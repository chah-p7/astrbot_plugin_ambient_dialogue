"""Short-lived conversational examples, never a learned persona or user profile."""
from __future__ import annotations

from collections import Counter
import re

from .policy import noise, normalized


MENTION = re.compile(r'<@!?([A-Za-z0-9_-]{1,128})>')
FORMAT_ORDER = re.compile(r'(?:以后|接下来|一直|每次|只能).{0,32}(?:回复|回答|回|发|扣)')


def mentions(row):
    # Old cached QQ messages may still contain unparsed Plain mentions.
    return list(dict.fromkeys([*row.get('mentions', []), *MENTION.findall(row['text'])]))


def plain_text(text):
    return normalized(MENTION.sub('', text))


def addressing(row, rows):
    targets = mentions(row)
    if targets:
        return targets, 'explicit_mention'
    if row.get('directed'):
        return ['bot'], 'directed_to_bot'
    # A short follow-on "那你…" can continue the same sender's explicit @.
    # Label the inference; it must never become an asserted relationship.
    if re.search(r'你|您', row['text']):
        previous = next((r for r in reversed(rows) if r['sender'] == row['sender']
                         and r['id'] != row['id'] and 0 <= row['at']-r['at'] <= 90), None)
        if previous and mentions(previous):
            return mentions(previous), 'recent_same_sender_mention'
    return [], 'unspecified'


def style_examples(rows, policy, now, current=None):
    def eligible(row):
        return (not any(row.get(k) for k in ('self', 'command', 'attachment', 'directed'))
                and 0 <= now-row['at'] <= policy.style_minutes*60
                and len(row['text']) <= 180 and not noise(plain_text(row['text']))
                and not re.search(r'\[(?:表情|图片|语音|视频|文件|卡片消息)', row['text'])
                and not FORMAT_ORDER.search(row['text']))

    available = {r['id']: r for r in rows if eligible(r)}
    examples = []
    for index, row in enumerate(rows):
        if row['id'] not in available or (current and row['id'] == current['id']):
            continue
        quote_id, targets = (row.get('quote') or {}).get('id'), mentions(row)
        lead, link = available.get(quote_id), 'quote'
        if lead and lead['at'] >= row['at']:
            lead = None
        if lead is None and targets and not quote_id:
            lead = next((r for r in reversed(rows[:index]) if r['sender'] in targets
                         and 0 <= row['at']-r['at'] <= 90 and r['id'] in available), None)
            link = 'mention'
        if lead and len(plain_text(lead['text'])) < 4:
            lead = None
        if lead is None and index and not (targets or quote_id):
            prior = rows[index-1]
            if prior['id'] in available and 0 <= row['at']-prior['at'] <= 45:
                if prior['sender'] == row['sender']:
                    lead, link = prior, 'same_speaker_continuation'
                elif re.search(r'[？?]|吗|几点|哪里|要不要|吃不吃|来不来', prior['text']):
                    lead, link = prior, 'nearby_only'
        examples.append((row, lead, link if lead else 'single'))
    # Prefer actual exchanges over isolated one-liners; retain ordinary speech,
    # not only jokes. Within each class use the newest evidence first.
    rank = {'quote': 3, 'mention': 3, 'same_speaker_continuation': 2, 'nearby_only': 1, 'single': 0}
    examples.sort(key=lambda x: x[0]['at']+rank[x[2]]*90, reverse=True)
    selected, counts, seen = [], Counter(), set()
    for row, lead, link in examples:
        if row['text'] in seen or counts[row['member']] >= policy.style_per_sender:
            continue
        selected.append((row, lead, link))
        seen.add(row['text'])
        counts[row['member']] += 1
        if len(selected) >= policy.style_messages:
            break
    return selected


def reply_feedback(rows, policy, now):
    """Bind explicit corrections to a confirmed reply; '?' is never a rating."""
    patterns = (
        ('wrong_addressee', r'没(?:和|跟)你说|不是(?:在)?(?:问|叫|跟|和)你|没你.{0,4}事|别抢答'),
        ('misunderstood', r'理解错|看错了|不是这个意思|答非所问|话都看不明白|胡编|乱编'),
        ('unwelcome_interjection', r'闭嘴|别插话|不要插话|停止.{0,12}(?:输出|发消息)|咋啥事都有你'),
    )
    confirmed, selected, seen = {}, [], set()
    for index, row in enumerate(rows):
        if row['self']:
            confirmed[row['id']] = row
            continue
        if (row.get('command') or row.get('attachment') or len(row['text']) > 180
                or now-row['at'] > policy.recent_minutes*60):
            continue
        kind = next((kind for kind, pattern in patterns if re.search(pattern, row['text'])), None)
        if not kind:
            continue
        targets = mentions(row)
        if targets and 'bot' not in targets and not row.get('directed'):
            continue
        reply = confirmed.get((row.get('quote') or {}).get('id'))
        if reply is None:
            prior = next((r for r in reversed(rows[:index])
                          if not r.get('attachment') and not noise(plain_text(r['text']))), None)
            if row.get('directed') or '机器人' in row['text'] or (prior and prior['self']):
                reply = next(reversed(confirmed.values()), None)
        if reply and 0 <= row['at']-reply['at'] <= 90:
            selected.append((reply, row, kind))
    result = []
    for reply, row, kind in reversed(selected):
        if reply['id'] not in seen:
            result.append((reply, row, kind))
            seen.add(reply['id'])
        if len(result) == 3:
            break
    return result
