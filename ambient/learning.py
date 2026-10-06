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
    source, link = quote_source(row, rows)
    if source:
        return ['bot' if source['self'] else source['sender']], link
    # A short follow-on "那你…" can continue the same sender's explicit @.
    # Label the inference; it must never become an asserted relationship.
    if re.search(r'你|您', row['text']):
        previous = next((r for r in reversed(rows) if r['sender'] == row['sender']
                         and r['id'] != row['id'] and 0 <= row['at']-r['at'] <= 90), None)
        if previous and mentions(previous):
            return mentions(previous), 'recent_same_sender_mention'
    return [], 'unspecified'


def quote_source(row, rows):
    """Resolve a real quote in this window; mentions/adjacency are not replies.

    The exact-quote and relevance approach is adapted from Tulpa (MIT); see
    docs/THIRD_PARTY.md. Never retarget a missing native ID using preview text.
    """
    quote = row.get('quote') or {}
    ident, text = quote.get('id'), plain_text(quote.get('text', ''))
    if not ident and (quote.get('truncated') or not 2 <= len(text) < 240):
        return None, ''
    candidates = [r for r in rows if r['id'] != row['id'] and r['at'] <= row['at']
                  and (r['id'] == ident if ident else plain_text(r['text']) == text)]
    if len(candidates) != 1:
        return None, ''
    source = candidates[0]
    # Check ambiguity before the author hint; a nickname cannot disambiguate it.
    author = quote.get('sender')
    if author and author not in (source['sender'], source.get('name')):
        return None, ''
    return source, 'native_quote' if ident else 'quoted_text_exact'


def signals(text):
    scenes = {name for name, pattern in (
        ('invitation', r'一起|来不来|玩吗|吃饭|打游戏|出去|约|几点|午饭|晚饭'),
        ('thanks', r'谢谢|感谢|多谢'),
        ('help', r'怎么|请教|咋|报错|代码|解决|服务器|接口|内存|程序'),
        ('banter', r'哈哈|笑死|草|绷|蚌|离谱|逆天|乐了|寄了'),
        ('arrangement', r'提交|老师|报告|开会|时间|作业|收到|出发'),
        ('correction', r'不是这个意思|理解错|看错|别插话|没跟你说|没和你说'),
    ) if re.search(pattern, text)}
    intents = {name for name, pattern in (
        ('decline', r'不去|不来|不了|没空|算了|拒绝|不打'),
        ('accept', r'好的|可以|行啊|没问题|收到'),
        ('question', r'[?？]|怎么|咋|几点|为啥'),
        ('tease', r'调戏|打趣|笑|哈哈|绷|离谱|逆天'),
    ) if re.search(pattern, text)}
    return scenes, intents


def words(text):
    # Small local matching vocabulary; no embeddings, model, index or archive.
    terms = re.findall(r'[a-z0-9_]{2,}', text.lower())
    for phrase in re.findall(r'[\u4e00-\u9fff]{2,}', text):
        terms.extend(phrase[i:i+2] for i in range(len(phrase)-1))
    return set(terms) - {'这个', '那个', '什么', '怎么', '你们', '我们', '是不是', '不是', '一个', '一下'}


def style_examples(rows, policy, now, current=None):
    def eligible(row):
        return (not any(row.get(k) for k in ('self', 'command', 'attachment', 'directed'))
                and 'bot' not in mentions(row)
                and 0 <= now-row['at'] <= policy.style_minutes*60
                and (not current or (row['id'] != current['id'] and row['at'] <= current['at']))
                and len(row['text']) <= 180 and not noise(plain_text(row['text']))
                and not re.search(r'\[(?:表情|图片|语音|视频|文件|卡片消息)', row['text'])
                and not FORMAT_ORDER.search(row['text']))

    available = {r['id']: r for r in rows if eligible(r)}
    examples = []
    for row in rows:
        if row['id'] not in available:
            continue
        lead, link = quote_source(row, rows)
        if lead and lead['self']:
            continue
        if lead and (lead['id'] not in available or len(plain_text(lead['text'])) < 2):
            lead = None
        examples.append((row, lead, link if lead else 'single'))
    query = current['text'] if current else ''
    source, _ = quote_source(current, rows) if current else (None, '')
    if source:
        query += ' '+source['text']
    scenes, intents = signals(query)
    terms = words(query)

    def rank(example):
        row, lead, _ = example
        incoming = lead or row
        old_scenes, _ = signals(incoming['text'])
        _, old_intents = signals(row['text'])
        overlap = len(terms & words(incoming['text']))
        similar = bool(scenes & old_scenes or overlap)
        same = bool(current and incoming['sender'] == current['sender'])
        tier = 3 if same and similar else 2 if same else 1 if similar else 0
        return tier, bool(lead), len(intents & old_intents), overlap, row['at']

    # Prefer the current person's matching scene, then the person or scene.
    # Unlinked lines still describe room voice, never an invented exchange.
    examples.sort(key=rank, reverse=True)
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
        source, _ = quote_source(row, rows)
        reply = confirmed.get(source['id']) if source else None
        if reply is None and not row.get('quote'):
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
