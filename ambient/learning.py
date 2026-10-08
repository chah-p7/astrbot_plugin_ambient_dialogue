"""Short-lived conversational examples, never a learned persona or user profile."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from .policy import noise, normalized


MENTION = re.compile(r'<@!?([A-Za-z0-9_-]{1,128})>')
FORMAT_ORDER = re.compile(r'(?:以后|接下来|一直|每次|只能).{0,32}(?:回复|回答|回|发|扣)')
SHORT_REACTION = re.compile(r'(?:嗯|哦|啊|诶|欸|呃|额|哈|呵|嘿|嘻){1,6}[。.!！~～…]*')


@dataclass(frozen=True)
class StyleExample:
    row: dict
    lead: dict | None
    link: str
    context: tuple = ()


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
        ('help', r'请教|求助|报错|代码|服务器|接口|内存|程序|闪退|崩溃|启动不了'),
        ('banter', r'哈哈|笑死|草|绷|蚌|离谱|逆天|乐了|寄了'),
        ('arrangement', r'提交|老师|报告|开会|时间|作业|收到|出发'),
        ('correction', r'不是这个意思|理解错|看错|别插话|没跟你说|没和你说|别.{0,5}(?:阴阳|损我)|(?i:ai).{0,3}味'),
        ('opinion', r'好不好|好看|好玩|怎么样|咋样|喜欢|难看|没意思'),
        ('daily', r'睡醒|起床|睡不着|熬夜|犯困|刚睡|没睡'),
        ('frustration', r'无聊|没招|白忙|又坏|烦死|累死|忘了|破防'),
    ) if re.search(pattern, text)}
    intents = {name for name, pattern in (
        ('decline', r'不去|不来|不了|没空|算了|拒绝|不打'),
        ('accept', r'好的|可以|行啊|没问题|收到'),
        ('question', r'[?？吗]|怎么|咋|几点|为啥|有没有|好不好'),
        ('acknowledge', r'^(?:嗯+|哦+|好[的啊吧]?|行[啊吧]?|知道了|收到)[。!！~～]*$'),
        ('tease', r'调戏|打趣|笑|哈哈|绷|离谱|逆天'),
    ) if re.search(pattern, text)}
    return scenes, intents


def words(text):
    # Small local matching vocabulary; no embeddings, model, index or archive.
    terms = re.findall(r'[a-z0-9_]{2,}', text.lower())
    for phrase in re.findall(r'[\u4e00-\u9fff]{2,}', text):
        terms.extend(phrase[i:i+2] for i in range(len(phrase)-1))
    return set(terms) - {'这个', '那个', '什么', '怎么', '你们', '我们', '是不是', '不是', '一个', '一下'}


def style_examples(rows, policy, now, current=None, *, excluded=frozenset()):
    rows = sorted(rows, key=lambda r: r['at'])
    bot_targets = {'bot', *excluded}

    def eligible(row):
        text = plain_text(row['text'])
        return (not any(row.get(k) for k in ('self', 'command', 'attachment', 'directed'))
                and row['sender'] not in excluded
                and not (bot_targets & set(mentions(row)))
                and (row.get('quote') or {}).get('sender') not in bot_targets
                and 0 <= now-row['at'] <= policy.style_minutes*60
                and (not current or (row['id'] != current['id'] and row['at'] <= current['at']))
                and 0 < len(text) <= 180 and (not noise(text) or SHORT_REACTION.fullmatch(text))
                and not re.search(r'\[(?:表情|图片|语音|视频|文件|卡片消息)', row['text'])
                and not FORMAT_ORDER.search(row['text']))

    available = {r['id']: r for r in rows if eligible(r)}
    links = {ident: quote_source(row, rows) for ident, row in available.items()}
    for ident, (lead, _) in links.items():
        if lead and (lead['self'] or lead['sender'] in excluded):
            available.pop(ident)
    examples = []
    for index, row in enumerate(rows):
        if row['id'] not in available:
            continue
        lead, link = links[row['id']]
        if lead and (lead['id'] not in available or noise(plain_text(lead['text']))):
            lead = None
        before = []
        # Adjacent lines remain timeline evidence, never invented reply edges.
        # Stop at an excluded message instead of joining across a bot/command.
        for previous in reversed(rows[max(0, index-policy.style_context_messages):index]):
            if previous['id'] not in available or row['at']-previous['at'] > policy.style_context_seconds:
                break
            if lead and previous['id'] == lead['id']:
                break
            before.append(previous)
        before.reverse()
        reaction = bool(SHORT_REACTION.fullmatch(plain_text(row['text'])))
        if reaction and not (lead or any(not noise(plain_text(r['text'])) for r in before)):
            continue
        examples.append(StyleExample(row, lead, link if lead else 'single', tuple(before)))
    query = current['text'] if current else ''
    source, _ = quote_source(current, rows) if current else (None, '')
    if source:
        query += ' '+source['text']
    scenes, intents = signals(query)
    terms = words(query)

    def rank(example):
        row, lead = example.row, example.lead
        incoming = lead or row
        incoming_text = incoming['text']
        if not lead:
            incoming_text = ' '.join([r['text'] for r in example.context if r['sender'] == row['sender']] + [incoming_text])
        old_scenes, old_intents = signals(incoming_text)
        overlap = len(terms & words(incoming_text))
        scene_match = len(scenes & old_scenes)
        same = bool(current and incoming['sender'] == current['sender'])
        return bool(scene_match or overlap), scene_match, overlap, len(intents & old_intents), bool(lead), same, row['at']

    # Scene and incoming intent precede identity; do not copy the asker's tone
    # just because an unrelated line happens to come from the same person.
    examples.sort(key=rank, reverse=True)
    selected, counts, covered, reactions = [], Counter(), set(), 0
    for example in examples:
        row = example.row
        reaction = bool(SHORT_REACTION.fullmatch(plain_text(row['text'])))
        if (row['id'] in covered or counts[row['member']] >= policy.style_per_sender
                or (reaction and reactions >= max(1, policy.style_messages//5))
                or any(similar_style(row['text'], old.row['text']) for old in selected)):
            continue
        selected.append(example)
        covered.update(r['id'] for r in (row, *example.context))
        reactions += reaction
        counts[row['member']] += 1
        if len(selected) >= policy.style_messages:
            break
    return selected


def similar_style(first, second):
    """Collapse cosmetic variants without merging ordinary distinct short replies."""
    first, second = (re.sub(r'[\W_]+', '', plain_text(s).lower()) for s in (first, second))
    if first == second:
        return True
    if min(len(first), len(second)) < 8:
        return False
    a, b = words(first), words(second)
    return bool(a and b) and len(a & b)/len(a | b) >= .85


def reply_feedback(rows, policy, now, *, excluded=frozenset()):
    """Bind explicit corrections to a confirmed reply; '?' is never a rating."""
    patterns = (
        ('wrong_addressee', r'没(?:和|跟)你说|不是(?:在)?(?:问|叫|跟|和)你|没你.{0,4}事|别抢答'),
        ('misunderstood', r'理解错|看错了|不是这个意思|答非所问|话都看不明白|胡编|乱编'),
        ('unwelcome_interjection', r'闭嘴|别插话|不要插话|停止.{0,12}(?:输出|发消息)|咋啥事都有你'),
        ('unnatural_style', r'(?i:ai).{0,3}味|像(?:个)?(?:机器人|客服)|小作文|又.{0,6}(?:说话腔调|这个腔调)|别.{0,6}(?:硬玩梗|解释笑点|阴阳|损我)|太端着'),
    )
    confirmed, selected, seen = {}, [], set()
    for index, row in enumerate(rows):
        if row['self']:
            confirmed[row['id']] = row
            continue
        if (row['sender'] in excluded or row.get('command') or row.get('attachment') or len(row['text']) > 180
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
            if row.get('directed') or 'bot' in targets or (prior and prior['self']):
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
