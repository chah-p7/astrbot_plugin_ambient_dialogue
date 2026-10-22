"""Bounded human dialogue cases with verified links, not generated punchlines."""
import re

from .learning import FORMAT_ORDER, mentions, plain_text, quote_source, reaction_only, signals, words
from .policy import noise, stable_id


CORRECTION = re.compile(r'没(?:和|跟)你说|不是(?:在)?问你|理解错|不是这个意思|别硬玩梗|别解释笑点|别乱说|别插话')


def case_key(case):
    return stable_id(*(plain_text(r['text']).casefold() for r in case['rows'][:2]))


def extract_cases(rows, policy, *, excluded=()):
    links = {r['id']: quote_source(r, rows) for r in rows if r.get('quote')}
    followups = {}
    for row in rows:
        source, _ = links.get(row['id'], (None, ''))
        if source and not row['self'] and not row.get('command') and row['sender'] not in excluded and row['at'] >= source['at']:
            followups.setdefault(source['id'], []).append(row)
    def eligible(row):
        text = plain_text(row['text'])
        return (not any(row.get(k) for k in ('self', 'command', 'directed', 'attachment'))
                and row['sender'] not in excluded and 'bot' not in mentions(row)
                and not set(mentions(row)) & set(excluded)
                and 2 <= len(text) <= 180 and not noise(text) and not reaction_only(text)
                and not FORMAT_ORDER.search(text) and not re.search(r'https?://|\[(?:图片|语音|表情|文件|视频|卡片)', text))
    for row in rows:
        if not eligible(row):
            continue
        lead, link = links.get(row['id'], (None, ''))
        if not lead or not eligible(lead) or lead['sender'] == row['sender']:
            continue
        # Do not archive factual Q&A, personal names or unrelated single lines
        # as reusable banter. Sources must have a recognizable social situation.
        scenes, _ = signals(plain_text(lead['text']))
        if not scenes or CORRECTION.search(row['text']):
            continue
        reactions = followups.get(row['id'], [])
        if any(CORRECTION.search(r['text']) for r in reactions):
            continue
        following = next((r for r in reactions if r['at']-row['at'] <= 90 and eligible(r)), None)
        selected = [lead, row]
        if following and eligible(following):
            selected.append(following)
        if sum(len(r['text']) for r in selected) > policy.style_case_chars:
            selected = selected[:2]
        if sum(len(r['text']) for r in selected) > policy.style_case_chars:
            continue
        # Keep source accounts privately for exclusion, correction and deletion.
        payload = [{k: r.get(k) for k in ('id', 'sender', 'member', 'name', 'text', 'at', 'self', 'mentions')}
                   for r in selected]
        yield {'id': stable_id(*(r['id'] for r in selected[:2])), 'at': row['at'],
               'rows': payload, 'link': link, 'scenes': sorted(scenes)}


def select_cases(cases, rows, current, policy, now, *, focused=(), excluded=()):
    if not current or not policy.style_archive_limit:
        return []
    query = ' '.join(plain_text(r['text']) for r in [current, *focused] if not r['self'])
    terms, (scenes, _) = words(query), signals(query)
    current_ids = {r['id'] for r in rows if 0 <= now-r['at'] <= policy.style_minutes*60}
    ranked = []
    for case in cases:
        if not 0 <= now-case['at'] <= policy.style_archive_days*86400:
            continue
        if (any(r['sender'] in excluded for r in case['rows'])
                or any(r['at'] >= current['at'] for r in case['rows'])
                or all(r['id'] in current_ids for r in case['rows'][:2])):
            continue
        lead = case['rows'][0]
        overlap = len(words(plain_text(lead['text'])) & terms)
        matches = len(set(case['scenes']) & scenes)
        if matches or overlap >= 2:
            ranked.append((matches, overlap, case['at'], case))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    result, seen = [], set()
    for *_, case in ranked:
        key = case_key(case)
        if key not in seen:
            result.append(case)
            seen.add(key)
        if len(result) == 2:
            break
    return result
