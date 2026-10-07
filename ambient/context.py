from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import statistics
import time

from .policy import normalized, stable_id
from .learning import MENTION, addressing, mentions as message_mentions, plain_text, quote_source, reply_feedback, style_examples


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


@dataclass(frozen=True)
class Route:
    platform: str
    account: str
    group: str
    adapter: str

    @property
    def umo(self):
        return f'{self.platform}:GroupMessage:{self.group}'

    @property
    def key(self):
        return compact([self.platform, self.account, self.group])

    def member(self, account):
        return stable_id(self.key, self.platform, str(account))


def route_for(context, event):
    platform_id, group = str(event.get_platform_id()), str(event.get_group_id() or '')
    platform = context.get_platform_inst(platform_id)
    if not platform or not group or str(platform.meta().id) != platform_id:
        raise ValueError('native_group_unavailable')
    adapter = platform.meta().name
    # QQ message.self_id alternates between mention IDs and qq_official.
    account = str(getattr(platform, 'appid', '') or '') if adapter == 'qq_official' else str(event.get_self_id() or '')
    if adapter not in {'qq_official', 'aiocqhttp'} or not account or account in {'unknown_selfid', 'qq_official'}:
        raise ValueError('stable_account_unavailable')
    route = Route(platform_id, account, group, adapter)
    if str(event.unified_msg_origin) != route.umo:
        raise ValueError('native_group_mismatch')
    return route


def event_message(event, route, *, now=None):
    now = time.time() if now is None else now
    obj = event.message_obj
    identity = str(getattr(obj, 'message_id', '') or '')
    sender = str(event.get_sender_id() or '')
    if not identity or not sender or len(identity) > 256:
        return None
    pieces, mentions, quote = [], [], None
    attachment = False
    for part in getattr(obj, 'message', ()):
        kind = type(part).__name__
        if kind == 'Plain':
            pieces.append(str(part.text))
        elif kind == 'At':
            mentions.append(str(getattr(part, 'qq', '')))
        elif kind == 'Reply':
            quoted_text = normalized(getattr(part, 'message_str', ''))
            quote = {'id': str(getattr(part, 'id', ''))[:256],
                     'sender': str(getattr(part, 'sender_id', '') or ''),
                     'text': quoted_text[:240], 'truncated': len(quoted_text) > 240}
        elif kind in {'Image', 'Record', 'Video', 'File', 'Face'}:
            attachment = True
            pieces.append({'Image': '[图片]', 'Record': '[语音]', 'Video': '[视频]',
                           'File': '[文件]', 'Face': '[表情]'}[kind])
    text = normalized(' '.join(pieces))[:1800]
    mentions.extend(MENTION.findall(text))
    event_self = str(event.get_self_id() or '')
    # GROUP_AT_MESSAGE_CREATE may carry the adapter's qq_official sentinel.
    # It is still an explicit At(self_id), even when no mention ID is supplied.
    bot_ids = {route.account, event_self} - {'', 'unknown_selfid'}
    mentions = list(dict.fromkeys('bot' if target in bot_ids else target for target in mentions))
    text = plain_text(text)
    directed = bool(getattr(event, 'is_at_or_wake_command', False))
    attention_only = not text and directed and ('bot' in mentions or quote is not None)
    if not text:
        if not attention_only:
            return None
        text = '[引用消息，无附加正文]' if quote else '[仅@机器人，无正文]'
    raw = getattr(obj, 'raw_message', None)
    timestamp = getattr(raw, 'timestamp', None)
    if timestamp is None and isinstance(raw, dict):
        timestamp = raw.get('time')
    try:
        at = (timestamp.timestamp() if isinstance(timestamp, datetime) else
              float(timestamp) if isinstance(timestamp, (int, float)) else
              datetime.fromisoformat(str(timestamp).replace('Z', '+00:00')).timestamp())
    except (ValueError, TypeError, OverflowError):
        at = now
    at = min(now, at)
    self_message = sender == route.account or (event_self not in {'', 'qq_official', 'unknown_selfid'} and sender == event_self)
    return {'id': identity, 'sender': sender, 'member': route.member(sender),
            'name': normalized(event.get_sender_name())[:40], 'text': text,
            'at': at, 'self': self_message, 'command': text.startswith(('/', '／')),
            'attachment': attachment, 'mentions': mentions, 'quote': quote,
            'attention_only': attention_only, 'directed': directed}


class Window:
    def __init__(self, policy, rows=()):
        self.policy = policy
        self.rows = deque(sorted(rows, key=lambda r: r['at']), maxlen=policy.raw_limit)

    def add(self, row, now):
        self.trim(now)
        if any(r['id'] == row['id'] for r in self.rows):
            return False
        self.rows.append(row)
        self.rows = deque(sorted(self.rows, key=lambda r: r['at']), maxlen=self.policy.raw_limit)
        return True

    def trim(self, now):
        retention = (self.policy.raw_hours * 3600 if self.policy.raw_hours else
                     max(self.policy.recent_minutes, self.policy.style_minutes) * 60)
        self.rows = deque((r for r in self.rows if now-r['at'] <= retention), maxlen=self.policy.raw_limit)

    def forget(self, member):
        self.rows = deque((r for r in self.rows if r['member'] != member), maxlen=self.policy.raw_limit)


def build_pack(window, snapshot, current, *, now, memory_status='none'):
    """Raw evidence, bounded locally. Model never sees stable platform IDs."""
    p = window.policy
    window.trim(now)
    rows = list(window.rows)
    people = [current] if current else []
    people.extend(reversed(rows))
    aliases = {}
    for r in people:
        if r['self']:
            continue
        if r['member'] not in aliases:
            aliases[r['member']] = 'u'+str(len(aliases)+1)
    accounts = {r['sender']: aliases.get(r['member'], 'bot') for r in people}
    ids = {r['id']: 'm'+str(i+1) for i, r in enumerate(rows)}

    def render(r):
        result = {'ref': ids.get(r['id'], 'current'), 'speaker': 'bot' if r['self'] else aliases[r['member']],
                  'text': plain_text(r['text']), 'at': int(r['at']), 'seconds_ago': max(0, int(now-r['at']))}
        if r['self']:
            result['source'] = r.get('source', 'interjection')
        if not r['self']:
            result['name'] = r.get('name', '')
        if r.get('attention_only'):
            result['attention_only'] = True
        if r.get('quote'):
            q = r['quote']
            source, link = quote_source(r, rows)
            result['quote'] = {'ref': ids.get(source['id'], 'outside_window') if source else 'outside_window',
                               'speaker': accounts.get(source['sender'], 'unknown') if source else accounts.get(q.get('sender'), 'unknown'),
                               'text': plain_text(source['text'] if source else q.get('text', ''))[:240],
                               'link': link or 'unresolved'}
        if message_mentions(r):
            result['mentions'] = [accounts.get(a, 'bot' if a == 'bot' else 'outside_window') for a in message_mentions(r)]
        if not r['self']:
            targets, basis = addressing(r, rows)
            if targets:
                result['addressing'] = {'targets': [accounts.get(a, 'bot' if a == 'bot' else 'outside_window') for a in targets],
                                        'basis': basis}
        return result

    recent = [r for r in rows if now-r['at'] <= p.recent_minutes*60 and
              (not current or r['id'] != current['id']) and not r['command']][-p.recent_messages:]
    style = []
    def style_line(row):
        return {'ref': ids[row['id']], 'speaker': aliases[row['member']], 'text': plain_text(row['text'])}

    for row, lead, link in style_examples(rows, p, now, current):
        # Keep source anchors and whole messages; use the budget for real speech
        # instead of duplicating names, timestamps and addressing on every sample.
        sample = style_line(row)
        sample['link'] = link
        if lead:
            sample['lead_in'] = style_line(lead)
        style.append(sample)
    feedback = [{'reply': render(reply), 'reaction': render(reaction), 'kind': kind}
                for reply, reaction, kind in reply_feedback(rows, p, now)]

    def bounded(items, limit):
        kept = []
        for item in items:
            if len(compact(kept+[item])) <= limit:
                kept.append(item)
        return kept

    facts = snapshot.get('facts', []) + snapshot.get('summary', [])
    # Explicit boundaries take precedence when a snapshot exceeds its budget.
    facts.sort(key=lambda r: (r['kind'] not in {'boundary', 'promise', 'todo'}, -r['at']))
    memory = []
    for r in facts + snapshot.get('interaction_notes', []):
        if r['member'] not in aliases or (r.get('expires', 0) and r['expires'] <= now):
            continue
        memory.append({'speaker': aliases[r['member']], 'kind': r.get('kind', 'interaction_note'),
                       'text': r['text'], 'at': int(r['at']), 'state': 'committed'})
    for r in snapshot.get('candidates', []):
        if r['member'] in aliases:
            memory.append({'speaker': aliases[r['member']], 'kind': r['kind'], 'text': r['text'],
                           'at': int(r['at']), 'state': 'pending'})
    context_rows = bounded([render(r) for r in reversed(recent)], p.recent_chars)
    context_rows.reverse()
    feedback_rows = bounded(feedback, min(700, p.style_chars//2))
    style_rows = bounded(style, p.style_chars-len(compact(feedback_rows)))
    memory_rows = bounded(memory, p.memory_chars)
    human = [r for r in rows if not r['self'] and not r['command'] and now-r['at'] <= p.style_minutes*60]
    stats = {'median_chars': statistics.median([len(r['text']) for r in human]) if human else 0,
             'example_median_chars': statistics.median([len(r['text']) for r in style_rows]) if style_rows else 0,
             'short_ratio': round(sum(len(r['text']) <= 20 for r in human)/max(1, len(human)), 2),
             'question_ratio': round(sum('?' in r['text'] or '？' in r['text'] for r in human)/max(1, len(human)), 2),
             'emoji_ratio': round(sum(r['attachment'] for r in human)/max(1, len(human)), 2),
             'confirmed_self_ratio': round(sum(r['self'] for r in rows)/max(1, len(rows)), 2)}
    pack = {'generated_at': int(now), 'current_message': render(current) if current else None,
            'current_speaker': aliases.get(current['member']) if current else None,
            'memory_request_status': memory_status, 'recent_context': context_rows,
            'memory_snapshot': memory_rows, 'style_samples': style_rows, 'reply_feedback': feedback_rows,
            'rhythm_stats': stats if len(compact(stats)) <= p.stats_chars else {},
            'omitted_count': len(recent)-len(context_rows)+len(style)-len(style_rows)+len(memory)-len(memory_rows)}
    if pack['current_message']:
        pack['current_message']['text'] = pack['current_message']['text'][:p.pack_chars//3]
    # Total budget includes JSON overhead. Drop lowest priority fields first.
    if len(compact(pack)) > p.pack_chars:
        pack['rhythm_stats'] = {}
    for key in ('style_samples', 'reply_feedback', 'memory_snapshot', 'recent_context'):
        while pack[key] and len(compact(pack)) > p.pack_chars:
            pack[key].pop(0 if key == 'recent_context' else -1)
            pack['omitted_count'] += 1
    return pack
