from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import re
import statistics
import time

from .policy import display_name, normalized, stable_id
from .learning import (MENTION, MEDIA_ONLY, TOPIC_CHANGE, addressing, continues_topic,
                       mentions as message_mentions, plain_text, quote_source,
                       reaction_only, reply_feedback, same_speaker_context, short_answer, style_examples)
from .recall import recall_rows
from .aliases import identity_lookups, known_names, name_key, name_targets
from .examples import select_cases


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def participants(snapshot, people, current, focused=()):
    """Resolve only stable accounts; name search returns candidates, never merges."""
    members = {r['id']: r for r in snapshot.get('members', [])}
    aliases, accounts, names, latest = {}, {}, {}, {}
    for r in people:
        if r['self']:
            accounts[r['sender']] = 'bot'
            continue
        member = r['member']
        aliases.setdefault(member, 'u'+str(len(aliases)+1))
        accounts[r['sender']] = aliases[member]
        latest.setdefault(member, r)
    for member, r in members.items():
        names[member] = display_name(r['name'], r['account'])
    for member, r in latest.items():
        name = display_name(r.get('name', ''), r['sender'])
        if name and (r is current or r['at'] >= members.get(member, {}).get('seen', 0)):
            names[member] = name
        names.setdefault(member, '')
    for r in people:
        if not r['self'] and not names.get(r['member']):
            names[r['member']] = display_name(r.get('name', ''), r['sender'])
    for alias in snapshot.get('aliases', []):
        if alias['kind'] == 'rejected' and name_key(names.get(alias['member'], '')) == alias['key']:
            names[alias['member']] = ''

    by_account = {r['account']: member for member, r in members.items()}
    by_account.update({r['sender']: r['member'] for r in people if not r['self']})
    priority = [current['member']] if current and not current['self'] else []
    referenced = []
    for row in people:
        refs = message_mentions(row) + [(row.get('quote') or {}).get('sender')]
        for account in refs:
            if not account or account == 'bot' or accounts.get(account) == 'bot':
                continue
            # Unknown mentioned accounts still need distinct prompt-local IDs.
            # This does not create a persistent profile for a passive member.
            member = by_account.get(account, 'mentioned:'+account)
            if account not in accounts:
                aliases.setdefault(member, 'u'+str(len(aliases)+1))
                accounts[account] = aliases[member]
            (priority if row is current else referenced).append(member)
    named, tentative = set(), {}
    question = name_key(plain_text(current['text'])) if current else ''
    for member in members:
        accepted = known_names(snapshot, {**members[member], 'name': names.get(member, '')})
        pending = [a['name'] for a in snapshot.get('aliases', []) if a['member'] == member and a['kind'] in {'candidate', 'inferred'}]
        for name in [*accepted, *pending]:
            if 2 <= len(name) <= 40 and re.search(r'(?<![A-Za-z0-9_])'+re.escape(name_key(name))+r'(?![A-Za-z0-9_])', question):
                named.add(member)
                if name not in accepted:
                    tentative.setdefault(member, []).append(name)
                aliases.setdefault(member, 'u'+str(len(aliases)+1))
                accounts[members[member]['account']] = aliases[member]
                priority.append(member)
                break
    priority.extend(r['member'] for r in focused if not r['self'])
    priority.extend(latest)
    priority.extend(referenced)
    cards = []
    for member in dict.fromkeys(priority):
        card = {'speaker': aliases[member], 'name': names.get(member, ''),
                'known_in_group': member in members}
        if current and member == current['member']:
            card['known_in_group'] = current.get('known_before', card['known_in_group'])
        if member in named and (not current or member != current['member']):
            card['name_match_only'] = True
        accepted = [a['name'] for a in snapshot.get('aliases', []) if a['member'] == member
                    and a['kind'] not in {'candidate', 'inferred', 'rejected'} and a['name'] != card['name']]
        if accepted:
            card['aliases'] = list(dict.fromkeys(accepted))
        if member in tentative:
            card['unconfirmed_aliases'] = tentative[member]
        inferred = [a['name'] for a in snapshot.get('aliases', []) if a['member'] == member and a['kind'] == 'inferred']
        if inferred:
            card['inferred_aliases'] = inferred
        cards.append(card)
    return aliases, accounts, names, cards


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
    attachment, image_count = False, 0
    for part in getattr(obj, 'message', ()):
        kind = type(part).__name__
        if kind == 'Plain':
            pieces.append(str(part.text))
        elif kind == 'At':
            target = str(getattr(part, 'qq', ''))
            mentions.append(target)
            if target:
                pieces.append('<@'+target+'>')
        elif kind == 'Reply':
            quoted_text = normalized(getattr(part, 'message_str', ''))
            quote = {'id': str(getattr(part, 'id', ''))[:256],
                     'sender': str(getattr(part, 'sender_id', '') or ''),
                     'text': quoted_text[:240], 'truncated': len(quoted_text) > 240}
        elif kind in {'Image', 'Record', 'Video', 'File', 'Face'}:
            attachment = True
            image_count += kind == 'Image'
            pieces.append({'Image': '[图片]', 'Record': '[语音]', 'Video': '[视频]',
                           'File': '[文件]', 'Face': '[表情]'}[kind])
    text = normalized(' '.join(pieces))[:1800]
    mentions.extend(MENTION.findall(text))
    event_self = str(event.get_self_id() or '')
    # GROUP_AT_MESSAGE_CREATE may carry the adapter's qq_official sentinel.
    # It is still an explicit At(self_id), even when no mention ID is supplied.
    bot_ids = {route.account, event_self} - {'', 'unknown_selfid'}
    mentions = list(dict.fromkeys('bot' if target in bot_ids else target for target in mentions))
    text = MENTION.sub(lambda m: '<@bot>' if m[1] in bot_ids else m[0], text)
    directed = bool(getattr(event, 'is_at_or_wake_command', False))
    body = plain_text(text)
    attention_only = not body and directed and ('bot' in mentions or quote is not None)
    if not body:
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
            'name': display_name(event.get_sender_name(), sender), 'text': text,
            'at': at, 'self': self_message, 'command': body.startswith(('/', '／')),
            'attachment': attachment, 'mentions': mentions, 'quote': quote,
            'attention_only': attention_only, 'directed': directed,
            **({'image_count': image_count} if image_count else {})}


class Window:
    def __init__(self, policy, rows=(), *, style_excluded_senders=()):
        self.policy = policy
        self.style_excluded_senders = frozenset(style_excluded_senders)
        self.rows = deque(sorted(rows, key=lambda r: r['at']), maxlen=policy.raw_limit)
        self.identity_snapshot = {}
        self.image_notes = {}
        self.vision_enabled = False

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
        active = {r['id'] for r in self.rows}
        self.image_notes = {k: v for k, v in self.image_notes.items() if k in active}
        for key in list(self.image_notes):
            if sum(map(len, self.image_notes.values())) <= self.policy.vision_cache_limit:
                break
            self.image_notes.pop(key)

    def forget(self, member):
        self.rows = deque((r for r in self.rows if r['member'] != member), maxlen=self.policy.raw_limit)
        active = {r['id'] for r in self.rows}
        self.image_notes = {k: v for k, v in self.image_notes.items() if k in active}


RECALL_REPLY = re.compile(r'你(?:刚才|之前|前面|上句)(?:是)?(?:说|讲|问)|(?:刚才|之前|前面)你(?:说|讲|问)|再说一遍')


def focus_rows(rows, current, *, topic_seconds=900):
    """Pin verified quotes and supported continuations, never identity alone."""
    if not current:
        return []
    selected, seen = [], {current['id']}

    def chain(row, reason):
        for _ in range(4):
            source, _ = quote_source(row, rows)
            if not source or source['id'] in seen or source.get('command'):
                break
            seen.add(source['id'])
            selected.append((source, reason))
            row, reason = source, 'quote_ancestor'

    chain(current, 'direct_quote')
    if not current.get('quote'):
        # Walk a continuous same-speaker burst, stopping at a topic boundary.
        for row in same_speaker_context(rows, current):
            if row['id'] not in seen:
                seen.add(row['id'])
                selected.append((row, 'same_speaker_recent'))
                chain(row, 'reply_anchor')
        # Only a confirmed reply to this account, never the room's last bot line.
        previous = next((r for r in reversed(rows) if r['self'] and not r.get('attachment')
                         and (r.get('quote') or {}).get('sender') == current['sender']
                         and 0 <= current['at']-r['at'] <= max(180, topic_seconds)), None)
        anchor, _ = quote_source(previous, rows) if previous else (None, '')
        # A vague "why" can follow the last room message, but cannot jump over
        # intervening conversations to an old bot response to the same account.
        last = next((r for r in reversed(rows) if r['id'] != current['id'] and not r.get('command')
                     and not reaction_only(r['text'])), None)
        changed = previous and any(r['sender'] == current['sender'] and previous['at'] < r['at'] <= current['at']
                                   and TOPIC_CHANGE.search(plain_text(r['text'])) for r in rows)
        topic_rows = [current] + [r for r, _ in selected]
        answered = previous and any(
            0 <= r['at']-previous['at'] <= topic_seconds and not message_mentions(r)
            and short_answer(previous['text'], r['text'])
            and next((s for s in reversed(rows) if s['at'] < r['at'] and not s.get('command')
                      and not reaction_only(s['text'])), None) is previous
            for r in topic_rows if not r['self'])
        related = previous and not changed and (current['at']-previous['at'] <= 180 and (
            RECALL_REPLY.search(plain_text(current['text']))
            or continues_topic(current, previous, adjacent=last is previous)
            or (anchor and any(continues_topic(r, anchor) for r in topic_rows)))
            or answered)
        if related and previous['id'] not in seen:
            seen.add(previous['id'])
            selected.append((previous, 'last_reply_to_speaker'))
            chain(previous, 'reply_anchor')
    return selected[:6]


def visual_rows(rows, current, focused=None):
    """Only attach a visible image occurrence; ambiguity never selects a stranger's photo."""
    if not current:
        return []
    focused = focus_rows(rows, current) if focused is None else focused
    if current.get('image_count') and not current.get('quote') and not re.search(
            r'这[两几]张|对比|比较|上一张|前一张|和刚才', current['text']):
        return [current]
    selected = [r for r in [current, *(r for r, _ in focused)] if r.get('image_count')]
    if not current.get('quote'):
        selected = [r for r in selected if r['id'] == current['id'] or not any(
            s.get('image_count') and s['sender'] != r['sender'] and r['at'] < s['at'] < current['at'] for s in rows)]
    if selected or current.get('quote') or TOPIC_CHANGE.search(current['text']):
        return list({r['id']: r for r in selected}.values())
    if not re.search(r'(?:刚才|前面|上面|这|那)(?:张|个|几张)?(?:图|截图|照片|表情)|图[里上中]|这是什么', current['text']):
        return []
    recent = [r for r in rows if 0 < current['at']-r['at'] <= 90 and not r['self']]
    images = [r for r in recent if r.get('image_count') and not r.get('command')]
    if len(images) != 1 or any(r['at'] > images[0]['at'] and TOPIC_CHANGE.search(r['text']) for r in recent):
        return []
    return images


def model_text(text, accounts):
    """Preserve mention positions while hiding stable platform identifiers."""
    return normalized(MENTION.sub(lambda m: '@'+accounts.get(m[1], 'bot' if m[1] == 'bot' else 'unknown'), text))


def build_pack(window, snapshot, current, *, now, memory_status='none'):
    """Raw evidence, bounded locally. Model never sees stable platform IDs."""
    p = window.policy
    window.trim(now)
    rows = [r for r in window.rows if r['at'] <= min(now, current['at'] if current else now)]
    if current and any(r['id'] == current['id'] for r in rows):
        rows = rows[:next(i for i, r in enumerate(rows) if r['id'] == current['id'])+1]
    focused = focus_rows(rows, current, topic_seconds=p.topic_minutes*60)
    images = visual_rows(rows, current, focused) if window.vision_enabled else []
    focused.extend((r, 'image_reference') for r in images if r['id'] != current['id']
                   and all(s['id'] != r['id'] for s, _ in focused))
    recalled, lookup = recall_rows(rows, current, snapshot, excluded=window.style_excluded_senders)
    saved_examples = select_cases(snapshot.get('style_cases', []), rows, current, p, now,
                                  focused=[r for r, _ in focused], excluded=window.style_excluded_senders)
    people = [current] if current else []
    people.extend(reversed(rows))
    people.extend(r for case in saved_examples for r in case['rows'])
    aliases, accounts, names, cards = participants(snapshot, people, current,
                                                  [r for r, _ in focused + recalled])
    automated = {accounts[a] for a in window.style_excluded_senders if a in accounts}
    for card in cards:
        if card['speaker'] in automated:
            card['automated'] = True
    ids = {r['id']: 'm'+str(i+1) for i, r in enumerate(rows)}

    def render(r):
        result = {'ref': ids.get(r['id'], 'current'), 'speaker': 'bot' if r['self'] else aliases[r['member']],
                  'text': model_text(r['text'], accounts), 'at': int(r['at']), 'seconds_ago': max(0, int(now-r['at']))}
        if r['self']:
            result['source'] = r.get('source', 'interjection')
        if not r['self']:
            result['name'] = names.get(r['member'], '')
            if r['sender'] in window.style_excluded_senders:
                result['automated'] = True
        if r.get('attention_only'):
            result['attention_only'] = True
        if MEDIA_ONLY.fullmatch(plain_text(r['text'])):
            result['media_content_unavailable'] = not (any(s['id'] == r['id'] for s in images) and
                any(n['state'] == 'ready' for n in window.image_notes.get(r['id'], [])))
        if r.get('image_count'):
            result['image_count'] = r['image_count']
        if r.get('quote'):
            q = r['quote']
            source, link = quote_source(r, rows)
            quoted_speaker = (accounts.get(q.get('sender'), 'unknown')
                              if q.get('id') not in ids else 'unknown')
            result['quote'] = {'ref': ids.get(source['id'], 'outside_window') if source else 'outside_window',
                               'speaker': accounts.get(source['sender'], 'unknown') if source else quoted_speaker,
                               'text': model_text(source['text'] if source else q.get('text', ''), accounts)[:240],
                               'link': link or 'unresolved'}
        if message_mentions(r):
            result['mentions'] = [accounts.get(a, 'bot' if a == 'bot' else 'outside_window') for a in message_mentions(r)]
            if not MENTION.search(r['text']) and not r.get('attention_only'):
                result['mention_positions_unknown'] = True
        if not r['self']:
            targets, basis = addressing(r, rows)
            if not targets:
                targets = name_targets(r, snapshot)
                basis = 'confirmed_name_vocative'
            if targets:
                result['addressing'] = {'targets': [accounts.get(a, 'bot' if a == 'bot' else 'outside_window') for a in targets],
                                        'basis': basis}
        return result

    recent = [r for r in rows if now-r['at'] <= p.recent_minutes*60 and
              (not current or r['id'] != current['id']) and not r['command']][-p.recent_messages:]
    style = []
    def style_line(row):
        return {'ref': ids[row['id']], 'speaker': aliases[row['member']],
                'text': model_text(row['text'], accounts), 'seconds_ago': max(0, int(now-row['at']))}

    for example in style_examples(rows, p, now, current, excluded=window.style_excluded_senders,
                                  focused=[r for r, _ in focused]):
        # Keep source anchors and whole messages; use the budget for real speech
        # instead of duplicating names, timestamps and addressing on every sample.
        sample = style_line(example.row)
        sample['link'] = example.link
        if example.lead:
            sample['lead_in'] = style_line(example.lead)
        if example.context:
            sample['nearby'] = [style_line(r) for r in example.context]
        style.append(sample)
    for i, case in enumerate(saved_examples):
        def case_line(row, position):
            return {'ref': f's{i+1}.{position}', 'speaker': aliases[row['member']],
                    'text': model_text(row['text'], accounts), 'seconds_ago': max(0, int(now-row['at']))}
        sample = {**case_line(case['rows'][1], 2), 'link': case['link'], 'source': 'saved_human_example',
                  'lead_in': case_line(case['rows'][0], 1)}
        if len(case['rows']) > 2:
            sample['follow_up'] = case_line(case['rows'][2], 3)
        style.insert(i, sample)
    style = style[:p.style_messages]
    feedback = [{'reply': render(reply), 'reaction': render(reaction), 'kind': kind}
                for reply, reaction, kind in reply_feedback(rows, p, now, excluded=window.style_excluded_senders)]

    def bounded(items, limit):
        kept = []
        for item in items:
            if len(compact(kept+[item])) <= limit:
                kept.append(item)
        return kept

    # Pending corrections replace the old value in this prompt immediately;
    # state still says pending until the existing batch commits it.
    facts = {r['id']: {**r, 'state': 'committed'}
             for r in snapshot.get('summary', []) + snapshot.get('facts', [])}
    facts.update({r['id']: {**r, 'state': 'pending'} for r in snapshot.get('candidates', [])})
    facts = list(facts.values())
    # Explicit boundaries take precedence when a snapshot exceeds its budget.
    facts.sort(key=lambda r: (r['kind'] not in {'boundary', 'promise', 'todo'},
                             r['member'] != (current or {}).get('member'), r['kind'] != 'address', -r['at']))
    memory = []
    for r in facts + snapshot.get('interaction_notes', []):
        if r['member'] not in aliases or (r.get('expires', 0) and r['expires'] <= now):
            continue
        memory.append({'speaker': aliases[r['member']], 'kind': r.get('kind', 'interaction_note'),
                       'text': r['text'], 'at': int(r['at']), 'state': r.get('state', 'committed')})
    focus = bounded([{**render(r), 'relevance': reason} for r, reason in focused], min(p.recent_chars, p.pack_chars//2))
    focus_ids = {r['ref'] for r in focus}
    recall = bounded([{**render(r), 'relevance': reason} for r, reason in recalled if ids[r['id']] not in focus_ids],
                     min(1600, max(0, p.recent_chars-len(compact(focus)))))
    evidence_ids = focus_ids | {r['ref'] for r in recall}
    lookup['matched_messages'] = len(recall) + sum(ids[r['id']] in focus_ids for r, _ in recalled)
    context_rows = bounded([render(r) for r in reversed(recent) if ids[r['id']] not in evidence_ids],
                           max(0, p.recent_chars-len(compact(focus))-len(compact(recall))))
    context_rows.reverse()
    feedback_rows = bounded(feedback, min(700, p.style_chars//2))
    style_rows = bounded(style, p.style_chars-len(compact(feedback_rows)))
    member_cards = bounded(cards[:8], min(800, p.memory_chars//2))
    memory_rows = bounded(memory, max(0, p.memory_chars-len(compact(member_cards))))
    human = [r for r in rows if not r['self'] and not r['command']
             and r['sender'] not in window.style_excluded_senders and now-r['at'] <= p.style_minutes*60]
    stats = {'median_chars': statistics.median([len(plain_text(r['text'])) for r in human]) if human else 0,
             'example_median_chars': statistics.median([len(r['text']) for r in style_rows]) if style_rows else 0,
             'short_ratio': round(sum(len(plain_text(r['text'])) <= 20 for r in human)/max(1, len(human)), 2),
             'question_ratio': round(sum('?' in r['text'] or '？' in r['text'] for r in human)/max(1, len(human)), 2),
             'emoji_ratio': round(sum(r['attachment'] for r in human)/max(1, len(human)), 2),
             'confirmed_self_ratio': round(sum(r['self'] for r in rows)/max(1, len(rows)), 2)}
    pack = {'generated_at': int(now), 'current_message': render(current) if current else None,
            'current_speaker': aliases.get(current['member']) if current else None,
            'memory_request_status': memory_status, 'focus_context': focus, 'recent_context': context_rows,
            'recall_context': recall, 'memory_lookup': lookup,
            'member_cards': member_cards,
            'memory_snapshot': memory_rows, 'style_samples': style_rows, 'reply_feedback': feedback_rows,
            'rhythm_stats': stats if len(compact(stats)) <= p.stats_chars else {},
            'omitted_count': sum(ids[r['id']] not in evidence_ids for r in recent)-len(context_rows)+len(style)-len(style_rows)+len(memory)-len(memory_rows)}
    visual = [{'source_ref': ids.get(r['id'], 'current'), 'sender': aliases[r['member']], 'at': int(r['at']),
               **{k:v for k,v in note.items() if k != 'digest'}} for r in images for note in window.image_notes.get(r['id'], [])]
    if images:
        pack['visual_context'] = bounded(visual, p.vision_context_chars)
        pack['visual_context_incomplete'] = len(pack['visual_context']) != sum(r['image_count'] for r in images)
    lookup_names = identity_lookups(current, snapshot) if current else []
    if lookup_names:
        queries = [{'name': n['name'], 'status': n['status'],
                    'speakers': [accounts[a] for a in n['accounts'][:8] if a in accounts]} for n in lookup_names]
        by_account = {m['account']: m for m in snapshot.get('members', [])}
        for query, result in zip(queries, lookup_names):
            if result['accounts']:
                query['matches'] = [{'speaker': accounts[a], 'name': by_account[a]['name'],
                                     'automated': a in window.style_excluded_senders}
                                    for a in result['accounts'][:8] if a in accounts]
        if len(queries) == 1:
            pack['identity_lookup'] = queries[0]
        else:
            pack['identity_lookups'] = queries
    if pack['current_message']:
        pack['current_message']['text'] = pack['current_message']['text'][:p.pack_chars//3]
    # Total budget includes JSON overhead. Drop lowest priority fields first.
    if len(compact(pack)) > p.pack_chars:
        pack['rhythm_stats'] = {}
    for key in ('style_samples', 'reply_feedback', 'memory_snapshot', 'recent_context', 'member_cards', 'recall_context', 'focus_context', 'visual_context'):
        if key not in pack:
            continue
        while pack[key] and len(compact(pack)) > p.pack_chars:
            pack[key].pop(0 if key == 'recent_context' else -1)
            pack['omitted_count'] += 1
            if key == 'visual_context':
                pack['visual_context_incomplete'] = True
    lookup['matched_messages'] = len(pack['recall_context']) + sum(
        any(item['ref'] == ids[r['id']] for item in pack['focus_context']) for r, _ in recalled)
    return pack
