"""Bounded, source-only dialogue packs for alias discovery and cross-scene checks."""
import json
import re
from datetime import datetime, timezone

from .aliases import valid_name
from .learning import MENTION, TOPIC_CHANGE, addressing, mentions, plain_text, quote_source


def contains_name(text, name):
    return bool(re.search(re.escape(name) + r'(?![A-Za-z0-9_])', plain_text(text)))


def human_text(row, excluded=()):
    text = plain_text(row['text'])
    return (not row['self'] and row['sender'] not in excluded and not row.get('command') and not row.get('directed')
            and bool(re.sub(r'[\W_0-9]+', '', re.sub(r'\[[^]]*\]', '', text))))


def response_chains(selected, refs, names, excluded=()):
    """Observable turn structure, never a stored identity assignment."""
    hints = []
    for i, anchor in enumerate(selected):
        if not human_text(anchor, excluded) or not any(contains_name(anchor['text'], n) for n in names):
            continue
        later = [r for r in selected[i+1:] if human_text(r, excluded) and 0 <= r['at']-anchor['at'] <= 120]
        statements = [anchor, *[r for r in later if r['sender'] == anchor['sender']
                      and r['at']-anchor['at'] <= 30 and re.match(r'^[他她]', plain_text(r['text']))]]
        for answer in later:
            if answer['sender'] == anchor['sender'] or not re.fullmatch(r'对|是|是啊|没错|确实', plain_text(answer['text'])):
                continue
            continuations = [r for r in later if r['sender'] == answer['sender']
                and 0 < r['at']-answer['at'] <= 45 and re.match(r'^我', plain_text(r['text']))]
            if continuations:
                if any(TOPIC_CHANGE.search(r['text']) for r in later if r['at'] <= continuations[0]['at']):
                    continue
                statement = [r for r in statements if r['at'] <= answer['at']][-1]
                hints.append({'mention': refs[anchor['id']], 'statement': refs[statement['id']],
                              'answer': refs[answer['id']], 'continuation': refs[continuations[0]['id']],
                              'basis': 'agreement_then_first_person', 'status': 'unconfirmed'})
    return hints[:12]


def source_row(row, rows):
    """A short original snippet, not a model paraphrase or a member attribute."""
    item = {k: row[k] for k in ('id', 'sender', 'member', 'at', 'self')}
    item.update(text=row['text'][:280], mentions=mentions(row), command=False)
    source, _ = quote_source(row, rows)
    if source:
        item['quote'] = {'id': source['id'], 'sender': source['sender'], 'text': ''}
    return item


def render_batch(selected, rows, snapshot, policy, *, excluded=(), cursor=None, names=()):
    """Only the selected original messages are evidence; no saved alias mappings."""
    members = {m['account']: m for m in snapshot.get('members', [])}
    accounts = {a: 'u'+str(i+1) for i, a in enumerate(dict.fromkeys(
        [r['sender'] for r in selected] + [a for r in selected for a in mentions(r)]))}
    refs = {r['id']: 'm'+str(i+1) for i, r in enumerate(selected)}
    cards = [{'speaker': accounts[a], 'name': m['name']}
             for a, m in members.items() if a not in excluded and a in accounts]
    rendered, previous, scene, last = [], {}, 0, None
    for row in selected:
        if last is None or row['at']-last['at'] > policy.topic_minutes*60:
            scene += 1
        item = {'id': refs[row['id']], 'speaker': accounts[row['sender']],
                'name': members.get(row['sender'], {}).get('name', row.get('name', '')),
                'at': int(row['at']), 'scene': scene,
                'text': MENTION.sub(lambda m: '@'+accounts.get(m[1], 'unknown'), row['text'])[:280]}
        if row['self'] or row['sender'] in excluded:
            # Keep the reply edge, never teach the model another bot's guesses.
            item.update(text='[自动账号发言，不作为外号证据]', automated=True)
        source, basis = quote_source(row, rows)
        if source and source['id'] in refs:
            item.update(reply_to=refs[source['id']], reply_basis=basis,
                        quoted_speaker=accounts[source['sender']])
        elif row.get('quote'):
            item['unresolved_quote'] = True
        targets, basis = addressing(row, rows)
        if targets:
            item['addressing'] = {'to': [accounts.get(a, 'bot' if a == 'bot' else 'unknown') for a in targets],
                                  'basis': basis}
        prior = previous.get(row['sender'])
        if (prior and 0 <= row['at']-prior['at'] <= 45 and not row.get('quote')
                and not mentions(row) and not TOPIC_CHANGE.search(row['text'])):
            item['same_speaker_after'] = refs[prior['id']]
        rendered.append(item)
        previous[row['sender']], last = row, row
    data = {'members': cards, 'messages': rendered}
    if names:
        data['names_to_review'] = list(names)
        data['name_mentions'] = {n: [refs[r['id']] for r in selected
                                    if human_text(r, excluded) and contains_name(r['text'], n)] for n in names}
        data['response_chains'] = response_chains(selected, refs, names, excluded)
    return {'data': data, 'rows': selected, 'accounts': accounts, 'refs': refs,
            'excluded': list(excluded), 'review_names': list(names),
            'evidence_gap_seconds': policy.topic_minutes*60,
            'cursor': cursor or [selected[-1]['at'], selected[-1]['id']],
            'revision': snapshot.get('revision', 0)}


def pack_size(batch):
    return len(json.dumps(batch['data'], ensure_ascii=False, separators=(',', ':')))


def model_prompt(data):
    """Readable speaker turns; the model returns a source message, never an account ID."""
    names = {m['speaker']: m['name'] for m in data['members']}
    lines = ['候选真人：'+json.dumps(list(names.values()),ensure_ascii=False)]
    if data.get('names_to_review'):
        lines.append('待查称呼：'+json.dumps(data['names_to_review'],ensure_ascii=False))
    for row in data['messages']:
        stamp = datetime.fromtimestamp(row['at'],timezone.utc).strftime('%m-%d %H:%M:%S UTC')
        label = '[自动账号]' if row.get('automated') else row['name'] or row['speaker']
        line = f"{row['id']} | 场景{row['scene']} | {stamp} | {label}: "+json.dumps(row['text'],ensure_ascii=False)
        if row.get('reply_to'):
            line += '；引用 '+row['reply_to']+'，原作者 '+names.get(row.get('quoted_speaker'),'[自动账号或未建档成员]')
        if row.get('addressing'):
            line += '；对话对象 '+','.join(names.get(a,'[机器人或未建档成员]') for a in row['addressing']['to'])
        if row.get('same_speaker_after'):
            line += '；与 '+row['same_speaker_after']+' 同人连发'
        lines.append(line)
    if data.get('name_mentions'):
        lines.append('外号原话编号：'+json.dumps(data['name_mentions'],ensure_ascii=False))
    if data.get('response_chains'):
        lines.append('待核验的承接链：'+json.dumps(data['response_chains'],ensure_ascii=False))
    return '\n'.join(lines)


def prepare_batch(rows, snapshot, policy, *, excluded=(), after=None):
    rows = sorted(rows, key=lambda r: (r['at'], r['id']))
    after = tuple(after or (0, ''))
    fresh = [r for r in rows if (r['at'], r['id']) > after]
    if not fresh:
        return None
    useful = [r for r in rows if human_text(r, excluded)]
    new = [r for r in useful if (r['at'], r['id']) > after][:policy.alias_context_messages]
    if not new:
        return {**render_batch([], rows, snapshot, policy, excluded=excluded,
                               cursor=[fresh[-1]['at'], fresh[-1]['id']]), 'skip_model': True}
    overlap = [r for r in useful if (r['at'], r['id']) <= after and new[0]['at']-r['at'] <= policy.topic_minutes*60][-8:]
    while new:
        selected = overlap+new
        sources = [s for r in selected if (s := quote_source(r, rows)[0]) and
                   abs(r['at']-s['at']) <= policy.topic_minutes*60]
        selected = sorted({r['id']: r for r in [*sources, *selected]}.values(), key=lambda r: (r['at'], r['id']))
        cursor = [new[-1]['at'], new[-1]['id']]
        if not any((r['at'], r['id']) > tuple(cursor) for r in useful):
            cursor = [fresh[-1]['at'], fresh[-1]['id']]
        batch = render_batch(selected, rows, snapshot, policy, excluded=excluded, cursor=cursor)
        if pack_size(batch) <= policy.alias_context_chars:
            return batch
        # Old overlap must never starve fresh input or leave the cursor stuck.
        if overlap:
            overlap.pop(0)
        else:
            new.pop()
    return None


def prepare_review(rows, snapshot, policy, names, *, excluded=(), cursor=None):
    """Round-robin surrounding turns across occurrences, instead of truncating old-to-new."""
    names = list(dict.fromkeys(n for n in names if isinstance(n, str) and valid_name(n)))[:12]
    rows = sorted({r['id']: r for r in rows}.values(), key=lambda r: (r['at'], r['id']))
    useful = [r for r in rows if human_text(r, excluded)]
    scenes = []
    for name in names:
        indices = [i for i, r in enumerate(useful) if contains_name(r['text'], name)]
        def strength(i):
            row = useful[i]
            source = quote_source(row, rows)[0]
            if re.search(r'你也是'+re.escape(name)+r'|今天你是'+re.escape(name), plain_text(row['text'])):
                return -1
            if row.get('directed') or (source and (source['self'] or source['sender'] in excluded)):
                return 0
            if source and source['sender'] != row['sender']:
                return 4
            if re.match(r'^你', plain_text(row['text'])) and any(s['sender'] == row['sender']
                and 0 <= row['at']-s['at'] <= 45 and addressing(s, rows)[0]
                for s in useful[max(0, i-4):i]):
                return 4
            if any(s['sender'] != row['sender'] and 0 < s['at']-row['at'] <= 90
                   and re.match(r'^(?:对[，,]?\s*)?我(?:发现|觉得|也|就|是|的|在)|^本人', plain_text(s['text']))
                   for s in useful[max(0, i-1):i+12]):
                return 3
            return 2 if mentions(row) else 1
        # Rank actual interaction edges, not the last/most talkative member.
        indices.sort(key=lambda i: (strength(i), useful[i]['at']), reverse=True)
        anchors = []
        for i in indices:
            if all(abs(useful[i]['at']-useful[j]['at']) >= 60 for j in anchors):
                anchors.append(i)
            if len(anchors) == 3:
                break
        for i in anchors:
            anchor = useful[i]
            near = [r for j, r in enumerate(useful) if -5 <= j-i <= 8
                    and abs(r['at']-anchor['at']) <= policy.topic_minutes*60]
            near.sort(key=lambda r: (abs(r['at']-anchor['at']), r['at']))
            source = quote_source(anchor, rows)[0]
            if source and abs(anchor['at']-source['at']) <= policy.topic_minutes*60:
                near = [anchor, source, *near]
            scenes.append(list({r['id']: r for r in near}.values()))
    all_rows = sorted({r['id']: r for scene in scenes for r in scene}.values(), key=lambda r: (r['at'], r['id']))
    if not all_rows:
        return None
    full = render_batch(all_rows, rows, snapshot, policy, excluded=excluded, cursor=cursor, names=names)
    rendered = dict(zip((r['id'] for r in all_rows), full['data']['messages']))
    selected = {}
    def subset():
        chosen = sorted(selected.values(), key=lambda r: (r['at'], r['id']))
        refs = {full['refs'][r['id']] for r in chosen}
        messages = []
        for row in chosen:
            item = dict(rendered[row['id']])
            for key in ('reply_to', 'same_speaker_after'):
                if item.get(key) and item[key] not in refs:
                    item.pop(key)
                    if key == 'reply_to':
                        item.pop('reply_basis', None)
                        item.pop('quoted_speaker', None)
                        item['unresolved_quote'] = True
            messages.append(item)
        data = {**full['data'], 'messages': messages,
                'name_mentions': {n: [m for m in ids if m in refs] for n, ids in full['data']['name_mentions'].items()},
                'response_chains': [c for c in full['data']['response_chains']
                                    if all(c[k] in refs for k in ('mention','statement','answer','continuation'))]}
        return {**full, 'data': data, 'rows': chosen, 'link_review': True}
    for depth in range(max(map(len, scenes), default=0)):
        for scene in scenes:
            if depth >= len(scene):
                continue
            row = scene[depth]
            if row['id'] in selected:
                continue
            selected[row['id']] = row
            if pack_size(subset()) > policy.alias_context_chars:
                del selected[row['id']]
    return subset() if selected else None
