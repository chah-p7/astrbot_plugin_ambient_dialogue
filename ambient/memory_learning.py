"""Bounded alias evidence and human examples inside the existing group database."""
import json
import time

from .aliases import name_key, observations, valid_name
from .alias_evidence import contains_name, human_text, source_row
from .examples import CORRECTION, case_key, extract_cases
from .learning import quote_source
from .policy import stable_id


class MemoryLearning:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def create(db):
        db.executescript('''
            CREATE TABLE IF NOT EXISTS aliases (
                id TEXT PRIMARY KEY, member TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
                key TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
                at REAL NOT NULL, source TEXT NOT NULL, evidence TEXT NOT NULL,
                UNIQUE(member,key));
            CREATE TABLE IF NOT EXISTS style_cases (
                id TEXT PRIMARY KEY, at REAL NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS learning_resets (member TEXT PRIMARY KEY, at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS alias_scan (id INTEGER PRIMARY KEY CHECK(id=1), at REAL NOT NULL, source TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS alias_scenes (
                id TEXT PRIMARY KEY, key TEXT NOT NULL, at REAL NOT NULL, payload TEXT NOT NULL);
        ''')

    def record_alias(self, db, member, name, kind, at, source, witness, basis='', support=()):
        if not valid_name(name, platform=kind == 'platform'):
            return False
        key, ident = name_key(name), stable_id(member, 'alias', name_key(name))
        previous = db.execute('SELECT * FROM aliases WHERE id=?', (ident,)).fetchone()
        if previous:
            if ((at < previous['at'] and kind != 'candidate')
                    or (previous['kind'] == 'rejected' and kind not in {'self', 'rejected'})):
                return False
            if previous['kind'] in {'self', 'platform', 'reviewed'} and kind == 'candidate':
                return False
            if previous['kind'] == 'self' and kind == 'platform':
                return False
            if previous['kind'] == 'community' and kind == 'candidate' and basis == 'context':
                return False
        if kind != 'rejected' and self.store.size() >= self.store.policy.quota_bytes:
            return False
        evidence = json.loads(previous['evidence']) if previous else []
        if any(e['source'] == source and e['by'] == witness for e in evidence):
            return False
        entry = {'source': source, 'by': witness, 'at': at}
        if basis:
            entry['basis'] = basis
            entry['support'] = [{**{k: r[k] for k in ('member', 'source', 'at')},
                                 **({'row': r['row']} if 'row' in r else {})} for r in support[:8]]
        evidence.append(entry)
        evidence = sorted(evidence, key=lambda e: e['at'])[-6:]
        if kind == 'candidate':
            kind = self.evidence_kind(evidence)
        recorded_at = max(at, previous['at'] if previous else at)
        value = json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))
        db.execute('INSERT OR REPLACE INTO aliases VALUES(?,?,?,?,?,?,?,?)',
                   (ident, member, key, name, kind, recorded_at, source[:256], value))
        if kind == 'rejected':
            # A denial beats both an old saved address and its pending copy.
            for table in ('facts', 'candidates'):
                for row in db.execute('SELECT id,text FROM '+table+" WHERE member=? AND kind='address'", (member,)).fetchall():
                    if name_key(row['text']) == key:
                        db.execute('DELETE FROM '+table+' WHERE id=?', (row['id'],))
        return True

    @staticmethod
    def accepted(evidence):
        return (len(evidence) >= 3 and len({e['by'] for e in evidence}) >= 2
                and max(e['at'] for e in evidence)-min(e['at'] for e in evidence) >= 60)

    @classmethod
    def evidence_kind(cls, evidence):
        explicit = [e for e in evidence if e.get('basis') != 'context']
        if cls.accepted(explicit):
            return 'community'
        if (len({e['source'] for e in evidence}) >= 2 and len({e['by'] for e in evidence}) >= 2
                and max(e['at'] for e in evidence)-min(e['at'] for e in evidence) >= 60):
            return 'inferred' if any(e.get('basis') == 'context' for e in evidence) else 'candidate'
        return 'candidate'

    def cursor(self):
        with self.store.connection() as db:
            row = db.execute('SELECT at,source FROM alias_scan WHERE id=1').fetchone()
            return list(row) if row else [0, '']

    def context_rows(self):
        """Only bounded original snippets already used as alias evidence."""
        with self.store.connection() as db:
            evidence = [json.loads(r[0]) for r in db.execute("SELECT evidence FROM aliases WHERE kind!='rejected'")]
            scenes = [json.loads(r[0]) for r in db.execute('SELECT payload FROM alias_scenes')]
        rows = {s['row']['id']: s['row'] for group in evidence for e in group
                for s in e.get('support', []) if isinstance(s.get('row'), dict)}
        rows.update({r['id']: r for scene in scenes for r in scene})
        return sorted(rows.values(), key=lambda r: (r['at'], r['id']))

    def archive_scenes(self, db, batch, resets, now):
        """Keep unresolved name evidence so another day's response can clarify it."""
        rows = [r for r in batch['rows'] if human_text(r, batch['excluded'])
                and 0 <= now-r['at'] <= self.store.policy.alias_days*86400]
        changed = 0
        for name in batch.get('review_names', []):
            anchors = [r for r in rows if contains_name(r['text'], name)][-3:]
            for anchor in anchors:
                nearby = [r for r in rows if abs(r['at']-anchor['at']) <= 180]
                source = quote_source(anchor, rows)[0]
                if source:
                    nearby.append(source)
                nearby = sorted({r['id']: r for r in nearby}.values(),
                                key=lambda r: (r['id'] != anchor['id'], r != source, abs(r['at']-anchor['at'])))[:8]
                if any(r['at'] <= resets.get(r['member'], 0) for r in nearby):
                    continue
                ident = stable_id(name_key(name), anchor['id'])
                payload = json.dumps([source_row(r, rows) for r in nearby], ensure_ascii=False, separators=(',', ':'))
                old = db.execute('SELECT payload FROM alias_scenes WHERE id=?', (ident,)).fetchone()
                if old and old[0] == payload:
                    continue
                if self.store.size() >= self.store.policy.quota_bytes:
                    break
                db.execute('INSERT OR REPLACE INTO alias_scenes VALUES(?,?,?,?)',
                           (ident, name_key(name), anchor['at'], payload))
                changed += 1
        return changed

    def review_aliases(self, mappings, source, *, now=None):
        """Explicit operator verification, never callable from model proposals."""
        now = time.time() if now is None else now
        if not isinstance(source, str) or not source.strip():
            raise ValueError('alias_review_source_required')
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            members = {r[0] for r in db.execute('SELECT id FROM members')}
            if any(m['member'] not in members or not valid_name(m['name']) for m in mappings):
                raise ValueError('invalid_alias_review')
            changed = sum(self.record_alias(db, m['member'], m['name'], 'reviewed', now,
                                           source, 'operator') for m in mappings)
            self.trim(db, now)
            if changed:
                db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
        return changed

    def apply_context(self, items, batch, *, now):
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT revision FROM meta WHERE id=1').fetchone()[0] != batch['revision']:
                return 'memory_changed_retry'
            if self.store.size() >= self.store.policy.quota_bytes:
                return 'quota_paused'
            members = {r[0] for r in db.execute('SELECT id FROM members')}
            resets = dict(db.execute('SELECT member,at FROM learning_resets'))
            def usable(item):
                evidence = [{'member': item['member'], 'at': item['at']},
                            {'member': item['witness'], 'at': item['at']}, *item.get('support', [])]
                return (item['member'] in members and 0 <= now-item['at'] <= self.store.policy.alias_days*86400
                        and all(e['at'] > resets.get(e['member'], 0) for e in evidence))
            items = [item for item in items if usable(item)]
            changed = 0
            if batch.get('review_names'):
                # Only retract automatic evidence actually revisited in this pack.
                # Unseen old evidence, explicit names and operator corrections survive.
                covered = {r['id'] for r in batch['rows']}
                for name in batch['review_names']:
                    targets = {i['member'] for i in items if name_key(i['name']) == name_key(name)}
                    if len(targets) != 1:
                        continue
                    for old in db.execute("SELECT * FROM aliases WHERE key=? AND kind IN ('candidate','inferred')",
                                          (name_key(name),)).fetchall():
                        if old['member'] in targets:
                            continue
                        evidence = json.loads(old['evidence'])
                        kept = [e for e in evidence if e.get('basis') != 'context' or e['source'] not in covered]
                        if kept == evidence:
                            continue
                        if kept:
                            db.execute('UPDATE aliases SET evidence=?,kind=? WHERE id=?',
                                       (json.dumps(kept), self.evidence_kind(kept), old['id']))
                        else:
                            db.execute('DELETE FROM aliases WHERE id=?', (old['id'],))
                        changed += 1
            for item in items:
                changed += self.record_alias(db, **item)
            archived = self.archive_scenes(db, batch, resets, now)
            self.trim(db, now)
            if changed or archived:
                db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
            old = db.execute('SELECT at,source FROM alias_scan WHERE id=1').fetchone()
            if old is None or tuple(batch['cursor']) > tuple(old):
                db.execute('INSERT OR REPLACE INTO alias_scan VALUES(1,?,?)', tuple(batch['cursor']))
            return 'learned' if changed else 'evidence_saved' if archived else 'no_new_alias'

    def learn(self, rows, *, now=None, excluded=(), cases=True):
        now = time.time() if now is None else now
        p = self.store.policy
        retention = p.raw_hours*3600 if p.raw_hours else max(p.recent_minutes, p.style_minutes)*60
        rows = sorted((r for r in rows if 0 <= now-r['at'] <= retention), key=lambda r: r['at'])[-p.raw_limit:]
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            before = db.total_changes
            members = [dict(r) for r in db.execute('SELECT * FROM members')]
            by_id = {m['id']: m for m in members}
            resets = dict(db.execute('SELECT member,at FROM learning_resets'))
            for r in rows:
                member = by_id.get(r['member'])
                if (member and r['at'] > resets.get(member['id'], 0) and not r['self'] and r['sender'] not in excluded and r.get('name')
                        and name_key(r['name']) != name_key(member['name'])):
                    self.record_alias(db, member['id'], r['name'], 'platform', r['at'], r['id'], member['id'])
            for item in observations(rows, members, excluded=excluded):
                if item['at'] > max(resets.get(item['member'], 0), resets.get(item['witness'], 0)):
                    self.record_alias(db, **item)
            for table in ('facts', 'candidates'):
                for r in db.execute('SELECT * FROM '+table+" WHERE kind='address'").fetchall():
                    if r['at'] >= now-p.alias_days*86400:
                        self.record_alias(db, r['member'], r['text'], 'self', r['at'], r['source'], r['member'])
            if cases and p.style_archive_limit and self.store.size() < p.quota_bytes:
                for case in extract_cases(rows, p, excluded=excluded):
                    if any(r['at'] <= resets.get(r['member'], 0) for r in case['rows']):
                        continue
                    if self.store.size() >= p.quota_bytes:
                        break
                    payload = json.dumps(case, ensure_ascii=False, separators=(',', ':'))
                    old = db.execute('SELECT payload FROM style_cases WHERE id=?', (case['id'],)).fetchone()
                    if not old or old[0] != payload:
                        db.execute('INSERT OR REPLACE INTO style_cases VALUES(?,?,?)', (case['id'], case['at'], payload))
            if cases:
                self.invalidate_cases(db, rows, excluded)
            self.trim(db, now, excluded=excluded)
            changed = db.total_changes-before
            if changed:
                db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
        return changed

    @staticmethod
    def invalidate_cases(db, rows, excluded):
        cases = [json.loads(r[0]) for r in db.execute('SELECT payload FROM style_cases')]
        sources = {r['id']: r for c in cases for r in c['rows']}
        sources.update({r['id']: r for r in rows})
        corrected = set()
        for row in rows:
            if row['self'] or row.get('command') or row['sender'] in excluded or not CORRECTION.search(row['text']):
                continue
            source, _ = quote_source(row, list(sources.values()))
            if source:
                corrected.add(source['id'])
        for case in cases:
            if case['rows'][1]['id'] in corrected:
                db.execute('DELETE FROM style_cases WHERE id=?', (case['id'],))

    def trim(self, db, now, *, excluded=(), compress=False):
        p = self.store.policy
        db.execute('DELETE FROM learning_resets WHERE at<?', (now-max(p.alias_days*86400, p.raw_hours*3600),))
        # Rejected labels last as long as the profile, preventing old evidence
        # from recreating a corrected name. All categories remain count bounded.
        db.execute("DELETE FROM aliases WHERE kind!='rejected' AND at<?", (now-p.alias_days*86400,))
        db.execute('''DELETE FROM aliases WHERE id IN (
            SELECT id FROM (SELECT id,ROW_NUMBER() OVER(PARTITION BY member,
                CASE WHEN kind='rejected' THEN 'rejected' WHEN kind='candidate' THEN 'candidate' ELSE 'accepted' END
                ORDER BY CASE kind WHEN 'self' THEN 0 WHEN 'platform' THEN 1 ELSE 2 END,at DESC,id) AS n
                FROM aliases) WHERE n>?)''', (p.alias_limit,))
        db.execute("DELETE FROM aliases WHERE kind='candidate' AND id NOT IN (SELECT id FROM aliases WHERE kind='candidate' ORDER BY at DESC LIMIT ?)", (p.candidate_limit,))
        db.execute('DELETE FROM style_cases WHERE at<?', (now-p.style_archive_days*86400,))
        db.execute('DELETE FROM alias_scenes WHERE at<?', (now-p.alias_days*86400,))
        db.execute('''DELETE FROM alias_scenes WHERE id IN (SELECT id FROM (
            SELECT id,ROW_NUMBER() OVER(PARTITION BY key ORDER BY at DESC,id) AS n FROM alias_scenes) WHERE n>3)''')
        db.execute('DELETE FROM alias_scenes WHERE id NOT IN (SELECT id FROM alias_scenes ORDER BY at DESC LIMIT ?)',
                   (p.candidate_limit,))
        seen = set()
        for row in db.execute('SELECT id,payload FROM style_cases ORDER BY at DESC,id').fetchall():
            case = json.loads(row['payload'])
            key = case_key(case)
            if key in seen or any(r['sender'] in excluded for r in case['rows']):
                db.execute('DELETE FROM style_cases WHERE id=?', (row['id'],))
            else:
                seen.add(key)
        db.execute('DELETE FROM style_cases WHERE id NOT IN (SELECT id FROM style_cases ORDER BY at DESC,id LIMIT ?)',
                   (p.style_archive_limit,))
        if compress:
            db.execute("DELETE FROM aliases WHERE kind='candidate'")
            db.execute('DELETE FROM alias_scenes')
            db.execute('DELETE FROM style_cases')

    @staticmethod
    def snapshot(db, member=None):
        aliases = [dict(r) for r in db.execute('SELECT id,member,key,name,kind,at,source FROM aliases'+
                    ('' if member is None else ' WHERE member=?')+' ORDER BY at DESC', () if member is None else (member,))]
        cases = [json.loads(r[0]) for r in db.execute('SELECT payload FROM style_cases ORDER BY at DESC')]
        if member is not None:
            cases = [c for c in cases if any(r['member'] == member for r in c['rows'])]
        return {'aliases': aliases, 'style_cases': cases}

    def forget(self, db, member, now):
        db.execute('INSERT OR REPLACE INTO learning_resets VALUES(?,?)', (member, now))
        for row in db.execute('SELECT id,payload FROM alias_scenes').fetchall():
            if any(r['member'] == member for r in json.loads(row['payload'])):
                db.execute('DELETE FROM alias_scenes WHERE id=?', (row['id'],))
        for row in db.execute('SELECT id,payload FROM style_cases').fetchall():
            if any(r['member'] == member for r in json.loads(row['payload'])['rows']):
                db.execute('DELETE FROM style_cases WHERE id=?', (row['id'],))
        for row in db.execute("SELECT * FROM aliases WHERE member!=? AND kind IN ('candidate','community','inferred')", (member,)).fetchall():
            old = json.loads(row['evidence'])
            evidence = [e for e in old if e['by'] != member and not any(s['member'] == member for s in e.get('support', []))]
            if not evidence:
                db.execute('DELETE FROM aliases WHERE id=?', (row['id'],))
            elif evidence != old:
                db.execute('UPDATE aliases SET evidence=?,kind=? WHERE id=?',
                           (json.dumps(evidence), self.evidence_kind(evidence), row['id']))
