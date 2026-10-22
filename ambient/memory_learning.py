"""Bounded alias evidence and human examples inside the existing group database."""
import json
import time

from .aliases import name_key, observations, valid_name
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
        ''')

    def record_alias(self, db, member, name, kind, at, source, witness):
        if not valid_name(name, platform=kind == 'platform'):
            return False
        key, ident = name_key(name), stable_id(member, 'alias', name_key(name))
        previous = db.execute('SELECT * FROM aliases WHERE id=?', (ident,)).fetchone()
        if previous:
            if at < previous['at'] or (previous['kind'] == 'rejected' and kind not in {'self', 'rejected'}):
                return False
            if previous['kind'] in {'self', 'platform'} and kind == 'candidate':
                return False
            if previous['kind'] == 'self' and kind == 'platform':
                return False
        if kind != 'rejected' and self.store.size() >= self.store.policy.quota_bytes:
            return False
        evidence = json.loads(previous['evidence']) if previous else []
        if any(e['source'] == source and e['by'] == witness for e in evidence):
            return False
        evidence.append({'source': source, 'by': witness, 'at': at})
        evidence = evidence[-6:]
        if kind == 'candidate' and self.accepted(evidence):
            kind = 'community'
        value = json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))
        db.execute('INSERT OR REPLACE INTO aliases VALUES(?,?,?,?,?,?,?,?)',
                   (ident, member, key, name, kind, at, source[:256], value))
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
        db.execute('DELETE FROM learning_resets WHERE at<?', (now-max(86400, p.raw_hours*3600),))
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
        for row in db.execute('SELECT id,payload FROM style_cases').fetchall():
            if any(r['member'] == member for r in json.loads(row['payload'])['rows']):
                db.execute('DELETE FROM style_cases WHERE id=?', (row['id'],))
        for row in db.execute("SELECT * FROM aliases WHERE member!=? AND kind IN ('candidate','community')", (member,)).fetchall():
            old = json.loads(row['evidence'])
            evidence = [e for e in old if e['by'] != member]
            if not evidence:
                db.execute('DELETE FROM aliases WHERE id=?', (row['id'],))
            elif evidence != old:
                db.execute('UPDATE aliases SET evidence=?,kind=? WHERE id=?',
                           (json.dumps(evidence), 'community' if self.accepted(evidence) else 'candidate', row['id']))
