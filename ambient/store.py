from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time

from .policy import Policy, normalized, stable_id


PROTECTED = frozenset({'boundary', 'promise', 'todo'})
KINDS = PROTECTED | {'address', 'preference', 'fact'}


class Store:
    """One physical database per native platform/account/group, with no raw conversation log.

    Summary entries retain stable fact IDs so correction/deletion also removes
    compressed copies. A transaction owns summary replacement and source purge;
    no model result can race with a correction and resurrect old information.
    """

    def __init__(self, root: Path, group: str, policy: Policy):
        self.group, self.policy = group, policy
        self.directory = Path(root) / stable_id(group)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'state.sqlite3'
        self.lock = RLock()
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS meta (
                    id INTEGER PRIMARY KEY CHECK(id=1), group_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '[]', revision INTEGER NOT NULL DEFAULT 0,
                    paused INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS members (
                    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, account TEXT NOT NULL,
                    name TEXT NOT NULL, seen REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY, member TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, key TEXT NOT NULL, text TEXT NOT NULL,
                    source TEXT NOT NULL, quote TEXT NOT NULL, at REAL NOT NULL,
                    expires REAL NOT NULL DEFAULT 0,
                    UNIQUE(member, kind, key));
                CREATE INDEX IF NOT EXISTS fact_member ON facts(member,kind);
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, member TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, key TEXT NOT NULL, text TEXT NOT NULL,
                    source TEXT NOT NULL, quote TEXT NOT NULL, at REAL NOT NULL,
                    expires REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS interaction_notes (
                    id TEXT PRIMARY KEY, member TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
                    text TEXT NOT NULL, source TEXT NOT NULL, at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS raw_messages (
                    id TEXT PRIMARY KEY, at REAL NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS raw_time ON raw_messages(at);
                CREATE TABLE IF NOT EXISTS migrations (
                    source_hash TEXT PRIMARY KEY, at REAL NOT NULL, report TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS interjection_gate (
                    id INTEGER PRIMARY KEY CHECK(id=1), last_at REAL NOT NULL DEFAULT 0,
                    anchor TEXT NOT NULL DEFAULT '', reply_hash TEXT NOT NULL DEFAULT '');
                INSERT OR IGNORE INTO interjection_gate(id) VALUES(1);
            ''')
            db.execute('INSERT OR IGNORE INTO meta(id,group_id) VALUES(1,?)', (group,))
            if db.execute('SELECT group_id FROM meta').fetchone()[0] != group:
                raise ValueError('Ambient群标识不匹配')

    @contextmanager
    def connection(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA secure_delete=ON')
            try:
                with db:
                    yield db
            finally:
                db.close()

    def size(self):
        return sum(p.stat().st_size for p in self.directory.iterdir()
                   if p.is_file() and not p.is_symlink())

    def identify(self, namespace, account, name, *, now=None):
        if not namespace or not account:
            raise ValueError('缺少稳定平台成员标识')
        now = time.time() if now is None else now
        member = stable_id(self.group, namespace, account)
        name = normalized(name)[:80] or str(account)[:80]
        with self.connection() as db:
            old = db.execute('SELECT * FROM members WHERE id=?', (member,)).fetchone()
            if not old and self.size() >= self.policy.quota_bytes:
                raise ValueError('本群记忆额度已满，暂不能新增成员档案；仍可聊天')
            # No per-message growth/history. A daily activity bucket is enough
            # for inactive-profile eviction; nick changes overwrite in place.
            if not old or old['name'] != name or now - old['seen'] >= 86400:
                db.execute('INSERT INTO members VALUES(?,?,?,?,?) ON CONFLICT(id) '
                           'DO UPDATE SET name=excluded.name,seen=excluded.seen',
                           (member, namespace, account, name, now))
        return member

    def _summary(self, db):
        return json.loads(db.execute('SELECT summary FROM meta WHERE id=1').fetchone()[0])

    def _save_summary(self, db, items):
        # This write must succeed before the source rows can be deleted.
        db.execute('UPDATE meta SET summary=? WHERE id=1',
                   (json.dumps(items, ensure_ascii=False, separators=(',', ':')),))

    def _bounded_summary(self, items, now):
        unique = {row['id']: row for row in items
                  if now - row['at'] < self.policy.topic_days * 86400}
        result, count = [], 0
        for row in sorted(unique.values(), key=lambda x: x['at'], reverse=True):
            if count + len(row['text']) > self.policy.summary_chars:
                continue
            result.append(row)
            count += len(row['text'])
        return result

    def enqueue(self, member, item, source, *, now=None):
        now = time.time() if now is None else now
        if any(not isinstance(item.get(k), str) for k in ('kind', 'key', 'text')):
            raise ValueError('kind、key 和 text 必须为文本')
        kind, key, value = item['kind'], normalized(item['key']), normalized(item['text'])
        if kind == 'address':
            key = 'address'
        if kind not in KINDS or not key or len(key) > 64 or not value or len(value) > self.policy.fact_chars:
            raise ValueError('记忆类型、键或内容无效')
        fact_id = stable_id(member, kind, key)
        with self.connection() as db:
            old = db.execute('SELECT text FROM facts WHERE id=?', (fact_id,)).fetchone()
            pending = db.execute('SELECT text FROM candidates WHERE id=?', (fact_id,)).fetchone()
            compressed = next((r for r in self._summary(db) if r['id'] == fact_id), None)
            if pending and pending[0] == value:
                return 'pending'
            if not pending and ((old and old[0] == value) or (compressed and compressed['text'] == value)):
                return 'unchanged'
            previous = pending or old
            if self.size() >= self.policy.quota_bytes and not (previous and len(value.encode()) <= len(previous[0].encode())):
                raise ValueError('本群记忆已达硬上限；请先删除、完成旧记录或提高额度')
            if not pending and db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] >= self.policy.candidate_limit:
                raise ValueError('记忆候选队列已满，请稍后重试')
            db.execute('INSERT OR REPLACE INTO candidates VALUES(?,?,?,?,?,?,?,?,?)',
                       (fact_id, member, kind, key, value, str(source)[:128],
                        normalized(item.get('quote', value))[:self.policy.fact_chars], now, 0))
            db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
        return 'pending'

    def _trim_preferences(self, db):
        db.execute('''DELETE FROM facts WHERE id IN (
            SELECT id FROM (SELECT id,ROW_NUMBER() OVER(
                PARTITION BY member ORDER BY at DESC,id) AS n FROM facts WHERE kind='preference')
            WHERE n>?)''', (self.policy.preference_limit,))

    def _apply_batch(self, db):
        rows = db.execute('SELECT * FROM candidates ORDER BY at LIMIT ?', (self.policy.batch_size,)).fetchall()
        if not rows:
            return 0
        summary = self._summary(db)
        for row in rows:
            db.execute('INSERT OR REPLACE INTO facts VALUES(?,?,?,?,?,?,?,?,?)', tuple(row))
            summary = [r for r in summary if r['id'] != row['id']]
            db.execute('DELETE FROM candidates WHERE id=?', (row['id'],))
        self._save_summary(db, summary)
        self._trim_preferences(db)
        return len(rows)

    def maintain(self, *, force=False, now=None):
        now = time.time() if now is None else now
        with self.lock:
            before = self.size()
            compress = force or before >= self.policy.quota_bytes * self.policy.compress_ratio
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                original = db.total_changes
                self._trim_raw(db, now)
                if compress:
                    db.execute('DELETE FROM raw_messages')
                applied = self._apply_batch(db)
                # An explicit expiry may retire a protected commitment. No
                # heuristic guesses an expiry or silently completes a todo.
                db.execute('DELETE FROM facts WHERE expires>0 AND expires<=?', (now,))
                self._trim_preferences(db)
                old = self._summary(db)
                facts = [dict(r) for r in db.execute("SELECT * FROM facts WHERE kind='fact'")]
                summary = self._bounded_summary(old + facts, now)
                if facts or summary != old:
                    self._save_summary(db, summary)
                    db.execute("DELETE FROM facts WHERE kind='fact'")
                db.execute('''DELETE FROM members WHERE seen<? AND NOT EXISTS
                    (SELECT 1 FROM facts WHERE member=members.id AND kind IN ('boundary','promise','todo'))
                    AND NOT EXISTS(SELECT 1 FROM candidates WHERE member=members.id)''',
                           (now - self.policy.inactive_days * 86400,))
                # Cascade deletes cover relational indexes. Summary copies are
                # also removed before reclaiming the actual SQLite file.
                members = {r[0] for r in db.execute('SELECT id FROM members')}
                filtered = [r for r in summary if r['member'] in members]
                if filtered != summary:
                    self._save_summary(db, filtered)
                changed = db.total_changes != original
                if changed:
                    db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
            self._cleanup_artifacts(now)
            if changed or compress:
                self._vacuum()
            if compress:
                # First expired/duplicate information above, then oldest
                # unprotected profiles. Preserve all protected facts and their
                # identity/source even if the target cannot be attained.
                while self.size() > self.policy.quota_bytes * self.policy.target_ratio:
                    with self.connection() as db:
                        removable = db.execute('''SELECT id FROM members WHERE NOT EXISTS
                            (SELECT 1 FROM facts WHERE member=members.id AND kind IN ('boundary','promise','todo'))
                            AND NOT EXISTS(SELECT 1 FROM candidates WHERE member=members.id)
                            ORDER BY seen LIMIT 32''').fetchall()
                        if not removable:
                            break
                        ids = {r[0] for r in removable}
                        self._save_summary(db, [r for r in self._summary(db) if r['member'] not in ids])
                        db.executemany('DELETE FROM members WHERE id=?', [(i,) for i in ids])
                        db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
                    self._vacuum()
            paused = self.size() >= self.policy.quota_bytes
            with self.connection() as db:
                db.execute('UPDATE meta SET paused=? WHERE id=1 AND paused<>?', (int(paused), int(paused)))
            return {'before_bytes': before, 'after_bytes': self.size(), 'applied': applied,
                    'paused': paused, 'target_reached': self.size() <= self.policy.quota_bytes * self.policy.target_ratio}

    def _vacuum(self):
        with self.connection() as db:
            # DELETE journaling has no enduring WAL. VACUUM rebuilds all B-tree
            # indexes and releases free pages, unlike a logical DELETE alone.
            db.execute('VACUUM')

    def _cleanup_artifacts(self, now):
        # Only our explicitly named, expired single-file artifacts. Never
        # traverse links or touch detailed-mode data, host logs or global backups.
        for path in self.directory.iterdir():
            if (path.is_file() and not path.is_symlink()
                    and path.name.startswith('ambient-') and path.suffix in {'.bak', '.tmp', '.log'}
                    and now - path.stat().st_mtime >= 86400):
                path.unlink()



    def interjection_gate(self):
        with self.connection() as db:
            row = dict(db.execute('SELECT * FROM interjection_gate WHERE id=1').fetchone())
        return row

    def claim_interjection(self, anchor, reply_hash, revision, *, now=None):
        """Persist only two hashes and the last attempt time, never chats.

        A claim precedes the transport side effect and survives uncertain sends
        and restarts. It cannot be undone by a generic boolean send result.
        """
        now = time.time() if now is None else now
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM interjection_gate WHERE id=1').fetchone()
            if (db.execute('SELECT revision FROM meta').fetchone()[0] != revision
                    or now - row['last_at'] < self.policy.interject_cooldown_seconds
                    or anchor == row['anchor'] or reply_hash == row['reply_hash']):
                return False
            db.execute('UPDATE interjection_gate SET last_at=?,anchor=?,reply_hash=? WHERE id=1',
                       (now, anchor, reply_hash))
        return True

    def _trim_raw(self, db, now):
        db.execute('DELETE FROM raw_messages WHERE at<?',
                   (now - self.policy.raw_hours * 3600,))
        if not self.policy.raw_hours:
            db.execute('DELETE FROM raw_messages')
        db.execute('DELETE FROM raw_messages WHERE id IN (SELECT id FROM raw_messages '
                   'ORDER BY at DESC,id LIMIT -1 OFFSET ?)', (self.policy.raw_limit,))

    def append_raw(self, message, *, now):
        with self.connection() as db:
            self._trim_raw(db, now)
            if self.policy.raw_hours and self.size() < self.policy.quota_bytes:
                db.execute('INSERT OR IGNORE INTO raw_messages VALUES(?,?,?)',
                           (message['id'], message['at'], json.dumps(message, ensure_ascii=False)))
                self._trim_raw(db, now)

    def load_raw(self, *, now):
        with self.connection() as db:
            self._trim_raw(db, now)
            return [json.loads(r[0]) for r in db.execute(
                'SELECT payload FROM raw_messages ORDER BY at,id')]

    def snapshot(self, member=None):
        with self.connection() as db:
            meta = dict(db.execute('SELECT * FROM meta').fetchone())
            result = {**meta, 'summary': json.loads(meta['summary'])}
            for table in ('members', 'facts', 'candidates', 'interaction_notes'):
                column = 'id' if table == 'members' else 'member'
                result[table] = [dict(r) for r in db.execute('SELECT * FROM '+table+
                    ('' if member is None else ' WHERE '+column+'=?'),
                    () if member is None else (member,))]
            if member is not None:
                result['summary'] = [r for r in result['summary'] if r['member'] == member]
            result['raw_count'] = db.execute('SELECT COUNT(*) FROM raw_messages').fetchone()[0]
        return {**result, 'used_bytes': self.size(), 'quota_bytes': self.policy.quota_bytes,
                'paused': self.size() >= self.policy.quota_bytes}

    def edit(self, operation, values, *, member, now=None):
        now = time.time() if now is None else now
        if operation == 'set':
            if values.get('kind') == 'address':
                values = {**values, 'key': 'address'}
            self.enqueue(member, values, values.get('source', 'explicit-command'), now=now)
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT * FROM candidates WHERE id=?',
                    (stable_id(member, values['kind'], normalized(values['key'])),)).fetchone()
                if row:
                    self._save_summary(db, [r for r in self._summary(db) if r['id'] != row['id']])
                    db.execute('INSERT OR REPLACE INTO facts VALUES(?,?,?,?,?,?,?,?,?)', tuple(row))
                    db.execute('DELETE FROM candidates WHERE id=?', (row['id'],))
                    self._trim_preferences(db)
                    db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
            return 'committed'
        if operation not in {'delete', 'complete', 'forget_member'}:
            raise ValueError('未知记忆操作')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            fact_id = str(values.get('id', ''))
            for table in ('facts', 'candidates', 'interaction_notes'):
                db.execute('DELETE FROM '+table+' WHERE member=?'+
                    ('' if operation == 'forget_member' else ' AND id=?'),
                    (member,) if operation == 'forget_member' else (member, fact_id))
            self._save_summary(db, [r for r in self._summary(db) if not
                (r['member'] == member and (operation == 'forget_member' or r['id'] == fact_id))])
            if operation == 'forget_member':
                db.execute('DELETE FROM members WHERE id=?', (member,))
                # Delete this sender's raw cache too; runtime clears its RAM copy.
                for row in db.execute('SELECT id,payload FROM raw_messages').fetchall():
                    if json.loads(row['payload']).get('member') == member:
                        db.execute('DELETE FROM raw_messages WHERE id=?', (row['id'],))
            db.execute('UPDATE meta SET revision=revision+1 WHERE id=1')
        self._vacuum()
        return 'deleted'
