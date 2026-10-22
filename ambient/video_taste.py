"""Small group-local evidence of human video shares; no inferred person profiles."""
from collections import Counter
import json
import math
import time

from .learning import words
from .policy import stable_id
from .video_sources import BVID, references


def limit(options, key, default, low, high):
    value = options.get(key, default)
    try:
        return max(low, min(high, int(value))) if not isinstance(value, bool) else default
    except (TypeError, ValueError, OverflowError):
        return default


def features(video, note=None):
    terms = words(video.get('title', ''))
    result = {'word:'+w for w in terms}
    result.update('trait:'+s for s in (note or {}).get('traits', []))
    result.update(k+':'+video[k] for k in ('category', 'author') if video.get(k))
    return result


class TasteProfile:
    def __init__(self, examples, notes, now):
        self.examples = examples
        self.weights, self.support = Counter(), Counter()
        counts = Counter(e['member'] for e in examples)
        # Repeated copies of one video never become multiple independent votes.
        seen = set()
        for e in sorted(examples, key=lambda r:r['at'], reverse=True):
            video = e['video']
            key = (e['member'], video.get('id') or e['ref'])
            if key in seen:
                continue
            seen.add(key)
            weight = 2**(-max(0, now-e['at'])/(30*86400))/max(1, math.sqrt(counts[e['member']]))
            for term in features(video, notes.get(video.get('id'))):
                self.weights[term] += weight
                self.support[term] += 1
        self.size = len(seen)

    def affinity(self, video, note=None):
        score = 0
        for term in features(video, note):
            kind = term.partition(':')[0]
            scale = {'word': .35, 'trait': .8, 'category': .35, 'author': .5}[kind]
            score += scale*min(2, self.weights[term])*min(1, self.support[term]/3)
        return min(3, score)

    def seeds(self):
        result, senders = [], set()
        ordered = sorted(self.examples, key=lambda r:r['at'], reverse=True)
        for e in sorted(ordered, key=lambda r:sum(x['member']==r['member'] for x in ordered)):
            identity = e['video'].get('id', '')
            if BVID.fullmatch(identity) and identity not in result and e['member'] not in senders:
                result.append(identity)
                senders.add(e['member'])
                if len(result) == 3:
                    break
        return result


class VideoTaste:
    def __init__(self, store):
        self.store = store
        self.days, self.maximum = 90, 128

    def configure(self, options):
        self.days = limit(options, 'video_learning_days', 90, 7, 365)
        self.maximum = limit(options, 'video_learning_samples', 128, 16, 512)

    def create(self, db):
        db.executescript('''
            CREATE TABLE IF NOT EXISTS video_examples (
                ref TEXT, member TEXT, source TEXT, at REAL, payload TEXT,
                resolved INTEGER DEFAULT 0, retry REAL DEFAULT 0, PRIMARY KEY(ref,member));
            CREATE TABLE IF NOT EXISTS video_notes (
                bvid TEXT PRIMARY KEY, signature TEXT, at REAL, payload TEXT);
        ''')

    def trim(self, db, now, compress=False):
        maximum = max(16, self.maximum//2) if compress else self.maximum
        db.execute('DELETE FROM video_examples WHERE at<?', (now-self.days*86400,))
        db.execute('DELETE FROM video_examples WHERE rowid IN (SELECT rowid FROM video_examples '
                   'ORDER BY at DESC LIMIT -1 OFFSET ?)', (maximum,))
        db.execute('DELETE FROM video_notes WHERE at<?', (now-14*86400,))
        db.execute('DELETE FROM video_notes WHERE rowid IN (SELECT rowid FROM video_notes '
                   'ORDER BY at DESC LIMIT -1 OFFSET ?)', (maximum,))

    def observe(self, row, excluded=(), *, now=None):
        now = time.time() if now is None else now
        if (row.get('self') or row.get('command') or row.get('sender') in excluded
                or not row.get('member') or not 0 <= now-row['at'] <= self.days*86400):
            return 0
        refs = row.get('video_refs') or references(row.get('text', ''))
        if not refs:
            return 0
        with self.store.connection() as db:
            reset = db.execute('SELECT at FROM learning_resets WHERE member=?', (row['member'],)).fetchone()
            if reset and row['at'] <= reset[0]:
                return 0
            if self.store.size() >= self.store.policy.quota_bytes:
                return 0
            added = 0
            for item in refs[:3]:
                # The member/source are supplied by transport, never by a model.
                ref = item['ref']
                payload = {'id': ref if BVID.fullmatch(ref) else '', 'title': item.get('title', '')[:100]}
                added += db.execute('INSERT OR IGNORE INTO video_examples(ref,member,source,at,payload) VALUES(?,?,?,?,?)',
                    (ref, row['member'], stable_id(row['id']), row['at'], json.dumps(payload, ensure_ascii=False))).rowcount
            self.trim(db, now)
        return added

    def pending(self, now):
        with self.store.connection() as db:
            self.trim(db, now)
            return [dict(r) for r in db.execute('SELECT * FROM video_examples WHERE resolved=0 AND retry<=? '
                                               'ORDER BY at DESC LIMIT 8', (now,))]

    def resolve(self, row, video, *, now):
        with self.store.connection() as db:
            # UPDATE cannot recreate evidence deleted by "forget me" during I/O.
            if video and self.store.size() < self.store.policy.quota_bytes:
                db.execute('UPDATE video_examples SET payload=?,resolved=1 WHERE ref=? AND member=?',
                    (json.dumps(video, ensure_ascii=False), row['ref'], row['member']))
            else:
                db.execute('UPDATE video_examples SET retry=? WHERE ref=? AND member=?',
                           (now+6*3600, row['ref'], row['member']))

    def notes(self, candidates, *, now):
        with self.store.connection() as db:
            return {r['bvid']: json.loads(r['payload']) for r in db.execute('SELECT * FROM video_notes WHERE at>=?', (now-14*86400,))
                    if r['bvid'] in candidates and r['signature'] == stable_id(candidates[r['bvid']])}

    def save_notes(self, candidates, notes, *, now):
        with self.store.connection() as db:
            for identity, note in notes.items():
                if self.store.size() >= self.store.policy.quota_bytes:
                    break
                db.execute('INSERT OR REPLACE INTO video_notes VALUES(?,?,?,?)',
                           (identity, stable_id(candidates[identity]), now, json.dumps(note, ensure_ascii=False)))
            self.trim(db, now)

    def examples(self, *, now):
        with self.store.connection() as db:
            self.trim(db, now)
            return [{**dict(r), 'video': json.loads(r['payload'])} for r in db.execute('SELECT * FROM video_examples WHERE at>=?', (now-self.days*86400,))
                    if json.loads(r['payload']).get('title')]

    def forget(self, db, member):
        db.execute('DELETE FROM video_examples WHERE member=?', (member,))
