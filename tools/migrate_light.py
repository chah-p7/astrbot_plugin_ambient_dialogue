"""Offline, one-time Light export/import. Never edits AstrBot configuration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ambient.context import Route, compact
from ambient.policy import Policy, stable_id
from ambient.store import Store


TABLES = ('members', 'facts', 'candidates', 'relations', 'bot_state')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_light(source, destination, route, logical_group, old_bot, configs=()):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if Path(str(source)+'-wal').exists() and Path(str(source)+'-wal').stat().st_size:
        raise ValueError('Stop writers and checkpoint the source before export')
    before = checksum(source)
    db = sqlite3.connect(source.as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN')
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'source_corrupt'
        meta = dict(db.execute('SELECT * FROM meta WHERE id=1').fetchone())
        assert meta['group_id'] == logical_group, 'source_group_mismatch'
        tables = {t: [dict(r) for r in db.execute('SELECT * FROM '+t+' ORDER BY rowid')] for t in TABLES}
        summary = json.loads(meta['summary'])
    finally:
        db.close()
    assert checksum(source) == before, 'source_changed_during_export'
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, destination/'light-source.sqlite3')
    assert checksum(destination/'light-source.sqlite3') == before
    for i, config in enumerate(configs):
        shutil.copy2(config, destination/f'config-{i}-{Path(config).name}')
    bundle = {'schema_version': 1, 'source_sha256': before, 'route': asdict(route),
              'logical_group': logical_group, 'old_bot': old_bot, 'tables': tables,
              'summary': summary, 'source_revision': meta['revision']}
    bundle['export_sha256'] = digest(bundle)
    path = destination/'light-export.json'
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (destination/'bot-state-manual-review.json').write_text(json.dumps(tables['bot_state'], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    suggestion = {'manual_apply_only': True, 'ambient_group': {'__template_key': 'group', 'umo': route.umo,
        'account': route.account, 'enabled': True, 'capture_only': False, 'interject_enabled': True, 'quota_mb': 0},
        'remove_botlife_logical_group': logical_group,
        'target_profile_plugin_change': {'enable': 'astrbot_plugin_ambient_dialogue', 'disable': 'astrbot_plugin_suite'},
        'remove_retired_fields': ['group_scopes.*.chat_mode', 'group_scopes.*.light_*', 'memory.light_chat_*']}
    (destination/'configuration-suggestion.json').write_text(json.dumps(suggestion, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return path


def import_light(export, target_root, backup_dir, policy=None):
    bundle = json.loads(Path(export).read_text(encoding='utf-8'))
    export_hash = bundle.pop('export_sha256')
    assert bundle.get('schema_version') == 1 and digest(bundle) == export_hash, 'export_integrity_failed'
    route = Route(**bundle['route'])
    store = Store(Path(target_root)/'groups', route.key, policy or Policy())
    tables = bundle['tables']
    mapping = {r['id']: route.member(r['account']) for r in tables['members']}
    assert len(set(mapping.values())) == len(mapping), 'ambiguous_member_mapping'
    members = [{**r, 'id': mapping[r['id']], 'namespace': route.platform, 'seen': float(r['seen'])} for r in tables['members']]

    def fact(r):
        m = mapping[r['member']]
        return {**r, 'member': m, 'id': stable_id(m, r['kind'], r['key']), 'at': float(r['at']), 'expires': float(r.get('expires', 0))}
    facts = [fact(r) for r in tables['facts']]
    pending = [fact(r) for r in tables['candidates']]
    summary = [fact(r) for r in bundle['summary']]
    notes = [{'id': stable_id(mapping[r['member']], 'interaction_note', r['bot']),
        'member': mapping[r['member']], 'text': '；'.join(x for x in (r['familiarity'], r['trust'], r['note']) if x),
        'source': r['source'], 'at': float(r['at'])} for r in tables['relations'] if r['bot'] == bundle['old_bot']]
    expected = {'members': members, 'facts': facts, 'candidates': pending, 'summary': summary, 'interaction_notes': notes}
    report = {'source_sha256': bundle['source_sha256'], 'export_sha256': export_hash,
              'source_counts': {k: len(v) for k,v in tables.items()},
              'target_counts': {k: len(v) for k,v in expected.items()},
              'protected_facts': sum(r['kind'] in {'boundary', 'promise', 'todo'} for r in facts+pending+summary),
              'bot_state_imported': False, 'interjection_gate_imported': False,
              'mapping_sha256': digest(mapping), 'target_content_sha256': digest(expected)}
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT report FROM migrations WHERE source_hash=?', (export_hash,)).fetchone()
        if previous:
            return {**json.loads(previous[0]), 'already_imported': True}
        for table in ('members', 'facts', 'candidates', 'interaction_notes', 'migrations'):
            assert db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] == 0, 'target_memory_not_empty'
        assert store._summary(db) == [], 'target_summary_not_empty'
        assert db.execute('SELECT last_at FROM interjection_gate WHERE id=1').fetchone()[0] == 0, 'target_already_sending'
    backup = Path(backup_dir)
    backup.mkdir(parents=True, exist_ok=True)
    target_backup = backup/'ambient-before-import.sqlite3'
    assert not target_backup.exists(), 'target_backup_already_exists'
    with store.connection() as db:
        target = sqlite3.connect(target_backup)
        try:
            db.backup(target)
        finally:
            target.close()
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        # Revalidate under the committing transaction, protecting even accidental concurrent use.
        assert db.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 0, 'target_changed'
        for table in ('members', 'facts', 'candidates', 'interaction_notes'):
            for row in expected[table]:
                columns = list(row)
                assert set(columns) <= {r[1] for r in db.execute('PRAGMA table_info('+table+')')}, 'unexpected_export_column'
                db.execute('INSERT INTO '+table+' ('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')', list(row.values()))
        store._save_summary(db, summary)
        db.execute('UPDATE meta SET revision=? WHERE id=1', (bundle['source_revision']+1,))
        db.execute('INSERT INTO migrations VALUES(?,?,?)', (export_hash, time.time(), compact(report)))
        assert not db.execute('PRAGMA foreign_key_check').fetchall(), 'target_foreign_key_failure'
        for table in ('members', 'facts', 'candidates', 'interaction_notes'):
            actual = [dict(r) for r in db.execute('SELECT * FROM '+table)]
            assert digest(sorted(actual, key=lambda r:r['id'])) == digest(sorted(expected[table], key=lambda r:r['id'])), 'target_readback_mismatch'
        assert store._summary(db) == summary
    with store.connection() as db:
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'target_corrupt'
    assert checksum(Path(export).parent/'light-source.sqlite3') == bundle['source_sha256'], 'source_backup_changed'
    report['readback_verified'] = True
    (backup/'migration-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    for arg in ('source', 'destination', 'platform', 'account', 'group', 'adapter', 'logical-group', 'old-bot'):
        export.add_argument('--'+arg, required=True)
    export.add_argument('--config', action='append', default=[])
    imp = sub.add_parser('import')
    for arg in ('export', 'target-root', 'backup-dir'):
        imp.add_argument('--'+arg, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        print(export_light(args.source, args.destination, Route(args.platform, args.account, args.group, args.adapter), args.logical_group, args.old_bot, args.config))
    else:
        print(json.dumps(import_light(args.export, args.target_root, args.backup_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
