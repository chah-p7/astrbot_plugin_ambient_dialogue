"""Offline pack replay and blinded draft review. No model calls or QQ sends."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys


def replay_case(case):
    from ambient.context import Window, build_pack, compact
    from ambient.policy import Policy

    rows = case['rows']
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or case['current_id'] not in ids:
        raise ValueError('Each message needs a unique ID and an existing current_id')
    if any(a['at'] > b['at'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Rows must follow observed chronological order')
    # Cut by position as well as timestamp: a later answer may have the same
    # second as the current question. Reference answers never enter the pack.
    current_index = ids.index(case['current_id'])
    visible = rows[:current_index+1]
    current = visible[-1]
    if case.get('reference_id') in ids[:current_index+1]:
        raise ValueError('The held-out answer must follow the current message')
    policy = Policy.from_config({'limits': case.get('limits', {})})
    window = Window(policy, visible, style_excluded_senders=case.get('excluded_senders', []))
    pack = build_pack(window, {}, current, now=current['at'])
    samples = pack['style_samples']
    refs = {r['ref'] for s in samples for r in [s, *s.get('nearby', []), *([s['lead_in']] if 'lead_in' in s else [])]}
    return {'id': case['id'], 'pack': pack, 'metrics': {
        'visible_rows': len(visible), 'held_out_rows': len(rows)-len(visible),
        'samples': len(samples), 'verified_pairs': sum('lead_in' in s for s in samples),
        'nearby_fragments': sum(bool(s.get('nearby')) for s in samples),
        'unique_source_refs': len(refs), 'pack_chars': len(compact(pack)),
    }}


def blind_review(cases, seed):
    """Keep the key separate; naturalness is a human judgement, not a length score."""
    rng = random.Random(seed)
    review, key, seen = [], [], set()
    for case in cases:
        if case['id'] in seen or len(case['variants']) != 2:
            raise ValueError('Use unique case IDs and exactly two draft variants')
        seen.add(case['id'])
        variants = list(case['variants'].items())
        rng.shuffle(variants)
        review.append({'id': case['id'], 'current': case['current'],
                       'reference': case.get('reference', ''),
                       'A': variants[0][1], 'B': variants[1][1],
                       'winner': '', 'notes': ''})
        key.append({'id': case['id'], 'A': variants[0][0], 'B': variants[1][0]})
    return review, key


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    replay = sub.add_parser('replay', help='Render cases using a reviewed local plugin checkout')
    replay.add_argument('cases', type=Path)
    replay.add_argument('output', type=Path)
    replay.add_argument('--plugin-root', type=Path, default=Path(__file__).resolve().parents[1])
    blind = sub.add_parser('blind', help='Shuffle two supplied drafts; write review and a separate answer key')
    blind.add_argument('cases', type=Path)
    blind.add_argument('output', type=Path)
    blind.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8-sig'))
    if args.mode == 'replay':
        sys.path.insert(0, str(args.plugin_root.resolve()))
        from ambient.policy import SYSTEM_RULES
        records = [replay_case(case) for case in cases]
        write_json(args.output, {'system_rules': SYSTEM_RULES, 'cases': records})
    else:
        review, key = blind_review(cases, args.seed)
        write_json(args.output, review)
        write_json(args.output.with_name(args.output.stem+'.key.json'), key)
    print(json.dumps({'mode': args.mode, 'cases': len(cases), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
