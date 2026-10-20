"""Compare frozen checkpoint/gate bundles on exactly the same manifest and policy.

Candidates JSON is a list of objects: name, checkpoint, gate_npz, gate_meta.
Paths are relative to that JSON file. No training or threshold fitting occurs.
"""
import argparse
import json
from pathlib import Path

import EVALUATE_TV3 as evaluator


def load_candidates(path):
    path = Path(path)
    candidates = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('Candidates must be a nonempty JSON list.')
    result, names = [], set()
    for item in candidates:
        name = item.get('name', '')
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Candidate names allow only letters, numbers, underscore and hyphen.')
        if name in names:
            raise ValueError('Candidate names must be unique.')
        names.add(name)
        resolved = {'name': name}
        for key in ('checkpoint', 'gate_npz', 'gate_meta'):
            resolved[key] = (path.parent / item[key]).resolve()
            if not resolved[key].is_file():
                raise ValueError(f'Missing {key} for {name}.')
        result.append(resolved)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--quality-config', type=Path, default=evaluator.DEFAULT_QUALITY_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Output already exists; choose a new directory.')
        candidates = load_candidates(args.candidates)
        evaluator.load_rows(args.manifest, args.split)
        args.output.mkdir(parents=True)
        rows = []
        for candidate in candidates:
            values = ['--root', str(args.root), '--manifest', str(args.manifest), '--split', args.split,
                      '--device', args.device, '--quality', '--quality-config', str(args.quality_config),
                      '--output', str(args.output / candidate['name'])]
            for flag, key in [('--checkpoint', 'checkpoint'), ('--gate-npz', 'gate_npz'), ('--gate-meta', 'gate_meta')]:
                values += [flag, str(candidate[key])]
            report = evaluator.run(evaluator.build_parser().parse_args(values))
            metric = report['metrics']
            rows.append(dict(name=candidate['name'], device=args.device,
                             checkpoint_sha256=report['checkpoint_sha256'],
                             manifest_sha256=report['manifest_sha256'],
                             cnn_only=metric['cnn_only'], policy=metric['policy'],
                             latency=metric['latency']))
        summary = dict(split=args.split, independent_final_test_verified=False, candidates=rows,
                       note='Comparison does not promote a model. Use external development data for selection; keep final data untouched.')
        (args.output / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Saved {len(rows)} candidate results to {args.output}')
    except Exception as exc:
        parser.exit(1, f'[ERROR] {exc}\n')


if __name__ == '__main__':
    main()
