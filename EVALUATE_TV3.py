"""Reproducible manifest-based CNN, gate and quality-policy evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil

import numpy as np
from dataclasses import asdict, replace
from pathlib import Path

from freshlens_ai.constants import CLASSES, FRUITS, STATUSES, PROJECT_DIR
from freshlens_ai.data.image_io import safe_path
from freshlens_ai.evaluation.tv3 import summarize
from freshlens_ai.inference.cnn_predict import FreshLensPredictor
from freshlens_ai.inference.quality import DEFAULT_QUALITY_CONFIG, load_quality_config, quality_config_hash
from freshlens_ai.utils.file_io import file_sha256

MODEL_DIR = PROJECT_DIR / 'models' / 'cnn_efficientnet_b0'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', required=True, help='Explicit manifest split, e.g. val, test, calibration.')
    parser.add_argument('--checkpoint', type=Path, default=MODEL_DIR / 'best.pt')
    parser.add_argument('--gate-npz', type=Path, default=MODEL_DIR / 'open_set_gate.npz')
    parser.add_argument('--gate-meta', type=Path, default=MODEL_DIR / 'open_set_gate.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quality-config', type=Path, default=DEFAULT_QUALITY_CONFIG)
    parser.add_argument('--quality', action='store_true', help='Enable experimental quality rejection.')
    parser.add_argument('--min-size', type=int, default=None)
    parser.add_argument('--blur-threshold', type=float, default=None)
    parser.add_argument('--dark-threshold', type=float, default=None)
    parser.add_argument('--bright-threshold', type=float, default=None)
    return parser


def load_rows(manifest, split):
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'path', 'fruit', 'status', 'split'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires path, fruit, status, split columns.')
        all_rows = list(reader)
    groups, paths, hashes = {}, set(), {}
    for row in all_rows:
        if row['path'] in paths:
            raise ValueError(f"Repeated manifest path: {row['path']}")
        paths.add(row['path'])
        if row.get('sha256'):
            digest = row['sha256']
            if digest in hashes and hashes[digest] != row['split']:
                raise ValueError(f"Image hash crosses splits: {row['path']}")
            hashes[digest] = row['split']
        if row.get('group_id'):
            group = row['group_id']
            if group in groups and groups[group] != row['split']:
                raise ValueError(f'Group crosses splits: {group}')
            groups[group] = row['split']
        if row['fruit'] not in (*FRUITS, 'other', 'unknown', 'unsupported'):
            raise ValueError(f"Invalid fruit label: {row['fruit']}. Use 'other' for unknown images.")
        if row['fruit'] in FRUITS and row['status'] not in STATUSES:
            raise ValueError(f"Invalid condition: {row['status']}")
        if row.get('quality_label', '') not in ('', 'good', 'bad'):
            raise ValueError('quality_label must be good, bad or empty.')
    rows = [r for r in all_rows if r['split'] == split]
    if not rows:
        raise ValueError(f'No records for split {split!r}.')
    return rows


def gallery_name(index, row, result):
    raw = f"{index:06d}_GT_{row['fruit']}_{row['status']}_PRED_{result['joint_class']}_{result['status']}"
    return re.sub(r'[^A-Za-z0-9_.-]', '_', raw)[:180] + '.png'


def run(args):
    if args.output.exists():
        raise ValueError('Output already exists; select a new directory to preserve earlier evidence.')
    rows = load_rows(args.manifest, args.split)
    overrides = {name: getattr(args, name) for name in
                 ('min_size', 'blur_threshold', 'dark_threshold', 'bright_threshold')
                 if getattr(args, name) is not None}
    config = replace(load_quality_config(args.quality_config), **overrides)
    predictor = FreshLensPredictor(args.checkpoint, args.gate_npz, args.device,
                                  gate_meta_path=args.gate_meta, quality_config=config)
    # Validate the complete selected population before producing any metrics.
    image_hashes = []
    for row in rows:
        path = safe_path(args.root, row['path'])
        digest = file_sha256(path)
        if row.get('sha256') and digest != row['sha256']:
            raise ValueError(f"Image hash mismatch: {row['path']}")
        image_hashes.append(digest)
    predictor.predict(safe_path(args.root, rows[0]['path']), check_quality=args.quality)
    records, gallery = [], []
    for index, (row, digest) in enumerate(zip(rows, image_hashes)):
        path = safe_path(args.root, row['path'])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Image changed during evaluation: {path}')
        # Decode/inference failures abort the run instead of shrinking the denominator.
        res = predictor.predict(data, check_quality=args.quality)
        known = row['fruit'] in FRUITS
        true_joint = f"{row['fruit']}::{row['status']}" if known else 'other'
        record = dict(path=row['path'], sha256=digest, group_id=row.get('group_id', ''),
                      source=row.get('source', ''), known=known, true_fruit=row['fruit'], true_condition=row['status'],
                      true_joint=true_joint, quality_label=row.get('quality_label', ''))
        for key in ('fruit', 'condition', 'joint_class', 'supported', 'gate_supported', 'status',
                    'rejection_reason', 'fruit_confidence', 'condition_confidence', 'joint_confidence',
                    'support_probability', 'support_threshold', 'latency_ms'):
            record[key] = res[key]
        record.update(quality_passed=res['quality']['passed'], blur_score=res['quality']['blur_score'],
                      brightness=res['quality']['brightness'], width=res['quality']['width'], height=res['quality']['height'],
                      joint_scores_json=json.dumps(res['joint_scores']))
        error = (known and (true_joint != res['joint_class'] or not res['supported'])) or (not known and res['supported'])
        record['gallery_file'] = 'error_gallery/' + gallery_name(index, row, res) if error else ''
        if error:
            gallery.append((data, record['gallery_file']))
        records.append(record)
    metrics = summarize(records)
    report = dict(
        split=args.split, device=str(predictor.device), checkpoint_epoch=predictor.metadata['epoch'],
        checkpoint_sha256=file_sha256(args.checkpoint), gate_npz_sha256=file_sha256(args.gate_npz),
        gate_meta_sha256=file_sha256(args.gate_meta), manifest_sha256=file_sha256(args.manifest),
        quality_enabled=args.quality, quality_config=asdict(config),
        quality_config_sha256=quality_config_hash(config), tta_enabled=False,
        independent_final_test_verified=False,
        dataset_note='Split name does not establish independence. No threshold is fitted by this evaluator.',
        selected_images_without_manifest_hash=sum(not r.get('sha256') for r in rows),
        selected_images_without_group=sum(not r.get('group_id') for r in rows),
        metrics=metrics,
    )
    args.output.mkdir(parents=True)
    (args.output / "quality_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (args.output / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    shutil.copyfile(args.manifest, args.output / 'manifest.csv')
    from freshlens_ai.data import rgb_from_bytes
    for data, relative in gallery:
        target = args.output / relative
        target.parent.mkdir(exist_ok=True)
        rgb_from_bytes(data).save(target)
    if metrics['cnn_only']['joint'] is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = metrics['cnn_only']['joint']['confusion_matrix']
        fig, ax = plt.subplots(figsize=(10, 9))
        ConfusionMatrixDisplay(np.asarray(cm), display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title('CNN joint classes — all known images, including rejections')
        fig.tight_layout()
        fig.savefig(args.output / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
    print(f"Saved {len(records)} predictions and metrics to {args.output}")
    return report


def main():
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
