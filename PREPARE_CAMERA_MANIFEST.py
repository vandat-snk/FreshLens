"""Validate human camera labels and export reproducible image quality metadata.

Labels CSV: path,fruit,status,split,group_id,source,quality_label.
Images stay in their original directory; this tool never changes labels or fits thresholds.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from EVALUATE_TV3 import load_rows
from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.data import rgb_from_bytes
from freshlens_ai.data.image_io import safe_path
from freshlens_ai.inference.quality import measure_quality
from freshlens_ai.utils.file_io import file_sha256


def prepare(root, labels, output, reference=None):
    root, labels, output = Path(root), Path(labels), Path(output)
    if output.exists():
        raise ValueError('Output already exists; choose a new directory.')
    with labels.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'path', 'fruit', 'status', 'split', 'group_id', 'source', 'quality_label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Camera labels require: ' + ', '.join(sorted(required)))
        rows = list(reader)
    if not rows:
        raise ValueError('No camera labels found.')
    references = []
    if reference:
        with Path(reference).open(encoding='utf-8-sig', newline='') as handle:
            references = list(csv.DictReader(handle))
    reference_hashes = {r['sha256'] for r in references if r.get('sha256')}
    reference_groups = {r['group_id'] for r in references if r.get('group_id')}
    prepared, quality, seen_hashes = [], [], set()
    for row in rows:
        if not row['group_id'].strip() or not row['source'].strip():
            raise ValueError('Every camera image needs a physical group_id and source.')
        if row['split'] not in ('dev', 'calibration', 'final'):
            raise ValueError('Use dev, calibration or final for camera splits.')
        path = safe_path(root, row['path'])
        digest = file_sha256(path)
        if digest in reference_hashes or row['group_id'] in reference_groups:
            raise ValueError(f"Camera image/group overlaps reference manifest: {row['path']}")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate camera image bytes: {row['path']}")
        seen_hashes.add(digest)
        prepared.append({**{k: row[k] for k in sorted(required)}, 'sha256': digest})
        measured = measure_quality(rgb_from_bytes(path.read_bytes()))
        quality.append(dict(path=row['path'], **measured))
    # Reuse the evaluator's label and cross-split group validation before writing.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / 'manifest.csv'
        _write_csv(candidate, prepared)
        for split in {r['split'] for r in prepared}:
            load_rows(candidate, split)
    report = dict(
        images=len(prepared), groups=len({r['group_id'] for r in prepared}),
        counts=dict(Counter(f"{r['split']}/{r['fruit']}/{r['status']}" for r in prepared)),
        labels_sha256=file_sha256(labels),
        reference_manifest_sha256=file_sha256(reference) if reference else None,
        reference_overlap_check_performed=bool(reference),
        independent_final_test_verified=False,
        note='Hash/group checks cannot establish unseen physical specimens or absence of tuning. Human review is required.',
    )
    output.mkdir(parents=True)
    _write_csv(output / 'manifest.csv', prepared)
    report['manifest_sha256'] = file_sha256(output / 'manifest.csv')
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def _write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=PROJECT_DIR / 'data/cnn_dataset_v3/manifest.csv')
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.root, args.labels, args.output, args.reference), ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f'[ERROR] {exc}\n')


if __name__ == '__main__':
    main()
