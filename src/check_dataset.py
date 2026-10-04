"""Quality gate for raw images and manifests before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DATASET_ROOT, FRUIT_CLASSES, OTHER_CLASS, STATUS_CLASSES
from .dataset import load_manifest, prepare_dataset, scan_raw_dataset, sha256_file
from .data_integrity import metadata_errors
from .image_processing import preprocess_image


def check_dataset(dataset_root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    records = load_manifest(manifest_path) if manifest_path is not None else scan_raw_dataset(dataset_root / "raw", dataset_root)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    hash_to_paths: defaultdict[str, list[str]] = defaultdict(list)
    split_to_groups: defaultdict[str, set[str]] = defaultdict(set)
    label_counts = Counter()
    checked_records = []

    for record in records:
        record = dict(record)
        checked_records.append(record)
        path = dataset_root / str(record.get("path") or "")
        fruit = record.get("fruit", "")
        status = record.get("status", "")
        split = record.get("split", "")
        label_counts[f"{fruit}::{status or 'none'}::{split or 'unsplit'}"] += 1
        if not record.get("path") or not path.is_file():
            errors.append({"path": str(path), "reason": "missing_file"})
            continue
        if split:
            split_to_groups[split].add(str(record.get("group_id", "")))
        try:
            actual_hash = sha256_file(path)
            expected_hash = str(record.get("sha256") or "").lower()
            if expected_hash and expected_hash != actual_hash:
                errors.append({"path": str(path), "reason": "sha256_mismatch"})
            record["sha256"] = actual_hash
            with Image.open(path) as image:
                image.verify()
            processed = preprocess_image(path)
            if processed.processed_bgr.shape != (224, 224, 3):
                errors.append({"path": str(path), "reason": "canonical_shape_mismatch"})
            if processed.processed_bgr.dtype.name != "uint8":
                errors.append({"path": str(path), "reason": "canonical_dtype_mismatch"})
            if processed.quality.get("is_very_dark") or processed.quality.get("is_very_bright"):
                warnings.append({"path": str(path), "reason": "extreme_brightness"})
        except Exception as exc:  # noqa: BLE001 - report the bad file and continue
            errors.append({"path": str(path), "reason": f"unreadable:{type(exc).__name__}:{exc}"})
        hash_to_paths[str(record.get("sha256") or "")].append(record["path"])

    errors.extend(metadata_errors(checked_records, require_split=manifest_path is not None))

    duplicate_groups = [paths for digest, paths in hash_to_paths.items() if digest and len(paths) > 1]
    for paths in duplicate_groups:
        warnings.append({"path": " | ".join(paths), "reason": "duplicate_sha256"})
    split_names = ("train", "val", "test")
    group_overlaps = []
    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            overlap = sorted(split_to_groups[first] & split_to_groups[second])
            if overlap:
                group_overlaps.append({"first": first, "second": second, "groups": overlap})

    missing_label_buckets = []
    for fruit in (*FRUIT_CLASSES, OTHER_CLASS):
        expected_statuses = ("fresh", "rotten") if fruit != OTHER_CLASS else ("none",)
        for status in expected_statuses:
            for split in split_names:
                key = f"{fruit}::{status}::{split}"
                if label_counts[key] == 0:
                    missing_label_buckets.append(key)
    if missing_label_buckets:
        warnings.append({"path": "manifest", "reason": "empty_buckets:" + ",".join(missing_label_buckets)})

    return {
        "dataset_root": str(dataset_root),
        "records": len(records),
        "split_counts": dict(Counter(record.get("split") or "unsplit" for record in records)),
        "errors": errors,
        "warnings": warnings,
        "is_valid": not errors,
        "label_counts": dict(label_counts),
        "duplicate_groups": duplicate_groups,
        "group_overlaps": group_overlaps,
        "acceptance_note": "Thiếu bucket do quá ít group là cảnh báo; không được nhân bản cùng vật thể chỉ để lấp bucket.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu FreshLens")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prepare-first", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="Luu ket qua kiem tra thanh JSON")
    args = parser.parse_args()
    manifest = args.manifest or (args.root / "manifest.csv")
    if args.prepare_first:
        manifest, _ = prepare_dataset(args.root)
    if not manifest.is_file():
        parser.error(f"Chua co manifest: {manifest}. Hay export tu MongoDB hoac tao manifest local truoc.")
    report = check_dataset(args.root, manifest)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["is_valid"] else 1)


if __name__ == "__main__":
    main()
