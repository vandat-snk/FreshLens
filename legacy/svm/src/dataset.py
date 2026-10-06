"""Dataset inventory, group-aware splitting and manifest generation.

Expected raw layout (the root may be on E:):

    E:\\FreshLens\\dataset\\raw\\apple\\fresh\\*.jpg
    E:\\FreshLens\\dataset\\raw\\apple\\rotten\\*.jpg
    E:\\FreshLens\\dataset\\raw\\banana\\fresh\\*.jpg
    E:\\FreshLens\\dataset\\raw\\orange\\rotten\\*.jpg
    E:\\FreshLens\\dataset\\raw\\tomato\\fresh\\*.jpg
    E:\\FreshLens\\dataset\\raw\\other\\*.jpg

Use a ``groups.csv`` beside ``raw`` when several views belong to one object
or one photo session. Its columns are ``relative_path,group_id,source``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DATASET_ROOT,
    FRUIT_CLASSES,
    MANIFEST_PATH,
    OTHER_CLASS,
    RANDOM_SEED,
    STATUS_CLASSES,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_group_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = dataset_root / "groups.csv"
    if not metadata_path.exists():
        return {}
    output: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = (row.get("relative_path") or row.get("path") or "").replace("\\", "/")
            if relative:
                output[relative] = {
                    "group_id": row.get("group_id") or Path(relative).stem.split("__")[0],
                    "source": row.get("source") or "unspecified",
                }
    return output


def _group_id_for(path: Path, raw_root: Path) -> str:
    """Derive a conservative fallback group id from a filename.

    Naming ``apple_001__view02.jpg`` keeps all views together. If a filename
    has no ``__`` marker, the stem itself is treated as one group, which is
    safer than accidentally leaking sibling files into multiple splits.
    """

    stem = path.stem
    return stem.split("__", 1)[0] or stem


def _infer_labels(path: Path, raw_root: Path) -> tuple[str, str]:
    relative_parts = [part.lower() for part in path.relative_to(raw_root).parts[:-1]]
    fruit = next((part for part in relative_parts if part in FRUIT_CLASSES or part == OTHER_CLASS), None)
    status = next((part for part in relative_parts if part in STATUS_CLASSES), "")
    if fruit is None:
        # Unknown directories are deliberately treated as outside the supported
        # scope. This prevents silently pretending an unsupported fruit is one
        # of the four learned classes.
        fruit = OTHER_CLASS
        status = ""
    if fruit == OTHER_CLASS:
        status = ""
    return fruit, status


def scan_raw_dataset(raw_root: Path, dataset_root: Path | None = None) -> list[dict[str, Any]]:
    raw_root = Path(raw_root)
    dataset_root = Path(dataset_root or raw_root.parent)
    group_metadata = _read_group_metadata(dataset_root)
    records: list[dict[str, Any]] = []
    if not raw_root.exists():
        return records
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_to_dataset = path.relative_to(dataset_root).as_posix()
        fruit, status = _infer_labels(path, raw_root)
        metadata = group_metadata.get(relative_to_dataset, {})
        records.append({
            "path": relative_to_dataset,
            "fruit": fruit,
            "status": status,
            "source": metadata.get("source", path.parts[-3] if len(path.parts) >= 3 else "unspecified"),
            "group_id": metadata.get("group_id", _group_id_for(path, raw_root)),
            "sha256": sha256_file(path),
            "split": "",
        })
    return records


def _bucket_key(record: dict[str, Any]) -> str:
    return f"{record['fruit']}::{record['status'] or 'none'}"


def split_by_group(
    records: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Split records without sharing a group id between train/val/test.

    Groups are allocated inside each label bucket to preserve class coverage
    when enough independent groups exist. With too few groups, the manifest
    records the honest result (a class may be absent from val/test) instead of
    duplicating an object across splits.
    """

    if not records:
        return []
    if abs(sum(ratios) - 1.0) > 1e-6 or any(value <= 0 for value in ratios):
        raise ValueError("Tỷ lệ train/validation/test phải dương và tổng bằng 1")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["group_id"])].append(record)
    buckets: dict[str, list[str]] = defaultdict(list)
    for group_id, group_records in groups.items():
        labels = {_bucket_key(record) for record in group_records}
        # A mixed-label group stays indivisible and gets its own bucket.
        bucket = "|".join(sorted(labels))
        buckets[bucket].append(group_id)
    assignment: dict[str, str] = {}
    split_names = ("train", "val", "test")
    for group_ids in buckets.values():
        rng.shuffle(group_ids)
        total = len(group_ids)
        train_end = int(round(total * ratios[0]))
        val_end = train_end + int(round(total * ratios[1]))
        if total >= 3:
            train_end = min(max(train_end, 1), total - 2)
            val_end = min(max(val_end, train_end + 1), total - 1)
        elif total == 2:
            train_end, val_end = 1, 1
        else:
            train_end, val_end = 1, 1
        for index, group_id in enumerate(group_ids):
            assignment[group_id] = split_names[0] if index < train_end else split_names[1] if index < val_end else split_names[2]
    output = []
    for record in records:
        cloned = dict(record)
        cloned["split"] = assignment[str(record["group_id"])]
        output.append(cloned)
    return output


def save_manifest(records: Iterable[dict[str, Any]], path: Path = MANIFEST_PATH) -> Path:
    rows = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["path", "fruit", "status", "source", "group_id", "sha256", "split"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def resolve_record_path(record: dict[str, Any], dataset_root: Path = DATASET_ROOT) -> Path:
    return Path(dataset_root) / str(record["path"])


def manifest_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    split_counts = Counter(row.get("split", "") for row in rows)
    label_counts = Counter(f"{row.get('fruit')}::{row.get('status') or 'none'}" for row in rows)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_split[row.get("split", "")].add(str(row.get("group_id", "")))
    overlaps = {
        first: sorted(groups_by_split[first] & groups_by_split[second])
        for index, first in enumerate(("train", "val", "test"))
        for second in ("train", "val", "test")[index + 1:]
        if groups_by_split[first] & groups_by_split[second]
    }
    return {
        "total_images": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "unique_groups": len({row.get("group_id") for row in rows}),
        "group_overlap": overlaps,
        "supported_fruit_classes": list(FRUIT_CLASSES),
        "other_images": sum(1 for row in rows if row.get("fruit") == OTHER_CLASS),
    }


def prepare_dataset(
    dataset_root: Path = DATASET_ROOT,
    output_path: Path | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[Path, dict[str, Any]]:
    dataset_root = Path(dataset_root)
    records = scan_raw_dataset(dataset_root / "raw", dataset_root)
    split_records = split_by_group(records, seed=seed)
    manifest_path = save_manifest(split_records, output_path or (dataset_root / "manifest.csv"))
    summary = manifest_summary(split_records)
    summary_path = manifest_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Quét raw dataset và tạo manifest chia theo group_id")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT, help="Thư mục dataset, ví dụ E:/FreshLens/dataset")
    parser.add_argument("--output", type=Path, default=None, help="Đường dẫn manifest CSV")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    manifest_path, summary = prepare_dataset(args.root, args.output, args.seed)
    print(json.dumps({"manifest": str(manifest_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

