"""Metadata checks and stable dataset splits; no database or network access."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .config import FRUIT_CLASSES, OTHER_CLASS, RANDOM_SEED, STATUS_CLASSES
from .dataset import split_by_group

SPLITS = ("train", "val", "test")


def metadata_errors(records: list[dict[str, Any]], *, require_split: bool = True) -> list[dict[str, str]]:
    """Validate labels, group boundaries and duplicate-content boundaries."""
    errors: list[dict[str, str]] = []
    if not records:
        return [{"path": "manifest", "reason": "empty_dataset"}]
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    hash_labels: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for index, record in enumerate(records, 1):
        location = str(record.get("path") or f"record_{index}")
        fruit = record.get("fruit", "")
        status = record.get("status") or ""
        group = str(record.get("group_id") or "").strip()
        split = record.get("split") or ""
        digest = str(record.get("sha256") or "").strip().lower()
        if fruit not in (*FRUIT_CLASSES, OTHER_CLASS):
            errors.append({"path": location, "reason": f"unsupported_fruit:{fruit}"})
        if fruit in FRUIT_CLASSES and status not in STATUS_CLASSES:
            errors.append({"path": location, "reason": "supported_fruit_requires_fresh_or_rotten"})
        if fruit == OTHER_CLASS and status:
            errors.append({"path": location, "reason": "other_must_not_have_fresh_or_rotten"})
        if not group:
            errors.append({"path": location, "reason": "missing_group_id"})
        elif group != str(record.get("group_id")):
            errors.append({"path": location, "reason": "group_id_has_outer_whitespace"})
        if (require_split and split not in SPLITS) or (split and split not in SPLITS):
            errors.append({"path": location, "reason": f"invalid_split:{split or 'empty'}"})
        if group and split in SPLITS:
            groups[group].add(split)
        if digest:
            hash_labels[digest].add((str(fruit), str(status)))
            if split in SPLITS:
                hashes[digest].add(split)
    for group, values in groups.items():
        if len(values) > 1:
            errors.append({"path": group, "reason": "group_leakage:" + ",".join(sorted(values))})
    for digest, values in hashes.items():
        if len(values) > 1:
            errors.append({"path": digest[:12], "reason": "duplicate_content_across_splits"})
    for digest, labels in hash_labels.items():
        if len(labels) > 1:
            errors.append({"path": digest[:12], "reason": "same_content_conflicting_labels"})
    return errors


def assign_missing_splits(
    records: list[dict[str, Any]], seed: int = RANDOM_SEED, *, force_resplit: bool = False
) -> list[dict[str, Any]]:
    """Freeze existing splits. New independent groups go to train.

    On the first snapshot, use the existing 60/20/20 group splitter. Records
    with the same SHA are kept together even if their declared groups differ.
    A view added to an existing group inherits that group's split.
    """
    output = [dict(row) for row in records]
    problems = metadata_errors(output, require_split=False)
    if problems:
        first = problems[0]
        raise ValueError(f"Metadata khong hop le ({len(problems)} loi): {first['reason']} tai {first['path']}")

    # Union physical groups that share exactly the same encoded image.
    parent = {str(row["group_id"]): str(row["group_id"]) for row in output}

    def root(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    seen_hash: dict[str, str] = {}
    for row in output:
        group = str(row["group_id"])
        digest = str(row.get("sha256") or "").lower()
        if digest in seen_hash:
            a, b = root(group), root(seen_hash[digest])
            parent[max(a, b)] = min(a, b)
        elif digest:
            seen_hash[digest] = group

    fixed: dict[str, str] = {}
    for row in output:
        split = row.get("split") or ""
        component = root(str(row["group_id"]))
        if split in SPLITS:
            if component in fixed and fixed[component] != split:
                raise ValueError("Anh trung noi dung lien ket cac group o nhieu split. Can ra soat du lieu.")
            fixed[component] = split

    if not fixed or force_resplit:
        canonical = [dict(row, group_id=root(str(row["group_id"]))) for row in output]
        canonical.sort(key=lambda row: (row["group_id"], str(row.get("path", "")), str(row.get("sha256", ""))))
        assignment = {row["group_id"]: row["split"] for row in split_by_group(canonical, seed=seed)}
    else:
        assignment = dict(fixed)
    for row in output:
        row["split"] = assignment.get(root(str(row["group_id"])), "train")
    return output


def metadata_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact, shareable summary: omit URLs, credentials and local paths."""
    problems = metadata_errors(records, require_split=False)
    return {
        "total_images": len(records),
        "split_counts": dict(Counter(str(row.get("split") or "unsplit") for row in records)),
        "label_counts": dict(sorted(Counter(f"{row.get('fruit') or 'missing'}::{row.get('status') or 'none'}" for row in records).items())),
        "unique_groups": len({str(row.get("group_id")) for row in records if row.get("group_id")}),
        "with_cloud_url": sum(bool(row.get("url")) for row in records),
        "with_local_path_reference": sum(bool(row.get("path")) for row in records),
        "missing_image_reference": sum(not row.get("url") and not row.get("path") for row in records),
        "metadata_error_count": len(problems),
        "metadata_error_counts": dict(Counter(error["reason"].split(":")[0] for error in problems)),
        "note": "Thong ke metadata; chua kiem tra file anh, anh gan trung, accuracy hay nhom chup thuc te.",
    }
