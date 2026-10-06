"""Build a new TV1 dataset snapshot without altering the V3 baseline."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
import json
from collections import defaultdict
from pathlib import Path

from freshlens_ai.constants import CLASSES, PROJECT_DIR, SPLITS
from freshlens_ai.data.dataset_audit import (
    OTHER_COLUMNS, QUALITY_COLUMNS, SUPPORTED_COLUMNS, audit_dataset,
    identity_audit, mapped_image_path, path_key, read_csv, validate_rows,
)
from freshlens_ai.data.image_io import rgb_from_bytes, safe_path
from freshlens_ai.data.locked_dataset import load_locked_dataset
from freshlens_ai.utils.file_io import atomic_json, file_sha256, sha256_bytes, write_csv

CAPTURE_COLUMNS = (
    "capture_device", "specimen_id", "specimen_evidence", "session_id",
    "lighting", "background", "viewpoint", "distance", "hard_case_tags",
    "provenance", "license",
)
INVENTORY_COLUMNS = ("path", "data_kind", "fruit", "status", "category", "subcategory",
                     "source", "group_id", "split", *CAPTURE_COLUMNS)
QUALITY_MANIFEST_COLUMNS = ("width", "height", "brightness", "blur_score", "decode_status", "quality_status")
BASELINE_FILES = ("manifest.csv", "final_split_report.json", "dataset_lock.json")


@lru_cache(maxsize=1)
def legacy_split_core():
    """Reuse the audited grouping/split primitives; legacy source stays unchanged."""
    path = PROJECT_DIR / "legacy/development/step2b_grouping/step2_core.py"
    spec = importlib.util.spec_from_file_location("freshlens_legacy_split_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_snapshot(directory):
    directory = Path(directory)
    before = {name: file_sha256(directory / name) for name in BASELINE_FILES}
    _, identity = load_locked_dataset(directory)
    rows = read_csv(directory / "manifest.csv", SUPPORTED_COLUMNS)
    after = {name: file_sha256(directory / name) for name in BASELINE_FILES}
    if before != after:
        raise ValueError("Baseline changed during validation")
    return rows, before, identity


def load_inventory(path, root, strip_prefix=""):
    if path is None:
        return [], []
    rows = read_csv(path, INVENTORY_COLUMNS)
    supported, other = [], []
    specimen_groups, group_specimens = {}, {}
    for row in rows:
        kind = row.pop("data_kind")
        if kind not in ("supported", "other"):
            raise ValueError("data_kind must be supported or other: " + row["path"])
        if kind == "other" and (row["fruit"] or row["status"]):
            raise ValueError("Other data must not have production fruit/status labels")
        if kind == "supported" and (row["category"] or row["subcategory"]):
            raise ValueError("Supported data must not have an other category")
        if row["capture_device"] not in ("", "phone", "laptop_camera", "other_camera"):
            raise ValueError("Unknown capture_device: " + row["path"])
        if row["capture_device"] in ("phone", "laptop_camera", "other_camera"):
            required = ("specimen_id", "specimen_evidence", "session_id", "lighting", "background", "viewpoint", "distance")
            if any(not row[key] for key in required):
                raise ValueError("Camera row needs specimen evidence and capture metadata: " + row["path"])
        if row["specimen_id"]:
            if not row["specimen_evidence"]:
                raise ValueError("specimen_id requires evidence")
            specimen, group = row["specimen_id"], row["group_id"]
            if specimen in specimen_groups and specimen_groups[specimen] != group:
                raise ValueError("Same physical specimen has different group_id")
            if group in group_specimens and group_specimens[group] != specimen:
                raise ValueError("Same group_id declares different physical specimens")
            specimen_groups[specimen] = group
            group_specimens[group] = specimen
        if root is None:
            raise ValueError("New inventory rows require --root with real image files")
        data = safe_path(root, mapped_image_path(row["path"], strip_prefix)).read_bytes()
        rgb_from_bytes(data)  # Never add a broken/multiframe image to a new manifest.
        row["sha256"] = sha256_bytes(data)
        problems = validate_rows([row], kind, allow_unsplit=True)
        if problems:
            raise ValueError("; ".join(problems))
        (supported if kind == "supported" else other).append(row)
    return supported, other


def assign_new_splits(baseline, supported, other, seed=42):
    """Freeze declared splits and coassign transitive group/SHA links globally."""
    rows = [dict(row) for row in baseline + supported + other]
    identity = identity_audit(rows)
    if identity["duplicate_path"]["keys"]:
        raise ValueError("Duplicate path; review inventory instead of overwriting baseline rows")
    content_labels = defaultdict(set)
    for i, row in enumerate(rows):
        label = ("supported", row["fruit"], row["status"]) if i < len(baseline) + len(supported) else ("other", row["category"], row["subcategory"])
        content_labels[row["sha256"]].add(label)
    if any(len(labels) > 1 for labels in content_labels.values()):
        raise ValueError("Same SHA256 has conflicting labels/categories; human review required")
    core = legacy_split_core()
    union = core.UnionFind(len(rows))
    owners = {}
    for i, row in enumerate(rows):
        for key in ("group_id", "sha256"):
            value = (key, row[key])
            if value in owners:
                union.union(i, owners[value])
            else:
                owners[value] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[union.find(i)].append(i)
    assignments, free, group_ids = {}, [], {}
    adapted = []
    for i, row in enumerate(rows):
        adapted.append(dict(row) if i < len(baseline) + len(supported)
                       else dict(row, fruit="other", status=row["category"] + ":" + row["subcategory"]))
    for ids in components.values():
        fixed = {rows[i]["split"] for i in ids if rows[i].get("split")}
        if len(fixed) > 1:
            raise ValueError("Group/SHA component crosses frozen splits; no records moved")
        component_id = min(rows[i]["group_id"] for i in ids)
        for i in ids:
            group_ids[i] = component_id
        if fixed:
            assignments[component_id] = next(iter(fixed))
        elif len({core.label(adapted[i]) for i in ids}) > 1:
            assignments[component_id] = "train"
        else:
            free.extend(ids)
    assignments.update(core.assign_splits(adapted, free, group_ids, seed))
    for i, row in enumerate(rows):
        row["split"] = assignments[group_ids[i]]
    # Keep every original baseline field, including its group ID, untouched.
    if rows[:len(baseline)] != baseline:
        raise ValueError("Baseline rows unexpectedly changed")
    n_supported = len(baseline) + len(supported)
    return rows[:n_supported], rows[n_supported:]


def build_dataset(baseline_dir, output_dir, root=None, inventory=None, seed=42, thresholds=None, strip_prefix="", require_images=False, path_map_csv=None):
    baseline_dir, output_dir = Path(baseline_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_relative_to(baseline_dir):
        raise ValueError("Use a NEW output directory outside the baseline; nothing overwritten")
    baseline, fingerprints, _ = baseline_snapshot(baseline_dir)
    from freshlens_ai.data.dataset_audit import load_path_mapping
    overrides = load_path_mapping(path_map_csv, baseline, strip_prefix)
    inventory_hash = file_sha256(inventory) if inventory is not None else None
    added_supported, added_other = load_inventory(inventory, root, strip_prefix)
    supported, other = assign_new_splits(baseline, added_supported, added_other, seed)
    report, quality = audit_dataset(supported, other, root, thresholds, strip_prefix, fingerprints=require_images, progress=root is not None, path_overrides=overrides)
    if require_images and not report["all_image_files_verified"]:
        raise ValueError("Verified image build requires every file, matching SHA256 and successful decode")
    if not report["metadata_valid"]:
        raise ValueError("Dataset metadata/leakage audit failed")
    if report["quality"]["sha256_mismatches"]:
        raise ValueError("Image bytes disagree with baseline manifest; baseline not changed")
    if report["quality"]["decode_errors"]:
        raise ValueError("Decode errors; review raw files before building")
    if {name: file_sha256(baseline_dir / name) for name in BASELINE_FILES} != fingerprints:
        raise ValueError("Baseline changed during build")
    if inventory is not None and file_sha256(inventory) != inventory_hash:
        raise ValueError("Inventory changed during build")
    quality_by_path = {q["path"]: q for q in quality}
    for row in supported + other:
        row.update({key: quality_by_path[row["path"]][key] for key in QUALITY_MANIFEST_COLUMNS})
    report.update(
        dataset_name="Dataset V2 (TV1), storage " + output_dir.name,
        baseline_fingerprints=fingerprints, inventory_sha256=inventory_hash,
        added_supported_records=len(added_supported), added_other_records=len(added_other),
        seed=seed, split_policy="Freeze baseline/declared splits; group+SHA union; new pure components 60/20/20 by label; mixed components train",
        baseline_records_preserved=True,
        ready_for_pilot_training=report["all_image_files_verified"] and not added_supported and not added_other,
        candidate_data_loader_available=True,
        production_loader_compatible=False,
        build_status=("VERIFIED_BASELINE_ONLY" if not added_supported and not added_other else "VERIFIED_IMAGES_REVIEW_REQUIRED") if report["all_image_files_verified"] else "METADATA_ONLY_OR_PARTIAL",
        limitations=["No current raw baseline or new image claim without verification.",
                     "Different group IDs do not prove different physical fruit specimens.",
                     "Exact byte SHA does not detect crops, compression or different views.",
                     "New near-duplicate review and an independently collected frozen final test are still required.",
                     "Candidate quality thresholds are diagnostic, not production inference rules."],
    )
    # Atomic mkdir fails if another process created the destination. No overwrite mode.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "manifest.csv", supported, (*SUPPORTED_COLUMNS, *CAPTURE_COLUMNS, *QUALITY_MANIFEST_COLUMNS))
    write_csv(output_dir / "other_manifest.csv", other, (*OTHER_COLUMNS, *CAPTURE_COLUMNS, *QUALITY_MANIFEST_COLUMNS))
    write_csv(output_dir / "image_quality.csv", quality, QUALITY_COLUMNS)
    report["output_manifest_sha256"] = file_sha256(output_dir / "manifest.csv")
    report["other_manifest_sha256"] = file_sha256(output_dir / "other_manifest.csv")
    atomic_json(output_dir / "dataset_report.json", report)
    files = ("manifest.csv", "other_manifest.csv", "image_quality.csv", "dataset_report.json")
    atomic_json(output_dir / "dataset_lock.json", {
        "schema_version": "freshlens_dataset_v2_lock_1", "seed": seed,
        "baseline_fingerprints": fingerprints,
        "files": {name: file_sha256(output_dir / name) for name in files},
        "supported_records": len(supported), "other_records": len(other),
        "final_test_frozen": False, "ready_for_pilot_training": report["ready_for_pilot_training"],
        "image_path_mapping": report["image_path_mapping"],
        "production_loader_compatible": False,
    })
    return report


def verify_candidate_lock(directory):
    directory = Path(directory)
    lock = json.loads((directory / "dataset_lock.json").read_text(encoding="utf-8-sig"))
    if lock.get("schema_version") != "freshlens_dataset_v2_lock_1":
        raise ValueError("Unsupported candidate lock schema")
    expected = {"manifest.csv", "other_manifest.csv", "image_quality.csv", "dataset_report.json"}
    if set(lock.get("files", {})) != expected:
        raise ValueError("Candidate lock file inventory mismatch")
    for name, digest in lock["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Candidate file changed after build: " + name)
    return lock


def load_candidate_dataset(directory, require_ready=True):
    """TV2 adapter: validate candidate lock and map paths in memory for FruitDataset.

    Caller supplies the original image root to FruitDataset. Manifest bytes,
    labels, groups, and splits on disk are never changed by this loader.
    """
    directory = Path(directory)
    lock = verify_candidate_lock(directory)
    report = json.loads((directory / "dataset_report.json").read_text(encoding="utf-8-sig"))
    if require_ready and not (lock.get("ready_for_pilot_training") and report.get("all_image_files_verified")):
        raise ValueError("Candidate is not ready for pilot training")
    mapping = lock.get("image_path_mapping", {"strip_prefix": ""})
    if mapping != report.get("image_path_mapping", {"strip_prefix": ""}):
        raise ValueError("Candidate mapping disagrees with locked report")
    supported = read_csv(directory / "manifest.csv", SUPPORTED_COLUMNS)
    other = read_csv(directory / "other_manifest.csv", OTHER_COLUMNS)
    problems = validate_rows(supported) + validate_rows(other, "other")
    identity = identity_audit(supported + other)
    if (problems or identity["duplicate_path"]["keys"] or identity["cross_split_sha256"]["keys"]
            or identity["cross_split_group_id"]["keys"]):
        raise ValueError("Invalid candidate manifest or cross-split leakage")
    if len(supported) != lock["supported_records"] or len(other) != lock["other_records"]:
        raise ValueError("Candidate record counts disagree with lock")
    for row in supported:
        row["manifest_path"] = row["path"]
        row["path"] = mapping.get("overrides", {}).get(row["path"], mapped_image_path(row["path"], mapping.get("strip_prefix", "")))
        row["target"] = CLASSES.index(row["fruit"] + "::" + row["status"])
    return supported, {
        "manifest_sha256": lock["files"]["manifest.csv"],
        "other_manifest_sha256": lock["files"]["other_manifest.csv"],
        "split_counts": report["supported_split_counts"],
        "image_path_mapping": mapping,
        "physical_specimen_independence_confirmed": False,
    }
