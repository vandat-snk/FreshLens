"""Read-only metadata, byte identity, and image quality audit for TV1."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image

from freshlens_ai.constants import CLASSES, SPLITS
from freshlens_ai.data.image_io import rgb_from_bytes, safe_path
from freshlens_ai.utils.file_io import file_sha256, sha256_bytes

SUPPORTED_COLUMNS = ("path", "fruit", "status", "source", "group_id", "sha256", "split")
OTHER_COLUMNS = ("path", "category", "subcategory", "source", "group_id", "sha256", "split")
OTHER_CATEGORIES = ("other_fruit", "non_fruit", "multiple")
QUALITY_COLUMNS = (
    "path", "sha256", "actual_sha256", "sha256_status", "decode_status",
    "quality_status", "width", "height", "aspect_ratio", "brightness",
    "blur_score", "low_resolution", "low_brightness", "blur", "format",
    "original_mode", "channels", "exif_present", "exif_orientation", "file_size",
)
DEFAULT_THRESHOLDS = {"min_side": 96, "min_brightness": 35.0, "min_blur_score": 50.0}


def path_key(value):
    return unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()


def relative_path(value):
    value = value.replace("\\", "/")
    pure = PurePosixPath(value)
    if (not value or pure.is_absolute() or PureWindowsPath(value).drive
            or ".." in pure.parts or ":" in value or "\x00" in value):
        raise ValueError(f"Unsafe relative image path: {value!r}")
    return pure.as_posix()


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Missing CSV columns: " + ", ".join(required))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Repeated CSV column name")
        rows = []
        for raw in reader:
            if None in raw:
                raise ValueError("CSV row has more values than header")
            row = {key: (value or "").strip() for key, value in raw.items()}
            row["path"] = relative_path(row["path"])
            rows.append(row)
        return rows


def identity_audit(rows):
    """Report declared identities; actual byte hashes are audited separately."""
    result = {}
    for key in ("path", "sha256", "group_id"):
        members = defaultdict(list)
        for row in rows:
            value = path_key(row[key]) if key == "path" else row.get(key, "")
            if value:
                members[value].append(row)
        duplicates = {value: [r["path"] for r in group]
                      for value, group in members.items() if len(group) > 1}
        if key != "group_id":
            result["duplicate_" + key] = {
                "keys": len(duplicates),
                "extra_records": sum(len(group) - 1 for group in duplicates.values()),
                "members": duplicates,
            }
        across = {value: {"splits": sorted({r.get("split", "") for r in group}),
                          "paths": [r["path"] for r in group]}
                  for value, group in members.items()
                  if len({r.get("split", "") for r in group}) > 1}
        result["cross_split_" + key] = {"keys": len(across), "members": across}
    return result


def validate_rows(rows, kind="supported", allow_unsplit=False):
    required = SUPPORTED_COLUMNS if kind == "supported" else OTHER_COLUMNS
    errors = []
    for index, row in enumerate(rows, 2):
        for key in required:
            if not row.get(key) and not (key == "split" and allow_unsplit):
                errors.append(f"line {index}: missing {key}")
        try:
            relative_path(row.get("path", ""))
        except ValueError as exc:
            errors.append(str(exc))
        if kind == "supported":
            if row.get("fruit", "") + "::" + row.get("status", "") not in CLASSES:
                errors.append(f"line {index}: unsupported fruit/status")
        elif row.get("category") not in OTHER_CATEGORIES:
            errors.append(f"line {index}: invalid other category")
        if row.get("split") not in SPLITS and not (allow_unsplit and not row.get("split")):
            errors.append(f"line {index}: invalid split")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            errors.append(f"line {index}: invalid SHA256")
    return errors


def inspect_image(root, row, thresholds=None):
    """Measure oriented RGB at native resolution, before resize/augmentation."""
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    result = dict.fromkeys(QUALITY_COLUMNS, "")
    result.update(path=row["path"], sha256=row["sha256"],
                  sha256_status="NOT_VERIFIED", decode_status="NOT_AVAILABLE",
                  quality_status="NOT_AVAILABLE")
    if root is None:
        return result
    try:
        path = safe_path(root, row["path"])
    except Exception:
        # Distinguish missing/unsafe paths from an image decode failure.
        result["decode_status"] = "MISSING_FILE"
        return result
    data = path.read_bytes()
    result["file_size"] = len(data)
    result["actual_sha256"] = sha256_bytes(data)
    result["sha256_status"] = "MATCH" if result["actual_sha256"] == row["sha256"] else "MISMATCH"
    try:
        with Image.open(io.BytesIO(data)) as opened:
            exif = opened.getexif()
            result.update(format=opened.format or "UNKNOWN", original_mode=opened.mode,
                          channels=len(opened.getbands()), exif_present=bool(exif),
                          exif_orientation=exif.get(274, ""))
        image = rgb_from_bytes(data)
        gray = np.asarray(image.convert("L"), dtype=np.float64)
        padded = np.pad(gray, 1, mode="reflect")
        laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                     + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * gray)
        brightness = float(gray.mean())
        blur_score = float(laplacian.var())
        flags = {
            "low_resolution": min(image.size) < thresholds["min_side"],
            "low_brightness": brightness < thresholds["min_brightness"],
            "blur": blur_score < thresholds["min_blur_score"],
        }
        result.update(width=image.width, height=image.height,
                      aspect_ratio=round(image.width / image.height, 6),
                      brightness=round(brightness, 6), blur_score=round(blur_score, 6),
                      decode_status="OK", quality_status="LOW_QUALITY" if any(flags.values()) else "OK",
                      **flags)
    except Exception as exc:
        result.update(decode_status="DECODE_ERROR", quality_status="DECODE_ERROR",
                      error_type=type(exc).__name__)
    return result


def audit_dataset(supported, other=(), root=None, thresholds=None):
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if any(not np.isfinite(float(v)) or float(v) < 0 for v in thresholds.values()):
        raise ValueError("Quality thresholds must be finite and nonnegative")
    if set(thresholds) != set(DEFAULT_THRESHOLDS):
        raise ValueError("Invalid quality threshold keys")
    supported, other = list(supported), list(other)
    rows = supported + other
    quality = [inspect_image(root, row, thresholds) for row in rows]
    decoded = [q for q in quality if q["decode_status"] == "OK"]
    actual_rows = [dict(row, sha256=q["actual_sha256"]) for row, q in zip(rows, quality)
                   if q["actual_sha256"]]
    metadata_errors = validate_rows(supported) + validate_rows(other, "other")
    identity = identity_audit(rows)
    group_splits = defaultdict(set)
    for row in rows:
        if row.get("group_id"):
            group_splits[row["group_id"]].add(row.get("split", ""))
    camera = [r for r in rows if r.get("capture_device") in ("phone", "laptop_camera", "other_camera")]
    specimen_groups, specimen_splits, group_specimens = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        if row.get("specimen_id"):
            specimen_groups[row["specimen_id"]].add(row.get("group_id", ""))
            specimen_splits[row["specimen_id"]].add(row.get("split", ""))
            group_specimens[row.get("group_id", "")].add(row["specimen_id"])
    specimen_violations = {
        "specimens_in_multiple_groups": {k: sorted(v) for k, v in specimen_groups.items() if len(v) > 1},
        "specimens_across_splits": {k: sorted(v) for k, v in specimen_splits.items() if len(v) > 1},
        "groups_with_conflicting_specimens": {k: sorted(v) for k, v in group_specimens.items() if len(v) > 1},
    }
    for row in camera:
        if any(not row.get(key) for key in ("specimen_id", "specimen_evidence", "session_id", "lighting", "background", "viewpoint", "distance")):
            metadata_errors.append("Camera metadata/evidence incomplete: " + row["path"])
    for name, members in specimen_violations.items():
        if members:
            metadata_errors.append(name + ": " + ", ".join(sorted(members)))
    errors = len(metadata_errors) + identity["duplicate_path"]["keys"]
    errors += identity["cross_split_sha256"]["keys"] + identity["cross_split_group_id"]["keys"]
    sha_mismatches = sum(q["sha256_status"] == "MISMATCH" for q in quality)
    report = {
        "schema_version": "freshlens_dataset_v2_audit_1",
        "audit_script_sha256": file_sha256(__file__),
        "records": len(rows), "supported_records": len(supported), "other_records": len(other),
        "split_counts": {s: sum(r.get("split") == s for r in rows) for s in SPLITS},
        "supported_split_counts": {s: sum(r.get("split") == s for r in supported) for s in SPLITS},
        "fruit_counts": dict(sorted(Counter(r["fruit"] for r in supported).items())),
        "status_counts": dict(sorted(Counter(r["status"] for r in supported).items())),
        "label_counts": {name: sum(r["fruit"] + "::" + r["status"] == name for r in supported) for name in CLASSES},
        "label_split_counts": {name: {s: sum(r["fruit"] + "::" + r["status"] == name and r["split"] == s
                                                  for r in supported) for s in SPLITS} for name in CLASSES},
        "other_category_counts": {c: sum(r["category"] == c for r in other) for c in OTHER_CATEGORIES},
        "other_subcategory_counts": dict(sorted(Counter(r["subcategory"] for r in other).items())),
        "source_counts": dict(sorted(Counter(r.get("source", "") for r in rows).items())),
        "unique_groups": len(group_splits),
        "groups_by_split": {s: sum(s in splits for splits in group_splits.values()) for s in SPLITS},
        "missing_metadata": {k: sum(not r.get(k) for r in rows) for k in ("path", "source", "group_id", "sha256", "split")},
        "metadata_errors": metadata_errors,
        "declared_identity": identity,
        "actual_byte_identity": identity_audit(actual_rows) if actual_rows else None,
        "raw_data_status": "RAW DATA NOT AVAILABLE" if not actual_rows else ("AVAILABLE" if len(actual_rows) == len(rows) else "PARTIAL"),
        "quality": {
            "records_inspected": len(actual_rows), "decoded_images": len(decoded),
            "missing_files": sum(q["decode_status"] == "MISSING_FILE" for q in quality) if root is not None else None,
            "not_checked": sum(q["decode_status"] == "NOT_AVAILABLE" for q in quality),
            "sha256_mismatches": sha_mismatches if actual_rows else None,
            "decode_errors": sum(q["decode_status"] == "DECODE_ERROR" for q in quality) if actual_rows else None,
            **{key: sum(q[key] is True for q in decoded) if decoded else None for key in ("low_resolution", "low_brightness", "blur")},
            "quality_status_counts": dict(Counter(q["quality_status"] for q in quality)),
            "width_range": [min(q["width"] for q in decoded), max(q["width"] for q in decoded)] if decoded else None,
            "height_range": [min(q["height"] for q in decoded), max(q["height"] for q in decoded)] if decoded else None,
            "formats": dict(Counter(q["format"] for q in decoded)),
            "original_modes": dict(Counter(q["original_mode"] for q in decoded)),
            "channels": dict(Counter(str(q["channels"]) for q in decoded)),
            "exif_present": sum(bool(q["exif_present"]) for q in decoded) if decoded else None,
            "exif_orientations": dict(Counter(str(q["exif_orientation"] or "missing") for q in decoded)),
            "candidate_thresholds": thresholds,
            "measurement": "native oriented RGB -> Pillow L; variance of 4-neighbour Laplacian, reflect boundary; no resizing",
            "production_thresholds_changed": False,
        },
        "camera": {
            "records": len(camera), "label_counts": dict(Counter(r.get("fruit", "other") + "::" + r.get("status", "") for r in camera)),
            "device_counts": dict(Counter(r["capture_device"] for r in camera)),
            **{k + "_counts": dict(Counter(r.get(k, "") for r in camera)) for k in ("lighting", "background", "viewpoint", "distance", "hard_case_tags")},
            "grouping_evidence_missing": sum(not r.get("specimen_id") or not r.get("specimen_evidence") for r in camera),
            "grouping_violations": specimen_violations,
            "physical_specimen_independence_confirmed": False,
        },
        "hard_case_records": sum(bool(r.get("hard_case_tags")) for r in rows),
        "other_data_status": "OTHER DATA INCOMPLETE",
        "final_test_status": "FINAL TEST NOT YET FROZEN",
        "metadata_valid": errors == 0,
        "all_image_files_verified": bool(rows) and len(decoded) == len(rows) and sha_mismatches == 0,
        "physical_specimen_independence_confirmed": False,
    }
    return report, quality
