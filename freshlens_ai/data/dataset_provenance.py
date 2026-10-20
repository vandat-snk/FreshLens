"""Conservative read-only provenance review of raw inventory and unmatched images."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from freshlens_ai.constants import CLASSES, PROJECT_DIR
from freshlens_ai.data.dataset_audit import (
    QUALITY_COLUMNS, inspect_image, mapped_image_path, path_key, read_csv, relative_path,
)
from freshlens_ai.data.dataset_v2 import baseline_snapshot, legacy_split_core
from freshlens_ai.data.image_io import safe_path
from freshlens_ai.utils.file_io import file_sha256

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CATEGORIES = ("exact_duplicate", "probable_augmentation", "probable_new_image", "unknown", "corrupt")


def raw_relative_path(raw_root, reference):
    """Accept repository-relative catalog paths or paths relative to raw_root."""
    relative = relative_path(reference)
    root = Path(raw_root).resolve()
    project_path = (PROJECT_DIR / relative).resolve()
    if project_path.is_relative_to(root):
        return project_path.relative_to(root).as_posix()
    # A catalog already relative to its raw root is also valid.
    if PurePosixPath(relative).parts[0] in ("apple", "banana", "orange", "tomato"):
        return relative
    raise ValueError("Catalog path does not belong to configured raw root: " + relative)


def classify_evidence(decode_status, byte_reference="", pixel_reference="", named_parent=False,
                      known_transform=False, verified_new_provenance=False):
    """Similarity alone never proves derivation or a new independent sample."""
    if decode_status != "OK":
        return "corrupt", "decode_or_read_failed"
    if byte_reference:
        return "exact_duplicate", "sha256_exact_match"
    if pixel_reference:
        return "exact_duplicate", "decoded_rgb_pixels_exact_match"
    if known_transform and named_parent:
        return "probable_augmentation", "explicit_transform_filename_and_available_named_parent"
    if verified_new_provenance:
        return "probable_new_image", "declared_source_and_label_evidence_require_human_review"
    return "unknown", "insufficient_parent_or_independent_source_evidence"


def analyze_provenance(baseline_dir, originals_root, raw_root, original_quality_csv,
                       matched_csv, duplicates_csv, unmatched_csv, strip_prefix="raw", progress=True, workers=4):
    baseline, baseline_hashes, _ = baseline_snapshot(baseline_dir)
    inputs = {name: Path(value) for name, value in {
        "original_quality": original_quality_csv, "matched_catalog": matched_csv,
        "duplicate_catalog": duplicates_csv, "unmatched_catalog": unmatched_csv,
    }.items()}
    fingerprints = {name: file_sha256(path) for name, path in inputs.items()}
    quality = read_csv(original_quality_csv, QUALITY_COLUMNS)
    by_path = {q["path"]: q for q in quality}
    if len(by_path) != len(quality) or set(by_path) != {r["path"] for r in baseline}:
        raise ValueError("Original quality CSV does not cover the exact baseline manifest")
    core = legacy_split_core()
    original_sha, original_pixels, families, phash_members = {}, {}, defaultdict(list), defaultdict(list)
    tree = core.HashTree()
    for index, row in enumerate(baseline, 1):
        q = by_path[row["path"]]
        if (q["decode_status"] != "OK" or q["sha256_status"] != "MATCH"
                or q["actual_sha256"] != row["sha256"] or q["sha256"] != row["sha256"]
                or not re.fullmatch(r"[0-9a-f]{64}", q["pixel_sha256"])
                or not re.fullmatch(r"[0-9a-f]{16}", q["phash63"])):
            raise ValueError("Original audit/fingerprints incomplete or mismatching: " + row["path"])
        if file_sha256(safe_path(originals_root, q["resolved_relative_path"])) != row["sha256"]:
            raise ValueError("Original changed since audit: " + row["path"])
        original_sha[row["sha256"]] = row
        original_pixels.setdefault(q["pixel_sha256"], row)
        info = core.filename_info(row)
        if info["family"]:
            families[(row["fruit"], row["status"], info["family"])].append(row)
        value = int(q["phash63"], 16)
        phash_members[value].append(row)
        tree.add(value)
        if progress and index % 1000 == 0:
            print(f"[PROVENANCE] baseline fingerprint inputs {index}/{len(baseline)}", flush=True)

    matched = read_csv(matched_csv, ("raw_path", "manifest_path", "fruit", "status", "source", "group_id", "sha256", "split"), "raw_path")
    duplicates = read_csv(duplicates_csv, ("raw_path", "manifest_path", "fruit", "status", "source", "group_id", "sha256", "split"), "raw_path")
    unmatched = read_csv(unmatched_csv, ("raw_path", "sha256", "extension"), "raw_path")
    catalogs = matched + duplicates + unmatched
    root = Path(raw_root).resolve()
    catalog_paths = [raw_relative_path(root, r["raw_path"]) for r in catalogs]
    if len({path_key(p) for p in catalog_paths}) != len(catalog_paths):
        raise ValueError("Raw catalogs overlap or repeat paths")
    actual_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    actual_relatives = {path_key(p.relative_to(root).as_posix()) for p in actual_files}
    if actual_relatives != {path_key(p) for p in catalog_paths}:
        raise ValueError("Raw filesystem and SHA partition catalogs do not cover identical image files")
    baseline_by_path = {r["path"]: r for r in baseline}
    if len(matched) != len(baseline) or {r["manifest_path"] for r in matched} != set(baseline_by_path):
        raise ValueError("Matched catalog does not cover all baseline records exactly once")
    for index, raw in enumerate(matched + duplicates, 1):
        reference = baseline_by_path.get(raw["manifest_path"])
        if reference is None or any(raw[k] != reference[k] for k in ("fruit", "status", "source", "group_id", "sha256", "split")):
            raise ValueError("Matched/duplicate catalog disagrees with baseline: " + raw["raw_path"])
        relative = raw_relative_path(root, raw["raw_path"])
        if tuple(PurePosixPath(relative).parts[:2]) != (reference["fruit"], reference["status"]):
            raise ValueError("Raw folder label contradicts baseline catalog")
        if file_sha256(safe_path(root, relative)) != raw["sha256"]:
            raise ValueError("Raw SHA partition changed: " + raw["raw_path"])
        if progress and index % 1000 == 0:
            print(f"[PROVENANCE] matched/duplicate bytes verified {index}/{len(matched) + len(duplicates)}", flush=True)

    results, seen_sha, seen_pixels = [], {}, {}
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Provenance workers must be between 1 and 8")
    ordered = sorted(unmatched, key=lambda r:path_key(r["raw_path"]))
    def inspect_unmatched(raw):
        relative = raw_relative_path(root, raw["raw_path"])
        return inspect_image(root, dict(path=relative, sha256=raw["sha256"]), fingerprints=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (raw, q) in enumerate(zip(ordered, executor.map(inspect_unmatched, ordered)), 1):
            relative = raw_relative_path(root, raw["raw_path"])
            parts = PurePosixPath(relative).parts
            fruit_hint, status_hint = parts[:2]
            if fruit_hint + "::" + status_hint not in CLASSES:
                raise ValueError("Unmatched path has an unsupported folder label")
            if q["sha256_status"] == "MISMATCH":
                raise ValueError("Unmatched SHA catalog is stale: " + raw["raw_path"])
            byte_row = original_sha.get(q["actual_sha256"])
            pixel_row = original_pixels.get(q["pixel_sha256"])
            byte_reference = byte_row["path"] if byte_row else seen_sha.get(q["actual_sha256"], "")
            pixel_reference = pixel_row["path"] if pixel_row else seen_pixels.get(q["pixel_sha256"], "")
            info = core.filename_info(dict(path=relative, fruit=fruit_hint, status=status_hint))
            parents = families.get((fruit_hint, status_hint, info["family"]), []) if info["family"] else []
            known_transform = info["kind"] == "offline_augmentation"
            near = []
            if q["phash63"]:
                for value, distance in tree.query(int(q["phash63"], 16), 4):
                    near.extend((distance, r["path"], r["group_id"], r["split"]) for r in phash_members[value])
            near.sort()
            declared_new = (raw.get("label_verified") == "true" and bool(raw.get("provenance"))
                            and bool(raw.get("source")) and bool(raw.get("group_id"))
                            and bool(raw.get("independent_capture_evidence"))
                            and raw.get("license") not in (None, "", "not_verified"))
            category, reason = classify_evidence(q["decode_status"], byte_reference, pixel_reference,
                                                 bool(parents), known_transform, declared_new)
            reference = byte_row or pixel_row
            output = {
                "raw_path": raw["raw_path"], "raw_relative_path": relative,
                "folder_fruit_hint": fruit_hint, "folder_status_hint": status_hint,
                "declared_sha256": raw["sha256"], "actual_sha256": q["actual_sha256"],
                "sha256_status": q["sha256_status"], "decode_status": q["decode_status"],
                "classification": category, "reason": reason,
                "duplicate_reference": byte_reference or pixel_reference,
                "duplicate_reference_group": reference["group_id"] if reference else "",
                "duplicate_reference_split": reference["split"] if reference else "",
                "filename_kind": info["kind"], "filename_family": info["family"],
                "named_parent_count": len(parents),
                "named_parent_paths": ";".join(r["path"] for r in parents),
                "named_parent_groups": ";".join(sorted({r["group_id"] for r in parents})),
                "named_parent_splits": ";".join(sorted({r["split"] for r in parents})),
                "phash_candidate_count": len(near),
                "nearest_phash_distance": near[0][0] if near else "",
                "nearest_manifest_path": near[0][1] if near else "",
                "near_match_splits": ";".join(sorted({n[3] for n in near})),
                "independent_specimen_confirmed": False,
                "eligible_for_candidate": False,
                "admission_note": "excluded_pending_verified_label_provenance_and_group_review",
                **{key:q[key] for key in ("width", "height", "format", "original_mode", "channels", "exif_present",
                                         "exif_orientation", "pixel_sha256", "phash63", "brightness", "blur_score")},
            }
            results.append(output)
            if q["actual_sha256"]:
                seen_sha.setdefault(q["actual_sha256"], raw["raw_path"])
            if q["pixel_sha256"]:
                seen_pixels.setdefault(q["pixel_sha256"], raw["raw_path"])
            if progress and (index % 250 == 0 or index == len(unmatched)):
                print(f"[PROVENANCE] unmatched {index}/{len(unmatched)}", flush=True)
    if fingerprints != {name:file_sha256(path) for name,path in inputs.items()}:
        raise ValueError("A provenance input changed during review")
    if baseline_snapshot(baseline_dir)[1] != baseline_hashes:
        raise ValueError("Baseline changed during provenance review")
    counts = Counter(r["classification"] for r in results)
    report = {
        "schema_version": "freshlens_unmatched_provenance_1",
        "analysis_script_sha256": file_sha256(__file__),
        "workers": workers, "classification_order": "sorted canonical paths; ordered executor results",
        "input_fingerprints": fingerprints, "baseline_fingerprints": baseline_hashes,
        "raw_image_files": len(actual_files),
        "non_image_files": [p.relative_to(root).as_posix() for p in root.rglob("*")
                            if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES],
        "partition_counts": {"matched_originals": len(matched), "duplicate_originals": len(duplicates), "unmatched_images": len(unmatched)},
        "partition_verified": True, "originals_rehashed": len(baseline),
        "matched_and_duplicate_raw_rehashed": len(matched) + len(duplicates),
        "unmatched_rehashed": len(results),
        "classification_counts": {c:counts[c] for c in CATEGORIES},
        "classification_by_folder_hint": {
            label: {c:sum(r["classification"] == c and r["folder_fruit_hint"] + "::" + r["folder_status_hint"] == label
                          for r in results) for c in CATEGORIES} for label in CLASSES
        },
        "filename_kind_counts": dict(Counter(r["filename_kind"] for r in results)),
        "reason_counts": dict(Counter(r["reason"] for r in results)),
        "unmatched_with_phash_candidates": sum(r["phash_candidate_count"] > 0 for r in results),
        "unmatched_with_exif": sum(r["exif_present"] is True for r in results),
        "format_counts": dict(Counter(r["format"] for r in results)),
        "admitted_new_samples": 0,
        "criteria": {
            "exact_duplicate": "Exact encoded SHA or exact oriented/composited RGB pixels at identical dimensions; reference recorded.",
            "probable_augmentation": "Recognized explicit transform filename AND an available same-folder-label named original; derivation probable, not visually confirmed.",
            "probable_new_image": "Explicit independent-capture/source/license/label evidence in input metadata; still needs human review before admission.",
            "unknown": "Insufficient provenance; no augmentation parent inferred from absence/presence of a generic augmented marker or pHash alone.",
            "corrupt": "Image cannot be read or decoded.",
            "phash": "Reuse legacy 63 non-DC DCT bits; Hamming <=4; candidates only, no automatic merging or label changes.",
        },
        "limitations": [
            "Folder labels are hints, not independently reviewed new-image labels.",
            "Filename derivation is not proven by a probable_augmentation classification.",
            "pHash can miss crops/rotations and can match different specimens.",
            "Perceptual analysis is dataset-integrity review, not model/threshold/augmentation tuning on test.",
            "75 duplicate catalog entries do not create 75 independent new samples.",
            "No raw file was changed, moved, deleted or automatically admitted.",
        ],
    }
    return report, results
