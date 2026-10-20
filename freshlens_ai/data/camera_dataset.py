"""Unassigned supplementary camera/hard-case inventory; never modifies baseline data."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from freshlens_ai.constants import CLASSES
from freshlens_ai.data.dataset_audit import (
    QUALITY_COLUMNS, SUPPORTED_COLUMNS, inspect_image, identity_audit, path_key, read_csv, relative_path,
)
from freshlens_ai.data.dataset_provenance import IMAGE_SUFFIXES
from freshlens_ai.data.dataset_v2 import legacy_split_core, verify_candidate_lock
from freshlens_ai.utils.file_io import atomic_json, file_sha256, write_csv

HARD_FIELDS = ("lighting", "distance", "angle", "background", "occlusion",
               "multiple_objects", "noise", "blur", "glare", "severity")
PROVENANCE_FIELDS = ("source_url", "license", "license_url", "source_name", "author",
                     "attribution", "download_url", "retrieved_date", "label_evidence",
                     "specimen_id", "specimen_evidence", "session_id", "capture_device", "capture_context", "provenance")
REVIEW_FIELDS = ("eligible_for_training", "eligible_for_evaluation", "requested_split", "review_status", "reviewer", "review_evidence", "perceptual_review_status", "perceptual_review_evidence")
METADATA_COLUMNS = ("path", "source", "group_id", "sha256", *PROVENANCE_FIELDS, *HARD_FIELDS, *REVIEW_FIELDS)
MANIFEST_COLUMNS = (
    "path", "fruit", "status", "label_type", "category", "subcategory", "source", "group_id", "group_evidence_status", "split",
    "sha256", "expected_sha256", "sha256_status", "width", "height", "brightness",
    "blur_score", "quality_flags", "decode_ok", "exif", "decode_status", "quality_status", "format",
    "original_mode", "channels", "exif_present", "exif_orientation", "pixel_sha256",
    "phash63", "duplicate_of", "duplicate_type", "duplicate_matches",
    "perceptual_candidates", "perceptual_status", "duplicate_internal", "duplicate_with_baseline", "hard_case_tags", *REVIEW_FIELDS,
    *PROVENANCE_FIELDS, *HARD_FIELDS,
)
SCHEMA = "freshlens_camera_inventory_1"


def image_label(path, kind="supported"):
    if kind == "other":
        other_category(path)
        return "UNKNOWN", "UNKNOWN"
    parts = Path(relative_path(path)).parts
    if len(parts) < 3 or parts[0] + "::" + parts[1] not in CLASSES:
        raise ValueError("Camera path must belong to one of eight fruit/status folders: " + path)
    return parts[0], parts[1]


def baseline_index(directory):
    """Use immutable V5 lock and actual baseline fingerprint sidecar, read-only."""
    lock = verify_candidate_lock(directory)
    baseline = read_csv(Path(directory) / "manifest.csv", SUPPORTED_COLUMNS)
    quality = read_csv(Path(directory) / "image_quality.csv", QUALITY_COLUMNS)
    lookup = {r["path"]: r for r in quality}
    if len(lookup) != len(quality) or set(lookup) != {r["path"] for r in baseline}:
        raise ValueError("Baseline quality coverage mismatch")
    for row in baseline:
        q = lookup[row["path"]]
        if q["actual_sha256"] != row["sha256"] or q["decode_status"] != "OK":
            raise ValueError("Baseline fingerprints are not fully verified")
    return [dict(r, pixel_sha256=lookup[r["path"]]["pixel_sha256"],
                 phash63=lookup[r["path"]]["phash63"]) for r in baseline], lock


def scan_camera(root, baseline_dir, metadata=None, kind="supported"):
    root = Path(root).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("Inventory root is not a directory")
    if kind not in ("supported", "other"):
        raise ValueError("Unknown inventory kind")
    baseline, lock = baseline_index(baseline_dir)
    manual = read_csv(metadata, ("path",)) if metadata else []
    declared = {}
    for row in manual:
        image_label(row["path"], kind)
        key = path_key(row["path"])
        if key in declared:
            raise ValueError("Repeated metadata path")
        declared[key] = row
    discovered, ignored = {}, []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            ignored.append(rel)
            continue
        image_label(rel, kind)
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("Camera image link escapes or aliases root")
        if path_key(rel) in discovered:
            raise ValueError("Case-insensitive duplicate path")
        discovered[path_key(rel)] = rel
    paths = dict(discovered)
    for key, row in declared.items():
        paths.setdefault(key, row["path"])
    metadata_hash = file_sha256(metadata) if metadata else None
    rows = []
    for key, path in sorted(paths.items()):
        info = declared.get(key, {})
        fruit, status = image_label(path, kind)
        source = info.get("source") or "UNKNOWN"
        if source not in ("camera", "web", "external", "dataset_external", "UNKNOWN"):
            raise ValueError("Source must be camera/web/external/dataset_external/UNKNOWN")
        expected = info.get("sha256", "")
        q = inspect_image(root, dict(path=path, sha256=expected), fingerprints=True)
        sha_status = q["sha256_status"] if expected else ("COMPUTED" if q["actual_sha256"] else "NOT_AVAILABLE")
        digest = q["actual_sha256"]
        flags = [name for name, flag in (("TOO_SMALL", q["low_resolution"]),
                 ("DARK", q["low_brightness"]), ("BLUR_CANDIDATE", q["blur"])) if flag is True]
        if q["decode_status"] != "OK":
            flags.append(q["decode_status"])
        if expected and sha_status == "MISMATCH":
            flags.append("SHA256_MISMATCH")
        if q["brightness"] != "" and float(q["brightness"]) > 220:
            flags.append("HIGH_BRIGHTNESS_CANDIDATE")
        group = info.get("group_id") if known(info.get("group_id")) else "UNKNOWN"
        category, subcategory = other_category(path) if kind == "other" else ("NOT_AVAILABLE", "NOT_AVAILABLE")
        row = dict(
            path=path, fruit=fruit, status=status, label_type="OTHER" if kind == "other" else "SUPPORTED",
            category=category, subcategory=subcategory, source=source, group_id=group,
            group_evidence_status="DECLARED" if known(group) else "UNVERIFIED",
            split="UNASSIGNED", sha256=digest or "NOT_AVAILABLE",
            expected_sha256=expected or "NOT_AVAILABLE", sha256_status=sha_status,
            quality_flags=";".join(flags) or "NONE",
            **{k:q[k] if q[k] != "" else "NOT_AVAILABLE" for k in (
                "width", "height", "brightness", "blur_score", "decode_status",
                "quality_status", "format", "original_mode", "channels", "exif_present",
                "exif_orientation", "pixel_sha256", "phash63")},
            eligible_for_training=enabled(info.get("eligible_for_training")) if kind == "supported" else False,
            eligible_for_evaluation=enabled(info.get("eligible_for_evaluation")),
            decode_ok=q["decode_status"] == "OK",
            exif=json.dumps({"present":q["exif_present"] or False,"orientation":q["exif_orientation"] or "NOT_AVAILABLE"}),
        )
        row.update({k:info.get(k) or "UNKNOWN" for k in (*PROVENANCE_FIELDS, *HARD_FIELDS)})
        row.update({k:info.get(k) or "UNKNOWN" for k in ("review_status", "reviewer", "review_evidence", "perceptual_review_status", "perceptual_review_evidence")})
        row["requested_split"] = info.get("requested_split") or info.get("split") or "UNASSIGNED"
        if row["requested_split"] not in ("UNASSIGNED", "train", "val", "test"):
            raise ValueError("Invalid requested_split")
        row["hard_case_tags"] = hard_case_tags(info)
        rows.append(row)
    # Flag every duplicate member, including the first camera image in a cluster.
    byte_members, pixel_members = defaultdict(list), defaultdict(list)
    tree, phash_members = legacy_split_core().HashTree(), defaultdict(list)
    for scope, items in (("baseline", baseline), ("camera", rows)):
        for item in items:
            ref = scope + ":" + item["path"]
            if item["sha256"] not in ("", "NOT_AVAILABLE"):
                byte_members[item["sha256"]].append(ref)
            if item.get("pixel_sha256") not in ("", None, "NOT_AVAILABLE"):
                pixel_members[item["pixel_sha256"]].append(ref)
            if item.get("phash63") not in ("", None, "NOT_AVAILABLE"):
                value = int(item["phash63"], 16)
                tree.add(value)
                phash_members[value].append(ref)
    for row in rows:
        own = "camera:" + row["path"]
        matches = []
        for duplicate_kind, index, value in (("SHA256", byte_members, row["sha256"]),
                                  ("DECODED_PIXEL", pixel_members, row["pixel_sha256"])):
            matches.extend(dict(reference=ref, type=duplicate_kind) for ref in sorted(index.get(value, [])) if ref != own)
        near = []
        if row["phash63"] != "NOT_AVAILABLE":
            for value, distance in tree.query(int(row["phash63"], 16), 4):
                near.extend(dict(reference=ref, distance=distance) for ref in phash_members[value] if ref != own)
        near.sort(key=lambda v:(v["distance"],v["reference"]))
        row.update(
            duplicate_internal=any(m["reference"].startswith("camera:") for m in matches),
            duplicate_with_baseline=any(m["reference"].startswith("baseline:") for m in matches),
            duplicate_of=";".join(sorted({m["reference"] for m in matches})) or "NONE",
            duplicate_type=";".join(sorted({m["type"] for m in matches})) or "NONE",
            duplicate_matches=json.dumps(matches, sort_keys=True),
            perceptual_candidates=json.dumps(near, sort_keys=True),
            perceptual_status="REVIEW_CANDIDATE" if near else "NO_BASELINE_CANDIDATE",
        )
    if metadata and file_sha256(metadata) != metadata_hash:
        raise ValueError("Metadata changed during scan")
    if verify_candidate_lock(baseline_dir) != lock:
        raise ValueError("Baseline lock changed during scan")
    import numpy as np
    measurements = {}
    for metric in ("width", "height", "brightness", "blur_score"):
        values = np.asarray([float(r[metric]) for r in rows if r["decode_ok"]], dtype=float)
        measurements[metric] = {k:round(float(v),6) for k,v in zip(
            ("min","median","mean","max"),(values.min(),np.median(values),values.mean(),values.max())
        )} if len(values) else "NOT_AVAILABLE"
    report = dict(
        schema_version=SCHEMA, inventory_kind=kind, records=len(rows),
        supported_combinations=list(CLASSES),
        class_counts={label:sum(r["fruit"]+"::"+r["status"]==label for r in rows) for label in CLASSES},
        source_counts=dict(Counter(r["source"] for r in rows)),
        decode_counts=dict(Counter(r["decode_status"] for r in rows)),
        quality_flags=dict(Counter(flag for r in rows for flag in r["quality_flags"].split(";") if flag != "NONE")),
        duplicate_camera_records=sum(any(m["reference"].startswith("camera:") for m in json.loads(r["duplicate_matches"])) for r in rows),
        duplicate_baseline_records=sum(any(m["reference"].startswith("baseline:") for m in json.loads(r["duplicate_matches"])) for r in rows),
        perceptual_baseline_candidates=sum(any(m["reference"].startswith("baseline:") for m in json.loads(r["perceptual_candidates"])) for r in rows),
        measurements=measurements,
        fruit_counts=dict(Counter(r["fruit"] for r in rows if r["label_type"]=="SUPPORTED")),
        status_counts=dict(Counter(r["status"] for r in rows if r["label_type"]=="SUPPORTED")),
        duplicate_path=inventory_leakage(rows)["identity"]["duplicate_path"],
        duplicate_sha=inventory_leakage(rows)["identity"]["duplicate_sha256"],
        duplicate_pixels=identity_audit([dict(r,sha256=r["pixel_sha256"] if known(r["pixel_sha256"]) else "") for r in rows])["duplicate_sha256"],
        hard_case_distribution=dict(Counter(tag for r in rows for tag in r["hard_case_tags"].split(";"))),
        provenance_completeness={k:sum(known(r.get(k)) for r in rows) for k in PROVENANCE_FIELDS},
        leakage=inventory_leakage(rows),
        category_counts=dict(Counter(r["category"] for r in rows)) if kind == "other" else {},
        sha_mismatches=sum(r["sha256_status"] == "MISMATCH" for r in rows),
        metadata_records=len(manual), metadata_sha256=metadata_hash,
        unknown_source_records=sum(r["source"]=="UNKNOWN" for r in rows),
        unknown_group_evidence=sum(r["group_evidence_status"]=="UNVERIFIED" for r in rows),
        ignored_non_image_files=ignored, baseline_lock_sha256=file_sha256(Path(baseline_dir)/"dataset_lock.json"),
        baseline_fingerprint_evidence="LOCKED_MANIFEST_AND_QUALITY; baseline image bytes not rehashed by camera command",
        split_status="UNASSIGNED", eligible_for_training=False,
        physical_specimen_independence_confirmed=False,
        candidate_thresholds={"min_side":96,"min_brightness":35,"min_blur_score":50,"high_brightness":220},
        limitations=[
            "Folder fruit/status labels require human review; metadata never relabels images.",
            "Unknown specimens keep group_id=UNKNOWN; known groups still need evidence.",
            "Brightness/blur are diagnostic candidates; glare, noise, occlusion and defects are not inferred.",
            "Perceptual baseline matches (legacy pHash <=4) require review; no automatic grouping.",
            "No camera file is deleted, filtered, split or automatically included in training.",
        ],
    )
    return rows, report


def write_camera_inventory(root, baseline_dir, output, metadata=None, kind="supported"):
    output, root, baseline_dir = Path(output).resolve(), Path(root).resolve(), Path(baseline_dir).resolve()
    project = Path(__file__).resolve().parents[2]
    # Keep all image/input datasets read-only; output is one dedicated report directory.
    protected = [root, baseline_dir, project/"data", project/"freshlens_ai", project/"scripts",
                 project/"tests", project/"docs", project/".git"]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("Output must be outside dataset/code/input directories")
    if metadata and (Path(metadata).resolve().is_relative_to(output)):
        raise ValueError("Output must not contain metadata input")
    report_path, manifest_path = output/"audit_report.json", output/"manifest.csv"
    if output.exists():
        if {p.name for p in output.iterdir()} != {"audit_report.json","manifest.csv"}:
            raise ValueError("Refuse to overwrite unrelated output files")
        saved = json.loads(report_path.read_text(encoding="utf-8-sig"))
        saved_kind = saved.get("inventory_kind", "supported")
        if saved_kind == "DECODED_PIXEL":
            # Compatibility with the short-lived loop-variable bug; verify the owned manifest before migration.
            previous = read_csv(manifest_path, ("path", "label_type"))
            expected_type = "OTHER" if kind == "other" else "SUPPORTED"
            if (file_sha256(manifest_path) != saved.get("manifest_sha256") or len(previous) != saved.get("records")
                    or any(row["label_type"] != expected_type for row in previous)):
                raise ValueError("Existing inventory cannot be safely migrated")
            saved_kind = kind
        if saved.get("schema_version") != SCHEMA or saved_kind != kind:
            raise ValueError("Existing output is not this inventory kind")
    rows, report = scan_camera(root, baseline_dir, metadata, kind)
    write_csv(manifest_path, rows, MANIFEST_COLUMNS)
    report["manifest_sha256"] = file_sha256(manifest_path)
    atomic_json(report_path, report)
    return report

def known(value):
    return str(value or "").strip().upper() not in ("", "UNKNOWN", "NOT_AVAILABLE", "UNASSIGNED", "NONE")


def enabled(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in ("", "false", "unknown", "not_available"):
        return False
    if text != "true":
        raise ValueError("Eligibility must be true or false")
    return True


def hard_case_tags(info):
    # Human annotations only; brightness flags never populate semantic metadata.
    mapping = {
        "lighting": {"normal":"normal", "low_light":"low_light", "dark":"low_light",
                     "uneven":"uneven_lighting", "uneven_lighting":"uneven_lighting",
                     "overexposed":"overexposed"},
        "distance": {"far":"far", "small_object":"small_object"},
        "angle": {"tilted":"tilted"},
        "background": {"complex":"complex_background"},
        "occlusion": {"hand_held":"hand_held", "partial":"occlusion", "true":"occlusion"},
        "multiple_objects": {"true":"multiple_objects"},
        "noise": {"true":"noisy", "noisy":"noisy"},
        "blur": {"true":"blurry", "blurry":"blurry"},
        "glare": {"true":"glare"},
        "severity": {"small_defect":"small_defect", "severe_rotten":"severe_rotten"},
    }
    return ";".join(sorted({tag for field, values in mapping.items()
                            if (tag := values.get(str(info.get(field, "")).lower()))})) or "UNKNOWN"


def other_category(path):
    parts = list(Path(relative_path(path)).parts)
    if parts and parts[0] == "other":
        parts.pop(0)
    aliases = {"fruit":"other_fruit", "other_fruit":"other_fruit",
               "non_fruit":"non_fruit", "multiple":"multiple"}
    if len(parts) < 3 or parts[0] not in aliases:
        raise ValueError("OTHER path must be fruit|non_fruit|multiple/subcategory/file")
    return aliases[parts[0]], parts[1]


def inventory_leakage(rows):
    from freshlens_ai.data.dataset_audit import identity_audit
    adapted = [dict(r, group_id=r.get("group_id") if known(r.get("group_id")) else "",
                    sha256=r.get("sha256") if known(r.get("sha256")) else "",
                    split=r.get("requested_split", "") if r.get("split") == "UNASSIGNED" else r.get("split", ""))
               for r in rows]
    identity = identity_audit(adapted)
    # UNASSIGNED/blank entries are not assigned splits.
    assigned = [r for r in adapted if r.get("split") in ("train", "val", "test")]
    actual = identity_audit(assigned)
    specimens = defaultdict(set)
    specimen_groups = defaultdict(set)
    for original, r in zip(rows, adapted):
        if known(original.get("specimen_id")):
            specimen_groups[original["specimen_id"]].add(original.get("group_id", "UNKNOWN"))
            if r["split"] in ("train", "val", "test"):
                specimens[original["specimen_id"]].add(r["split"])
    return dict(
        identity=identity, assigned_identity=actual,
        cross_split_group=actual["cross_split_group_id"]["keys"],
        cross_split_sha=actual["cross_split_sha256"]["keys"],
        specimen_cross_split={k:sorted(v) for k,v in specimens.items() if len(v)>1},
        specimen_multiple_groups={k:sorted(v) for k,v in specimen_groups.items() if len(v)>1},
        unknown_groups=sum(not known(r.get("group_id")) for r in rows),
        physical_specimen_independence_confirmed=False,
    )

