"""Reviewed multi-source candidate builder; immutable V5 baseline and explicit admission."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from freshlens_ai.constants import CLASSES
from freshlens_ai.data.camera_dataset import (
    MANIFEST_COLUMNS, PROVENANCE_FIELDS, HARD_FIELDS, baseline_index, enabled, known, scan_camera, inventory_leakage,
)
from freshlens_ai.data.dataset_audit import (
    OTHER_COLUMNS, QUALITY_COLUMNS, SUPPORTED_COLUMNS, audit_dataset, inspect_image,
    identity_audit, mapped_image_path, path_key, read_csv, relative_path,
)
from freshlens_ai.data.dataset_v2 import legacy_split_core, verify_candidate_lock
from freshlens_ai.data.image_io import safe_path
from freshlens_ai.utils.file_io import atomic_json, file_sha256, write_csv

SCHEMA = "freshlens_reviewed_candidate_1"
COLUMNS = tuple(dict.fromkeys((*SUPPORTED_COLUMNS, *MANIFEST_COLUMNS, "root_id", "image_path")))


def sources_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, list):
        raise ValueError("Sources JSON must be a list")
    seen = set()
    for source in config:
        if not isinstance(source, dict):
            raise ValueError("Every source must be an object")
        identifier = source.get("id", "")
        if (not identifier or not identifier.replace("_", "").replace("-", "").isalnum()
                or identifier in seen or identifier in ("raw", "baseline")):
            raise ValueError("Source id must be unique, stable and path-safe")
        seen.add(identifier)
        if source.get("kind") not in ("supported", "other") or not source.get("root"):
            raise ValueError("Source needs kind supported|other and root")
    return config


def admission(row, other=False):
    if row["review_status"] != "APPROVED" or any(not known(row.get(k)) for k in
        ("reviewer", "review_evidence", "label_evidence", "group_id", "specimen_evidence", "source", "source_name")):
        raise ValueError("Eligible row requires APPROVED review, label/provenance/group evidence: " + row["path"])
    if row["decode_status"] != "OK" or row["sha256_status"] == "MISMATCH":
        raise ValueError("Eligible row failed SHA/decode: " + row["path"])
    if row["source"] in ("web", "external", "dataset_external"):
        if any(not known(row.get(k)) for k in ("source_url", "license")):
            raise ValueError("External eligible row requires source_url and license")
    if row["source"] == "camera":
        if any(not known(row.get(k)) for k in ("specimen_id", "capture_device", "capture_context", "session_id")):
            raise ValueError("Camera eligible row requires specimen/capture evidence")
    exact = {m["reference"] for m in json.loads(row["duplicate_matches"])}
    near = [m for m in json.loads(row["perceptual_candidates"]) if m["reference"] not in exact]
    if near and (row.get("perceptual_review_status") != "APPROVED" or not known(row.get("perceptual_review_evidence"))):
        raise ValueError("Perceptual candidates require explicit review evidence")
    if row["requested_split"] == "test":
        raise ValueError("New supplementary data cannot enter test; independent evaluation needs a separate frozen protocol")
    if row["requested_split"] not in ("UNASSIGNED", "train", "val"):
        raise ValueError("Invalid reviewed split")
    if other and enabled(row["eligible_for_training"]):
        raise ValueError("OTHER cannot be eligible for 8-class training")


def split_reviewed(rows, baseline_count, seed):
    """Union group/SHA/pixel links; freeze baseline, new components only train/val."""
    core = legacy_split_core()
    union = core.UnionFind(len(rows))
    owners = {}
    label_owners = defaultdict(set)
    for i, row in enumerate(rows):
        label = (row.get("label_type", "SUPPORTED"), row.get("fruit"), row.get("status"),
                 row.get("category", ""), row.get("subcategory", ""))
        for key in ("sha256", "pixel_sha256"):
            if known(row.get(key)):
                label_owners[(key, row[key])].add(label[:3] if label[0] == "SUPPORTED" else (label[0],*label[3:]))
        for key in ("group_id", "sha256", "pixel_sha256", "specimen_id"):
            if known(row.get(key)):
                identity = (key,row[key])
                if identity in owners:
                    union.union(i,owners[identity])
                else:
                    owners[identity] = i
    if any(len(v)>1 for v in label_owners.values()):
        raise ValueError("Exact content has conflicting labels/categories; review required")
    components = defaultdict(list)
    for i in range(len(rows)):
        components[union.find(i)].append(i)
    for indices in components.values():
        frozen = {rows[i]["split"] for i in indices if i < baseline_count}
        declared = {rows[i]["requested_split"] for i in indices if i >= baseline_count and rows[i]["requested_split"] in ("train","val")}
        if len(frozen | declared)>1:
            raise ValueError("Group/SHA/pixel/specimen component crosses frozen splits")
        if "test" in frozen and any(i >= baseline_count for i in indices):
            raise ValueError("New source overlaps baseline test; no admission")
        if frozen or declared:
            split = next(iter(frozen | declared))
        else:
            # Reproducible group assignment without introducing a new test protocol.
            import hashlib
            identity = min(rows[i]["group_id"] for i in indices)
            value = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16],16)
            split = "val" if value % 5 == 0 else "train"
        for i in indices:
            if i >= baseline_count:
                rows[i]["split"] = split
    leak = inventory_leakage(rows)
    if leak["cross_split_group"] or leak["cross_split_sha"] or leak["specimen_cross_split"] or leak["specimen_multiple_groups"]:
        raise ValueError("Group/specimen leakage detected")
    pixels = identity_audit([dict(r, sha256=r.get("pixel_sha256","")) for r in rows])
    if pixels["cross_split_sha256"]["keys"]:
        raise ValueError("Decoded pixel leakage detected")
    return leak


def build_reviewed_candidate(baseline_dir, baseline_root, config_path, output=None, seed=42, dry_run=False):
    baseline_dir, baseline_root = Path(baseline_dir).resolve(), Path(baseline_root).resolve()
    if not dry_run:
        if output is None:
            raise ValueError("New output directory required")
        output = Path(output).resolve()
        project = Path(__file__).resolve().parents[2]
        if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in
            (baseline_dir,baseline_root,project/"data/raw",project/"data/cnn_dataset_v3",project/"data/camera_v1",
             project/"freshlens_ai",project/"scripts",project/"docs",project/"tests",project/".git")):
            raise ValueError("Use a NEW output outside protected inputs/code")
    config = sources_config(config_path) if config_path else []
    config_hash = file_sha256(config_path) if config_path else None
    baseline, lock = baseline_index(baseline_dir)
    if output and any(output.is_relative_to(Path(s["root"]).resolve()) or Path(s["root"]).resolve().is_relative_to(output) for s in config):
        raise ValueError("Candidate output overlaps source")
    mapping = lock.get("image_path_mapping", {})
    original_quality = read_csv(baseline_dir/"image_quality.csv",QUALITY_COLUMNS)
    by_path = {r["path"]:r for r in original_quality}
    rows, quality, excluded = [], [], {}
    for row in baseline:
        image_path = mapping.get("overrides",{}).get(row["path"],mapped_image_path(row["path"],mapping.get("strip_prefix","")))
        if file_sha256(safe_path(baseline_root,image_path)) != row["sha256"]:
            raise ValueError("Baseline image changed: "+row["path"])
        q = dict(by_path[row["path"]])
        # CSV types restored for shared audit's numeric diagnostics.
        for key in ("width","height","file_size","channels"):
            if q.get(key): q[key] = int(q[key])
        for key in ("brightness","blur_score","aspect_ratio"):
            if q.get(key): q[key] = float(q[key])
        for key in ("low_resolution","low_brightness","blur","exif_present"):
            q[key] = str(q.get(key,"")).lower() == "true"
        base_row = {key:row.get(key) or "UNKNOWN" for key in MANIFEST_COLUMNS}
        base_row.update(row)
        base_row.update({key:q[key] if q.get(key) != "" else "NOT_AVAILABLE" for key in
            ("width","height","brightness","blur_score","decode_status","quality_status","format",
             "original_mode","channels","exif_present","exif_orientation","pixel_sha256","phash63")})
        flags=[name for name,key in (("TOO_SMALL","low_resolution"),("DARK","low_brightness"),
               ("BLUR_CANDIDATE","blur")) if q[key]]
        base_row.update(label_type="SUPPORTED",root_id="baseline",image_path=image_path,
            eligible_for_training=row["split"]=="train",eligible_for_evaluation=False,requested_split=row["split"],
            decode_ok=True,quality_flags=";".join(flags) or "NONE",sha256_status="MATCH",expected_sha256=row["sha256"],
            exif=json.dumps({"present":q["exif_present"],"orientation":q.get("exif_orientation") or "NOT_AVAILABLE"}),
            group_evidence_status="BASELINE_DECLARED",review_status="BASELINE_INHERITED",
            duplicate_of="NOT_AVAILABLE",duplicate_type="NOT_AVAILABLE",duplicate_matches="[]",
            perceptual_candidates="[]",perceptual_status="NOT_CHECKED_BY_REVIEWED_BUILDER",
            hard_case_tags=row.get("hard_case_tags") or "UNKNOWN")
        rows.append(base_row)
        quality.append(q)
    fingerprints = {"baseline_lock":file_sha256(baseline_dir/"dataset_lock.json"),"sources_json":config_hash}
    for source in config:
        root = Path(source["root"])
        metadata = source.get("metadata")
        if metadata: fingerprints[source["id"]+"_metadata"] = file_sha256(metadata)
        inventory, _ = scan_camera(root,baseline_dir,metadata,source["kind"])
        chosen = [r for r in inventory if enabled(r["eligible_for_evaluation"] if source["kind"]=="other" else r["eligible_for_training"])]
        excluded[source["id"]] = len(inventory)-len(chosen)
        for row in chosen:
            admission(row,source["kind"]=="other")
            relative = row["path"]
            q = inspect_image(root,dict(path=relative,sha256=row["sha256"]),fingerprints=True)
            if q["sha256_status"] != "MATCH" or q["decode_status"] != "OK":
                raise ValueError("Reviewed image changed or failed decode")
            row = dict(row, path=source["id"]+"/"+relative,root_id=source["id"],image_path=relative)
            row["split"] = ""
            row["viewpoint"] = row.get("angle", "UNKNOWN")
            q["path"] = row["path"]
            rows.append(row); quality.append(q)
    if identity_audit(rows)["duplicate_path"]["keys"]:
        raise ValueError("Duplicate candidate path")
    baseline_count = len(baseline)
    leakage = split_reviewed(rows,baseline_count,seed)
    supported = [r for r in rows if r["label_type"] != "OTHER"]
    other = [r for r in rows if r["label_type"] == "OTHER"]
    quality_by_path = {q["path"]:q for q in quality}
    report,_ = audit_dataset(supported,other,quality_records=[quality_by_path[r["path"]] for r in supported+other])
    if not report["metadata_valid"] or not report["all_image_files_verified"]:
        raise ValueError("Candidate integrity or metadata audit failed")
    if [{k:r[k] for k in SUPPORTED_COLUMNS} for r in rows[:baseline_count]] != [{k:r[k] for k in SUPPORTED_COLUMNS} for r in baseline]:
        raise ValueError("Baseline fields unexpectedly changed")
    if verify_candidate_lock(baseline_dir) != lock or (config_path and file_sha256(config_path) != config_hash):
        raise ValueError("Build inputs changed")
    for source in config:
        if source.get("metadata") and file_sha256(source["metadata"]) != fingerprints[source["id"]+"_metadata"]:
            raise ValueError("Review metadata changed during build")
    report.update(schema_version=SCHEMA,baseline_records_preserved=True,source_fingerprints=fingerprints,
        added_supported_records=len(supported)-baseline_count,added_other_records=len(other),excluded_unreviewed=excluded,
        seed=seed,split_policy="Freeze baseline; group/SHA/pixel/specimen union; new groups deterministic 80/20 train/val; no new test",
        group_leakage=leakage,phash_new_source_candidates=sum(bool(json.loads(r.get("perceptual_candidates","[]"))) for r in rows[baseline_count:]),eligible_for_training_records=sum(r.get("split")=="train" and r["label_type"]=="SUPPORTED" for r in rows),
        hard_case_distribution=dict(Counter(tag for r in rows[baseline_count:] for tag in r.get("hard_case_tags","UNKNOWN").split(";"))),
        provenance_completeness={key:sum(known(r.get(key)) for r in rows[baseline_count:]) for key in ("source_name","source_url","license","specimen_evidence","review_evidence")},
        new_records=len(rows)-baseline_count,baseline_quality_evidence="Locked V5 quality reused only after rehashing every baseline image",
        production_loader_compatible=False,ready_for_pilot_training=True,final_test_status="FINAL TEST NOT YET FROZEN",
        root_ids=["baseline"]+[s["id"] for s in config],dry_run=dry_run)
    if dry_run:
        return report
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/"manifest.csv",supported,COLUMNS)
    write_csv(output/"other_manifest.csv",other,COLUMNS)
    write_csv(output/"image_quality.csv",quality,QUALITY_COLUMNS)
    atomic_json(output/"dataset_report.json",report)
    atomic_json(output/"dataset_lock.json",dict(schema_version=SCHEMA,
        files={name:file_sha256(output/name) for name in ("manifest.csv","other_manifest.csv","image_quality.csv","dataset_report.json")},
        baseline_lock_sha256=fingerprints["baseline_lock"],source_fingerprints=fingerprints))
    return report


def load_reviewed_candidate(directory):
    directory = Path(directory)
    lock = json.loads((directory/"dataset_lock.json").read_text(encoding="utf-8-sig"))
    if lock.get("schema_version") != SCHEMA:
        raise ValueError("Not a reviewed candidate lock")
    required={"manifest.csv","other_manifest.csv","image_quality.csv","dataset_report.json"}
    if set(lock.get("files",{})) != required:
        raise ValueError("Reviewed candidate file inventory mismatch")
    for name,digest in lock["files"].items():
        if file_sha256(directory/name) != digest:
            raise ValueError("Reviewed candidate modified after build")
    supported=read_csv(directory/"manifest.csv",SUPPORTED_COLUMNS)
    other=read_csv(directory/"other_manifest.csv",("path","category","subcategory","sha256","split"))
    for row in supported:
        if row["fruit"]+"::"+row["status"] not in CLASSES or row.get("label_type") != "SUPPORTED":
            raise ValueError("Unsupported production label in supported manifest")
        row["target"]=CLASSES.index(row["fruit"]+"::"+row["status"])
    from freshlens_ai.data.dataset_audit import validate_rows
    if validate_rows(supported) or validate_rows(other, "other"):
        raise ValueError("Invalid reviewed manifest labels or metadata")
    leakage = inventory_leakage(supported+other)
    pixels=identity_audit([dict(r,sha256=r.get("pixel_sha256", "")) for r in supported+other])
    if (leakage["identity"]["duplicate_path"]["keys"] or leakage["cross_split_group"] or leakage["cross_split_sha"] or leakage["specimen_cross_split"] or pixels["cross_split_sha256"]["keys"]):
        raise ValueError("Reviewed candidate leakage")
    if any(r.get("label_type") != "OTHER" or enabled(r.get("eligible_for_training")) for r in other):
        raise ValueError("OTHER cannot enter supported training")
    return supported,other


def make_reviewed_dataset(directory, roots, split, training=False):
    """TV2 opt-in Dataset adapter; no training/inference entrypoint changes."""
    if split not in ("train", "val", "test"):
        raise ValueError("Split must be train/val/test")
    if training and split != "train":
        raise ValueError("Training augmentation cannot be used for validation/test")
    from torch.utils.data import Dataset
    from freshlens_ai.data.transforms import image_transform
    from freshlens_ai.data.image_io import read_record_image
    supported,_=load_reviewed_candidate(directory)
    selected=[r for r in supported if r["split"]==split]
    class ReviewedDataset(Dataset):
        def __init__(self):
            self.rows=selected
            self.transform=image_transform(training)
        def __len__(self):
            return len(self.rows)
        def __getitem__(self,index):
            row=self.rows[index]
            image=read_record_image(roots[row["root_id"]],dict(row,path=relative_path(row["image_path"])))
            return self.transform(image),row["target"],index
    return ReviewedDataset()


def audit_reviewed_candidate(directory, baseline_root, config_path=None):
    from concurrent.futures import ThreadPoolExecutor
    supported, other = load_reviewed_candidate(directory)
    config = sources_config(config_path) if config_path else []
    roots = {"baseline": baseline_root, **{s["id"]:s["root"] for s in config}}
    rows = supported + other
    def measure(row):
        if row["root_id"] not in roots:
            raise ValueError("Missing image root for " + row["root_id"])
        q=inspect_image(roots[row["root_id"]],dict(row,path=row["image_path"]),fingerprints=True)
        q["path"]=row["path"]
        return q
    with ThreadPoolExecutor(max_workers=4) as pool:
        quality=list(pool.map(measure,rows))
    for row in supported+other:
        row["viewpoint"] = row.get("angle",row.get("viewpoint","UNKNOWN"))
    return audit_dataset(supported,other,quality_records=quality)
