"""Coassign suspected related images before a NEW FreshLens CNN experiment.

Grouping is a conservative evaluation policy, not a declaration of duplicate
identity or a relabelling decision. All images and original labels are kept.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import step2_core as core


VERSION = "2.1.0"
INPUT_FILES = ("manifest.csv", "split_report.json", "near_duplicates.csv")
PAIR_COLUMNS = ("pair_id", "path_a", "path_b", "label_a", "label_b", "split_a", "split_b",
                "group_a", "group_b", "hamming_distance", "cross_split", "different_labels", "decision")


def fail(message):
    raise core.DataError(message)


def boolean(text):
    text = str(text).lower()
    if text not in ("true", "false"):
        fail("Invalid boolean in pair CSV: " + text)
    return text == "true"


def load_inputs(directory):
    directory = Path(directory)
    snapshots = {name: (directory / name).read_bytes() for name in INPUT_FILES}
    report = json.loads(snapshots["split_report.json"].decode("utf-8-sig"))
    rows = core.read_manifest(directory / "manifest.csv")
    if report.get("version") != "2.0.0":
        fail("This step expects a report produced by STEP2_SPLIT.py version 2.0.0.")
    fingerprint = core.sha_bytes(snapshots["manifest.csv"])
    if report.get("output_manifest_sha256") != fingerprint:
        fail("The input manifest does not match split_report.json. Use the unchanged V2 output directory.")
    if report.get("output_records") != len(rows):
        fail("Input report and manifest record counts disagree.")
    if report.get("output_split_counts") != dict(Counter(row["split"] for row in rows)):
        fail("Input report and manifest split counts disagree.")
    if report.get("unique_output_groups") != len({row["group_id"] for row in rows}):
        fail("Input report and manifest group counts disagree.")
    if report.get("near_duplicate_review", {}).get("complete") is not True:
        fail("The previous near-duplicate audit was incomplete. Resolve that before grouping.")
    old_group_splits = defaultdict(set)
    for row in rows:
        old_group_splits[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in old_group_splits.values()):
        fail("An existing group spans multiple splits in the input manifest.")
    reader = csv.DictReader(io.StringIO(snapshots["near_duplicates.csv"].decode("utf-8-sig")))
    if not set(PAIR_COLUMNS).issubset(reader.fieldnames or []):
        fail("The pair CSV has missing columns.")
    pairs = [dict(row) for row in reader]
    validate_pairs(rows, pairs)
    audit = report["near_duplicate_review"]
    if (audit.get("candidate_pairs") != len(pairs)
            or audit.get("cross_split_pairs") != sum(boolean(pair["cross_split"]) for pair in pairs)
            or audit.get("different_label_pairs") != sum(boolean(pair["different_labels"]) for pair in pairs)):
        fail("The pair CSV does not match the counts in split_report.json.")
    return rows, report, pairs, snapshots


def validate_pairs(rows, pairs):
    lookup = {row["path"]: row for row in rows}
    seen_paths, seen_ids = set(), set()
    for pair in pairs:
        try:
            pair_number, distance = int(pair["pair_id"]), int(pair["hamming_distance"])
        except (ValueError, TypeError):
            fail("Invalid pair ID or Hamming distance in pair CSV.")
        if pair_number < 1 or pair_number in seen_ids or not 0 <= distance <= 4:
            fail("Repeated/invalid pair ID or unsupported Hamming distance.")
        seen_ids.add(pair_number)
        key = tuple(sorted((pair["path_a"], pair["path_b"])))
        if key[0] == key[1] or key in seen_paths:
            fail("Repeated pair or self-pair in CSV: " + str(key))
        seen_paths.add(key)
        for side in ("a", "b"):
            path = pair["path_" + side]
            row = lookup.get(path)
            if row is None:
                fail("Pair references a file outside the manifest: " + path)
            if (pair["label_" + side] != core.label(row)
                    or pair["split_" + side] != row["split"]
                    or pair["group_" + side] != row["group_id"]):
                fail("Pair metadata does not match the manifest: " + path)
        if pair["group_a"] == pair["group_b"]:
            fail("Input pair is already inside one group; it is not an unchanged V2 audit.")
        if boolean(pair["cross_split"]) != (pair["split_a"] != pair["split_b"]):
            fail("Incorrect cross_split flag in pair CSV.")
        if boolean(pair["different_labels"]) != (pair["label_a"] != pair["label_b"]):
            fail("Incorrect different_labels flag in pair CSV.")


def verify_pixels_and_candidates(rows, pairs, root):
    infos = [core.filename_info(row) for row in rows]
    if any(info["kind"] != "candidate_original" for info in infos):
        fail("Input contains augmented/unmapped filenames. Use the manifest produced by step 2.")
    facts = core.inspect_images(rows, infos, root)
    pixels, families = {}, {}
    for i, row in enumerate(rows):
        digest = facts[i]["pixel_sha256"]
        if digest in pixels:
            previous = rows[pixels[digest]]
            if core.label(previous) != core.label(row):
                fail("Same pixels have conflicting labels: " + previous["path"] + " | " + row["path"])
            fail("Exact pixel copies remain. Rerun step 2 before this step: " + row["path"])
        pixels[digest] = i
        family = infos[i]["family"]
        if family in families and families[family] != row["group_id"]:
            fail("A filename family was separated in the input; rerun step 2.")
        families[family] = row["group_id"]
    current = {
        "kept": list(range(len(rows))),
        "group_ids": {i: row["group_id"] for i, row in enumerate(rows)},
        "assignments": {row["group_id"]: row["split"] for row in rows},
    }
    print("[AUDIT] Rechecking candidate pairs on the current image files...", flush=True)
    recomputed, complete = core.find_candidates(rows, facts, current)
    if not complete:
        fail("Near-duplicate audit exceeded the pair limit; no new split was published.")
    def signature(pair):
        return (*sorted((pair["path_a"], pair["path_b"])), int(pair["hamming_distance"]))
    if {signature(pair) for pair in pairs} != {signature(pair) for pair in recomputed}:
        fail("The current image audit differs from near_duplicates.csv. Rerun step 2 into a new directory.")
    return facts


def coassign(rows, pairs, seed=42):
    """Preserve existing groups, then add every candidate link transitively."""
    union = core.UnionFind(len(rows))
    path_index = {row["path"]: i for i, row in enumerate(rows)}
    owner = {}
    for i, row in enumerate(rows):
        gid = row["group_id"]
        if gid in owner:
            union.union(i, owner[gid])
        else:
            owner[gid] = i
    for pair in pairs:
        union.union(path_index[pair["path_a"]], path_index[pair["path_b"]])
    components = defaultdict(list)
    for i in range(len(rows)):
        components[union.find(i)].append(i)
    new_groups, mixed_groups = {}, {}
    for ids in components.values():
        fingerprint = "\n".join(sorted(core.norm(rows[i]["path"]) for i in ids))
        gid = "v3_" + core.sha_bytes(fingerprint.encode("utf-8"))[:32]
        labels = sorted({core.label(rows[i]) for i in ids})
        for i in ids:
            new_groups[i] = gid
        if len(labels) > 1:
            mixed_groups[gid] = {"labels": labels, "images": len(ids), "assigned_split": "train"}
    # Preserve both labels in mixed groups. Keeping these groups in train means
    # unresolved specimen provenance does not enter validation/test scoring.
    pure = [i for i in range(len(rows)) if new_groups[i] not in mixed_groups]
    assignments = core.assign_splits(rows, pure, new_groups, seed)
    assignments.update({gid: "train" for gid in mixed_groups})
    output = [dict(row, group_id=new_groups[i], split=assignments[new_groups[i]]) for i, row in enumerate(rows)]
    output.sort(key=lambda row: core.norm(row["path"]))
    lookup = {row["path"]: row for row in output}
    old_to_new = defaultdict(set)
    new_group_splits = defaultdict(set)
    for row in rows:
        new_row = lookup[row["path"]]
        if any(row[key] != new_row[key] for key in core.COLUMNS if key not in ("group_id", "split")):
            fail("An image or label was unexpectedly changed.")
        old_to_new[row["group_id"]].add(new_row["group_id"])
        new_group_splits[new_row["group_id"]].add(new_row["split"])
    if any(len(groups) != 1 for groups in old_to_new.values()):
        fail("An existing group was broken apart.")
    if any(len(splits) != 1 for splits in new_group_splits.values()):
        fail("A new group spans more than one split.")
    for pair in pairs:
        a, b = lookup[pair["path_a"]], lookup[pair["path_b"]]
        if a["group_id"] != b["group_id"] or a["split"] != b["split"]:
            fail("A candidate pair still spans different groups/splits.")
    return output, mixed_groups


def visual_observations(rows, pairs, notes):
    lookup = {row["path"]: row for row in rows}
    by_paths = {tuple(sorted((pair["path_a"], pair["path_b"]))): str(pair["pair_id"]) for pair in pairs}
    matched = {}
    for note in notes.get("pairs", []):
        a, b = note["path_a"], note["path_b"]
        pair_id = by_paths.get(tuple(sorted((a, b))))
        if (pair_id is not None and lookup[a]["sha256"] == note.get("sha256_a")
                and lookup[b]["sha256"] == note.get("sha256_b")):
            matched[pair_id] = note["observation"]
    return matched


def build_report(rows, output, pairs, facts, mixed, observations, snapshots, seed):
    counts = Counter((core.label(row), row["split"]) for row in output)
    missing = [f"{fruit}::{status}::{split}" for fruit in core.FRUITS for status in core.STATUSES for split in core.SPLITS
               if not counts[(f"{fruit}::{status}", split)]]
    old_groups = len({row["group_id"] for row in rows})
    new_groups = len({row["group_id"] for row in output})
    lookup = {row["path"]: row for row in output}
    prior_groups_by_new = defaultdict(set)
    members = Counter(row["group_id"] for row in output)
    for row in rows:
        prior_groups_by_new[lookup[row["path"]]["group_id"]].add(row["group_id"])
    merges = [groups for groups in prior_groups_by_new.values() if len(groups) > 1]
    return {
        "version": VERSION,
        "operation": "conservative_group_coassignment_before_new_cnn_experiment",
        "input_fingerprints": {name: core.sha_bytes(data) for name, data in snapshots.items()},
        "script_sha256": core.sha_bytes(Path(__file__).read_bytes()),
        "helper_sha256": core.sha_bytes(Path(core.__file__).read_bytes()),
        "seed": seed,
        "split_policy": "60/20/20 by group within each pure label; mixed-label components kept in train",
        "input_records": len(rows), "output_records": len(output),
        "images_removed": 0, "labels_changed": 0,
        "local_image_files_verified": len(facts),
        "old_groups": old_groups, "new_groups": new_groups,
        "group_count_reduction": old_groups - new_groups,
        "merged_components": len(merges),
        "largest_component_old_groups": max((len(groups) for groups in merges), default=1),
        "largest_group_images": max(members.values(), default=0),
        "split_counts": dict(Counter(row["split"] for row in output)),
        "label_split_counts": {
            f"{fruit}::{status}": {split: counts[(f"{fruit}::{status}", split)] for split in core.SPLITS}
            for fruit in core.FRUITS for status in core.STATUSES
        },
        "groups_by_split": {split: len({row["group_id"] for row in output if row["split"] == split}) for split in core.SPLITS},
        "missing_label_split_buckets": missing,
        "pair_handling": {
            "total_candidate_pairs": len(pairs),
            "cross_split_pairs_before": sum(boolean(pair["cross_split"]) for pair in pairs),
            "cross_split_pairs_after": sum(lookup[pair["path_a"]]["split"] != lookup[pair["path_b"]]["split"] for pair in pairs),
            "different_label_pairs": sum(boolean(pair["different_labels"]) for pair in pairs),
            "visually_inspected_contact_sheet_pairs": len(observations),
            "unreviewed_pairs_grouped_as_precaution": len(pairs) - len(observations),
            "all_candidates_recomputed_from_image_files": len(facts) == len(rows),
            "all_candidate_links_inside_one_group": all(lookup[pair["path_a"]]["group_id"] == lookup[pair["path_b"]]["group_id"] for pair in pairs),
            "automatic_duplicate_identity_confirmation": False,
            "automatic_label_changes": False,
        },
        "mixed_label_groups_in_train": mixed,
        "ready_for_pilot_training": not missing and len(facts) == len(rows),
        "physical_specimen_independence_confirmed": False,
        "all_pairs_visually_reviewed": len(observations) == len(pairs),
        "accuracy_measured": False,
        "limitations": [
            "Conservative grouping contains the detected links; it does not prove that every group is a unique real fruit.",
            "Only eight pairs were supplied as a contact sheet for visual inspection; other pairs are grouped as a precaution.",
            "Perceptual hashes can miss larger rotations, crops and other views of the same physical fruit.",
            "Coassigning visually similar but unrelated images can reduce the effective number of independent groups.",
            "Keep this split fixed during model development; use a separate labelled camera-photo test for real-world performance.",
        ],
    }


def publish(directory, rows, output, pairs, report, observations, snapshots):
    directory = Path(directory).resolve()
    if directory.exists():
        fail("Output directory already exists. Choose a NEW --output-dir: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".freshlens_step2b_", dir=directory.parent))
    try:
        core.write_csv(temporary / "manifest.csv", output, core.COLUMNS)
        lookup = {row["path"]: row for row in output}
        old_members = defaultdict(list)
        for row in rows:
            old_members[row["group_id"]].append(row)
        mapping = []
        for gid, members in sorted(old_members.items()):
            new = lookup[members[0]["path"]]
            mapping.append({"old_group_id": gid, "old_split": members[0]["split"], "images": len(members),
                            "new_group_id": new["group_id"], "new_split": new["split"]})
        core.write_csv(temporary / "group_mapping.csv", mapping,
                       ("old_group_id", "old_split", "images", "new_group_id", "new_split"))
        actions = []
        for pair in pairs:
            a, b = lookup[pair["path_a"]], lookup[pair["path_b"]]
            actions.append({**pair, "new_group": a["group_id"], "new_split_a": a["split"], "new_split_b": b["split"],
                            "visually_inspected": str(pair["pair_id"]) in observations,
                            "visual_observation": observations.get(str(pair["pair_id"]), "not visually inspected"),
                            "action": "coassign_group_keep_both_images_and_original_labels"})
        core.write_csv(temporary / "pair_actions.csv", actions,
                       (*PAIR_COLUMNS, "new_group", "new_split_a", "new_split_b", "visually_inspected", "visual_observation", "action"))
        report["output_manifest_sha256"] = core.sha_bytes((temporary / "manifest.csv").read_bytes())
        (temporary / "final_split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lock = {
            "version": VERSION, "manifest_sha256": report["output_manifest_sha256"],
            "report_sha256": core.sha_bytes((temporary / "final_split_report.json").read_bytes()),
            "records": len(output), "seed": report["seed"],
            "ready_for_pilot_training": report["ready_for_pilot_training"],
            "instruction": "Use this fixed manifest for the new CNN experiment. Do not resplit to improve reported accuracy.",
        }
        (temporary / "dataset_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "source_split_report.json").write_bytes(snapshots["split_report.json"])
        if directory.exists():
            fail("Output directory appeared during processing; nothing overwritten.")
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(input_dir, root, output_dir, seed=42, visual_path=None):
    input_dir, root, output_dir = Path(input_dir), Path(root), Path(output_dir)
    if output_dir.exists():
        fail("Output directory already exists. Choose a NEW --output-dir: " + str(output_dir))
    if not root.is_dir():
        fail("Dataset root does not exist: " + str(root))
    rows, source_report, pairs, snapshots = load_inputs(input_dir)
    print(f"[INPUT] {len(rows)} images; {len(pairs)} candidate pairs", flush=True)
    facts = verify_pixels_and_candidates(rows, pairs, root)
    output, mixed = coassign(rows, pairs, seed)
    notes_path = Path(visual_path) if visual_path else Path(__file__).with_name("VISUAL_REVIEW.json")
    notes = json.loads(notes_path.read_text(encoding="utf-8-sig")) if notes_path.is_file() else {"pairs": []}
    observations = visual_observations(rows, pairs, notes)
    report = build_report(rows, output, pairs, facts, mixed, observations, snapshots, seed)
    for name, original in snapshots.items():
        if (input_dir / name).read_bytes() != original:
            fail("An input file changed during this run: " + name)
    publish(output_dir, rows, output, pairs, report, observations, snapshots)
    print(f"[OK] Saved: {output_dir.resolve()}", flush=True)
    print(f"[IMAGES] {len(output)}; labels changed: 0; images removed: 0", flush=True)
    print(f"[GROUPS] {report['old_groups']} -> {report['new_groups']}", flush=True)
    print("[SPLITS] " + json.dumps(report["split_counts"]), flush=True)
    print(f"[CROSS SPLIT PAIRS] {report['pair_handling']['cross_split_pairs_before']} -> {report['pair_handling']['cross_split_pairs_after']}", flush=True)
    status = "READY_FOR_PILOT_TRAINING" if report["ready_for_pilot_training"] else "REVIEW_REQUIRED"
    print(f"[{status}] Send final_split_report.json for the CNN step.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="FreshLens step 2B: group all detected candidate links, keep labels")
    parser.add_argument("--input-dir", type=Path, default=Path("data/cnn_dataset_v2"))
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory containing raw/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cnn_dataset_v3"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run(args.input_dir, args.root, args.output_dir, args.seed)
    except (core.DataError, OSError, json.JSONDecodeError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
