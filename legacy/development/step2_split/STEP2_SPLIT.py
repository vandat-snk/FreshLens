"""Build a versioned FreshLens split and a review report. Never writes to MongoDB.

The original manifest and image files are read only. Filename rules are tailored
to the supplied FreshLens inventory; a filename without an augmentation marker
is only a *candidate* original, not proof of an independent photograph.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
import textwrap
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath


VERSION = "2.0.0"
COLUMNS = ("path", "fruit", "status", "source", "group_id", "sha256", "split")
FRUITS = ("apple", "banana", "orange", "tomato")
STATUSES = ("fresh", "rotten")
SPLITS = ("train", "val", "test")
SCREEN = re.compile(
    r"^(?P<prefix>(?:(?:rotated_by_-?\d+(?:\.\d+)?|saltandpepper|"
    r"translation|vertical_flip|horizontal_flip)_)*)(?P<base>Screen\s+Shot\s+.+)$",
    re.IGNORECASE,
)
NUMBERED = re.compile(r"^(fresh|rotten)(apple|banana|orange|tomato)\s*\(\d+\)$", re.IGNORECASE)
UNKNOWN_AUG = re.compile(r"^Banana__Healthy_augmented_\d+$", re.IGNORECASE)


class DataError(ValueError):
    """Invalid input or an unresolved exact-content label conflict."""


def norm(text):
    return unicodedata.normalize("NFC", " ".join(text.casefold().split()))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def label(row):
    return row["fruit"] + "::" + row["status"]


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise DataError("Manifest must contain: " + ", ".join(COLUMNS))
        rows = []
        seen = set()
        for line, raw in enumerate(reader, 2):
            row = {key: (raw.get(key) or "").strip() for key in COLUMNS}
            name = row["path"].replace("\\", "/")
            pure = PurePosixPath(name)
            if (not name or pure.is_absolute() or PureWindowsPath(name).drive
                    or ".." in pure.parts or ":" in name or "\x00" in name):
                raise DataError(f"Unsafe relative path at line {line}: {name!r}")
            row["path"] = pure.as_posix()
            key = norm(row["path"])
            if key in seen:
                raise DataError(f"Repeated Windows path at line {line}: {name}")
            seen.add(key)
            if row["fruit"] not in FRUITS or row["status"] not in STATUSES:
                raise DataError(f"Unsupported label at line {line}: {label(row)}")
            if not row["group_id"] or row["split"] not in SPLITS:
                raise DataError(f"Missing group or invalid split at line {line}")
            row["sha256"] = row["sha256"].lower()
            if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise DataError(f"Missing/invalid SHA-256 at line {line}: {name}")
            rows.append(row)
    if not rows:
        raise DataError("Manifest is empty.")
    return sorted(rows, key=lambda row: norm(row["path"]))


def filename_info(row):
    stem = PurePosixPath(row["path"]).stem
    match = SCREEN.fullmatch(stem)
    if match:
        augmented = bool(match["prefix"])
        return {
            "family": "screen::" + norm(match["base"]),
            "kind": "offline_augmentation" if augmented else "candidate_original",
            "reason": "offline_augmentation_known_parent" if augmented else "",
        }
    if UNKNOWN_AUG.fullmatch(stem):
        return {"family": "", "kind": "offline_augmentation_unknown_parent",
                "reason": "offline_augmentation_parent_unknown"}
    match = NUMBERED.fullmatch(stem)
    if match:
        if match[1].lower() != row["status"] or match[2].lower() != row["fruit"]:
            raise DataError("Filename and metadata labels disagree: " + row["path"])
        # Ignore extension/case conservatively, but do NOT infer that neighbouring
        # numbers are independent fruit specimens or automatically group them.
        return {"family": "numbered::" + norm(stem),
                "kind": "candidate_original", "reason": ""}
    return {"family": "", "kind": "unrecognized_filename",
            "reason": "unrecognized_filename_needs_source_mapping"}


def audit_filenames(rows, infos):
    families = defaultdict(list)
    for i, info in enumerate(infos):
        if info["family"].startswith("screen::"):
            families[info["family"]].append(i)
    leaked = {key: ids for key, ids in families.items()
              if len({rows[i]["split"] for i in ids}) > 1}
    return {
        "evidence": "filename rules only; augmented image pixels not inspected here",
        "input_records": len(rows),
        "input_split_counts": dict(Counter(row["split"] for row in rows)),
        "filename_kind_counts": dict(Counter(info["kind"] for info in infos)),
        "screen_families": len(families),
        "screen_families_across_old_splits": len(leaked),
        "rows_in_these_families": sum(map(len, leaked.values())),
        "screen_families_without_named_original": sum(
            not any(infos[i]["kind"] == "candidate_original" for i in ids)
            for ids in families.values()
        ),
        "candidate_originals_before_pixel_check": sum(
            info["kind"] == "candidate_original" for info in infos
        ),
        "candidate_counts_by_label": dict(sorted(Counter(
            label(row) for row, info in zip(rows, infos)
            if info["kind"] == "candidate_original"
        ).items())),
        "examples": [
            {"family": key, "files": [
                {"path": rows[i]["path"], "old_split": rows[i]["split"]} for i in ids
            ]} for key, ids in sorted(leaked.items())[:3]
        ],
    }


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_groups(rows, infos, facts):
    """Preserve old group links and add filename / exact-content links."""
    union = UnionFind(len(rows))
    owners = {}
    content_labels = {}
    for i, (row, info) in enumerate(zip(rows, infos)):
        keys = [("declared", norm(row["group_id"])), ("sha256", row["sha256"])]
        if info["family"]:
            keys.append(("filename", info["family"]))
        if i in facts:
            keys.append(("pixels", facts[i]["pixel_sha256"]))
        for key in keys:
            if key[0] in ("sha256", "pixels"):
                if key in content_labels and content_labels[key] != label(row):
                    other = rows[owners[key]]["path"]
                    raise DataError(f"Same content has conflicting labels: {other} | {row['path']}")
                content_labels[key] = label(row)
            if key in owners:
                union.union(i, owners[key])
            else:
                owners[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[union.find(i)].append(i)
    output = {}
    for ids in components.values():
        fingerprint = "\n".join(sorted(norm(rows[i]["path"]) for i in ids))
        gid = "v2_" + sha_bytes(fingerprint.encode("utf-8"))[:32]
        for i in ids:
            output[i] = gid
    return output


def safe_image_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DataError("Image resolves outside dataset root: " + relative)
    if not path.is_file():
        raise DataError("Image file not found: " + str(path))
    return path


def rgb_image(image):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def phash63(image):
    """63 non-DC low-frequency DCT bits; a review heuristic, not identity."""
    import numpy as np
    from PIL import Image

    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coordinates = np.arange(32, dtype=np.float64)
    frequencies = np.arange(8, dtype=np.float64)[:, None]
    basis = math.sqrt(2 / 32) * np.cos(math.pi * (2 * coordinates + 1) * frequencies / 64)
    basis[0] /= math.sqrt(2)
    coefficients = (basis @ gray @ basis.T).round(6).ravel()[1:]
    median = np.median(coefficients)
    value = 0
    for bit in coefficients > median:
        value = (value << 1) | int(bit)
    return value


def inspect_images(rows, infos, root):
    from PIL import Image

    indices = [i for i, info in enumerate(infos) if info["kind"] == "candidate_original"]
    facts = {}
    for position, i in enumerate(indices, 1):
        row = rows[i]
        path = safe_image_path(root, row["path"])
        data = path.read_bytes()
        if sha_bytes(data) != row["sha256"]:
            raise DataError("File changed since manifest export (SHA-256 mismatch): " + row["path"])
        try:
            with Image.open(io.BytesIO(data)) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise DataError("Multi-frame image is not supported: " + row["path"])
                image = rgb_image(opened)
                image.load()
                digest = hashlib.sha256()
                digest.update(f"RGB:{image.width}:{image.height}:".encode("ascii"))
                digest.update(image.tobytes())
                facts[i] = {
                    "pixel_sha256": digest.hexdigest(),
                    "phash63": phash63(image),
                    "width": image.width,
                    "height": image.height,
                }
        except DataError:
            raise
        except Exception as exc:
            raise DataError(f"Cannot decode {row['path']}: {type(exc).__name__}: {exc}") from exc
        if position % 250 == 0 or position == len(indices):
            print(f"[CHECK] {position}/{len(indices)} candidate images", flush=True)
    return facts


def assign_splits(rows, kept, group_ids, seed):
    groups = defaultdict(list)
    for i in kept:
        groups[group_ids[i]].append(i)
    buckets = defaultdict(list)
    for gid, ids in groups.items():
        bucket = "|".join(sorted({label(rows[i]) for i in ids}))
        buckets[bucket].append(gid)
    assignments = {}
    for bucket, gids in sorted(buckets.items()):
        gids = sorted(gids, key=lambda gid: sha_bytes(f"{seed}:{bucket}:{gid}".encode("utf-8")))
        count = len(gids)
        if count >= 3:
            n_train = min(max(round(count * .6), 1), count - 2)
            n_val = min(max(round(count * .2), 1), count - n_train - 1)
        elif count == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0
        for j, gid in enumerate(gids):
            assignments[gid] = "train" if j < n_train else "val" if j < n_train + n_val else "test"
    return assignments


def prepare_split(rows, infos, facts, seed=42):
    expected = {i for i, info in enumerate(infos) if info["kind"] == "candidate_original"}
    if set(facts) != expected:
        raise DataError("Every candidate image must pass the local image check before splitting.")
    group_ids = build_groups(rows, infos, facts)
    reasons = {i: info["reason"] for i, info in enumerate(infos) if info["reason"]}
    pixel_owners = {}
    kept = []
    duplicate_of = {}
    for i in sorted(expected, key=lambda j: norm(rows[j]["path"])):
        digest = facts[i]["pixel_sha256"]
        if digest in pixel_owners:
            reasons[i] = "exact_pixel_duplicate"
            duplicate_of[i] = rows[pixel_owners[digest]]["path"]
        else:
            pixel_owners[digest] = i
            kept.append(i)
    assignments = assign_splits(rows, kept, group_ids, seed)
    output = [dict(rows[i], group_id=group_ids[i], split=assignments[group_ids[i]]) for i in kept]
    # Check postconditions independently by each identity key.
    overlaps = {}
    for key in ("group", "filename", "bytes", "pixels"):
        seen = defaultdict(set)
        for i in kept:
            value = {"group": group_ids[i], "filename": infos[i]["family"],
                     "bytes": rows[i]["sha256"], "pixels": facts[i]["pixel_sha256"]}[key]
            if value:
                seen[value].add(assignments[group_ids[i]])
        overlaps[key] = sum(len(splits) > 1 for splits in seen.values())
    if any(overlaps.values()):
        raise DataError("Internal split check failed: " + str(overlaps))
    return {"manifest": output, "kept": kept, "group_ids": group_ids,
            "assignments": assignments, "reasons": reasons,
            "duplicate_of": duplicate_of, "overlaps": overlaps}


class HashTree:
    """BK-tree with Hamming distance; queries return all matching hash values."""
    def __init__(self):
        self.root = None

    def add(self, value):
        if self.root is None:
            self.root = [value, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = [value, {}]
                return
            node = node[1][distance]

    def query(self, value, threshold):
        stack = [self.root] if self.root is not None else []
        matches = []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                matches.append((node[0], distance))
            stack.extend(child for edge, child in node[1].items()
                         if distance - threshold <= edge <= distance + threshold)
        return sorted(matches)


def find_candidates(rows, facts, prepared, threshold=4, max_pairs=20000):
    tree = HashTree()
    seen = defaultdict(list)
    candidates = []
    for i in prepared["kept"]:
        gid_i = prepared["group_ids"][i]
        split_i = prepared["assignments"][gid_i]
        value = facts[i]["phash63"]
        for neighbour, distance in tree.query(value, threshold):
            for j in seen[neighbour]:
                gid_j = prepared["group_ids"][j]
                if gid_j == gid_i:
                    continue
                if len(candidates) >= max_pairs:
                    return sort_candidates(candidates), False
                split_j = prepared["assignments"][gid_j]
                candidates.append({
                    "path_a": rows[j]["path"], "path_b": rows[i]["path"],
                    "label_a": label(rows[j]), "label_b": label(rows[i]),
                    "split_a": split_j, "split_b": split_i,
                    "group_a": gid_j, "group_b": gid_i,
                    "hamming_distance": distance,
                    "cross_split": split_j != split_i,
                    "different_labels": label(rows[j]) != label(rows[i]),
                    "decision": "unreviewed",
                })
        tree.add(value)
        seen[value].append(i)
    return sort_candidates(candidates), True


def sort_candidates(pairs):
    return sorted(pairs, key=lambda pair: (
        not pair["different_labels"], not pair["cross_split"],
        pair["hamming_distance"], pair["path_a"], pair["path_b"],
    ))


def write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contact_sheets(root, directory, pairs, limit=48):
    """Small sheets for review only; no training image is changed."""
    from PIL import Image, ImageDraw, ImageFont

    selected = [(i, pair) for i, pair in enumerate(pairs, 1)
                if pair["cross_split"] or pair["different_labels"]][:limit]
    if not selected:
        return []
    directory.mkdir()
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    pages = []
    for page_start in range(0, len(selected), 8):
        subset = selected[page_start:page_start + 8]
        canvas = Image.new("RGB", (620 * min(2, len(subset)), math.ceil(len(subset) / 2) * 290), "white")
        draw = ImageDraw.Draw(canvas)
        for position, (number, pair) in enumerate(subset):
            left, top = (position % 2) * 620, (position // 2) * 290
            draw.rectangle((left, top, left + 619, top + 289), outline="#cccccc")
            title = f"Pair {number} | distance={pair['hamming_distance']} | {pair['split_a']} / {pair['split_b']}"
            draw.text((left + 8, top + 7), title, fill="black", font=font)
            for side, key in enumerate(("a", "b")):
                x = left + 8 + 305 * side
                with Image.open(safe_image_path(root, pair[f"path_{key}"])) as opened:
                    thumbnail = rgb_image(opened)
                    thumbnail.thumbnail((290, 175), Image.Resampling.LANCZOS)
                    canvas.paste(thumbnail, (x + (290 - thumbnail.width) // 2, top + 30))
                draw.text((x, top + 208), pair[f"label_{key}"], fill="black", font=font)
                name = PurePosixPath(pair[f"path_{key}"]).name
                for line_number, line in enumerate(textwrap.wrap(name, width=39)[:3]):
                    draw.text((x, top + 226 + 17 * line_number), line, fill="black", font=font)
        filename = f"pairs_{page_start // 8 + 1:02d}.jpg"
        canvas.save(directory / filename, quality=90)
        pages.append("review_images/" + filename)
    return pages


def build_report(rows, infos, facts, prepared, pairs, complete, seed, fingerprint):
    counts = Counter((label(row), row["split"]) for row in prepared["manifest"])
    missing = [f"{fruit}::{status}::{split}" for fruit in FRUITS for status in STATUSES for split in SPLITS
               if counts[(f"{fruit}::{status}", split)] == 0]
    mixed = sum(pair["different_labels"] for pair in pairs)
    across = sum(pair["cross_split"] for pair in pairs)
    group_labels = defaultdict(set)
    for row in prepared["manifest"]:
        group_labels[row["group_id"]].add(label(row))
    mixed_groups = {group: sorted(labels) for group, labels in group_labels.items() if len(labels) > 1}
    needs_review = not complete or bool(across or mixed or mixed_groups or missing)
    return {
        "version": VERSION,
        "input_manifest_sha256": fingerprint,
        "script_sha256": sha_bytes(Path(__file__).read_bytes()),
        "seed": seed,
        "target_ratios_by_group": {"train": .6, "val": .2, "test": .2},
        "filename_audit_before": audit_filenames(rows, infos),
        "local_image_files_checked": len(facts),
        "output_records": len(prepared["manifest"]),
        "excluded_records": len(prepared["reasons"]),
        "exclusion_counts": dict(sorted(Counter(prepared["reasons"].values()).items())),
        "output_split_counts": dict(Counter(row["split"] for row in prepared["manifest"])),
        "output_label_split_counts": {
            f"{fruit}::{status}": {split: counts[(f"{fruit}::{status}", split)] for split in SPLITS}
            for fruit in FRUITS for status in STATUSES
        },
        "unique_output_groups": len(group_labels),
        "cross_split_overlap_counts": prepared["overlaps"],
        "mixed_label_groups": mixed_groups,
        "missing_label_split_buckets": missing,
        "near_duplicate_review": {
            "method": "63 non-DC DCT bits, Hamming distance <= 4; candidates only",
            "complete": complete, "candidate_pairs": len(pairs),
            "cross_split_pairs": across, "different_label_pairs": mixed,
            "automatic_perceptual_merging": False,
            "maximum_pairs": 20000,
        },
        "review_required": needs_review,
        "ready_for_pilot_training": not needs_review,
        "evaluation_independence_confirmed": False,
        "limitations": [
            "No augmentation marker is not proof of an original/independent photograph.",
            "Image transformations marked in filenames are excluded from this experiment, not deleted.",
            "Perceptual hashes can miss crops/rotations/renames and can flag different fruits that look similar.",
            "Physical-fruit, capture-session and source-dataset grouping needs provenance or human review.",
            "A separately collected camera-photo test is needed to assess real upload performance.",
            "This run is dataset preparation, not training and not an accuracy measurement.",
        ],
    }


def publish(root, output_dir, rows, infos, facts, prepared, pairs, report):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise DataError("Output directory already exists; use a NEW --output-dir: " + str(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".freshlens_step2_", dir=output_dir.parent))
    try:
        write_csv(temporary / "manifest.csv", prepared["manifest"], COLUMNS)
        lineage = []
        for i, (row, info) in enumerate(zip(rows, infos)):
            gid = prepared["group_ids"][i]
            lineage.append({
                **row, "old_group_id": row["group_id"], "old_split": row["split"],
                "group_id": gid, "split": prepared["assignments"].get(gid, ""),
                "filename_family": info["family"], "filename_kind": info["kind"],
                "included": i not in prepared["reasons"],
                "exclusion_reason": prepared["reasons"].get(i, ""),
                "duplicate_of": prepared["duplicate_of"].get(i, ""),
            })
        lineage_columns = (*COLUMNS, "old_group_id", "old_split", "filename_family", "filename_kind",
                           "included", "exclusion_reason", "duplicate_of")
        write_csv(temporary / "lineage.csv", lineage, lineage_columns)
        write_csv(temporary / "excluded.csv", (row for row in lineage if not row["included"]), lineage_columns)
        pair_columns = ("pair_id", "path_a", "path_b", "label_a", "label_b", "split_a", "split_b",
                        "group_a", "group_b", "hamming_distance", "cross_split", "different_labels", "decision")
        write_csv(temporary / "near_duplicates.csv", (
            dict(pair_id=i, **pair) for i, pair in enumerate(pairs, 1)), pair_columns)
        write_csv(temporary / "image_fingerprints.csv", (
            {"path": rows[i]["path"], "sha256": rows[i]["sha256"],
             **facts[i], "phash63": f"{facts[i]['phash63']:016x}"} for i in sorted(facts)
        ), ("path", "sha256", "pixel_sha256", "phash63", "width", "height"))
        report["review_images"] = contact_sheets(root, temporary / "review_images", pairs)
        report["review_images_pair_limit"] = 48
        report["output_manifest_sha256"] = sha_bytes((temporary / "manifest.csv").read_bytes())
        (temporary / "split_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise DataError("Output directory appeared during processing; nothing overwritten.")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(manifest, root, output_dir, seed=42):
    manifest, root, output_dir = Path(manifest), Path(root), Path(output_dir)
    if output_dir.exists():
        raise DataError("Output directory already exists; use a NEW --output-dir: " + str(output_dir))
    if not root.is_dir():
        raise DataError("Dataset root does not exist: " + str(root))
    fingerprint = sha_bytes(manifest.read_bytes())
    rows = read_manifest(manifest)
    infos = [filename_info(row) for row in rows]
    audit = audit_filenames(rows, infos)
    print(f"[INPUT] {len(rows)} records", flush=True)
    print(f"[OLD SPLIT] {audit['screen_families_across_old_splits']} filename families cross splits", flush=True)
    print(f"[CANDIDATES] {audit['candidate_originals_before_pixel_check']} image files to verify", flush=True)
    facts = inspect_images(rows, infos, root)
    prepared = prepare_split(rows, infos, facts, seed)
    print("[AUDIT] Finding perceptual-hash candidates for review...", flush=True)
    pairs, complete = find_candidates(rows, facts, prepared)
    if sha_bytes(manifest.read_bytes()) != fingerprint:
        raise DataError("Input manifest changed during processing. Run again with a stable input file.")
    report = build_report(rows, infos, facts, prepared, pairs, complete, seed, fingerprint)
    publish(root, output_dir, rows, infos, facts, prepared, pairs, report)
    print(f"[OK] Saved: {output_dir.resolve()}", flush=True)
    print("[SPLITS] " + json.dumps(report["output_split_counts"]), flush=True)
    print(f"[EXCLUDED] {report['excluded_records']} records (source images kept)", flush=True)
    print(f"[NEAR DUPLICATES] {len(pairs)} candidates; {report['near_duplicate_review']['cross_split_pairs']} cross splits", flush=True)
    status = "REVIEW_REQUIRED" if report["review_required"] else "READY_FOR_PILOT_TRAINING"
    print(f"[{status}] Send split_report.json and near_duplicates.csv before the CNN step.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="FreshLens step 2: audited split into a NEW directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory containing raw/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cnn_dataset_v2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run(args.manifest, args.root, args.output_dir, args.seed)
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}. This script requires Pillow and NumPy.", file=sys.stderr)
        return 1
    except (DataError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
