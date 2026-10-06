"""Locked dataset loading and shared RGB preprocessing for FreshLens CNN."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T


FRUITS = ("apple", "banana", "orange", "tomato")
STATUSES = ("fresh", "rotten")
CLASSES = tuple(f"{fruit}::{status}" for fruit in FRUITS for status in STATUSES)
SPLITS = ("train", "val", "test")
PREPROCESS = {
    "version": "rgb_exif_white_letterbox224_imagenet_v1",
    "image_size": 224, "fill_rgb": [255, 255, 255],
    "mean": [.485, .456, .406], "std": [.229, .224, .225],
}
PROJECT_DIR = Path(__file__).resolve().parents[3]


class DataError(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_locked_dataset(directory):
    """Inspect metadata for all splits; this function never opens image files."""
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.csv").read_bytes()
    report_bytes = (directory / "final_split_report.json").read_bytes()
    report = json.loads(report_bytes.decode("utf-8-sig"))
    lock = json.loads((directory / "dataset_lock.json").read_text(encoding="utf-8-sig"))
    fingerprint = sha(manifest_bytes)
    if (fingerprint != report.get("output_manifest_sha256")
            or fingerprint != lock.get("manifest_sha256")
            or sha(report_bytes) != lock.get("report_sha256")):
        raise DataError("Manifest/report/lock do not match. Use the unchanged cnn_dataset_v3 output.")
    if report.get("version") != "2.1.0" or not report.get("ready_for_pilot_training") or not lock.get("ready_for_pilot_training"):
        raise DataError("Dataset has not passed step 2B for pilot training.")
    pairs = report.get("pair_handling", {})
    if (pairs.get("cross_split_pairs_after") != 0
            or pairs.get("all_candidate_links_inside_one_group") is not True
            or pairs.get("all_candidates_recomputed_from_image_files") is not True):
        raise DataError("Unresolved candidate links in the data report.")
    reader = csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig")))
    columns = ("path", "fruit", "status", "source", "group_id", "sha256", "split")
    if not set(columns).issubset(reader.fieldnames or []):
        raise DataError("The manifest is missing required columns.")
    rows, seen_paths = [], set()
    groups, hashes = defaultdict(set), defaultdict(set)
    for raw in reader:
        row = {key: (raw.get(key) or "").strip() for key in columns}
        relative = row["path"].replace("\\", "/")
        pure = PurePosixPath(relative)
        if (not relative or pure.is_absolute() or PureWindowsPath(relative).drive
                or ".." in pure.parts or ":" in relative or "\x00" in relative):
            raise DataError("Unsafe image path: " + repr(relative))
        row["path"] = pure.as_posix()
        if row["path"].casefold() in seen_paths:
            raise DataError("Repeated image path: " + row["path"])
        seen_paths.add(row["path"].casefold())
        name = row["fruit"] + "::" + row["status"]
        if name not in CLASSES or row["split"] not in SPLITS or not row["group_id"]:
            raise DataError("Invalid label/split/group: " + row["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise DataError("Invalid image SHA-256: " + row["path"])
        row["target"] = CLASSES.index(name)
        groups[row["group_id"]].add(row["split"])
        hashes[row["sha256"]].add(row["split"])
        rows.append(row)
    if any(len(values) > 1 for values in groups.values()) or any(len(values) > 1 for values in hashes.values()):
        raise DataError("An image/group spans more than one split.")
    if len(rows) != report.get("output_records") or len(rows) != lock.get("records"):
        raise DataError("Dataset record counts do not match the lock/report.")
    if len(groups) != report.get("new_groups"):
        raise DataError("Dataset group count does not match the report.")
    if dict(Counter(row["split"] for row in rows)) != report.get("split_counts"):
        raise DataError("Dataset split counts do not match the report.")
    counts = Counter((CLASSES[row["target"]], row["split"]) for row in rows)
    expected = {name: {split: counts[(name, split)] for split in SPLITS} for name in CLASSES}
    if expected != report.get("label_split_counts") or any(not count for values in expected.values() for count in values.values()):
        raise DataError("Missing labels or report/manifest label counts disagree.")
    return rows, {
        "manifest_sha256": fingerprint, "report_sha256": sha(report_bytes),
        "split_counts": report["split_counts"], "new_groups": report["new_groups"],
        "physical_specimen_independence_confirmed": report.get("physical_specimen_independence_confirmed", False),
    }


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DataError("Image not found inside dataset root: " + relative)
    return path


def rgb_from_bytes(data):
    with Image.open(io.BytesIO(data)) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise DataError("Please use a single-frame photo.")
        image = ImageOps.exif_transpose(opened)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        image.load()
        return image


def read_record_image(root, row):
    data = safe_path(root, row["path"]).read_bytes()
    if sha(data) != row["sha256"]:
        raise DataError("Image changed since data preparation: " + row["path"])
    try:
        return rgb_from_bytes(data)
    except Exception as exc:
        raise DataError("Cannot decode image: " + row["path"]) from exc


def verify_images(root, selected_rows):
    for index, row in enumerate(selected_rows, 1):
        read_record_image(root, row)
        if index % 250 == 0 or index == len(selected_rows):
            print(f"[DATA CHECK] {index}/{len(selected_rows)} selected images", flush=True)


class Letterbox:
    def __call__(self, image):
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(255, 255, 255), centering=(.5, .5))


def image_transform(training=False):
    steps = [Letterbox()]
    if training:
        steps.extend([
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=15, translate=(.04, .04), scale=(.9, 1.02),
                           interpolation=T.InterpolationMode.BILINEAR, fill=(255, 255, 255)),
            T.ColorJitter(brightness=.12, contrast=.12, saturation=.08, hue=.01),
        ])
    steps.extend([T.ToTensor(), T.Normalize(PREPROCESS["mean"], PREPROCESS["std"])])
    return T.Compose(steps)


class FruitDataset(Dataset):
    def __init__(self, root, rows, training=False):
        self.root, self.rows = Path(root), list(rows)
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Hash and decode the same bytes on each read, including during resume.
        return self.transform(read_record_image(self.root, row)), row["target"], index
