r"""Download one locked FreshLens CNN dataset version from Cloudinary.

The manifest stays authoritative. The script:
1) reads the exact committed manifest,
2) asks MongoDB Atlas for the cloud URL of each SHA-256,
3) downloads missing files,
4) verifies exact SHA-256 after every download.

Usage:
    python scripts/sync_cnn_dataset.py ^
      --manifest "data\cnn_dataset_v3\manifest.csv" ^
      --version cnn_v1 ^
      --dest "dataset_cache\cnn_v1"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from freshlens_ai.storage import get_db, test_connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_COLUMNS = ("path", "fruit", "status", "source", "group_id", "sha256", "split")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(REQUIRED_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest thieu cot: {REQUIRED_COLUMNS}")
        rows = []
        for raw in reader:
            row = {k: (raw.get(k) or "").strip() for k in REQUIRED_COLUMNS}
            pure = PurePosixPath(row["path"].replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"Path khong an toan: {row['path']}")
            row["path"] = pure.as_posix()
            rows.append(row)
        return rows


def safe_dest(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path vuot khoi destination: {relative}")
    return path


def download_exact(url: str, dest: Path, expected_sha: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "FreshLens-Dataset-Sync/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as f:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                f.write(chunk)
        actual = sha256_file(tmp)
        if actual.lower() != expected_sha.lower():
            raise RuntimeError(
                f"SHA256 download khong khop. Expected={expected_sha}, actual={actual}, url={url}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "cnn_dataset_v3" / "manifest.csv")
    parser.add_argument("--version", default="cnn_v1")
    parser.add_argument("--dest", type=Path, default=PROJECT_ROOT / "dataset_cache" / "cnn_v1")
    parser.add_argument("--limit", type=int, default=0, help="Chi sync N anh dau de test")
    args = parser.parse_args()

    rows = load_manifest(args.manifest.resolve())
    selected = rows[: args.limit] if args.limit else rows
    dest_root = args.dest.resolve()

    if not test_connection():
        raise RuntimeError("MongoDB Atlas ket noi that bai.")

    db = get_db()
    images = db["dataset_images"]
    versions = db["dataset_versions"]

    version_doc = versions.find_one({"dataset_version": args.version})
    if not args.limit:
        if not version_doc or version_doc.get("status") != "published":
            raise RuntimeError(f"Dataset {args.version} chua duoc publish complete.")
        if int(version_doc.get("record_count", -1)) != len(rows):
            raise RuntimeError("So record trong Atlas khong khop manifest local.")

    downloaded = cached = 0
    for i, row in enumerate(selected, 1):
        dest = safe_dest(dest_root, row["path"])

        if dest.is_file() and sha256_file(dest).lower() == row["sha256"].lower():
            cached += 1
        else:
            doc = images.find_one(
                {"dataset_version": args.version, "sha256": row["sha256"]},
                {"_id": 0, "cloud_url": 1, "path": 1, "fruit": 1, "status": 1, "split": 1},
            )
            if not doc or not doc.get("cloud_url"):
                raise RuntimeError(f"Khong co cloud_url cho: {row['path']}")
            # Basic metadata consistency checks.
            for key in ("path", "fruit", "status", "split"):
                if str(doc.get(key, "")) != str(row.get(key, "")):
                    raise RuntimeError(f"Metadata Atlas khong khop manifest ({key}): {row['path']}")
            download_exact(str(doc["cloud_url"]), dest, row["sha256"])
            downloaded += 1

        if i % 100 == 0 or i == len(selected):
            print(f"[SYNC] {i}/{len(selected)} downloaded={downloaded} cached={cached}")

    print(f"[OK] Dataset ready: {dest_root}")
    print(f"[OK] downloaded={downloaded}; cached={cached}; total={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
