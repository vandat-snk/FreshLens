r"""Publish the locked FreshLens CNN dataset to Cloudinary + MongoDB Atlas.

Important:
- Reads the existing locked manifest (cnn_dataset_v3).
- Uploads ORIGINAL BYTES as Cloudinary RAW resources to preserve SHA-256.
- Stores metadata in MongoDB Atlas; images are not stored inside MongoDB.
- Safe to re-run: existing matching records are skipped.

Usage:
    python scripts/publish_cnn_dataset.py ^
      --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
      --manifest "data\cnn_dataset_v3\manifest.csv" ^
      --version cnn_v1

Test first:
    python scripts/publish_cnn_dataset.py ... --limit 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pymongo import ASCENDING

from freshlens_ai.storage import get_db, test_connection

try:
    import cloudinary
    import cloudinary.uploader
except ImportError as exc:
    raise SystemExit("Chua cai cloudinary. Chay: pip install -r requirements-db.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_COLUMNS = ("path", "fruit", "status", "source", "group_id", "sha256", "split")
VALID_SPLITS = {"train", "val", "test"}
VALID_FRUITS = {"apple", "banana", "orange", "tomato"}
VALID_STATUS = {"fresh", "rotten"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> tuple[list[dict[str, str]], bytes]:
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if not set(REQUIRED_COLUMNS).issubset(reader.fieldnames or []):
        raise ValueError(f"Manifest thieu cot. Can: {REQUIRED_COLUMNS}")
    rows = []
    for i, raw in enumerate(reader):
        row = {k: (raw.get(k) or "").strip() for k in REQUIRED_COLUMNS}
        if row["fruit"] not in VALID_FRUITS:
            raise ValueError(f"Dong {i+2}: fruit khong hop le: {row['fruit']}")
        if row["status"] not in VALID_STATUS:
            raise ValueError(f"Dong {i+2}: status khong hop le: {row['status']}")
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Dong {i+2}: split khong hop le: {row['split']}")
        if len(row["sha256"]) != 64:
            raise ValueError(f"Dong {i+2}: sha256 khong hop le")
        pure = PurePosixPath(row["path"].replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Dong {i+2}: path khong an toan: {row['path']}")
        row["path"] = pure.as_posix()
        row["manifest_index"] = i
        rows.append(row)
    return rows, payload


def safe_local_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path vuot ra khoi dataset root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay anh: {path}")
    return path


def configure_cloudinary() -> None:
    required = ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError("Thieu bien Cloudinary trong .env: " + ", ".join(missing))
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )


def upload_raw(path: Path, sha256: str, version: str) -> tuple[str, str]:
    """Upload exact bytes as RAW so a later download can match original SHA-256."""
    suffix = path.suffix.lower() or ".bin"
    public_id = f"{sha256}{suffix}"
    folder = f"freshlens_datasets/{version}"
    result = cloudinary.uploader.upload(
        str(path),
        resource_type="raw",
        folder=folder,
        public_id=public_id,
        overwrite=False,
        unique_filename=False,
        use_filename=False,
    )
    return str(result["secure_url"]), str(result["public_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Dataset root chua raw/")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "cnn_dataset_v3" / "manifest.csv")
    parser.add_argument("--version", default="cnn_v1")
    parser.add_argument("--limit", type=int, default=0, help="Chi publish N anh dau de test; 0 = tat ca")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = args.manifest.resolve()
    rows, manifest_bytes = load_manifest(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)

    if args.limit:
        rows_to_process = rows[: args.limit]
        print(f"[TEST MODE] Chi xu ly {len(rows_to_process)}/{len(rows)} anh.")
    else:
        rows_to_process = rows

    print(f"[DATASET] version={args.version}")
    print(f"[DATASET] records={len(rows)}")
    print(f"[DATASET] manifest_sha256={manifest_sha}")
    print(f"[ROOT] {root}")

    # Validate selected files/hashes before writing anything.
    for i, row in enumerate(rows_to_process, 1):
        path = safe_local_path(root, row["path"])
        actual = sha256_file(path)
        if actual.lower() != row["sha256"].lower():
            raise RuntimeError(f"SHA256 khong khop: {row['path']}")
        if i % 250 == 0 or i == len(rows_to_process):
            print(f"[VERIFY] {i}/{len(rows_to_process)}")

    if args.dry_run:
        print("[OK] Dry-run thanh cong. Khong ghi MongoDB/Cloudinary.")
        return 0

    configure_cloudinary()

    if not test_connection():
        raise RuntimeError("MongoDB Atlas ket noi that bai.")

    db = get_db()
    images = db["dataset_images"]
    versions = db["dataset_versions"]

    images.create_index(
        [("dataset_version", ASCENDING), ("sha256", ASCENDING)],
        unique=True,
        name="dataset_version_sha_unique",
    )
    images.create_index([("dataset_version", ASCENDING), ("manifest_index", ASCENDING)])
    images.create_index([("dataset_version", ASCENDING), ("split", ASCENDING)])
    images.create_index([("dataset_version", ASCENDING), ("fruit", ASCENDING), ("status", ASCENDING)])

    ok = skipped = 0
    for i, row in enumerate(rows_to_process, 1):
        path = safe_local_path(root, row["path"])
        query = {"dataset_version": args.version, "sha256": row["sha256"]}
        existing = images.find_one(query)

        if existing and existing.get("cloud_url"):
            # Prevent silently reusing metadata with a different label/path.
            for key in ("path", "fruit", "status", "split", "group_id"):
                if str(existing.get(key, "")) != str(row.get(key, "")):
                    raise RuntimeError(
                        f"MongoDB co cung SHA nhung metadata khac ({key}): {row['path']}"
                    )
            skipped += 1
        else:
            url, public_id = upload_raw(path, row["sha256"], args.version)
            doc = {
                "dataset_version": args.version,
                "manifest_index": int(row["manifest_index"]),
                "path": row["path"],
                "fruit": row["fruit"],
                "status": row["status"],
                "source": row["source"],
                "group_id": row["group_id"],
                "sha256": row["sha256"],
                "split": row["split"],
                "cloud_url": url,
                "cloud_public_id": public_id,
                "cloud_resource_type": "raw",
                "published_at": datetime.now(timezone.utc),
            }
            images.update_one(query, {"$set": doc}, upsert=True)
            ok += 1

        if i % 100 == 0 or i == len(rows_to_process):
            print(f"[PUBLISH] {i}/{len(rows_to_process)} new={ok} skipped={skipped}")

    # Only mark a dataset version complete when publishing ALL records.
    if not args.limit:
        split_counts = Counter(r["split"] for r in rows)
        label_counts = Counter(f"{r['fruit']}::{r['status']}" for r in rows)
        versions.update_one(
            {"dataset_version": args.version},
            {
                "$set": {
                    "dataset_version": args.version,
                    "manifest_sha256": manifest_sha,
                    "record_count": len(rows),
                    "split_counts": dict(split_counts),
                    "label_counts": dict(label_counts),
                    "status": "published",
                    "published_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
        print(f"[OK] Dataset {args.version} published: {len(rows)} images.")
    else:
        print("[OK] Test publish finished. Chua danh dau dataset version la complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
