r"""Publish the locked FreshLens CNN dataset to Cloudinary + MongoDB Atlas.

Key guarantees:
- The committed manifest is authoritative.
- ORIGINAL BYTES are uploaded as Cloudinary RAW resources.
- SHA-256 is the content identity. File-extension case (.JPG/.jpg) is not identity.
- Existing MongoDB/Cloudinary records are reused safely.
- Interrupted runs are resumable.
- --reconcile-only never uploads missing assets.
- A dataset version is marked published only after final MongoDB + Cloudinary checks pass.

Usage:
    python scripts/publish_cnn_dataset.py ^
      --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
      --manifest "data\cnn_dataset_v3\manifest.csv" ^
      --version cnn_v1

Dry-run:
    python scripts/publish_cnn_dataset.py ^
      --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
      --manifest "data\cnn_dataset_v3\manifest.csv" ^
      --version cnn_v1 ^
      --dry-run

Reconcile Cloudinary -> MongoDB only:
    python scripts/publish_cnn_dataset.py ^
      --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
      --manifest "data\cnn_dataset_v3\manifest.csv" ^
      --version cnn_v1 ^
      --reconcile-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pymongo import ASCENDING

from freshlens_ai.storage import get_db, test_connection

try:
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
except ImportError as exc:
    raise SystemExit(
        "Cloudinary dependency is missing. "
        "Run: pip install -r requirements-db.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_COLUMNS = (
    "path",
    "fruit",
    "status",
    "source",
    "group_id",
    "sha256",
    "split",
)

VALID_SPLITS = {"train", "val", "test"}
VALID_FRUITS = {"apple", "banana", "orange", "tomato"}
VALID_STATUS = {"fresh", "rotten"}


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for a local file."""
    h = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Calculate SHA-256 for bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_url(url: str) -> str:
    """Download one remote RAW asset and calculate its SHA-256."""
    h = hashlib.sha256()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FreshLens-Dataset-Publisher/2.2"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_manifest(path: Path) -> tuple[list[dict[str, str]], bytes]:
    """Load and validate the authoritative locked manifest."""
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())

    if not set(REQUIRED_COLUMNS).issubset(reader.fieldnames or []):
        raise ValueError(
            f"Manifest is missing required columns. Required: {REQUIRED_COLUMNS}"
        )

    rows: list[dict[str, str]] = []

    for i, raw in enumerate(reader):
        row = {
            key: (raw.get(key) or "").strip()
            for key in REQUIRED_COLUMNS
        }

        if row["fruit"] not in VALID_FRUITS:
            raise ValueError(
                f"Row {i + 2}: invalid fruit: {row['fruit']}"
            )

        if row["status"] not in VALID_STATUS:
            raise ValueError(
                f"Row {i + 2}: invalid status: {row['status']}"
            )

        if row["split"] not in VALID_SPLITS:
            raise ValueError(
                f"Row {i + 2}: invalid split: {row['split']}"
            )

        sha = row["sha256"].lower()

        if (
            len(sha) != 64
            or any(ch not in "0123456789abcdef" for ch in sha)
        ):
            raise ValueError(
                f"Row {i + 2}: invalid SHA-256."
            )

        row["sha256"] = sha

        pure = PurePosixPath(
            row["path"].replace("\\", "/")
        )

        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(
                f"Row {i + 2}: unsafe path: {row['path']}"
            )

        row["path"] = pure.as_posix()
        row["manifest_index"] = i
        rows.append(row)

    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError(
            "Manifest contains duplicate SHA-256 values."
        )

    return rows, payload


def safe_local_path(root: Path, relative: str) -> Path:
    """Resolve one manifest path safely below dataset root."""
    root = root.resolve()
    path = (root / relative).resolve()

    if not path.is_relative_to(root):
        raise ValueError(
            f"Path escapes dataset root: {relative}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return path


def configure_cloudinary() -> None:
    """Load Cloudinary credentials from environment."""
    required = (
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    )

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            "Missing Cloudinary environment variables: "
            + ", ".join(missing)
        )

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )


def upload_raw(
    path: Path,
    sha256: str,
    version: str,
) -> tuple[str, str]:
    """Upload exact local bytes as a Cloudinary RAW resource."""
    suffix = path.suffix.lower() or ".bin"

    result = cloudinary.uploader.upload(
        str(path),
        resource_type="raw",
        folder=f"freshlens_datasets/{version}",
        public_id=f"{sha256}{suffix}",
        overwrite=False,
        unique_filename=False,
        use_filename=False,
    )

    return (
        str(result["secure_url"]),
        str(result["public_id"]),
    )


def sha_from_public_id(public_id: str) -> str | None:
    """Extract SHA-256 identity from a FreshLens Cloudinary public_id."""
    name = PurePosixPath(public_id).name
    candidate = name.split(".", 1)[0].lower()

    if (
        len(candidate) == 64
        and all(ch in "0123456789abcdef" for ch in candidate)
    ):
        return candidate

    return None


def list_cloud_resources(version: str) -> dict[str, dict]:
    """Return one Cloudinary RAW resource per SHA-256."""
    prefix = f"freshlens_datasets/{version}/"

    resources_by_sha: dict[str, dict] = {}
    cursor = None

    while True:
        response = cloudinary.api.resources(
            resource_type="raw",
            type="upload",
            prefix=prefix,
            max_results=500,
            next_cursor=cursor,
        )

        for item in response.get("resources", []):
            public_id = str(item.get("public_id", ""))

            if not public_id:
                raise RuntimeError(
                    "Cloudinary returned a RAW resource without public_id."
                )

            sha = sha_from_public_id(public_id)

            if sha is None:
                raise RuntimeError(
                    "Cloudinary RAW asset does not use a valid "
                    f"SHA-based public_id: {public_id}"
                )

            if sha in resources_by_sha:
                previous = str(
                    resources_by_sha[sha].get("public_id", "")
                )

                raise RuntimeError(
                    "Multiple Cloudinary RAW assets have the same "
                    "SHA-256 identity. "
                    f"sha={sha}; first={previous}; second={public_id}"
                )

            resources_by_sha[sha] = item

        cursor = response.get("next_cursor")

        if not cursor:
            break

    return resources_by_sha


def validate_existing_metadata(
    existing: dict,
    row: dict,
) -> None:
    """Reject MongoDB records whose authoritative metadata differs."""
    for key in (
        "path",
        "fruit",
        "status",
        "split",
        "group_id",
    ):
        if str(existing.get(key, "")) != str(row.get(key, "")):
            raise RuntimeError(
                "MongoDB contains the same SHA with different metadata "
                f"({key}): {row['path']}"
            )


def mongo_document(
    row: dict,
    version: str,
    cloud_url: str,
    public_id: str,
) -> dict:
    """Build canonical MongoDB metadata for one image."""
    return {
        "dataset_version": version,
        "manifest_index": int(row["manifest_index"]),
        "path": row["path"],
        "fruit": row["fruit"],
        "status": row["status"],
        "source": row["source"],
        "group_id": row["group_id"],
        "sha256": row["sha256"],
        "split": row["split"],
        "cloud_url": cloud_url,
        "cloud_public_id": public_id,
        "cloud_resource_type": "raw",
        "published_at": datetime.now(timezone.utc),
    }


def create_indexes(images) -> None:
    """Create indexes required by dataset publishing."""
    images.create_index(
        [
            ("dataset_version", ASCENDING),
            ("sha256", ASCENDING),
        ],
        unique=True,
        name="dataset_version_sha_unique",
    )

    images.create_index(
        [
            ("dataset_version", ASCENDING),
            ("manifest_index", ASCENDING),
        ]
    )

    images.create_index(
        [
            ("dataset_version", ASCENDING),
            ("split", ASCENDING),
        ]
    )

    images.create_index(
        [
            ("dataset_version", ASCENDING),
            ("fruit", ASCENDING),
            ("status", ASCENDING),
        ]
    )


def verify_uploaded_asset(
    *,
    url: str,
    public_id: str,
    expected_sha: str,
    version: str,
    source_path: str,
) -> None:
    """Verify identity, folder, and exact remote bytes after an upload."""
    uploaded_sha = sha_from_public_id(public_id)

    if uploaded_sha != expected_sha:
        raise RuntimeError(
            "Uploaded Cloudinary asset has an unexpected SHA identity. "
            f"Expected={expected_sha}; public_id={public_id}"
        )

    expected_prefix = f"freshlens_datasets/{version}/"

    if not public_id.startswith(expected_prefix):
        raise RuntimeError(
            "Uploaded Cloudinary asset is outside the expected dataset "
            f"folder. public_id={public_id}"
        )

    remote_sha = sha256_url(url)

    if remote_sha.lower() != expected_sha:
        raise RuntimeError(
            "Uploaded Cloudinary RAW bytes do not match the manifest "
            f"SHA-256. path={source_path}; "
            f"expected={expected_sha}; actual={remote_sha}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or reconcile a locked FreshLens CNN dataset."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing manifest-relative image paths."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "cnn_dataset_v3"
            / "manifest.csv"
        ),
    )

    parser.add_argument(
        "--version",
        default="cnn_v1",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process only the first N rows for testing; 0 = all."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate local files and SHA-256 without writing "
            "MongoDB/Cloudinary."
        ),
    )

    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help=(
            "Backfill MongoDB records for assets already present "
            "on Cloudinary; never upload missing assets."
        ),
    )

    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")

    root = args.root.resolve()
    manifest = args.manifest.resolve()

    rows, manifest_bytes = load_manifest(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)

    rows_to_process = (
        rows[: args.limit]
        if args.limit
        else rows
    )

    print(f"[DATASET] version={args.version}")
    print(f"[DATASET] records={len(rows)}")
    print(f"[DATASET] manifest_sha256={manifest_sha}")
    print(f"[ROOT] {root}")

    if args.limit:
        print(
            "[TEST MODE] Processing "
            f"{len(rows_to_process)}/{len(rows)} rows."
        )

    # Validate all selected local files before any remote write.
    for i, row in enumerate(rows_to_process, 1):
        path = safe_local_path(root, row["path"])
        actual = sha256_file(path)

        if actual.lower() != row["sha256"].lower():
            raise RuntimeError(
                f"SHA-256 mismatch: {row['path']}"
            )

        if i % 250 == 0 or i == len(rows_to_process):
            print(
                f"[VERIFY] {i}/{len(rows_to_process)}"
            )

    if args.dry_run:
        print(
            "[OK] Dry-run successful. "
            "No MongoDB/Cloudinary writes."
        )
        return 0

    configure_cloudinary()

    if not test_connection():
        raise RuntimeError(
            "MongoDB Atlas connection failed."
        )

    db = get_db()
    images = db["dataset_images"]
    versions = db["dataset_versions"]

    create_indexes(images)

    print(
        "[CLOUD] Reading existing RAW asset inventory..."
    )

    cloud_resources = list_cloud_resources(
        args.version
    )

    print(
        "[CLOUD] Existing RAW assets="
        f"{len(cloud_resources)}"
    )

    uploaded = 0
    reconciled = 0
    skipped = 0
    missing = 0

    for i, row in enumerate(rows_to_process, 1):
        path = safe_local_path(
            root,
            row["path"],
        )

        sha = row["sha256"].lower()

        query = {
            "dataset_version": args.version,
            "sha256": sha,
        }

        existing = images.find_one(query)
        cloud_item = cloud_resources.get(sha)

        # Case 1:
        # MongoDB and Cloudinary already contain this SHA.
        if existing and existing.get("cloud_url"):
            validate_existing_metadata(
                existing,
                row,
            )

            if not cloud_item:
                raise RuntimeError(
                    "MongoDB references an asset whose SHA is missing "
                    f"from Cloudinary: {row['path']}"
                )

            cloud_public_id = str(
                cloud_item.get("public_id", "")
            )

            mongo_public_id = str(
                existing.get("cloud_public_id", "")
            )

            if (
                mongo_public_id
                and cloud_public_id
                and mongo_public_id != cloud_public_id
            ):
                raise RuntimeError(
                    "MongoDB references a different Cloudinary public_id "
                    f"for the same SHA: {row['path']}"
                )

            backfill: dict[str, str] = {}

            if not mongo_public_id and cloud_public_id:
                backfill["cloud_public_id"] = cloud_public_id

            if not existing.get("cloud_resource_type"):
                backfill["cloud_resource_type"] = "raw"

            if backfill:
                images.update_one(
                    query,
                    {"$set": backfill},
                )

            skipped += 1

        # Case 2:
        # Cloudinary contains this SHA but MongoDB is missing/incomplete.
        elif cloud_item:
            cloud_public_id = str(
                cloud_item.get("public_id", "")
            )

            url = str(
                cloud_item.get("secure_url")
                or cloud_item.get("url")
                or ""
            )

            if not cloud_public_id:
                raise RuntimeError(
                    "Existing Cloudinary resource has no public_id "
                    f"for SHA: {sha}"
                )

            if not url:
                raise RuntimeError(
                    "Existing Cloudinary resource has no URL: "
                    f"{cloud_public_id}"
                )

            remote_sha = sha256_url(url)

            if remote_sha.lower() != sha:
                raise RuntimeError(
                    "Existing Cloudinary RAW bytes do not match "
                    f"manifest SHA-256: {row['path']}"
                )

            doc = mongo_document(
                row,
                args.version,
                url,
                cloud_public_id,
            )

            images.update_one(
                query,
                {"$set": doc},
                upsert=True,
            )

            reconciled += 1

        # Case 3A:
        # Missing from both systems; reconcile-only never uploads.
        elif args.reconcile_only:
            missing += 1

        # Case 3B:
        # Missing from both systems; normal publish uploads original bytes.
        else:
            url, uploaded_public_id = upload_raw(
                path,
                sha,
                args.version,
            )

            # IMPORTANT:
            # Cloudinary may preserve original extension case
            # (.JPG instead of .jpg). We validate by SHA-256,
            # expected folder, and exact remote bytes.
            verify_uploaded_asset(
                url=url,
                public_id=uploaded_public_id,
                expected_sha=sha,
                version=args.version,
                source_path=row["path"],
            )

            doc = mongo_document(
                row,
                args.version,
                url,
                uploaded_public_id,
            )

            images.update_one(
                query,
                {"$set": doc},
                upsert=True,
            )

            # Inventory is indexed by SHA-256, not public_id.
            cloud_resources[sha] = {
                "public_id": uploaded_public_id,
                "secure_url": url,
            }

            uploaded += 1

        if i % 100 == 0 or i == len(rows_to_process):
            print(
                f"[PUBLISH] {i}/{len(rows_to_process)} "
                f"uploaded={uploaded} "
                f"reconciled={reconciled} "
                f"skipped={skipped} "
                f"missing={missing}"
            )

    # Never mark limited/reconcile-only runs complete.
    if args.limit or args.reconcile_only:
        if args.reconcile_only:
            print(
                "[OK] Reconciliation finished. "
                "Missing assets were NOT uploaded."
            )
        else:
            print(
                "[OK] Limited publish finished. "
                "Dataset version was not marked complete."
            )

        return 0

    # Final MongoDB verification.
    mongo_docs = list(
        images.find(
            {
                "dataset_version": args.version,
            },
            {
                "_id": 0,
                "sha256": 1,
                "cloud_url": 1,
                "cloud_public_id": 1,
            },
        )
    )

    expected_shas = {
        row["sha256"].lower()
        for row in rows
    }

    all_mongo_shas = {
        str(doc.get("sha256", "")).lower()
        for doc in mongo_docs
        if doc.get("sha256")
    }

    missing_mongo_shas = (
        expected_shas - all_mongo_shas
    )

    extra_mongo_shas = (
        all_mongo_shas - expected_shas
    )

    if missing_mongo_shas:
        raise RuntimeError(
            "Final MongoDB inventory is missing manifest assets. "
            f"Missing={len(missing_mongo_shas)}"
        )

    if extra_mongo_shas:
        raise RuntimeError(
            "Final MongoDB dataset version contains assets outside "
            f"the manifest. Extra={len(extra_mongo_shas)}"
        )

    if len(mongo_docs) != len(rows):
        raise RuntimeError(
            "Final MongoDB record count does not match manifest. "
            f"MongoDB={len(mongo_docs)}; manifest={len(rows)}"
        )

    incomplete_mongo = [
        doc
        for doc in mongo_docs
        if (
            not doc.get("cloud_url")
            or not doc.get("cloud_public_id")
        )
    ]

    if incomplete_mongo:
        raise RuntimeError(
            "Final MongoDB inventory contains incomplete cloud metadata. "
            f"Incomplete={len(incomplete_mongo)}"
        )

    # Final Cloudinary verification by SHA identity.
    final_cloud_resources = list_cloud_resources(
        args.version
    )

    actual_cloud_shas = set(
        final_cloud_resources
    )

    missing_cloud_shas = (
        expected_shas - actual_cloud_shas
    )

    extra_cloud_shas = (
        actual_cloud_shas - expected_shas
    )

    if missing_cloud_shas:
        raise RuntimeError(
            "Final Cloudinary inventory is missing manifest assets. "
            f"Missing={len(missing_cloud_shas)}"
        )

    if extra_cloud_shas:
        raise RuntimeError(
            "Final Cloudinary dataset folder contains assets outside "
            f"the manifest. Extra={len(extra_cloud_shas)}"
        )

    if len(final_cloud_resources) != len(rows):
        raise RuntimeError(
            "Final Cloudinary asset count does not match manifest. "
            f"Cloudinary={len(final_cloud_resources)}; "
            f"manifest={len(rows)}"
        )

    split_counts = Counter(
        row["split"]
        for row in rows
    )

    label_counts = Counter(
        f"{row['fruit']}::{row['status']}"
        for row in rows
    )

    versions.update_one(
        {
            "dataset_version": args.version,
        },
        {
            "$set": {
                "dataset_version": args.version,
                "manifest_sha256": manifest_sha,
                "record_count": len(rows),
                "split_counts": dict(split_counts),
                "label_counts": dict(label_counts),
                "cloud_resource_type": "raw",
                "status": "published",
                "published_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )

    print(
        "[VERIFY] MongoDB complete="
        f"{len(mongo_docs)}/{len(rows)}"
    )

    print(
        "[VERIFY] Cloudinary complete="
        f"{len(final_cloud_resources)}/{len(rows)}"
    )

    print(
        f"[OK] Dataset {args.version} published successfully: "
        f"{len(rows)} images."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
