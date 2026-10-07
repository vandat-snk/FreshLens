r"""Publish the locked FreshLens CNN dataset to Cloudinary + MongoDB Atlas.

Important:
- Reads the existing locked manifest (cnn_dataset_v3).
- Uploads ORIGINAL BYTES as Cloudinary RAW resources to preserve SHA-256.
- Stores metadata in MongoDB Atlas; images are not stored inside MongoDB.
- Safe to re-run: existing matching records are skipped.
- Can reconcile Cloudinary assets that exist but are missing MongoDB metadata.
- Cloud asset identity is SHA-256, not filename extension/case.

Usage:
    python scripts/publish_cnn_dataset.py ^
      --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
      --manifest "data\cnn_dataset_v3\manifest.csv" ^
      --version cnn_v1
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
        "Cloudinary dependency is missing. Run: pip install -r requirements-db.txt"
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
    """Download a remote RAW asset and calculate its SHA-256."""
    h = hashlib.sha256()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FreshLens-Dataset-Publisher/2.1"},
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
            or any(
                char not in "0123456789abcdef"
                for char in sha
            )
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

    missing = [
        key
        for key in required
        if not os.getenv(key)
    ]

    if missing:
        raise RuntimeError(
            "Missing Cloudinary environment variables: "
            + ", ".join(missing)
        )

    cloudinary.config(
        cloud_name=os.environ[
            "CLOUDINARY_CLOUD_NAME"
        ],
        api_key=os.environ[
            "CLOUDINARY_API_KEY"
        ],
        api_secret=os.environ[
            "CLOUDINARY_API_SECRET"
        ],
        secure=True,
    )


def expected_public_id(
    path: Path,
    sha256: str,
    version: str,
) -> str:
    """Return deterministic public_id used only for NEW uploads."""
    suffix = path.suffix.lower() or ".bin"

    return (
        f"freshlens_datasets/"
        f"{version}/"
        f"{sha256}{suffix}"
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
        and all(
            char in "0123456789abcdef"
            for char in candidate
        )
    ):
        return candidate

    return None


def list_cloud_resources(
    version: str,
) -> dict[str, dict]:
    """Return RAW resources indexed by SHA-256."""
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

        for item in response.get(
            "resources",
            [],
        ):
            public_id = str(
                item.get(
                    "public_id",
                    "",
                )
            )

            if not public_id:
                raise RuntimeError(
                    "Cloudinary returned a RAW resource "
                    "without public_id."
                )

            sha = sha_from_public_id(
                public_id
            )

            if sha is None:
                raise RuntimeError(
                    "Cloudinary RAW asset does not use "
                    "a valid SHA-based public_id: "
                    f"{public_id}"
                )

            if sha in resources_by_sha:
                previous = str(
                    resources_by_sha[sha].get(
                        "public_id",
                        "",
                    )
                )

                raise RuntimeError(
                    "Multiple Cloudinary RAW assets have "
                    "the same SHA-256 identity. "
                    f"sha={sha}; "
                    f"first={previous}; "
                    f"second={public_id}"
                )

            resources_by_sha[
                sha
            ] = item

        cursor = response.get(
            "next_cursor"
        )

        if not cursor:
            break

    return resources_by_sha


def validate_existing_metadata(
    existing: dict,
    row: dict,
) -> None:
    """Reject Mongo records whose authoritative metadata differs."""
    for key in (
        "path",
        "fruit",
        "status",
        "split",
        "group_id",
    ):
        if str(
            existing.get(
                key,
                "",
            )
        ) != str(
            row.get(
                key,
                "",
            )
        ):
            raise RuntimeError(
                "MongoDB contains the same SHA "
                "with different metadata "
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
        "dataset_version":
            version,
        "manifest_index":
            int(row["manifest_index"]),
        "path":
            row["path"],
        "fruit":
            row["fruit"],
        "status":
            row["status"],
        "source":
            row["source"],
        "group_id":
            row["group_id"],
        "sha256":
            row["sha256"],
        "split":
            row["split"],
        "cloud_url":
            cloud_url,
        "cloud_public_id":
            public_id,
        "cloud_resource_type":
            "raw",
        "published_at":
            datetime.now(
                timezone.utc
            ),
    }


def create_indexes(
    images,
) -> None:
    """Create indexes required by dataset publishing."""
    images.create_index(
        [
            (
                "dataset_version",
                ASCENDING,
            ),
            (
                "sha256",
                ASCENDING,
            ),
        ],
        unique=True,
        name="dataset_version_sha_unique",
    )

    images.create_index(
        [
            (
                "dataset_version",
                ASCENDING,
            ),
            (
                "manifest_index",
                ASCENDING,
            ),
        ]
    )

    images.create_index(
        [
            (
                "dataset_version",
                ASCENDING,
            ),
            (
                "split",
                ASCENDING,
            ),
        ]
    )

    images.create_index(
        [
            (
                "dataset_version",
                ASCENDING,
            ),
            (
                "fruit",
                ASCENDING,
            ),
            (
                "status",
                ASCENDING,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or reconcile a locked "
            "FreshLens CNN dataset."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing "
            "manifest-relative image paths."
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
            "Process only the first N rows "
            "for testing; 0 = all."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate local files and SHA-256 "
            "without writing MongoDB/Cloudinary."
        ),
    )

    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help=(
            "Backfill MongoDB records for assets "
            "already present on Cloudinary; "
            "never upload missing assets."
        ),
    )

    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError(
            "--limit cannot be negative."
        )

    root = args.root.resolve()
    manifest = args.manifest.resolve()

    rows, manifest_bytes = (
        load_manifest(
            manifest
        )
    )

    manifest_sha = sha256_bytes(
        manifest_bytes
    )

    rows_to_process = (
        rows[: args.limit]
        if args.limit
        else rows
    )

    print(
        f"[DATASET] version="
        f"{args.version}"
    )

    print(
        f"[DATASET] records="
        f"{len(rows)}"
    )

    print(
        "[DATASET] manifest_sha256="
        f"{manifest_sha}"
    )

    print(
        f"[ROOT] {root}"
    )

    if args.limit:
        print(
            "[TEST MODE] Processing "
            f"{len(rows_to_process)}/"
            f"{len(rows)} rows."
        )

    # -----------------------------------------------------
    # Local validation.
    # No remote write happens before all selected rows pass.
    # -----------------------------------------------------
    for i, row in enumerate(
        rows_to_process,
        1,
    ):
        path = safe_local_path(
            root,
            row["path"],
        )

        actual = sha256_file(
            path
        )

        if (
            actual.lower()
            != row["sha256"].lower()
        ):
            raise RuntimeError(
                "SHA-256 mismatch: "
                f"{row['path']}"
            )

        if (
            i % 250 == 0
            or i == len(
                rows_to_process
            )
        ):
            print(
                f"[VERIFY] "
                f"{i}/"
                f"{len(rows_to_process)}"
            )

    if args.dry_run:
        print(
            "[OK] Dry-run successful. "
            "No MongoDB/Cloudinary writes."
        )

        return 0

    # -----------------------------------------------------
    # External services.
    # -----------------------------------------------------
    configure_cloudinary()

    if not test_connection():
        raise RuntimeError(
            "MongoDB Atlas connection failed."
        )

    db = get_db()

    images = db[
        "dataset_images"
    ]

    versions = db[
        "dataset_versions"
    ]

    create_indexes(
        images
    )

    print(
        "[CLOUD] Reading existing "
        "RAW asset inventory..."
    )

    cloud_resources = (
        list_cloud_resources(
            args.version
        )
    )

    print(
        "[CLOUD] Existing RAW assets="
        f"{len(cloud_resources)}"
    )

    uploaded = 0
    reconciled = 0
    skipped = 0
    missing = 0

    # -----------------------------------------------------
    # Reconcile / upload.
    #
    # IMPORTANT:
    # Cloudinary inventory is indexed by SHA-256.
    # Therefore .JPG / .jpg differences do not make
    # an existing asset look missing.
    # -----------------------------------------------------
    for i, row in enumerate(
        rows_to_process,
        1,
    ):
        path = safe_local_path(
            root,
            row["path"],
        )

        sha = row[
            "sha256"
        ].lower()

        query = {
            "dataset_version":
                args.version,
            "sha256":
                sha,
        }

        existing = images.find_one(
            query
        )

        cloud_item = (
            cloud_resources.get(
                sha
            )
        )

        # -------------------------------------------------
        # Case 1:
        # MongoDB and Cloudinary already contain this SHA.
        # -------------------------------------------------
        if (
            existing
            and existing.get(
                "cloud_url"
            )
        ):
            validate_existing_metadata(
                existing,
                row,
            )

            if not cloud_item:
                raise RuntimeError(
                    "MongoDB references an asset "
                    "whose SHA is missing from "
                    "Cloudinary: "
                    f"{row['path']}"
                )

            cloud_public_id = str(
                cloud_item.get(
                    "public_id",
                    "",
                )
            )

            mongo_public_id = str(
                existing.get(
                    "cloud_public_id",
                    "",
                )
            )

            if (
                mongo_public_id
                and cloud_public_id
                and mongo_public_id
                != cloud_public_id
            ):
                raise RuntimeError(
                    "MongoDB references a different "
                    "Cloudinary public_id for the "
                    "same SHA: "
                    f"{row['path']}"
                )

            # Safe metadata backfill only.
            backfill: dict[str, str] = {}

            if (
                not mongo_public_id
                and cloud_public_id
            ):
                backfill[
                    "cloud_public_id"
                ] = cloud_public_id

            if not existing.get(
                "cloud_resource_type"
            ):
                backfill[
                    "cloud_resource_type"
                ] = "raw"

            if backfill:
                images.update_one(
                    query,
                    {
                        "$set":
                            backfill,
                    },
                )

            skipped += 1

        # -------------------------------------------------
        # Case 2:
        # Cloudinary contains this SHA but MongoDB is
        # missing/incomplete.
        #
        # Verify remote bytes before backfilling MongoDB.
        # -------------------------------------------------
        elif cloud_item:
            cloud_public_id = str(
                cloud_item.get(
                    "public_id",
                    "",
                )
            )

            url = str(
                cloud_item.get(
                    "secure_url"
                )
                or cloud_item.get(
                    "url"
                )
                or ""
            )

            if not cloud_public_id:
                raise RuntimeError(
                    "Existing Cloudinary resource "
                    "has no public_id for SHA: "
                    f"{sha}"
                )

            if not url:
                raise RuntimeError(
                    "Existing Cloudinary resource "
                    "has no URL: "
                    f"{cloud_public_id}"
                )

            remote_sha = (
                sha256_url(
                    url
                )
            )

            if (
                remote_sha.lower()
                != sha
            ):
                raise RuntimeError(
                    "Existing Cloudinary RAW "
                    "bytes do not match "
                    "manifest SHA-256: "
                    f"{row['path']}"
                )

            doc = mongo_document(
                row,
                args.version,
                url,
                cloud_public_id,
            )

            images.update_one(
                query,
                {
                    "$set":
                        doc,
                },
                upsert=True,
            )

            reconciled += 1

        # -------------------------------------------------
        # Case 3A:
        # Missing from both systems.
        # Reconcile-only NEVER uploads it.
        # -------------------------------------------------
        elif args.reconcile_only:
            missing += 1

        # -------------------------------------------------
        # Case 3B:
        # Missing from both systems during normal publish.
        # Upload original bytes and write Mongo metadata.
        # -------------------------------------------------
        else:
            expected_new_public_id = (
                expected_public_id(
                    path,
                    sha,
                    args.version,
                )
            )

            (
                url,
                uploaded_public_id,
            ) = upload_raw(
                path,
                sha,
                args.version,
            )

            if (
                uploaded_public_id
                != expected_new_public_id
            ):
                raise RuntimeError(
                    "Unexpected Cloudinary "
                    "public_id after upload. "
                    f"Expected="
                    f"{expected_new_public_id}; "
                    f"actual="
                    f"{uploaded_public_id}"
                )

            doc = mongo_document(
                row,
                args.version,
                url,
                uploaded_public_id,
            )

            images.update_one(
                query,
                {
                    "$set":
                        doc,
                },
                upsert=True,
            )

            # IMPORTANT:
            # cloud_resources is keyed by SHA.
            cloud_resources[
                sha
            ] = {
                "public_id":
                    uploaded_public_id,
                "secure_url":
                    url,
            }

            uploaded += 1

        if (
            i % 100 == 0
            or i == len(
                rows_to_process
            )
        ):
            print(
                f"[PUBLISH] "
                f"{i}/"
                f"{len(rows_to_process)} "
                f"uploaded={uploaded} "
                f"reconciled={reconciled} "
                f"skipped={skipped} "
                f"missing={missing}"
            )

    # -----------------------------------------------------
    # Limited or reconcile-only runs NEVER mark the
    # dataset version complete.
    # -----------------------------------------------------
    if (
        args.limit
        or args.reconcile_only
    ):
        if args.reconcile_only:
            print(
                "[OK] Reconciliation finished. "
                "Missing assets were NOT uploaded."
            )
        else:
            print(
                "[OK] Limited publish finished. "
                "Dataset version was not "
                "marked complete."
            )

        return 0

    # -----------------------------------------------------
    # Final MongoDB verification by SHA identity.
    # -----------------------------------------------------
    mongo_docs = list(
        images.find(
            {
                "dataset_version":
                    args.version,
            },
            {
                "_id": 0,
                "sha256": 1,
                "cloud_url": 1,
            },
        )
    )

    expected_shas = {
        row["sha256"].lower()
        for row in rows
    }

    mongo_shas = {
        str(
            doc.get(
                "sha256",
                "",
            )
        ).lower()
        for doc in mongo_docs
        if doc.get(
            "sha256"
        )
        and doc.get(
            "cloud_url"
        )
    }

    missing_mongo_shas = (
        expected_shas
        - mongo_shas
    )

    extra_mongo_shas = (
        mongo_shas
        - expected_shas
    )

    if missing_mongo_shas:
        raise RuntimeError(
            "Final MongoDB inventory is "
            "missing manifest assets. "
            f"Missing="
            f"{len(missing_mongo_shas)}"
        )

    if extra_mongo_shas:
        raise RuntimeError(
            "Final MongoDB dataset version "
            "contains assets outside the "
            "manifest. "
            f"Extra="
            f"{len(extra_mongo_shas)}"
        )

    if (
        len(mongo_shas)
        != len(rows)
    ):
        raise RuntimeError(
            "Final MongoDB record count "
            "does not match manifest. "
            f"MongoDB="
            f"{len(mongo_shas)}; "
            f"manifest="
            f"{len(rows)}"
        )

    # -----------------------------------------------------
    # Final Cloudinary verification by SHA identity.
    # -----------------------------------------------------
    final_cloud_resources = (
        list_cloud_resources(
            args.version
        )
    )

    actual_cloud_shas = set(
        final_cloud_resources
    )

    missing_cloud_shas = (
        expected_shas
        - actual_cloud_shas
    )

    extra_cloud_shas = (
        actual_cloud_shas
        - expected_shas
    )

    if missing_cloud_shas:
        raise RuntimeError(
            "Final Cloudinary inventory "
            "is missing manifest assets. "
            f"Missing="
            f"{len(missing_cloud_shas)}"
        )

    if extra_cloud_shas:
        raise RuntimeError(
            "Final Cloudinary dataset folder "
            "contains assets outside the "
            "manifest. "
            f"Extra="
            f"{len(extra_cloud_shas)}"
        )

    if (
        len(final_cloud_resources)
        != len(rows)
    ):
        raise RuntimeError(
            "Final Cloudinary asset count "
            "does not match manifest. "
            f"Cloudinary="
            f"{len(final_cloud_resources)}; "
            f"manifest="
            f"{len(rows)}"
        )

    split_counts = Counter(
        row["split"]
        for row in rows
    )

    label_counts = Counter(
        (
            f"{row['fruit']}"
            f"::{row['status']}"
        )
        for row in rows
    )

    versions.update_one(
        {
            "dataset_version":
                args.version,
        },
        {
            "$set": {
                "dataset_version":
                    args.version,
                "manifest_sha256":
                    manifest_sha,
                "record_count":
                    len(rows),
                "split_counts":
                    dict(
                        split_counts
                    ),
                "label_counts":
                    dict(
                        label_counts
                    ),
                "cloud_resource_type":
                    "raw",
                "status":
                    "published",
                "published_at":
                    datetime.now(
                        timezone.utc
                    ),
            }
        },
        upsert=True,
    )

    print(
        "[VERIFY] MongoDB complete="
        f"{len(mongo_shas)}/"
        f"{len(rows)}"
    )

    print(
        "[VERIFY] Cloudinary complete="
        f"{len(final_cloud_resources)}/"
        f"{len(rows)}"
    )

    print(
        f"[OK] Dataset "
        f"{args.version} "
        f"published successfully: "
        f"{len(rows)} images."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )