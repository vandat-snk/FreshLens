"""Validation and loading of the immutable FreshLens CNN dataset manifest."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

from freshlens_ai.constants import CLASSES, SPLITS
from freshlens_ai.errors import DataError
from freshlens_ai.utils.file_io import sha256_bytes


def load_locked_dataset(directory):
    """Inspect metadata for all splits; this function never opens image files."""
    directory = Path(directory)

    manifest_bytes = (directory / "manifest.csv").read_bytes()
    report_bytes = (directory / "final_split_report.json").read_bytes()

    report = json.loads(report_bytes.decode("utf-8-sig"))
    lock = json.loads(
        (directory / "dataset_lock.json").read_text(encoding="utf-8-sig")
    )

    fingerprint = sha256_bytes(manifest_bytes)

    if (
        fingerprint != report.get("output_manifest_sha256")
        or fingerprint != lock.get("manifest_sha256")
        or sha256_bytes(report_bytes) != lock.get("report_sha256")
    ):
        raise DataError(
            "Manifest/report/lock do not match. "
            "Use the unchanged cnn_dataset_v3 output."
        )

    if (
        report.get("version") != "2.1.0"
        or not report.get("ready_for_pilot_training")
        or not lock.get("ready_for_pilot_training")
    ):
        raise DataError(
            "Dataset has not passed step 2B for pilot training."
        )

    pairs = report.get("pair_handling", {})

    if (
        pairs.get("cross_split_pairs_after") != 0
        or pairs.get("all_candidate_links_inside_one_group") is not True
        or pairs.get("all_candidates_recomputed_from_image_files") is not True
    ):
        raise DataError(
            "Unresolved candidate links in the data report."
        )

    reader = csv.DictReader(
        io.StringIO(manifest_bytes.decode("utf-8-sig"))
    )

    columns = (
        "path",
        "fruit",
        "status",
        "source",
        "group_id",
        "sha256",
        "split",
    )

    if not set(columns).issubset(reader.fieldnames or []):
        raise DataError(
            "The manifest is missing required columns."
        )

    rows = []
    seen_paths = set()
    groups = defaultdict(set)
    hashes = defaultdict(set)

    for raw in reader:
        row = {
            key: (raw.get(key) or "").strip()
            for key in columns
        }

        relative = row["path"].replace("\\", "/")
        pure = PurePosixPath(relative)

        if (
            not relative
            or pure.is_absolute()
            or PureWindowsPath(relative).drive
            or ".." in pure.parts
            or ":" in relative
            or "\x00" in relative
        ):
            raise DataError(
                "Unsafe image path: " + repr(relative)
            )

        row["path"] = pure.as_posix()

        if row["path"].casefold() in seen_paths:
            raise DataError(
                "Repeated image path: " + row["path"]
            )

        seen_paths.add(row["path"].casefold())

        name = row["fruit"] + "::" + row["status"]

        if (
            name not in CLASSES
            or row["split"] not in SPLITS
            or not row["group_id"]
        ):
            raise DataError(
                "Invalid label/split/group: " + row["path"]
            )

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            row["sha256"],
        ):
            raise DataError(
                "Invalid image SHA-256: " + row["path"]
            )

        row["target"] = CLASSES.index(name)

        groups[row["group_id"]].add(row["split"])
        hashes[row["sha256"]].add(row["split"])
        rows.append(row)

    if (
        any(len(values) > 1 for values in groups.values())
        or any(len(values) > 1 for values in hashes.values())
    ):
        raise DataError(
            "An image/group spans more than one split."
        )

    if (
        len(rows) != report.get("output_records")
        or len(rows) != lock.get("records")
    ):
        raise DataError(
            "Dataset record counts do not match the lock/report."
        )

    if len(groups) != report.get("new_groups"):
        raise DataError(
            "Dataset group count does not match the report."
        )

    if (
        dict(Counter(row["split"] for row in rows))
        != report.get("split_counts")
    ):
        raise DataError(
            "Dataset split counts do not match the report."
        )

    counts = Counter(
        (CLASSES[row["target"]], row["split"])
        for row in rows
    )

    expected = {
        name: {
            split: counts[(name, split)]
            for split in SPLITS
        }
        for name in CLASSES
    }

    if (
        expected != report.get("label_split_counts")
        or any(
            not count
            for values in expected.values()
            for count in values.values()
        )
    ):
        raise DataError(
            "Missing labels or report/manifest label counts disagree."
        )

    return rows, {
        "manifest_sha256": fingerprint,
        "report_sha256": sha256_bytes(report_bytes),
        "split_counts": report["split_counts"],
        "new_groups": report["new_groups"],
        "physical_specimen_independence_confirmed": report.get(
            "physical_specimen_independence_confirmed",
            False,
        ),
    }
