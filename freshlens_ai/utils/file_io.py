"""Atomic file I/O and hashing helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json(path, value) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")

        os.replace(
            temporary,
            path,
        )

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, rows, columns) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

        os.replace(
            temporary,
            path,
        )

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)