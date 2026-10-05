"""Safe image loading and byte-level integrity checks."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from freshlens_ai.errors import DataError
from freshlens_ai.utils.file_io import sha256_bytes


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()

    if (
        not path.is_relative_to(root)
        or not path.is_file()
    ):
        raise DataError(
            "Image not found inside dataset root: " + relative
        )

    return path


def rgb_from_bytes(data):
    with Image.open(io.BytesIO(data)) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise DataError(
                "Please use a single-frame photo."
            )

        image = ImageOps.exif_transpose(opened)

        if (
            "A" in image.getbands()
            or "transparency" in image.info
        ):
            rgba = image.convert("RGBA")

            image = Image.alpha_composite(
                Image.new("RGBA", rgba.size, "white"),
                rgba,
            ).convert("RGB")

        else:
            image = image.convert("RGB")

        image.load()
        return image


def read_record_image(root, row):
    data = safe_path(
        root,
        row["path"],
    ).read_bytes()

    if sha256_bytes(data) != row["sha256"]:
        raise DataError(
            "Image changed since data preparation: "
            + row["path"]
        )

    try:
        return rgb_from_bytes(data)

    except Exception as exc:
        raise DataError(
            "Cannot decode image: " + row["path"]
        ) from exc


def verify_images(root, selected_rows):
    for index, row in enumerate(
        selected_rows,
        1,
    ):
        read_record_image(
            root,
            row,
        )

        if (
            index % 250 == 0
            or index == len(selected_rows)
        ):
            print(
                f"[DATA CHECK] "
                f"{index}/{len(selected_rows)} selected images",
                flush=True,
            )
