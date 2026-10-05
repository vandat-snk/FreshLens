"""Stage G3 contract test for PREDICT_CNN_V2.py."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = PROJECT_DIR / "FreshLens_Buoc3_CNN"
LEGACY_PATH = LEGACY_DIR / "PREDICT_CNN.py"

CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "best.pt"
)


def load_legacy():
    sys.path.insert(
        0,
        str(LEGACY_DIR),
    )

    try:
        spec = importlib.util.spec_from_file_location(
            "legacy_predict_entrypoint",
            LEGACY_PATH,
        )

        module = importlib.util.module_from_spec(
            spec
        )

        assert spec.loader is not None

        spec.loader.exec_module(
            module
        )

        return module

    finally:
        sys.path.pop(
            0
        )


def synthetic_png_bytes():
    y = np.arange(
        240,
        dtype=np.uint16,
    )[:, None]

    x = np.arange(
        320,
        dtype=np.uint16,
    )[None, :]

    array = np.empty(
        (240, 320, 3),
        dtype=np.uint8,
    )

    array[..., 0] = (
        (x + y) % 256
    ).astype(np.uint8)

    array[..., 1] = (
        (2 * x + y) % 256
    ).astype(np.uint8)

    array[..., 2] = (
        (x + 3 * y) % 256
    ).astype(np.uint8)

    buffer = io.BytesIO()

    Image.fromarray(
        array
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def extract_json(stdout):
    start = stdout.find("{")

    if start < 0:
        raise AssertionError(
            "CLI output does not contain JSON:\n"
            + stdout
        )

    return json.loads(
        stdout[start:]
    )


def compare_nested(
    left,
    right,
    path="root",
):
    if isinstance(
        left,
        dict,
    ):
        assert isinstance(
            right,
            dict,
        ), path

        assert left.keys() == right.keys(), path

        for key in left:
            compare_nested(
                left[key],
                right[key],
                f"{path}.{key}",
            )

        return

    if isinstance(
        left,
        float,
    ):
        assert isinstance(
            right,
            (float, int),
        ), path

        assert np.isclose(
            left,
            right,
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ), (
            path,
            left,
            right,
        )

        return

    assert left == right, (
        path,
        left,
        right,
    )


def main():
    if not CHECKPOINT.is_file():
        raise SystemExit(
            "[ERROR] Missing production checkpoint: "
            + str(CHECKPOINT)
        )

    legacy = load_legacy()

    import PREDICT_CNN_V2 as modern

    # CLI default contract.
    legacy_parser = legacy.argparse.ArgumentParser(
        description="Predict fruit and condition from a photo"
    )

    legacy_parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )

    legacy_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=legacy.PROJECT_DIR
        / "models"
        / "cnn_efficientnet_b0"
        / "best.pt",
    )

    legacy_parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cpu",
    )

    old_args = legacy_parser.parse_args(
        [
            "--image",
            "sample.png",
        ]
    )

    new_args = modern.build_parser().parse_args(
        [
            "--image",
            "sample.png",
        ]
    )

    assert old_args.device == new_args.device == "cpu"

    assert (
        Path(old_args.checkpoint).resolve()
        == Path(new_args.checkpoint).resolve()
    )

    print(
        "[OK] CLI defaults match legacy PREDICT_CNN.py"
    )

    # The V2 entry point must not depend on legacy modules.
    source = (
        PROJECT_DIR
        / "PREDICT_CNN_V2.py"
    ).read_text(
        encoding="utf-8"
    )

    forbidden = (
        "from cnn_data import",
        "import cnn_data",
        "from cnn_model import",
        "import cnn_model",
        "from open_set import",
        "import open_set",
    )

    for marker in forbidden:
        assert marker not in source, (
            "PREDICT_CNN_V2.py still imports legacy code: "
            + marker
        )

    print(
        "[OK] V2 predictor entry point has no legacy cnn_* dependency"
    )

    # --help smoke test.
    help_run = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_DIR
                / "PREDICT_CNN_V2.py"
            ),
            "--help",
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    assert help_run.returncode == 0, (
        help_run.stdout
        + help_run.stderr
    )

    print(
        "[OK] PREDICT_CNN_V2.py --help starts successfully"
    )

    # End-to-end CLI output equivalence on CPU.
    with tempfile.TemporaryDirectory() as temporary:
        image_path = (
            Path(temporary)
            / "sample.png"
        )

        image_path.write_bytes(
            synthetic_png_bytes()
        )

        old_run = subprocess.run(
            [
                sys.executable,
                str(LEGACY_PATH),
                "--image",
                str(image_path),
                "--checkpoint",
                str(CHECKPOINT),
                "--device",
                "cpu",
            ],
            cwd=LEGACY_DIR,
            capture_output=True,
            text=True,
        )

        assert old_run.returncode == 0, (
            old_run.stdout
            + old_run.stderr
        )

        new_run = subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_DIR
                    / "PREDICT_CNN_V2.py"
                ),
                "--image",
                str(image_path),
                "--checkpoint",
                str(CHECKPOINT),
                "--device",
                "cpu",
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )

        assert new_run.returncode == 0, (
            new_run.stdout
            + new_run.stderr
        )

        old_result = extract_json(
            old_run.stdout
        )

        new_result = extract_json(
            new_run.stdout
        )

        compare_nested(
            old_result,
            new_result,
            "cli_result",
        )

    print(
        "[OK] End-to-end JSON prediction output matches legacy CLI"
    )

    print(
        "[PASS] Stage G3 predictor entry point is behavior-equivalent "
        "to FreshLens_Buoc3_CNN/PREDICT_CNN.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
