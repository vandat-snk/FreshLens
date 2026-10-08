"""Stage G4 tests for the modular FreshLens Streamlit app."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from freshlens_ai.inference import (
    clear_result,
    image_identity,
    result_for_image,
    store_result,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
APP = PROJECT_DIR / "APP_CNN_V2.py"
RUNNER = PROJECT_DIR / "RUN_APP_V2.cmd"


def main():
    # Session-state correctness: result must be tied to exact input bytes.
    state = {}

    image_a = b"image-a"
    image_b = b"image-b"

    result_a = {
        "fruit": "apple",
        "condition": "fresh",
    }

    assert (
        image_identity(image_a)
        != image_identity(image_b)
    )

    assert (
        result_for_image(
            state,
            image_a,
        )
        is None
    )

    store_result(
        state,
        image_a,
        result_a,
    )

    assert (
        result_for_image(
            state,
            image_a,
        )
        == result_a
    )

    assert (
        result_for_image(
            state,
            image_b,
        )
        is None
    )

    clear_result(
        state
    )

    assert (
        result_for_image(
            state,
            image_a,
        )
        is None
    )

    print(
        "[OK] Session-state result is bound to the exact current image"
    )

    source = APP.read_text(
        encoding="utf-8"
    )

    # Parse without executing Streamlit/model loading.
    ast.parse(
        source,
        filename=str(APP),
    )

    print(
        "[OK] APP_CNN_V2.py syntax parses successfully"
    )

    forbidden = (
        "from cnn_data import",
        "import cnn_data",
        "from cnn_model import",
        "import cnn_model",
        "from open_set import",
        "import open_set",
        "sys.path.insert",
    )

    for marker in forbidden:
        assert marker not in source, (
            "APP_CNN_V2.py still depends on legacy runtime: "
            + marker
        )

    required = (
        "from freshlens_ai.inference import",
        "from freshlens_ai.inference.cnn_predict import FreshLensPredictor",
        "result_for_image(",
        "store_result(",
        "clear_result(",
        "st.file_uploader(",
        "st.camera_input(",
    )

    for marker in required:
        assert marker in source, (
            "APP_CNN_V2.py missing expected modular behavior: "
            + marker
        )

    print(
        "[OK] App uses freshlens_ai runtime and has no legacy cnn_* imports"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(APP),
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        completed.stdout
        + completed.stderr
    )

    print(
        "[OK] APP_CNN_V2.py compiles successfully"
    )

    assert RUNNER.is_file()

    runner_source = RUNNER.read_text(
        encoding="utf-8"
    )

    assert (
        "streamlit run APP_CNN_V2.py"
        in runner_source
    )

    print(
        "[OK] RUN_APP_V2.cmd targets the modular Streamlit app"
    )

    print(
        "[PASS] Stage G4 app refactor passed static/runtime-state tests"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
