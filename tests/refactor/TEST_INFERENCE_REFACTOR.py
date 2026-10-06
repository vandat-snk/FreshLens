"""Behavior-equivalence test for Stage G1 FreshLens inference refactor."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
STEP5_FIX = PROJECT_DIR / "legacy" / "development" / "step5_fix"
STEP3 = PROJECT_DIR / "legacy" / "development" / "step3_cnn"

CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "best.pt"
)

GATE_NPZ = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "open_set_gate.npz"
)

GATE_META = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "open_set_gate.json"
)


def import_from_path(
    module_name,
    path,
    legacy_dir,
):
    sys.path.insert(
        0,
        str(legacy_dir),
    )

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
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
        sys.path.pop(0)


# Load FIX runtime first. Keep references even if legacy cnn_* names
# are later replaced while importing the Step 3 predictor.
legacy_open_set = import_from_path(
    "legacy_step5_fix_open_set",
    STEP5_FIX / "open_set.py",
    STEP5_FIX,
)

# Force Step 3 PREDICT_CNN.py to resolve its own legacy cnn_* copies.
sys.modules.pop(
    "cnn_data",
    None,
)
sys.modules.pop(
    "cnn_model",
    None,
)

legacy_predict = import_from_path(
    "legacy_step3_predict",
    STEP3 / "PREDICT_CNN.py",
    STEP3,
)

from freshlens_ai.inference import (  # noqa: E402
    analyze_bytes,
    gate_features,
    load_gate,
    normalize_rows,
    predict_image,
    sigmoid,
    supported_probability,
)
from freshlens_ai.models import load_model  # noqa: E402


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

        assert (
            left.keys()
            == right.keys()
        ), path

        for key in left:
            compare_nested(
                left[key],
                right[key],
                f"{path}.{key}",
            )

        return

    if isinstance(
        left,
        np.ndarray,
    ):
        assert isinstance(
            right,
            np.ndarray,
        ), path

        assert np.array_equal(
            left,
            right,
        ), path

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


def synthetic_png_bytes():
    # Deterministic RGB image. It does not need to be a real fruit:
    # this test checks implementation parity, not model quality.
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


def main():
    for path in (
        CHECKPOINT,
        GATE_NPZ,
        GATE_META,
    ):
        if not path.is_file():
            raise SystemExit(
                "[ERROR] Missing required production artifact: "
                + str(path)
            )

    rng = np.random.default_rng(
        2026
    )

    values = rng.normal(
        size=25
    )

    assert np.array_equal(
        legacy_open_set.sigmoid(
            values
        ),
        sigmoid(
            values
        ),
    )

    embeddings = rng.normal(
        size=(9, 1280),
    ).astype(np.float32)

    assert np.array_equal(
        legacy_open_set.normalize_rows(
            embeddings
        ),
        normalize_rows(
            embeddings
        ),
    )

    print(
        "[OK] sigmoid and embedding normalization match"
    )

    gate_old = legacy_open_set.load_gate(
        GATE_NPZ,
        GATE_META,
        CHECKPOINT,
    )

    gate_new = load_gate(
        GATE_NPZ,
        GATE_META,
        CHECKPOINT,
    )

    compare_nested(
        gate_old,
        gate_new,
        "load_gate",
    )

    assert (
        gate_new["meta"]["version"]
        == "1.1-external-calibration"
    )

    assert (
        gate_new["prototypes"].shape
        == (4, 4, 1280)
    )

    print(
        "[OK] Production gate 1.1 loads identically "
        "(4 fruits x 4 prototypes x 1280-D)"
    )

    raw = rng.random(
        (9, 8)
    )

    probabilities = (
        raw
        / raw.sum(
            axis=1,
            keepdims=True,
        )
    )

    old_features, old_details = (
        legacy_open_set.gate_features(
            embeddings,
            probabilities,
            gate_old["prototypes"],
        )
    )

    new_features, new_details = gate_features(
        embeddings,
        probabilities,
        gate_new["prototypes"],
    )

    assert np.array_equal(
        old_features,
        new_features,
    )

    compare_nested(
        old_details,
        new_details,
        "gate_details",
    )

    print(
        "[OK] Six open-set gate features match exactly"
    )

    old_support = (
        legacy_open_set.supported_probability(
            old_features,
            gate_old,
        )
    )

    new_support = supported_probability(
        new_features,
        gate_new,
    )

    assert np.array_equal(
        old_support,
        new_support,
    )

    print(
        "[OK] Supported probabilities match exactly"
    )

    device = torch.device(
        "cpu"
    )

    model, checkpoint = load_model(
        CHECKPOINT,
        device,
    )

    assert int(
        checkpoint["epoch"]
    ) == int(
        gate_new["meta"]["checkpoint_epoch"]
    )

    image_bytes = synthetic_png_bytes()

    old_plain = legacy_predict.predict_image(
        model,
        image_bytes,
        device,
    )

    new_plain = predict_image(
        model,
        image_bytes,
        device,
    )

    compare_nested(
        old_plain,
        new_plain,
        "predict_image",
    )

    print(
        "[OK] Plain single-image prediction contract matches legacy PREDICT_CNN.py"
    )

    old_open = legacy_open_set.analyze_bytes(
        model,
        image_bytes,
        device,
        gate_old,
    )

    new_open = analyze_bytes(
        model,
        image_bytes,
        device,
        gate_new,
    )

    compare_nested(
        old_open,
        new_open,
        "analyze_bytes",
    )

    print(
        "[OK] Open-set analyze_bytes result matches Step5_FIX runtime"
    )

    print(
        "[PASS] Stage G1 inference refactor is behavior-equivalent "
        "to legacy prediction/open-set runtime"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
