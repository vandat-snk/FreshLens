"""Behavior-equivalence test for the FreshLens 1.1 gate builder refactor."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = PROJECT_DIR / "FreshLens_Buoc5_FIX"
LEGACY_PATH = LEGACY_DIR / "BUILD_OPENSET_GATE.py"

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
            "legacy_gate_builder",
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


legacy = load_legacy()

from freshlens_ai.inference import (  # noqa: E402
    build_metadata,
    build_prototypes,
    choose_threshold,
    deterministic_split,
    fit_gate,
)
from freshlens_ai.models import read_checkpoint  # noqa: E402


def assert_array_equal(
    left,
    right,
    name,
):
    assert np.array_equal(
        left,
        right,
    ), name


def main():
    # Deterministic path split.
    rows = []

    for fruit in range(
        4
    ):
        for index in range(
            12
        ):
            rows.append(
                (
                    Path(
                        f"/fruit_{fruit}/image_{index}.jpg"
                    ),
                    fruit,
                    f"fruit_{fruit}",
                )
            )

    old_proto, old_calib = (
        legacy.deterministic_split(
            rows
        )
    )

    new_proto, new_calib = (
        deterministic_split(
            rows
        )
    )

    assert (
        old_proto
        == new_proto
    )

    assert (
        old_calib
        == new_calib
    )

    print(
        "[OK] Deterministic prototype/calibration split matches"
    )

    # Prototype clustering.
    rng = np.random.default_rng(
        42
    )

    embeddings = rng.normal(
        size=(
            48,
            1280,
        )
    ).astype(
        np.float32
    )

    fruit_targets = np.repeat(
        np.arange(
            4
        ),
        12,
    )

    old_prototypes = (
        legacy.build_prototypes(
            embeddings,
            fruit_targets,
            4,
        )
    )

    new_prototypes = build_prototypes(
        embeddings,
        fruit_targets,
        4,
    )

    assert_array_equal(
        old_prototypes,
        new_prototypes,
        "prototypes",
    )

    assert (
        new_prototypes.shape
        == (
            4,
            4,
            1280,
        )
    )

    print(
        "[OK] 4-prototype-per-fruit clustering matches exactly"
    )

    # Threshold selection.
    known_p = np.asarray(
        [
            0.99,
            0.98,
            0.96,
            0.91,
            0.83,
            0.77,
            0.72,
            0.69,
            0.64,
            0.60,
        ],
        dtype=np.float64,
    )

    unknown_p = np.asarray(
        [
            0.55,
            0.49,
            0.40,
            0.30,
            0.20,
            0.10,
        ],
        dtype=np.float64,
    )

    old_threshold = (
        legacy.choose_threshold(
            known_p,
            unknown_p,
            0.90,
        )
    )

    new_threshold = choose_threshold(
        known_p,
        unknown_p,
        0.90,
    )

    assert (
        old_threshold
        == new_threshold
    )

    print(
        "[OK] Threshold selection matches"
    )

    # Full scaler + logistic gate fit on deterministic synthetic features.
    known_embeddings = rng.normal(
        size=(
            20,
            1280,
        )
    ).astype(
        np.float32
    )

    unknown_embeddings = rng.normal(
        size=(
            18,
            1280,
        )
    ).astype(
        np.float32
    )

    known_raw = rng.random(
        (
            20,
            8,
        )
    )

    unknown_raw = rng.random(
        (
            18,
            8,
        )
    )

    known_probs = (
        known_raw
        / known_raw.sum(
            axis=1,
            keepdims=True,
        )
    )

    unknown_probs = (
        unknown_raw
        / unknown_raw.sum(
            axis=1,
            keepdims=True,
        )
    )

    old_known_x, _ = (
        legacy.gate_features(
            known_embeddings,
            known_probs,
            old_prototypes,
        )
    )

    old_unknown_x, _ = (
        legacy.gate_features(
            unknown_embeddings,
            unknown_probs,
            old_prototypes,
        )
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.concatenate(
        [
            old_known_x,
            old_unknown_x,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.ones(
                len(old_known_x)
            ),
            np.zeros(
                len(old_unknown_x)
            ),
        ]
    )

    old_scaler = (
        StandardScaler()
        .fit(
            x
        )
    )

    old_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    old_classifier.fit(
        old_scaler.transform(
            x
        ),
        y,
    )

    old_known_probability = (
        old_classifier.predict_proba(
            old_scaler.transform(
                old_known_x
            )
        )[:, 1]
    )

    old_unknown_probability = (
        old_classifier.predict_proba(
            old_scaler.transform(
                old_unknown_x
            )
        )[:, 1]
    )

    (
        old_gate_threshold,
        old_known_accept,
        old_unknown_reject,
    ) = legacy.choose_threshold(
        old_known_probability,
        old_unknown_probability,
        0.90,
    )

    (
        new_arrays,
        new_known_accept,
        new_unknown_reject,
    ) = fit_gate(
        known_embeddings,
        known_probs,
        unknown_embeddings,
        unknown_probs,
        new_prototypes,
        0.90,
    )

    assert_array_equal(
        old_prototypes,
        new_arrays[
            "prototypes"
        ],
        "gate.prototypes",
    )

    assert_array_equal(
        old_scaler.mean_.astype(
            np.float64
        ),
        new_arrays[
            "scaler_mean"
        ],
        "gate.scaler_mean",
    )

    assert_array_equal(
        old_scaler.scale_.astype(
            np.float64
        ),
        new_arrays[
            "scaler_scale"
        ],
        "gate.scaler_scale",
    )

    assert_array_equal(
        old_classifier.coef_[0].astype(
            np.float64
        ),
        new_arrays[
            "coef"
        ],
        "gate.coef",
    )

    assert_array_equal(
        np.asarray(
            [
                old_classifier.intercept_[0]
            ],
            dtype=np.float64,
        ),
        new_arrays[
            "intercept"
        ],
        "gate.intercept",
    )

    assert_array_equal(
        np.asarray(
            [
                old_gate_threshold
            ],
            dtype=np.float64,
        ),
        new_arrays[
            "decision_threshold"
        ],
        "gate.decision_threshold",
    )

    assert (
        old_known_accept
        == new_known_accept
    )

    assert (
        old_unknown_reject
        == new_unknown_reject
    )

    print(
        "[OK] StandardScaler + balanced LogisticRegression gate fit matches exactly"
    )

    # Metadata contract.
    if not CHECKPOINT.is_file():
        raise SystemExit(
            "[ERROR] Missing production checkpoint: "
            + str(
                CHECKPOINT
            )
        )

    checkpoint = read_checkpoint(
        CHECKPOINT
    )

    with tempfile.TemporaryDirectory() as temporary:
        external = Path(
            temporary
        ) / "external"

        meta = build_metadata(
            checkpoint_path=CHECKPOINT,
            checkpoint=checkpoint,
            external_root=external,
            known_total=81,
            prototype_count=60,
            known_calibration_count=21,
            unknown_calibration_count=22,
            threshold=0.700972028801349,
            known_accept=0.9047619047619048,
            unknown_reject=1.0,
        )

    assert (
        meta["version"]
        == "1.1-external-calibration"
    )

    assert (
        meta["checkpoint_epoch"]
        == 9
    )

    assert (
        meta["known_total_images"]
        == 81
    )

    assert (
        meta["prototype_images"]
        == 60
    )

    assert (
        meta["known_calibration_images"]
        == 21
    )

    assert (
        meta["unknown_calibration_images"]
        == 22
    )

    assert (
        meta["chosen_threshold"]
        == 0.700972028801349
    )

    assert (
        meta["known_validation_accept_rate"]
        == 0.9047619047619048
    )

    print(
        "[OK] Production gate 1.1 metadata contract is preserved"
    )

    print(
        "[PASS] Stage G2 gate-builder refactor is behavior-equivalent "
        "to FreshLens_Buoc5_FIX/BUILD_OPENSET_GATE.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
