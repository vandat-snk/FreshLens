"""Prediction decoding shared by training evaluation and deployed inference."""

from __future__ import annotations

import numpy as np

from freshlens_ai.errors import DataError


DECISION_RULE = (
    "fruit_marginal_then_condition_within_selected_fruit_v1"
)


def decode_probabilities(probabilities):
    """Decode Nx8 joint probabilities using the deployed fruit-first rule."""
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 8
        or not np.isfinite(probabilities).all()
    ):
        raise DataError(
            "Expected finite Nx8 class probabilities."
        )

    if (
        (probabilities < 0).any()
        or not np.allclose(
            probabilities.sum(axis=1),
            1,
            atol=1e-5,
        )
    ):
        raise DataError(
            "Invalid class probability distribution."
        )

    pairs = probabilities.reshape(
        -1,
        4,
        2,
    )

    fruit_probabilities = pairs.sum(
        axis=2,
    )

    fruit = fruit_probabilities.argmax(
        axis=1,
    )

    selected = pairs[
        np.arange(len(pairs)),
        fruit,
    ]

    condition_probabilities = selected / np.maximum(
        selected.sum(
            axis=1,
            keepdims=True,
        ),
        1e-15,
    )

    condition = condition_probabilities.argmax(
        axis=1,
    )

    joint = fruit * 2 + condition

    return {
        "joint": joint,
        "fruit": fruit,
        "condition": condition,
        "fruit_probabilities": fruit_probabilities,
        "condition_probabilities": condition_probabilities,
        "raw_joint_argmax": probabilities.argmax(axis=1),
    }
