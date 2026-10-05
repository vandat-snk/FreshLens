"""Single-image FreshLens CNN inference shared by CLI and app."""

from __future__ import annotations

import torch

from freshlens_ai.constants import CLASSES, FRUITS, STATUSES
from freshlens_ai.data import image_transform, rgb_from_bytes
from freshlens_ai.models import decode_probabilities


@torch.inference_mode()
def predict_image(model, image_bytes, device):
    """Preserve the exact prediction contract of legacy PREDICT_CNN.py."""
    model.eval()

    image = rgb_from_bytes(image_bytes)

    tensor = (
        image_transform(training=False)(image)
        .unsqueeze(0)
        .to(device)
    )

    logits = model(tensor)

    probabilities = (
        logits.float()
        .softmax(dim=1)
        .cpu()
        .numpy()
    )

    decoded = decode_probabilities(
        probabilities
    )

    fruit = int(
        decoded["fruit"][0]
    )

    condition = int(
        decoded["condition"][0]
    )

    return {
        "fruit": FRUITS[fruit],
        "condition": STATUSES[condition],
        "fruit_score": float(
            decoded["fruit_probabilities"][
                0,
                fruit,
            ]
        ),
        "condition_score_given_fruit": float(
            decoded[
                "condition_probabilities"
            ][
                0,
                condition,
            ]
        ),
        "fruit_scores": {
            name: float(
                decoded["fruit_probabilities"][
                    0,
                    i,
                ]
            )
            for i, name in enumerate(FRUITS)
        },
        "condition_scores_given_fruit": {
            name: float(
                decoded[
                    "condition_probabilities"
                ][
                    0,
                    i,
                ]
            )
            for i, name in enumerate(STATUSES)
        },
        "joint_scores": {
            name: float(
                probabilities[
                    0,
                    i,
                ]
            )
            for i, name in enumerate(CLASSES)
        },
        "scope": (
            "One main fruit; apple/banana/orange/tomato only. "
            "Scores are uncalibrated model outputs, not measured accuracy."
        ),
        "out_of_scope_detection_trained": False,
    }


predict_bytes = predict_image
