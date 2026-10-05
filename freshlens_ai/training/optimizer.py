"""Optimizer construction for FreshLens two-stage EfficientNet training."""

from __future__ import annotations

import torch

from freshlens_ai.models import configure_stage


def optimizer_for(model, stage):
    """Create the exact optimizer used by the legacy FreshLens trainer."""
    configure_stage(model, stage)

    if stage == "warmup":
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

    return torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": 1e-4,
            },
            {
                "params": model.classifier.parameters(),
                "lr": 3e-4,
            },
        ],
        weight_decay=1e-4,
    )
