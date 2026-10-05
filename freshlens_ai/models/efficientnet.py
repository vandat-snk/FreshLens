"""EfficientNet-B0 construction and two-stage training behavior for FreshLens."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from freshlens_ai.constants import CLASSES
from freshlens_ai.errors import DataError


def choose_device(name):
    """Resolve the requested runtime device using the legacy FreshLens rules."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DataError(
                "CUDA is not available. Run CHECK_GPU.py and install "
                "the CUDA PyTorch build first."
            )
        return torch.device("cuda")

    if name == "cpu":
        return torch.device("cpu")

    raise DataError("Device must be cuda or cpu.")


def make_model(pretrained=True, cache_dir=None):
    """Build the exact EfficientNet-B0 classifier used by FreshLens V1."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))

    weights = (
        EfficientNet_B0_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = efficientnet_b0(weights=weights)

    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(
            model.classifier[1].in_features,
            len(CLASSES),
        ),
    )

    return model


def configure_stage(model, stage):
    """Configure trainable parameters for warmup or fine-tuning."""
    if stage not in ("warmup", "finetune"):
        raise DataError("Unknown training stage.")

    for parameter in model.features.parameters():
        parameter.requires_grad_(stage == "finetune")

    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)

    training_mode(model, stage)


def training_mode(model, stage):
    """Apply the exact module train/eval behavior used by the legacy trainer."""
    model.train()

    if stage == "warmup":
        model.features.eval()

    else:
        # Batch size 8 is small. Preserve pretrained BatchNorm running stats.
        # Affine BatchNorm parameters remain trainable during fine-tuning.
        for module in model.features.modules():
            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                module.eval()
