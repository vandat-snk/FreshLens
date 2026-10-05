"""FreshLens checkpoint contract, atomic persistence, and model loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from freshlens_ai.constants import (
    CLASSES,
    FRUITS,
    PREPROCESS,
    STATUSES,
)
from freshlens_ai.errors import DataError
from freshlens_ai.models.efficientnet import make_model
from freshlens_ai.models.prediction import DECISION_RULE


FORMAT_VERSION = 1


def checkpoint_metadata():
    """Return metadata that must match every compatible FreshLens checkpoint."""
    return {
        "format_version": FORMAT_VERSION,
        "architecture": "efficientnet_b0",
        "classes": list(CLASSES),
        "fruits": list(FRUITS),
        "statuses": list(STATUSES),
        "preprocess": PREPROCESS,
        "decision_rule": DECISION_RULE,
    }


def cpu_tree(value):
    """Recursively clone tensors to CPU before checkpoint serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()

    if isinstance(value, dict):
        return {
            key: cpu_tree(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            cpu_tree(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            cpu_tree(item)
            for item in value
        )

    return value


def save_checkpoint(path, content):
    """Atomically save a FreshLens checkpoint."""
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
            "wb",
        ) as handle:
            torch.save(
                cpu_tree(content),
                handle,
            )

        os.replace(
            temporary,
            path,
        )

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path):
    """Read and validate the immutable FreshLens checkpoint contract."""
    content = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(content, dict):
        raise DataError(
            "This file is not a FreshLens CNN checkpoint."
        )

    for key, expected in checkpoint_metadata().items():
        if content.get(key) != expected:
            raise DataError(
                "Checkpoint metadata mismatch: " + key
            )

    return content


def load_model(path, device):
    """Load a validated checkpoint into the exact production architecture."""
    content = read_checkpoint(path)

    model = make_model(
        pretrained=False,
    )

    model.load_state_dict(
        content["model_state"],
        strict=True,
    )

    model.to(device).eval()

    return model, content
