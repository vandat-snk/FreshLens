"""Model, checkpoint contract and one prediction rule used everywhere."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from cnn_data import CLASSES, FRUITS, STATUSES, PREPROCESS, DataError


FORMAT_VERSION = 1
DECISION_RULE = "fruit_marginal_then_condition_within_selected_fruit_v1"


def choose_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DataError("CUDA is not available. Run CHECK_GPU.py and install the CUDA PyTorch build first.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise DataError("Device must be cuda or cpu.")


def make_model(pretrained=True, cache_dir=None):
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(.25), nn.Linear(model.classifier[1].in_features, len(CLASSES)))
    return model


def configure_stage(model, stage):
    if stage not in ("warmup", "finetune"):
        raise DataError("Unknown training stage.")
    for parameter in model.features.parameters():
        parameter.requires_grad_(stage == "finetune")
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    training_mode(model, stage)


def training_mode(model, stage):
    model.train()
    if stage == "warmup":
        model.features.eval()
    else:
        # Batch size 8 is small. Use pretrained running statistics in BatchNorm.
        # Affine parameters remain trainable during fine-tuning.
        for module in model.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def decode_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 8 or not np.isfinite(probabilities).all():
        raise DataError("Expected finite Nx8 class probabilities.")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise DataError("Invalid class probability distribution.")
    pairs = probabilities.reshape(-1, 4, 2)
    fruit_probabilities = pairs.sum(axis=2)
    fruit = fruit_probabilities.argmax(axis=1)
    selected = pairs[np.arange(len(pairs)), fruit]
    condition_probabilities = selected / np.maximum(selected.sum(axis=1, keepdims=True), 1e-15)
    condition = condition_probabilities.argmax(axis=1)
    joint = fruit * 2 + condition
    return {
        "joint": joint, "fruit": fruit, "condition": condition,
        "fruit_probabilities": fruit_probabilities,
        "condition_probabilities": condition_probabilities,
        "raw_joint_argmax": probabilities.argmax(axis=1),
    }


def checkpoint_metadata():
    return {"format_version": FORMAT_VERSION, "architecture": "efficientnet_b0",
            "classes": list(CLASSES), "fruits": list(FRUITS), "statuses": list(STATUSES),
            "preprocess": PREPROCESS, "decision_rule": DECISION_RULE}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(cpu_tree(content), handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path):
    content = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(content, dict):
        raise DataError("This file is not a FreshLens CNN checkpoint.")
    for key, expected in checkpoint_metadata().items():
        if content.get(key) != expected:
            raise DataError("Checkpoint metadata mismatch: " + key)
    return content


def load_model(path, device):
    content = read_checkpoint(path)
    model = make_model(pretrained=False)
    model.load_state_dict(content["model_state"], strict=True)
    model.to(device).eval()
    return model, content
