"""Behavior-equivalence tests for Stage D FreshLens training refactor."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[2]
LEGACY_DIR = PROJECT_DIR / "legacy" / "development" / "step3_cnn"

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import TRAIN_CNN as old_train  # noqa: E402
import cnn_model as old_model_api  # noqa: E402

from freshlens_ai.models import make_model  # noqa: E402
from freshlens_ai.training import (  # noqa: E402
    optimizer_for,
    seed_everything,
    train_epoch,
)


def optimizer_signature(optimizer):
    groups = []

    for group in optimizer.param_groups:
        groups.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": tuple(group["betas"]),
                "eps": float(group["eps"]),
                "amsgrad": bool(group["amsgrad"]),
                "maximize": bool(group["maximize"]),
            }
        )

    return groups


def trainable_signature(model):
    return {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }


def state_dict_equal(left, right):
    assert left.keys() == right.keys()

    for key in left:
        if not torch.equal(left[key], right[key]):
            raise AssertionError(
                "Model state differs at: " + key
            )


def test_optimizer(stage):
    torch.manual_seed(1234)
    old_model = old_model_api.make_model(
        pretrained=False,
    )

    torch.manual_seed(1234)
    new_model = make_model(
        pretrained=False,
    )

    old_optimizer = old_train.optimizer_for(
        old_model,
        stage,
    )

    new_optimizer = optimizer_for(
        new_model,
        stage,
    )

    assert (
        trainable_signature(old_model)
        == trainable_signature(new_model)
    )

    assert (
        optimizer_signature(old_optimizer)
        == optimizer_signature(new_optimizer)
    )


def test_seed():
    old_train.seed_everything(2026)

    old_values = (
        random.random(),
        float(np.random.rand()),
        torch.rand(5),
        bool(torch.backends.cudnn.benchmark),
        bool(torch.backends.cudnn.deterministic),
    )

    seed_everything(2026)

    new_values = (
        random.random(),
        float(np.random.rand()),
        torch.rand(5),
        bool(torch.backends.cudnn.benchmark),
        bool(torch.backends.cudnn.deterministic),
    )

    assert old_values[0] == new_values[0]
    assert old_values[1] == new_values[1]
    assert torch.equal(
        old_values[2],
        new_values[2],
    )
    assert old_values[3:] == new_values[3:]


def make_loader():
    generator = torch.Generator().manual_seed(9876)

    images = torch.randn(
        4,
        3,
        64,
        64,
        generator=generator,
    )

    targets = torch.tensor(
        [0, 1, 6, 7],
        dtype=torch.long,
    )

    indices = torch.arange(
        4,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        images,
        targets,
        indices,
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )


def test_train_epoch():
    device = torch.device("cpu")

    torch.manual_seed(2468)
    old_model = old_model_api.make_model(
        pretrained=False,
    ).to(device)

    torch.manual_seed(2468)
    new_model = make_model(
        pretrained=False,
    ).to(device)

    old_optimizer = old_train.optimizer_for(
        old_model,
        "warmup",
    )

    new_optimizer = optimizer_for(
        new_model,
        "warmup",
    )

    old_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    new_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    old_loader = make_loader()
    new_loader = make_loader()

    # Dropout is stochastic during classifier training, so reset
    # the same RNG state before each implementation.
    torch.manual_seed(777)

    old_loss = old_train.train_epoch(
        old_model,
        old_loader,
        old_optimizer,
        old_scaler,
        device,
        "warmup",
        2,
        0.05,
    )

    torch.manual_seed(777)

    new_loss = train_epoch(
        new_model,
        new_loader,
        new_optimizer,
        new_scaler,
        device,
        "warmup",
        2,
        0.05,
    )

    assert old_loss == new_loss

    state_dict_equal(
        old_model.state_dict(),
        new_model.state_dict(),
    )


def main():
    test_optimizer("warmup")
    test_optimizer("finetune")

    print(
        "[OK] Warmup and finetune optimizer configuration match"
    )

    test_seed()

    print(
        "[OK] Random/NumPy/Torch seeding and cuDNN flags match"
    )

    test_train_epoch()

    print(
        "[OK] One CPU warmup epoch matches loss and updated model state"
    )

    print(
        "[PASS] Stage D training refactor is behavior-equivalent "
        "to legacy TRAIN_CNN.py helpers"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
