"""Behavior-equivalence tests for the Stage C FreshLens model refactor."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = PROJECT_DIR / "FreshLens_Buoc3_CNN"

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import cnn_model as old  # noqa: E402

from freshlens_ai.models import (  # noqa: E402
    checkpoint_metadata,
    configure_stage,
    decode_probabilities,
    load_model,
    make_model,
    read_checkpoint,
    save_checkpoint,
)


def assert_state_dict_equal(left, right):
    assert left.keys() == right.keys()

    for key in left:
        assert torch.equal(
            left[key],
            right[key],
        ), key


def batchnorm_training_flags(model):
    return [
        module.training
        for module in model.features.modules()
        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        )
    ]


def trainable_signature(model):
    return {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }


def compare_nested(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
        return

    if isinstance(left, dict):
        assert isinstance(right, dict), path
        assert left.keys() == right.keys(), path
        for key in left:
            compare_nested(
                left[key],
                right[key],
                f"{path}.{key}",
            )
        return

    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)), path
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare_nested(
                a,
                b,
                f"{path}[{index}]",
            )
        return

    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        assert np.array_equal(left, right), path
        return

    assert left == right, path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR
        / "models"
        / "cnn_efficientnet_b0"
        / "best.pt",
    )

    args = parser.parse_args()

    # Architecture and initialization.
    torch.manual_seed(123456)
    old_model = old.make_model(
        pretrained=False,
    )

    torch.manual_seed(123456)
    new_model = make_model(
        pretrained=False,
    )

    assert str(old_model.classifier) == str(new_model.classifier)
    assert_state_dict_equal(
        old_model.state_dict(),
        new_model.state_dict(),
    )

    sample = torch.randn(
        2,
        3,
        224,
        224,
    )

    old_model.eval()
    new_model.eval()

    with torch.inference_mode():
        old_logits = old_model(sample)
        new_logits = new_model(sample)

    assert torch.equal(
        old_logits,
        new_logits,
    )

    print(
        "[OK] EfficientNet-B0 architecture, initialization and forward pass match"
    )

    # Warmup/fine-tune behavior.
    for stage in (
        "warmup",
        "finetune",
    ):
        torch.manual_seed(7)
        old_stage_model = old.make_model(
            pretrained=False,
        )

        torch.manual_seed(7)
        new_stage_model = make_model(
            pretrained=False,
        )

        old.configure_stage(
            old_stage_model,
            stage,
        )

        configure_stage(
            new_stage_model,
            stage,
        )

        assert (
            trainable_signature(old_stage_model)
            == trainable_signature(new_stage_model)
        )

        assert (
            old_stage_model.training
            == new_stage_model.training
        )

        assert (
            old_stage_model.features.training
            == new_stage_model.features.training
        )

        assert (
            batchnorm_training_flags(old_stage_model)
            == batchnorm_training_flags(new_stage_model)
        )

    print(
        "[OK] Warmup/fine-tune requires_grad and BatchNorm behavior match"
    )

    # Deployed decision rule.
    rng = np.random.default_rng(42)
    raw = rng.random(
        (25, 8),
    )
    probabilities = raw / raw.sum(
        axis=1,
        keepdims=True,
    )

    old_decoded = old.decode_probabilities(
        probabilities,
    )

    new_decoded = decode_probabilities(
        probabilities,
    )

    compare_nested(
        old_decoded,
        new_decoded,
        "decode_probabilities",
    )

    print(
        "[OK] Fruit-first deployed decision rule matches"
    )

    # Checkpoint metadata contract.
    assert (
        old.checkpoint_metadata()
        == checkpoint_metadata()
    )

    print(
        "[OK] Checkpoint metadata contract matches"
    )

    # Atomic checkpoint save/read behavior.
    content = {
        **checkpoint_metadata(),
        "model_state": new_model.state_dict(),
        "epoch": 123,
        "custom": {
            "tensor": torch.arange(5),
            "tuple": (1, 2, 3),
        },
    }

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "test.pt"

        save_checkpoint(
            path,
            content,
        )

        restored = read_checkpoint(
            path,
        )

        compare_nested(
            content,
            restored,
            "roundtrip_checkpoint",
        )

    print(
        "[OK] New checkpoint save/read round-trip passes"
    )

    # Production checkpoint compatibility.
    if args.checkpoint.is_file():
        old_checkpoint = old.read_checkpoint(
            args.checkpoint,
        )

        new_checkpoint = read_checkpoint(
            args.checkpoint,
        )

        assert (
            old_checkpoint.keys()
            == new_checkpoint.keys()
        )

        assert_state_dict_equal(
            old_checkpoint["model_state"],
            new_checkpoint["model_state"],
        )

        cpu = torch.device("cpu")

        old_loaded_model, old_metadata = old.load_model(
            args.checkpoint,
            cpu,
        )

        new_loaded_model, new_metadata = load_model(
            args.checkpoint,
            cpu,
        )

        assert old_metadata.keys() == new_metadata.keys()

        with torch.inference_mode():
            old_output = old_loaded_model(sample)
            new_output = new_loaded_model(sample)

        assert torch.equal(
            old_output,
            new_output,
        )

        print(
            "[OK] Existing production best.pt is compatible with refactored loader"
        )

    else:
        print(
            "[SKIP] Production checkpoint not found: "
            + str(args.checkpoint)
        )

    print(
        "[PASS] Stage C model refactor is behavior-equivalent to legacy cnn_model.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
