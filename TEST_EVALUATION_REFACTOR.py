"""Behavior-equivalence tests for Stage E FreshLens evaluation refactor."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = PROJECT_DIR / "FreshLens_Buoc3_CNN"

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import cnn_metrics as old  # noqa: E402

from freshlens_ai.evaluation import (  # noqa: E402
    calculate_metrics,
    classification_metrics,
    evaluate_model,
    plot_confusion,
    plot_history,
    save_predictions,
)


def compare_nested(left, right, path="root"):
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

    if isinstance(left, list):
        assert isinstance(right, list), path
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(
            zip(left, right)
        ):
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

    if isinstance(left, float):
        assert isinstance(right, (float, int)), path
        assert np.isclose(
            left,
            right,
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ), path
        return

    assert left == right, path


class TinyEvalDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [
                [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0],
            ],
            dtype=torch.float32,
        )

        self.targets = torch.arange(
            8,
            dtype=torch.long,
        )

        self.rows = [
            {
                "path": f"sample_{i}.jpg",
                "group_id": f"group_{i // 2}",
                "fruit": (
                    "apple",
                    "apple",
                    "banana",
                    "banana",
                    "orange",
                    "orange",
                    "tomato",
                    "tomato",
                )[i],
                "status": (
                    "fresh",
                    "rotten",
                )[i % 2],
                "target": i,
            }
            for i in range(8)
        ]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return (
            self.images[index],
            self.targets[index],
            index,
        )


class IdentityLogitModel(torch.nn.Module):
    def forward(self, x):
        return x


def main():
    truth = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7],
        dtype=np.int64,
    )

    prediction = np.array(
        [0, 1, 2, 2, 4, 5, 7, 7],
        dtype=np.int64,
    )

    old_basic = old.classification_metrics(
        truth,
        prediction,
        old.CLASSES,
    )

    new_basic = classification_metrics(
        truth,
        prediction,
        old.CLASSES,
    )

    compare_nested(
        old_basic,
        new_basic,
        "classification_metrics",
    )

    print(
        "[OK] classification_metrics matches"
    )

    rng = np.random.default_rng(2026)
    raw = rng.random(
        (8, 8)
    )

    probabilities = raw / raw.sum(
        axis=1,
        keepdims=True,
    )

    groups = [
        "g0",
        "g0",
        "g1",
        "g1",
        "g2",
        "g2",
        "g3",
        "g3",
    ]

    old_metrics = old.calculate_metrics(
        truth,
        probabilities,
        groups,
    )

    new_metrics = calculate_metrics(
        truth,
        probabilities,
        groups,
    )

    compare_nested(
        old_metrics,
        new_metrics,
        "calculate_metrics",
    )

    print(
        "[OK] calculate_metrics matches"
    )

    dataset = TinyEvalDataset()
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    model = IdentityLogitModel()
    device = torch.device("cpu")

    old_eval, old_probs = old.evaluate_model(
        model,
        loader,
        device,
        0.05,
    )

    new_eval, new_probs = evaluate_model(
        model,
        loader,
        device,
        0.05,
    )

    compare_nested(
        old_eval,
        new_eval,
        "evaluate_model.metrics",
    )

    assert np.array_equal(
        old_probs,
        new_probs,
    )

    print(
        "[OK] evaluate_model metrics and probabilities match"
    )

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)

        old_dir = temp / "old"
        new_dir = temp / "new"
        old_dir.mkdir()
        new_dir.mkdir()

        old.save_predictions(
            old_dir,
            "validation",
            dataset.rows,
            old_probs,
        )

        save_predictions(
            new_dir,
            "validation",
            dataset.rows,
            new_probs,
        )

        for filename in (
            "validation_predictions.csv",
            "validation_errors.csv",
        ):
            assert (
                (old_dir / filename).read_bytes()
                == (new_dir / filename).read_bytes()
            ), filename

        print(
            "[OK] Prediction/error CSV exports match byte-for-byte"
        )

        old.plot_confusion(
            old_dir / "confusion.png",
            old_eval,
            "Test confusion",
        )

        plot_confusion(
            new_dir / "confusion.png",
            new_eval,
            "Test confusion",
        )

        history = [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 0.8,
                "val_fruit_accuracy": 0.8,
                "val_joint_accuracy": 0.7,
                "val_joint_macro_f1": 0.69,
            },
            {
                "epoch": 2,
                "train_loss": 0.7,
                "val_loss": 0.6,
                "val_fruit_accuracy": 0.9,
                "val_joint_accuracy": 0.8,
                "val_joint_macro_f1": 0.79,
            },
        ]

        old.plot_history(
            old_dir / "history.png",
            history,
        )

        plot_history(
            new_dir / "history.png",
            history,
        )

        for path in (
            old_dir / "confusion.png",
            new_dir / "confusion.png",
            old_dir / "history.png",
            new_dir / "history.png",
        ):
            assert path.is_file()
            assert path.stat().st_size > 0

        print(
            "[OK] Confusion matrix and history plot generation both succeed"
        )

    print(
        "[PASS] Stage E evaluation refactor is behavior-equivalent "
        "to legacy cnn_metrics.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
