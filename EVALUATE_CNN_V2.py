"""FreshLens CNN V2 explicit evaluation entry point.

TRAIN_CNN_V2 never evaluates the reserved test split. This command must be
run explicitly after selecting a checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.data import FruitDataset, load_locked_dataset, verify_images
from freshlens_ai.errors import DataError
from freshlens_ai.evaluation import (
    evaluate_model,
    plot_confusion,
    save_predictions,
)
from freshlens_ai.models import choose_device, load_model
from freshlens_ai.utils.file_io import atomic_json, file_sha256


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a chosen CNN checkpoint "
            "on a full fixed split"
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            PROJECT_DIR
            / "data"
            / "cnn_dataset_v3"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_DIR
            / "models"
            / "cnn_efficientnet_b0"
            / "best.pt"
        ),
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    return parser


def main():
    args = build_parser().parse_args()

    try:
        if args.output.exists():
            raise DataError(
                "Evaluation output exists. Preserve it and choose "
                "a new directory if a rerun is needed."
            )

        if args.batch_size < 1:
            raise DataError(
                "Batch size must be positive."
            )

        rows, identity = load_locked_dataset(
            args.data
        )

        device = choose_device(
            args.device
        )

        model, checkpoint = load_model(
            args.checkpoint,
            device,
        )

        if (
            checkpoint["dataset_identity"]
            != identity
        ):
            raise DataError(
                "Checkpoint and evaluation dataset versions differ."
            )

        selected = [
            row
            for row in rows
            if row["split"] == args.split
        ]

        verify_images(
            args.root,
            selected,
        )

        loader = DataLoader(
            FruitDataset(
                args.root,
                selected,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        metrics, probabilities = evaluate_model(
            model,
            loader,
            device,
            checkpoint["config"]["label_smoothing"],
        )

        args.output.mkdir(
            parents=True
        )

        report = {
            "split": args.split,
            "checkpoint_sha256": file_sha256(
                args.checkpoint
            ),
            "checkpoint_epoch": checkpoint["epoch"],
            "dataset_identity": identity,
            "metrics": metrics,
            "scores_are_from_internal_dataset": True,
            "camera_generalization_measured": False,
        }

        atomic_json(
            args.output / "metrics.json",
            report,
        )

        save_predictions(
            args.output,
            args.split,
            selected,
            probabilities,
        )

        plot_confusion(
            args.output / "confusion_matrix.png",
            metrics,
            f"{args.split.title()}: fruit + condition",
        )

        print(
            f"[OK] {args.split}: "
            f"fruit accuracy={metrics['fruit']['accuracy']:.2%}; "
            f"joint accuracy={metrics['joint']['accuracy']:.2%}; "
            f"macro-F1={metrics['joint']['macro_f1']:.4f}"
        )

        print(
            f"[OK] Saved: {args.output}"
        )

    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
