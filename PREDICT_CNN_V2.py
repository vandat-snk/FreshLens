"""FreshLens V2 single-image prediction CLI.

The prediction implementation lives in freshlens_ai.inference. This file is
only a command-line entry point and intentionally preserves the legacy
PREDICT_CNN.py output contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference import predict_image
from freshlens_ai.models import choose_device, load_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict fruit and condition from a photo"
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
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
        "--device",
        choices=("cuda", "cpu"),
        default="cpu",
    )

    return parser


def run(args):
    device = choose_device(
        args.device
    )

    model, metadata = load_model(
        args.checkpoint,
        device,
    )

    result = predict_image(
        model,
        args.image.read_bytes(),
        device,
    )

    result["checkpoint_epoch"] = metadata[
        "epoch"
    ]

    return result


def main():
    args = build_parser().parse_args()

    try:
        result = run(
            args
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
