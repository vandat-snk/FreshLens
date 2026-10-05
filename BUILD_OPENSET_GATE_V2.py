"""Build FreshLens open-set gate 1.1 through the modular freshlens_ai package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from freshlens_ai.errors import DataError
from freshlens_ai.inference import (
    build_metadata,
    build_prototypes,
    collect_images,
    collect_known,
    deterministic_split,
    fit_gate,
    infer_paths,
)
from freshlens_ai.models import (
    choose_device,
    load_model,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build FreshLens open-set gate "
            "without original training images"
        )
    )

    parser.add_argument(
        "--external",
        type=Path,
        default=Path(
            r"E:\VanDat_\XuLyAnh\FreshLens_external_test_v2"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/cnn_efficientnet_b0/best.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/cnn_efficientnet_b0/open_set_gate.npz"
        ),
    )

    parser.add_argument(
        "--meta",
        type=Path,
        default=Path(
            "models/cnn_efficientnet_b0/open_set_gate.json"
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "cuda",
            "cpu",
        ),
        default="cuda",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-known-accept",
        type=float,
        default=0.90,
    )

    return parser


def run(args):
    known_rows = collect_known(
        args.external
    )

    unknown_paths = collect_images(
        args.external
        / "unknown"
    )

    if len(
        unknown_paths
    ) < 10:
        raise DataError(
            "Can it nhat 10 anh unknown; "
            f"hien co {len(unknown_paths)} tai "
            f"{args.external / 'unknown'}"
        )

    (
        proto_rows,
        calib_rows,
    ) = deterministic_split(
        known_rows
    )

    device = choose_device(
        args.device
    )

    model, checkpoint = load_model(
        args.checkpoint,
        device,
    )

    print(
        f"[OK] Device: {device}; "
        f"checkpoint epoch={checkpoint['epoch']}"
    )

    print(
        f"[DATA] "
        f"known_total={len(known_rows)} "
        f"prototype={len(proto_rows)} "
        f"known_calibration={len(calib_rows)} "
        f"unknown_calibration={len(unknown_paths)}"
    )

    proto_paths = [
        row[0]
        for row in proto_rows
    ]

    proto_targets = [
        row[1]
        for row in proto_rows
    ]

    proto_emb, _ = infer_paths(
        model,
        device,
        proto_paths,
        args.batch,
    )

    prototypes = build_prototypes(
        proto_emb,
        proto_targets,
    )

    known_emb, known_probs = infer_paths(
        model,
        device,
        [
            row[0]
            for row in calib_rows
        ],
        args.batch,
    )

    (
        unknown_emb,
        unknown_probs,
    ) = infer_paths(
        model,
        device,
        unknown_paths,
        args.batch,
    )

    (
        gate_arrays,
        known_accept,
        unknown_reject,
    ) = fit_gate(
        known_emb,
        known_probs,
        unknown_emb,
        unknown_probs,
        prototypes,
        args.min_known_accept,
    )

    threshold = float(
        gate_arrays[
            "decision_threshold"
        ][0]
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        **gate_arrays,
    )

    meta = build_metadata(
        checkpoint_path=(
            args.checkpoint
        ),
        checkpoint=checkpoint,
        external_root=(
            args.external
        ),
        known_total=(
            len(known_rows)
        ),
        prototype_count=(
            len(proto_rows)
        ),
        known_calibration_count=(
            len(calib_rows)
        ),
        unknown_calibration_count=(
            len(unknown_paths)
        ),
        threshold=threshold,
        known_accept=known_accept,
        unknown_reject=unknown_reject,
    )

    args.meta.write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n[OK] Open-set gate created"
    )

    print(
        f"[OK] Known accept: "
        f"{known_accept:.1%}"
    )

    print(
        f"[OK] Unknown reject: "
        f"{unknown_reject:.1%}"
    )

    print(
        f"[OK] Threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"[OK] Saved: "
        f"{args.output}"
    )

    print(
        f"[OK] Saved: "
        f"{args.meta}"
    )


def main():
    args = build_parser().parse_args()

    try:
        run(
            args
        )

    except (
        DataError,
        OSError,
        ValueError,
    ) as exc:
        build_parser().exit(
            1,
            f"[ERROR] "
            f"{type(exc).__name__}: "
            f"{exc}\n",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
