"""Stage F smoke/contract test for modular training/evaluation entry points."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


def parser_defaults(module, training):
    parser = module.build_parser()

    if training:
        args = parser.parse_args(
            [
                "--root",
                "X",
            ]
        )
        return {
            "data": str(args.data),
            "output": str(args.output),
            "device": args.device,
            "batch_size": args.batch_size,
            "accumulation": args.accumulation,
            "warmup_epochs": args.warmup_epochs,
            "finetune_epochs": args.finetune_epochs,
            "patience": args.patience,
            "workers": args.workers,
            "seed": args.seed,
            "label_smoothing": args.label_smoothing,
            "resume": args.resume,
            "from_scratch": args.from_scratch,
        }

    args = parser.parse_args(
        [
            "--root",
            "X",
            "--output",
            "Y",
        ]
    )

    return {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": args.device,
        "batch_size": args.batch_size,
    }


def main():
    train_v2 = importlib.import_module(
        "TRAIN_CNN_V2"
    )

    eval_v2 = importlib.import_module(
        "EVALUATE_CNN_V2"
    )

    print(
        "[OK] TRAIN_CNN_V2 and EVALUATE_CNN_V2 import successfully"
    )

    train_defaults = parser_defaults(
        train_v2,
        training=True,
    )

    assert train_defaults["device"] == "cuda"
    assert train_defaults["batch_size"] == 8
    assert train_defaults["accumulation"] == 2
    assert train_defaults["warmup_epochs"] == 3
    assert train_defaults["finetune_epochs"] == 20
    assert train_defaults["patience"] == 6
    assert train_defaults["workers"] == 0
    assert train_defaults["seed"] == 42
    assert train_defaults["label_smoothing"] == 0.05
    assert train_defaults["resume"] is False
    assert train_defaults["from_scratch"] is False

    eval_defaults = parser_defaults(
        eval_v2,
        training=False,
    )

    assert eval_defaults["split"] == "test"
    assert eval_defaults["device"] == "cuda"
    assert eval_defaults["batch_size"] == 8

    print(
        "[OK] CLI defaults match the legacy training/evaluation commands"
    )

    for script in (
        "TRAIN_CNN_V2.py",
        "EVALUATE_CNN_V2.py",
    ):
        source = (
            PROJECT_DIR / script
        ).read_text(
            encoding="utf-8"
        )

        forbidden = (
            "from cnn_data import",
            "import cnn_data",
            "from cnn_model import",
            "import cnn_model",
            "from cnn_metrics import",
            "import cnn_metrics",
        )

        for marker in forbidden:
            assert marker not in source, (
                script + " still depends on legacy module: " + marker
            )

    print(
        "[OK] V2 entry points no longer import legacy cnn_* modules"
    )

    for script in (
        "TRAIN_CNN_V2.py",
        "EVALUATE_CNN_V2.py",
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / script),
                "--help",
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, (
            script
            + " --help failed:\n"
            + completed.stdout
            + completed.stderr
        )

    print(
        "[OK] Both V2 command-line entry points start and expose --help"
    )

    print(
        "[PASS] Stage F orchestration refactor smoke/contract test passed"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
