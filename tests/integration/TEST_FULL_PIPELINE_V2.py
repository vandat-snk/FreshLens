"""Final integration test for the modular FreshLens CNN V2 pipeline.

Runs all refactor-equivalence tests and performs an end-to-end production
artifact smoke test without modifying the trained checkpoint or open-set gate.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from freshlens_ai.inference import analyze_bytes, load_gate, predict_image
from freshlens_ai.models import load_model


PROJECT_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "best.pt"
)

GATE_NPZ = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "open_set_gate.npz"
)

GATE_META = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "open_set_gate.json"
)


def run_test(name, command):
    print(
        f"\n{'=' * 72}\n"
        f"[RUN] {name}\n"
        f"{'=' * 72}",
        flush=True,
    )

    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code "
            f"{completed.returncode}"
        )


def synthetic_png_bytes():
    y = np.arange(
        240,
        dtype=np.uint16,
    )[:, None]

    x = np.arange(
        320,
        dtype=np.uint16,
    )[None, :]

    array = np.empty(
        (240, 320, 3),
        dtype=np.uint8,
    )

    array[..., 0] = (
        (x + y) % 256
    ).astype(np.uint8)

    array[..., 1] = (
        (2 * x + y) % 256
    ).astype(np.uint8)

    array[..., 2] = (
        (x + 3 * y) % 256
    ).astype(np.uint8)

    buffer = io.BytesIO()

    Image.fromarray(
        array
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def audit_modular_imports():
    """Audit real legacy imports/path hacks while allowing package-relative imports."""
    targets = [
        PROJECT_DIR / "TRAIN_CNN_V2.py",
        PROJECT_DIR / "EVALUATE_CNN_V2.py",
        PROJECT_DIR / "PREDICT_CNN_V2.py",
        PROJECT_DIR / "BUILD_OPENSET_GATE_V2.py",
        PROJECT_DIR / "APP_CNN_V2.py",
    ]

    targets.extend(
        sorted(
            (
                PROJECT_DIR
                / "freshlens_ai"
            ).rglob("*.py")
        )
    )

    forbidden_roots = {
        "cnn_data",
        "cnn_model",
        "cnn_metrics",
        "open_set",
    }

    violations = []

    for path in targets:
        source = path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            source,
            filename=str(path),
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]

                    if root in forbidden_roots:
                        violations.append(
                            (
                                path.relative_to(PROJECT_DIR),
                                f"import {alias.name}",
                            )
                        )

            elif isinstance(node, ast.ImportFrom):
                # IMPORTANT:
                # node.level > 0 means a valid package-relative import.
                # Example:
                #   from .open_set import analyze_bytes
                #
                # That is NOT the legacy top-level `open_set.py`.
                if node.level > 0:
                    continue

                module = node.module or ""
                root = module.split(".", 1)[0] if module else ""

                if root in forbidden_roots:
                    violations.append(
                        (
                            path.relative_to(PROJECT_DIR),
                            f"from {module} import ...",
                        )
                    )

            elif isinstance(node, ast.Call):
                # Detect an actual `sys.path.insert(...)` call,
                # rather than matching the words inside comments/docstrings.
                func = node.func

                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "insert"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "path"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "sys"
                ):
                    violations.append(
                        (
                            path.relative_to(PROJECT_DIR),
                            "sys.path.insert(...)",
                        )
                    )

    if violations:
        message = "\n".join(
            f"  {path}: {marker}"
            for path, marker in violations
        )

        raise RuntimeError(
            "Modular runtime still contains legacy runtime imports/path hacks:\n"
            + message
        )


def production_artifact_smoke():
    for path in (
        CHECKPOINT,
        GATE_NPZ,
        GATE_META,
    ):
        if not path.is_file():
            raise RuntimeError(
                "Missing production artifact: "
                + str(path)
            )

    meta = json.loads(
        GATE_META.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        meta.get("version")
        != "1.1-external-calibration"
    ):
        raise RuntimeError(
            "Unexpected open-set gate version: "
            + repr(
                meta.get("version")
            )
        )

    device = torch.device(
        "cpu"
    )

    model, checkpoint = load_model(
        CHECKPOINT,
        device,
    )

    gate = load_gate(
        GATE_NPZ,
        GATE_META,
        CHECKPOINT,
    )

    if (
        int(checkpoint["epoch"])
        != int(
            gate["meta"][
                "checkpoint_epoch"
            ]
        )
    ):
        raise RuntimeError(
            "Checkpoint epoch and gate checkpoint epoch differ."
        )

    if (
        gate["prototypes"].shape
        != (4, 4, 1280)
    ):
        raise RuntimeError(
            "Unexpected production prototype shape: "
            + repr(
                gate["prototypes"].shape
            )
        )

    image_bytes = synthetic_png_bytes()

    plain = predict_image(
        model,
        image_bytes,
        device,
    )

    opened = analyze_bytes(
        model,
        image_bytes,
        device,
        gate,
    )

    required_plain = {
        "fruit",
        "condition",
        "fruit_score",
        "condition_score_given_fruit",
        "fruit_scores",
        "joint_scores",
    }

    required_open = {
        "supported",
        "support_probability",
        "support_threshold",
        "fruit",
        "condition",
        "fruit_score",
        "condition_score_given_fruit",
        "fruit_scores",
        "joint_scores",
        "gate_detail",
    }

    if not required_plain.issubset(
        plain
    ):
        raise RuntimeError(
            "Plain predictor output contract is incomplete."
        )

    if not required_open.issubset(
        opened
    ):
        raise RuntimeError(
            "Open-set output contract is incomplete."
        )

    if (
        plain["fruit"]
        != opened["fruit"]
        or plain["condition"]
        != opened["condition"]
    ):
        raise RuntimeError(
            "Plain predictor and open-set runtime disagree "
            "on fruit/condition for the same model input."
        )

    return {
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "gate_version": gate[
            "meta"
        ]["version"],
        "prototype_shape": list(
            gate["prototypes"].shape
        ),
        "synthetic_prediction": {
            "fruit": opened[
                "fruit"
            ],
            "condition": opened[
                "condition"
            ],
            "supported": bool(
                opened[
                    "supported"
                ]
            ),
            "support_probability": float(
                opened[
                    "support_probability"
                ]
            ),
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete FreshLens V2 "
            "refactor/integration test suite"
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Original image root used by "
            "TEST_DATA_REFACTOR.py"
        ),
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

    return parser


def main():
    args = build_parser().parse_args()

    python = sys.executable

    tests = [
        (
            "Stage B - data",
            [
                python,
                "-m",
                "tests.refactor.TEST_DATA_REFACTOR",
                "--root",
                str(args.root),
                "--data",
                str(args.data),
            ],
        ),
        (
            "Stage C - model",
            [
                python,
                "-m",
                "tests.refactor.TEST_MODEL_REFACTOR",
            ],
        ),
        (
            "Stage D - training",
            [
                python,
                "-m",
                "tests.refactor.TEST_TRAINING_REFACTOR",
            ],
        ),
        (
            "Stage E - evaluation",
            [
                python,
                "-m",
                "tests.refactor.TEST_EVALUATION_REFACTOR",
            ],
        ),
        (
            "Stage F - orchestration",
            [
                python,
                "-m",
                "tests.refactor.TEST_ORCHESTRATION_REFACTOR",
            ],
        ),
        (
            "Stage G1 - inference runtime",
            [
                python,
                "-m",
                "tests.refactor.TEST_INFERENCE_REFACTOR",
            ],
        ),
        (
            "Stage G2 - gate builder",
            [
                python,
                "-m",
                "tests.refactor.TEST_GATE_BUILDER_REFACTOR",
            ],
        ),
        (
            "Stage G3 - predictor CLI",
            [
                python,
                "-m",
                "tests.refactor.TEST_PREDICT_ENTRYPOINT_REFACTOR",
            ],
        ),
        (
            "TV3 - shared runtime, evaluation and Grad-CAM",
            [python, "-m", "pytest", "tests/tv3", "-q"],
        ),
        (
            "Stage G4 - Streamlit app",
            [
                python,
                "-m",
                "tests.refactor.TEST_APP_REFACTOR",
            ],
        ),
    ]

    try:
        for name, command in tests:
            run_test(
                name,
                command,
            )

        print(
            f"\n{'=' * 72}\n"
            "[AUDIT] Modular import boundaries\n"
            f"{'=' * 72}",
            flush=True,
        )

        audit_modular_imports()

        print(
            "[OK] New V2 entry points and freshlens_ai "
            "contain no legacy runtime imports/path hacks"
        )

        print(
            f"\n{'=' * 72}\n"
            "[SMOKE] Production checkpoint + gate\n"
            f"{'=' * 72}",
            flush=True,
        )

        summary = production_artifact_smoke()

        print(
            "[OK] Production checkpoint loads"
        )

        print(
            "[OK] Production open-set gate loads "
            "and matches checkpoint"
        )

        print(
            "[OK] End-to-end CPU prediction + open-set analysis succeeds"
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "\n[PASS] FreshLens modular CNN V2 "
            "full integration suite passed"
        )

        print(
            "[OK] Repository-level V2 integration verification complete."
        )

        return 0

    except Exception as exc:
        print(
            f"\n[FAIL] "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
