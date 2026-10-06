"""Build the FreshLens 1.1 external-calibration open-set gate.

This module preserves the behavior of legacy/development/step5_fix/BUILD_OPENSET_GATE.py
while depending only on the modular freshlens_ai package.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from freshlens_ai.constants import FRUITS
from freshlens_ai.data import image_transform, rgb_from_bytes
from freshlens_ai.errors import DataError
from freshlens_ai.inference.open_set import (
    embedding_and_probabilities,
    gate_features,
)
from freshlens_ai.utils.file_io import file_sha256


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

KNOWN_FOLDERS = {
    "apple_fresh": 0,
    "apple_rotten": 0,
    "banana_fresh": 1,
    "banana_rotten": 1,
    "orange_fresh": 2,
    "orange_rotten": 2,
    "tomato_fresh": 3,
    "tomato_rotten": 3,
}


def collect_images(folder: Path):
    folder = Path(folder)

    if not folder.is_dir():
        return []

    result = []
    seen = set()

    for path in sorted(
        folder.rglob("*")
    ):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        data = path.read_bytes()

        digest = hashlib.sha256(
            data
        ).hexdigest()

        if digest in seen:
            continue

        try:
            rgb_from_bytes(
                data
            )
        except Exception:
            continue

        seen.add(
            digest
        )

        result.append(
            path
        )

    return result


def collect_known(root: Path):
    root = Path(root)
    rows = []

    for (
        folder_name,
        fruit_idx,
    ) in KNOWN_FOLDERS.items():
        files = collect_images(
            root / folder_name
        )

        if len(files) < 5:
            raise DataError(
                f"Can it nhat 5 anh trong "
                f"{root / folder_name}; "
                f"hien co {len(files)}"
            )

        for path in files:
            rows.append(
                (
                    path,
                    fruit_idx,
                    folder_name,
                )
            )

    return rows


def deterministic_split(rows):
    """Use ~75% known images for prototypes and ~25% for gate calibration."""
    proto = []
    calib = []

    by_fruit = {
        i: []
        for i in range(
            len(FRUITS)
        )
    }

    for row in rows:
        by_fruit[
            row[1]
        ].append(
            row
        )

    for (
        fruit_idx,
        items,
    ) in by_fruit.items():
        items = sorted(
            items,
            key=lambda x: hashlib.sha256(
                str(
                    x[0]
                ).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )

        for i, row in enumerate(
            items
        ):
            (
                calib
                if i % 4 == 0
                else proto
            ).append(
                row
            )

        if (
            sum(
                r[1] == fruit_idx
                for r in proto
            )
            < 3
            or sum(
                r[1] == fruit_idx
                for r in calib
            )
            < 1
        ):
            raise DataError(
                "Khong du anh de tach "
                "prototype/calibration cho "
                f"{FRUITS[fruit_idx]}"
            )

    return proto, calib


@torch.inference_mode()
def infer_paths(
    model,
    device,
    paths,
    batch_size,
):
    transform = image_transform(
        training=False
    )

    embeddings = []
    probabilities = []

    for start in range(
        0,
        len(paths),
        batch_size,
    ):
        chunk = paths[
            start:
            start + batch_size
        ]

        tensors = [
            transform(
                rgb_from_bytes(
                    path.read_bytes()
                )
            )
            for path in chunk
        ]

        images = torch.stack(
            tensors
        ).to(
            device
        )

        emb, probs = (
            embedding_and_probabilities(
                model,
                images,
            )
        )

        embeddings.append(
            emb.cpu().numpy()
        )

        probabilities.append(
            probs.cpu().numpy()
        )

        print(
            f"[EMBED] "
            f"{min(start + len(chunk), len(paths))}"
            f"/{len(paths)}",
            flush=True,
        )

    if not paths:
        return (
            np.empty(
                (0, 1280),
                np.float32,
            ),
            np.empty(
                (0, 8),
                np.float32,
            ),
        )

    return (
        np.concatenate(
            embeddings
        ),
        np.concatenate(
            probabilities
        ),
    )


def normalize(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    return values / np.maximum(
        np.linalg.norm(
            values,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )


def build_prototypes(
    embeddings,
    fruit_targets,
    prototypes_per_fruit=4,
):
    embeddings = normalize(
        embeddings
    )

    fruit_targets = np.asarray(
        fruit_targets,
        dtype=np.int64,
    )

    result = []

    for (
        fruit_idx,
        fruit,
    ) in enumerate(
        FRUITS
    ):
        selected = embeddings[
            fruit_targets
            == fruit_idx
        ]

        clusters = min(
            prototypes_per_fruit,
            len(selected),
        )

        if clusters < 1:
            raise DataError(
                "Khong co embedding "
                f"prototype cho {fruit}"
            )

        if clusters == 1:
            centers = normalize(
                selected.mean(
                    axis=0,
                    keepdims=True,
                )
            )

        else:
            km = MiniBatchKMeans(
                n_clusters=clusters,
                random_state=42,
                batch_size=64,
                n_init=10,
            )

            km.fit(
                selected
            )

            centers = normalize(
                km.cluster_centers_
            )

        if (
            clusters
            < prototypes_per_fruit
        ):
            centers = np.concatenate(
                [
                    centers,
                    np.repeat(
                        centers[-1:],
                        (
                            prototypes_per_fruit
                            - clusters
                        ),
                        axis=0,
                    ),
                ],
                axis=0,
            )

        result.append(
            centers[
                :prototypes_per_fruit
            ]
        )

        print(
            f"[PROTO] {fruit}: "
            f"{len(selected)} images -> "
            f"{prototypes_per_fruit} prototypes"
        )

    return np.stack(
        result
    ).astype(
        np.float32
    )


def choose_threshold(
    known_probability,
    unknown_probability,
    min_known_accept=0.90,
):
    candidates = np.unique(
        np.concatenate(
            [
                known_probability,
                unknown_probability,
                [
                    0.0,
                    1.0,
                ],
            ]
        )
    )

    best = None

    for threshold in candidates:
        known_accept = float(
            np.mean(
                known_probability
                >= threshold
            )
        )

        if (
            known_accept + 1e-12
            < min_known_accept
        ):
            continue

        unknown_reject = float(
            np.mean(
                unknown_probability
                < threshold
            )
        )

        key = (
            unknown_reject,
            known_accept,
            threshold,
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(
                    threshold
                ),
                known_accept,
                unknown_reject,
            )

    if best is None:
        threshold = float(
            np.quantile(
                known_probability,
                1.0
                - min_known_accept,
            )
        )

        return (
            threshold,
            float(
                np.mean(
                    known_probability
                    >= threshold
                )
            ),
            float(
                np.mean(
                    unknown_probability
                    < threshold
                )
            ),
        )

    return (
        best[1],
        best[2],
        best[3],
    )


def fit_gate(
    known_embeddings,
    known_probabilities,
    unknown_embeddings,
    unknown_probabilities,
    prototypes,
    min_known_accept=0.90,
):
    """Fit the exact StandardScaler + balanced LogisticRegression legacy gate."""
    known_x, _ = gate_features(
        known_embeddings,
        known_probabilities,
        prototypes,
    )

    unknown_x, _ = gate_features(
        unknown_embeddings,
        unknown_probabilities,
        prototypes,
    )

    x = np.concatenate(
        [
            known_x,
            unknown_x,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.ones(
                len(known_x)
            ),
            np.zeros(
                len(unknown_x)
            ),
        ]
    )

    scaler = StandardScaler().fit(
        x
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    classifier.fit(
        scaler.transform(
            x
        ),
        y,
    )

    known_p = classifier.predict_proba(
        scaler.transform(
            known_x
        )
    )[:, 1]

    unknown_p = classifier.predict_proba(
        scaler.transform(
            unknown_x
        )
    )[:, 1]

    (
        threshold,
        known_accept,
        unknown_reject,
    ) = choose_threshold(
        known_p,
        unknown_p,
        min_known_accept,
    )

    gate_arrays = {
        "prototypes": prototypes.astype(
            np.float32
        ),
        "scaler_mean": scaler.mean_.astype(
            np.float64
        ),
        "scaler_scale": scaler.scale_.astype(
            np.float64
        ),
        "coef": classifier.coef_[0].astype(
            np.float64
        ),
        "intercept": np.asarray(
            [
                classifier.intercept_[0]
            ],
            dtype=np.float64,
        ),
        "decision_threshold": np.asarray(
            [
                threshold
            ],
            dtype=np.float64,
        ),
    }

    return (
        gate_arrays,
        known_accept,
        unknown_reject,
    )


def build_metadata(
    *,
    checkpoint_path,
    checkpoint,
    external_root,
    known_total,
    prototype_count,
    known_calibration_count,
    unknown_calibration_count,
    threshold,
    known_accept,
    unknown_reject,
):
    """Return the exact production 1.1 metadata contract."""
    return {
        "version": (
            "1.1-external-calibration"
        ),
        "purpose": (
            "supported-vs-unsupported gate; "
            "CNN weights unchanged"
        ),
        "checkpoint_sha256": file_sha256(
            checkpoint_path
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "supported_fruits": list(
            FRUITS
        ),
        "calibration_source": str(
            external_root
        ),
        "known_total_images": int(
            known_total
        ),
        "prototype_images": int(
            prototype_count
        ),
        "known_calibration_images": int(
            known_calibration_count
        ),
        "unknown_calibration_images": int(
            unknown_calibration_count
        ),
        "chosen_threshold": float(
            threshold
        ),
        "known_calibration_accept_rate": float(
            known_accept
        ),
        "unknown_calibration_reject_rate": float(
            unknown_reject
        ),
        "known_validation_accept_rate": float(
            known_accept
        ),
        "important_note": (
            "The external V2 set had already been evaluated "
            "before this calibration. From this point it is "
            "calibration data, not a future untouched test set."
        ),
    }
