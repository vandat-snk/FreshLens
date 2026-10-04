from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.config import DEFAULT_FEATURES, FeatureConfig
from src.dataset import split_by_group
from src.features import extract_feature_groups, extract_features, feature_dimension, feature_dimensions
from src.image_processing import load_image, preprocess_image
from src.model import decide_fruit
from src.model import quality_gate_reason


def _image_bytes(mode: str = "RGB") -> bytes:
    if mode == "L":
        image = Image.new("L", (80, 48), color=140)
    elif mode == "RGBA":
        image = Image.new("RGBA", (80, 48), color=(220, 50, 40, 120))
    else:
        image = Image.new("RGB", (80, 48), color=(220, 50, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_grayscale_and_transparent_png_are_bgr_uint8() -> None:
    gray = load_image(_image_bytes("L"))
    alpha = load_image(_image_bytes("RGBA"))
    assert gray.shape == (48, 80, 3)
    assert alpha.shape == (48, 80, 3)
    assert gray.dtype == np.uint8
    assert alpha.dtype == np.uint8
    assert np.array_equal(gray[:, :, 0], gray[:, :, 1])


def test_canonical_pipeline_and_full_feature_dimension() -> None:
    processed = preprocess_image(_image_bytes())
    assert processed.processed_bgr.shape == (224, 224, 3)
    assert processed.processed_bgr.dtype == np.uint8
    groups = extract_feature_groups(processed, DEFAULT_FEATURES)
    assert {name: len(values) for name, values in groups.items()} == {"color": 132, "texture": 21, "shape": 12, "hog": 6084}
    vector = extract_features(processed, DEFAULT_FEATURES)
    assert vector.shape == (6249,)
    assert np.isfinite(vector).all()


def test_feature_ablation_dimensions_are_explicit() -> None:
    processed = preprocess_image(_image_bytes())
    config = FeatureConfig(color=True, texture=True, shape=False, hog=False)
    vector = extract_features(processed, config)
    assert vector.shape == (153,)
    assert feature_dimension(config) == 153
    assert feature_dimensions(config) == {"color": 132, "texture": 21}


def test_open_set_policy_has_accept_low_confidence_and_other_states() -> None:
    classes = np.asarray(["apple", "banana", "orange", "tomato", "other"])
    accepted = decide_fruit(np.asarray([0.85, 0.05, 0.04, 0.03, 0.03]), classes)
    low = decide_fruit(np.asarray([0.36, 0.34, 0.12, 0.10, 0.08]), classes)
    outside = decide_fruit(np.asarray([0.02, 0.03, 0.04, 0.05, 0.86]), classes)
    assert accepted["state"] == "ACCEPTED"
    assert low["state"] == "LOW_CONFIDENCE"
    assert outside["state"] == "OUT_OF_SCOPE"


def test_quality_gate_rejects_uniform_input_but_not_normal_input() -> None:
    assert quality_gate_reason({"is_usable": True, "is_very_dark": False, "is_very_bright": False, "blur_score_laplacian_variance": 0.0, "edge_strength_mean": 0.0})
    assert quality_gate_reason({"is_usable": True, "is_very_dark": False, "is_very_bright": False, "blur_score_laplacian_variance": 100.0, "edge_strength_mean": 2.0}) is None


def test_group_split_never_leaks_a_group() -> None:
    records = []
    for label in ("apple::fresh", "banana::fresh", "other::none"):
        for group_index in range(6):
            fruit, status = label.split("::")
            for view in range(2):
                records.append({
                    "path": f"raw/{label}/{group_index}_{view}.jpg",
                    "fruit": fruit,
                    "status": "" if status == "none" else status,
                    "group_id": f"{label}_{group_index}",
                    "source": "test",
                    "sha256": f"{label}_{group_index}_{view}",
                    "split": "",
                })
    split = split_by_group(records, seed=7)
    groups = {name: {row["group_id"] for row in split if row["split"] == name} for name in ("train", "val", "test")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
