"""Dependency-light smoke test for machines that have not installed pytest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_FEATURES
from src.features import extract_features, feature_dimension
from src.image_processing import load_image, preprocess_image
from src.model import decide_fruit
from src.predict import FreshLensPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/freshlens_model.joblib"))
    parser.add_argument("--image", type=Path, default=Path("data/demo_uploads/apple_fresh_demo.jpg"))
    args = parser.parse_args()
    grayscale = np.full((48, 80), 130, dtype=np.uint8)
    assert load_image(grayscale).shape == (48, 80, 3)
    processed = preprocess_image(args.image)
    vector = extract_features(processed, DEFAULT_FEATURES)
    assert vector.shape == (feature_dimension(DEFAULT_FEATURES),)
    assert np.isfinite(vector).all()
    assert decide_fruit(np.array([.8, .05, .05, .05, .05]), ["apple", "banana", "orange", "tomato", "other"])["accepted"]
    if args.model.exists():
        result = FreshLensPredictor(args.model).predict(args.image)
        assert result.model_run_id
        print({"state": result.state, "fruit": result.fruit, "status": result.status, "confidence": result.fruit_confidence})
    print("FreshLens smoke test: PASS")


if __name__ == "__main__":
    main()
