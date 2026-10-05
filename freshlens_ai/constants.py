"""Shared constants and paths for the FreshLens CNN pipeline."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

FRUITS = (
    "apple",
    "banana",
    "orange",
    "tomato",
)

STATUSES = (
    "fresh",
    "rotten",
)

CLASSES = tuple(
    f"{fruit}::{status}"
    for fruit in FRUITS
    for status in STATUSES
)

SPLITS = (
    "train",
    "val",
    "test",
)

PREPROCESS = {
    "version": "rgb_exif_white_letterbox224_imagenet_v1",
    "image_size": 224,
    "fill_rgb": [255, 255, 255],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(CLASSES)
}

IDX_TO_CLASS = {
    index: name
    for index, name in enumerate(CLASSES)
}
