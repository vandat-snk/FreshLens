"""Public data API; metadata and image audits do not require PyTorch."""

from importlib import import_module

from .image_io import read_record_image, rgb_from_bytes, safe_path, verify_images
from .locked_dataset import load_locked_dataset

__all__ = [
    "FruitDataset", "Letterbox", "image_transform", "load_locked_dataset",
    "read_record_image", "rgb_from_bytes", "safe_path", "verify_images",
]


def __getattr__(name):
    modules = {
        "FruitDataset": ".dataset",
        "Letterbox": ".transforms",
        "image_transform": ".transforms",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
