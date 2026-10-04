"""Robust image I/O and the one canonical preprocessing pipeline.

Contract: every internal image is a contiguous BGR uint8 NumPy array of
224x224 after letterboxing. Pillow is used for safe file decoding and EXIF
orientation; OpenCV is optional at import time and used when installed for
CLAHE/Gaussian/Canny.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import DEFAULT_PROCESSING, ProcessingConfig
from .schemas import ProcessedImage

try:  # Optional during lightweight unit tests; requirements install it.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only without OpenCV
    cv2 = None


ArrayLikeSource = str | Path | bytes | bytearray | BinaryIO | Image.Image | np.ndarray


def _as_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def _rgba_to_rgb(pil_image: Image.Image) -> Image.Image:
    rgba = pil_image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    corrected = ImageOps.exif_transpose(pil_image)
    if corrected.mode in {"RGBA", "LA"} or "transparency" in corrected.info:
        rgb = _rgba_to_rgb(corrected)
    else:
        rgb = corrected.convert("RGB")
    rgb_array = _as_uint8(np.asarray(rgb))
    return np.ascontiguousarray(rgb_array[:, :, ::-1])


def _array_to_bgr(array: np.ndarray, array_format: str = "BGR") -> np.ndarray:
    array = _as_uint8(array)
    if array.ndim == 2:
        return np.ascontiguousarray(np.repeat(array[:, :, None], 3, axis=2))
    if array.ndim != 3:
        raise ValueError(f"Ảnh phải có 2 hoặc 3 chiều, nhận được shape={array.shape}")
    if array.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(array, 3, axis=2))
    if array.shape[2] == 4:
        # Array input is explicit: RGBA is composited on white; BGRA is not a
        # safe default because callers should declare the format.
        if array_format.upper() != "RGBA":
            raise ValueError("Ảnh 4 kênh NumPy phải khai báo array_format='RGBA'")
        rgb = array[:, :, :3].astype(np.float32)
        alpha = array[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        return np.ascontiguousarray(_as_uint8(rgb)[:, :, ::-1])
    if array.shape[2] != 3:
        raise ValueError(f"Ảnh phải có 1, 3 hoặc 4 kênh, nhận được {array.shape[2]}")
    if array_format.upper() == "RGB":
        return np.ascontiguousarray(array[:, :, ::-1])
    if array_format.upper() != "BGR":
        raise ValueError("array_format chỉ nhận 'BGR', 'RGB' hoặc 'RGBA'")
    return np.ascontiguousarray(array)


def load_image(source: ArrayLikeSource, array_format: str = "BGR") -> np.ndarray:
    """Decode a path/bytes/file/Pillow/NumPy source into BGR uint8.

    EXIF rotation, grayscale inputs, transparent PNGs and Unicode paths are
    handled here so train and upload use exactly the same entry point.
    """

    if isinstance(source, np.ndarray):
        return _array_to_bgr(source, array_format=array_format)
    if isinstance(source, Image.Image):
        return _pil_to_bgr(source)

    if isinstance(source, (str, Path)):
        with Image.open(Path(source)) as image:
            return _pil_to_bgr(image)
    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(source))) as image:
            return _pil_to_bgr(image)
    if hasattr(source, "read"):
        payload = source.read()
        with Image.open(io.BytesIO(payload)) as image:
            return _pil_to_bgr(image)
    raise TypeError(f"Không hỗ trợ kiểu dữ liệu ảnh: {type(source)!r}")


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    bgr = _array_to_bgr(bgr)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(bgr), mode="RGB")


def resize_letterbox(
    bgr: np.ndarray,
    image_size: int = DEFAULT_PROCESSING.image_size,
    pad_value: int = DEFAULT_PROCESSING.pad_value,
) -> np.ndarray:
    """Resize without stretching and pad the short side to a square."""

    bgr = _array_to_bgr(bgr)
    height, width = bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Ảnh có kích thước rỗng")
    scale = min(image_size / width, image_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    rgb = Image.fromarray(bgr[:, :, ::-1], mode="RGB")
    resized = rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (pad_value, pad_value, pad_value))
    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2
    canvas.paste(resized, (left, top))
    return np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1], dtype=np.uint8)


def _fallback_contrast(bgr: np.ndarray) -> np.ndarray:
    """Dependency-free approximation used only if OpenCV is unavailable."""

    rgb = Image.fromarray(bgr_to_rgb(bgr), mode="RGB")
    channels = [ImageOps.autocontrast(channel) for channel in rgb.split()]
    return np.ascontiguousarray(np.asarray(Image.merge("RGB", channels))[:, :, ::-1], dtype=np.uint8)


def apply_clahe(
    bgr: np.ndarray,
    clip_limit: float = DEFAULT_PROCESSING.clahe_clip_limit,
    grid_size: int = DEFAULT_PROCESSING.clahe_grid_size,
) -> np.ndarray:
    """Apply CLAHE to Lab lightness (not independently to RGB channels)."""

    bgr = _array_to_bgr(bgr)
    if cv2 is None:
        return _fallback_contrast(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return np.ascontiguousarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def gaussian_blur(bgr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    bgr = _array_to_bgr(bgr)
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("gaussian_kernel phải là số lẻ dương")
    if cv2 is not None:
        return np.ascontiguousarray(cv2.GaussianBlur(bgr, (kernel_size, kernel_size), 0))
    radius = max(0.1, kernel_size / 4.0)
    blurred = bgr_to_pil(bgr).filter(ImageFilter.GaussianBlur(radius=radius))
    return _pil_to_bgr(blurred)


def grayscale(bgr: np.ndarray) -> np.ndarray:
    bgr = _array_to_bgr(bgr).astype(np.float32)
    gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    return np.rint(gray).clip(0, 255).astype(np.uint8)


def sobel_magnitude(bgr_or_gray: np.ndarray) -> np.ndarray:
    gray = bgr_or_gray if bgr_or_gray.ndim == 2 else grayscale(bgr_or_gray)
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gray_f = gray.astype(np.float32)
        gx = (np.roll(gray_f, -1, axis=1) - np.roll(gray_f, 1, axis=1)) / 2.0
        gy = (np.roll(gray_f, -1, axis=0) - np.roll(gray_f, 1, axis=0)) / 2.0
    magnitude = np.sqrt(gx * gx + gy * gy)
    return magnitude.astype(np.float32)


def canny_edges(bgr: np.ndarray) -> np.ndarray:
    """Return a diagnostic edge map; it is not a fruit/rot mask."""

    gray = grayscale(bgr)
    if cv2 is not None:
        return np.ascontiguousarray(cv2.Canny(gray, 60, 160))
    magnitude = sobel_magnitude(gray)
    high = float(np.percentile(magnitude, 85))
    low = float(np.percentile(magnitude, 65))
    if high <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    # A lightweight hysteresis approximation: retain strong pixels and weak
    # pixels near a strong one. This is only for the learning view.
    strong = magnitude >= high
    weak = magnitude >= low
    neighbours = sum(np.roll(np.roll(strong, dy, axis=0), dx, axis=1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return np.where(strong | (weak & (neighbours > 0)), 255, 0).astype(np.uint8)


def image_quality(bgr: np.ndarray) -> dict[str, float | int | bool]:
    gray = grayscale(bgr)
    magnitude = sobel_magnitude(gray)
    laplacian = (
        -4.0 * gray.astype(np.float32)
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    rgb = bgr_to_rgb(bgr).astype(np.float32) / 255.0
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = np.zeros_like(max_channel, dtype=np.float32)
    np.divide(max_channel - min_channel, max_channel, out=saturation, where=max_channel > 0)
    height, width = gray.shape
    return {
        "width": int(width),
        "height": int(height),
        "brightness_mean": round(float(gray.mean()), 3),
        "saturation_mean": round(float(saturation.mean()), 3),
        "blur_score_laplacian_variance": round(float(laplacian.var()), 3),
        "edge_strength_mean": round(float(magnitude.mean()), 3),
        "is_very_dark": bool(gray.mean() < 18),
        "is_very_bright": bool(gray.mean() > 244),
        "is_usable": bool(width >= 32 and height >= 32 and np.isfinite(gray).all()),
    }


def preprocess_image(
    source: ArrayLikeSource,
    config: ProcessingConfig = DEFAULT_PROCESSING,
) -> ProcessedImage:
    """Run the canonical train/predict preprocessing pipeline."""

    original = load_image(source)
    source_size = (int(original.shape[1]), int(original.shape[0]))
    letterboxed = resize_letterbox(original, config.image_size, config.pad_value)
    clahe = apply_clahe(letterboxed, config.clahe_clip_limit, config.clahe_grid_size) if config.use_clahe else letterboxed.copy()
    gaussian = gaussian_blur(clahe, config.gaussian_kernel) if config.use_gaussian else clahe.copy()
    edges = canny_edges(gaussian)
    return ProcessedImage(
        original_bgr=original,
        letterboxed_bgr=letterboxed,
        clahe_bgr=clahe,
        gaussian_bgr=gaussian,
        processed_bgr=gaussian,
        edges=edges,
        source_size=source_size,
        quality=image_quality(original),
    )


def image_sha256(source: ArrayLikeSource) -> str:
    """Hash the canonical normalized pixels, not the file name."""

    processed = preprocess_image(source)
    return hashlib.sha256(processed.processed_bgr.tobytes()).hexdigest()
