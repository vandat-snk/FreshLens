"""Explainable handcrafted feature groups for the image-processing project.

The full configuration intentionally mirrors the assignment sheet:

* color: 132 values
* texture: 21 values (uniform LBP P=16, R=2 + edge statistics)
* shape: 12 values (five geometry values + seven signed-log Hu moments)
* HOG: 6,084 values for 224x224, 9 orientations, 16-pixel cells, 2x2 blocks

The final vector is always concatenated in the same order. A changed feature
configuration therefore creates a new model artifact rather than silently
feeding a model the wrong vector.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import DEFAULT_FEATURES, FeatureConfig
from .image_processing import grayscale, sobel_magnitude
from .schemas import ProcessedImage


def _normalised_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values.ravel(), bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else np.zeros(bins, dtype=np.float32)


def _bgr_to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, ::-1].astype(np.float32) / 255.0


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV with all channels in [0, 1]."""

    rgb = np.asarray(rgb, dtype=np.float32).clip(0, 1)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_case = nonzero & (maximum == red)
    green_case = nonzero & (maximum == green)
    blue_case = nonzero & (maximum == blue)
    hue[red_case] = ((green[red_case] - blue[red_case]) / delta[red_case]) % 6.0
    hue[green_case] = (blue[green_case] - red[green_case]) / delta[green_case] + 2.0
    hue[blue_case] = (red[blue_case] - green[blue_case]) / delta[blue_case] + 4.0
    hue /= 6.0
    saturation = np.zeros_like(maximum, dtype=np.float32)
    np.divide(delta, maximum, out=saturation, where=maximum > 1e-8)
    return np.stack([hue, saturation, maximum], axis=2).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Small dependency-free sRGB->CIELAB conversion used for color features."""

    rgb = np.asarray(rgb, dtype=np.float32).clip(0, 1)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.einsum(
        "...c,dc->...d",
        linear,
        np.array(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]],
            dtype=np.float32,
        ),
    )
    reference = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    ratio = xyz / reference
    delta = 6 / 29
    f = np.where(ratio > delta**3, np.cbrt(np.maximum(ratio, 1e-12)), ratio / (3 * delta**2) + 4 / 29)
    lightness = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=2).astype(np.float32)


def color_features(bgr: np.ndarray) -> np.ndarray:
    """Return the 132-dimensional color group."""

    rgb = _bgr_to_rgb_float(bgr)
    hsv = rgb_to_hsv(rgb)
    lab = rgb_to_lab(rgb)
    parts: list[np.ndarray] = []
    # Three HSV histograms x 24 bins = 72.
    for channel in range(3):
        parts.append(_normalised_histogram(hsv[:, :, channel], 24, (0.0, 1.0)))
    # Three Lab histograms x 16 bins = 48.
    lab_scaled = np.stack(
        [lab[:, :, 0] / 100.0, (lab[:, :, 1] + 128.0) / 255.0, (lab[:, :, 2] + 128.0) / 255.0],
        axis=2,
    ).clip(0, 1)
    for channel in range(3):
        parts.append(_normalised_histogram(lab_scaled[:, :, channel], 16, (0.0, 1.0)))
    # Four moments per HSV channel = 12.
    for channel in range(3):
        values = hsv[:, :, channel]
        parts.append(np.array([
            values.mean(),
            values.std(),
            np.percentile(values, 10),
            np.percentile(values, 90),
        ], dtype=np.float32))
    output = np.concatenate(parts).astype(np.float32)
    if output.size != 132:
        raise AssertionError(f"Color feature dimension changed: {output.size}")
    return output


def uniform_lbp(gray: np.ndarray, points: int = 16, radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Compute a vectorised uniform LBP image and its P+2-bin code map."""

    if points < 4 or radius < 1:
        raise ValueError("LBP points/radius không hợp lệ")
    gray_float = gray.astype(np.float32)
    center = gray_float
    code = np.zeros_like(gray_float, dtype=np.uint32)
    for index in range(points):
        angle = 2.0 * math.pi * index / points
        dy = int(round(-radius * math.sin(angle)))
        dx = int(round(radius * math.cos(angle)))
        neighbour = np.roll(np.roll(gray_float, dy, axis=0), dx, axis=1)
        code |= ((neighbour >= center).astype(np.uint32) << index)
    transitions = np.zeros_like(code, dtype=np.uint8)
    for index in range(points):
        bit_a = (code >> index) & 1
        bit_b = (code >> ((index + 1) % points)) & 1
        transitions += (bit_a != bit_b).astype(np.uint8)
    bit_count = np.zeros_like(code, dtype=np.uint8)
    for index in range(points):
        bit_count += ((code >> index) & 1).astype(np.uint8)
    # Uniform patterns get bins 0..P; non-uniform patterns share P+1.
    mapped = np.where(transitions <= 2, bit_count, points + 1).astype(np.uint8)
    return mapped, transitions


def texture_features(bgr: np.ndarray, points: int = 16, radius: int = 2) -> np.ndarray:
    """Return the 18-bin uniform LBP histogram plus three edge statistics."""

    gray = grayscale(bgr)
    lbp, _ = uniform_lbp(gray, points=points, radius=radius)
    hist = np.bincount(lbp.ravel(), minlength=points + 2).astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    magnitude = sobel_magnitude(gray)
    threshold = float(np.percentile(magnitude, 75))
    edge_density = float((magnitude >= threshold).mean()) if threshold > 0 else 0.0
    stats = np.array([edge_density, magnitude.mean() / 255.0, magnitude.std() / 255.0], dtype=np.float32)
    output = np.concatenate([hist, stats]).astype(np.float32)
    if output.size != 21:
        raise AssertionError(f"Texture feature dimension changed: {output.size}")
    return output


def _object_mask(bgr: np.ndarray) -> np.ndarray:
    """Estimate a coarse foreground mask for shape descriptors only.

    This is deliberately not used to crop the model input. A mask inferred
    from color/contrast can include background, shadow or text; the app labels
    it as an estimate and displays Canny separately.
    """

    rgb = _bgr_to_rgb_float(bgr)
    hsv = rgb_to_hsv(rgb)
    gray = grayscale(bgr).astype(np.float32) / 255.0
    mask = (hsv[:, :, 1] > 0.12) | (gray < 0.86)
    # Remove a thin border so a dark frame/background does not become the
    # object. Keep the operation deterministic and dependency free.
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    count = int(mask.sum())
    total = mask.size
    if count < total * 0.01 or count > total * 0.98:
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        cy, cx = (mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0
        mask = ((yy - cy) / (mask.shape[0] * 0.43)) ** 2 + ((xx - cx) / (mask.shape[1] * 0.43)) ** 2 <= 1.0
    return mask


def _hu_moments(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if len(xx) < 5:
        return np.zeros(7, dtype=np.float32)
    area = float(len(xx))
    x = xx.astype(np.float64)
    y = yy.astype(np.float64)
    cx, cy = x.mean(), y.mean()

    def eta(p: int, q: int) -> float:
        moment = np.sum((x - cx) ** p * (y - cy) ** q)
        return float(moment / (area ** (1.0 + (p + q) / 2.0) + 1e-12))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    hu = np.array([
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11**2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    ], dtype=np.float64)
    signed_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return np.nan_to_num(signed_log, nan=0.0, posinf=30.0, neginf=-30.0).clip(-30, 30).astype(np.float32)


def shape_features(bgr: np.ndarray) -> np.ndarray:
    """Return five coarse geometry values plus seven signed-log Hu values."""

    mask = _object_mask(bgr)
    height, width = mask.shape
    area = float(mask.sum())
    area_ratio = area / float(height * width)
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        geometry = np.zeros(5, dtype=np.float32)
        return np.concatenate([geometry, np.zeros(7, dtype=np.float32)])
    bbox_width = float(xx.max() - xx.min() + 1)
    bbox_height = float(yy.max() - yy.min() + 1)
    bbox_aspect = bbox_width / max(bbox_height, 1.0)
    extent = area / max(bbox_width * bbox_height, 1.0)
    edge_count = np.abs(np.diff(mask.astype(np.int8), axis=0)).sum() + np.abs(np.diff(mask.astype(np.int8), axis=1)).sum()
    perimeter_ratio = float(edge_count) / max(2.0 * (width + height), 1.0)
    circularity = float(4.0 * math.pi * area / (edge_count * edge_count + 1e-6))
    geometry = np.array([area_ratio, bbox_aspect, extent, perimeter_ratio, circularity], dtype=np.float32)
    output = np.concatenate([geometry, _hu_moments(mask)]).astype(np.float32)
    if output.size != 12:
        raise AssertionError(f"Shape feature dimension changed: {output.size}")
    return output


def hog_features(
    bgr: np.ndarray,
    orientations: int = 9,
    cell_size: int = 16,
    block_cells: int = 2,
) -> np.ndarray:
    """Compute L2-Hys HOG with the assignment's 6,084-value default."""

    gray = grayscale(bgr).astype(np.float32) / 255.0
    if gray.shape[0] % cell_size or gray.shape[1] % cell_size:
        raise ValueError("Kích thước ảnh phải chia hết cho hog_cell_size")
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / orientations)
    cell_rows = gray.shape[0] // cell_size
    cell_cols = gray.shape[1] // cell_size
    hist = np.zeros((cell_rows, cell_cols, orientations), dtype=np.float32)
    for row in range(cell_rows):
        for col in range(cell_cols):
            row_slice = slice(row * cell_size, (row + 1) * cell_size)
            col_slice = slice(col * cell_size, (col + 1) * cell_size)
            bins = np.floor(angle[row_slice, col_slice]).astype(np.int32) % orientations
            weights = magnitude[row_slice, col_slice]
            hist[row, col] = np.bincount(bins.ravel(), weights=weights.ravel(), minlength=orientations)
    blocks: list[np.ndarray] = []
    for row in range(cell_rows - block_cells + 1):
        for col in range(cell_cols - block_cells + 1):
            block = hist[row:row + block_cells, col:col + block_cells].ravel()
            block = block / math.sqrt(float(np.sum(block * block)) + 1e-6)
            block = np.minimum(block, 0.2)
            block = block / math.sqrt(float(np.sum(block * block)) + 1e-6)
            blocks.append(block)
    output = np.concatenate(blocks).astype(np.float32) if blocks else np.zeros(0, dtype=np.float32)
    expected = (cell_rows - block_cells + 1) * (cell_cols - block_cells + 1) * block_cells * block_cells * orientations
    if output.size != expected:
        raise AssertionError(f"HOG feature dimension changed: {output.size} != {expected}")
    return output


def extract_feature_groups(processed: ProcessedImage, config: FeatureConfig = DEFAULT_FEATURES) -> dict[str, np.ndarray]:
    image = processed.processed_bgr
    groups: dict[str, np.ndarray] = {}
    if config.color:
        groups["color"] = color_features(image)
    if config.texture:
        groups["texture"] = texture_features(image, config.lbp_points, config.lbp_radius)
    if config.shape:
        groups["shape"] = shape_features(image)
    if config.hog:
        groups["hog"] = hog_features(image, config.hog_orientations, config.hog_cell_size, config.hog_block_cells)
    if not groups:
        raise ValueError("Phải bật ít nhất một nhóm đặc trưng")
    return groups


def extract_features(processed: ProcessedImage, config: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    groups = extract_feature_groups(processed, config)
    output = np.concatenate([groups[name] for name in ("color", "texture", "shape", "hog") if name in groups]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Vector đặc trưng chứa NaN/Inf")
    return output


def feature_dimensions(config: FeatureConfig = DEFAULT_FEATURES) -> dict[str, int]:
    dims: dict[str, int] = {}
    if config.color:
        dims["color"] = 132
    if config.texture:
        dims["texture"] = config.lbp_points + 5 if config.lbp_points != 16 else 21
        # The production contract is 18 LBP bins + 3 statistics for P=16.
        if config.lbp_points == 16:
            dims["texture"] = 21
    if config.shape:
        dims["shape"] = 12
    if config.hog:
        cells = 224 // config.hog_cell_size
        dims["hog"] = (cells - config.hog_block_cells + 1) ** 2 * config.hog_block_cells**2 * config.hog_orientations
    return dims


def feature_dimension(config: FeatureConfig = DEFAULT_FEATURES) -> int:
    return sum(feature_dimensions(config).values())


def diagnostic_views(processed: ProcessedImage) -> dict[str, Any]:
    """Create truthful intermediate views for the learning tab."""

    gray = grayscale(processed.processed_bgr)
    lbp, _ = uniform_lbp(gray, points=DEFAULT_FEATURES.lbp_points, radius=DEFAULT_FEATURES.lbp_radius)
    lbp_view = (lbp.astype(np.float32) / max(DEFAULT_FEATURES.lbp_points + 1, 1) * 255).astype(np.uint8)
    magnitude = sobel_magnitude(gray)
    hog_view = np.clip(magnitude / max(float(magnitude.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
    return {
        "gray": gray,
        "lbp": lbp_view,
        "edges": processed.edges,
        "gradient_magnitude": hog_view,
        "feature_dimensions": feature_dimensions(DEFAULT_FEATURES),
    }
