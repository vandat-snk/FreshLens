"""Experimental image quality measurements; thresholds require local validation."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from freshlens_ai.constants import PROJECT_DIR

DEFAULT_QUALITY_CONFIG = PROJECT_DIR / "config" / "quality.json"

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_size: int = 100
    blur_threshold: float = 100.0
    dark_threshold: float = 40.0
    bright_threshold: float = 215.0

    def __post_init__(self):
        values = [self.min_size, self.blur_threshold, self.dark_threshold, self.bright_threshold]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError('Quality thresholds must be numbers, not strings or booleans.')
        if not isinstance(self.min_size, int):
            raise ValueError('Minimum size must be an integer.')
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Quality thresholds must be finite.')
        if self.min_size < 1 or self.blur_threshold < 0:
            raise ValueError('Minimum size must be positive and blur threshold nonnegative.')
        if not 0 <= self.dark_threshold < self.bright_threshold <= 255:
            raise ValueError('Brightness thresholds must satisfy 0 <= dark < bright <= 255.')


def load_quality_config(path=DEFAULT_QUALITY_CONFIG):
    values = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(values, dict) or set(values) != set(QualityConfig.__dataclass_fields__):
        raise ValueError('Quality config requires exactly min_size, blur_threshold, dark_threshold, bright_threshold.')
    return QualityConfig(**values)


def quality_config_hash(config):
    content = json.dumps(asdict(config), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def measure_quality(image, config=None):
    config = config or QualityConfig()
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    reasons = []
    if min(width, height) < config.min_size:
        reasons.append('Ảnh quá nhỏ; vui lòng chụp gần hơn hoặc chọn ảnh lớn hơn.')
    if blur < config.blur_threshold:
        reasons.append('Ảnh có thể bị mờ; vui lòng giữ máy ổn định và lấy nét lại.')
    if brightness < config.dark_threshold:
        reasons.append('Ảnh quá tối; vui lòng tăng ánh sáng.')
    if brightness > config.bright_threshold:
        reasons.append('Ảnh quá sáng; vui lòng điều chỉnh ánh sáng hoặc góc chụp.')
    return dict(width=width, height=height, blur_score=blur, brightness=brightness,
                passed=not reasons, reasons=reasons, thresholds=asdict(config))
