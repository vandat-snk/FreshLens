"""Shared TV3 runtime for UI, CLI and evaluation, using the production contract."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from freshlens_ai.constants import CLASSES, FRUITS, STATUSES
from freshlens_ai.data import rgb_from_bytes
from freshlens_ai.inference.open_set import analyze_bytes, load_gate
from freshlens_ai.inference.quality import load_quality_config, measure_quality, quality_config_hash
from freshlens_ai.models import choose_device, load_model


def image_bytes(image_input):
    """Keep original bytes for EXIF/alpha processing; support existing TV3 callers."""
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)
    if isinstance(image_input, (str, Path)):
        return Path(image_input).read_bytes()
    if isinstance(image_input, np.ndarray):
        image_input = Image.fromarray(image_input)
    if not isinstance(image_input, Image.Image):
        raise TypeError('Expected image bytes, a path, a PIL image or an RGB array.')
    buffer = io.BytesIO()
    image_input.save(buffer, format='PNG')
    return buffer.getvalue()


class FreshLensPredictor:
    """Require a matching gate. Keep raw CNN results even when policy rejects."""
    FRUITS = FRUITS
    CONDITIONS = STATUSES
    JOINT_CLASSES = CLASSES

    def __init__(self, model_path, openset_gate_path=None, device=None,
                 gate_meta_path=None, quality_config=None):
        model_path = Path(model_path)
        self.device = choose_device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model, self.metadata = load_model(model_path, self.device)
        gate_path = Path(openset_gate_path) if openset_gate_path else model_path.with_name('open_set_gate.npz')
        meta_path = Path(gate_meta_path) if gate_meta_path else gate_path.with_suffix('.json')
        # A missing or incompatible gate is a startup error, never an implicit accept.
        self.openset_gate = load_gate(gate_path, meta_path, model_path)
        self.quality_config = quality_config if quality_config is not None else load_quality_config()

    def predict(self, image_input, use_tta=False, check_quality=False):
        if use_tta:
            raise ValueError('TTA is not validated with the production gate. Use use_tta=False.')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        data = image_bytes(image_input)
        quality = measure_quality(rgb_from_bytes(data), self.quality_config)
        result = analyze_bytes(self.model, data, self.device, self.openset_gate)
        gate_supported = result['supported']
        if check_quality and not quality['passed']:
            status = 'quality_rejection'
            reason = ' '.join(quality['reasons'])
        elif not gate_supported:
            status = 'openset_rejection'
            reason = 'Chưa đủ cơ sở nhận diện trong phạm vi Táo, Chuối, Cam và Cà chua.'
        else:
            status, reason = 'accepted', None
        supported = status == 'accepted'
        joint_class = f"{result['fruit']}::{result['condition']}"
        condition_scores = {
            name: result['joint_scores'][f"{result['fruit']}::{name}"] / max(result['fruit_score'], 1e-15)
            for name in STATUSES
        }
        result.update(
            quality_config_sha256=quality_config_hash(self.quality_config),
            condition_scores_given_fruit=condition_scores,
            supported=supported, is_supported=supported, status=status,
            rejection_type=None if supported else status, rejection_reason=reason,
            gate_supported=gate_supported, quality=quality, quality_enabled=check_quality,
            joint_class=joint_class, fruit_confidence=result['fruit_score'],
            condition_confidence=result['condition_score_given_fruit'],
            joint_confidence=result['joint_scores'][joint_class],
        )
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        result['latency_ms'] = (time.perf_counter() - start) * 1000
        return result
