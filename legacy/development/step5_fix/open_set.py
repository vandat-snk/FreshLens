"""Open-set gate for FreshLens CNN.

The CNN weights stay unchanged. This module only decides whether an input looks
sufficiently like one of the four supported fruits before showing the CNN label.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cnn_data import FRUITS, STATUSES, CLASSES, image_transform, rgb_from_bytes, DataError
from cnn_model import decode_probabilities


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50, 50)))


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def embedding_and_probabilities(model, tensor):
    """One EfficientNet pass returning normalized 1280-D embeddings + Nx8 softmax."""
    features = model.features(tensor)
    pooled = model.avgpool(features)
    embedding = torch.flatten(pooled, 1)
    logits = model.classifier(embedding)
    embedding = F.normalize(embedding.float(), dim=1)
    probabilities = logits.float().softmax(dim=1)
    return embedding, probabilities


def gate_features(embeddings, probabilities, prototypes):
    """Build small, interpretable features for supported-vs-unsupported gating."""
    embeddings = normalize_rows(embeddings)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    decoded = decode_probabilities(probabilities)
    fruit_probs = decoded["fruit_probabilities"]
    predicted = decoded["fruit"].astype(np.int64)

    rows = []
    details = []
    for i in range(len(embeddings)):
        fp = fruit_probs[i]
        order = np.sort(fp)
        fruit_score = float(fp[predicted[i]])
        margin = float(order[-1] - order[-2])
        entropy = float(-np.sum(fp * np.log(np.maximum(fp, 1e-15))) / math.log(len(FRUITS)))
        joint_max = float(probabilities[i].max())

        sims_by_fruit = []
        for fruit_idx in range(len(FRUITS)):
            sims = prototypes[fruit_idx] @ embeddings[i]
            sims_by_fruit.append(float(np.max(sims)))
        predicted_sim = sims_by_fruit[predicted[i]]
        other_sim = max(value for j, value in enumerate(sims_by_fruit) if j != predicted[i])
        sim_gap = predicted_sim - other_sim

        # High values generally mean "looks supported", except entropy.
        rows.append([fruit_score, margin, joint_max, predicted_sim, sim_gap, 1.0 - entropy])
        details.append({
            "predicted_fruit_index": int(predicted[i]),
            "fruit_score": fruit_score,
            "fruit_margin": margin,
            "joint_max": joint_max,
            "prototype_similarity": predicted_sim,
            "prototype_gap": sim_gap,
            "fruit_certainty": 1.0 - entropy,
        })
    return np.asarray(rows, dtype=np.float64), details


def load_gate(npz_path: Path, meta_path: Path, checkpoint_path: Path):
    npz_path, meta_path, checkpoint_path = map(Path, (npz_path, meta_path, checkpoint_path))
    if not npz_path.is_file() or not meta_path.is_file():
        raise DataError("Chưa có open-set gate. Hãy chạy SETUP_STEP5.cmd một lần trước.")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    expected = meta.get("checkpoint_sha256")
    actual = file_sha256(checkpoint_path)
    if expected != actual:
        raise DataError("best.pt đã thay đổi sau khi tạo gate. Hãy chạy lại SETUP_STEP5.cmd.")
    data = np.load(npz_path, allow_pickle=False)
    required = {"prototypes", "scaler_mean", "scaler_scale", "coef", "intercept", "decision_threshold"}
    if not required.issubset(data.files):
        raise DataError("Open-set gate không đúng định dạng.")
    return {
        "prototypes": data["prototypes"].astype(np.float32),
        "scaler_mean": data["scaler_mean"].astype(np.float64),
        "scaler_scale": data["scaler_scale"].astype(np.float64),
        "coef": data["coef"].astype(np.float64),
        "intercept": float(data["intercept"].reshape(-1)[0]),
        "decision_threshold": float(data["decision_threshold"].reshape(-1)[0]),
        "meta": meta,
    }


def supported_probability(features, gate):
    features = np.asarray(features, dtype=np.float64)
    scaled = (features - gate["scaler_mean"]) / np.maximum(gate["scaler_scale"], 1e-12)
    logits = scaled @ gate["coef"].reshape(-1) + gate["intercept"]
    return sigmoid(logits)


@torch.inference_mode()
def analyze_bytes(model, image_bytes, device, gate):
    image = rgb_from_bytes(image_bytes)
    tensor = image_transform(training=False)(image).unsqueeze(0).to(device)
    embedding, probs_t = embedding_and_probabilities(model, tensor)
    probabilities = probs_t.cpu().numpy()
    emb = embedding.cpu().numpy()
    decoded = decode_probabilities(probabilities)
    fruit_idx = int(decoded["fruit"][0])
    condition_idx = int(decoded["condition"][0])

    features, detail = gate_features(emb, probabilities, gate["prototypes"])
    support_prob = float(supported_probability(features, gate)[0])
    threshold = float(gate["decision_threshold"])
    supported = support_prob >= threshold

    result = {
        "supported": bool(supported),
        "support_probability": support_prob,
        "support_threshold": threshold,
        "fruit": FRUITS[fruit_idx],
        "condition": STATUSES[condition_idx],
        "fruit_score": float(decoded["fruit_probabilities"][0, fruit_idx]),
        "condition_score_given_fruit": float(decoded["condition_probabilities"][0, condition_idx]),
        "fruit_scores": {name: float(decoded["fruit_probabilities"][0, i]) for i, name in enumerate(FRUITS)},
        "joint_scores": {name: float(probabilities[0, i]) for i, name in enumerate(CLASSES)},
        "gate_detail": detail[0],
    }
    return result
