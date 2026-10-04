"""Single-photo prediction. Import predict_image for later camera/upload UI use."""

import argparse
import json
from pathlib import Path

import torch

from cnn_data import FRUITS, STATUSES, CLASSES, PROJECT_DIR, image_transform, rgb_from_bytes
from cnn_model import choose_device, decode_probabilities, load_model


@torch.inference_mode()
def predict_image(model, image_bytes, device):
    model.eval()
    image = rgb_from_bytes(image_bytes)
    tensor = image_transform(training=False)(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = logits.float().softmax(dim=1).cpu().numpy()
    decoded = decode_probabilities(probabilities)
    fruit, condition = int(decoded["fruit"][0]), int(decoded["condition"][0])
    return {
        "fruit": FRUITS[fruit], "condition": STATUSES[condition],
        "fruit_score": float(decoded["fruit_probabilities"][0, fruit]),
        "condition_score_given_fruit": float(decoded["condition_probabilities"][0, condition]),
        "fruit_scores": {name: float(decoded["fruit_probabilities"][0, i]) for i, name in enumerate(FRUITS)},
        "condition_scores_given_fruit": {name: float(decoded["condition_probabilities"][0, i]) for i, name in enumerate(STATUSES)},
        "joint_scores": {name: float(probabilities[0, i]) for i, name in enumerate(CLASSES)},
        "scope": "One main fruit; apple/banana/orange/tomato only. Scores are uncalibrated model outputs, not measured accuracy.",
        "out_of_scope_detection_trained": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict fruit and condition from a photo")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "best.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    try:
        device = choose_device(args.device)
        model, metadata = load_model(args.checkpoint, device)
        result = predict_image(model, args.image.read_bytes(), device)
        result["checkpoint_epoch"] = metadata["epoch"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
