"""Explain one raw fruit photo using the production preprocessing and decoder."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from freshlens_ai.constants import CLASSES, PROJECT_DIR
from freshlens_ai.data import image_transform, rgb_from_bytes
from freshlens_ai.data.transforms import Letterbox
from freshlens_ai.inference.gradcam import GradCAM, overlay_heatmap
from freshlens_ai.models import choose_device, load_model
from freshlens_ai.utils.file_io import file_sha256


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='Raw fruit photo, not a UI screenshot.')
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_DIR / 'models/cnn_efficientnet_b0/best.pt')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True, help='New PNG output path.')
    return parser


def run(args):
    if args.output.suffix.lower() != '.png':
        raise ValueError('Output must be a PNG path.')
    metadata_path = args.output.with_suffix('.json')
    if args.output.exists() or metadata_path.exists():
        raise ValueError('Choose a new output path to preserve previous results.')
    device = choose_device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    image = rgb_from_bytes(args.image.read_bytes())
    tensor = image_transform(training=False)(image).unsqueeze(0).to(device)
    with GradCAM(model, model.features[-1]) as cam:
        heatmap, index = cam.generate_heatmap(tensor)
    # Overlay on the actual padded canvas, never stretch a padded CAM onto the raw photo.
    overlay = overlay_heatmap(heatmap, np.asarray(Letterbox()(image)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    metadata_path.write_text(json.dumps({
        'target_class': CLASSES[index], 'class_index': index,
        'checkpoint_epoch': metadata['epoch'], 'checkpoint_sha256': file_sha256(args.checkpoint),
        'image_sha256': file_sha256(args.image), 'canvas': 'production white letterbox 224x224',
        'note': 'Grad-CAM shows contributions to the raw CNN prediction, not a rotten-region segmentation or gate acceptance.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved Grad-CAM for {CLASSES[index]} to {args.output}')


def main():
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
