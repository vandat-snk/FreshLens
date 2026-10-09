"""Predict with the same validated CNN/gate runtime as the UI and TV3 evaluator."""
import argparse
import json
from pathlib import Path

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference.cnn_predict import FreshLensPredictor
from freshlens_ai.inference.quality import DEFAULT_QUALITY_CONFIG, load_quality_config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_DIR / 'models/cnn_efficientnet_b0/best.pt')
    parser.add_argument('--gate-npz', type=Path, help='Defaults to open_set_gate.npz beside checkpoint.')
    parser.add_argument('--gate-meta', type=Path, help='Defaults to JSON beside gate NPZ.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--quality-config', type=Path, default=DEFAULT_QUALITY_CONFIG)
    parser.add_argument('--quality', action='store_true', help='Enable experimental quality rejection.')
    return parser


def run(args):
    predictor = FreshLensPredictor(args.checkpoint, args.gate_npz, args.device, gate_meta_path=args.gate_meta,
                                   quality_config=load_quality_config(args.quality_config))
    result = predictor.predict(args.image, check_quality=args.quality)
    result['checkpoint_epoch'] = predictor.metadata['epoch']
    return result


def main():
    try:
        print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
