"""Compare the new FreshLens data modules with the original cnn_data.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
LEGACY_DIR = PROJECT_DIR / "legacy" / "development" / "step3_cnn"

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import cnn_data as old  # noqa: E402

from freshlens_ai.constants import (  # noqa: E402
    CLASSES,
    FRUITS,
    PREPROCESS,
    SPLITS,
    STATUSES,
)
from freshlens_ai.data import (  # noqa: E402
    FruitDataset,
    image_transform,
    load_locked_dataset,
    read_record_image,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data" / "cnn_dataset_v3",
    )
    args = parser.parse_args()

    assert FRUITS == old.FRUITS
    assert STATUSES == old.STATUSES
    assert CLASSES == old.CLASSES
    assert SPLITS == old.SPLITS
    assert PREPROCESS == old.PREPROCESS
    print("[OK] Constants match legacy cnn_data.py")

    old_rows, old_identity = old.load_locked_dataset(args.data)
    new_rows, new_identity = load_locked_dataset(args.data)

    assert old_rows == new_rows
    assert old_identity == new_identity
    assert len(new_rows) == 5869
    print("[OK] Locked manifest rows and dataset identity match")

    sample_indices = sorted(
        set([0, len(new_rows) // 2, len(new_rows) - 1])
    )

    for index in sample_indices:
        row = new_rows[index]

        old_image = old.read_record_image(args.root, row)
        new_image = read_record_image(args.root, row)

        assert old_image.mode == new_image.mode == "RGB"
        assert old_image.size == new_image.size
        assert old_image.tobytes() == new_image.tobytes()

        old_eval = old.image_transform(training=False)(old_image)
        new_eval = image_transform(training=False)(new_image)

        assert torch.equal(old_eval, new_eval)

        torch.manual_seed(123456)
        old_train = old.image_transform(training=True)(old_image)

        torch.manual_seed(123456)
        new_train = image_transform(training=True)(new_image)

        assert torch.equal(old_train, new_train)

    print("[OK] Image decoding, eval preprocessing and seeded train augmentation match")

    old_dataset = old.FruitDataset(
        args.root,
        new_rows[:3],
        training=False,
    )

    new_dataset = FruitDataset(
        args.root,
        new_rows[:3],
        training=False,
    )

    for index in range(3):
        old_tensor, old_target, old_index = old_dataset[index]
        new_tensor, new_target, new_index = new_dataset[index]

        assert torch.equal(old_tensor, new_tensor)
        assert old_target == new_target
        assert old_index == new_index

    print("[OK] FruitDataset output matches")
    print("[PASS] Stage B data refactor is behavior-equivalent to legacy cnn_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
