"""Shared preprocessing and augmentation for FreshLens CNN."""

from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms as T

from freshlens_ai.constants import PREPROCESS


class Letterbox:
    def __call__(self, image):
        size = int(PREPROCESS["image_size"])
        fill = tuple(PREPROCESS["fill_rgb"])

        return ImageOps.pad(
            image,
            (size, size),
            method=Image.Resampling.BICUBIC,
            color=fill,
            centering=(0.5, 0.5),
        )


def image_transform(training=False):
    steps = [
        Letterbox(),
    ]

    if training:
        steps.extend(
            [
                T.RandomHorizontalFlip(),
                T.RandomAffine(
                    degrees=15,
                    translate=(0.04, 0.04),
                    scale=(0.9, 1.02),
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=tuple(PREPROCESS["fill_rgb"]),
                ),
                T.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.08,
                    hue=0.01,
                ),
            ]
        )

    steps.extend(
        [
            T.ToTensor(),
            T.Normalize(
                PREPROCESS["mean"],
                PREPROCESS["std"],
            ),
        ]
    )

    return T.Compose(steps)
