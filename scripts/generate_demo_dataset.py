"""Generate a small synthetic dataset for smoke-testing the complete app.

This is not a substitute for real photographs. It exists so the team can
verify manifest splitting, feature extraction, training, CLI prediction and
the UI immediately after installation. Replace it with licensed real images
before reporting model accuracy.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


FRUITS = ("apple", "banana", "orange", "tomato")
STATUSES = ("fresh", "rotten")


def _font(size: int = 22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _background(rng: random.Random, size: int = 256) -> Image.Image:
    base = rng.choice([(245, 240, 226), (225, 236, 226), (232, 232, 242), (250, 231, 217)])
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x = rng.randint(0, size)
        y = rng.randint(0, size)
        radius = rng.randint(8, 28)
        shade = tuple(max(0, min(255, value + rng.randint(-12, 12))) for value in base)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=shade)
    return image


def _spots(draw: ImageDraw.ImageDraw, rng: random.Random, box: tuple[int, int, int, int], count: int, rotten: bool) -> None:
    if not rotten:
        return
    left, top, right, bottom = box
    colors = [(75, 42, 25), (118, 66, 35), (65, 56, 34), (152, 91, 44)]
    for _ in range(count):
        x = rng.randint(left, right)
        y = rng.randint(top, bottom)
        radius = rng.randint(4, 13)
        color = rng.choice(colors)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if rng.random() < 0.55:
            draw.ellipse((x - radius // 2, y - radius // 2, x + radius * 2, y + radius * 2), outline=(30, 25, 20), width=2)


def draw_fruit(kind: str, status: str, seed: int, size: int = 256) -> Image.Image:
    rng = random.Random(seed)
    image = _background(rng, size)
    draw = ImageDraw.Draw(image)
    cx = size // 2 + rng.randint(-10, 10)
    cy = size // 2 + rng.randint(-8, 8)
    scale = rng.uniform(0.82, 1.05)
    rotten = status == "rotten"
    if kind == "apple":
        radius = int(65 * scale)
        color = rng.choice([(194, 38, 38), (220, 56, 39), (169, 39, 43)])
        draw.ellipse((cx - radius, cy - radius + 8, cx + radius, cy + radius + 17), fill=color, outline=(105, 28, 27), width=3)
        draw.ellipse((cx - radius + 2, cy - radius, cx - 5, cy + 12), fill=tuple(min(255, c + 20) for c in color))
        draw.line((cx, cy - radius + 16, cx + 9, cy - radius - 15), fill=(83, 47, 25), width=8)
        draw.ellipse((cx + 4, cy - radius - 20, cx + 43, cy - radius + 1), fill=(48, 125, 55))
        _spots(draw, rng, (cx - radius + 9, cy - radius + 22, cx + radius - 8, cy + radius + 5), 8, rotten)
    elif kind == "banana":
        points = []
        for index in range(21):
            t = index / 20
            x = cx - 76 + t * 152
            y = cy + 40 - 70 * math.sin(math.pi * t) + 4 * math.sin(5 * math.pi * t)
            points.append((x, y))
        draw.line(points, fill=(218, 181, 35), width=27, joint="curve")
        inner = [(x, y - 3) for x, y in points]
        draw.line(inner, fill=(248, 218, 84), width=16, joint="curve")
        draw.ellipse((points[0][0] - 9, points[0][1] - 13, points[0][0] + 12, points[0][1] + 12), fill=(90, 70, 30))
        draw.ellipse((points[-1][0] - 12, points[-1][1] - 12, points[-1][0] + 10, points[-1][1] + 13), fill=(75, 54, 27))
        if rotten:
            for _ in range(10):
                index = rng.randint(2, 18)
                x, y = points[index]
                radius = rng.randint(3, 8)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rng.choice([(96, 63, 25), (127, 77, 23), (77, 54, 24)]))
    elif kind == "orange":
        radius = int(68 * scale)
        color = rng.choice([(234, 108, 22), (242, 131, 25), (215, 83, 21)])
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(151, 63, 17), width=3)
        for _ in range(65):
            x = rng.randint(cx - radius + 8, cx + radius - 8)
            y = rng.randint(cy - radius + 8, cy + radius - 8)
            if (x - cx) ** 2 + (y - cy) ** 2 < (radius - 7) ** 2:
                draw.point((x, y), fill=(255, 180, 66))
        draw.ellipse((cx - 14, cy - radius - 7, cx + 13, cy + 12 - radius), fill=(103, 74, 24))
        _spots(draw, rng, (cx - radius + 12, cy - radius + 12, cx + radius - 12, cy + radius - 12), 10, rotten)
    elif kind == "tomato":
        width = int(78 * scale)
        height = int(64 * scale)
        color = rng.choice([(190, 37, 30), (215, 45, 29), (165, 42, 31)])
        draw.ellipse((cx - width, cy - height, cx + width, cy + height), fill=color, outline=(111, 30, 25), width=3)
        crown = [(cx, cy - height + 3), (cx - 22, cy - height - 23), (cx - 7, cy - height + 6), (cx - 58, cy - height - 4), (cx - 26, cy - height + 15), (cx + 26, cy - height + 15), (cx + 60, cy - height - 4), (cx + 7, cy - height + 6), (cx + 22, cy - height - 23)]
        draw.polygon(crown, fill=(48, 124, 48))
        _spots(draw, rng, (cx - width + 10, cy - height + 10, cx + width - 10, cy + height - 10), 10, rotten)
    else:
        raise ValueError(kind)
    if rotten:
        # A gentle brown veil represents surface deterioration without making
        # every rotten image identical.
        overlay = Image.new("RGBA", image.size, (75, 40, 20, rng.randint(0, 22)))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    angle = rng.uniform(-8, 8)
    return ImageOps.exif_transpose(image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False))


def draw_other(kind: str, seed: int, size: int = 256) -> Image.Image:
    rng = random.Random(seed)
    image = _background(rng, size)
    draw = ImageDraw.Draw(image)
    if kind == "number":
        draw.rounded_rectangle((58, 56, 198, 200), radius=18, fill=(250, 250, 250), outline=(35, 50, 70), width=5)
        draw.text((96, 72), str(rng.randint(0, 9)), fill=(30, 40, 80), font=_font(96))
    elif kind == "object":
        draw.rectangle((64, 55, 190, 202), fill=(40, 95, 155), outline=(20, 45, 80), width=5)
        draw.ellipse((91, 88, 166, 163), fill=(230, 180, 40), outline=(120, 80, 20), width=3)
    else:
        draw.rounded_rectangle((45, 70, 211, 182), radius=24, fill=(130, 75, 150), outline=(75, 38, 95), width=5)
        draw.line((80, 119, 176, 119), fill=(245, 210, 235), width=9)
    return image.rotate(rng.uniform(-10, 10), resample=Image.Resampling.BICUBIC)


def generate(root: Path, per_combo: int = 15, other_count: int = 30) -> None:
    raw = root / "raw"
    uploads = root.parent / "demo_uploads"
    raw.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    for fruit_index, fruit in enumerate(FRUITS):
        for status_index, status in enumerate(STATUSES):
            output = raw / fruit / status
            output.mkdir(parents=True, exist_ok=True)
            for index in range(per_combo):
                image = draw_fruit(fruit, status, seed=1000 + fruit_index * 100 + status_index * 30 + index)
                image.save(output / f"{fruit}_{status}_{index:03d}__view01.jpg", quality=93)
    other_kinds = ("number", "object", "food")
    for index in range(other_count):
        kind = other_kinds[index % len(other_kinds)]
        output = raw / "other" / kind
        output.mkdir(parents=True, exist_ok=True)
        image = draw_other(kind, seed=3000 + index)
        image.save(output / f"other_{kind}_{index:03d}.jpg", quality=93)
    # A few named files make the first UI demo obvious.
    draw_fruit("apple", "fresh", 9001).save(uploads / "apple_fresh_demo.jpg", quality=95)
    draw_fruit("banana", "rotten", 9002).save(uploads / "banana_rotten_demo.jpg", quality=95)
    draw_other("number", 9003).save(uploads / "outside_scope_number_demo.jpg", quality=95)
    print(f"Đã tạo demo dataset tại: {root}")
    print(f"Ảnh demo UI tại: {uploads}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo dataset giả lập để smoke-test FreshLens")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "demo")
    parser.add_argument("--per-combo", type=int, default=15)
    parser.add_argument("--other-count", type=int, default=30)
    args = parser.parse_args()
    generate(args.root, args.per_combo, args.other_count)


if __name__ == "__main__":
    main()

