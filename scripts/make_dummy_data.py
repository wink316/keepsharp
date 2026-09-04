from __future__ import annotations

"""Build a contest-shaped demo set when official 4K data is not yet downloaded.

Official layout this mimics:
- evaluation: 5 LQ/GT pairs covering face/text/plant/clock/bird
- test: jpg-ready stems case1..N (GT hidden)
"""

import argparse
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import list_images
from src.models.lite.degrade import degrade_image

SCENES = ["face", "text", "plant", "clock", "bird"]


def _is_official_like(folder: Path) -> bool:
    files = list_images(folder)
    if not files:
        return False
    with Image.open(files[0]) as image:
        return max(image.size) >= 2000


def _font(size: int):
    for name in ("arial.ttf", "calibri.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_scene(size: int, scene: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size), (rng.randint(25, 55), rng.randint(30, 60), rng.randint(35, 70)))
    draw = ImageDraw.Draw(image)
    m = size / 256.0

    if scene == "face":
        draw.ellipse((int(50 * m), int(30 * m), int(206 * m), int(210 * m)), fill=(214, 176, 154))
        draw.ellipse((int(90 * m), int(90 * m), int(118 * m), int(118 * m)), fill=(40, 32, 28))
        draw.ellipse((int(140 * m), int(90 * m), int(168 * m), int(118 * m)), fill=(40, 32, 28))
        draw.arc((int(100 * m), int(130 * m), int(156 * m), int(175 * m)), 15, 165, fill=(120, 60, 60), width=max(2, int(3 * m)))
    elif scene == "text":
        draw.rectangle((int(18 * m), int(70 * m), int(238 * m), int(186 * m)), fill=(248, 248, 242))
        draw.text((int(28 * m), int(88 * m)), "CAMERA STAR 2026", fill=(12, 12, 12), font=_font(max(16, int(22 * m))))
        draw.text((int(28 * m), int(128 * m)), "TEXT CTRL CASE", fill=(20, 20, 90), font=_font(max(14, int(20 * m))))
    elif scene == "plant":
        for x in range(int(12 * m), int(240 * m), max(8, int(14 * m))):
            color = (20 + rng.randint(0, 40), 110 + rng.randint(0, 80), 40 + rng.randint(0, 30))
            draw.polygon([(x, int(230 * m)), (x + int(6 * m), int(20 * m + rng.randint(0, 40))), (x + int(14 * m), int(230 * m))], fill=color)
    elif scene == "clock":
        cx = cy = size // 2
        r = int(80 * m)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(236, 232, 220), outline=(20, 20, 20), width=max(3, int(4 * m)))
        draw.line((cx, cy, cx, cy - int(50 * m)), fill=(20, 20, 20), width=max(3, int(4 * m)))
        draw.line((cx, cy, cx + int(38 * m), cy), fill=(160, 30, 30), width=max(2, int(3 * m)))
        for i, label in enumerate(["12", "3", "6", "9"]):
            draw.text((cx - 8 + [0, int(52 * m), 0, -int(58 * m)][i], cy - 10 + [-int(62 * m), 0, int(50 * m), 0][i]), label, fill=(10, 10, 10), font=_font(max(12, int(16 * m))))
    else:
        draw.ellipse((int(30 * m), int(70 * m), int(150 * m), int(190 * m)), fill=(70, 96, 150))
        draw.polygon([(int(140 * m), int(90 * m)), (int(230 * m), int(40 * m)), (int(210 * m), int(120 * m))], fill=(200, 205, 220))
        draw.ellipse((int(168 * m), int(48 * m), int(198 * m), int(78 * m)), fill=(20, 20, 20))
    return image


def write_dataset(size: int, n_test: int, force: bool) -> None:
    eval_lq = ROOT / "data" / "evaluation" / "lq"
    eval_gt = ROOT / "data" / "evaluation" / "gt"
    test_lq = ROOT / "data" / "test"
    train_hq = ROOT / "data" / "train_synth"
    if not force and (_is_official_like(eval_lq) or _is_official_like(test_lq)):
        print("Keep official-looking images. Pass --force only if you intend to replace them.")
        return

    for folder in (eval_lq, eval_gt, test_lq, train_hq):
        folder.mkdir(parents=True, exist_ok=True)
        if force:
            for path in list_images(folder):
                path.unlink()

    rng = random.Random(2026)
    for idx, scene in enumerate(SCENES, start=1):
        gt = draw_scene(size, scene, seed=1000 + idx)
        lq = degrade_image(gt, rng)
        name = f"case{idx}_{scene}"
        gt.save(eval_gt / f"{name}.png")
        lq.save(eval_lq / f"{name}.png")

    for idx in range(1, n_test + 1):
        scene = SCENES[(idx - 1) % len(SCENES)]
        gt = draw_scene(size, scene, seed=2000 + idx)
        degrade_image(gt, rng).save(test_lq / f"case{idx}.png")

    for idx in range(1, 41):
        scene = SCENES[(idx - 1) % len(SCENES)]
        draw_scene(size, scene, seed=3000 + idx).save(train_hq / f"hq_{idx:02d}_{scene}.png")

    print(f"Demo data ready: 5 eval pairs, {n_test} test LQ, 40 train HQ @ {size}px")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_dataset(args.size, args.n_test, args.force)


if __name__ == "__main__":
    main()
