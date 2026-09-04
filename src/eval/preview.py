from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.data.dataset import list_images, pair_key
from src.data.io import read_image


def _caption(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), (20, 20, 20))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), text, fill=(240, 240, 240))
    return canvas


def export_eval_preview(lq_dir: str | Path, pred_dir: str | Path, gt_dir: str | Path, dest: str | Path) -> Path:
    rows = []
    max_w = 0
    for gt_path in list_images(gt_dir):
        if gt_path.stem.lower().endswith("_lq"):
            continue
        stem = pair_key(gt_path.stem)
        lq_path = next((p for p in list_images(lq_dir) if pair_key(p.stem) == stem and not p.stem.lower().endswith("_gt")), None)
        pred_path = next((p for p in list_images(pred_dir) if pair_key(p.stem) == stem), None)
        if lq_path is None or pred_path is None:
            continue
        cells = []
        for label, path in (("LQ", lq_path), ("Pred", pred_path), ("GT", gt_path)):
            img = read_image(path)
            img.thumbnail((320, 320))
            cells.append(_caption(img, f"{stem} | {label}"))
        row = Image.new("RGB", (sum(c.width for c in cells), max(c.height for c in cells)), (20, 20, 20))
        x = 0
        for cell in cells:
            row.paste(cell, (x, 0))
            x += cell.width
        rows.append(row)
        max_w = max(max_w, row.width)

    if not rows:
        raise FileNotFoundError("No LQ/Pred/GT triples for preview")

    sheet = Image.new("RGB", (max_w, sum(r.height for r in rows)), (20, 20, 20))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest_path, quality=92)
    return dest_path
