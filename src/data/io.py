from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return Image.fromarray(array, mode="RGB")


def save_jpg(image: Image.Image, path: str | Path, quality: int = 95) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(dest, format="JPEG", quality=quality, subsampling=0, optimize=True)
