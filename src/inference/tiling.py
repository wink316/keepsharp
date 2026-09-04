from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from src.data.io import image_to_numpy, numpy_to_image


def _hann_window(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    window = np.outer(wy, wx)
    return np.clip(window, 1e-3, None)[..., None]


def iter_tiles(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    step = max(tile_size - overlap, 1)
    boxes = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            y1 = min(y + tile_size, height)
            x1 = min(x + tile_size, width)
            y0 = max(y1 - tile_size, 0)
            x0 = max(x1 - tile_size, 0)
            boxes.append((y0, x0, y1, x1))
    # unique boxes while keeping order
    seen = set()
    unique = []
    for box in boxes:
        if box not in seen:
            seen.add(box)
            unique.append(box)
    return unique


def enhance_tiled(
    image: Image.Image,
    enhance_fn: Callable[[Image.Image], Image.Image],
    tile_size: int = 1024,
    overlap: int = 128,
    min_size_to_tile: int = 1536,
) -> Image.Image:
    width, height = image.size
    if max(width, height) < min_size_to_tile:
        return enhance_fn(image)

    src = image_to_numpy(image).astype(np.float32)
    acc = np.zeros_like(src, dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    for y0, x0, y1, x1 in iter_tiles(height, width, tile_size, overlap):
        tile = numpy_to_image(src[y0:y1, x0:x1].astype(np.uint8))
        enhanced = enhance_fn(tile)
        if enhanced.size != tile.size:
            enhanced = enhanced.resize(tile.size, Image.Resampling.LANCZOS)
        patch = image_to_numpy(enhanced).astype(np.float32)
        win = _hann_window(y1 - y0, x1 - x0)
        acc[y0:y1, x0:x1] += patch * win
        weight[y0:y1, x0:x1] += win

    merged = acc / np.clip(weight, 1e-6, None)
    return numpy_to_image(merged)
