from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.data.io import image_to_numpy, numpy_to_image


def match_color_lab(source: Image.Image, reference: Image.Image) -> Image.Image:
    """Transfer LQ color statistics onto the enhanced image to reduce hue drift."""
    src = cv2.cvtColor(image_to_numpy(source), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = cv2.cvtColor(image_to_numpy(reference), cv2.COLOR_RGB2LAB).astype(np.float32)

    eps = 1e-6
    for c in range(3):
        s_mean, s_std = src[..., c].mean(), src[..., c].std() + eps
        r_mean, r_std = ref[..., c].mean(), ref[..., c].std() + eps
        src[..., c] = (src[..., c] - s_mean) * (r_std / s_std) + r_mean

    aligned = np.clip(src, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_LAB2RGB)
    return numpy_to_image(rgb)


def enforce_resolution(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def lock_content_highpass(enhanced: Image.Image, reference: Image.Image, sigma: float = 1.2) -> Image.Image:
    """Keep LQ low-frequency structure, take only high-frequency detail from Diffusion."""
    detail = image_to_numpy(enhanced).astype(np.float32)
    base = image_to_numpy(reference).astype(np.float32)
    radius = max(3, int(2 * round(3 * sigma) + 1))
    if radius % 2 == 0:
        radius += 1
    detail_low = cv2.GaussianBlur(detail, (radius, radius), sigma)
    base_low = cv2.GaussianBlur(base, (radius, radius), sigma)
    fused = np.clip(base_low + (detail - detail_low), 0, 255)
    return numpy_to_image(fused)


def fuse_fidelity(
    enhanced: Image.Image,
    reference: Image.Image,
    mix: float = 0.40,
    max_delta: float = 16.0,
) -> Image.Image:
    """Clamp Diffusion residual so the baseline cannot drift far from LQ."""
    pred = image_to_numpy(enhanced).astype(np.float32)
    lq = image_to_numpy(reference).astype(np.float32)
    residual = np.clip(pred - lq, -max_delta, max_delta)
    fused = np.clip(lq + float(mix) * residual, 0, 255)
    return numpy_to_image(fused)
