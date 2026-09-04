from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter


def degrade_image(hq: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Synthetic camera-like degradation for first-version training only."""
    rng = rng or random.Random()
    image = hq.convert("RGB")
    radius = rng.uniform(0.8, 2.4)
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    if rng.random() < 0.45:
        scale = rng.choice([0.5, 0.6, 0.75])
        small = image.resize((max(8, int(image.width * scale)), max(8, int(image.height * scale))), Image.Resampling.BILINEAR)
        image = small.resize(hq.size, Image.Resampling.BILINEAR)
    if rng.random() < 0.8:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=rng.randint(55, 88))
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    return image
