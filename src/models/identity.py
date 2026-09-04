from __future__ import annotations

from PIL import Image

from src.models.base import BaseEnhancer, EnhanceContext


class IdentityEnhancer(BaseEnhancer):
    """Smoke-test backend: keep content unchanged. Not a scoring solution."""

    name = "identity"

    def enhance(self, image: Image.Image, context: EnhanceContext) -> Image.Image:
        return image.convert("RGB")
