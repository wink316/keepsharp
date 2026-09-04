from __future__ import annotations

from dataclasses import dataclass

from src.models.base import EnhanceContext


@dataclass(frozen=True)
class SceneControl:
    strength: float
    guidance_scale: float
    prompt: str


class SceneController:
    """Maps a detected scene to Diffusion control knobs.

    Controllability strategy:
    - Lower strength on face / text / clock to avoid identity or glyph hallucination.
    - Slightly higher strength on plant / bird where texture synthesis helps.
    """

    def __init__(self, controller_cfg: dict, default_negative: str = "") -> None:
        self._cfg = controller_cfg
        self.default_negative = default_negative

    def apply(self, name: str, scene: str) -> EnhanceContext:
        spec = self._cfg.get(scene) or self._cfg.get("general") or {}
        control = SceneControl(
            strength=float(spec.get("strength", 0.25)),
            guidance_scale=float(spec.get("guidance_scale", 3.8)),
            prompt=str(spec.get("prompt", "high quality, photorealistic")),
        )
        return EnhanceContext(
            name=name,
            scene=scene,
            prompt=control.prompt,
            negative_prompt=self.default_negative,
            strength=control.strength,
            guidance_scale=control.guidance_scale,
        )
