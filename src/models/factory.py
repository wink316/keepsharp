from __future__ import annotations

from src.models.base import BaseEnhancer
from src.models.diffusion_enhancer import DiffusionEnhancer
from src.models.identity import IdentityEnhancer
from src.models.lite.enhancer import LiteDiffusionEnhancer
from src.models.pretrained_img2img import PretrainedImg2ImgEnhancer


def build_enhancer(backend: str, inference_cfg: dict) -> BaseEnhancer:
    backend = (backend or "diffusion").lower()
    if backend == "identity":
        return IdentityEnhancer()
    if backend in {"lite", "diffusion"}:
        model_id = str(inference_cfg.get("diffusion", {}).get("model_id") or "")
        if backend == "diffusion" and model_id:
            return DiffusionEnhancer(inference_cfg.get("diffusion", {}))
        return LiteDiffusionEnhancer(inference_cfg)
    if backend in {"sd_turbo", "pretrained"}:
        return PretrainedImg2ImgEnhancer(inference_cfg.get("sd_turbo", inference_cfg.get("diffusion", {})))
    if backend == "osediff":
        from src.models.osediff import OSEDiffEnhancer

        return OSEDiffEnhancer(inference_cfg)
    raise ValueError(f"Unknown backend: {backend}. Use identity, lite, diffusion, sd_turbo, or osediff.")
