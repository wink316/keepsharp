from __future__ import annotations

from PIL import Image

from src.models.base import BaseEnhancer, EnhanceContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiffusionEnhancer(BaseEnhancer):
    """Official-compliant backend: Diffusion architecture only.

    This wrapper is model-agnostic. Plug in an open research checkpoint
    (SeeSR / SUPIR / OSEDiff / StableSR, etc.). Commercial APIs are banned.
    """

    name = "diffusion"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.model_id = str(cfg.get("model_id") or "")
        self.num_inference_steps = int(cfg.get("num_inference_steps", 20))
        self.default_strength = float(cfg.get("strength", 0.28))
        self.default_guidance = float(cfg.get("guidance_scale", 4.5))
        self.negative_prompt = str(cfg.get("negative_prompt", ""))
        self.device = "cpu"
        self._pipe = None

    def enhance(self, image: Image.Image, context: EnhanceContext) -> Image.Image:
        pipe = self._load_pipeline()
        strength = context.strength or self.default_strength
        guidance = context.guidance_scale or self.default_guidance
        prompt = context.prompt or "high quality photorealistic image"
        negative = context.negative_prompt or self.negative_prompt

        result = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            image=image.convert("RGB"),
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=self.num_inference_steps,
        )
        output = result.images[0]
        if output.size != image.size:
            output = output.resize(image.size, Image.Resampling.LANCZOS)
        return output.convert("RGB")

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        if not self.model_id:
            raise RuntimeError(
                "Diffusion backend needs diffusion.model_id in configs/inference.yaml. "
                "Use an open research model; commercial APIs are prohibited."
            )
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise RuntimeError("Install extras: pip install -e .[torch]") from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading Diffusion pipeline: %s on %s", self.model_id, self.device)
        pipe = AutoPipelineForImage2Image.from_pretrained(self.model_id, torch_dtype=dtype)
        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self._pipe = pipe
        return pipe
