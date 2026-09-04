from __future__ import annotations

from PIL import Image

from src.models.base import BaseEnhancer, EnhanceContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PretrainedImg2ImgEnhancer(BaseEnhancer):
    """Open HuggingFace img2img Diffusion. Used only as a pretrained probe.

    Not a dedicated restoration network. Commercial APIs are not used.
    """

    name = "pretrained_img2img"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.model_id = str(cfg.get("model_id") or "stabilityai/sd-turbo")
        self.num_inference_steps = int(cfg.get("num_inference_steps", 4))
        self.default_strength = float(cfg.get("strength", 0.22))
        self.default_guidance = float(cfg.get("guidance_scale", 0.0))
        self.negative_prompt = str(cfg.get("negative_prompt") or "")
        self._pipe = None
        self.device = "cpu"

    def enhance(self, image: Image.Image, context: EnhanceContext) -> Image.Image:
        pipe = self._load_pipeline()
        prompt = context.prompt or "high quality photograph, sharp, natural details"
        result = pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt or None,
            image=image.convert("RGB"),
            strength=context.strength or self.default_strength,
            guidance_scale=context.guidance_scale if context.guidance_scale is not None else self.default_guidance,
            num_inference_steps=self.num_inference_steps,
        )
        output = result.images[0]
        if output.size != image.size:
            output = output.resize(image.size, Image.Resampling.LANCZOS)
        return output.convert("RGB")

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise RuntimeError("Install diffusers/transformers: pip install -e .[torch]") from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading pretrained img2img: %s on %s", self.model_id, self.device)
        from pathlib import Path

        source = self.model_id
        local = Path(source)
        if local.exists():
            source = str(local)
        pipe = AutoPipelineForImage2Image.from_pretrained(source, torch_dtype=dtype, local_files_only=local.exists())
        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self._pipe = pipe
        return pipe
