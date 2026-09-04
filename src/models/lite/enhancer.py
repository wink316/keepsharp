from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.io import image_to_numpy, numpy_to_image
from src.models.base import BaseEnhancer, EnhanceContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_tensor(image: Image.Image, device):
    import torch

    arr = image_to_numpy(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device) * 2.0 - 1.0


def _to_image(tensor) -> Image.Image:
    arr = tensor.detach().clamp(-1, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = ((arr + 1.0) * 0.5 * 255.0).round()
    return numpy_to_image(arr)


class LiteDiffusionEnhancer(BaseEnhancer):
    """Open, local conditional DDPM. No commercial API. Resolution-preserving."""

    name = "lite_diffusion"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        lite_cfg = cfg.get("lite", cfg)
        self.checkpoint = Path(lite_cfg.get("checkpoint", "weights/lite_ddpm.pt"))
        self.timesteps = int(lite_cfg.get("timesteps", 50))
        self.sample_steps = int(lite_cfg.get("sample_steps", 12))
        self.base = int(lite_cfg.get("base_channels", 32))
        self.device_name = str(cfg.get("device", "cuda"))
        self._loaded = False
        self.device = None
        self.model = None
        self.scheduler = None

    def enhance(self, image: Image.Image, context: EnhanceContext) -> Image.Image:
        self._ensure_loaded()
        import torch

        lq = image.convert("RGB")
        # pad to multiple of 8 for pooling/unpooling alignment
        w, h = lq.size
        pad_w = (8 - w % 8) % 8
        pad_h = (8 - h % 8) % 8
        if pad_w or pad_h:
            canvas = Image.new("RGB", (w + pad_w, h + pad_h))
            canvas.paste(lq, (0, 0))
            lq_in = canvas
        else:
            lq_in = lq

        lq_t = _to_tensor(lq_in, self.device)
        t_start = max(1, min(self.timesteps - 1, int(self.timesteps * float(context.strength))))
        noise = torch.randn_like(lq_t)
        t0 = torch.tensor([t_start], device=self.device, dtype=torch.long)
        xt = self.scheduler.q_sample(lq_t, t0, noise)

        step_ids = np.linspace(t_start, 0, num=self.sample_steps, dtype=int)
        self.model.eval()
        with torch.no_grad():
            for i, t in enumerate(step_ids):
                t_prev = int(step_ids[i + 1]) if i + 1 < len(step_ids) else -1
                t_batch = torch.tensor([int(t)], device=self.device, dtype=torch.long)
                eps = self.model(xt, lq_t, t_batch)
                xt = self.scheduler.ddim_step(xt, eps, int(t), t_prev)

        out = _to_image(xt)
        if pad_w or pad_h:
            out = out.crop((0, 0, w, h))
        if out.size != image.size:
            out = out.resize(image.size, Image.Resampling.LANCZOS)
        return out

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Lite Diffusion needs torch. Install: pip install torch torchvision") from exc

        from src.models.lite.scheduler import DDPMScheduler
        from src.models.lite.unet import LiteCondUNet

        self.device = torch.device(
            self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        if not self.checkpoint.exists():
            raise RuntimeError(f"Missing checkpoint {self.checkpoint}. Run: python scripts/train_lite.py")

        blob = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.timesteps = int(blob.get("timesteps", self.timesteps))
        self.base = int(blob.get("base_channels", self.base))
        self.model = LiteCondUNet(base=self.base).to(self.device)
        self.model.load_state_dict(blob["model"])
        self.scheduler = DDPMScheduler(timesteps=self.timesteps).to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Loaded Lite Diffusion from %s on %s", self.checkpoint, self.device)
