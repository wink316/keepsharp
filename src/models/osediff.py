"""OSEDiff one-step restoration (Wu et al., NeurIPS 2024).

Uses official LoRA (`osediff.pkl`) on SD 2.1-base. Scene prompts replace RAM/DAPE
so 8GB machines do not load a second tagging network. Resolution is preserved
(`upscale=1`); 4K is handled by the existing tile pipeline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.models.base import BaseEnhancer, EnhanceContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_tensor(image: Image.Image, device, dtype):
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor.to(device=device, dtype=dtype) * 2.0) - 1.0


def _to_image(tensor) -> Image.Image:
    arr = tensor.detach().float().clamp(-1, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = ((arr + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class OSEDiffEnhancer(BaseEnhancer):
    name = "osediff"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        osediff_cfg = cfg.get("osediff", cfg)
        self.sd_path = Path(osediff_cfg.get("sd_path", "weights/sd21-base"))
        self.lora_path = Path(osediff_cfg.get("lora_path", "weights/osediff.pkl"))
        self.device_name = str(cfg.get("device", "cuda"))
        self.mixed_precision = str(osediff_cfg.get("mixed_precision", "fp16"))
        self._loaded = False
        self.device = None
        self.weight_dtype = None
        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def enhance(self, image: Image.Image, context: EnhanceContext) -> Image.Image:
        self._ensure_loaded()
        import torch

        rgb = image.convert("RGB")
        w, h = rgb.size
        pad_w = (8 - w % 8) % 8
        pad_h = (8 - h % 8) % 8
        if pad_w or pad_h:
            canvas = Image.new("RGB", (w + pad_w, h + pad_h))
            canvas.paste(rgb, (0, 0))
            rgb_in = canvas
        else:
            rgb_in = rgb

        prompt = context.prompt or "high quality photograph, sharp, photorealistic, natural details"
        lq = _to_tensor(rgb_in, self.device, self.weight_dtype)
        with torch.no_grad():
            prompt_embeds = self._encode_prompt(prompt)
            latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
            t = torch.tensor([999], device=self.device, dtype=torch.long)
            eps = self.unet(latent, t, encoder_hidden_states=prompt_embeds).sample
            denoised = self._one_step(eps, latent, 999)
            out = self.vae.decode(denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        result = _to_image(out)
        if pad_w or pad_h:
            result = result.crop((0, 0, w, h))
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result

    def _one_step(self, model_output, sample, timestep: int = 999):
        """Predict x0 at t=999. Avoids newer DDPMScheduler.set_timesteps(1) index bugs."""
        alphas = self.scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_t = alphas[timestep]
        x0 = (sample - (1.0 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        return x0.to(dtype=sample.dtype)

    def _encode_prompt(self, prompt: str):
        tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        from transformers import AutoTokenizer, CLIPTextModel

        if not self.sd_path.exists():
            raise RuntimeError(f"Missing SD 2.1-base at {self.sd_path}. Run scripts/_download_weights.py")
        if not self.lora_path.exists() or self.lora_path.stat().st_size < 1_000_000:
            raise RuntimeError(f"Missing OSEDiff LoRA at {self.lora_path}. Run scripts/_download_weights.py")

        self.device = torch.device(
            self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.weight_dtype = torch.float16 if self.mixed_precision == "fp16" and self.device.type == "cuda" else torch.float32
        sd = str(self.sd_path)
        logger.info("Loading OSEDiff from %s + %s", sd, self.lora_path)

        self.tokenizer = AutoTokenizer.from_pretrained(sd, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd, subfolder="text_encoder", variant="fp16")
        self.scheduler = DDPMScheduler.from_pretrained(sd, subfolder="scheduler")
        self.scheduler.set_timesteps(1, device=str(self.device))
        self.vae = AutoencoderKL.from_pretrained(sd, subfolder="vae", variant="fp16")
        self.unet = UNet2DConditionModel.from_pretrained(sd, subfolder="unet", variant="fp16")

        blob = torch.load(self.lora_path, map_location="cpu", weights_only=False)
        self._load_lora(blob, LoraConfig)

        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self._loaded = True
        logger.info("OSEDiff ready on %s dtype=%s", self.device, self.weight_dtype)

    def _load_lora(self, model: dict, lora_config_cls) -> None:
        self.unet.add_adapter(
            lora_config_cls(
                r=model["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=model["unet_lora_encoder_modules"],
            ),
            adapter_name="default_encoder",
        )
        self.unet.add_adapter(
            lora_config_cls(
                r=model["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=model["unet_lora_decoder_modules"],
            ),
            adapter_name="default_decoder",
        )
        self.unet.add_adapter(
            lora_config_cls(
                r=model["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=model["unet_lora_others_modules"],
            ),
            adapter_name="default_others",
        )
        missing_unet = []
        for name, param in self.unet.named_parameters():
            if "lora" in name or "conv_in" in name:
                if name not in model["state_dict_unet"]:
                    missing_unet.append(name)
                    continue
                param.data.copy_(model["state_dict_unet"][name].to(param.dtype))
        if missing_unet:
            raise RuntimeError(f"OSEDiff UNet LoRA keys missing ({len(missing_unet)}): {missing_unet[:5]}")
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        self.vae.add_adapter(
            lora_config_cls(
                r=model["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=model["vae_lora_encoder_modules"],
            ),
            adapter_name="default_encoder",
        )
        missing_vae = []
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                if name not in model["state_dict_vae"]:
                    missing_vae.append(name)
                    continue
                param.data.copy_(model["state_dict_vae"][name].to(param.dtype))
        if missing_vae:
            raise RuntimeError(f"OSEDiff VAE LoRA keys missing ({len(missing_vae)}): {missing_vae[:5]}")
        self.vae.set_adapter(["default_encoder"])


def build_osediff_args(cfg: dict) -> SimpleNamespace:
    osediff_cfg = cfg.get("osediff", {})
    return SimpleNamespace(
        pretrained_model_name_or_path=str(osediff_cfg.get("sd_path", "weights/sd21-base")),
        osediff_path=str(osediff_cfg.get("lora_path", "weights/osediff.pkl")),
        mixed_precision=str(osediff_cfg.get("mixed_precision", "fp16")),
        merge_and_unload_lora=False,
    )
