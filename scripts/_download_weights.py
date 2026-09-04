"""Download remaining Phase-1 checkpoints. Disable Xet (previous 401 source)."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

OSEDIFF_PKL = "https://raw.githubusercontent.com/cswry/OSEDiff/main/preset/models/osediff.pkl"


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "keepsharp"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  wrote {dest.stat().st_size} bytes")


def snapshot(repo_id: str, dest: Path, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    print(f"HF snapshot {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=["*.bin", "*.ckpt"],
    )
    print(f"  done {repo_id}")


def main() -> int:
    pkl = WEIGHTS / "osediff.pkl"
    if not pkl.exists() or pkl.stat().st_size < 1_000_000:
        download_url(OSEDIFF_PKL, pkl)
    else:
        print("skip osediff.pkl", pkl.stat().st_size)

    if pkl.stat().st_size < 1_000_000:
        print("FAIL osediff.pkl looks like an LFS pointer, not weights")
        print(pkl.read_text(encoding="utf-8", errors="replace")[:300])
        return 2

    sd21 = WEIGHTS / "sd21-base"
    unet = sd21 / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    unet_fp32 = sd21 / "unet" / "diffusion_pytorch_model.safetensors"
    if not unet.exists() and not unet_fp32.exists():
        snapshot(
            "Manojb/stable-diffusion-2-1-base",
            sd21,
            allow_patterns=[
                "model_index.json",
                "scheduler/**",
                "tokenizer/**",
                "feature_extractor/**",
                "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
            ],
        )
    else:
        print("skip sd21-base")

    turbo = WEIGHTS / "sd-turbo"
    turbo_unet = turbo / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    turbo_unet_fp32 = turbo / "unet" / "diffusion_pytorch_model.safetensors"
    if not turbo_unet.exists() and not turbo_unet_fp32.exists():
        try:
            snapshot(
                "stabilityai/sd-turbo",
                turbo,
                allow_patterns=[
                    "model_index.json",
                    "scheduler/**",
                    "tokenizer/**",
                    "text_encoder/config.json",
                    "text_encoder/model.fp16.safetensors",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.fp16.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.fp16.safetensors",
                ],
            )
        except Exception as exc:
            print("FAIL sd-turbo", type(exc).__name__, exc)
            return 3
    else:
        print("skip sd-turbo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
