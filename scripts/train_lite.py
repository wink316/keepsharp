from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import list_images
from src.data.io import read_image
from src.models.lite.degrade import degrade_image
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train_lite")


def _collect_hq(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        for file in list_images(path):
            images.append(read_image(file))
    if not images:
        raise FileNotFoundError(f"No HQ images in {paths}")
    return images


def _random_crop(image: Image.Image, size: int, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    if min(image.size) < size:
        image = image.resize((max(size, image.width), max(size, image.height)), Image.Resampling.BICUBIC)
    x = rng.randint(0, image.width - size)
    y = rng.randint(0, image.height - size)
    return image.crop((x, y, x + size, y + size))


def _batch(images: list[Image.Image], crop: int, batch_size: int, rng: random.Random, device):
    import torch

    hq_list = []
    lq_list = []
    for _ in range(batch_size):
        hq = _random_crop(rng.choice(images), crop, rng)
        lq = degrade_image(hq, rng)
        hq_list.append(np.asarray(hq, dtype=np.float32) / 255.0)
        lq_list.append(np.asarray(lq, dtype=np.float32) / 255.0)
    hq_t = torch.from_numpy(np.stack(hq_list)).permute(0, 3, 1, 2).to(device) * 2 - 1
    lq_t = torch.from_numpy(np.stack(lq_list)).permute(0, 3, 1, 2).to(device) * 2 - 1
    return hq_t, lq_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local conditional DDPM baseline")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    lite_cfg = cfg.get("lite", {})

    import torch
    from src.models.lite.scheduler import DDPMScheduler
    from src.models.lite.unet import LiteCondUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hq_dirs = [ROOT / cfg["paths"]["eval_gt"], ROOT / "data" / "train_synth"]
    images = _collect_hq([p for p in hq_dirs if p.exists()])
    logger.info("Training on %d HQ images, device=%s", len(images), device)

    timesteps = int(lite_cfg.get("timesteps", 50))
    base = int(lite_cfg.get("base_channels", 32))
    model = LiteCondUNet(base=base).to(device)
    scheduler = DDPMScheduler(timesteps=timesteps).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(42)

    model.train()
    last_loss = 0.0
    for step in tqdm(range(1, args.steps + 1), desc="train"):
        hq, lq = _batch(images, args.crop, args.batch_size, rng, device)
        t = torch.randint(0, timesteps, (hq.size(0),), device=device)
        noise = torch.randn_like(hq)
        xt = scheduler.q_sample(hq, t, noise)
        pred = model(xt, lq, t)
        loss = torch.nn.functional.mse_loss(pred, noise)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        last_loss = float(loss.item())

    dest = ROOT / lite_cfg.get("checkpoint", "weights/lite_ddpm.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "timesteps": timesteps, "base_channels": base, "loss": last_loss},
        dest,
    )
    logger.info("Saved %s (last loss=%.4f)", dest, last_loss)


if __name__ == "__main__":
    main()
