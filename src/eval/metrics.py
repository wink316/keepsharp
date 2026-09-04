from __future__ import annotations

import math

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def psnr(pred: Image.Image, gt: Image.Image) -> float:
    return float(peak_signal_noise_ratio(_to_float(gt), _to_float(pred), data_range=1.0))


def ssim(pred: Image.Image, gt: Image.Image) -> float:
    return float(
        structural_similarity(_to_float(gt), _to_float(pred), channel_axis=2, data_range=1.0)
    )


def lpips_metric(pred: Image.Image, gt: Image.Image) -> float | None:
    try:
        import lpips
        import torch
    except ImportError:
        return None

    loss_fn = getattr(lpips_metric, "_fn", None)
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net="alex")
        lpips_metric._fn = loss_fn

    def _t(img: Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return ten

    with torch.no_grad():
        value = loss_fn(_t(pred), _t(gt))
    return float(value.item())


def niqe_metric(pred: Image.Image) -> float | None:
    try:
        import piq
        import torch
    except ImportError:
        return None

    arr = np.asarray(pred.convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return float(piq.niqe(ten, data_range=1.0).item())


def compute_pair_metrics(pred: Image.Image, gt: Image.Image) -> dict[str, float]:
    scores = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
    lpips_v = lpips_metric(pred, gt)
    niqe_v = niqe_metric(pred)
    if lpips_v is not None:
        scores["lpips"] = lpips_v
    if niqe_v is not None and not math.isnan(niqe_v):
        scores["niqe"] = niqe_v
    return scores


def unofficial_score(metrics: dict[str, float], weights: dict[str, float]) -> float | None:
    """Local proxy score. Official weights are unpublished until finalists."""
    needed = {"psnr", "ssim"}
    if not needed.issubset(metrics):
        return None

    psnr_n = min(max((metrics["psnr"] - 20.0) / 20.0, 0.0), 1.0)
    ssim_n = min(max(metrics["ssim"], 0.0), 1.0)
    lpips_n = 1.0 - min(max(metrics.get("lpips", 0.2), 0.0), 1.0)
    niqe_n = 1.0 - min(max((metrics.get("niqe", 5.0) - 2.0) / 10.0, 0.0), 1.0)

    parts = {"psnr": psnr_n, "ssim": ssim_n, "lpips": lpips_n, "niqe": niqe_n}
    usable = {k: v for k, v in parts.items() if k in weights}
    if not usable:
        return None
    total_w = sum(weights[k] for k in usable)
    return float(sum(usable[k] * weights[k] for k in usable) / total_w)
