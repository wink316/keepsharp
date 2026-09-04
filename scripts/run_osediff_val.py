"""Run official val for OSEDiff without overwriting the Lite benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ImagePairDataset
from src.data.io import save_jpg
from src.eval.evaluate import evaluate_dir
from src.eval.preview import export_eval_preview
from src.inference.pipeline import EnhancementPipeline
from src.models.controllers import SceneController
from src.models.factory import build_enhancer
from src.models.scene_router import SceneRouter
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("osediff_val")


def _vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None


def run(name: str, cfg: dict, lq_dir: Path, gt_dir: Path, out_dir: Path) -> dict:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    enhancer = build_enhancer("osediff", cfg)
    router = SceneRouter(cfg.get("scene_router", {}))
    controller = SceneController(cfg.get("controllers", {}), cfg.get("diffusion", {}).get("negative_prompt", ""))
    pipeline = EnhancementPipeline(enhancer, router, controller, {**cfg, "submit": cfg.get("submit", {})})

    out_dir.mkdir(parents=True, exist_ok=True)
    samples = list(ImagePairDataset(lq_dir, gt_dir))
    t0 = time.perf_counter()
    per_image = []
    for sample in samples:
        item_t0 = time.perf_counter()
        enhanced, scene = pipeline.process_image(sample.load_lq(), sample.name)
        save_jpg(enhanced, out_dir / f"{sample.name}.jpg", quality=95)
        sec = time.perf_counter() - item_t0
        per_image.append({"name": sample.name, "scene": scene, "sec": round(sec, 2)})
        logger.info("%s %s %.2fs", name, sample.name, sec)

    elapsed = time.perf_counter() - t0
    metrics = evaluate_dir(out_dir, gt_dir, cfg.get("metrics", {}).get("unofficial_weights"))
    try:
        export_eval_preview(lq_dir, out_dir, gt_dir, out_dir / "preview.jpg")
    except Exception as exc:
        logger.warning("preview failed: %s", exc)

    return {
        "backend": name,
        "count": len(per_image),
        "total_sec": round(elapsed, 2),
        "sec_per_image": round(elapsed / max(len(per_image), 1), 2),
        "peak_vram_gb": None if _vram_gb() is None else round(_vram_gb(), 2),
        "mean": metrics["mean"],
        "unofficial_score": metrics["unofficial_score"],
        "per_image_metrics": metrics["per_image"],
        "per_image_time": per_image,
        "output_dir": str(out_dir),
        "fidelity_fuse": bool(cfg.get("consistency", {}).get("fidelity_fuse")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()
    base = load_config(ROOT / args.config)
    lq_dir = ROOT / base["paths"]["eval_lq"]
    gt_dir = ROOT / base["paths"]["eval_gt"]
    out_root = ROOT / base["paths"]["output_dir"] / "phase1"

    raw_cfg = deepcopy(base)
    raw_cfg.setdefault("consistency", {})["fidelity_fuse"] = False
    fuse_cfg = deepcopy(base)
    fuse_cfg.setdefault("consistency", {})["fidelity_fuse"] = True

    results = []
    for name, cfg, folder in (
        ("osediff_raw", raw_cfg, out_root / "osediff_raw"),
        ("osediff_fuse", fuse_cfg, out_root / "osediff_fuse"),
    ):
        logger.info("==== %s ====", name)
        try:
            results.append(run(name, cfg, lq_dir, gt_dir, folder))
        except Exception as exc:
            logger.exception("%s failed", name)
            results.append({"backend": name, "error": f"{type(exc).__name__}: {exc}"})
            break

    report = {
        "device": "RTX 4060 Laptop 8GB",
        "protocol": "OSEDiff LoRA on SD2.1-base, scene prompts instead of RAM/DAPE, upscale=1, tile 512/64",
        "eval_dir": str(lq_dir),
        "results": results,
    }
    dest = out_root / "osediff_val.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
