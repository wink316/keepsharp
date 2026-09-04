from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

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

logger = get_logger("benchmark_val")


def _vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None


def _reset_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_backend(name: str, cfg: dict, lq_dir: Path, gt_dir: Path, out_root: Path, max_images: int | None) -> dict:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    enhancer = build_enhancer(name, cfg)
    router = SceneRouter(cfg.get("scene_router", {}))
    controller = SceneController(cfg.get("controllers", {}), cfg.get("diffusion", {}).get("negative_prompt", ""))
    pipeline = EnhancementPipeline(enhancer, router, controller, {**cfg, "submit": cfg.get("submit", {})})

    dataset = ImagePairDataset(lq_dir, gt_dir)
    samples = list(dataset)
    if max_images:
        samples = samples[:max_images]

    _reset_vram()
    t0 = time.perf_counter()
    records = []
    for sample in samples:
        item_t0 = time.perf_counter()
        enhanced, scene = pipeline.process_image(sample.load_lq(), sample.name)
        save_jpg(enhanced, out_dir / f"{sample.name}.jpg", quality=int(cfg.get("submit", {}).get("jpeg_quality", 95)))
        records.append({"name": sample.name, "scene": scene, "sec": round(time.perf_counter() - item_t0, 2)})
        logger.info("%s %s %.2fs", name, sample.name, records[-1]["sec"])
    elapsed = time.perf_counter() - t0
    metrics = evaluate_dir(out_dir, gt_dir, cfg.get("metrics", {}).get("unofficial_weights"))
    try:
        export_eval_preview(lq_dir, out_dir, gt_dir, out_dir / "preview.jpg")
    except Exception as exc:
        logger.warning("preview failed: %s", exc)

    return {
        "backend": name,
        "count": len(records),
        "total_sec": round(elapsed, 2),
        "sec_per_image": round(elapsed / max(len(records), 1), 2),
        "peak_vram_gb": None if _vram_gb() is None else round(_vram_gb(), 2),
        "mean": metrics["mean"],
        "unofficial_score": metrics["unofficial_score"],
        "per_image": records,
        "output_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 validation benchmark")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--backends", default="identity,lite")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all 5 official pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    lq_dir = ROOT / cfg["paths"]["eval_lq"]
    gt_dir = ROOT / cfg["paths"]["eval_gt"]
    out_root = ROOT / cfg["paths"]["output_dir"] / "phase1"
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    max_images = args.max_images or None

    results = []
    for name in backends:
        logger.info("==== backend %s ====", name)
        try:
            results.append(run_backend(name, cfg, lq_dir, gt_dir, out_root, max_images))
        except Exception as exc:
            logger.exception("backend %s failed", name)
            results.append({"backend": name, "error": str(exc)})

    report = {
        "device": "cuda",
        "eval_dir": str(lq_dir),
        "results": results,
    }
    dest = out_root / "benchmark.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Wrote %s", dest)


if __name__ == "__main__":
    main()
