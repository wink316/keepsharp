from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.controllers import SceneController
from src.models.factory import build_enhancer
from src.models.scene_router import SceneRouter
from src.inference.pipeline import EnhancementPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controllable enhancement inference")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--split", choices=["eval", "test"], default="eval")
    parser.add_argument("--backend", default=None, help="identity | lite | diffusion | sd_turbo | osediff")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    backend = args.backend or cfg.get("backend", "identity")

    lq_dir = Path(args.input) if args.input else Path(cfg["paths"]["eval_lq" if args.split == "eval" else "test_lq"])
    out_dir = Path(args.output) if args.output else Path(cfg["paths"]["output_dir"]) / args.split

    enhancer = build_enhancer(backend, cfg)
    router = SceneRouter(cfg.get("scene_router", {}))
    controller = SceneController(cfg.get("controllers", {}), cfg.get("diffusion", {}).get("negative_prompt", ""))
    pipeline = EnhancementPipeline(enhancer, router, controller, {**cfg, "submit": cfg.get("submit", {})})

    records = pipeline.run_dir(lq_dir, out_dir)
    summary = out_dir / "infer_summary.json"
    summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d images to %s", len(records), out_dir)


if __name__ == "__main__":
    main()
