from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate import evaluate_dir
from src.eval.preview import export_eval_preview
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate enhanced images against official GT pairs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pred", default=None)
    parser.add_argument("--gt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    pred_dir = Path(args.pred) if args.pred else Path(cfg["paths"]["output_dir"]) / "eval"
    gt_dir = Path(args.gt) if args.gt else Path(cfg["paths"]["eval_gt"])
    result = evaluate_dir(pred_dir, gt_dir, cfg.get("metrics", {}).get("unofficial_weights"))
    print(json.dumps({"count": result["count"], "mean": result["mean"], "unofficial_score": result["unofficial_score"]}, ensure_ascii=False, indent=2))
    out = pred_dir / "eval_report.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved report to %s", out)
    try:
        preview = export_eval_preview(cfg["paths"]["eval_lq"], pred_dir, gt_dir, pred_dir / "preview.jpg")
        logger.info("Saved preview to %s", preview)
    except Exception as exc:
        logger.warning("Preview skipped: %s", exc)


if __name__ == "__main__":
    main()
