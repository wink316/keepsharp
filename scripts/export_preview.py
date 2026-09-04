from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.preview import export_eval_preview
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("preview")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    dest = Path(cfg["paths"]["output_dir"]) / "eval" / "preview.jpg"
    export_eval_preview(cfg["paths"]["eval_lq"], Path(cfg["paths"]["output_dir"]) / "eval", cfg["paths"]["eval_gt"], dest)
    logger.info("Wrote %s", dest)


if __name__ == "__main__":
    main()
