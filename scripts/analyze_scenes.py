from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import list_images
from src.data.io import read_image
from src.models.scene_router import SceneRouter
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("analyze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify challenge scenes for strategy planning")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--input", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    input_dir = Path(args.input) if args.input else Path(cfg["paths"]["test_lq"])
    router = SceneRouter(cfg.get("scene_router", {}))
    rows = []
    for path in list_images(input_dir):
        scene = router.infer(read_image(path), name=path.stem)
        rows.append({"name": path.stem, "scene": scene})
    counts = Counter(r["scene"] for r in rows)
    report = {"total": len(rows), "counts": dict(counts), "items": rows}
    dest = Path(cfg["paths"]["output_dir"]) / "scene_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Scene distribution: %s", counts)
    logger.info("Saved %s", dest)


if __name__ == "__main__":
    main()
