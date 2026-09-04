from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import list_images
from src.submit.pack import pack_submission
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("pack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack official Tianchi zip")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None, help="Folder of 100 jpg results")
    parser.add_argument("--test-dir", default=None, help="Official test LQ folder for name check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    submit_cfg = cfg["submit"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["output_dir"]) / "test"
    test_dir = Path(args.test_dir) if args.test_dir else Path(cfg["paths"]["test_lq"])
    expected = [p.stem for p in list_images(test_dir)] or None

    zip_path = pack_submission(
        output_dir=output_dir,
        dest_dir=cfg["paths"]["submission_dir"],
        prefix=submit_cfg["zip_prefix"],
        work_name=submit_cfg["work_name"],
        team_name=submit_cfg["team_name"],
        phone=submit_cfg["phone"],
        expected_stems=expected,
        max_zip_gb=float(submit_cfg["max_zip_gb"]),
    )
    logger.info("Ready to upload: %s", zip_path)


if __name__ == "__main__":
    main()
