from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ImagePairDataset
from src.data.io import read_image, save_jpg
from src.eval.evaluate import evaluate_dir
from src.eval.preview import export_eval_preview
from src.inference.consistency import fuse_fidelity
from src.utils.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml")
    data = ImagePairDataset(cfg["paths"]["eval_lq"], cfg["paths"]["eval_gt"])
    src = ROOT / "data" / "outputs" / "phase1" / "lite"
    dests = [
        ROOT / "data" / "outputs" / "phase1" / "baseline",
        ROOT / "data" / "outputs" / "eval",
    ]
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)

    for sample in data:
        pred = read_image(src / f"{sample.name}.jpg")
        safe = fuse_fidelity(pred, sample.load_lq(), mix=0.25, max_delta=12)
        for dest in dests:
            save_jpg(safe, dest / f"{sample.name}.jpg", quality=95)

    report = evaluate_dir(dests[0], cfg["paths"]["eval_gt"], cfg.get("metrics", {}).get("unofficial_weights"))
    (dests[0] / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    export_eval_preview(cfg["paths"]["eval_lq"], dests[0], cfg["paths"]["eval_gt"], dests[0] / "preview.jpg")
    print(json.dumps({"mean": report["mean"], "unofficial_score": report["unofficial_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
