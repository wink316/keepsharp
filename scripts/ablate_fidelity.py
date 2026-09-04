from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ImagePairDataset
from src.data.io import read_image
from src.eval.metrics import compute_pair_metrics
from src.inference.consistency import fuse_fidelity, lock_content_highpass, match_color_lab


def main() -> None:
    data = ImagePairDataset(ROOT / "赛题一" / "验证集", ROOT / "赛题一" / "验证集")
    pred_dir = ROOT / "data" / "outputs" / "phase1" / "lite"
    settings = [
        ("raw_lq", None),
        ("raw_lite", lambda e, lq: e),
        ("hp_only", lambda e, lq: lock_content_highpass(e, lq, sigma=1.6)),
        ("fuse_m25_d12", lambda e, lq: fuse_fidelity(e, lq, mix=0.25, max_delta=12)),
        ("fuse_m40_d16", lambda e, lq: fuse_fidelity(e, lq, mix=0.40, max_delta=16)),
        ("fuse_m55_d20", lambda e, lq: fuse_fidelity(e, lq, mix=0.55, max_delta=20)),
        (
            "hp_then_fuse",
            lambda e, lq: fuse_fidelity(lock_content_highpass(e, lq, sigma=1.6), lq, mix=0.40, max_delta=16),
        ),
        (
            "fuse_then_color",
            lambda e, lq: match_color_lab(fuse_fidelity(e, lq, mix=0.40, max_delta=16), lq),
        ),
    ]

    report = []
    for name, fn in settings:
        rows = []
        for sample in data:
            lq = sample.load_lq()
            gt = sample.load_gt()
            pred = lq if fn is None else fn(read_image(pred_dir / f"{sample.name}.jpg"), lq)
            if pred.size != gt.size:
                pred = pred.resize(gt.size)
            metrics = compute_pair_metrics(pred, gt)
            metrics["name"] = sample.name
            rows.append(metrics)
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "name"}
        report.append({"setting": name, "mean": mean, "per_image": rows})
        print(name, {k: round(v, 4) for k, v in mean.items()})

    dest = ROOT / "data" / "outputs" / "phase1" / "fidelity_ablation.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
