from __future__ import annotations

from pathlib import Path

from src.data.dataset import list_images, pair_key
from src.data.io import read_image
from src.eval.metrics import compute_pair_metrics, unofficial_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_dir(
    pred_dir: str | Path,
    gt_dir: str | Path,
    weights: dict[str, float] | None = None,
) -> dict:
    pred_map = {pair_key(p.stem): p for p in list_images(pred_dir)}
    gt_files = [p for p in list_images(gt_dir) if not p.stem.lower().endswith("_lq")]
    if not gt_files:
        raise FileNotFoundError(f"No GT images in {gt_dir}")

    rows = []
    missing = []
    for gt_path in gt_files:
        key = pair_key(gt_path.stem)
        pred_path = pred_map.get(key)
        if pred_path is None:
            missing.append(key)
            continue
        pred = read_image(pred_path)
        gt = read_image(gt_path)
        if pred.size != gt.size:
            pred = pred.resize(gt.size)
        metrics = compute_pair_metrics(pred, gt)
        metrics["name"] = key
        rows.append(metrics)

    if missing:
        logger.warning("Missing predictions: %s", ", ".join(missing))
    if not rows:
        raise RuntimeError("No matched pred/GT pairs to evaluate.")

    keys = [k for k in rows[0] if k != "name"]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    proxy = unofficial_score(mean, weights or {}) if weights else None
    return {"count": len(rows), "mean": mean, "unofficial_score": proxy, "per_image": rows}
