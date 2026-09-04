from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.data.io import IMAGE_EXTS, read_image


def pair_key(stem: str) -> str:
    lower = stem.lower()
    if lower.endswith("_lq") or lower.endswith("_gt"):
        return stem[:-3]
    return stem


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.stem.lower())


@dataclass
class ImageSample:
    name: str
    lq_path: Path
    gt_path: Path | None = None

    def load_lq(self) -> Image.Image:
        return read_image(self.lq_path)

    def load_gt(self) -> Image.Image | None:
        if self.gt_path is None:
            return None
        return read_image(self.gt_path)


class ImagePairDataset:
    """Pairs LQ / GT by stem. Official eval set has 5 pairs; test set has LQ only."""

    def __init__(self, lq_dir: str | Path, gt_dir: str | Path | None = None) -> None:
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.samples = self._build()

    def _build(self) -> list[ImageSample]:
        lq_files = [p for p in list_images(self.lq_dir) if not p.stem.lower().endswith("_gt")]
        gt_map = {}
        if self.gt_dir:
            for path in list_images(self.gt_dir):
                if path.stem.lower().endswith("_lq"):
                    continue
                gt_map[pair_key(path.stem)] = path
        return [
            ImageSample(name=pair_key(p.stem), lq_path=p, gt_path=gt_map.get(pair_key(p.stem)))
            for p in lq_files
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
