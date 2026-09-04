from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from src.data.dataset import ImagePairDataset
from src.data.io import save_jpg
from src.inference.consistency import enforce_resolution, fuse_fidelity, lock_content_highpass, match_color_lab
from src.inference.tiling import enhance_tiled
from src.models.base import BaseEnhancer
from src.models.controllers import SceneController
from src.models.scene_router import SceneRouter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancementPipeline:
    def __init__(
        self,
        enhancer: BaseEnhancer,
        router: SceneRouter,
        controller: SceneController,
        inference_cfg: dict,
    ) -> None:
        self.enhancer = enhancer
        self.router = router
        self.controller = controller
        self.tile_cfg = inference_cfg.get("tiling", {})
        self.consistency_cfg = inference_cfg.get("consistency", {})
        self.jpeg_quality = int(inference_cfg.get("submit", {}).get("jpeg_quality", 95))

    def process_image(self, image, name: str):
        scene = self.router.infer(image, name=name)
        context = self.controller.apply(name, scene)
        logger.info("Enhance %s as scene=%s strength=%.2f", name, scene, context.strength)

        def _run(tile):
            return self.enhancer.enhance(tile, context)

        if self.tile_cfg.get("enabled", True):
            enhanced = enhance_tiled(
                image,
                _run,
                tile_size=int(self.tile_cfg.get("tile_size", 1024)),
                overlap=int(self.tile_cfg.get("overlap", 128)),
                min_size_to_tile=int(self.tile_cfg.get("min_size_to_tile", 1536)),
            )
        else:
            enhanced = _run(image)

        if self.consistency_cfg.get("lock_resolution", True):
            enhanced = enforce_resolution(enhanced, image.size)
        if self.consistency_cfg.get("lock_content", False):
            enhanced = lock_content_highpass(
                enhanced, image, sigma=float(self.consistency_cfg.get("lock_sigma", 1.6))
            )
        if self.consistency_cfg.get("fidelity_fuse", True):
            enhanced = fuse_fidelity(
                enhanced,
                image,
                mix=float(self.consistency_cfg.get("fuse_mix", 0.25)),
                max_delta=float(self.consistency_cfg.get("fuse_max_delta", 12.0)),
            )
        if self.consistency_cfg.get("color_match", False):
            enhanced = match_color_lab(enhanced, image)
        return enhanced, scene

    def run_dir(
        self,
        lq_dir: str | Path,
        output_dir: str | Path,
        gt_dir: str | Path | None = None,
    ) -> list[dict]:
        dataset = ImagePairDataset(lq_dir, gt_dir)
        if not dataset:
            raise FileNotFoundError(f"No images found in {lq_dir}")

        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        records = []
        for sample in tqdm(dataset, desc="infer"):
            enhanced, scene = self.process_image(sample.load_lq(), sample.name)
            out_path = dest / f"{sample.name}.jpg"
            save_jpg(enhanced, out_path, quality=self.jpeg_quality)
            records.append({"name": sample.name, "scene": scene, "output": str(out_path)})
        return records
