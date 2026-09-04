from pathlib import Path

from PIL import Image

from src.inference.pipeline import EnhancementPipeline
from src.models.controllers import SceneController
from src.models.identity import IdentityEnhancer
from src.models.scene_router import SceneRouter


def test_identity_pipeline_writes_jpg(tmp_path: Path) -> None:
    lq = tmp_path / "lq"
    out = tmp_path / "out"
    lq.mkdir()
    Image.new("RGB", (32, 24), color=(12, 34, 56)).save(lq / "case1.png")

    pipeline = EnhancementPipeline(
        enhancer=IdentityEnhancer(),
        router=SceneRouter({"enabled": True, "backend": "heuristic"}),
        controller=SceneController({"general": {"strength": 0.2, "guidance_scale": 3.0, "prompt": "x"}}),
        inference_cfg={
            "tiling": {"enabled": False},
            "consistency": {"color_match": False, "lock_resolution": True},
            "submit": {"jpeg_quality": 90},
        },
    )
    records = pipeline.run_dir(lq, out)
    assert records[0]["name"] == "case1"
    assert (out / "case1.jpg").exists()
