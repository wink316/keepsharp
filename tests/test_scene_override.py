from PIL import Image

from src.models.scene_router import SceneRouter


def test_official_override() -> None:
    router = SceneRouter({"enabled": True, "overrides": {"case1": "text", "case5": "clock"}})
    dummy = Image.new("RGB", (16, 16), (10, 10, 10))
    assert router.infer(dummy, "case1") == "text"
    assert router.infer(dummy, "case5") == "clock"
