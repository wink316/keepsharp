from src.models.factory import build_enhancer
from src.models.osediff import OSEDiffEnhancer


def test_factory_registers_osediff_without_loading_weights() -> None:
    enhancer = build_enhancer(
        "osediff",
        {"osediff": {"sd_path": "missing-sd", "lora_path": "missing-lora"}},
    )
    assert isinstance(enhancer, OSEDiffEnhancer)
    assert enhancer.name == "osediff"
    assert enhancer._loaded is False
