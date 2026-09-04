from pathlib import Path

from PIL import Image

from src.data.dataset import ImagePairDataset, pair_key


def test_pair_key_official_suffix() -> None:
    assert pair_key("case1_lq") == "case1"
    assert pair_key("case1_gt") == "case1"
    assert pair_key("case1") == "case1"


def test_official_folder_pairing(tmp_path: Path) -> None:
    folder = tmp_path / "val"
    folder.mkdir()
    Image.new("RGB", (8, 8), (10, 10, 10)).save(folder / "case1_lq.jpg")
    Image.new("RGB", (8, 8), (20, 20, 20)).save(folder / "case1_gt.jpg")
    data = ImagePairDataset(folder, folder)
    assert len(data) == 1
    assert data.samples[0].name == "case1"
    assert data.samples[0].gt_path is not None
