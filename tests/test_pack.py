from pathlib import Path

from src.submit.pack import build_zip_name, pack_submission, validate_outputs


def test_validate_rejects_png(tmp_path: Path) -> None:
    (tmp_path / "case1.png").write_bytes(b"x")
    errors = validate_outputs(tmp_path)
    assert any("Not jpg" in e for e in errors)


def test_zip_name_and_layout(tmp_path: Path) -> None:
    from PIL import Image

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(output_dir / "case1.jpg")

    zip_path = pack_submission(
        output_dir=output_dir,
        dest_dir=tmp_path / "sub",
        prefix="赛题一",
        work_name="demo",
        team_name="alpha",
        phone="13000000000",
        expected_stems=["case1"],
    )
    assert zip_path.name == build_zip_name("赛题一", "demo", "alpha", "13000000000")
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        assert "output_dir/case1.jpg" in zf.namelist()
