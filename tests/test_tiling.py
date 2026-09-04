from PIL import Image

from src.inference.tiling import enhance_tiled, iter_tiles
from src.inference.consistency import enforce_resolution, fuse_fidelity, lock_content_highpass


def test_iter_tiles_covers_canvas() -> None:
    boxes = iter_tiles(2000, 3000, tile_size=1024, overlap=128)
    assert boxes
    covered = set()
    for y0, x0, y1, x1 in boxes:
        assert y1 - y0 <= 1024
        assert x1 - x0 <= 1024
        covered.add((y0, x0, y1, x1))
    assert (0, 0, 1024, 1024) in covered


def test_tiled_identity_keeps_size() -> None:
    image = Image.new("RGB", (1800, 1600), color=(80, 90, 100))
    out = enhance_tiled(image, lambda x: x, tile_size=512, overlap=64, min_size_to_tile=256)
    assert out.size == image.size


def test_resolution_lock() -> None:
    src = Image.new("RGB", (64, 48), color=1)
    dst = Image.new("RGB", (32, 32), color=2)
    locked = enforce_resolution(dst, src.size)
    assert locked.size == (64, 48)


def test_content_lock_keeps_size() -> None:
    lq = Image.new("RGB", (48, 48), color=(80, 90, 100))
    pred = Image.new("RGB", (48, 48), color=(90, 100, 110))
    out = lock_content_highpass(pred, lq)
    assert out.size == lq.size


def test_fidelity_fuse_stays_near_lq() -> None:
    lq = Image.new("RGB", (32, 32), color=(40, 50, 60))
    pred = Image.new("RGB", (32, 32), color=(200, 10, 10))
    out = fuse_fidelity(pred, lq, mix=0.25, max_delta=12)
    assert out.size == lq.size
    px = out.getpixel((0, 0))
    assert abs(px[0] - 40) <= 4
