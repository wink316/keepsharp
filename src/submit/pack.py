from __future__ import annotations

import re
import zipfile
from pathlib import Path

from src.data.dataset import list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)

FORBIDDEN_CN_PUNCT = re.compile(r"[，。！？；：、“”‘’（）【】《》…—]")


def validate_outputs(output_dir: str | Path, expected_stems: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    files = list_images(output_dir)
    if not files:
        errors.append(f"No images in {output_dir}")
        return errors

    stems = []
    for path in files:
        if path.suffix.lower() != ".jpg":
            errors.append(f"Not jpg: {path.name}")
        if FORBIDDEN_CN_PUNCT.search(path.name):
            errors.append(f"Chinese punctuation in name: {path.name}")
        stems.append(path.stem)

    if expected_stems:
        missing = sorted(set(expected_stems) - set(stems))
        extra = sorted(set(stems) - set(expected_stems))
        if missing:
            errors.append(f"Missing outputs: {missing}")
        if extra:
            errors.append(f"Unexpected outputs: {extra}")
    return errors


def build_zip_name(prefix: str, work_name: str, team_name: str, phone: str) -> str:
    parts = [prefix, work_name, team_name, phone]
    cleaned = [FORBIDDEN_CN_PUNCT.sub("", p).replace(" ", "") for p in parts]
    return "_".join(cleaned) + ".zip"


def pack_submission(
    output_dir: str | Path,
    dest_dir: str | Path,
    prefix: str,
    work_name: str,
    team_name: str,
    phone: str,
    expected_stems: list[str] | None = None,
    max_zip_gb: float = 10.0,
) -> Path:
    errors = validate_outputs(output_dir, expected_stems)
    if errors:
        raise ValueError("Submission validation failed:\n- " + "\n- ".join(errors))

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / build_zip_name(prefix, work_name, team_name, phone)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in list_images(output_dir):
            if path.suffix.lower() != ".jpg":
                continue
            zf.write(path, arcname=f"output_dir/{path.name}")

    size_gb = zip_path.stat().st_size / (1024**3)
    if size_gb > max_zip_gb:
        raise ValueError(f"Zip is {size_gb:.2f} GB, exceeds {max_zip_gb} GB limit")
    logger.info("Packed %s (%.2f GB)", zip_path, size_gb)
    return zip_path
