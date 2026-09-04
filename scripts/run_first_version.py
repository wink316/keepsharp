from __future__ import annotations

"""Train the local Diffusion baseline and export a complete first-version deliverable."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.check_call(args, cwd=ROOT)


def main() -> None:
    run([PY, "scripts/make_dummy_data.py", "--size", "512", "--n-test", "20", "--force"])
    run([PY, "scripts/train_lite.py", "--steps", "400", "--batch-size", "8", "--crop", "128"])
    run([PY, "scripts/infer.py", "--split", "eval", "--backend", "diffusion"])
    run([PY, "scripts/evaluate.py"])
    run([PY, "scripts/export_preview.py"])
    run([PY, "scripts/infer.py", "--split", "test", "--backend", "diffusion"])
    run([PY, "scripts/pack_submit.py"])
    print("First version ready: data/outputs + submissions/")


if __name__ == "__main__":
    main()
