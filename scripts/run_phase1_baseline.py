from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.check_call(args, cwd=ROOT)


def main() -> None:
    run([PY, "-m", "pytest", "-q"])
    run([PY, "scripts/export_safe_baseline.py"])
    print("Phase-1 baseline artifacts: data/outputs/phase1/baseline and data/outputs/eval")


if __name__ == "__main__":
    main()
