"""Resumable HTTP download with fallback URLs."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "keepsharp"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    print(f"GET {url} resume={existing}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = getattr(resp, "status", None)
        print("status", status, "headers", resp.headers.get("Content-Length"), resp.headers.get("Content-Range"))
        mode = "ab" if existing and status == 206 else "wb"
        if mode == "wb" and existing:
            print("server ignored Range, rewriting")
        written = existing if mode == "ab" else 0
        with dest.open(mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if written % (64 * 1024 * 1024) < 1024 * 1024:
                    print(f"  {written / (1024**3):.2f} GB")
    print("done", dest, dest.stat().st_size)


if __name__ == "__main__":
    dest = Path(sys.argv[1])
    urls = sys.argv[2:]
    last = None
    for url in urls:
        try:
            download(url, dest)
            raise SystemExit(0)
        except Exception as exc:
            last = exc
            print("FAIL", type(exc).__name__, exc)
    raise SystemExit(f"all urls failed: {last}")
