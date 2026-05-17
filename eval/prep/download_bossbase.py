"""Download BOSSbase 1.01 (10 000 grayscale 512x512 PGM covers).

Source: http://agents.fel.cvut.cz/boss/index.php?mode=VIEW&tmpl=materials
        (Czech Technical University mirror — Binghamton's DDE Lab original
         link http://dde.binghamton.edu/download/ImageDB/ has been intermittent)

Expected output: data/bossbase/raw/*.pgm (10000 files, ~150 MB total)

Idempotent: skips download if data/bossbase/raw already has 10000 PGMs.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "bossbase" / "raw"
ZIP_PATH = ROOT / "data" / "bossbase" / "BOSSbase_1.01.zip"

# Mirrors in priority order. First-success wins. Verify SHA after download.
MIRRORS = [
    "http://agents.fel.cvut.cz/stegodata/PGMs.tgz",
    "http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip",
]

EXPECTED_FILE_COUNT = 10_000
EXPECTED_DIM = (512, 512)
# SHA256 of the canonical Binghamton zip; pinned after first successful download
# and updated here. Set to None initially so first-run computes + reports.
EXPECTED_SHA256 = None


def sha256_of(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> bool:
    print(f"[download] {url}\n           -> {dest}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100.0 * written / total
                        print(f"\r           {written / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
            print()
        return True
    except Exception as exc:
        print(f"           FAIL: {exc}", flush=True)
        return False


def already_extracted() -> bool:
    if not RAW_DIR.exists():
        return False
    pgms = list(RAW_DIR.glob("*.pgm"))
    return len(pgms) == EXPECTED_FILE_COUNT


def extract(archive: Path) -> int:
    """Extract PGMs from zip or tgz into RAW_DIR. Return count."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".pgm")]
            for m in members:
                # Flatten paths — write to RAW_DIR/<basename>
                data = zf.read(m)
                (RAW_DIR / Path(m).name).write_bytes(data)
        return len(members)
    if suffix in {".tgz", ".tar"}:
        import tarfile
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers() if m.name.lower().endswith(".pgm")]
            for m in members:
                f = tf.extractfile(m)
                if f is not None:
                    (RAW_DIR / Path(m.name).name).write_bytes(f.read())
        return len(members)
    raise ValueError(f"unknown archive type: {suffix}")


def main() -> int:
    if already_extracted():
        print(f"[bossbase] {EXPECTED_FILE_COUNT} PGMs already in {RAW_DIR}, skipping download", flush=True)
        return 0

    if not ZIP_PATH.exists() and not (ZIP_PATH.with_suffix(".tgz")).exists():
        ok = False
        for url in MIRRORS:
            target = ZIP_PATH if url.endswith(".zip") else ZIP_PATH.with_suffix(".tgz")
            if download(url, target):
                ok = True
                break
        if not ok:
            print(f"[bossbase] all mirrors failed", file=sys.stderr)
            return 1

    archive = ZIP_PATH if ZIP_PATH.exists() else ZIP_PATH.with_suffix(".tgz")
    sha = sha256_of(archive)
    print(f"[bossbase] archive sha256: {sha}", flush=True)
    if EXPECTED_SHA256 and sha != EXPECTED_SHA256:
        print(f"[bossbase] SHA mismatch: expected {EXPECTED_SHA256}", file=sys.stderr)
        return 1

    print(f"[bossbase] extracting {archive.name} -> {RAW_DIR}", flush=True)
    n = extract(archive)
    print(f"[bossbase] extracted {n} PGMs", flush=True)

    if n != EXPECTED_FILE_COUNT:
        print(f"[bossbase] WARN: expected {EXPECTED_FILE_COUNT} PGMs, got {n}", file=sys.stderr)

    # Spot-check a sample
    sample = next(RAW_DIR.glob("*.pgm"))
    from PIL import Image
    with Image.open(sample) as im:
        if im.size != EXPECTED_DIM:
            print(f"[bossbase] WARN: sample {sample.name} dim {im.size} != {EXPECTED_DIM}", file=sys.stderr)
        else:
            print(f"[bossbase] sample OK: {sample.name} {im.size} mode={im.mode}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
