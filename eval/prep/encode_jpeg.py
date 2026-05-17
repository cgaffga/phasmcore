"""Re-encode BOSSbase PGMs as JPEG at the configured quality factor.

Reads:  data/bossbase/raw/*.pgm
Writes: data/bossbase/jpeg_qf{Q}/*.jpg

Phase 1 ships QF=75. Use --quality 95 for the second-tier comparison set later.

Pillow uses libjpeg-turbo on modern macOS installs (verified via
`PIL.__version__` + libjpeg-turbo dylib check). The choice of JPEG library
matters for J-UNIWARD cost computation in the third decimal of PE — if we
ever see calibration drift vs Boroumand 2019, the libjpeg version is one
of the suspects.

Idempotent: skips files whose output already exists.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "bossbase" / "raw"


def encode_one(args: tuple[Path, Path, int]) -> tuple[str, bool, str]:
    src, dst, quality = args
    if dst.exists():
        return src.name, True, "skipped"
    try:
        with Image.open(src) as im:
            im = im.convert("L")  # ensure grayscale
            # subsampling=0 = no chroma subsampling (we're grayscale anyway, but
            # being explicit avoids surprises if anyone runs this on a color cover)
            # optimize=False — match Binghamton/J-UNIWARD reference behavior
            im.save(dst, format="JPEG", quality=quality, subsampling=0, optimize=False)
        return src.name, True, "ok"
    except Exception as exc:
        return src.name, False, str(exc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quality", type=int, default=75, help="JPEG quality factor (75 or 95)")
    p.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count(); set lower if Pillow contends")
    args = p.parse_args()

    if not SRC_DIR.exists():
        print(f"[encode_jpeg] {SRC_DIR} not found — run prep/download_bossbase.py first", file=sys.stderr)
        return 1

    out_dir = ROOT / "data" / "bossbase" / f"jpeg_qf{args.quality}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(SRC_DIR.glob("*.pgm"))
    if not sources:
        print(f"[encode_jpeg] no .pgm files in {SRC_DIR}", file=sys.stderr)
        return 1

    workers = args.workers or None
    print(f"[encode_jpeg] encoding {len(sources)} PGMs at QF={args.quality} -> {out_dir}", flush=True)

    tasks = [(src, out_dir / (src.stem + ".jpg"), args.quality) for src in sources]
    n_ok, n_skip, n_fail = 0, 0, 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(encode_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="encoding"):
            name, ok, status = fut.result()
            if not ok:
                tqdm.write(f"  FAIL {name}: {status}")
                n_fail += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_ok += 1

    print(f"[encode_jpeg] done: {n_ok} ok, {n_skip} skipped, {n_fail} failed", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
