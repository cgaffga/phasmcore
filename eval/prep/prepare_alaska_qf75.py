"""Prepare ALASKA-2 cover + J-UNIWARD@0.20 stego pairs at QF75.

E16 pipeline support. ALASKA-2 raw covers (data/alaska2/cover/*.jpg) have mixed
quality factors (QF~95/85). Re-encode the Y channel at QF75 to match the §6.2
BOSSbase QF75 protocol, then generate J-UNIWARD@0.20 stegos via conseal using
the same fix-off-by-one implementation as E9 (deterministic per-cover seeding).

Writes:
  data/alaska2/jpeg_qf75/*.jpg               re-encoded grayscale covers
  data/stego/juniward_alaska/qf75/payload_020/*.jpg   J-UNI stegos

Idempotent on both stages.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import conseal as cl
import jpeglib
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = 0.20
SEED_BASE = 2026

# Defaults — overridable via CLI
DEFAULT_SRC_COVERS = ROOT / "data" / "alaska2" / "cover"
DEFAULT_COVER_OUT = ROOT / "data" / "alaska2" / "jpeg_qf75"
DEFAULT_STEGO_OUT = ROOT / "data" / "stego" / "juniward_alaska" / "qf75" / "payload_020"


def reencode_one(args: tuple[Path, Path]) -> tuple[str, bool, str]:
    src, dst = args
    if dst.exists():
        return src.name, True, "skipped"
    try:
        with Image.open(src) as im:
            im_gray = im.convert("L")
            im_gray.save(dst, format="JPEG", quality=75, subsampling=0, optimize=False)
        return src.name, True, "ok"
    except Exception as exc:
        return src.name, False, f"{type(exc).__name__}: {exc}"


def derive_seed(cover_name: str, payload: float, seed_base: int) -> int:
    h = hashlib.sha256(f"{cover_name}|{payload:.4f}|{seed_base}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def embed_one(args: tuple[Path, Path, float, int]) -> tuple[str, bool, str]:
    src, dst, payload, seed_base = args
    if dst.exists():
        return src.name, True, "skipped"
    try:
        cover_dct = jpeglib.read_dct(str(src))
        cover_spatial = jpeglib.read_spatial(str(src)).spatial[..., 0]
        seed = derive_seed(src.name, payload, seed_base)
        stego_y = cl.juniward.simulate_single_channel(
            x0=cover_spatial,
            y0=cover_dct.Y,
            qt=cover_dct.qt[0],
            alpha=payload,
            implementation=cl.JUNIWARD_FIX_OFF_BY_ONE,
            seed=seed,
        )
        stego_jpeg = jpeglib.read_dct(str(src))
        stego_jpeg.Y = stego_y
        stego_jpeg.write_dct(str(dst))
        return src.name, True, "ok"
    except Exception as exc:
        return src.name, False, f"{type(exc).__name__}: {exc}"


def stage_reencode(src_covers: Path, cover_out: Path, limit: int) -> int:
    cover_out.mkdir(parents=True, exist_ok=True)
    sources = sorted(src_covers.glob("*.jpg"))
    if limit > 0:
        sources = sources[:limit]
    if not sources:
        print(f"[prepare_alaska] no .jpg files in {src_covers}", file=sys.stderr)
        return 1
    tasks = [(src, cover_out / src.name) for src in sources]
    n_ok = n_skip = n_fail = 0
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(reencode_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="reencode QF75"):
            name, ok, status = fut.result()
            if not ok:
                tqdm.write(f"  FAIL {name}: {status}")
                n_fail += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_ok += 1
    print(f"[prepare_alaska] reencode: {n_ok} ok, {n_skip} skipped, {n_fail} failed -> {cover_out}")
    return 0 if n_fail == 0 else 1


def stage_juniward(cover_dir: Path, stego_out: Path) -> int:
    stego_out.mkdir(parents=True, exist_ok=True)
    sources = sorted(cover_dir.glob("*.jpg"))
    if not sources:
        print(f"[prepare_alaska] no covers in {cover_dir} — run --stage reencode first", file=sys.stderr)
        return 1
    tasks = [(src, stego_out / src.name, PAYLOAD, SEED_BASE) for src in sources]
    n_ok = n_skip = n_fail = 0
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(embed_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"J-UNI@{PAYLOAD}"):
            name, ok, status = fut.result()
            if not ok:
                tqdm.write(f"  FAIL {name}: {status}")
                n_fail += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_ok += 1
    print(f"[prepare_alaska] J-UNI: {n_ok} ok, {n_skip} skipped, {n_fail} failed -> {stego_out}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src-covers", type=Path, default=DEFAULT_SRC_COVERS,
                   help=f"source ALASKA covers (default: {DEFAULT_SRC_COVERS.relative_to(ROOT)})")
    p.add_argument("--cover-out", type=Path, default=DEFAULT_COVER_OUT,
                   help=f"re-encoded QF75 covers output (default: {DEFAULT_COVER_OUT.relative_to(ROOT)})")
    p.add_argument("--stego-out", type=Path, default=DEFAULT_STEGO_OUT,
                   help=f"J-UNI stegos output (default: {DEFAULT_STEGO_OUT.relative_to(ROOT)})")
    p.add_argument("--limit", type=int, default=0, help="0 = all (default); else cap to first N")
    args = p.parse_args()

    rc = stage_reencode(args.src_covers, args.cover_out, args.limit)
    if rc != 0:
        return rc
    return stage_juniward(args.cover_out, args.stego_out)


if __name__ == "__main__":
    raise SystemExit(main())
