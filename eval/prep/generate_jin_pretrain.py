"""Build the JIN (J-UNIWARD ImageNet) pretraining set — Tier 1/2 detector-strength.

JIN recipe (Butora, Yousfi & Fridrich, "How to Pretrain for Steganalysis",
IH&MMSec 2021): pretrain a CNN on a binary cover/stego task where natural-image
JPEGs are embedded with *luminance* J-UNIWARD at a random payload drawn uniformly
from [0.4, 0.6] bpnzAC. A JIN-pretrained EfficientNet is a stronger JPEG-steganalysis
detector than the plain-ImageNet-pretrained one — this addresses Patrick Bas's
"why not JIN?" review note on the §6.3 shadow detector.

Covers: ImageNet-1k val JPEGs -> grayscale, center-cropped to 256x256, re-encoded
at QF75 so the *pretraining domain matches the BOSSbase QF75 shadow downstream*
(the detector is then strongest in exactly the regime we stress-test). The JIN
paper kept ImageNet's native (diverse) QFs for general robustness; we deliberately
match QF75 instead because our test is QF75-specific. Pass --quality / --size to change.

Stego: luminance J-UNIWARD via conseal (JUNIWARD_FIX_OFF_BY_ONE — same implementation
as generate_juniward.py) at per-image random payload ~ U[--payload-min,--payload-max],
seeded deterministically from the filename (reproducibility contract).

Reads:  data/imagenet/**/*.{JPEG,jpg,png}  (recursive)
Writes: data/jin/covers/qf{Q}/<stem>.jpg  and  data/jin/stego/qf{Q}/<stem>.jpg
        (matching filenames -> drops straight into train_efficientnet.PairedStegoDataset)

Idempotent: skips pairs whose cover+stego outputs already exist.
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
IMPL = cl.JUNIWARD_FIX_OFF_BY_ONE


def derive(stem: str, seed_base: int) -> tuple[int, float]:
    """Deterministic per-image (embedding seed, payload fraction in [0,1]).

    Same stem + seed_base -> identical (seed, payload), so the whole JIN set is
    byte-reproducible.
    """
    h = hashlib.sha256(f"{stem}|{seed_base}".encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    frac = int.from_bytes(h[4:8], "big") / 0xFFFFFFFF
    return seed, frac


def prep_one(task) -> tuple[str, bool, str]:
    src, cover_dst, stego_dst, size, quality, pmin, pmax, seed_base = task
    if cover_dst.exists() and stego_dst.exists():
        return src.name, True, "skipped"
    try:
        # 1. Cover: grayscale -> center size x size -> QF{quality} JPEG.
        img = Image.open(src).convert("L")
        w, h = img.size
        if min(w, h) < size:                       # upscale small images to fit the crop
            scale = size / min(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
        cover_dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(cover_dst, format="JPEG", quality=quality, subsampling=0)

        # 2. Stego: luminance J-UNIWARD at random payload ~ U[pmin, pmax].
        seed, frac = derive(src.stem, seed_base)
        payload = pmin + (pmax - pmin) * frac
        cover_dct = jpeglib.read_dct(str(cover_dst))
        cover_spatial = jpeglib.read_spatial(str(cover_dst)).spatial[..., 0]
        stego_y = cl.juniward.simulate_single_channel(
            x0=cover_spatial,
            y0=cover_dct.Y,
            qt=cover_dct.qt[0],
            alpha=payload,
            implementation=IMPL,
            seed=seed,
        )
        stego = jpeglib.read_dct(str(cover_dst))
        stego.Y = stego_y
        stego_dst.parent.mkdir(parents=True, exist_ok=True)
        stego.write_dct(str(stego_dst))
        return src.name, True, f"ok p={payload:.3f}"
    except Exception as exc:
        return src.name, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--imagenet-dir", type=Path, default=ROOT / "data/imagenet")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/jin")
    p.add_argument("--quality", type=int, default=75, help="cover JPEG QF (match the downstream domain)")
    p.add_argument("--size", type=int, default=256, help="square crop edge (JIN uses 256)")
    p.add_argument("--payload-min", type=float, default=0.4)
    p.add_argument("--payload-max", type=float, default=0.6)
    p.add_argument("--seed-base", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0, help="0 = all; e.g. --limit 200 for a smoke test")
    p.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count(); lower if memory is tight")
    args = p.parse_args()

    exts = {".jpeg", ".jpg", ".png"}
    srcs = sorted(q for q in args.imagenet_dir.rglob("*") if q.suffix.lower() in exts)
    if args.limit:
        srcs = srcs[: args.limit]
    if not srcs:
        print(f"[jin] no images under {args.imagenet_dir}", file=sys.stderr)
        return 1

    cover_dir = args.out_dir / "covers" / f"qf{args.quality}"
    stego_dir = args.out_dir / "stego" / f"qf{args.quality}"
    print(f"[jin] {len(srcs)} source images")
    print(f"[jin] -> covers {cover_dir}")
    print(f"[jin] -> stego  {stego_dir}")
    print(f"[jin] {args.size}x{args.size} grayscale QF{args.quality} covers; "
          f"J-UNIWARD payload ~ U[{args.payload_min},{args.payload_max}] bpnzAC")

    tasks = [
        (s, cover_dir / f"{s.stem}.jpg", stego_dir / f"{s.stem}.jpg",
         args.size, args.quality, args.payload_min, args.payload_max, args.seed_base)
        for s in srcs
    ]
    ok = skip = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers or None) as ex:
        futs = [ex.submit(prep_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="jin-prep"):
            name, good, status = fut.result()
            if not good:
                tqdm.write(f"  FAIL {name}: {status}")
                fail += 1
            elif status == "skipped":
                skip += 1
            else:
                ok += 1

    print(f"[jin] DONE: {ok} pairs built, {skip} skipped, {fail} failed -> {args.out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
