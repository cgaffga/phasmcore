"""Gap #3 GFR+FLD baseline — cover-vs-stego rich-feature extractor.

Reuses the GFR-lite + SRMQ1-lite + DCTR-lite kernels from
extract_rich_model_features.py, but operates on a single cover dir and
a single stego dir (the §6.2 cover-vs-stego setting), not paired
n1/n2 shadow dirs.

Output: data/<features-out>/cover/*.npy + data/<features-out>/stego/*.npy

Usage:
  python detectors/extract_handcrafted_cover_stego.py \\
    --cover-dir data/bossbase/jpeg_qf75 \\
    --stego-dir data/stego/phasm_ghost/qf75/payload_040 \\
    --out-dir   data/gfrfld_phasm_qf75_pf040 \\
    --workers 8 --limit 2000

For the canonical GFR+FLD baseline cited by Kodovský 2012 we follow the
ensemble paradigm: this script writes per-image feature vectors; the
FLD ensemble is trained downstream by train_fld_ensemble.py.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detectors"))

from extract_rich_model_features import (  # noqa: E402
    srm_features_one_channel,
    dctr_features_one_channel,
    gfr_features_one_channel,
)


def features_for_path(path: Path) -> np.ndarray:
    """Concatenated SRM + DCTR + GFR features for one greyscale JPEG."""
    img = Image.open(path).convert("L")
    if img.size != (512, 512):
        w, h = img.size
        if w >= 512 and h >= 512:
            left, top = (w - 512) // 2, (h - 512) // 2
            img = img.crop((left, top, left + 512, top + 512))
        else:
            img = img.resize((512, 512), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return np.concatenate([
        srm_features_one_channel(arr),
        dctr_features_one_channel(arr),
        gfr_features_one_channel(arr),
    ]).astype(np.float32)


def _worker(task: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = task
    try:
        if dst.exists():
            return (dst.name, "cached")
        feats = features_for_path(src)
        np.save(dst, feats)
        return (dst.name, "ok")
    except Exception as exc:  # noqa: BLE001
        return (dst.name, f"ERR: {type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cover-dir", type=Path, required=True)
    p.add_argument("--stego-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all images per dir")
    args = p.parse_args()

    out_cover = args.out_dir / "cover"
    out_stego = args.out_dir / "stego"
    out_cover.mkdir(parents=True, exist_ok=True)
    out_stego.mkdir(parents=True, exist_ok=True)

    cover_files = sorted(args.cover_dir.glob("*.jpg"))
    stego_files = sorted(args.stego_dir.glob("*.jpg"))

    # Restrict to filenames present in BOTH dirs (paired by cover).
    paired = sorted({f.name for f in cover_files} & {f.name for f in stego_files})
    if args.limit:
        paired = paired[: args.limit]
    print(f"[extract] paired covers in both dirs: {len(paired)}")

    tasks: list[tuple[Path, Path]] = []
    for name in paired:
        tasks.append((args.cover_dir / name, (out_cover / name).with_suffix(".npy")))
        tasks.append((args.stego_dir / name, (out_stego / name).with_suffix(".npy")))

    okc = okc_cached = okc_err = 0
    with mp.Pool(args.workers) as pool:
        for name, status in tqdm(pool.imap_unordered(_worker, tasks),
                                  total=len(tasks), desc="features"):
            if status == "ok":
                okc += 1
            elif status == "cached":
                okc_cached += 1
            else:
                okc_err += 1
                tqdm.write(f"  FAIL {name}: {status}")

    feat_dim = features_for_path(args.cover_dir / paired[0]).shape[0]
    print(f"[extract] ok={okc} cached={okc_cached} fail={okc_err}")
    print(f"[extract] feature dim per image: {feat_dim}")
    return 0 if okc_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
