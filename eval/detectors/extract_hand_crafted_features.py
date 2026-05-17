"""Phase 4e MINIMAL: hand-crafted steganalysis features for shadow-count detection.

Auto-detects grayscale vs color JPEGs. For color, computes features on Y, Cb,
and Cr channels, plus Y/chroma asymmetry features (Phasm only modifies Y, so
chroma-cover-statistics serve as a same-image control).

Three feature groups per channel, all NumPy:
- SPAM (Pevný 2010, Subtractive Pixel Adjacency Model) — 1st + 2nd-order Markov
  on truncated pixel differences along 4 directions. ~1568 features per channel.
- DCT-domain coefficient histograms — per-position binned counts, plus per-position
  moments. ~1472 features per channel.
- LSB-plane statistics — spatial and DCT-domain LSB rates, autocorrelations,
  χ² goodness-of-fit, sign-split. ~390 features per channel.

Color asymmetry block (Y stats vs chroma stats):
- DCT LSB-rate asymmetry per coefficient position (Y − Cb, Y − Cr) → 128 features.
- Spatial LSB-rate asymmetry → 4 features.

Total ~3430 features per grayscale 512×512 JPEG.
Total ~10470 features per color 512×512 JPEG.

Adversary model (Phase 4e):
- Has stego JPEG + (already-extracted) one decoy message.
- Wants to know if a SECOND shadow exists at a different passphrase.
- Strictly non-CNN — tests whether the design leaks signal a CNN missed.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import jpeglib
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

T_SPAM = 3  # truncation threshold for SPAM differences
DCT_HIST_RANGE = 9  # histogram bins in [-9, 9] per DCT position


# ---------- SPAM features (Pevný-Bas-Fridrich 2010) ----------

def _truncate(d: np.ndarray, T: int = T_SPAM) -> np.ndarray:
    """Truncate differences to [-T, T] integer range."""
    return np.clip(d, -T, T).astype(np.int8)


def spam_1st_order(diffs: np.ndarray, T: int = T_SPAM) -> np.ndarray:
    """1st-order Markov on truncated differences. Returns 7×7 = 49 features."""
    n_states = 2 * T + 1
    # Cast to int32 before arithmetic — int8 overflows on multiplications below.
    flat_a = (diffs[:, :-1].ravel().astype(np.int32) + T)
    flat_b = (diffs[:, 1:].ravel().astype(np.int32) + T)
    co = np.bincount(flat_a * n_states + flat_b, minlength=n_states ** 2).astype(np.float64)
    co = co.reshape(n_states, n_states)
    row_sums = co.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return (co / row_sums).ravel()


def spam_2nd_order(diffs: np.ndarray, T: int = T_SPAM) -> np.ndarray:
    """2nd-order Markov. Returns 7×7×7 = 343 features."""
    n_states = 2 * T + 1
    flat_a = (diffs[:, :-2].ravel().astype(np.int32) + T)
    flat_b = (diffs[:, 1:-1].ravel().astype(np.int32) + T)
    flat_c = (diffs[:, 2:].ravel().astype(np.int32) + T)
    idx = flat_a * (n_states * n_states) + flat_b * n_states + flat_c
    co = np.bincount(idx, minlength=n_states ** 3).astype(np.float64)
    co = co.reshape(n_states, n_states, n_states)
    pair_sums = co.sum(axis=2, keepdims=True)
    pair_sums[pair_sums == 0] = 1
    return (co / pair_sums).ravel()


def spam_features(img: np.ndarray) -> np.ndarray:
    """Compute 1st + 2nd-order SPAM for 4 directions × ±. Total ~1568 features.

    Directions: horizontal, vertical, diagonal, minor-diagonal.
    For each: differences in +d and -d are computed separately and concatenated.
    """
    img = img.astype(np.int16)
    parts: list[np.ndarray] = []

    # Horizontal: D[i,j] = I[i,j+1] - I[i,j]
    d = _truncate(img[:, 1:] - img[:, :-1])
    parts.append(spam_1st_order(d))
    parts.append(spam_2nd_order(d))
    # Horizontal reverse
    d_rev = _truncate(img[:, :-1] - img[:, 1:])
    parts.append(spam_1st_order(d_rev))
    parts.append(spam_2nd_order(d_rev))

    # Vertical
    d = _truncate(img[1:, :] - img[:-1, :]).T  # transpose so SPAM walks rows
    parts.append(spam_1st_order(d))
    parts.append(spam_2nd_order(d))

    # Diagonal (top-left → bottom-right)
    d = _truncate(img[1:, 1:] - img[:-1, :-1])
    parts.append(spam_1st_order(d))
    parts.append(spam_2nd_order(d))

    # Minor diagonal (top-right → bottom-left)
    d = _truncate(img[1:, :-1] - img[:-1, 1:])
    parts.append(spam_1st_order(d))
    parts.append(spam_2nd_order(d))

    return np.concatenate(parts).astype(np.float32)


# ---------- DCT histogram features ----------

def dct_histogram_features(dct_y: np.ndarray) -> np.ndarray:
    """Per-position binned histograms of DCT coefficients + per-position moments.

    Input: dct_y of shape (blocks_v, blocks_h, 8, 8).
    Output: 64 positions × (19 hist bins + 4 moments) = 64 × 23 = 1472 features.

    Bins span [-9, 9] inclusive with 1-wide bins. Coefficients outside the range
    are accumulated into the edge bins (clipping).
    """
    blocks_v, blocks_h, _, _ = dct_y.shape
    n_blocks = blocks_v * blocks_h
    # Reorder to (8, 8, n_blocks): per-position vectors
    coeffs = dct_y.transpose(2, 3, 0, 1).reshape(8, 8, n_blocks).astype(np.int32)
    feats: list[float] = []
    bins = np.arange(-DCT_HIST_RANGE, DCT_HIST_RANGE + 1)  # 19 bins
    for u in range(8):
        for v in range(8):
            c = coeffs[u, v]
            c_clip = np.clip(c, -DCT_HIST_RANGE, DCT_HIST_RANGE)
            hist = np.bincount(c_clip + DCT_HIST_RANGE, minlength=2 * DCT_HIST_RANGE + 1)
            hist = hist.astype(np.float32) / n_blocks
            feats.extend(hist.tolist())
            # Moments (on un-clipped coefficients)
            feats.append(float(c.mean()))
            feats.append(float(c.std()))
            feats.append(float(c.min()))
            feats.append(float(c.max()))
    return np.array(feats, dtype=np.float32)


# ---------- LSB-plane statistics ----------

def lsb_features(img: np.ndarray, dct_y: np.ndarray) -> np.ndarray:
    """LSB statistics in spatial + DCT domains.

    Spatial-domain LSB plane:
    - mean LSB rate
    - LSB-LSB autocorrelation at 4 offsets
    - χ² goodness-of-fit to uniform LSB
    DCT-domain LSB (per-position):
    - LSB rate per coefficient position (64 features) — Phasm-relevant positions
    - LSB rate split by coefficient sign (positive / negative / zero) — 192 features
    """
    feats: list[float] = []
    pix = img.astype(np.int32)
    lsb_plane = (pix & 1).astype(np.float32)

    # Spatial LSB rate
    feats.append(float(lsb_plane.mean()))
    # Spatial χ² (compare LSB histogram to uniform)
    c0 = float((lsb_plane == 0).sum())
    c1 = float((lsb_plane == 1).sum())
    n = c0 + c1
    exp = n / 2
    chi2 = ((c0 - exp) ** 2 + (c1 - exp) ** 2) / exp
    feats.append(chi2)
    # Autocorrelations of LSB plane at 4 spatial offsets
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = lsb_plane[: -abs(dy) or None, : -abs(dx) or None]
        b = lsb_plane[abs(dy):, abs(dx) if dx > 0 else None: dx if dx < 0 else None]
        # Align shapes for negative dx
        if dx < 0:
            a = lsb_plane[: -dy or None, -dx:]
            b = lsb_plane[dy:, : dx]
        m = min(a.shape[0], b.shape[0])
        n2 = min(a.shape[1], b.shape[1])
        a = a[:m, :n2].ravel()
        b = b[:m, :n2].ravel()
        if a.size > 0 and a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = 0.0
        feats.append(corr)

    # DCT LSB per-position (64 features)
    blocks_v, blocks_h, _, _ = dct_y.shape
    coeffs = dct_y.transpose(2, 3, 0, 1).reshape(8, 8, -1).astype(np.int32)
    for u in range(8):
        for v in range(8):
            c = coeffs[u, v]
            lsb_rate = float((np.abs(c) & 1).mean())
            feats.append(lsb_rate)

    # DCT LSB split by sign (per-position; 3 features × 64 = 192)
    for u in range(8):
        for v in range(8):
            c = coeffs[u, v]
            pos = c > 0
            neg = c < 0
            zero = c == 0
            lsb = np.abs(c) & 1
            feats.append(float(lsb[pos].mean()) if pos.any() else 0.0)
            feats.append(float(lsb[neg].mean()) if neg.any() else 0.0)
            feats.append(float(zero.mean()))  # zero-rate proxy for sparsity

    return np.array(feats, dtype=np.float32)


# ---------- pipeline ----------

def _crop_blocks(blocks: np.ndarray, target_bv: int = 64, target_bh: int = 64) -> np.ndarray:
    """Center-crop block-array to (target_bv, target_bh, 8, 8) layout."""
    bv, bh = blocks.shape[:2]
    if (bv, bh) == (target_bv, target_bh):
        return blocks
    ty = max(0, (bv - target_bv) // 2)
    tx = max(0, (bh - target_bh) // 2)
    return blocks[ty:ty + target_bv, tx:tx + target_bh]


def chroma_asymmetry_features(
    dct_y: np.ndarray,
    dct_cb: np.ndarray,
    dct_cr: np.ndarray,
    spatial_y: np.ndarray,
    spatial_cb: np.ndarray,
    spatial_cr: np.ndarray,
) -> np.ndarray:
    """Y/chroma asymmetry features. Phasm only modifies Y, so chroma serves as
    an in-image control: a feature that changes more between Y and chroma than
    expected from baseline natural-image asymmetry is a Phasm-modification
    fingerprint.
    """
    feats: list[float] = []
    # Per-DCT-position LSB-rate asymmetry: Y − Cb and Y − Cr (64 + 64 features)
    for chroma_dct in (dct_cb, dct_cr):
        c_y = dct_y.transpose(2, 3, 0, 1).reshape(8, 8, -1).astype(np.int32)
        c_c = chroma_dct.transpose(2, 3, 0, 1).reshape(8, 8, -1).astype(np.int32)
        for u in range(8):
            for v in range(8):
                ly = float((np.abs(c_y[u, v]) & 1).mean())
                lc = float((np.abs(c_c[u, v]) & 1).mean())
                feats.append(ly - lc)

    # Spatial LSB-rate asymmetry
    ly = float((spatial_y.astype(np.int32) & 1).mean())
    lcb = float((spatial_cb.astype(np.int32) & 1).mean())
    lcr = float((spatial_cr.astype(np.int32) & 1).mean())
    feats.append(ly - lcb)
    feats.append(ly - lcr)
    feats.append(lcb - lcr)
    feats.append(ly - 0.5 * (lcb + lcr))

    return np.array(feats, dtype=np.float32)


def extract_features_for_path(path: Path) -> np.ndarray:
    """Compute the full hand-crafted feature vector for one image.

    Auto-detects grayscale vs color. Returns a longer vector for color images
    (3 channels × per-channel features + chroma-asymmetry block).
    """
    pil = Image.open(path)
    if pil.size != (512, 512):
        w, h = pil.size
        left = (w - 512) // 2
        top = (h - 512) // 2
        pil = pil.crop((left, top, left + 512, top + 512))

    dct = jpeglib.read_dct(str(path))
    is_color = (dct.Cb is not None) and (dct.Cr is not None)

    if not is_color:
        spatial = np.asarray(pil.convert("L"), dtype=np.uint8)
        dct_y = _crop_blocks(dct.Y)
        spam = spam_features(spatial)
        hist = dct_histogram_features(dct_y)
        lsb = lsb_features(spatial, dct_y)
        return np.concatenate([spam, hist, lsb])

    # Color path
    ycbcr = np.asarray(pil.convert("YCbCr"), dtype=np.uint8)
    spatial_y = ycbcr[:, :, 0]
    spatial_cb = ycbcr[:, :, 1]
    spatial_cr = ycbcr[:, :, 2]

    dct_y = _crop_blocks(dct.Y)
    dct_cb = _crop_blocks(dct.Cb)
    dct_cr = _crop_blocks(dct.Cr)

    parts: list[np.ndarray] = []
    for spatial, dct_ch in ((spatial_y, dct_y), (spatial_cb, dct_cb), (spatial_cr, dct_cr)):
        parts.append(spam_features(spatial))
        parts.append(dct_histogram_features(dct_ch))
        parts.append(lsb_features(spatial, dct_ch))

    parts.append(chroma_asymmetry_features(
        dct_y, dct_cb, dct_cr, spatial_y, spatial_cb, spatial_cr,
    ))

    return np.concatenate(parts)


def _worker(args: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = args
    try:
        if dst.exists():
            return (dst.name, "cached")
        feats = extract_features_for_path(src)
        np.save(dst, feats.astype(np.float32))
        return (dst.name, "ok")
    except Exception as e:
        return (dst.name, f"ERR: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n1-dir", type=Path,
                   default=ROOT / "data" / "path4e_color" / "n1")
    p.add_argument("--n2-dir", type=Path,
                   default=ROOT / "data" / "path4e_color" / "n2")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "data" / "path4e_features_color")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only extract first N covers (smoke test)")
    args = p.parse_args()

    out_n1 = args.out_dir / "n1"
    out_n2 = args.out_dir / "n2"
    out_n1.mkdir(parents=True, exist_ok=True)
    out_n2.mkdir(parents=True, exist_ok=True)

    # Find paired covers
    n1_names = {p.name for p in args.n1_dir.glob("*.jpg")}
    n2_names = {p.name for p in args.n2_dir.glob("*.jpg")}
    paired = sorted(n1_names & n2_names)
    print(f"[extract] {len(paired)} paired covers")

    if args.limit > 0:
        paired = paired[: args.limit]
        print(f"[extract] limited to {len(paired)} covers")

    jobs: list[tuple[Path, Path]] = []
    for name in paired:
        stem = Path(name).stem
        jobs.append((args.n1_dir / name, out_n1 / f"{stem}.npy"))
        jobs.append((args.n2_dir / name, out_n2 / f"{stem}.npy"))

    print(f"[extract] {len(jobs)} jobs, {args.workers} workers")

    # Smoke test on first image to validate dimensions
    feats0 = extract_features_for_path(args.n1_dir / paired[0])
    print(f"[extract] feature vector length: {len(feats0)}")
    print(f"[extract] dtype: {feats0.dtype}")
    print(f"[extract] sample stats: min={feats0.min():.4f} max={feats0.max():.4f} "
          f"mean={feats0.mean():.4f}")

    results: dict[str, int] = {"ok": 0, "cached": 0, "err": 0}
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for name, status in tqdm(pool.imap_unordered(_worker, jobs, chunksize=4),
                                     total=len(jobs)):
                if status == "ok":
                    results["ok"] += 1
                elif status == "cached":
                    results["cached"] += 1
                else:
                    results["err"] += 1
                    print(f"[extract] {name}: {status}")
    else:
        for j in tqdm(jobs):
            name, status = _worker(j)
            if status == "ok":
                results["ok"] += 1
            elif status == "cached":
                results["cached"] += 1
            else:
                results["err"] += 1
                print(f"[extract] {name}: {status}")

    print(f"[extract] done: ok={results['ok']} cached={results['cached']} err={results['err']}")


if __name__ == "__main__":
    main()
