"""Phase 4e FULL — rich-model steganalysis features (SRMQ1/DCTR-style) for
shadow-count discrimination.

Adds three new feature groups to the Phase 4e minimal extractor:

- **SRMQ1-lite** (~1500 features per channel) — Spatial Rich Model with q=1:
  6 residual filters (1st-order H/V, 2nd-order H/V, KV, 3×3 square mean),
  each quantized + truncated at T=2 (5 states), with 1st-order Markov
  co-occurrence (25 features) and 2nd-order co-occurrence (125 features).
  Total per filter: 150 features × 6 filters = 900 features per channel.

- **DCTR-lite** (~512 features per channel) — DCT Residual:
  8 DCT-mode filters (low-frequency modes), each convolved with image →
  quantized residual → 8-block-sub-region histogram (8 bins × 8 = 64
  features per filter). 8 filters × 64 = 512 features per channel.

- **GFR-lite** (~480 features per channel) — Gabor Filter Residual:
  4 scales × 6 orientations = 24 Gabor filters, magnitude response
  quantized + truncated, histogrammed → 20 features per filter.
  24 filters × 20 features = 480 features per channel.

Total: ~1892 features per channel = ~5676 features for color JPEGs.

Output: per-image .npy in `data/path4e_features_rich_color/{n1,n2}/`.

Use alongside Phase 4e minimal features by concatenating both sets at
training time (see train_handcrafted_classifier_full.py).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import jpeglib
import numpy as np
from PIL import Image
from scipy import signal
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

T_SRM = 2          # SRMQ1 truncation
Q_SRM = 1          # SRMQ1 quantization
DCTR_T = 4         # DCTR truncation
DCTR_Q = 1         # DCTR quantization


# ---------- SRMQ1-lite ----------

def _kv_filter() -> np.ndarray:
    """Ker-Bohme 5×5 filter — high-frequency residual extraction."""
    return np.array([
        [-1, +2, -2, +2, -1],
        [+2, -6, +8, -6, +2],
        [-2, +8, -12, +8, -2],
        [+2, -6, +8, -6, +2],
        [-1, +2, -2, +2, -1],
    ], dtype=np.float32) / 12.0


def _square_3x3() -> np.ndarray:
    """3×3 square mean residual (subtracts local average)."""
    k = np.full((3, 3), 1 / 9, dtype=np.float32)
    k[1, 1] -= 1
    return k


def _residual(img: np.ndarray, kind: str) -> np.ndarray:
    """Compute one residual map. `img` is uint8 grayscale."""
    img = img.astype(np.float32)
    if kind == "h1":
        return img[:, 1:] - img[:, :-1]
    if kind == "v1":
        return img[1:, :] - img[:-1, :]
    if kind == "h2":
        return img[:, :-2] - 2 * img[:, 1:-1] + img[:, 2:]
    if kind == "v2":
        return img[:-2, :] - 2 * img[1:-1, :] + img[2:, :]
    if kind == "kv":
        k = _kv_filter()
        return signal.convolve2d(img, k, mode="valid", boundary="symm")
    if kind == "sq3":
        k = _square_3x3()
        return signal.convolve2d(img, k, mode="valid", boundary="symm")
    raise ValueError(kind)


def _qtruncate(r: np.ndarray, q: int = Q_SRM, t: int = T_SRM) -> np.ndarray:
    """SRMQ quantize-then-truncate. Returns int array in [-t, t]."""
    q_r = np.round(r / q)
    return np.clip(q_r, -t, t).astype(np.int32)


def _markov1(diffs: np.ndarray, t: int = T_SRM) -> np.ndarray:
    """1st-order Markov co-occurrence (along horizontal axis). Returns (2t+1)² features."""
    n = 2 * t + 1
    flat_a = (diffs[:, :-1].ravel() + t)
    flat_b = (diffs[:, 1:].ravel() + t)
    co = np.bincount(flat_a * n + flat_b, minlength=n * n).astype(np.float64)
    co = co.reshape(n, n)
    rows = co.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    return (co / rows).ravel()


def _markov2(diffs: np.ndarray, t: int = T_SRM) -> np.ndarray:
    """2nd-order Markov co-occurrence. Returns (2t+1)³ features."""
    n = 2 * t + 1
    a = (diffs[:, :-2].ravel() + t)
    b = (diffs[:, 1:-1].ravel() + t)
    c = (diffs[:, 2:].ravel() + t)
    idx = a * (n * n) + b * n + c
    co = np.bincount(idx, minlength=n ** 3).astype(np.float64)
    co = co.reshape(n, n, n)
    pair = co.sum(axis=2, keepdims=True)
    pair[pair == 0] = 1
    return (co / pair).ravel()


def srm_features_one_channel(img: np.ndarray) -> np.ndarray:
    """SRMQ1-lite for one grayscale channel. ~900 features."""
    parts: list[np.ndarray] = []
    for kind in ("h1", "v1", "h2", "v2", "kv", "sq3"):
        r = _residual(img, kind)
        q = _qtruncate(r)
        # 1st-order Markov on q
        parts.append(_markov1(q))
        # 2nd-order Markov on q
        parts.append(_markov2(q))
    return np.concatenate(parts).astype(np.float32)


# ---------- DCTR-lite ----------

def _dctr_bases() -> list[np.ndarray]:
    """8 low-frequency 8×8 DCT mode filters (used as convolution kernels)."""
    bases: list[np.ndarray] = []
    indices = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    for u, v in indices:
        F = np.zeros((8, 8), dtype=np.float32)
        for x in range(8):
            for y in range(8):
                F[x, y] = (
                    (1.0 if u == 0 else np.sqrt(2)) *
                    (1.0 if v == 0 else np.sqrt(2)) *
                    np.cos(np.pi * (2 * x + 1) * u / 16) *
                    np.cos(np.pi * (2 * y + 1) * v / 16)
                ) / 8.0
        bases.append(F)
    return bases


_DCTR_BASES = _dctr_bases()


def _dctr_residual(img: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """One DCTR residual."""
    return signal.convolve2d(img.astype(np.float32), filt, mode="valid")


def dctr_features_one_channel(img: np.ndarray) -> np.ndarray:
    """DCTR-lite: 8 mode filters × 64-bin (per 8×8 phase position) histogram.

    Output: 8 × (2*DCTR_T + 1)² = 8 × 81 = 648 features.
    """
    parts: list[np.ndarray] = []
    for filt in _DCTR_BASES:
        r = _dctr_residual(img, filt)
        q = np.clip(np.round(r / DCTR_Q), -DCTR_T, DCTR_T).astype(np.int32)
        # 8 phase positions × histogram of (2T+1) bins = (2T+1)² features per filter
        # Use a 2D phase grid: row phase mod 8, col phase mod 8 — flatten + histogram per (row_phase, col_phase) is too many.
        # Simpler: histogram entire residual into (2T+1)² 2D bins via 2D co-occurrence of adjacent residuals.
        # Marginal hist
        n = 2 * DCTR_T + 1
        marginal = np.bincount(q.ravel() + DCTR_T, minlength=n).astype(np.float32)
        marginal = marginal / max(1, marginal.sum())
        # 1st-order Markov as well
        flat_a = (q[:, :-1].ravel() + DCTR_T)
        flat_b = (q[:, 1:].ravel() + DCTR_T)
        co = np.bincount(flat_a * n + flat_b, minlength=n * n).astype(np.float32)
        co = co / max(1, co.sum())
        parts.append(marginal)
        parts.append(co)
    return np.concatenate(parts).astype(np.float32)


# ---------- GFR-lite ----------

def _gabor_kernel(scale: int, theta: float, ksize: int = 15) -> np.ndarray:
    """2D Gabor kernel, real part. Stable, deterministic."""
    sigma = scale * 0.8
    lam = scale * 2.0
    gamma = 0.5
    psi = 0.0
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)
    k = np.exp(-0.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2) * \
        np.cos(2 * np.pi * x_t / lam + psi)
    k -= k.mean()  # zero-DC
    n = np.linalg.norm(k)
    if n > 0:
        k /= n
    return k.astype(np.float32)


_GABOR_BANK = [
    _gabor_kernel(scale=s, theta=t * np.pi / 6)
    for s in (2, 3, 4, 6)
    for t in range(6)
]  # 24 kernels


def gfr_features_one_channel(img: np.ndarray) -> np.ndarray:
    """GFR-lite: 24 Gabor filters, each producing a 21-bin truncated histogram.

    Output: 24 × 21 = 504 features.
    """
    parts: list[np.ndarray] = []
    img_f = img.astype(np.float32)
    for k in _GABOR_BANK:
        r = signal.convolve2d(img_f, k, mode="valid")
        q = np.clip(np.round(r), -10, 10).astype(np.int32)
        hist = np.bincount(q.ravel() + 10, minlength=21).astype(np.float32)
        hist = hist / max(1, hist.sum())
        parts.append(hist)
    return np.concatenate(parts).astype(np.float32)


# ---------- pipeline ----------

def extract_rich_features_for_path(path: Path) -> np.ndarray:
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
        return np.concatenate([
            srm_features_one_channel(spatial),
            dctr_features_one_channel(spatial),
            gfr_features_one_channel(spatial),
        ])

    ycbcr = np.asarray(pil.convert("YCbCr"), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for ch in (0, 1, 2):
        c = ycbcr[:, :, ch]
        parts.append(srm_features_one_channel(c))
        parts.append(dctr_features_one_channel(c))
        parts.append(gfr_features_one_channel(c))
    return np.concatenate(parts)


def _worker(args: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = args
    try:
        if dst.exists():
            return (dst.name, "cached")
        feats = extract_rich_features_for_path(src)
        np.save(dst, feats.astype(np.float32))
        return (dst.name, "ok")
    except Exception as e:
        return (dst.name, f"ERR: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n1-dir", type=Path, default=ROOT / "data" / "path4e_color" / "n1")
    p.add_argument("--n2-dir", type=Path, default=ROOT / "data" / "path4e_color" / "n2")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "data" / "path4e_features_rich_color")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    out_n1 = args.out_dir / "n1"
    out_n2 = args.out_dir / "n2"
    out_n1.mkdir(parents=True, exist_ok=True)
    out_n2.mkdir(parents=True, exist_ok=True)

    n1_names = {p.name for p in args.n1_dir.glob("*.jpg")}
    n2_names = {p.name for p in args.n2_dir.glob("*.jpg")}
    paired = sorted(n1_names & n2_names)
    print(f"[rich] {len(paired)} paired covers")
    if args.limit:
        paired = paired[: args.limit]

    feats0 = extract_rich_features_for_path(args.n1_dir / paired[0])
    print(f"[rich] feature vector length: {len(feats0)}")
    print(f"[rich] dtype: {feats0.dtype} min={feats0.min():.4f} max={feats0.max():.4f}")

    jobs: list[tuple[Path, Path]] = []
    for name in paired:
        stem = Path(name).stem
        jobs.append((args.n1_dir / name, out_n1 / f"{stem}.npy"))
        jobs.append((args.n2_dir / name, out_n2 / f"{stem}.npy"))

    results = {"ok": 0, "cached": 0, "err": 0}
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for name, status in tqdm(pool.imap_unordered(_worker, jobs, chunksize=2),
                                     total=len(jobs)):
                if status == "ok":
                    results["ok"] += 1
                elif status == "cached":
                    results["cached"] += 1
                else:
                    results["err"] += 1
                    print(f"[rich] {name}: {status}")
    else:
        for j in tqdm(jobs):
            name, status = _worker(j)
            results[status if status in results else "err"] += 1

    print(f"[rich] done: ok={results['ok']} cached={results['cached']} err={results['err']}")


if __name__ == "__main__":
    main()
