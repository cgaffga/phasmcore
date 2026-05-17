"""Phase 4f — Cost-pool-aware white-box informed attack on Phasm Ghost shadows.

Open-source threat model: the adversary has Phasm's source code (the engine is
open source) and can:

1. Replicate the J-UNIWARD cost computation on the stego image's Y channel.
2. Identify the EXACT candidate DCT positions Phasm would use for shadow
   embedding — the cost-pool (cheapest 5%, 10%, 20%, 50%, 100% of nzAC).
3. Check ONLY those positions for LSB anomalies.

This is structurally stronger than generic SPAM/DCTR/GFR features (Phase 4e):
- Phase 4e: scan ALL coefficient positions for LSB-rate skew → AUC 0.70
- Phase 4f: scan ONLY the pool positions Phasm would use → expected AUC > 0.75
  if the cost-pool design is the leak channel.

For each image, extract LSB statistics per cost-pool fraction. Phasm's source
defines COST_FRACTIONS = [20, 10, 5, 2, 1] (= cheapest 5%, 10%, 20%, 50%, 100%
of all positions).

Per fraction:
- pool_lsb_rate: mean LSB of |coeff| within pool
- pool_lsb_chi2: χ² of LSB distribution within pool vs uniform 0.5
- pool_neg_lsb_rate: mean LSB of |coeff| for negative coefficients in pool
- pool_pos_lsb_rate: mean LSB of |coeff| for positive coefficients in pool
- outside_lsb_rate: mean LSB of |coeff| outside pool
- diff_lsb_rate: pool − outside
- pool_size: number of positions (deterministic from fraction)

5 fractions × 7 features = 35 features per image.

Plus chroma-control: same features computed for Cb and Cr (which Phasm does NOT
modify). 3 × 35 = 105 features total.

The contrast (Y pool LSB skew − Cb pool LSB skew) is the strongest possible
signal of Phasm modification: Y is modified at exactly the cost-pool positions,
Cb/Cr are not modified at all, so the asymmetry isolates the Phasm signature.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import jpeglib
import numpy as np
from conseal.juniward import _costmap
from conseal.juniward._costmap import Implementation
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# From Phasm core/src/stego/ghost/shadow.rs:81
COST_FRACTIONS = [20, 10, 5, 2, 1]


def compute_juniward_cost(spatial: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """Compute J-UNIWARD cost map, shape (blocks_v, blocks_h, 8, 8)."""
    return _costmap.compute_cost(
        x0=spatial.astype(np.float64),
        qt=qt.astype(np.float64),
        implementation=Implementation.JUNIWARD_FIX_OFF_BY_ONE,
        sigma=2 ** -6,
        dtype=np.float64,
    )


def channel_pool_features(dct_y: np.ndarray, cost_y: np.ndarray) -> np.ndarray:
    """Extract per-fraction LSB-pool features for one channel.

    Phasm's cost-pool selection:
    1. Take all nzAC positions (coefficient != 0, NOT DC).
    2. Sort by cost (cheapest first).
    3. For each fraction f, the pool = cheapest len/f positions.

    Returns 5 × 7 = 35 features.
    """
    # Flatten DCT and cost to per-coefficient arrays
    coeffs = dct_y.reshape(-1)
    costs = cost_y.reshape(-1)

    # Build the "AC mask": exclude DC (every 64th position is DC at pos 0 of block)
    n = len(coeffs)
    ac_mask = np.ones(n, dtype=bool)
    ac_mask[::64] = False  # DC positions
    # Also exclude zero coefficients (Phasm uses nzAC only)
    nzAC_mask = ac_mask & (coeffs != 0)
    nzAC_indices = np.flatnonzero(nzAC_mask)

    if len(nzAC_indices) < 100:
        # Not enough positions — return zeros
        return np.zeros(5 * 7, dtype=np.float32)

    # Costs of nzAC positions (the same cost-sort Phasm uses)
    nzAC_costs = costs[nzAC_indices]
    sort_order = np.argsort(nzAC_costs)
    sorted_nzAC = nzAC_indices[sort_order]
    n_nzAC = len(sorted_nzAC)

    # LSB of |coeff| (Phasm modifies LSB of magnitudes)
    abs_coeffs = np.abs(coeffs)
    lsbs = (abs_coeffs & 1).astype(np.int32)
    signs = np.sign(coeffs.astype(np.int32))

    feats: list[float] = []
    for fraction in COST_FRACTIONS:
        pool_size = n_nzAC // fraction
        if pool_size < 10:
            feats.extend([0.0] * 7)
            continue

        pool_idx = sorted_nzAC[:pool_size]
        outside_idx = sorted_nzAC[pool_size:]

        pool_lsbs = lsbs[pool_idx]
        pool_signs = signs[pool_idx]
        outside_lsbs = lsbs[outside_idx]

        pool_lsb_rate = float(pool_lsbs.mean())
        pool_neg_lsb_rate = float(pool_lsbs[pool_signs < 0].mean()) if (pool_signs < 0).any() else 0.0
        pool_pos_lsb_rate = float(pool_lsbs[pool_signs > 0].mean()) if (pool_signs > 0).any() else 0.0
        outside_lsb_rate = float(outside_lsbs.mean()) if len(outside_lsbs) else 0.0
        diff_lsb_rate = pool_lsb_rate - outside_lsb_rate

        # χ² vs uniform 0.5 in pool
        c0 = float((pool_lsbs == 0).sum())
        c1 = float((pool_lsbs == 1).sum())
        m = c0 + c1
        exp = m / 2 if m > 0 else 1.0
        pool_chi2 = ((c0 - exp) ** 2 + (c1 - exp) ** 2) / exp if exp > 0 else 0.0

        feats.append(pool_lsb_rate)
        feats.append(pool_chi2)
        feats.append(pool_neg_lsb_rate)
        feats.append(pool_pos_lsb_rate)
        feats.append(outside_lsb_rate)
        feats.append(diff_lsb_rate)
        feats.append(float(pool_size))

    return np.array(feats, dtype=np.float32)


def cross_channel_asymmetry(
    y_feats: np.ndarray, cb_feats: np.ndarray, cr_feats: np.ndarray,
) -> np.ndarray:
    """Y − Cb and Y − Cr asymmetry features.

    Since Phasm only modifies Y (Cb/Cr are unmodified cover), the Y vs chroma
    contrast is the cleanest signal of Phasm shadow embedding.

    Returns 5 × 7 = 35 features (one per fraction × {Y−Cb, Y−Cr split-by-feature}).
    """
    # For each of the 5 fractions × 7 features, compute Y-vs-chroma
    return np.concatenate([y_feats - cb_feats, y_feats - cr_feats], dtype=np.float32)


def extract_cost_pool_features_for_path(path: Path) -> np.ndarray:
    """Full Phase 4f feature vector for one stego image.

    For color: 3 × 35 (per-channel) + 2 × 35 (Y−Cb, Y−Cr asymmetry) = 175 features.
    For grayscale: 1 × 35 = 35 features.
    """
    pil = Image.open(path)
    if pil.size != (512, 512):
        w, h = pil.size
        pil = pil.crop(((w - 512) // 2, (h - 512) // 2,
                       (w - 512) // 2 + 512, (h - 512) // 2 + 512))

    dct = jpeglib.read_dct(str(pil.filename if hasattr(pil, "filename") else path))
    is_color = (dct.Cb is not None) and (dct.Cr is not None)

    ycbcr = np.asarray(pil.convert("YCbCr"), dtype=np.float64)
    spatial_y = ycbcr[:, :, 0]

    cost_y = compute_juniward_cost(spatial_y, dct.qt[0])
    y_feats = channel_pool_features(dct.Y, cost_y)

    if not is_color:
        return y_feats

    spatial_cb = ycbcr[:, :, 1]
    spatial_cr = ycbcr[:, :, 2]
    # Chroma cost: use chroma qt (usually qt[1])
    qt_chroma = dct.qt[1] if len(dct.qt) > 1 else dct.qt[0]
    cost_cb = compute_juniward_cost(spatial_cb, qt_chroma)
    cost_cr = compute_juniward_cost(spatial_cr, qt_chroma)
    cb_feats = channel_pool_features(dct.Cb, cost_cb)
    cr_feats = channel_pool_features(dct.Cr, cost_cr)

    asym = cross_channel_asymmetry(y_feats, cb_feats, cr_feats)
    return np.concatenate([y_feats, cb_feats, cr_feats, asym], dtype=np.float32)


def _worker(args: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = args
    try:
        if dst.exists():
            return (dst.name, "cached")
        feats = extract_cost_pool_features_for_path(src)
        np.save(dst, feats.astype(np.float32))
        return (dst.name, "ok")
    except Exception as e:
        return (dst.name, f"ERR: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n1-dir", type=Path, default=ROOT / "data" / "path4e_color" / "n1")
    p.add_argument("--n2-dir", type=Path, default=ROOT / "data" / "path4e_color" / "n2")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "data" / "path4f_features")
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
    print(f"[4f] {len(paired)} paired covers")
    if args.limit:
        paired = paired[: args.limit]

    feats0 = extract_cost_pool_features_for_path(args.n1_dir / paired[0])
    print(f"[4f] feature vector length: {len(feats0)}")
    print(f"[4f] dtype: {feats0.dtype} min={feats0.min():.4f} max={feats0.max():.4f}")

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
                    print(f"[4f] {name}: {status}")
    else:
        for j in tqdm(jobs):
            name, status = _worker(j)
            results[status if status in results else "err"] += 1

    print(f"[4f] done: ok={results['ok']} cached={results['cached']} err={results['err']}")


if __name__ == "__main__":
    main()
