"""E18 Stage 2 — Cost-pool-aware structural quantitative estimator.

A passive-warden attacker with stego-only knowledge:
  1. Computes the J-UNIWARD cost map directly from the stego (cost is
     stable under +/-1 DCT modifications; same conseal implementation
     Phasm uses internally, JUNIWARD_FIX_OFF_BY_ONE).
  2. Selects Tier-1 = the cheapest `tier1_fraction` of NZ AC positions
     (publicly knowable, no passphrase).
  3. Computes a Westfeld-Pfitzmann chi-square statistic on
     (2k, 2k+1) coefficient-pair counts within Tier-1. LSB embedding
     equalises these pairs, so chi-square decreases with embedding rate.

We also compute an ORACLE flip-count statistic (cover known) as the
upper-bound a perfect cover-aware quantitative attacker could reach.
This separates the "cover-unknown" tax from any algorithmic defence
the cost-pool selection itself provides.

Output: runs/<date>-e18-quant-stego/stage2_results.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import conseal as cl
import jpeglib
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / f"runs/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-e18-quant-stego"
COVER_DIR = ROOT / "data/bossbase/jpeg_qf75"
SHADOW_DIRS = {n: ROOT / f"data/path3_shadow_eval/shadow_n{n}" for n in range(5)}

TIER1_FRACTION = 0.20  # 20% — matches the Phasm default
K_MAX = 12              # number of (2k, 2k+1) pairs to test


def compute_tier1_mask(dct_y: np.ndarray, spatial_y: np.ndarray, qt: np.ndarray,
                       fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Tier-1 mask: cheapest `fraction` of NZ AC positions in Y."""
    rho_p1, rho_m1 = cl.juniward.compute_cost_adjusted(
        x0=spatial_y, y0=dct_y, qt=qt,
        implementation=cl.juniward.JUNIWARD_FIX_OFF_BY_ONE,
    )
    cost = np.minimum(rho_p1, rho_m1)
    nzac = dct_y != 0
    nzac[:, :, 0, 0] = False  # exclude DC
    if not nzac.any():
        return np.zeros_like(nzac, dtype=bool), cost
    threshold = np.percentile(cost[nzac], fraction * 100)
    tier1 = nzac & (cost <= threshold)
    return tier1, cost


def chi_square_in_tier1(dct_y: np.ndarray, tier1_mask: np.ndarray,
                        k_max: int = K_MAX) -> tuple[float, int, int]:
    """Westfeld-Pfitzmann chi-square on |coefficient| pairs (2k-1, 2k)
    inside Tier-1. Lower chi_sq → more LSB embedding (closer to pair equality)."""
    values = np.abs(dct_y[tier1_mask]).flatten()
    chi_sq = 0.0
    pairs_used = 0
    for k in range(1, k_max + 1):
        n_odd = int((values == 2 * k - 1).sum())
        n_even = int((values == 2 * k).sum())
        total = n_odd + n_even
        if total < 10:
            continue
        expected = total / 2.0
        chi_sq += ((n_odd - expected) ** 2 + (n_even - expected) ** 2) / expected
        pairs_used += 1
    return chi_sq / max(pairs_used, 1), pairs_used, int(tier1_mask.sum())


def oracle_lsb_flips_in_tier1(cover_y: np.ndarray, stego_y: np.ndarray,
                              tier1_mask: np.ndarray) -> int:
    """ORACLE: actual LSB-1 changes in Tier-1 positions (requires cover)."""
    diff_lsb = (cover_y & 1) ^ (stego_y & 1)
    return int((diff_lsb & tier1_mask).sum())


def universal_subset() -> list[str]:
    names_per_n = {n: set(p.name for p in SHADOW_DIRS[n].glob("*.jpg")) for n in range(5)}
    return sorted(set.intersection(*names_per_n.values()))


def analyse_image(name: str) -> dict:
    """Compute Stage 2 statistics for a single cover and its 5 stego variants."""
    cover_path = COVER_DIR / name
    cover_dct = jpeglib.read_dct(str(cover_path)).Y
    cover_spatial = jpeglib.read_spatial(str(cover_path)).spatial[..., 0]
    cover_qt = jpeglib.read_dct(str(cover_path)).qt[0]

    # Tier-1 from COVER (the attacker would compute from stego; we use cover
    # as the reference for "what Phasm selected" since the cover defines the
    # cost map Phasm uses for embedding decisions. We separately compute
    # Tier-1 from each stego below for the realistic attack.)
    tier1_from_cover, _ = compute_tier1_mask(cover_dct, cover_spatial, cover_qt, TIER1_FRACTION)

    chi_per_n = {}
    oracle_per_n = {}
    chi_from_stego_per_n = {}

    # N=-1 sentinel: pure cover (no Phasm modification at all)
    cover_chi, _, _ = chi_square_in_tier1(cover_dct, tier1_from_cover)
    chi_per_n["cover"] = float(cover_chi)
    chi_from_stego_per_n["cover"] = float(cover_chi)
    oracle_per_n["cover"] = 0

    for n in range(5):
        stego_path = SHADOW_DIRS[n] / name
        stego_dct = jpeglib.read_dct(str(stego_path)).Y
        stego_spatial = jpeglib.read_spatial(str(stego_path)).spatial[..., 0]

        # Two Tier-1 perspectives:
        # (i) ORACLE (cover-known): uses cover Tier-1 mask → exact attacker would have if they had cover
        # (ii) STEGO-only (realistic): recompute Tier-1 from the stego
        tier1_from_stego, _ = compute_tier1_mask(stego_dct, stego_spatial, cover_qt, TIER1_FRACTION)

        chi_oracle, _, _ = chi_square_in_tier1(stego_dct, tier1_from_cover)
        chi_realistic, _, _ = chi_square_in_tier1(stego_dct, tier1_from_stego)
        flips_oracle = oracle_lsb_flips_in_tier1(cover_dct, stego_dct, tier1_from_cover)

        chi_per_n[n] = float(chi_oracle)
        chi_from_stego_per_n[n] = float(chi_realistic)
        oracle_per_n[n] = flips_oracle

    return {
        "name": name,
        "chi_oracle_tier1": chi_per_n,        # cover-known Tier-1, chi statistic
        "chi_realistic_tier1": chi_from_stego_per_n,  # stego-derived Tier-1
        "oracle_flips_tier1": oracle_per_n,  # cover-known LSB-flip count
    }


def aggregate(per_image: list[dict]) -> dict:
    """Compute Spearman ρ + per-image ρ + linear MAE + rank-2 accuracy for each estimator."""
    estimators = ["chi_oracle_tier1", "chi_realistic_tier1", "oracle_flips_tier1"]
    out = {}
    for est in estimators:
        # Build point cloud: only N=0..4 (skip "cover" sentinel for the N-correlation)
        pts = []  # (image_name, N, statistic)
        for rec in per_image:
            for n in range(5):
                v = rec[est].get(n)
                if v is not None:
                    pts.append((rec["name"], n, float(v)))
        if not pts:
            out[est] = {"error": "no points"}
            continue
        stats_arr = np.array([p[2] for p in pts])
        ns = np.array([p[1] for p in pts])

        rho_pooled, p_pooled = stats.spearmanr(stats_arr, ns)

        # Per-image Spearman
        per_img_rhos = []
        for rec in per_image:
            n_present = sorted(k for k in rec[est].keys() if isinstance(k, int))
            if len(n_present) < 3:
                continue
            x = np.array(n_present, dtype=float)
            y = np.array([rec[est][n] for n in n_present], dtype=float)
            if y.std() == 0:
                continue
            r, _ = stats.spearmanr(y, x)
            if not np.isnan(r):
                per_img_rhos.append(r)
        per_img_rhos = np.array(per_img_rhos)

        # Linear regressor on point cloud
        rng = np.random.default_rng(42)
        names = sorted({p[0] for p in pts})
        perm = rng.permutation(len(names))
        train_names = set(np.array(names)[perm[: len(names) // 2]])
        train_pts = [(s, n) for (nm, n, s) in pts if nm in train_names]
        test_pts = [(s, n) for (nm, n, s) in pts if nm not in train_names]
        if not train_pts or not test_pts:
            out[est] = {"error": "split issue"}
            continue
        A = np.array([[s, 1] for s, _ in train_pts])
        b = np.array([n for _, n in train_pts])
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a_, b_ = coef
        test_s = np.array([s for s, _ in test_pts])
        test_n = np.array([n for _, n in test_pts])
        pred = a_ * test_s + b_
        mae = float(np.mean(np.abs(pred - test_n)))

        # Rank-2 accuracy: pairs of (statistic, N) where N differs
        correct = 0
        total = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if pts[i][1] == pts[j][1]:
                    continue
                pred_i = a_ * pts[i][2] + b_
                pred_j = a_ * pts[j][2] + b_
                truth_i_lt_j = pts[i][1] < pts[j][1]
                pred_i_lt_j = pred_i < pred_j
                if truth_i_lt_j == pred_i_lt_j:
                    correct += 1
                total += 1
        rank_acc = correct / total if total else float("nan")

        out[est] = {
            "n_points": len(pts),
            "rho_pooled": float(rho_pooled),
            "rho_pooled_p": float(p_pooled),
            "rho_perimage_mean": float(per_img_rhos.mean()) if len(per_img_rhos) else None,
            "rho_perimage_n_used": int(len(per_img_rhos)),
            "linear_mae_in_N_units": mae,
            "linear_a": float(a_),
            "linear_b": float(b_),
            "rank_2_accuracy": rank_acc,
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universal = universal_subset()
    print(f"[E18-S2] Universal subset: {len(universal)} images (N=0..4 all present)")
    print(f"[E18-S2] Tier-1 fraction = {TIER1_FRACTION}, k_max = {K_MAX}")
    print()

    per_image = []
    for i, name in enumerate(universal):
        try:
            rec = analyse_image(name)
            per_image.append(rec)
            if (i + 1) % 5 == 0 or i == len(universal) - 1:
                print(f"  [{i+1}/{len(universal)}] processed {name}")
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")

    summary = aggregate(per_image)

    out = {
        "config": {
            "tier1_fraction": TIER1_FRACTION,
            "k_max": K_MAX,
            "n_universal": len(universal),
            "n_processed": len(per_image),
        },
        "summary": summary,
        "per_image": per_image,
    }
    (OUT_DIR / "stage2_results.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"[E18-S2] wrote {OUT_DIR / 'stage2_results.json'}")
    print()

    print("=" * 100)
    print(f"{'Estimator':<28} {'rho_pool':>10} {'rho_perimg':>12} {'lin MAE (N)':>13} {'rank-2 acc':>11}")
    print("-" * 100)
    label = {
        "chi_oracle_tier1": "chi^2 (cover-known T1)",
        "chi_realistic_tier1": "chi^2 (stego-only T1)",
        "oracle_flips_tier1": "ORACLE (cover-known flip count)",
    }
    for est, r in summary.items():
        if "error" in r:
            continue
        print(
            f"{label.get(est, est):<28} {r['rho_pooled']:>+10.3f} "
            f"{r['rho_perimage_mean']:>+12.3f} {r['linear_mae_in_N_units']:>13.3f} "
            f"{r['rank_2_accuracy']:>11.3f}"
        )
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
