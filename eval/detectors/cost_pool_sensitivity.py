"""E5: Cost-pool fraction sensitivity sweep.

Phasm's shadow embedding uses cost-pool fractions
[5%, 10%, 20%, 50%, 100%] (indices 0..4, COST_FRACTIONS = [20,10,5,2,1]).
The Phase 4f white-box attacker extracts 7 LSB statistics within each
fraction's candidate set, giving 35 features per channel.

This script trains a classifier on EACH FRACTION'S features alone (7
features per slice, or 7 × 5 channels = 35 for color including asymmetry)
to identify which cost-pool fraction carries the strongest signal in the
attacker's view.

Interpretation:
  - AUC near 0.5 at fraction k  →  cost-pool at fraction k is well-hidden
  - AUC well above 0.5 at fraction k  →  cost-pool at fraction k leaks
  - The fraction Phasm actually USED for embedding will show the strongest
    signal (because that's where modifications landed)

Output: runs/<date>-cost-pool-sensitivity/RESULTS.md + json

Usage:
  python detectors/cost_pool_sensitivity.py \\
    --features-dir data/path4f_features \\
    --out-dir runs/2026-05-14-cost-pool-sensitivity-color \\
    --layout color
  python detectors/cost_pool_sensitivity.py \\
    --features-dir data/path4f_features_gray \\
    --out-dir runs/2026-05-14-cost-pool-sensitivity-gray \\
    --layout gray
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from xgboost import XGBClassifier

# Phasm's cost fractions, in order (COST_FRACTIONS = [20, 10, 5, 2, 1])
# These define pool sizes as nzAC // fraction.
POOL_SIZES_PCT = [5, 10, 20, 50, 100]  # 5%, 10%, 20%, 50%, 100% of nzAC
N_STATS_PER_FRACTION = 7  # pool_lsb_rate, chi2, neg_lsb, pos_lsb, outside_lsb, diff, pool_size


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return 0.5
    ranks = rankdata(scores)
    rank_sum_pos = ranks[pos].sum()
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def split_paired(paired: list[str], seed: int, n_folds: int) -> np.ndarray:
    return np.array([
        int(hashlib.sha256(f"{n}|{seed}".encode()).hexdigest()[:8], 16) % n_folds
        for n in paired
    ])


def fraction_slice_indices(layout: str, frac_idx: int) -> list[int]:
    """Return the feature indices corresponding to a given fraction."""
    n = N_STATS_PER_FRACTION
    start = frac_idx * n
    end = start + n
    if layout == "gray":
        # Only Y features: indices [k*7 : k*7+7]
        return list(range(start, end))
    # Color: Y / Cb / Cr / Y-Cb asym / Y-Cr asym, each 35-long, in that order
    bases = [0, 35, 70, 105, 140]
    out = []
    for b in bases:
        out.extend(range(b + start, b + end))
    return out


def train_one_slice(X: np.ndarray, y: np.ndarray, cover_idx: np.ndarray,
                    paired: list[str], seed: int, n_folds: int = 5,
                    max_depth: int = 8, lr: float = 0.1,
                    n_est: int = 500) -> dict:
    fold_per_cover = split_paired(paired, seed, n_folds)
    fold_idx = fold_per_cover[cover_idx]
    folds = []
    for f in range(n_folds):
        test = (fold_idx == f)
        clf = XGBClassifier(
            n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=8, random_state=seed + f,
        )
        clf.fit(X[~test], y[~test])
        scores = clf.predict_proba(X[test])[:, 1]
        folds.append(auc_score(scores, y[test]))
    return {
        "auc_mean": float(np.mean(folds)),
        "auc_std": float(np.std(folds)),
        "auc_per_fold": [float(x) for x in folds],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--layout", choices=["color", "gray"], required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n1_dir = args.features_dir / "n1"
    n2_dir = args.features_dir / "n2"
    n1_names = {p.stem for p in n1_dir.glob("*.npy")}
    n2_names = {p.stem for p in n2_dir.glob("*.npy")}
    paired = sorted(n1_names & n2_names)
    print(f"[E5] {len(paired)} paired covers, layout={args.layout}")

    feats_n1 = np.stack([np.load(n1_dir / f"{p}.npy") for p in paired])
    feats_n2 = np.stack([np.load(n2_dir / f"{p}.npy") for p in paired])
    X_full = np.concatenate([feats_n1, feats_n2], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(paired)), np.ones(len(paired))]).astype(np.int64)
    cover_idx = np.concatenate([np.arange(len(paired)), np.arange(len(paired))])
    X_full[~np.isfinite(X_full)] = 0.0
    print(f"[E5] X={X_full.shape} y={y.shape} ({y.sum()} positives)")

    # Per-fraction AUC
    per_fraction = []
    for frac_idx, pct in enumerate(POOL_SIZES_PCT):
        idx = fraction_slice_indices(args.layout, frac_idx)
        X_slice = X_full[:, idx]
        res = train_one_slice(X_slice, y, cover_idx, paired, args.seed,
                              n_folds=args.n_folds)
        per_fraction.append({
            "fraction_pct": pct,
            "frac_idx": frac_idx,
            "n_features": len(idx),
            **res,
        })
        print(f"[E5] frac {pct:>4}%  ({len(idx):3d} features): "
              f"AUC={res['auc_mean']:.4f} +/- {res['auc_std']:.4f}")

    # Baseline (all features) for reference
    print(f"[E5] === full feature set baseline ===")
    full = train_one_slice(X_full, y, cover_idx, paired, args.seed,
                           n_folds=args.n_folds)
    print(f"[E5] all features ({X_full.shape[1]} features): "
          f"AUC={full['auc_mean']:.4f} +/- {full['auc_std']:.4f}")

    results = {
        "phase": "5_cost_pool_sensitivity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_paired_covers": len(paired),
        "layout": args.layout,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "per_fraction": per_fraction,
        "all_features_baseline": full,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[E5] saved {args.out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
