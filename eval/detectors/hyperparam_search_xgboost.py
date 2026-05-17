"""Phase 4e FULL — XGBoost hyperparameter search on the Phase 4e minimal feature set.

Grid: max_depth × learning_rate × n_estimators × subsample.
Inner 5-fold CV, paired-by-cover (same split as minimal). Reports best AUC,
best params, and full grid.

The search is over reasonable XGBoost regularization knobs; goal is to test
whether the AUC ceiling on the 11,214 hand-crafted features moves above the
default-XGBoost 0.685 baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    all_s = np.concatenate([neg, pos])
    ranks = rankdata(all_s)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def split_paired(paired: list[str], seed: int = 2026, n_folds: int = 5) -> np.ndarray:
    keys = np.array([int(hashlib.sha256(f"{n_}|{seed}".encode()).hexdigest()[:8], 16)
                     for n_ in paired])
    order = np.argsort(keys)
    folds = np.empty(len(paired), dtype=int)
    for rank, idx in enumerate(order):
        folds[idx] = rank % n_folds
    return folds


def evaluate_params(X: np.ndarray, y: np.ndarray, fold_idx: np.ndarray,
                    params: dict, n_folds: int, seed: int) -> tuple[float, float, list[float]]:
    fold_aucs: list[float] = []
    for fold in range(n_folds):
        test_mask = (fold_idx == fold)
        X_tr, y_tr = X[~test_mask], y[~test_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        clf = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=8,
            random_state=seed + fold,
        )
        clf.fit(X_tr, y_tr)
        scores_te = clf.predict_proba(X_te)[:, 1]
        fold_aucs.append(auc_score(scores_te, y_te))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), fold_aucs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-dir", type=Path,
                   default=ROOT / "data" / "path4e_features_color")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "runs" / "2026-05-11-path4e-handcrafted-full")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n1_dir = args.features_dir / "n1"
    n2_dir = args.features_dir / "n2"

    n1_names = {p.stem for p in n1_dir.glob("*.npy")}
    n2_names = {p.stem for p in n2_dir.glob("*.npy")}
    paired = sorted(n1_names & n2_names)
    print(f"[search] {len(paired)} paired covers")

    X1 = np.stack([np.load(n1_dir / f"{n}.npy") for n in paired])
    X2 = np.stack([np.load(n2_dir / f"{n}.npy") for n in paired])
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(paired), int), np.ones(len(paired), int)])
    cover_idx = np.concatenate([np.arange(len(paired)), np.arange(len(paired))])

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    fold_per_cover = split_paired(paired, seed=args.seed, n_folds=args.n_folds)
    fold_idx = fold_per_cover[cover_idx]

    print(f"[search] X shape: {X.shape}, y: {y.shape} ({int(y.sum())} positives)")

    # Hyperparameter grid (~36 combos)
    grid = {
        "max_depth":     [4, 6, 8, 10],
        "learning_rate": [0.05, 0.1, 0.3],
        "n_estimators":  [100, 300, 500],
        "subsample":     [1.0],     # leave column subsampling separate
    }
    combos = list(itertools.product(*grid.values()))
    print(f"[search] {len(combos)} parameter combinations × 5 folds = "
          f"{len(combos) * args.n_folds} model fits")

    keys = list(grid.keys())
    results: list[dict] = []
    t0 = time.time()
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        ts = time.time()
        mean_auc, std_auc, fold_aucs = evaluate_params(X, y, fold_idx,
                                                       params, args.n_folds, args.seed)
        dur = time.time() - ts
        results.append({
            **params, "auc_mean": mean_auc, "auc_std": std_auc,
            "auc_per_fold": fold_aucs, "wall_s": dur,
        })
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"[search] {i+1:2d}/{len(combos)}  "
              f"d={params['max_depth']} lr={params['learning_rate']:>4} "
              f"n={params['n_estimators']:>4} sub={params['subsample']:.1f}  "
              f"AUC={mean_auc:.4f}±{std_auc:.4f}  "
              f"({dur:5.1f}s, ETA {eta/60:.1f}m)")

    # Sort by mean AUC desc
    results.sort(key=lambda r: -r["auc_mean"])
    print(f"\n[search] === TOP 10 ===")
    for r in results[:10]:
        print(f"  AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
              f"d={r['max_depth']:2d} lr={r['learning_rate']:>4} "
              f"n={r['n_estimators']:>4} sub={r['subsample']:.1f}")

    best = results[0]
    print(f"\n[search] === BEST ===")
    print(f"  AUC: {best['auc_mean']:.4f} ± {best['auc_std']:.4f}")
    print(f"  Params: max_depth={best['max_depth']} lr={best['learning_rate']} "
          f"n_estimators={best['n_estimators']} subsample={best['subsample']}")

    # Persist
    summary = {
        "phase": "4e_full_hyperparam_search",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_paired_covers": len(paired),
        "n_features": int(X.shape[1]),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "grid": grid,
        "n_combos": len(combos),
        "best": best,
        "all_results": results,
    }
    (args.out_dir / "hyperparam_search.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[search] saved {args.out_dir / 'hyperparam_search.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
