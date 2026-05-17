"""Gap #3 GFR+FLD baseline — FLD-ensemble classifier (Kodovský 2012).

Trains an ensemble of Fisher Linear Discriminants on random feature
subsets and reports the conventional steganalysis decision-error rate
$P_E = (P_{\\mathrm{FA}} + P_{\\mathrm{MD}})/2$ on a held-out test set.

Defaults follow Kodovský et al. 2012:
  L = 33 base learners
  d_sub = 600 features per learner (subset)
  bagging = 50% of training samples per learner
  decision = majority vote of sign(linear projection) across the ensemble

Input layout (from extract_handcrafted_cover_stego.py):
  data/<features-in>/cover/*.npy
  data/<features-in>/stego/*.npy

Each .npy is a 1D feature vector. Cover and stego dirs must share
filenames (one paired cover→stego per name).

Usage:
  python detectors/train_fld_ensemble.py \\
    --features-dir data/gfrfld_phasm_qf75_pf040 \\
    --out-dir       runs/2026-05-15-gfrfld-phasm-qf75-pf040 \\
    --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parent.parent


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return 0.5
    ranks = rankdata(scores)
    rs_pos = ranks[pos].sum()
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    return float((rs_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def split_paired(paired: list[str], seed: int, n_folds: int) -> np.ndarray:
    return np.array([
        int(hashlib.sha256(f"{n}|{seed}".encode()).hexdigest()[:8], 16) % n_folds
        for n in paired
    ])


def fld_ensemble_train(
    X_tr: np.ndarray, y_tr: np.ndarray,
    L: int, d_sub: int, bag_frac: float, rng: np.random.Generator,
) -> list[tuple[LinearDiscriminantAnalysis, np.ndarray]]:
    """Train L FLDs on random (sample-bagged, feature-subset) views."""
    n_train, n_features = X_tr.shape
    bag_size = max(2, int(round(bag_frac * n_train)))
    learners: list[tuple[LinearDiscriminantAnalysis, np.ndarray]] = []
    for _ in range(L):
        cols = rng.choice(n_features, size=min(d_sub, n_features), replace=False)
        rows = rng.choice(n_train, size=bag_size, replace=False)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_tr[rows][:, cols], y_tr[rows])
        learners.append((lda, cols))
    return learners


def fld_ensemble_score(
    X: np.ndarray,
    learners: list[tuple[LinearDiscriminantAnalysis, np.ndarray]],
) -> np.ndarray:
    """Mean stego-probability across L learners (used for AUC + PE)."""
    n = X.shape[0]
    accum = np.zeros(n, dtype=np.float64)
    for lda, cols in learners:
        probs = lda.predict_proba(X[:, cols])[:, 1]
        accum += probs
    return accum / len(learners)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--L", type=int, default=33,
                   help="Number of FLD base learners (Kodovský 2012: 33)")
    p.add_argument("--d-sub", type=int, default=600,
                   help="Feature-subset size per learner")
    p.add_argument("--bag-frac", type=float, default=0.5,
                   help="Bagging fraction per learner")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cover_dir = args.features_dir / "cover"
    stego_dir = args.features_dir / "stego"
    cover_names = {p.stem for p in cover_dir.glob("*.npy")}
    stego_names = {p.stem for p in stego_dir.glob("*.npy")}
    paired = sorted(cover_names & stego_names)
    print(f"[fld] paired cover+stego: {len(paired)}")

    cov = np.stack([np.load(cover_dir / f"{n}.npy") for n in paired])
    stg = np.stack([np.load(stego_dir / f"{n}.npy") for n in paired])
    X = np.concatenate([cov, stg], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(paired)), np.ones(len(paired))]).astype(np.int64)
    cover_idx = np.concatenate([np.arange(len(paired)), np.arange(len(paired))])

    # Handle non-finite entries (rare but possible from DCT extractor)
    if not np.all(np.isfinite(X)):
        n_bad = (~np.isfinite(X)).sum()
        print(f"[fld] WARNING: {n_bad} non-finite entries → 0")
        X[~np.isfinite(X)] = 0.0

    n_features = X.shape[1]
    print(f"[fld] X={X.shape}, features={n_features}")

    fold_per_cover = split_paired(paired, args.seed, args.n_folds)
    fold_idx = fold_per_cover[cover_idx]

    fold_aucs: list[float] = []
    fold_pes: list[float] = []
    fold_accs: list[float] = []
    rng = np.random.default_rng(args.seed)

    for f in range(args.n_folds):
        test_mask = (fold_idx == f)
        train_mask = ~test_mask
        learners = fld_ensemble_train(
            X[train_mask], y[train_mask], args.L, args.d_sub, args.bag_frac, rng,
        )
        scores_te = fld_ensemble_score(X[test_mask], learners)
        a = auc_score(scores_te, y[test_mask])
        preds = (scores_te > 0.5).astype(int)
        acc = float((preds == y[test_mask]).mean())
        fa = float(((preds == 1) & (y[test_mask] == 0)).sum() / max(1, (y[test_mask] == 0).sum()))
        md = float(((preds == 0) & (y[test_mask] == 1)).sum() / max(1, (y[test_mask] == 1).sum()))
        pe = (fa + md) / 2
        fold_aucs.append(a)
        fold_pes.append(pe)
        fold_accs.append(acc)
        print(f"[fld] fold {f}: n_tr={int(train_mask.sum())} n_te={int(test_mask.sum())} "
              f"AUC={a:.4f} PE={pe:.4f} acc={acc:.4f} FA={fa:.4f} MD={md:.4f}")

    mean_auc, std_auc = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    mean_pe, std_pe = float(np.mean(fold_pes)), float(np.std(fold_pes))
    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    print(f"\n[fld] === RESULTS ===")
    print(f"[fld] AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"[fld] PE:  {mean_pe:.4f} ± {std_pe:.4f}")
    print(f"[fld] Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    results = {
        "phase": "gap3_gfr_fld_ensemble",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_paired_covers": len(paired),
        "n_features": int(n_features),
        "n_folds": args.n_folds,
        "L": args.L,
        "d_sub": args.d_sub,
        "bag_frac": args.bag_frac,
        "seed": args.seed,
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "auc_per_fold": fold_aucs,
        "pe_mean": mean_pe,
        "pe_std": std_pe,
        "pe_per_fold": fold_pes,
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "acc_per_fold": fold_accs,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[fld] saved {args.out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
