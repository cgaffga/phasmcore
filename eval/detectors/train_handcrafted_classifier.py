"""Phase 4e MINIMAL: train XGBoost classifier on hand-crafted features for
shadow-count discrimination (n=1 vs n=2 Phasm Ghost shadows).

5-fold cross-validation, paired-by-cover. Reports:
- Mean AUC ± std across folds
- Top-K feature importances (gain)
- Confusion matrix at threshold 0.5
- Optional per-fold per-image scores for downstream analysis

Inputs:
- features/<cover_name>.npy from extract_hand_crafted_features.py
  Two subdirectories: n1/ and n2/.

Adversary model: feature-space attack (non-CNN), strict comparison to
Phase 4c+4d CNN result on the equivalent data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U rank-based ROC AUC (no sklearn dep on scoring)."""
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_s = np.concatenate([neg, pos])
    ranks = rankdata(all_s)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def split_paired(paired: list[str], seed: int = 2026) -> np.ndarray:
    """Deterministic by-cover-name fold assignment.

    Returns array of fold indices [0..n_folds-1] with one entry per paired cover.
    Uses sha256 of cover name + seed for stable assignment.
    """
    n = len(paired)
    keys = np.array([
        int(hashlib.sha256(f"{n_}|{seed}".encode()).hexdigest()[:8], 16)
        for n_ in paired
    ])
    order = np.argsort(keys)
    folds = np.empty(n, dtype=int)
    for rank, idx in enumerate(order):
        folds[idx] = rank % 5
    return folds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-dir", type=Path,
                   default=ROOT / "data" / "path4e_features_color")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "runs" / "2026-05-11-path4e-handcrafted-minimal")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=20,
                   help="Top-K feature importances to report")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n1_dir = args.features_dir / "n1"
    n2_dir = args.features_dir / "n2"
    if not n1_dir.exists() or not n2_dir.exists():
        print(f"[trainer] missing {n1_dir} or {n2_dir}", file=sys.stderr)
        return 1

    n1_names = {p.stem for p in n1_dir.glob("*.npy")}
    n2_names = {p.stem for p in n2_dir.glob("*.npy")}
    paired = sorted(n1_names & n2_names)
    print(f"[trainer] {len(paired)} paired covers (n1+n2 features both present)")

    if not paired:
        return 1

    # Load all features
    X1 = np.stack([np.load(n1_dir / f"{name}.npy") for name in paired])
    X2 = np.stack([np.load(n2_dir / f"{name}.npy") for name in paired])
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(paired), dtype=int),
                        np.ones(len(paired), dtype=int)])
    cover_idx = np.concatenate([np.arange(len(paired)), np.arange(len(paired))])

    n_features = X.shape[1]
    print(f"[trainer] X shape: {X.shape}, y: {y.shape} ({y.sum()} positives), "
          f"n_features: {n_features}")
    print(f"[trainer] feature stats: any NaN={bool(np.isnan(X).any())}, "
          f"min={X.min():.4f}, max={X.max():.4f}")

    # Replace inf/NaN if present
    if np.isnan(X).any() or np.isinf(X).any():
        print("[trainer] WARNING: NaN/inf in features — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Paired-by-cover CV: fold assignment is per cover, both copies (n1 and n2)
    # of the same cover share the fold
    fold_per_cover = split_paired(paired, seed=args.seed)
    fold_idx = fold_per_cover[cover_idx]
    print(f"[trainer] fold sizes: " +
          ", ".join(f"f{f}={int((fold_idx == f).sum())}" for f in range(args.n_folds)))

    fold_aucs: list[float] = []
    fold_accs: list[float] = []
    per_image_scores: dict[str, dict] = {}
    importance_accum = np.zeros(n_features, dtype=np.float64)

    for fold in range(args.n_folds):
        test_mask = (fold_idx == fold)
        train_mask = ~test_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=8,
            random_state=args.seed + fold,
        )
        clf.fit(X_tr, y_tr)

        scores_te = clf.predict_proba(X_te)[:, 1]
        preds_te = (scores_te > 0.5).astype(int)
        a = auc_score(scores_te, y_te)
        acc = float((preds_te == y_te).mean())

        fold_aucs.append(a)
        fold_accs.append(acc)
        importance_accum += clf.feature_importances_

        print(f"[trainer] fold {fold}: train n={len(y_tr)} test n={len(y_te)} "
              f"AUC={a:.4f} acc={acc:.4f}")

        # Stash per-test-image scores
        test_indices = np.flatnonzero(test_mask)
        for i, idx in enumerate(test_indices):
            cidx = int(cover_idx[idx])
            cname = paired[cidx]
            cls = int(y[idx])
            per_image_scores.setdefault(cname, {})[f"n{cls + 1}"] = {
                "fold": fold,
                "score": float(scores_te[i]),
                "label": cls,
            }

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print(f"\n[trainer] === RESULTS ===")
    print(f"[trainer] AUC: {mean_auc:.4f} ± {std_auc:.4f} (5-fold CV)")
    print(f"[trainer] Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    # Top-K feature importances
    top_idx = np.argsort(-importance_accum)[: args.top_k]
    top_features = [(int(i), float(importance_accum[i])) for i in top_idx]
    print(f"\n[trainer] Top {args.top_k} feature importances (sum across folds):")
    for rank, (i, imp) in enumerate(top_features, 1):
        print(f"  #{rank:2d}  feat_{i:5d}  importance={imp:.4f}")

    # Persist
    results = {
        "phase": "4e_minimal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_paired_covers": len(paired),
        "n_features": int(n_features),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "xgb_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "auc_per_fold": fold_aucs,
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "acc_per_fold": fold_accs,
        "top_features": top_features,
    }
    out_json = args.out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[trainer] saved {out_json}")

    out_scores = args.out_dir / "per_image_scores.json"
    out_scores.write_text(json.dumps(per_image_scores, indent=2))
    print(f"[trainer] saved {out_scores}")

    # Save importance vector
    np.save(args.out_dir / "feature_importance.npy", importance_accum.astype(np.float32))
    print(f"[trainer] saved feature_importance.npy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
