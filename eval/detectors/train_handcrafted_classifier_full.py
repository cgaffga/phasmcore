"""Phase 4e FULL — XGBoost training on combined (minimal + rich-model) features.

Loads features from BOTH `data/path4e_features_color/` (minimal: SPAM + DCT-hist
+ LSB) AND `data/path4e_features_rich_color/` (rich: SRMQ1-lite + DCTR-lite +
GFR-lite), concatenates per image, and runs 5-fold paired-by-cover CV with the
BEST hyperparameters found in the hyperparameter_search.json output.

If hyperparameter_search.json is absent, falls back to a sensible deeper-tree
default (max_depth=8, lr=0.1, n_estimators=500) — better-than-default given
the result of the search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def load_features(features_dir: Path, names: list[str], subset: str) -> np.ndarray:
    sub_dir = features_dir / subset
    return np.stack([np.load(sub_dir / f"{n}.npy") for n in names])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-min", type=Path,
                   default=ROOT / "data" / "path4e_features_color")
    p.add_argument("--features-rich", type=Path,
                   default=ROOT / "data" / "path4e_features_rich_color")
    p.add_argument("--search-json", type=Path,
                   default=ROOT / "runs" / "2026-05-11-path4e-handcrafted-full" /
                   "hyperparam_search.json")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "runs" / "2026-05-11-path4e-handcrafted-full")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Locate paired covers present in BOTH feature sets
    min_n1 = {p.stem for p in (args.features_min / "n1").glob("*.npy")}
    min_n2 = {p.stem for p in (args.features_min / "n2").glob("*.npy")}
    rich_n1 = {p.stem for p in (args.features_rich / "n1").glob("*.npy")}
    rich_n2 = {p.stem for p in (args.features_rich / "n2").glob("*.npy")}
    paired = sorted(min_n1 & min_n2 & rich_n1 & rich_n2)
    print(f"[trainer] {len(paired)} covers paired in BOTH minimal + rich sets")
    if not paired:
        return 1

    X_min_1 = load_features(args.features_min, paired, "n1")
    X_min_2 = load_features(args.features_min, paired, "n2")
    X_rich_1 = load_features(args.features_rich, paired, "n1")
    X_rich_2 = load_features(args.features_rich, paired, "n2")

    X1 = np.hstack([X_min_1, X_rich_1])
    X2 = np.hstack([X_min_2, X_rich_2])
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(paired), int), np.ones(len(paired), int)])
    cover_idx = np.concatenate([np.arange(len(paired)), np.arange(len(paired))])

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    n_features = X.shape[1]
    print(f"[trainer] X shape: {X.shape}  ({X_min_1.shape[1]} minimal + "
          f"{X_rich_1.shape[1]} rich) total {n_features}")

    fold_per_cover = split_paired(paired, seed=args.seed, n_folds=args.n_folds)
    fold_idx = fold_per_cover[cover_idx]

    # Pull best params from hyperparam search
    if args.search_json.exists():
        search = json.loads(args.search_json.read_text())
        best = search["best"]
        params = {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
        }
        print(f"[trainer] using BEST params from {args.search_json.name}:")
    else:
        params = {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.1, "subsample": 1.0}
        print(f"[trainer] no search JSON; falling back to deeper default:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    fold_aucs: list[float] = []
    fold_accs: list[float] = []
    per_image_scores: dict = {}
    importance_accum = np.zeros(n_features, dtype=np.float64)

    for fold in range(args.n_folds):
        test_mask = (fold_idx == fold)
        X_tr, y_tr = X[~test_mask], y[~test_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        clf = XGBClassifier(
            **params,
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
        test_indices = np.flatnonzero(test_mask)
        for i, idx in enumerate(test_indices):
            cidx = int(cover_idx[idx])
            cname = paired[cidx]
            cls = int(y[idx])
            per_image_scores.setdefault(cname, {})[f"n{cls + 1}"] = {
                "fold": fold, "score": float(scores_te[i]), "label": cls,
            }

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print(f"\n[trainer] === RESULTS ===")
    print(f"[trainer] AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"[trainer] Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    top_idx = np.argsort(-importance_accum)[: args.top_k]
    top_features = [(int(i), float(importance_accum[i])) for i in top_idx]
    minimal_size = X_min_1.shape[1]
    print(f"\n[trainer] Top {args.top_k} feature importances "
          f"(minimal: idx < {minimal_size}; rich: idx >= {minimal_size}):")
    for rank, (i, imp) in enumerate(top_features, 1):
        which = "MIN" if i < minimal_size else "RICH"
        print(f"  #{rank:2d}  feat_{i:5d} [{which}]  importance={imp:.4f}")

    results = {
        "phase": "4e_full_minimal+rich",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_paired_covers": len(paired),
        "n_features_minimal": int(X_min_1.shape[1]),
        "n_features_rich": int(X_rich_1.shape[1]),
        "n_features_total": int(n_features),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "xgb_params": params,
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "auc_per_fold": fold_aucs,
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "acc_per_fold": fold_accs,
        "top_features": top_features,
    }
    (args.out_dir / "results_minimal_plus_rich.json").write_text(json.dumps(results, indent=2))
    np.save(args.out_dir / "feature_importance_full.npy",
            importance_accum.astype(np.float32))
    (args.out_dir / "per_image_scores_full.json").write_text(
        json.dumps(per_image_scores, indent=2))
    print(f"\n[trainer] saved results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
