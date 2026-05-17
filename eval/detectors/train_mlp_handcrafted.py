"""E4: MLP on hand-crafted features — confirm the CNN-vs-XGBoost gap.

Trains a small feed-forward network on Phase 4e features (3,694-dim
hand-crafted) with the same 5-fold paired-by-cover split as
`train_handcrafted_classifier.py`. Reports per-fold AUC.

Outcome distinguishes two hypotheses for the Phase 4e-gray AUC 0.803 vs
Phase 4c+4d SRNet AUC 0.556 gap on identical data:
  - "features matter" → MLP gets close to XGBoost (~0.8)
  - "trees > NNs here" → MLP underperforms XGBoost (~0.6 or lower)

Usage:
  python detectors/train_mlp_handcrafted.py \\
    --features-dir data/path4e_features_gray \\
    --out-dir runs/2026-05-14-path4e-mlp-gray \\
    --seed 2026
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.5
    ranks = rankdata(scores)
    rank_sum_pos = ranks[pos].sum()
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def split_paired(paired: list[str], seed: int, n_folds: int) -> np.ndarray:
    return np.array([
        int(hashlib.sha256(f"{n}|{seed}".encode()).hexdigest()[:8], 16) % n_folds
        for n in paired
    ])


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(512, 128, 64)):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(X_tr, y_tr, X_te, y_te, device, seed: int,
               epochs: int = 80, batch_size: int = 32, lr: float = 1e-3):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True) + 1e-7
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    model = MLP(n_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    X_te_t = torch.from_numpy(X_te).float().to(device)

    n = X_tr_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t).cpu().numpy()
    return logits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[mlp] device={device}")

    n1_dir = args.features_dir / "n1"
    n2_dir = args.features_dir / "n2"
    n1_names = {p.stem for p in n1_dir.glob("*.npy")}
    n2_names = {p.stem for p in n2_dir.glob("*.npy")}
    paired = sorted(n1_names & n2_names)
    print(f"[mlp] {len(paired)} paired covers")

    feats_n1 = np.stack([np.load(n1_dir / f"{p}.npy") for p in paired])
    feats_n2 = np.stack([np.load(n2_dir / f"{p}.npy") for p in paired])
    X = np.concatenate([feats_n1, feats_n2], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(paired)), np.ones(len(paired))]).astype(np.int64)
    cover_idx = np.concatenate([np.arange(len(paired)), np.arange(len(paired))])
    print(f"[mlp] X={X.shape} y={y.shape} ({y.sum()} positives)")

    if not np.all(np.isfinite(X)):
        n_bad = (~np.isfinite(X)).sum()
        print(f"[mlp] WARNING: {n_bad} non-finite entries → 0")
        X[~np.isfinite(X)] = 0.0

    fold_per_cover = split_paired(paired, args.seed, args.n_folds)
    fold_idx = fold_per_cover[cover_idx]
    print(f"[mlp] fold sizes: " +
          ", ".join(f"f{f}={int((fold_idx == f).sum())}" for f in range(args.n_folds)))

    fold_aucs = []
    fold_accs = []
    for fold in range(args.n_folds):
        test_mask = (fold_idx == fold)
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        logits = train_fold(X_tr, y_tr, X_te, y_te, device,
                            seed=args.seed + fold,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr)
        a = auc_score(logits, y_te)
        preds = (logits > 0.0).astype(int)
        acc = float((preds == y_te).mean())
        fold_aucs.append(a)
        fold_accs.append(acc)
        print(f"[mlp] fold {fold}: train n={len(y_tr)} test n={len(y_te)} "
              f"AUC={a:.4f} acc={acc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"\n[mlp] === RESULTS ===")
    print(f"[mlp] AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"[mlp] Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    results = {
        "phase": "4e_mlp_confirmation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_paired_covers": len(paired),
        "n_features": int(X.shape[1]),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": device,
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "auc_per_fold": fold_aucs,
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "acc_per_fold": fold_accs,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[mlp] saved {args.out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
