"""Train a binary classifier that discriminates Phasm-with-1-shadow from Phasm-with-2-shadows.

Phase 4c: the dedicated deep adversary for the deniability claim.

Adversary model: has stego JPEG + one passphrase + extracted decoy message.
Question: is there an additional shadow beyond the one I extracted?
Binary task: class 0 = exactly 1 shadow; class 1 = exactly 2 shadows.

Setup:
- Reuses the Deep-Steganalysis SRNet architecture (4.78M params).
- Initializes from our 0.4 J-UNIWARD SRNet checkpoint (curriculum learning —
  the J-UNIWARD features are a reasonable starting point for a related-task
  fine-tune).
- Trains on paired (n1, n2) cover IDs to ensure no cover leaks between train
  and test splits.

Expected output: test AUC. If AUC ≈ 0.5, deniability holds against the
deep adversary. If AUC ≫ 0.6, deniability is leaking shadow-count info.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEEP_STEG = ROOT / "third_party" / "Deep-Steganalysis"


def install_deep_steg(channels: int, height: int) -> None:
    if not DEEP_STEG.exists():
        raise FileNotFoundError(f"Missing {DEEP_STEG}")
    stub = types.ModuleType("config")
    stub.mode = "train"
    stub.stego_img_channel = channels
    stub.stego_img_height = height
    sys.modules["config"] = stub
    sys.path.insert(0, str(DEEP_STEG))


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class ShadowCountDataset(Dataset):
    """Returns (image, label) where label=0 for n1 stego, label=1 for n2 stego."""

    def __init__(self, items: list[tuple[Path, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("L")
        if img.size != (512, 512):
            w, h = img.size
            left = (w - 512) // 2
            top = (h - 512) // 2
            img = img.crop((left, top, left + 512, top + 512))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def split_by_name_hash(paired_names: list[str], n_train: int, n_val: int, n_test: int) -> tuple[list, list, list]:
    """Deterministic by-cover-name split. Same name -> same fold forever."""
    keyed = sorted(paired_names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    return keyed[:n_train], keyed[n_train:n_train + n_val], keyed[n_train + n_val:n_train + n_val + n_test]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Vectorized rank-based ROC AUC (Mann-Whitney U statistic).

    No sklearn dependency. Handles ties (uses average rank). Robust to NaN
    by dropping them before computation.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    mask = ~np.isnan(scores)
    if mask.sum() < len(scores):
        scores = scores[mask]
        labels = labels[mask]
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U via averaged ranks (handles ties correctly)
    from scipy.stats import rankdata
    all_scores = np.concatenate([neg_scores, pos_scores])
    ranks = rankdata(all_scores)
    rank_sum_pos = ranks[n_neg:].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    scores = []
    labels = []
    correct = 0
    total = 0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        logits = model(imgs)
        # Use raw logit difference for AUC scoring, NOT softmax. On MPS,
        # softmax during eval-mode batched forward passes occasionally
        # produces NaN even when underlying logits are finite (BN
        # running-stat edge case). Logit diff is monotonic w/ pos-class
        # probability — AUC ranking preserved.
        score_pos = (logits[:, 1] - logits[:, 0]).cpu().numpy().tolist()
        scores.extend(score_pos)
        labels.extend(lbls.cpu().numpy().tolist())
        preds = logits.argmax(dim=1)
        correct += int(preds.eq(lbls).sum())
        total += int(lbls.numel())
    s = np.array(scores)
    l = np.array(labels)
    return {
        "n": total,
        "accuracy": correct / max(total, 1),
        "auc": auc(s, l),
        "pe": 1 - correct / max(total, 1),  # binary error rate
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n1-dir", type=Path, default=Path("data/path4c_classifier/n1"))
    p.add_argument("--n2-dir", type=Path, default=Path("data/path4c_classifier/n2"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Clip gradient norm to this value before each optimizer step (0 to disable)")
    p.add_argument("--lr-warmup-steps", type=int, default=200,
                   help="Linear LR warmup over the first N batches (0 to disable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--init-from", type=Path,
                   default=Path("runs/2026-05-10-srnet-juniward-pf040-s042/checkpoints/best.pt"),
                   help="Init weights (curriculum from 0.4 J-UNIWARD SRNet)")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device()
    print(f"[shadow_classifier] device={device}")

    # Paired covers — must succeed for both n1 and n2
    n1_names = {p.name for p in args.n1_dir.glob("*.jpg")}
    n2_names = {p.name for p in args.n2_dir.glob("*.jpg")}
    paired = sorted(n1_names & n2_names)
    print(f"[shadow_classifier] {len(paired)} paired covers (both n1 and n2 succeeded)")

    if len(paired) < args.n_train + args.n_val + args.n_test:
        print(f"WARN: paired={len(paired)} < requested split sum {args.n_train + args.n_val + args.n_test}")
        # Auto-scale split
        total = len(paired)
        args.n_train = int(total * 0.6)
        args.n_val = int(total * 0.1)
        args.n_test = total - args.n_train - args.n_val
        print(f"  scaled to: train={args.n_train} val={args.n_val} test={args.n_test}")

    train_names, val_names, test_names = split_by_name_hash(
        paired, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test
    )

    def build_items(names: list[str]) -> list[tuple[Path, int]]:
        items = []
        for name in names:
            items.append((args.n1_dir / name, 0))
            items.append((args.n2_dir / name, 1))
        return items

    train_items = build_items(train_names)
    val_items = build_items(val_names)
    test_items = build_items(test_names)
    print(f"[shadow_classifier] examples: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    train_loader = DataLoader(ShadowCountDataset(train_items), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(ShadowCountDataset(val_items), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(ShadowCountDataset(test_items), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    install_deep_steg(channels=1, height=512)
    from models.SRNet import Model as SRNetModel  # noqa
    model = SRNetModel().to(device)

    if args.init_from.exists():
        state = torch.load(str(args.init_from), map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        prev = state.get("val_pe", "?")
        print(f"[shadow_classifier] curriculum init from {args.init_from} (prev val_pe={prev})")
    else:
        print(f"[shadow_classifier] init_from {args.init_from} missing — training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=list(range(30, args.epochs, 30)), gamma=0.5)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = args.run_name or f"shadow-classifier-s{args.seed:03d}"
    run_dir = ROOT / "runs" / f"{today}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    # Save initial-state checkpoint as a fallback — guarantees something exists
    # to load at the end of training even if no epoch beats best_val_auc.
    fallback_path = run_dir / "checkpoints" / "last.pt"
    torch.save({"model": model.state_dict(), "epoch": -1, "val_auc": 0.5}, fallback_path)
    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "args": vars(args),
        "n_paired_covers": len(paired),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"[shadow_classifier] run dir = {run_dir}")

    history = []
    best_val_auc = 0.5
    best_path = run_dir / "checkpoints" / "best.pt"
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_loss = 0.0
        ep_n = 0
        nan_steps = 0
        for imgs, lbls in tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            # LR warmup over first N steps — prevents BN running-stats blow-up
            if args.lr_warmup_steps > 0 and global_step < args.lr_warmup_steps:
                warm_lr = args.lr * (global_step + 1) / args.lr_warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            if torch.isnan(loss):
                nan_steps += 1
                global_step += 1
                continue  # skip this step rather than poisoning weights with NaN gradient
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            ep_loss += loss.item() * lbls.numel()
            ep_n += int(lbls.numel())
            global_step += 1
        scheduler.step()

        val = evaluate(model, val_loader, device)
        ep_time = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": ep_loss / max(ep_n, 1),
            "val_auc": val["auc"],
            "val_accuracy": val["accuracy"],
            "epoch_seconds": ep_time,
            "nan_steps": nan_steps,
        })
        msg = (f"  ep {epoch:02d}: loss={history[-1]['train_loss']:.4f} "
               f"val_auc={val['auc']:.4f} val_acc={val['accuracy']:.4f} "
               f"nan_steps={nan_steps}/{len(train_loader)} ({ep_time:.0f}s)")
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")  # also to stderr (unbuffered)
        sys.stderr.flush()

        # Always save last-epoch checkpoint (fallback if no val_auc beats baseline)
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val["auc"]}, fallback_path)
        if not np.isnan(val["auc"]) and val["auc"] > best_val_auc:
            best_val_auc = val["auc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val["auc"]}, best_path)

    # Load best if exists, else fall back to last-epoch
    load_path = best_path if best_path.exists() else fallback_path
    state = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[shadow_classifier] FINAL test AUC={test_metrics['auc']:.4f} acc={test_metrics['accuracy']:.4f} "
          f"(loaded {'best' if load_path == best_path else 'last-epoch fallback'})", flush=True)
    (run_dir / "results.json").write_text(json.dumps({
        "history": history,
        "best_val_auc": best_val_auc,
        "best_epoch": state["epoch"],
        "test": test_metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
