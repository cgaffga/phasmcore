"""Train SRNet (Boroumand-Chen-Fridrich 2019) on cover/stego pairs.

Wraps the SRNet model from `third_party/Deep-Steganalysis/models/SRNet.py`
with our own training loop:

  - Device: prefers MPS on Apple Silicon, falls back to CUDA, then CPU
  - Dataset: grayscale (1-channel) — matches Boroumand 2019 protocol
  - Pairing: (cover[i], stego[i]) batches keep the model exposed to both
    classes evenly per step (standard SRNet training convention)
  - Split: 4000 train / 1000 val / 5000 test by sorted filename hash
  - Optimizer: Adam, lr=2e-4, weight_decay=1e-5 (matches Deep-Steganalysis)
  - LR sched: MultiStepLR every 30 epochs, gamma=0.5
  - Saves checkpoint, predictions, and a `config.json` for reproducibility

Run one seed at a time (script intentionally single-seed for clean log
separation); call from a bash loop or `--seed` cycle for 3+ seeds.

Usage:
    uv run python detectors/train_srnet.py \\
        --payload 0.4 --quality 75 \\
        --epochs 80 --batch-size 8 --seed 42 \\
        --run-name juniward-baseline-pf040-s042
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


# ---------------------------------------------------------------------------
# Deep-Steganalysis interop: it imports `config` for stego_img_channel, etc.
# Provide a minimal stub before importing their model module.
# ---------------------------------------------------------------------------
def install_deep_steg_path(channels: int, height: int) -> None:
    if not DEEP_STEG.exists():
        raise FileNotFoundError(
            f"{DEEP_STEG} not found — clone it first:\n"
            f"  cd third_party && git clone --depth 1 "
            f"https://github.com/albblgb/Deep-Steganalysis.git"
        )
    config_stub = types.ModuleType("config")
    config_stub.mode = "train"
    config_stub.stego_img_channel = channels
    config_stub.stego_img_height = height
    sys.modules["config"] = config_stub
    sys.path.insert(0, str(DEEP_STEG))


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset: paired cover/stego, grayscale
# ---------------------------------------------------------------------------
class PairedStegoDataset(Dataset):
    """Returns one (cover, stego) pair per index. The training loop unpacks
    the pair into two batch elements with labels (0=cover, 1=stego)."""

    def __init__(self, cover_dir: Path, stego_dir: Path, names: list[str]):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        cover = self._load(self.cover_dir / name)
        stego = self._load(self.stego_dir / name)
        return {"cover": cover, "stego": stego}

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # shape (1, H, W)


def split_by_name_hash(names: list[str], n_train: int, n_val: int, n_test: int) -> tuple[list, list, list]:
    """Deterministic split by sha256(name). Same names -> same split forever."""
    if n_train + n_val + n_test > len(names):
        raise ValueError(f"split sum {n_train + n_val + n_test} > {len(names)}")
    keyed = sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    return keyed[:n_train], keyed[n_train:n_train + n_val], keyed[n_train + n_val:n_train + n_val + n_test]


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------
def unpack_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cover = batch["cover"].to(device, non_blocking=True)
    stego = batch["stego"].to(device, non_blocking=True)
    inputs = torch.cat([cover, stego], dim=0)
    labels = torch.cat([
        torch.zeros(cover.size(0), dtype=torch.long, device=device),
        torch.ones(stego.size(0), dtype=torch.long, device=device),
    ])
    return inputs, labels


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n_total = 0
    n_correct = 0
    n_fp = 0
    n_fn = 0
    for batch in loader:
        inputs, labels = unpack_batch(batch, device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels)
        n_total += labels.numel()
        n_correct += int(correct.sum())
        n_fp += int(((preds == 1) & (labels == 0)).sum())
        n_fn += int(((preds == 0) & (labels == 1)).sum())
    n_per_class = n_total // 2
    return {
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "pe": (n_fp + n_fn) / (2 * n_per_class),  # = (FPR + FNR) / 2
        "fp": n_fp,
        "fn": n_fn,
    }


def train_one_seed(args, device: torch.device, run_dir: Path) -> dict:
    cover_dir = ROOT / "data" / "bossbase" / f"jpeg_qf{args.quality}"
    stego_dir = (
        ROOT / "data" / "stego" / "juniward"
        / f"qf{args.quality}" / f"payload_{int(round(args.payload * 100)):03d}"
    )

    if not cover_dir.exists():
        raise FileNotFoundError(f"covers not found: {cover_dir}")
    if not stego_dir.exists():
        raise FileNotFoundError(f"stegos not found: {stego_dir}")

    cover_names = {p.name for p in cover_dir.glob("*.jpg")}
    stego_names = {p.name for p in stego_dir.glob("*.jpg")}
    paired = sorted(cover_names & stego_names)
    print(f"[train_srnet] {len(paired)} paired cover/stego available")

    train_names, val_names, test_names = split_by_name_hash(
        paired, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test
    )
    print(f"[train_srnet] split: train={len(train_names)} val={len(val_names)} test={len(test_names)}")

    train_loader = DataLoader(
        PairedStegoDataset(cover_dir, stego_dir, train_names),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        PairedStegoDataset(cover_dir, stego_dir, val_names),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        PairedStegoDataset(cover_dir, stego_dir, test_names),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    install_deep_steg_path(channels=1, height=args.image_size)
    from models.SRNet import Model as SRNetModel  # noqa: E402
    model = SRNetModel().to(device)
    print(f"[train_srnet] SRNet params: {sum(p.numel() for p in model.parameters()):,}")

    if args.init_from is not None:
        init_state = torch.load(str(args.init_from), map_location=device, weights_only=True)
        model.load_state_dict(init_state["model"])
        prev_pe = init_state.get("val_pe", "?")
        print(f"[train_srnet] curriculum init from {args.init_from} (prev val_pe={prev_pe})")

    criterion = nn.CrossEntropyLoss()
    # Lower LR when curriculum-fine-tuning — fine adjustments not aggressive overrides
    effective_lr = args.lr * 0.5 if args.init_from else args.lr
    optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=list(range(30, args.epochs, 30)), gamma=0.5)

    history = []
    best_val_pe = 1.0
    best_path = run_dir / "checkpoints" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_loss = 0.0
        ep_n = 0
        for batch in tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False):
            inputs, labels = unpack_batch(batch, device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * labels.numel()
            ep_n += labels.numel()
        scheduler.step()

        val = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": ep_loss / max(ep_n, 1),
            "val_pe": val["pe"],
            "val_accuracy": val["accuracy"],
            "epoch_seconds": epoch_time,
        })
        print(f"  ep {epoch:02d}: loss={history[-1]['train_loss']:.4f} val_pe={val['pe']:.4f} ({epoch_time:.1f}s)")

        if val["pe"] < best_val_pe:
            best_val_pe = val["pe"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_pe": val["pe"]}, best_path)

    # Final test on best checkpoint
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[train_srnet] FINAL test PE = {test_metrics['pe']:.4f} (acc={test_metrics['accuracy']:.4f})")
    return {"history": history, "test": test_metrics, "best_epoch": state["epoch"], "best_val_pe": best_val_pe}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--payload", type=float, required=True, help="bpnzAC payload rate (0.1, 0.4, ...)")
    p.add_argument("--quality", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--run-name", type=str, default=None,
                   help="run subdirectory name; auto-generated if omitted")
    p.add_argument("--init-from", type=Path, default=None,
                   help="path to a previous checkpoint (best.pt) to initialize from. "
                        "Enables curriculum learning: train at high payload first, then "
                        "fine-tune at lower payload from the high-payload checkpoint. "
                        "When set, learning rate is halved for fine adjustments.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device()
    print(f"[train_srnet] device = {device}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = args.run_name or (
        f"juniward-baseline-pf{int(round(args.payload * 100)):03d}-s{args.seed:03d}"
    )
    run_dir = ROOT / "runs" / f"{today}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_srnet] run dir = {run_dir}")

    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "args": vars(args),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    result = train_one_seed(args, device, run_dir)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
