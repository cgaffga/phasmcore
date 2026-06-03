"""Train ImageNet-pretrained EfficientNet-B0 on JPEG cover/stego pairs (E15).

Yousfi et al. 2020 (WIFS) — ImageNet-pretrained CNNs surpass from-scratch SRNet
as the JPEG-steganalysis SOTA. This script gives us a comparable detector to
re-run the §6.3 shadow-N experiment against, validating that shadow security
holds vs the stronger post-SRNet detector lineage (review-finding F7 follow-up).

Mirrors the train_srnet.py structure so a/b comparison is clean:
  - Same dataset, same paired-cover convention, same hash-deterministic
    4000/1000/5000 train/val/test split, same loss + eval protocol.
  - Different: torchvision EfficientNet-B0 with ImageNet weights, 3-channel
    input (grayscale duplicated across channels for clean pretrained reuse),
    ImageNet normalisation, lower default LR (pretrained fine-tunes need less),
    fewer default epochs.

Usage:
    uv run python detectors/train_efficientnet.py \\
        --payload 0.2 --quality 75 \\
        --epochs 40 --batch-size 8 --seed 42 \\
        --run-name e15-effnet-juniward-qf75-pf020-s042
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# ImageNet RGB normalisation. We duplicate grayscale to 3 channels at load time
# (preserves pretrained feature distribution; the equivalent of applying the
# norm independently per channel collapses to a single channel-mean-shift since
# all three channels are identical).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
# Dataset — same paired structure as SRNet, but 3-channel + ImageNet norm
# ---------------------------------------------------------------------------
class PairedStegoDataset(Dataset):
    def __init__(self, cover_dir: Path, stego_dir: Path, names: list[str]):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.names = names
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        cover = self._load(self.cover_dir / name)
        stego = self._load(self.stego_dir / name)
        return {"cover": cover, "stego": stego}

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")  # grayscale, 0..255
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)            # (1, H, W)
        t = t.repeat(3, 1, 1)                              # (3, H, W) — RGB-duplicate
        t = (t - self._mean) / self._std                   # ImageNet normalize
        return t


def split_by_name_hash(names: list[str], n_train: int, n_val: int, n_test: int) -> tuple[list, list, list]:
    if n_train + n_val + n_test > len(names):
        raise ValueError(f"split sum {n_train + n_val + n_test} > {len(names)}")
    keyed = sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    return (
        keyed[:n_train],
        keyed[n_train:n_train + n_val],
        keyed[n_train + n_val:n_train + n_val + n_test],
    )


# ---------------------------------------------------------------------------
# Model — torchvision EfficientNet-B0 with 2-class head
# ---------------------------------------------------------------------------
def build_model(device: torch.device, init: str = "imagenet",
                init_ckpt: Path | None = None) -> nn.Module:
    # init="imagenet": torchvision IMAGENET1K_V1 classification weights — the
    #   default every E15/E17 run used. init="checkpoint": start from a prior
    #   steganalysis-trained backbone (e.g. the JIN-pretrained EffNet) — the
    #   stronger-pretraining detector for the "why not JIN?" review point.
    weights = EfficientNet_B0_Weights.DEFAULT if init == "imagenet" else None
    model = efficientnet_b0(weights=weights)
    # Replace 1000-class head with binary classifier.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    if init == "checkpoint":
        if init_ckpt is None:
            raise ValueError("--init checkpoint requires --init-ckpt")
        state = torch.load(init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
    return model.to(device)


# ---------------------------------------------------------------------------
# Training / eval loops (identical structure to train_srnet.py)
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
        n_total += labels.numel()
        n_correct += int(preds.eq(labels).sum())
        n_fp += int(((preds == 1) & (labels == 0)).sum())
        n_fn += int(((preds == 0) & (labels == 1)).sum())
    n_per_class = n_total // 2
    return {
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "pe": (n_fp + n_fn) / (2 * n_per_class),
        "fp": n_fp,
        "fn": n_fn,
    }


def train_one_seed(args, device: torch.device, run_dir: Path) -> dict:
    if args.cover_dir and args.stego_dir:
        # Explicit dirs (e.g. the JIN pretraining corpus data/jin/{covers,stego}/qf75).
        cover_dir = Path(args.cover_dir)
        stego_dir = Path(args.stego_dir)
    else:
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
    print(f"[train_effnet] {len(paired)} paired cover/stego available")

    train_names, val_names, test_names = split_by_name_hash(
        paired, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test
    )
    print(f"[train_effnet] split: train={len(train_names)} val={len(val_names)} test={len(test_names)}")

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

    model = build_model(device, args.init, args.init_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_effnet] EfficientNet-B0 (ImageNet pretrained) params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LR schedule: half every 15 epochs (matches the SRNet 30-epoch schedule
    # ratio'd down for shorter pretrained fine-tune runs).
    scheduler = MultiStepLR(optimizer, milestones=list(range(15, args.epochs, 15)), gamma=0.5)

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
        print(f"  ep {epoch:02d}: loss={history[-1]['train_loss']:.4f} val_pe={val['pe']:.4f} ({epoch_time:.1f}s)", flush=True)

        if val["pe"] < best_val_pe:
            best_val_pe = val["pe"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_pe": val["pe"], "arch": "efficientnet_b0_imagenet1k"},
                best_path,
            )

    # Final test on best checkpoint
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[train_effnet] FINAL test PE = {test_metrics['pe']:.4f} (acc={test_metrics['accuracy']:.4f})")
    return {
        "history": history,
        "test": test_metrics,
        "best_epoch": state["epoch"],
        "best_val_pe": best_val_pe,
        "arch": "efficientnet_b0_imagenet1k",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--payload", type=float, default=None,
                   help="bpnzAC payload rate (BOSSbase path; omit when using --cover-dir/--stego-dir)")
    p.add_argument("--cover-dir", type=str, default=None,
                   help="override cover dir, e.g. JIN: data/jin/covers/qf75")
    p.add_argument("--stego-dir", type=str, default=None,
                   help="override stego dir, e.g. JIN: data/jin/stego/qf75")
    p.add_argument("--init", choices=["imagenet", "checkpoint"], default="imagenet",
                   help="backbone init: imagenet (default, all E15/E17 runs) or checkpoint (JIN backbone via --init-ckpt)")
    p.add_argument("--init-ckpt", type=Path, default=None, help="checkpoint .pt for --init checkpoint")
    p.add_argument("--quality", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam LR. Pretrained fine-tuning typically uses 5e-5 to 1e-4; default 1e-4 sits in the middle.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device()
    print(f"[train_effnet] device = {device}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = args.run_name or (
        f"e15-effnet-juniward-pf{int(round(args.payload * 100)):03d}-s{args.seed:03d}"
    )
    run_dir = ROOT / "runs" / f"{today}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_effnet] run dir = {run_dir}")

    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "arch": "efficientnet_b0_imagenet1k",
        "args": vars(args),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    result = train_one_seed(args, device, run_dir)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
