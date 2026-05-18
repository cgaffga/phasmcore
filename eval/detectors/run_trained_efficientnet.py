"""Run a fine-tuned EfficientNet-B0 checkpoint on a directory of JPEGs.

Mirrors run_trained_srnet.py but uses the torchvision EfficientNet-B0
architecture with the binary-classifier head produced by
train_efficientnet.py. Grayscale inputs are duplicated across 3 channels
and ImageNet-normalised at load time (must match the trainer's preprocessing).

Usage:
    uv run python detectors/run_trained_efficientnet.py \\
        --image-dir <path-to-jpegs> \\
        --checkpoint runs/<date>-e15-effnet-juniward-qf75-pf020-s042/checkpoints/best.pt \\
        --output runs/<date>-<eval-slug>/effnet_<set>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def select_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_model() -> nn.Module:
    """Construct the same model architecture the trainer built (no weights yet)."""
    # weights=None — we'll load the fine-tuned state_dict over a fresh shell
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def preprocess(image_path: Path, target_size: int = 512) -> torch.Tensor:
    """Match train_efficientnet.py: grayscale -> 3-channel -> ImageNet normalize."""
    img = Image.open(image_path).convert("L")
    if img.size != (target_size, target_size):
        w, h = img.size
        if w >= target_size and h >= target_size:
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            img = img.crop((left, top, left + target_size, top + target_size))
        else:
            img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0)                      # (1, H, W)
    t = t.repeat(3, 1, 1)                                        # (3, H, W)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = select_device(args.device)
    print(f"[run_trained_effnet] device={device}")

    state = torch.load(str(args.checkpoint), map_location=device, weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[run_trained_effnet] checkpoint loaded: {args.checkpoint}")
    print(f"[run_trained_effnet] checkpoint val_pe at save time: {state.get('val_pe', 'unknown')}")

    images = (
        sorted(args.image_dir.glob("*.jpg"))
        + sorted(args.image_dir.glob("*.jpeg"))
        + sorted(args.image_dir.glob("*.pgm"))
        + sorted(args.image_dir.glob("*.png"))
    )
    if not images:
        print(f"[run_trained_effnet] no images in {args.image_dir}", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    print(f"[run_trained_effnet] {len(images)} images")

    results: dict[str, dict] = {}
    batch_x: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    @torch.no_grad()
    def flush() -> None:
        if not batch_x:
            return
        x = torch.stack(batch_x).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        for path, p_ in zip(batch_paths, probs):
            results[path.name] = {
                "cover_prob": float(p_[0]),
                "stego_prob": float(p_[1]),
                "predicted": "stego" if p_[1] > p_[0] else "cover",
            }
        batch_x.clear()
        batch_paths.clear()

    for path in tqdm(images, desc="scoring"):
        batch_x.append(preprocess(path, args.image_size))
        batch_paths.append(path)
        if len(batch_x) >= args.batch_size:
            flush()
    flush()

    h = hashlib.sha256()
    with args.checkpoint.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    summary = {
        "image_dir": str(args.image_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_first8": h.hexdigest()[:8],
        "arch": "efficientnet_b0_imagenet1k",
        "n_images": len(images),
        "mean_stego_prob": float(np.mean([r["stego_prob"] for r in results.values()])),
        "frac_predicted_stego": float(
            sum(1 for r in results.values() if r["predicted"] == "stego") / len(results)
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_image": results}, indent=2))
    print(f"[run_trained_effnet] wrote {args.output}")
    print(f"[run_trained_effnet] mean stego prob: {summary['mean_stego_prob']:.4f}")
    print(f"[run_trained_effnet] fraction predicted stego: {summary['frac_predicted_stego']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
