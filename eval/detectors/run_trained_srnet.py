"""Run a from-scratch-trained SRNet checkpoint on a directory of JPEGs.

Loads the `best.pt` checkpoint produced by `train_srnet.py` and scores
each JPEG (or PGM/PNG) with per-image cover/stego logits → softmax →
stego_prob. Output mirrors `run_pretrained_effnet.py` and
`run_pretrained_srnet_aletheia.py` — JSON with summary + per-image
scores.

Use this once Path 1c training completes to score Phasm Ghost,
J-UNIWARD reference, and cover with our own trained detector.

Usage:
    uv run python detectors/run_trained_srnet.py \\
        --image-dir <path-to-jpegs> \\
        --checkpoint runs/<datestamp>-srnet-juniward-pf040-s042/checkpoints/best.pt \\
        --output runs/<datestamp>-<eval-slug>/srnet_trained_<set>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEEP_STEG = ROOT / "third_party" / "Deep-Steganalysis"


def install_deep_steg_path(channels: int, height: int) -> None:
    if not DEEP_STEG.exists():
        raise FileNotFoundError(f"Missing {DEEP_STEG}")
    config_stub = types.ModuleType("config")
    config_stub.mode = "test"
    config_stub.stego_img_channel = channels
    config_stub.stego_img_height = height
    sys.modules["config"] = config_stub
    sys.path.insert(0, str(DEEP_STEG))


def select_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def preprocess(image_path: Path, target_size: int = 512) -> torch.Tensor:
    """Match train_srnet.py's PairedStegoDataset: grayscale, [0,1], shape (1,H,W)."""
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
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all images; useful for cheap eval on a fixed subset")
    p.add_argument("--device", type=str, default=None,
                   help="Override auto-selected device (cpu/mps/cuda)")
    args = p.parse_args()

    device = select_device(args.device)
    print(f"[run_trained_srnet] device={device}")

    install_deep_steg_path(channels=1, height=args.image_size)
    from models.SRNet import Model as SRNetModel  # noqa: E402

    state = torch.load(str(args.checkpoint), map_location=device, weights_only=True)
    model = SRNetModel().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[run_trained_srnet] checkpoint loaded: {args.checkpoint}")
    print(f"[run_trained_srnet] checkpoint val_pe at save time: {state.get('val_pe', 'unknown')}")

    images = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.jpeg")) + sorted(args.image_dir.glob("*.pgm")) + sorted(args.image_dir.glob("*.png"))
    if not images:
        print(f"[run_trained_srnet] no images in {args.image_dir}", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    print(f"[run_trained_srnet] {len(images)} images")

    results = {}
    batch_x: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    @torch.no_grad()
    def flush():
        if not batch_x:
            return
        x = torch.stack(batch_x).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        for path, p in zip(batch_paths, probs):
            results[path.name] = {
                "cover_prob": float(p[0]),
                "stego_prob": float(p[1]),
                "predicted": "stego" if p[1] > p[0] else "cover",
            }
        batch_x.clear()
        batch_paths.clear()

    for path in tqdm(images, desc="scoring"):
        batch_x.append(preprocess(path, args.image_size))
        batch_paths.append(path)
        if len(batch_x) >= args.batch_size:
            flush()
    flush()

    # Hash checkpoint for reproducibility
    h = hashlib.sha256()
    with args.checkpoint.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    summary = {
        "image_dir": str(args.image_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_first8": h.hexdigest()[:8],
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
    print(f"[run_trained_srnet] wrote {args.output}")
    print(f"[run_trained_srnet] mean stego prob: {summary['mean_stego_prob']:.4f}")
    print(f"[run_trained_srnet] fraction predicted stego: {summary['frac_predicted_stego']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
