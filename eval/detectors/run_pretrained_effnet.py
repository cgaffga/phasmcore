"""Run aletheia's pretrained EfficientNet-B0 J-UNIWARD/ALASKA-2 detector on JPEGs.

Loads the .h5 weights from data/pretrained_models/aletheia/ (downloaded from
github.com/daniellerch/aletheia/aletheia-models, MIT-licensed) into a Keras
EfficientNet-B0 → GlobalAveragePooling2D → Dense(2, softmax) head, then runs
inference on a directory of JPEG images.

The model was trained on **color JPEG, 512x512, mixed QF (75/90/95)** with
J-UNIWARD embedding at varying payloads. We use it as a Path 1a smoke
test for Phasm Ghost detection — the cover-domain mismatch (color vs Phasm's
grayscale Y-channel target) means this is a directional signal, not a
calibrated benchmark.

Output: per-image stego-probability written to a results JSON + a
summary printed to stdout (mean stego prob, fraction predicted stego).

Usage:
    uv run python detectors/run_pretrained_effnet.py \\
        --image-dir <path-to-jpegs> \\
        --variant A           # or B; aletheia ships both as a 2-model ensemble
        --output runs/<datestamp>-<slug>/effnet_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "data" / "pretrained_models" / "aletheia"

# Suppress TF noise BEFORE importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_model(variant: str = "A"):
    """Load aletheia's effnetb0-X-alaska2-juniw.h5 into a Keras Sequential."""
    import efficientnet.tfkeras as efn
    import tensorflow as tf
    from tensorflow.keras import layers as L

    weights = WEIGHTS_DIR / f"effnetb0-{variant}-alaska2-juniw.h5"
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights missing: {weights}\n"
            f"Download via: curl -sSL https://github.com/daniellerch/aletheia/raw/master/aletheia-models/effnetb0-{variant}-alaska2-juniw.h5 -o {weights}"
        )

    base = efn.EfficientNetB0(
        input_shape=(512, 512, 3), weights=None, include_top=False
    )
    model = tf.keras.Sequential([
        base,
        L.GlobalAveragePooling2D(),
        L.Dense(2, activation="softmax", dtype="float32"),
    ])
    model.load_weights(str(weights))
    return model


def preprocess(image_path: Path, target_size: int = 512):
    """Load JPEG -> (1, H, W, 3) float32 in [0, 1]. Replicates Y to RGB if grayscale."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        if im.mode == "L":
            im = im.convert("RGB")  # replicate Y to 3 channels
        elif im.mode != "RGB":
            im = im.convert("RGB")
        if im.size != (target_size, target_size):
            # Center-crop to target size if larger; pad with edge if smaller
            w, h = im.size
            if w >= target_size and h >= target_size:
                left = (w - target_size) // 2
                top = (h - target_size) // 2
                im = im.crop((left, top, left + target_size, top + target_size))
            else:
                # Resize keeping aspect ratio then pad
                im = im.resize((target_size, target_size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # (1, 512, 512, 3)


def run(image_dir: Path, variant: str, output: Path, batch_size: int = 8) -> dict:
    import numpy as np
    from tqdm import tqdm

    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(f"no JPEGs in {image_dir}")

    print(f"[effnet] loading variant {variant}...")
    model = load_model(variant)
    print(f"[effnet] {len(images)} images in {image_dir}")

    results = {}
    batch: list = []
    batch_paths: list = []

    def flush():
        if not batch:
            return
        x = np.concatenate(batch, axis=0)
        probs = model.predict(x, verbose=0)
        for path, p in zip(batch_paths, probs):
            results[path.name] = {
                "cover_prob": float(p[0]),
                "stego_prob": float(p[1]),
                "predicted": "stego" if p[1] > p[0] else "cover",
            }
        batch.clear()
        batch_paths.clear()

    for path in tqdm(images, desc=f"effnet-{variant}"):
        batch.append(preprocess(path))
        batch_paths.append(path)
        if len(batch) >= batch_size:
            flush()
    flush()

    summary = {
        "image_dir": str(image_dir),
        "variant": variant,
        "weights_sha256_first8": None,  # filled in below
        "n_images": len(images),
        "mean_stego_prob": float(np.mean([r["stego_prob"] for r in results.values()])),
        "frac_predicted_stego": float(
            sum(1 for r in results.values() if r["predicted"] == "stego") / len(results)
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Hash the weights for reproducibility
    import hashlib
    h = hashlib.sha256()
    weights_path = WEIGHTS_DIR / f"effnetb0-{variant}-alaska2-juniw.h5"
    with weights_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    summary["weights_sha256_first8"] = h.hexdigest()[:8]

    payload = {"summary": summary, "per_image": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[effnet] wrote {output}")
    print(f"[effnet] mean stego prob: {summary['mean_stego_prob']:.4f}")
    print(f"[effnet] fraction predicted stego: {summary['frac_predicted_stego']:.2%}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image-dir", type=Path, required=True, help="directory of JPEGs to score")
    p.add_argument("--variant", choices=["A", "B"], default="A",
                   help="aletheia ships A and B variants — 2-model ensemble pattern")
    p.add_argument("--output", type=Path, required=True, help="output JSON path")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    run(args.image_dir, args.variant, args.output, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
