"""Run aletheia's pretrained SRNet S-UNIWARD/BOSSbase detector on images.

Loads a `.keras` SRNet model from data/pretrained_models/aletheia/ (downloaded
from github.com/daniellerch/aletheia/aletheia-models, MIT-licensed) and runs
inference on a directory of images (PGM, PNG, or JPEG — all decoded to
512x512x3 spatial pixels via Y-replication).

The aletheia SRNet variants we use:
- srnet-A-sd-suniw-0.10.keras: SRNet trained on S-UNIWARD spatial-domain at 0.10 bpp
- srnet-A-sd-suniw-0.20.keras: SRNet trained on S-UNIWARD spatial-domain at 0.20 bpp

`sd` = spatial domain. The model expects raw decoded pixel values in spatial domain
(not JPEG DCT). When evaluating Phasm Ghost output (JPEG with J-UNIWARD embedding
in DCT), we decompress the JPEG to spatial pixels first — this introduces a
domain-mismatch caveat (model trained on uncompressed S-UNIWARD; we feed
JPEG-decompressed J-UNIWARD).

Output: per-image stego_prob written to a JSON + summary printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "data" / "pretrained_models" / "aletheia"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_model(payload: str = "0.10"):
    """Load aletheia's srnet-A-sd-suniw-{payload}.keras.

    payload ∈ {"0.10", "0.20"}.
    """
    from tensorflow.keras.models import load_model as keras_load_model

    weights = WEIGHTS_DIR / f"srnet-A-sd-suniw-{payload}.keras"
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights missing: {weights}\n"
            f"Download via: curl -sSL https://github.com/daniellerch/aletheia/raw/master/aletheia-models/srnet-A-sd-suniw-{payload}.keras -o {weights}"
        )
    return keras_load_model(str(weights), compile=False)


def preprocess(image_path: Path, target_size: int = 512):
    """Load image (PGM/PNG/JPEG) → (1, target_size, target_size, 3) float32 in [0, 255]."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        if im.mode == "L":
            im = im.convert("RGB")  # replicate Y to 3 channels
        elif im.mode != "RGB":
            im = im.convert("RGB")
        if im.size != (target_size, target_size):
            w, h = im.size
            if w >= target_size and h >= target_size:
                left = (w - target_size) // 2
                top = (h - target_size) // 2
                im = im.crop((left, top, left + target_size, top + target_size))
            else:
                im = im.resize((target_size, target_size), Image.LANCZOS)
        # Aletheia uses div255=True for SRNet (see aletheialib/models.py:231).
        # Without this, every prediction saturates to stego_prob=1.0.
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def run(image_dir: Path, payload: str, output: Path, batch_size: int = 8) -> dict:
    import numpy as np
    from tqdm import tqdm

    images = (
        sorted(image_dir.glob("*.jpg"))
        + sorted(image_dir.glob("*.jpeg"))
        + sorted(image_dir.glob("*.png"))
        + sorted(image_dir.glob("*.pgm"))
    )
    if not images:
        raise FileNotFoundError(f"no images in {image_dir}")

    print(f"[srnet] loading payload {payload}...")
    model = load_model(payload)
    print(f"[srnet] {len(images)} images in {image_dir}")

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

    for path in tqdm(images, desc=f"srnet-suniw-{payload}"):
        batch.append(preprocess(path))
        batch_paths.append(path)
        if len(batch) >= batch_size:
            flush()
    flush()

    # Hash the weights for reproducibility
    h = hashlib.sha256()
    weights_path = WEIGHTS_DIR / f"srnet-A-sd-suniw-{payload}.keras"
    with weights_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    summary = {
        "image_dir": str(image_dir),
        "model": f"srnet-A-sd-suniw-{payload}",
        "weights_sha256_first8": h.hexdigest()[:8],
        "n_images": len(images),
        "mean_stego_prob": float(np.mean([r["stego_prob"] for r in results.values()])),
        "frac_predicted_stego": float(
            sum(1 for r in results.values() if r["predicted"] == "stego") / len(results)
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload_obj = {"summary": summary, "per_image": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload_obj, indent=2))
    print(f"[srnet] wrote {output}")
    print(f"[srnet] mean stego prob: {summary['mean_stego_prob']:.4f}")
    print(f"[srnet] fraction predicted stego: {summary['frac_predicted_stego']:.2%}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--payload", choices=["0.10", "0.20"], default="0.10",
                   help="Which aletheia SRNet variant: 0.10 or 0.20 bpp")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    run(args.image_dir, args.payload, args.output, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
