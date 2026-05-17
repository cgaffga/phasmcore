"""Cross-distribution detector evaluation: detector × (cover, stego) -> PE.

Wraps `run_trained_srnet.py` to run inference on a (cover_dir, stego_dir)
pair and compute the standard steganalysis decision-error rate
$P_E = (P_\\text{FA} + P_\\text{MD}) / 2$ at threshold 0.5.

Used by:
  - §6.2 cross-stego transfer (J-UNI detector → Phasm Ghost stego)
  - E11 cross-QF generalization (QF75 detector → QF95 stego)
  - E12 cross-source generalization (BOSSbase detector → ALASKA stego,
    auto color→Y via Pillow `convert("L")` in run_trained_srnet)

Usage:
  python detectors/cross_eval_pe.py \\
    --checkpoint runs/2026-05-15-e9-srnet-juniward-qf75-pf020-s042/checkpoints/best.pt \\
    --cover-dir data/alaska2/cover \\
    --stego-dir data/path2_alaska2_eval/phasm_ghost_100 \\
    --out-dir runs/2026-05-16-e12-cross-source-bossbase-to-alaska-pf020 \\
    --label "BOSSbase J-UNI@QF75@0.20 -> ALASKA Phasm Ghost@100% capacity" \\
    --limit 1000

Outputs:
  out_dir/cover_scores.json
  out_dir/stego_scores.json
  out_dir/results.json    {detector_path, label, n, FA, MD, PE, AUC, ...}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
RUN_TRAINED_SRNET = ROOT / "detectors" / "run_trained_srnet.py"


def _auc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    all_s = np.concatenate([neg_scores, pos_scores])
    labels = np.concatenate([
        np.zeros(len(neg_scores)),
        np.ones(len(pos_scores)),
    ])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    ranks = rankdata(all_s)
    rs_pos = ranks[labels == 1].sum()
    n_p = int(labels.sum())
    n_n = len(labels) - n_p
    return float((rs_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def _scan_one(checkpoint: Path, image_dir: Path, output: Path,
              device: str, limit: int) -> None:
    """Invoke run_trained_srnet.py on a directory."""
    if output.exists():
        print(f"  [reuse] {output}")
        return
    cmd = [
        sys.executable, str(RUN_TRAINED_SRNET),
        "--image-dir", str(image_dir),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--limit", str(limit),
        "--output", str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"inference failed (rc={proc.returncode})")
    print(f"  [done]  {output}")


def _load_scores(json_path: Path) -> np.ndarray:
    j = json.loads(json_path.read_text())
    return np.array([r["stego_prob"] for r in j["per_image"].values()])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cover-dir", type=Path, required=True)
    p.add_argument("--stego-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="",
                   help="Human-readable description for results.json")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu/mps/cuda — defaults to cpu to avoid GPU contention")
    p.add_argument("--limit", type=int, default=1000,
                   help="Max images per dir (default 1000 for fast turnaround)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cover_json = args.out_dir / "cover_scores.json"
    stego_json = args.out_dir / "stego_scores.json"

    print(f"[cross_eval] cover dir: {args.cover_dir}")
    _scan_one(args.checkpoint, args.cover_dir, cover_json, args.device, args.limit)
    print(f"[cross_eval] stego dir: {args.stego_dir}")
    _scan_one(args.checkpoint, args.stego_dir, stego_json, args.device, args.limit)

    cover_scores = _load_scores(cover_json)
    stego_scores = _load_scores(stego_json)

    fa = float((cover_scores > 0.5).mean())
    md = float((stego_scores <= 0.5).mean())
    pe = (fa + md) / 2
    auc = _auc(cover_scores, stego_scores)

    n_cover = len(cover_scores)
    n_stego = len(stego_scores)
    print()
    print(f"  label: {args.label}")
    print(f"  n_cover={n_cover}  n_stego={n_stego}")
    print(f"  cover mean stego_prob = {cover_scores.mean():.4f}  (std {cover_scores.std():.4f})")
    print(f"  stego mean stego_prob = {stego_scores.mean():.4f}  (std {stego_scores.std():.4f})")
    print(f"  FA = {fa:.4f}   MD = {md:.4f}   PE = (FA+MD)/2 = {pe:.4f}")
    print(f"  AUC = {auc:.4f}")

    results = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "cover_dir": str(args.cover_dir),
        "stego_dir": str(args.stego_dir),
        "n_cover": n_cover,
        "n_stego": n_stego,
        "cover_mean_stego_prob": float(cover_scores.mean()),
        "stego_mean_stego_prob": float(stego_scores.mean()),
        "FA": fa,
        "MD": md,
        "PE": pe,
        "AUC": auc,
        "device": args.device,
        "limit": args.limit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"  saved {args.out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
