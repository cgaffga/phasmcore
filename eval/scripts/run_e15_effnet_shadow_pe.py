"""E15 — EfficientNet-pretrained PE-vs-shadow-N curve (mirrors E10 with EffNet).

For each N in {0, 1, 2, 3, 4} (where N=0 means primary-only, no shadows),
score the BOSSbase QF75 cover + Phasm Ghost stego set with the
EfficientNet-B0 (ImageNet-pretrained) fine-tuned at QF75@0.20.

The reference comparison is the §6.3 E10 SRNet curve. If EffNet shows
the same washing-out trend (PE rises with N), the shadow-security
claim is validated against the post-SRNet detector lineage; if EffNet
sees through the wash-out, the claim must be bounded to from-scratch
SRNet and that limitation flagged in §6/§7.

Cover dir : data/bossbase/jpeg_qf75/
Stego dirs: data/path3_shadow_eval/shadow_n{0..4}/    (100/91/79/59/40)

Outputs in runs/2026-05-17-e15-effnet-shadow-pe-curve/:
  cover_scores.json, shadow_n{0..4}_scores.json
  per_n_pe.json     { N: { universal_subset: {...}, per_n_max: {...} } }
  RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN_TRAINED_EFFNET = ROOT / "detectors" / "run_trained_efficientnet.py"

COVER_DIR = ROOT / "data/bossbase/jpeg_qf75"
SHADOW_DIRS = {n: ROOT / f"data/path3_shadow_eval/shadow_n{n}" for n in range(5)}


def find_default_checkpoint() -> Path:
    """Pick the most recent E15 EffNet checkpoint."""
    candidates = sorted(ROOT.glob("runs/*e15-effnet-juniward-qf75-pf020*/checkpoints/best.pt"))
    if not candidates:
        raise SystemExit(
            "no E15 EffNet checkpoint found under runs/*e15-effnet-juniward-qf75-pf020*; "
            "train one first via detectors/train_efficientnet.py."
        )
    return candidates[-1]


def scan_one(checkpoint: Path, image_dir: Path, output: Path, device: str, limit: int) -> None:
    if output.exists():
        print(f"  [reuse] {output.name}")
        return
    cmd = [
        sys.executable, str(RUN_TRAINED_EFFNET),
        "--image-dir", str(image_dir),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--limit", str(limit),
        "--output", str(output),
    ]
    print(f"  [run]   {image_dir.name} -> {output.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"inference failed (rc={proc.returncode})")


def load_scores(json_path: Path) -> dict[str, float]:
    j = json.loads(json_path.read_text())
    return {name: r["stego_prob"] for name, r in j["per_image"].items()}


def pe(cover_probs: np.ndarray, stego_probs: np.ndarray) -> dict:
    fa = float((cover_probs > 0.5).mean()) if len(cover_probs) else 0.0
    md = float((stego_probs <= 0.5).mean()) if len(stego_probs) else 0.0
    return {
        "n_cover": int(len(cover_probs)),
        "n_stego": int(len(stego_probs)),
        "cover_mean": float(cover_probs.mean()) if len(cover_probs) else None,
        "stego_mean": float(stego_probs.mean()) if len(stego_probs) else None,
        "FA": fa, "MD": md, "PE": (fa + md) / 2,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="EffNet checkpoint best.pt (auto-detects most recent E15 run if omitted)")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "runs/2026-05-17-e15-effnet-shadow-pe-curve")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--limit", type=int, default=2000)
    args = p.parse_args()

    checkpoint = args.checkpoint or find_default_checkpoint()
    print(f"[E15] checkpoint = {checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Universal subset = filenames present in ALL 5 shadow dirs.
    names_per_n = {n: set(p.name for p in d.iterdir() if p.suffix == ".jpg")
                   for n, d in SHADOW_DIRS.items()}
    needed_covers = set().union(*names_per_n.values())
    universal_names = set.intersection(*names_per_n.values())
    print(f"[E15] cover|stego pairs requested = {len(needed_covers)}")
    print(f"[E15] universal subset (all 5 N exist) = {len(universal_names)}")

    # Transient symlink dir so the inference scans only the needed covers.
    cover_subset_dir = args.out_dir / "cover_subset"
    cover_subset_dir.mkdir(exist_ok=True)
    for name in needed_covers:
        src = COVER_DIR / name
        if not src.exists():
            continue
        lnk = cover_subset_dir / name
        if not lnk.exists():
            lnk.symlink_to(src)

    print(f"[E15] inference (device={args.device})...")
    cover_json = args.out_dir / "cover_scores.json"
    scan_one(checkpoint, cover_subset_dir, cover_json, args.device, args.limit)
    stego_jsons: dict[int, Path] = {}
    for n in range(5):
        sj = args.out_dir / f"shadow_n{n}_scores.json"
        scan_one(checkpoint, SHADOW_DIRS[n], sj, args.device, args.limit)
        stego_jsons[n] = sj

    cover_map = load_scores(cover_json)
    stego_maps = {n: load_scores(stego_jsons[n]) for n in range(5)}

    results = {}
    for n in range(5):
        cov_u = np.array([cover_map[name] for name in universal_names
                          if name in cover_map])
        ste_u = np.array([stego_maps[n][name] for name in universal_names
                          if name in stego_maps[n]])
        names_n = set(stego_maps[n].keys())
        cov_m = np.array([cover_map[name] for name in names_n
                          if name in cover_map])
        ste_m = np.array([stego_maps[n][name] for name in names_n])
        results[n] = {
            "universal_subset": pe(cov_u, ste_u),
            "per_n_max":       pe(cov_m, ste_m),
        }

    def _rel(p: Path) -> str:
        p_abs = p if p.is_absolute() else (ROOT / p).resolve()
        try:
            return str(p_abs.relative_to(ROOT))
        except ValueError:
            return str(p_abs)

    out = {
        "checkpoint": _rel(checkpoint),
        "arch": "efficientnet_b0_imagenet1k",
        "cover_dir": _rel(COVER_DIR),
        "shadow_dirs": {n: _rel(SHADOW_DIRS[n]) for n in range(5)},
        "n_universal_subset": len(universal_names),
        "device": args.device,
        "per_N": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "per_n_pe.json").write_text(json.dumps(out, indent=2))

    print()
    print(f"=== EfficientNet PE per N (universal subset, n={len(universal_names)}) ===")
    print(f"  {'N':>2}  {'cover_mean':>10}  {'stego_mean':>10}  {'FA':>6}  {'MD':>6}  {'PE':>6}")
    for n in range(5):
        r = results[n]["universal_subset"]
        print(f"  {n:>2}  {r['cover_mean']:>10.4f}  {r['stego_mean']:>10.4f}  "
              f"{r['FA']:>6.3f}  {r['MD']:>6.3f}  {r['PE']:>6.3f}")
    print()
    print(f"=== EfficientNet PE per N (per-N max) ===")
    print(f"  {'N':>2}  {'n_stego':>8}  {'cover_mean':>10}  {'stego_mean':>10}  {'PE':>6}")
    for n in range(5):
        r = results[n]["per_n_max"]
        print(f"  {n:>2}  {r['n_stego']:>8d}  {r['cover_mean']:>10.4f}  "
              f"{r['stego_mean']:>10.4f}  {r['PE']:>6.3f}")
    print()
    print(f"saved {args.out_dir / 'per_n_pe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
