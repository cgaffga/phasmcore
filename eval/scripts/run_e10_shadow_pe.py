"""E10 — SRNet PE-vs-shadow-N curve.

For each N in {0, 1, 2, 3, 4} (where N=0 means primary-only, no shadows),
score the BOSSbase QF75 cover + Phasm Ghost stego set with the
QF75@0.20 from-scratch SRNet detector (the headline §6.2 detector).

Reports PE per N on:
  (a) the universal subset (covers where stegos exist for ALL N in 0..4)
  (b) per-N maximum (whatever paired set is available for that N)

Cover dir : data/bossbase/jpeg_qf75/                  (10000 BOSSbase QF75)
Stego dirs: data/path3_shadow_eval/shadow_n{0..4}/    (100/91/79/59/40)
Detector  : runs/2026-05-15-e9-srnet-juniward-qf75-pf020-s042/checkpoints/best.pt
            (test PE 0.293 on its own QF75@0.20 test set)

Outputs in runs/2026-05-17-e10-shadow-pe-curve/:
  cover_scores.json        all-cover scores
  shadow_n{0..4}_scores.json
  per_n_pe.json            { N: { universal_pe, max_pe, n_universal, n_max } }
  RESULTS.md
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN_TRAINED_SRNET = ROOT / "detectors" / "run_trained_srnet.py"

CHECKPOINT = ROOT / "runs/2026-05-15-e9-srnet-juniward-qf75-pf020-s042/checkpoints/best.pt"
COVER_DIR  = ROOT / "data/bossbase/jpeg_qf75"
SHADOW_DIRS = {n: ROOT / f"data/path3_shadow_eval/shadow_n{n}" for n in range(5)}
OUT_DIR = ROOT / "runs/2026-05-17-e10-shadow-pe-curve"
DEVICE = "cpu"


def _scan_one(image_dir: Path, output: Path, limit: int) -> None:
    if output.exists():
        print(f"  [reuse] {output.name}")
        return
    cmd = [
        sys.executable, str(RUN_TRAINED_SRNET),
        "--image-dir", str(image_dir),
        "--checkpoint", str(CHECKPOINT),
        "--device", DEVICE,
        "--limit", str(limit),
        "--output", str(output),
    ]
    print(f"  [run]   {image_dir.name} -> {output.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"inference failed (rc={proc.returncode})")


def _load_scores(json_path: Path) -> dict[str, float]:
    j = json.loads(json_path.read_text())
    return {name: r["stego_prob"] for name, r in j["per_image"].items()}


def _pe(cover_probs: np.ndarray, stego_probs: np.ndarray) -> dict:
    fa = float((cover_probs > 0.5).mean()) if len(cover_probs) else 0.0
    md = float((stego_probs <= 0.5).mean()) if len(stego_probs) else 0.0
    pe = (fa + md) / 2
    return {
        "n_cover": int(len(cover_probs)),
        "n_stego": int(len(stego_probs)),
        "cover_mean": float(cover_probs.mean()) if len(cover_probs) else None,
        "stego_mean": float(stego_probs.mean()) if len(stego_probs) else None,
        "FA": fa, "MD": md, "PE": pe,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Identify all filenames present across the 5 shadow dirs and limit
    # cover scoring to that subset (much faster than scanning 10k covers).
    names_per_n = {n: set(p.name for p in d.iterdir() if p.suffix == ".jpg")
                   for n, d in SHADOW_DIRS.items()}
    needed_covers = set().union(*names_per_n.values())  # any stego we'll score
    universal_names = set.intersection(*names_per_n.values())  # in all 5 N
    print(f"[E10] cover|stego pairs requested = {len(needed_covers)}")
    print(f"[E10] universal subset (all 5 N exist) = {len(universal_names)}")

    # Build a transient cover-subset dir via symlinks. Run-trained-srnet
    # scans a directory — symlinks are cheaper than copies.
    cover_subset_dir = OUT_DIR / "cover_subset"
    cover_subset_dir.mkdir(exist_ok=True)
    for name in needed_covers:
        src = COVER_DIR / name
        if not src.exists():
            continue
        lnk = cover_subset_dir / name
        if not lnk.exists():
            lnk.symlink_to(src)

    n_cover_files = sum(1 for _ in cover_subset_dir.iterdir())
    print(f"[E10] cover subset dir has {n_cover_files} symlinks")

    # Inference: cover + 5 stego dirs.
    print(f"[E10] inference (device={DEVICE})...")
    cover_json = OUT_DIR / "cover_scores.json"
    _scan_one(cover_subset_dir, cover_json, limit=2000)
    stego_jsons = {}
    for n in range(5):
        sj = OUT_DIR / f"shadow_n{n}_scores.json"
        _scan_one(SHADOW_DIRS[n], sj, limit=2000)
        stego_jsons[n] = sj

    # Load per-image scores.
    cover_map = _load_scores(cover_json)
    stego_maps = {n: _load_scores(stego_jsons[n]) for n in range(5)}

    # Per-N PE: (a) universal subset, (b) max available subset.
    results = {}
    for n in range(5):
        # (a) universal subset
        cov_u = np.array([cover_map[name] for name in universal_names
                          if name in cover_map])
        ste_u = np.array([stego_maps[n][name] for name in universal_names
                          if name in stego_maps[n]])
        # (b) max available — all stegos for this N + their paired covers
        names_n = set(stego_maps[n].keys())
        cov_m = np.array([cover_map[name] for name in names_n
                          if name in cover_map])
        ste_m = np.array([stego_maps[n][name] for name in names_n])

        results[n] = {
            "universal_subset": _pe(cov_u, ste_u),
            "per_n_max":       _pe(cov_m, ste_m),
        }

    out = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "cover_dir": str(COVER_DIR.relative_to(ROOT)),
        "shadow_dirs": {n: str(SHADOW_DIRS[n].relative_to(ROOT)) for n in range(5)},
        "n_universal_subset": len(universal_names),
        "device": DEVICE,
        "per_N": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "per_n_pe.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"=== PE per N (universal subset, n={len(universal_names)}) ===")
    print(f"  {'N':>2}  {'cover_mean':>10}  {'stego_mean':>10}  {'FA':>6}  {'MD':>6}  {'PE':>6}")
    for n in range(5):
        r = results[n]["universal_subset"]
        print(f"  {n:>2}  {r['cover_mean']:>10.4f}  {r['stego_mean']:>10.4f}  "
              f"{r['FA']:>6.3f}  {r['MD']:>6.3f}  {r['PE']:>6.3f}")
    print()
    print(f"=== PE per N (per-N max) ===")
    print(f"  {'N':>2}  {'n_stego':>8}  {'cover_mean':>10}  {'stego_mean':>10}  {'PE':>6}")
    for n in range(5):
        r = results[n]["per_n_max"]
        print(f"  {n:>2}  {r['n_stego']:>8d}  {r['cover_mean']:>10.4f}  "
              f"{r['stego_mean']:>10.4f}  {r['PE']:>6.3f}")
    print()
    print(f"saved {OUT_DIR / 'per_n_pe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
