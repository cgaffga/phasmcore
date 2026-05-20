"""E18 — Quantitative steganalysis of Phasm Ghost cost-pool shadows.

Stage 1: re-analyse existing E10/E15/E17 per-image stego_prob JSONs as
quantitative N predictors. No training; pure analysis on numbers we
already have.

For each of three detectors (E10 SRNet from-scratch, E15 EffNet-B0 40-ep
fine-tune, E17 EffNet-B0 100-ep fine-tune):
  - Build the universal subset: images where N=0..4 all have scores.
  - Compute pooled Spearman rho(stego_prob, N) across the (image, N) grid.
  - Compute per-image Spearman rho: for each image, correlate (N, prob)
    across N=0..4; report the mean and 95%-CI of those per-image rhos.
  - Fit a linear regressor predicted_N = a*prob + b on a 50/50 holdout;
    report MAE.
  - Rank-2 accuracy: among all image-pairs (i, j) with N_i < N_j,
    P(predicted_N_i < predicted_N_j). The continuous analog of binary
    AUC.

Output: runs/2026-05-20-e18-quant-stego/stage1_results.json + RESULTS.md
table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

DETECTORS = {
    "E10-SRNet": ROOT / "runs/2026-05-17-e10-shadow-pe-curve",
    "E15-EffNet-40ep": ROOT / "runs/2026-05-17-e15-effnet-shadow-pe-curve",
    "E17-EffNet-100ep": ROOT / "runs/2026-05-18-e17-effnet-shadow-pe-curve-ep100",
}

OUT_DIR = ROOT / f"runs/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-e18-quant-stego"


def load_per_image(json_path: Path) -> dict[str, float]:
    d = json.loads(json_path.read_text())
    return {name: rec["stego_prob"] for name, rec in d["per_image"].items()}


def per_image_matrix(detector_dir: Path) -> dict[str, dict[int, float]]:
    """Return {image_name: {N: stego_prob, ...}} for N in 0..4."""
    out: dict[str, dict[int, float]] = {}
    for n in range(5):
        scores = load_per_image(detector_dir / f"shadow_n{n}_scores.json")
        for name, prob in scores.items():
            out.setdefault(name, {})[n] = prob
    return out


def analyse(detector_name: str, detector_dir: Path) -> dict:
    matrix = per_image_matrix(detector_dir)
    # Universal subset
    universal = sorted(name for name, by_n in matrix.items() if len(by_n) == 5)
    if not universal:
        return {"detector": detector_name, "error": "no universal-subset images"}

    # Build the (logit, N) point cloud
    points = []  # list of (image_name, N, prob) triples
    for name in universal:
        for n in range(5):
            points.append((name, n, matrix[name][n]))
    probs_arr = np.array([p[2] for p in points])
    ns_arr = np.array([p[1] for p in points])

    # --- Pooled Spearman ---
    rho_pooled, p_pooled = stats.spearmanr(probs_arr, ns_arr)

    # --- Per-image Spearman ---
    per_image_rhos = []
    for name in universal:
        ns_image = np.array([0, 1, 2, 3, 4])
        probs_image = np.array([matrix[name][n] for n in range(5)])
        rho_i, _ = stats.spearmanr(probs_image, ns_image)
        if not np.isnan(rho_i):
            per_image_rhos.append(rho_i)
    per_image_rhos = np.array(per_image_rhos)
    rho_perimage_mean = float(per_image_rhos.mean())
    rho_perimage_std = float(per_image_rhos.std(ddof=1))
    rho_perimage_ci95 = (
        rho_perimage_mean - 1.96 * rho_perimage_std / np.sqrt(len(per_image_rhos)),
        rho_perimage_mean + 1.96 * rho_perimage_std / np.sqrt(len(per_image_rhos)),
    )

    # --- Linear regression (50/50 holdout, deterministic split by hash of image name) ---
    rng = np.random.default_rng(42)
    names_arr = np.array(universal)
    perm = rng.permutation(len(names_arr))
    split = len(names_arr) // 2
    train_names = set(names_arr[perm[:split]])
    test_names = set(names_arr[perm[split:]])

    train_probs = [matrix[n][k] for n in train_names for k in range(5)]
    train_ns = [k for n in train_names for k in range(5)]
    test_probs = [matrix[n][k] for n in test_names for k in range(5)]
    test_ns = [k for n in test_names for k in range(5)]

    # Fit y = a*x + b on train
    A = np.vstack([train_probs, np.ones(len(train_probs))]).T
    coef, _, _, _ = np.linalg.lstsq(A, train_ns, rcond=None)
    a, b = coef
    test_pred = a * np.array(test_probs) + b
    mae = float(np.mean(np.abs(test_pred - np.array(test_ns))))

    # --- Rank-2 accuracy ---
    # Among all triples (image, N_lo, N_hi) with N_lo < N_hi, what fraction
    # have prob(N_lo) ranked correctly vs prob(N_hi) by the linear regressor?
    # For monotonic-decreasing detectors (SRNet wash-out), N_lo prob > N_hi prob.
    # For monotonic-increasing (typical detector), N_lo prob < N_hi prob.
    # We need the sign of `a` to decide direction; use predicted N consistently.
    correct = 0
    total = 0
    for n_lo in range(5):
        for n_hi in range(n_lo + 1, 5):
            for name in universal:
                pred_lo = a * matrix[name][n_lo] + b
                pred_hi = a * matrix[name][n_hi] + b
                if pred_lo < pred_hi:
                    correct += 1
                total += 1
    rank_acc = correct / total if total else float("nan")

    # --- Per-N logit distribution ---
    per_n_stats = {}
    for n in range(5):
        probs_n = np.array([matrix[name][n] for name in universal])
        per_n_stats[n] = {
            "mean": float(probs_n.mean()),
            "std": float(probs_n.std(ddof=1)),
            "n_samples": int(len(probs_n)),
        }

    return {
        "detector": detector_name,
        "n_universal": len(universal),
        "rho_pooled": float(rho_pooled),
        "rho_pooled_p": float(p_pooled),
        "rho_perimage_mean": rho_perimage_mean,
        "rho_perimage_std": rho_perimage_std,
        "rho_perimage_ci95": rho_perimage_ci95,
        "linear_regressor": {
            "a": float(a),
            "b": float(b),
            "test_mae_in_N_units": mae,
        },
        "rank_2_accuracy": rank_acc,
        "per_n_probs": per_n_stats,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, dir_ in DETECTORS.items():
        if not dir_.exists():
            print(f"[E18-S1] SKIP {name}: {dir_} not found")
            continue
        print(f"[E18-S1] {name}...")
        results[name] = analyse(name, dir_)

    out_path = OUT_DIR / "stage1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[E18-S1] wrote {out_path}")
    print()

    # Pretty table
    print("=" * 92)
    print(f"{'Detector':<22} {'n_univ':>6} {'rho_pool':>10} {'rho_perimg':>12} {'lin MAE (N)':>13} {'rank-2 acc':>11}")
    print("-" * 92)
    for name, r in results.items():
        print(
            f"{name:<22} {r['n_universal']:>6} {r['rho_pooled']:>+10.3f} "
            f"{r['rho_perimage_mean']:>+12.3f} {r['linear_regressor']['test_mae_in_N_units']:>13.3f} "
            f"{r['rank_2_accuracy']:>11.3f}"
        )
    print("=" * 92)
    print()
    print("Per-N stego_prob means (sanity — matches paper Table IV):")
    for name, r in results.items():
        means = [r["per_n_probs"][n]["mean"] for n in range(5)]
        print(f"  {name:<22}  " + "  ".join(f"N={n}:{m:.3f}" for n, m in enumerate(means)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
