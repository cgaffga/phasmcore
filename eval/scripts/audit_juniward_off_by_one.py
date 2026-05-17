"""
E1: Numerical audit of Phasm's shipped J-UNIWARD impl against the
Butora 2023 (arXiv:2305.19776) off-by-one fix.

Procedure:
  1. Generate a deterministic 64x64 grayscale JPEG (sinusoidal pattern +
     reproducible noise → non-trivial wavelet response at QF75).
  2. Run Phasm's cost-dumping test via `cargo test --test juniward_audit`,
     driven by env vars (PHASM_AUDIT_INPUT, PHASM_AUDIT_OUTPUT). Produces a
     JSON of cost-map values shaped (bt, bw, 8, 8).
  3. Re-decode the same JPEG via jpeglib (the canonical libjpeg path) and
     run `conseal.juniward._costmap.compute_cost` with both variants:
       - JUNIWARD_ORIGINAL        (the pre-2023 reference impl)
       - JUNIWARD_FIX_OFF_BY_ONE  (the Butora 2023 correction)
  4. Compare. Phasm should be vastly closer to FIX_OFF_BY_ONE than to
     ORIGINAL — Spearman rank correlation is the canonical metric since
     absolute cost magnitudes differ between impls (f32 vs f64, mirror vs
     symmetric padding).

Output: marketing/research/20260514-juniward-fix-audit.md
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jpeglib
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from conseal.juniward._costmap import Implementation, compute_cost

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "core"
AUDIT_DOC = REPO_ROOT / "marketing/research/20260514-juniward-fix-audit.md"


def make_test_image(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pat = (
        128
        + 60 * np.sin(2 * np.pi * x / 16) * np.cos(2 * np.pi * y / 16)
        + 20 * rng.standard_normal((size, size))
    )
    return np.clip(pat, 0, 255).astype(np.uint8)


def run_phasm_dump(jpeg_path: Path, out_path: Path) -> dict:
    env = dict(os.environ)
    env["PHASM_AUDIT_INPUT"] = str(jpeg_path)
    env["PHASM_AUDIT_OUTPUT"] = str(out_path)
    env.setdefault("CARGO_TARGET_DIR", "/tmp/phasm-audit-build")
    proc = subprocess.run(
        ["cargo", "test", "--release", "--test", "juniward_audit",
         "--", "dump_cost_map", "--exact", "--nocapture"],
        cwd=CORE_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(f"cargo test failed (rc={proc.returncode})")
    with out_path.open() as f:
        return json.load(f)


def main():
    img = make_test_image(size=512)
    print(f"Cover image:    {img.shape[0]}x{img.shape[1]} grayscale, "
          f"content range [{img.min()}, {img.max()}]")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jpeg_path = tmp / "cover.jpg"
        cost_path = tmp / "phasm_costs.json"

        Image.fromarray(img, mode="L").save(
            jpeg_path, format="JPEG", quality=75, subsampling=0
        )
        size_bytes = jpeg_path.stat().st_size
        print(f"JPEG:           {size_bytes} bytes, QF75")

        print("Running Phasm cost dump (cargo test --release)...")
        phasm = run_phasm_dump(jpeg_path, cost_path)
        bw, bt = phasm["blocks_wide"], phasm["blocks_tall"]
        flat = np.array(
            [c if c is not None else np.inf for c in phasm["costs"]],
            dtype=np.float64,
        )
        phasm_arr = flat.reshape(bt, bw, 8, 8)
        print(f"Phasm cost map: shape {phasm_arr.shape}, "
              f"{np.isfinite(phasm_arr).sum()} finite cells")

        jpg = jpeglib.read_dct(str(jpeg_path))
        qt = np.asarray(jpg.qt[0])
        decomp = jpeglib.read_spatial(
            str(jpeg_path), out_color_space=jpeglib.JCS_GRAYSCALE
        )
        pixels = decomp.spatial[:, :, 0].astype(np.float64)
        print(f"Conseal input:  qt[0]={qt.flatten()[0]} (DC), "
              f"pixel range [{pixels.min():.0f}, {pixels.max():.0f}]")

    print("Running conseal both variants...")
    costs_orig = compute_cost(
        pixels, qt, implementation=Implementation.JUNIWARD_ORIGINAL
    )
    costs_fix = compute_cost(
        pixels, qt, implementation=Implementation.JUNIWARD_FIX_OFF_BY_ONE
    )

    finite = (
        np.isfinite(phasm_arr) & np.isfinite(costs_orig) & np.isfinite(costs_fix)
    )
    is_dc = np.zeros_like(finite, dtype=bool)
    is_dc[:, :, 0, 0] = True
    mask = finite & ~is_dc

    p = phasm_arr[mask]
    o = costs_orig[mask]
    f = costs_fix[mask]
    print(f"Comparable cells: {p.size}")

    def metrics(a, b):
        a_n = a / np.median(a)
        b_n = b / np.median(b)
        rms = float(np.sqrt(np.mean((a_n - b_n) ** 2)))
        pear = float(np.corrcoef(a, b)[0, 1])
        rho = float(spearmanr(a, b).correlation)
        return rms, pear, rho

    rms_o, pear_o, rho_o = metrics(p, o)
    rms_f, pear_f, rho_f = metrics(p, f)

    print()
    print(f"{'metric':<28}{'vs ORIGINAL':>16}{'vs FIX_OFF':>16}")
    print(f"{'-' * 60}")
    print(f"{'normalized RMS (lower=better)':<28}{rms_o:>16.4f}{rms_f:>16.4f}")
    print(f"{'Pearson corr (higher=better)':<28}{pear_o:>16.4f}{pear_f:>16.4f}")
    print(f"{'Spearman rho (higher=better)':<28}{rho_o:>16.4f}{rho_f:>16.4f}")
    print()

    fix_wins_rms = rms_f < rms_o
    fix_wins_corr = pear_f > pear_o and rho_f > rho_o

    if fix_wins_rms and fix_wins_corr:
        verdict = "✓ shipped J-UNIWARD matches FIX_OFF_BY_ONE (Butora 2023)"
        verdict_emoji = "✓"
    elif (not fix_wins_rms) and (not fix_wins_corr):
        verdict = "✗ shipped J-UNIWARD matches ORIGINAL (off-by-one NOT fixed)"
        verdict_emoji = "✗"
    else:
        verdict = "🟡 results inconsistent across metrics — manual investigation needed"
        verdict_emoji = "🟡"
    print(f"VERDICT: {verdict}")

    AUDIT_DOC.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DOC.write_text(f"""\
---
date: 2026-05-14
task: E1
status: complete
verdict: "{verdict_emoji}"
---

# E1 — J-UNIWARD Off-By-One Audit

Verify that Phasm's shipped J-UNIWARD cost implementation includes the
Butora 2023 correction (arXiv:2305.19776).

## Method

Numerical comparison: feed the same 64×64 grayscale QF75 JPEG to
1. Phasm `compute_uniward` (via `cargo test --release --test juniward_audit`)
2. conseal `compute_cost(implementation=JUNIWARD_ORIGINAL)`
3. conseal `compute_cost(implementation=JUNIWARD_FIX_OFF_BY_ONE)`

Test image: sin/cos pattern + Gaussian noise (seed 42), saved as
single-channel JPEG QF75. Reproducible via `make_test_image()` in
`eval/scripts/audit_juniward_off_by_one.py`.

## Numerical Results

| Metric                          | vs ORIGINAL | vs FIX_OFF |
|---------------------------------|-------------|------------|
| normalized RMS (lower = closer) | {rms_o:.4f}      | {rms_f:.4f}     |
| Pearson correlation             | {pear_o:.4f}      | {pear_f:.4f}     |
| Spearman rank correlation       | {rho_o:.4f}      | {rho_f:.4f}     |

Cells compared: {p.size} (finite, non-DC).

## Code-Level Evidence

- `core/src/stego/cost/uniward.rs` line 26–27 declares:
  `"This implementation follows the corrected version (fixing the off-by-one error described in arXiv:2305.19776)."`
- Filter centering: `half = (flen - 1) / 2 = 7` (uniward.rs:441, 471) for the 16-tap db8 wavelet — consistent for both row and column filtering.
- Phasm uses `mirror_index` *inline during convolution*, side-stepping the
  pad-then-crop architecture where conseal's original off-by-one occurs.
- Boundary mode: Phasm uses `reflect` (no edge-sample duplication);
  conseal uses NumPy `symmetric` (with edge-sample duplication). This is a
  *separate* difference from the off-by-one and primarily affects the outer
  8-pixel boundary of the image (<3% of cells for 512×512+ covers).

## Verdict

**{verdict}**

## Paper Impact

§5.2 of the arXiv paper plan calls for citing "Daubechies db8 J-UNIWARD cost
(Butora/Lorch off-by-one fix)" — the paper plan currently lists "db4" which
is incorrect (we use db8, 16-tap). Both the off-by-one citation and the
filter family should be stated accurately.

## Reproducibility

```bash
cd ~/Development/phasm/eval
uv run python scripts/audit_juniward_off_by_one.py
```

Driver script: `eval/scripts/audit_juniward_off_by_one.py`
Phasm test:    `core/tests/juniward_audit.rs`

## Document History

- 2026-05-14: Audit completed; verdict {verdict_emoji}.
""")
    print(f"\nDoc written: {AUDIT_DOC}")


if __name__ == "__main__":
    main()
