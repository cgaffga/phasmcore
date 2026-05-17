"""Capacity sweep for paper §6.5.

Calls `phasm capacity --json` on a curated sample from each cover bucket
and records per-image dimensions, chroma layout, QF estimate, Y-channel
nzAC count, and the five capacity numbers (ghost / ghostDeepCover / armor
/ fortress / shadow). Computes derived bpnzAC rates so the numbers
connect to standard steganalysis literature.

Output: per-image CSV + bucket summary JSON + scatter plot
(nzAC vs ghost capacity, colored by bucket).

Buckets:
  - bossbase_qf75_gray:   100 stratified samples (canonical testbed)
  - alaska2_color:        100 stratified samples (production-color baseline)
  - real_world_jpeg:      9 test-vectors + Desktop/02445.jpg (multi-MP photos)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import jpeglib
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# Standard libjpeg Annex K luminance ref table value at position (0,1)
QT_REF_01 = 11


def estimate_qf(qt0: np.ndarray) -> int:
    """Rough QF estimate from luminance quant table position (0,1).

    libjpeg formula (reverse):
        for QF >= 50:  qt_val ≈ ref * (200 - 2*QF) / 100
        for QF < 50:   qt_val ≈ ref * 5000 / (QF * 100)
    Returns nearest integer in [1, 100].
    """
    q = int(qt0[0, 1])
    if q <= 0:
        return 100
    # scale_factor = q * 100 / ref
    scale = q * 100.0 / QT_REF_01
    if scale <= 100:
        qf = (200 - scale) / 2
    else:
        qf = 5000 / scale
    return int(round(max(1, min(100, qf))))


def chroma_layout(im) -> str:
    """Return 'gray', '4:4:4', '4:2:2', '4:2:0', or 'other'."""
    if im.Cb is None:
        return "gray"
    yh, yw = im.Y.shape[:2]
    ch, cw = im.Cb.shape[:2]
    if yh == ch and yw == cw:
        return "4:4:4"
    if yh == ch and yw == 2 * cw:
        return "4:2:2"
    if yh == 2 * ch and yw == 2 * cw:
        return "4:2:0"
    return f"other_{yh}x{yw}/{ch}x{cw}"


def y_nzAC(dct_y: np.ndarray) -> int:
    """Count Y-channel non-zero AC coefficients (DC at [0,0] excluded)."""
    total_nz = int((dct_y != 0).sum())
    dc_nz = int((dct_y[..., 0, 0] != 0).sum())
    return max(0, total_nz - dc_nz)


def phasm_capacity(path: Path, bin_path: str) -> dict:
    """Call `phasm capacity --json`. Returns dict with 5 fields or {}."""
    try:
        out = subprocess.run(
            [bin_path, "capacity", str(path), "--json"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        return json.loads(out)
    except subprocess.SubprocessError as e:
        print(f"  capacity failed for {path.name}: {e}", file=sys.stderr)
        return {}
    except json.JSONDecodeError:
        return {}


def probe(path: Path, bin_path: str) -> dict | None:
    """Probe one image. Returns a flat dict of metrics."""
    try:
        pil = Image.open(path)
        width, height = pil.size
    except Exception as e:
        return {"name": path.name, "error": f"PIL: {e}"}

    try:
        dct = jpeglib.read_dct(str(path))
    except Exception as e:
        return {"name": path.name, "error": f"jpeglib: {e}"}

    layout = chroma_layout(dct)
    nz = y_nzAC(dct.Y) if dct.Y is not None else 0
    qf = estimate_qf(dct.qt[0]) if dct.qt is not None else 0

    cap = phasm_capacity(path, bin_path)

    ghost = int(cap.get("ghost", 0))
    ghost_dc = int(cap.get("ghostDeepCover", 0))
    armor = int(cap.get("armor", 0))
    fortress = int(cap.get("fortress", 0))
    shadow = int(cap.get("shadow", 0))

    ghost_bpnzAC = (ghost * 8 / nz) if nz > 0 else 0.0
    shadow_bpnzAC = (shadow * 8 / nz) if nz > 0 else 0.0
    shadow_per_ghost = (shadow / ghost) if ghost > 0 else 0.0

    return {
        "name": path.name,
        "width": width,
        "height": height,
        "megapixels": (width * height) / 1e6,
        "chroma": layout,
        "qf_est": qf,
        "nzAC_Y": nz,
        "ghost": ghost,
        "ghostDeepCover": ghost_dc,
        "armor": armor,
        "fortress": fortress,
        "shadow": shadow,
        "ghost_bpnzAC": ghost_bpnzAC,
        "shadow_bpnzAC": shadow_bpnzAC,
        "shadow_per_ghost_ratio": shadow_per_ghost,
    }


def find_phasm_binary() -> str | None:
    bin_dir = ROOT / "bin"
    if bin_dir.is_dir():
        candidates = sorted(bin_dir.glob("phasm-*"), reverse=True)
        for c in candidates:
            if c.is_file():
                return str(c)
    import shutil
    return shutil.which("phasm")


def stratified_sample(paths: list[Path], n: int, seed: int = 2026) -> list[Path]:
    """Stratify by sha256(name) for deterministic sample selection."""
    if len(paths) <= n:
        return paths
    keyed = sorted(paths, key=lambda p: hashlib.sha256(f"{p.name}|{seed}".encode()).hexdigest())
    return keyed[:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "runs" / "2026-05-12-capacity-sweep")
    p.add_argument("--n-bossbase", type=int, default=100)
    p.add_argument("--n-alaska2", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    bin_path = find_phasm_binary()
    if not bin_path:
        print("ERROR: no phasm binary", file=sys.stderr)
        return 1
    print(f"[sweep] phasm bin: {bin_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Bucket assembly.
    buckets: list[tuple[str, list[Path]]] = []

    boss_dir = ROOT / "data" / "bossbase" / "jpeg_qf75"
    if boss_dir.exists():
        boss_files = sorted(boss_dir.glob("*.jpg"))
        boss_sample = stratified_sample(boss_files, args.n_bossbase, args.seed)
        buckets.append(("bossbase_qf75_gray", boss_sample))

    alaska_dir = ROOT / "data" / "alaska2" / "cover"
    if alaska_dir.exists():
        alaska_files = sorted(alaska_dir.glob("*.jpg"))
        alaska_sample = stratified_sample(alaska_files, args.n_alaska2, args.seed)
        buckets.append(("alaska2_color", alaska_sample))

    realworld: list[Path] = []
    tv = ROOT.parent / "core" / "test-vectors" / "image"
    if tv.exists():
        for jp in sorted(tv.glob("*.jpg")):
            realworld.append(jp)
    big = Path.home() / "Desktop" / "02445.jpg"
    if big.exists():
        realworld.append(big)
    if realworld:
        buckets.append(("real_world_jpeg", realworld))

    rows: list[dict] = []
    for bname, paths in buckets:
        print(f"[sweep] bucket {bname}: {len(paths)} images")
        t0 = time.time()
        for path in tqdm(paths, desc=bname):
            row = probe(path, bin_path)
            if row is None:
                continue
            row["bucket"] = bname
            rows.append(row)
        print(f"[sweep]   bucket wall: {time.time() - t0:.1f}s")

    # Write per-image CSV.
    import csv
    out_csv = args.out_dir / "per_image.csv"
    if rows:
        fields = sorted({k for r in rows for k in r.keys()})
        with out_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[sweep] wrote {out_csv} ({len(rows)} rows)")

    # Bucket summary JSON.
    summary: dict[str, dict] = {}
    for bname, _ in buckets:
        bucket_rows = [r for r in rows if r.get("bucket") == bname and "error" not in r]
        if not bucket_rows:
            continue

        def stats(field: str) -> dict:
            vals = [r[field] for r in bucket_rows if isinstance(r.get(field), (int, float))]
            if not vals:
                return {}
            arr = np.array(vals, dtype=np.float64)
            return {
                "n": len(vals),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "std": float(arr.std()),
            }

        summary[bname] = {
            "n_images": len(bucket_rows),
            "ghost_bytes": stats("ghost"),
            "ghostDeepCover_bytes": stats("ghostDeepCover"),
            "armor_bytes": stats("armor"),
            "fortress_bytes": stats("fortress"),
            "shadow_bytes": stats("shadow"),
            "ghost_bpnzAC": stats("ghost_bpnzAC"),
            "shadow_bpnzAC": stats("shadow_bpnzAC"),
            "shadow_per_ghost_ratio": stats("shadow_per_ghost_ratio"),
            "nzAC_Y": stats("nzAC_Y"),
            "megapixels": stats("megapixels"),
            "qf_est": stats("qf_est"),
        }

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[sweep] wrote {out_json}")

    # Scatter plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "bossbase_qf75_gray": "#4a7",
            "alaska2_color": "#46c",
            "real_world_jpeg": "#c64",
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for bname, _ in buckets:
            br = [r for r in rows if r.get("bucket") == bname and "error" not in r]
            if not br:
                continue
            nz = [r["nzAC_Y"] for r in br]
            gh = [r["ghost"] for r in br]
            sh = [r["shadow"] for r in br]
            axes[0].scatter(nz, gh, label=bname, alpha=0.6, s=20, color=colors.get(bname, "#888"))
            axes[1].scatter(nz, sh, label=bname, alpha=0.6, s=20, color=colors.get(bname, "#888"))

        axes[0].set_xlabel("Y-channel nzAC coefficient count")
        axes[0].set_ylabel("Phasm Ghost primary capacity (bytes)")
        axes[0].set_title("Primary STC capacity vs cover nzAC")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Y-channel nzAC coefficient count")
        axes[1].set_ylabel("Phasm shadow per-layer capacity (bytes)")
        axes[1].set_title("Shadow per-layer capacity vs cover nzAC")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = args.out_dir / "capacity_scatter.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[sweep] wrote {fig_path}")
    except Exception as e:
        print(f"[sweep] scatter plot skipped: {e}", file=sys.stderr)

    # Pretty-print summary
    print()
    print("=" * 78)
    for bname, s in summary.items():
        print(f"\n## {bname} (n={s['n_images']})")
        print(f"  megapixels:    median {s['megapixels']['median']:7.3f} "
              f"[{s['megapixels']['min']:.3f} – {s['megapixels']['max']:.3f}]")
        print(f"  QF estimate:   median {s['qf_est']['median']:>6.0f} "
              f"[{s['qf_est']['min']:.0f} – {s['qf_est']['max']:.0f}]")
        print(f"  nzAC_Y:        median {s['nzAC_Y']['median']:>10,.0f}")
        print(f"  ghost bytes:   median {s['ghost_bytes']['median']:>10,.0f}  "
              f"mean {s['ghost_bytes']['mean']:>10,.0f}  "
              f"std {s['ghost_bytes']['std']:>10,.0f}")
        print(f"  shadow bytes:  median {s['shadow_bytes']['median']:>10,.0f}  "
              f"mean {s['shadow_bytes']['mean']:>10,.0f}  "
              f"std {s['shadow_bytes']['std']:>10,.0f}")
        print(f"  ghost bpnzAC:  median {s['ghost_bpnzAC']['median']:.4f}  "
              f"mean {s['ghost_bpnzAC']['mean']:.4f}")
        print(f"  shadow bpnzAC: median {s['shadow_bpnzAC']['median']:.4f}  "
              f"mean {s['shadow_bpnzAC']['mean']:.4f}")
        print(f"  shadow/ghost:  median {s['shadow_per_ghost_ratio']['median']:5.2f}×  "
              f"mean {s['shadow_per_ghost_ratio']['mean']:5.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
