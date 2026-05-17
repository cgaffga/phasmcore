"""E13 (SI-UNIWARD slice) — SI-UNIWARD Phasm Ghost stegos at N=0..4 shadows.

Compares J-UNIWARD-primary Phasm Ghost (from E10 — `data/path3_shadow_eval/`)
against SI-UNIWARD-primary Phasm Ghost on the same 40 universal-subset
covers. SI-UNIWARD auto-activates when Phasm sees non-JPEG input (PGM/PNG),
exploiting the JPEG quantisation rounding errors that exist between the
uncompressed source and the final QF75 JPEG.

Pipeline (per cover):
  1. PGM -> PNG (Pillow, lossless)
  2. phasm encode <png> --mode ghost --qf 75 ...   -> stego (auto SI-UNIWARD)
     for each N in {0, 1, 2, 3, 4}

Cover for the inference comparison: data/bossbase/jpeg_qf75/<name>.jpg
(same as E10; the libjpeg-encoded canonical BOSSbase QF75 JPEG). The
SI-UNIWARD output JPEG is encoded by Phasm's own pure-Rust codec at QF75,
so there's a tiny codec-distribution shift between cover and stego --
unavoidable for this ablation. We accept it since the cross-N trend is the
headline.

Intermediate PGMs and PNGs are removed after stego generation to keep disk
clean (per CLAUDE.md "regenerable inputs not in git").

Outputs in runs/2026-05-17-e13-siuniward-ablation/:
  stego_si/n{0..4}/<name>.jpg
  cover_subset/         (symlinks to data/bossbase/jpeg_qf75/<name>.jpg)
  cover_scores.json, stego_n{0..4}_si_scores.json
  per_n_pe.json
  RESULTS.md
"""
from __future__ import annotations

import hashlib
import json
import shutil
import string
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
OUT_DIR = ROOT / "runs/2026-05-17-e13-siuniward-ablation"
PGM_DIR = OUT_DIR / "pgms" / "BOSSbase_1.01"
PNG_DIR = OUT_DIR / "pngs"
STEGO_BASE = OUT_DIR / "stego_si"
COVER_SUBSET = OUT_DIR / "cover_subset"
COVER_SRC = ROOT / "data/bossbase/jpeg_qf75"
NAMES_FILE = OUT_DIR / "universal_names.txt"   # written by earlier prep step

CHECKPOINT = ROOT / "runs/2026-05-15-e9-srnet-juniward-qf75-pf020-s042/checkpoints/best.pt"
RUN_TRAINED_SRNET = ROOT / "detectors/run_trained_srnet.py"

PRIMARY_PASS = "phasm-eval-primary-2026"
SHADOW_PASSES = [
    "phasm-eval-shadow1-2026",
    "phasm-eval-shadow2-2026",
    "phasm-eval-shadow3-2026",
    "phasm-eval-shadow4-2026",
]
MSG_BYTES = 64
SEED = 2026
WORKERS = 8


def find_phasm() -> str:
    cands = sorted(BIN_DIR.glob("phasm-*"), reverse=True)
    for c in cands:
        if c.is_file():
            return str(c)
    raise SystemExit("phasm binary not found in eval/bin/")


def derive_message(cover_name: str, slot: str) -> str:
    h = hashlib.sha256(f"{cover_name}|{slot}|{SEED}".encode()).digest()
    chars = string.ascii_letters + string.digits
    out: list[str] = []
    counter = 0
    while len(out) < MSG_BYTES:
        counter += 1
        h2 = hashlib.sha256(h + counter.to_bytes(4, "big")).digest()
        for b in h2:
            out.append(chars[b % len(chars)])
            if len(out) >= MSG_BYTES:
                break
    return "".join(out)


def pgm_to_png(args: tuple[Path, Path]) -> tuple[str, bool, str]:
    src, dst = args
    if dst.exists():
        return src.name, True, "skipped"
    try:
        Image.open(src).save(dst)
        return src.name, True, "ok"
    except Exception as exc:
        return src.name, False, f"{type(exc).__name__}: {exc}"


def encode_one(args: tuple[Path, Path, int, str]) -> tuple[str, int, bool, str]:
    src_png, dst_jpg, n_shadows, phasm_bin = args
    if dst_jpg.exists():
        return src_png.name, n_shadows, True, "skipped"
    primary = derive_message(src_png.name, "primary")
    cmd = [
        phasm_bin, "encode", str(src_png),
        "--mode", "ghost",
        "--qf", "75",
        "-o", str(dst_jpg),
        "-m", primary,
        "-p", PRIMARY_PASS,
        "--quiet",
    ]
    for i in range(n_shadows):
        slot = f"shadow{i+1}"
        cmd += [f"--m{i+2}", derive_message(src_png.name, slot),
                f"--p{i+2}", SHADOW_PASSES[i]]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return src_png.name, n_shadows, False, "timeout"
    if res.returncode != 0:
        return src_png.name, n_shadows, False, f"phasm exit {res.returncode}: {res.stderr.strip()[:200]}"
    return src_png.name, n_shadows, True, "ok"


def scan_dir(image_dir: Path, output_json: Path, limit: int = 200) -> None:
    if output_json.exists():
        print(f"  [reuse] {output_json.name}")
        return
    cmd = [
        sys.executable, str(RUN_TRAINED_SRNET),
        "--image-dir", str(image_dir),
        "--checkpoint", str(CHECKPOINT),
        "--device", "cpu",
        "--limit", str(limit),
        "--output", str(output_json),
    ]
    print(f"  [run]   {image_dir.name} -> {output_json.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"inference failed (rc={proc.returncode})")


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


def cleanup(paths: list[Path]) -> None:
    """Remove transient inputs after stego generation."""
    for p in paths:
        if p.exists():
            print(f"  [cleanup] rm -rf {p.relative_to(OUT_DIR)}")
            shutil.rmtree(p)


def main() -> int:
    if not NAMES_FILE.exists():
        raise SystemExit(f"missing {NAMES_FILE}; run prep step first")
    names_pgm = [ln.strip() for ln in NAMES_FILE.read_text().splitlines() if ln.strip()]
    print(f"[E13-SI] universal subset: {len(names_pgm)} covers")

    phasm_bin = find_phasm()
    print(f"[E13-SI] phasm: {phasm_bin}")

    # 1. PGM -> PNG
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pgm_tasks = [(PGM_DIR / n, PNG_DIR / n.replace(".pgm", ".png")) for n in names_pgm]
    pgm_present = [t for t in pgm_tasks if t[0].exists()]
    if len(pgm_present) < len(pgm_tasks):
        print(f"[E13-SI] WARN: only {len(pgm_present)}/{len(pgm_tasks)} PGMs available — "
              f"extract missing ones via `7z x BOSSbase_1.01.7z BOSSbase_1.01/<n>.pgm -o./pgms`")
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(pgm_to_png, t) for t in pgm_present]
        for f in futs:
            name, ok, status = f.result()
            if not ok:
                print(f"  PGM->PNG FAIL {name}: {status}")
    n_png = sum(1 for _ in PNG_DIR.glob("*.png"))
    print(f"[E13-SI] PNGs ready: {n_png}")

    # 2. Encode SI-UNIWARD Phasm Ghost at N=0..4
    for N in range(5):
        (STEGO_BASE / f"n{N}").mkdir(parents=True, exist_ok=True)
    enc_tasks = []
    for png in sorted(PNG_DIR.glob("*.png")):
        for N in range(5):
            enc_tasks.append((png, STEGO_BASE / f"n{N}" / png.name.replace(".png", ".jpg"),
                              N, phasm_bin))
    print(f"[E13-SI] encoding {len(enc_tasks)} stego pairs (workers={WORKERS})...")
    stats = {n: {"ok": 0, "skip": 0, "fail": 0} for n in range(5)}
    fails: list[str] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(encode_one, t) for t in enc_tasks]
        for f in as_completed(futs):
            name, n, ok, status = f.result()
            if not ok:
                stats[n]["fail"] += 1
                fails.append(f"  N={n} {name}: {status}")
            elif status == "skipped":
                stats[n]["skip"] += 1
            else:
                stats[n]["ok"] += 1
    for n in range(5):
        s = stats[n]
        print(f"[E13-SI] N={n}: ok={s['ok']} skip={s['skip']} fail={s['fail']}")
    if fails:
        print("  Failures (first 5):")
        for f in fails[:5]:
            print(f)

    # 3. Cleanup PGMs + PNGs before inference (per user instruction)
    cleanup([PGM_DIR.parent, PNG_DIR])

    # 4. Build cover subset (symlinks to BOSSbase QF75 covers matching our 40)
    COVER_SUBSET.mkdir(exist_ok=True)
    for name_pgm in names_pgm:
        name_jpg = name_pgm.replace(".pgm", ".jpg")
        src = COVER_SRC / name_jpg
        if not src.exists():
            print(f"  WARN: cover {src} missing")
            continue
        lnk = COVER_SUBSET / name_jpg
        if not lnk.exists():
            lnk.symlink_to(src)

    # 5. Inference: cover + 5 stego_si dirs
    print(f"[E13-SI] inference (CPU)...")
    cover_json = OUT_DIR / "cover_scores.json"
    scan_dir(COVER_SUBSET, cover_json)
    stego_jsons = {}
    for N in range(5):
        sj = OUT_DIR / f"stego_n{N}_si_scores.json"
        scan_dir(STEGO_BASE / f"n{N}", sj)
        stego_jsons[N] = sj

    def load(p: Path) -> dict[str, float]:
        j = json.loads(p.read_text())
        return {nm: r["stego_prob"] for nm, r in j["per_image"].items()}

    cov = load(cover_json)
    ste = {N: load(stego_jsons[N]) for N in range(5)}

    # Universal subset = names present in cover AND in all 5 stego_si dirs
    universal = set(cov.keys())
    for N in range(5):
        universal &= set(ste[N].keys())
    print(f"[E13-SI] universal subset after generation: n={len(universal)}")

    results = {}
    for N in range(5):
        cov_u = np.array([cov[nm] for nm in universal])
        ste_u = np.array([ste[N][nm] for nm in universal])
        names_n = set(ste[N].keys())
        cov_m = np.array([cov[nm] for nm in names_n if nm in cov])
        ste_m = np.array([ste[N][nm] for nm in names_n])
        results[N] = {
            "universal_subset": pe(cov_u, ste_u),
            "per_n_max":        pe(cov_m, ste_m),
        }

    out = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "cover_dir": str(COVER_SUBSET.relative_to(ROOT)),
        "stego_dirs": {N: str((STEGO_BASE / f"n{N}").relative_to(ROOT)) for N in range(5)},
        "primary_cost_map": "SI-UNIWARD (auto, PNG cover input)",
        "n_universal_subset": len(universal),
        "device": "cpu",
        "per_N": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "per_n_pe.json").write_text(json.dumps(out, indent=2))

    print()
    print(f"=== SI-UNIWARD PE per N (universal subset, n={len(universal)}) ===")
    print(f"  {'N':>2}  {'cover_mean':>10}  {'stego_mean':>10}  {'FA':>6}  {'MD':>6}  {'PE':>6}")
    for N in range(5):
        r = results[N]["universal_subset"]
        print(f"  {N:>2}  {r['cover_mean']:>10.4f}  {r['stego_mean']:>10.4f}  "
              f"{r['FA']:>6.3f}  {r['MD']:>6.3f}  {r['PE']:>6.3f}")
    print()
    print(f"=== SI-UNIWARD PE per N (per-N max) ===")
    print(f"  {'N':>2}  {'n_stego':>8}  {'cover_mean':>10}  {'stego_mean':>10}  {'PE':>6}")
    for N in range(5):
        r = results[N]["per_n_max"]
        print(f"  {N:>2}  {r['n_stego']:>8d}  {r['cover_mean']:>10.4f}  "
              f"{r['stego_mean']:>10.4f}  {r['PE']:>6.3f}")
    print()
    print(f"saved {OUT_DIR / 'per_n_pe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
