"""Generate paired Phasm-Ghost-with-shadows stegos for Phase 4e color.

Reads:  data/alaska2/cover/*.jpg
Writes: data/path4e_color/n1/*.jpg  (primary + 1 shadow)
        data/path4e_color/n2/*.jpg  (primary + 2 shadows)

Each cover is encoded TWICE: once with n=1 shadow, once with n=2 shadows.
Pairs that fail capacity for either condition are skipped (caller filters via
filename intersection later, same convention as Phase 4c).

Per-cover messages are deterministic (sha256-derived from cover name + slot + seed).
Each passphrase is fixed across the whole run — passphrases do not affect
detectability statistics, only ciphertext bytes.

Usage:
  uv run python prep/generate_phasm_shadows.py
  uv run python prep/generate_phasm_shadows.py --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import string
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"

PRIMARY_PASS = "phasm-eval-primary-2026"
SHADOW_PASSES = [
    "phasm-eval-shadow1-2026",
    "phasm-eval-shadow2-2026",
]
MSG_BYTES = 64  # short text — matches Phasm spec "<1KB short messages"


def find_phasm_binary() -> str | None:
    """Resolve phasm binary path. Priority: eval/bin/phasm-* > $PATH."""
    if BIN_DIR.is_dir():
        candidates = sorted(BIN_DIR.glob("phasm-*"), key=lambda p: p.name, reverse=True)
        for c in candidates:
            if c.is_file():
                return str(c)
    import shutil
    return shutil.which("phasm")


def derive_message(cover_name: str, slot: str, seed: int, n_bytes: int = MSG_BYTES) -> str:
    """Reproducible printable-ASCII message."""
    h = hashlib.sha256(f"{cover_name}|{slot}|{seed}".encode()).digest()
    chars = string.ascii_letters + string.digits
    out: list[str] = []
    counter = 0
    while len(out) < n_bytes:
        counter += 1
        h2 = hashlib.sha256(h + counter.to_bytes(4, "big")).digest()
        for b in h2:
            out.append(chars[b % len(chars)])
            if len(out) >= n_bytes:
                break
    return "".join(out)


def encode_one(args: tuple[Path, Path, int, int, str]) -> tuple[str, int, bool, str]:
    """Encode one (cover, n_shadows) pair.

    n_shadows=1: primary + 1 shadow message.
    n_shadows=2: primary + 2 shadow messages.
    """
    src, dst, n_shadows, seed, phasm_bin = args
    if dst.exists():
        return src.name, n_shadows, True, "skipped"

    primary_msg = derive_message(src.name, "primary", seed)
    cmd = [
        phasm_bin, "encode", str(src),
        "--mode", "ghost",
        "-o", str(dst),
        "-m", primary_msg,
        "-p", PRIMARY_PASS,
        "--quiet",
    ]
    for i in range(n_shadows):
        slot = f"shadow{i+1}"
        msg = derive_message(src.name, slot, seed)
        cmd += [f"--m{i+2}", msg, f"--p{i+2}", SHADOW_PASSES[i]]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return src.name, n_shadows, False, "timeout"
    except Exception as exc:
        return src.name, n_shadows, False, f"{type(exc).__name__}: {exc}"

    if result.returncode != 0:
        return src.name, n_shadows, False, f"phasm exit {result.returncode}: {result.stderr[:200]}"
    if not dst.exists():
        return src.name, n_shadows, False, "phasm produced no output"
    return src.name, n_shadows, True, "ok"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cover-dir", type=Path,
                   default=ROOT / "data" / "alaska2" / "cover")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "data" / "path4e_color")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all covers (smoke testing: --limit 5)")
    args = p.parse_args()

    phasm_bin = find_phasm_binary()
    if phasm_bin is None:
        print("[gen] phasm binary not found in eval/bin/ or $PATH", file=sys.stderr)
        return 2
    print(f"[gen] using phasm: {phasm_bin}", flush=True)

    covers = sorted(args.cover_dir.glob("*.jpg"))
    if args.limit:
        covers = covers[: args.limit]
    if not covers:
        print(f"[gen] no covers in {args.cover_dir}", file=sys.stderr)
        return 1
    print(f"[gen] {len(covers)} covers, seed={args.seed}, workers={args.workers}")

    out_n1 = args.out_dir / "n1"
    out_n2 = args.out_dir / "n2"
    out_n1.mkdir(parents=True, exist_ok=True)
    out_n2.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path, int, int, str]] = []
    for c in covers:
        tasks.append((c, out_n1 / c.name, 1, args.seed, phasm_bin))
        tasks.append((c, out_n2 / c.name, 2, args.seed, phasm_bin))

    stats = {1: {"ok": 0, "skip": 0, "fail": 0},
             2: {"ok": 0, "skip": 0, "fail": 0}}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(encode_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="encode"):
            name, n, ok, status = fut.result()
            if not ok:
                stats[n]["fail"] += 1
                tqdm.write(f"  FAIL {name} n={n}: {status}")
            elif status == "skipped":
                stats[n]["skip"] += 1
            else:
                stats[n]["ok"] += 1

    for n in (1, 2):
        s = stats[n]
        print(f"[gen] n={n}: ok={s['ok']} skipped={s['skip']} failed={s['fail']}", flush=True)

    fail_total = stats[1]["fail"] + stats[2]["fail"]
    return 0 if fail_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
