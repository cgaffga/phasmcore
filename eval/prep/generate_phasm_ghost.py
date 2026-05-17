"""Generate Phasm Ghost stego pairs by shelling out to the `phasm` CLI.

Reads:  data/bossbase/jpeg_qf{Q}/*.jpg  (or any --cover-dir of JPEGs)
Writes: data/stego/phasm_ghost/qf{Q}/payload_{P}/*.jpg
        e.g. data/stego/phasm_ghost/qf75/payload_040/00001.jpg

For each cover, generates a random ASCII message sized to hit the requested
bpnzAC payload (computed from cover capacity). Embeds via:

    phasm encode <cover> --mode ghost -o <stego> -m <message> -p <pass>

Per-cover message is deterministic (sha256-derived from cover name + payload
+ seed_base). The passphrase is fixed across the whole run for reproducibility
— it does not affect detection statistics, only the encrypted payload bytes.

Idempotent: skips files whose output already exists.
Requires `phasm` in $PATH. Verify with `which phasm`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import string
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jpeglib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"

DEFAULT_PASSPHRASE = "phasm-eval-deterministic-2026"


def find_phasm_binary() -> str | None:
    """Resolve phasm binary path. Priority: env var > eval/bin/phasm-* > $PATH."""
    if (path := os.environ.get("PHASM_BIN")):
        return path if Path(path).is_file() else None
    if BIN_DIR.is_dir():
        candidates = sorted(
            BIN_DIR.glob("phasm-*"),
            key=lambda p: p.name,
            reverse=True,  # newest by name (we use SHA-DATE so date sorts last)
        )
        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return str(c)
    return shutil.which("phasm")


def derive_message(cover_name: str, payload: float, seed_base: int, n_bytes: int) -> str:
    """Reproducible printable-ASCII message of exactly n_bytes."""
    h = hashlib.sha256(f"{cover_name}|{payload:.4f}|{seed_base}".encode()).digest()
    # Expand hash to enough bytes; map each byte to printable ASCII.
    chars = string.ascii_letters + string.digits + string.punctuation.replace('"', '').replace("'", "")
    out = []
    counter = 0
    while len(out) < n_bytes:
        counter += 1
        h2 = hashlib.sha256(h + counter.to_bytes(4, "big")).digest()
        for b in h2:
            out.append(chars[b % len(chars)])
            if len(out) >= n_bytes:
                break
    return "".join(out)


def cover_capacity_ghost(cover_path: Path, phasm_bin: str) -> int:
    """Phasm Ghost capacity in bytes for a JPEG cover.

    Uses `phasm capacity <cover> --json` whose shape is
    {"ghost":N,"ghostDeepCover":N,"armor":N,"fortress":N,"shadow":N}.
    """
    try:
        out = subprocess.run(
            [phasm_bin, "capacity", str(cover_path), "--json"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        return int(json.loads(out)["ghost"])
    except (subprocess.SubprocessError, FileNotFoundError, KeyError, ValueError):
        pass
    # Fallback heuristic if CLI output shape changes: use nzAC * 0.5 bpnzAC / 8
    dct = jpeglib.read_dct(str(cover_path))
    nzac = int((dct.Y != 0).sum()) - int((dct.Y[..., 0, 0] != 0).sum())
    return max(8, nzac // 16)


def n_bytes_for_payload(cover_path: Path, payload_bpnzAC: float) -> int:
    """Convert bpnzAC payload to bytes for this cover."""
    dct = jpeglib.read_dct(str(cover_path))
    nzac = int((dct.Y != 0).sum()) - int((dct.Y[..., 0, 0] != 0).sum())
    bits = nzac * payload_bpnzAC
    return max(8, int(bits // 8))


def encode_one(args: tuple[Path, Path, float, int, str, str]) -> tuple[str, bool, str]:
    src, dst, payload, seed_base, passphrase, phasm_bin = args
    if dst.exists():
        return src.name, True, "skipped"
    try:
        n_bytes = n_bytes_for_payload(src, payload)
        message = derive_message(src.name, payload, seed_base, n_bytes)
        result = subprocess.run(
            [
                phasm_bin, "encode",
                str(src),
                "--mode", "ghost",
                "-o", str(dst),
                "-m", message,
                "-p", passphrase,
                "--quiet",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return src.name, False, f"phasm exit {result.returncode}: {result.stderr[:200]}"
        if not dst.exists():
            return src.name, False, "phasm produced no output file"
        return src.name, True, f"ok ({n_bytes}B msg)"
    except subprocess.TimeoutExpired:
        return src.name, False, "timeout"
    except Exception as exc:
        return src.name, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cover-dir", type=Path, default=None,
                   help="Cover JPEG directory (default: data/bossbase/jpeg_qf{quality})")
    p.add_argument("--quality", type=int, default=75)
    p.add_argument("--payload", type=float, nargs="+", default=[0.1, 0.4],
                   help="bpnzAC payload rates (default: 0.1 0.4)")
    p.add_argument("--passphrase", type=str, default=DEFAULT_PASSPHRASE)
    p.add_argument("--seed-base", type=int, default=2026)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = os.cpu_count(); reduce if phasm contends for memory")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all images; useful for smoke testing (e.g. --limit 5)")
    args = p.parse_args()

    phasm_bin = find_phasm_binary()
    if phasm_bin is None:
        print(
            "[generate_phasm_ghost] `phasm` binary not found.\n"
            "  Looked in: $PHASM_BIN, eval/bin/phasm-*, $PATH\n"
            "  Build via: cd ~/Development/phasm/core && \\\n"
            "    CARGO_TARGET_DIR=/tmp/phasm-build cargo build --release -p phasm-cli\n"
            "  Then: cp /tmp/phasm-build/release/phasm <eval-bin>/phasm-<sha>-<date>",
            file=sys.stderr,
        )
        return 2
    print(f"[generate_phasm_ghost] using phasm: {phasm_bin}", flush=True)

    cover_dir = args.cover_dir or (ROOT / "data" / "bossbase" / f"jpeg_qf{args.quality}")
    if not cover_dir.exists():
        print(f"[generate_phasm_ghost] {cover_dir} not found", file=sys.stderr)
        return 1

    sources = sorted(cover_dir.glob("*.jpg"))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"[generate_phasm_ghost] no .jpg files in {cover_dir}", file=sys.stderr)
        return 1

    print(
        f"[generate_phasm_ghost] {len(sources)} covers, payloads={args.payload}, "
        f"seed_base={args.seed_base}, passphrase=<{len(args.passphrase)} chars>",
        flush=True,
    )

    overall_ok = overall_skip = overall_fail = 0
    workers = args.workers or None

    for payload in args.payload:
        out_dir = (
            ROOT / "data" / "stego" / "phasm_ghost"
            / f"qf{args.quality}"
            / f"payload_{int(round(payload * 100)):03d}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = [
            (src, out_dir / src.name, payload, args.seed_base, args.passphrase, phasm_bin)
            for src in sources
        ]
        n_ok = n_skip = n_fail = 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(encode_one, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"phasm-ghost @ {payload}"):
                name, ok, status = fut.result()
                if not ok:
                    tqdm.write(f"  FAIL {name}: {status}")
                    n_fail += 1
                elif status == "skipped":
                    n_skip += 1
                else:
                    n_ok += 1

        print(
            f"[generate_phasm_ghost] payload={payload}: "
            f"{n_ok} ok, {n_skip} skipped, {n_fail} failed -> {out_dir}",
            flush=True,
        )
        overall_ok += n_ok
        overall_skip += n_skip
        overall_fail += n_fail

    print(
        f"[generate_phasm_ghost] DONE: {overall_ok} embedded, "
        f"{overall_skip} skipped, {overall_fail} failed",
        flush=True,
    )
    return 0 if overall_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
