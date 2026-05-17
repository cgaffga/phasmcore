"""Generate J-UNIWARD reference stego pairs from BOSSbase JPEG covers.

Uses `conseal` (pure Python with optional Rust backend) — no Octave required.
We embed at each requested bpnzAC rate using the Butora 2023 off-by-one-fixed
implementation by default (`JUNIWARD_FIX_OFF_BY_ONE`); pass `--implementation
original` to use the canonical MATLAB-equivalent buggy version that a decade
of published numbers were computed against. We record the choice in the per-
run config so the calibration analysis can compare apples to apples.

Reads:  data/bossbase/jpeg_qf{Q}/*.jpg
Writes: data/stego/juniward/qf{Q}/payload_{P}/*.jpg
        e.g. data/stego/juniward/qf75/payload_010/00001.jpg

Payload P encoding: bpnzAC * 100, zero-padded to 3 digits. So 0.1 -> 010,
0.4 -> 040, 0.05 -> 005. Keeps directory listings sortable.

Per-image seed is derived deterministically from the cover filename so the
same cover + payload + implementation + seed_base produces byte-identical
stego. Reproducibility contract.

Idempotent: skips files whose output already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import conseal as cl
import jpeglib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

IMPLEMENTATIONS = {
    "original": cl.JUNIWARD_ORIGINAL,
    "fix-off-by-one": cl.JUNIWARD_FIX_OFF_BY_ONE,
}


def derive_seed(cover_name: str, payload: float, seed_base: int) -> int:
    """Reproducible per-image seed. Same cover + payload + base = same seed."""
    h = hashlib.sha256(f"{cover_name}|{payload:.4f}|{seed_base}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def embed_one(args: tuple[Path, Path, float, str, int]) -> tuple[str, bool, str]:
    src, dst, payload, impl_name, seed_base = args
    if dst.exists():
        return src.name, True, "skipped"
    try:
        # Read DCT + spatial — conseal needs both
        cover_dct = jpeglib.read_dct(str(src))
        cover_spatial = jpeglib.read_spatial(str(src)).spatial[..., 0]

        impl = IMPLEMENTATIONS[impl_name]
        seed = derive_seed(src.name, payload, seed_base)

        stego_y = cl.juniward.simulate_single_channel(
            x0=cover_spatial,
            y0=cover_dct.Y,
            qt=cover_dct.qt[0],
            alpha=payload,
            implementation=impl,
            seed=seed,
        )

        # Write back as JPEG with same Huffman tables / quant tables
        stego_jpeg = jpeglib.read_dct(str(src))
        stego_jpeg.Y = stego_y
        stego_jpeg.write_dct(str(dst))
        return src.name, True, "ok"
    except Exception as exc:
        return src.name, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quality", type=int, default=75, help="JPEG QF of the cover set")
    p.add_argument(
        "--payload",
        type=float,
        nargs="+",
        default=[0.1, 0.4],
        help="bpnzAC payload rates (default: 0.1 0.4)",
    )
    p.add_argument(
        "--implementation",
        choices=list(IMPLEMENTATIONS.keys()),
        default="fix-off-by-one",
        help="J-UNIWARD variant. 'original' matches the buggy Binghamton MATLAB; "
             "'fix-off-by-one' incorporates the Butora 2023 correction.",
    )
    p.add_argument("--seed-base", type=int, default=2026, help="Reproducibility base seed")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = os.cpu_count(); reduce if memory is tight (J-UNIWARD cost map is ~20 MB per image)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all images; useful for smoke testing (e.g. --limit 10)",
    )
    args = p.parse_args()

    src_dir = ROOT / "data" / "bossbase" / f"jpeg_qf{args.quality}"
    if not src_dir.exists():
        print(f"[generate_juniward] {src_dir} not found — run prep/encode_jpeg.py first", file=sys.stderr)
        return 1

    sources = sorted(src_dir.glob("*.jpg"))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"[generate_juniward] no .jpg files in {src_dir}", file=sys.stderr)
        return 1

    print(
        f"[generate_juniward] {len(sources)} covers, payloads={args.payload}, "
        f"implementation={args.implementation}, seed_base={args.seed_base}",
        flush=True,
    )

    overall_ok = 0
    overall_skip = 0
    overall_fail = 0
    workers = args.workers or None

    for payload in args.payload:
        out_dir = (
            ROOT / "data" / "stego" / "juniward" / f"qf{args.quality}"
            / f"payload_{int(round(payload * 100)):03d}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = [
            (src, out_dir / src.name, payload, args.implementation, args.seed_base)
            for src in sources
        ]
        n_ok = n_skip = n_fail = 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(embed_one, t) for t in tasks]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"payload {payload}",
            ):
                name, ok, status = fut.result()
                if not ok:
                    tqdm.write(f"  FAIL {name}: {status}")
                    n_fail += 1
                elif status == "skipped":
                    n_skip += 1
                else:
                    n_ok += 1

        print(
            f"[generate_juniward] payload={payload}: "
            f"{n_ok} ok, {n_skip} skipped, {n_fail} failed -> {out_dir}",
            flush=True,
        )
        overall_ok += n_ok
        overall_skip += n_skip
        overall_fail += n_fail

    print(
        f"[generate_juniward] DONE: {overall_ok} embedded, "
        f"{overall_skip} skipped, {overall_fail} failed",
        flush=True,
    )
    return 0 if overall_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
