"""Parallel download of ALASKA-2 Cover/ files from Kaggle competition API.

Uses the Kaggle Python API directly (reusing a single gRPC client across all
downloads). Bundled Cover.zip endpoint returns 404; individual files work.

Reads:  ~/.kaggle/kaggle.json (or eval/.secrets/kaggle.json)
Writes: data/alaska2_full/Cover/*.jpg

Idempotent: skips files already on disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "alaska2_full" / "Cover"
COMPETITION = "alaska2-image-steganalysis"


def list_cover_files(api: KaggleApi, limit: int) -> list[str]:
    files: list[str] = []
    token = None
    while len(files) < limit:
        result = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        page_files = result.files if hasattr(result, "files") else result
        cover_files = [f.name for f in page_files if f.name.startswith("Cover/")]
        for name in cover_files:
            if len(files) < limit:
                files.append(name)
        token = getattr(result, "next_page_token", None)
        if not token:
            break
        if not cover_files and files:
            break
    return files[:limit]


def download_one(args: tuple[KaggleApi, str, Path]) -> tuple[str, bool, str]:
    api, fname, out_dir = args
    dst = out_dir / Path(fname).name
    if dst.exists() and dst.stat().st_size > 0:
        return fname, True, "skipped"
    try:
        api.competition_download_file(
            COMPETITION, fname, path=str(out_dir), quiet=True, force=True
        )
        if dst.exists() and dst.stat().st_size > 0:
            return fname, True, "ok"
        return fname, False, "file missing after dl"
    except Exception as e:
        msg = str(e)
        if "404" in msg or "429" in msg or "Not Found" in msg:
            return fname, False, "RATE_LIMITED"
        return fname, False, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, default=10000, help="how many Cover/*.jpg to fetch")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads per chunk")
    p.add_argument("--chunk", type=int, default=50, help="files per chunk before rate-check")
    p.add_argument("--between-chunks", type=float, default=2.0, help="sleep between healthy chunks (s)")
    p.add_argument("--backoff-initial", type=float, default=30.0, help="initial backoff on rate-limit (s)")
    p.add_argument("--backoff-max", type=float, default=600.0, help="max backoff (s)")
    p.add_argument("--max-minutes", type=int, default=0, help="0 = run until done; else stop after N min")
    args = p.parse_args()

    api = KaggleApi()
    api.authenticate()

    print(f"[download_alaska] listing first {args.limit} Cover/ files...")
    t0 = time.time()
    files = list_cover_files(api, args.limit)
    print(f"[download_alaska] {len(files)} files listed in {time.time()-t0:.1f}s")
    if len(files) < args.limit:
        print(f"  WARN: only got {len(files)} (asked for {args.limit})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Adaptive throttled loop: process in chunks, back off on rate-limit.
    # Kaggle's competition file-download endpoint rate-limits per session;
    # repeated 404s usually mean "wait, then try again". We keep retrying
    # the unfinished list with exponential backoff until everything is done
    # or hard-failures dominate.
    n_done = n_ok = n_skip = n_fail = 0
    backoff = float(args.backoff_initial)
    t_start = time.time()
    round_idx = 0
    remaining = list(files)

    while remaining:
        round_idx += 1
        chunk = remaining[: args.chunk]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(download_one, (api, f, OUT_DIR)): f for f in chunk}
            chunk_ok = chunk_rate = chunk_skip = chunk_fail = 0
            still_failing = []
            for fut in as_completed(futures):
                name, ok, status = fut.result()
                if ok and status == "skipped":
                    chunk_skip += 1
                    n_skip += 1
                elif ok:
                    chunk_ok += 1
                    n_ok += 1
                elif status == "RATE_LIMITED":
                    chunk_rate += 1
                    still_failing.append(name)
                else:
                    chunk_fail += 1
                    n_fail += 1
                    print(f"  HARD-FAIL {name}: {status}", file=sys.stderr)
        n_done += chunk_ok + chunk_skip
        rate_global = n_done / max(time.time() - t_start, 1e-3)
        print(
            f"[round {round_idx}] chunk={len(chunk)} ok={chunk_ok} skip={chunk_skip} "
            f"rate-limited={chunk_rate} hard-fail={chunk_fail} | "
            f"global ok+skip={n_done} hard-fail={n_fail} rate={rate_global:.2f}/s "
            f"(elapsed {(time.time()-t_start)/60:.1f}min)",
            flush=True,
        )
        # Drop successes from remaining
        remaining = still_failing + remaining[args.chunk :]

        # Throttle based on rate-limit fraction in this chunk
        if chunk_rate == 0:
            backoff = max(args.backoff_initial, backoff * 0.7)
            time.sleep(args.between_chunks)
        else:
            # Aggressive backoff when rate-limited
            rate_frac = chunk_rate / max(len(chunk), 1)
            if rate_frac > 0.5:
                print(f"  >50% rate-limited; backing off {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, args.backoff_max)
            else:
                time.sleep(args.between_chunks * 2)

        if args.max_minutes > 0 and (time.time() - t_start) / 60 > args.max_minutes:
            print(f"[download_alaska] hit max_minutes={args.max_minutes}; stopping early")
            break

    print(f"[download_alaska] DONE in {(time.time()-t_start)/60:.1f}min: "
          f"{n_ok} downloaded, {n_skip} skipped, {n_fail} hard-failed, "
          f"{len(remaining)} unfinished -> {OUT_DIR}")
    return 0 if n_fail == 0 and not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
