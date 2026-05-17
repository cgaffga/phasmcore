# Phasm CLI binaries

Local-only `phasm` builds used by `prep/generate_phasm_ghost.py`. NOT installed
into `$PATH` (per user preference). Wrapper picks up the newest filename in
this directory automatically (sorts by name; SHA-DATE convention sorts dates last).

## Manifest

| Filename | Phasm version | Parent SHA | Build date | Built from | Notes |
|---|---|---|---|---|---|
| `phasm-16e76521934d-20260510` | 0.2.6 | `16e7652...` (parent main) | 2026-05-10 | `cd core && CARGO_TARGET_DIR=/tmp/phasm-eval-build cargo build --release -p phasm-cli` | First Path 1a evaluation binary |

## How to add a new binary

```bash
cd ~/Development/phasm
SHA=$(git rev-parse --short=12 HEAD)
DATE=$(date +%Y%m%d)
cd core
CARGO_TARGET_DIR=/tmp/phasm-eval-build cargo build --release -p phasm-cli
cp /tmp/phasm-eval-build/release/phasm "$OLDPWD/eval/bin/phasm-${SHA}-${DATE}"
chmod +x "$OLDPWD/eval/bin/phasm-${SHA}-${DATE}"
# Then add a row to the manifest above and commit only this file
```

Use `CARGO_TARGET_DIR=/tmp/phasm-eval-build` to avoid touching the user's
active `core/target/` cache (the rust h264 work in flight there is unrelated
and we shouldn't churn its incremental-compile state).

## Pinning a specific binary in a run

The wrapper resolves binary path in this order:
1. `$PHASM_BIN` (env var)
2. Newest `eval/bin/phasm-*` (alphabetical sort)
3. `which phasm`

To pin an older binary for a re-run: `PHASM_BIN=eval/bin/phasm-<old-sha>-<date> uv run python prep/generate_phasm_ghost.py ...`
