#!/bin/sh
# Run cargo with the phasm rav1e fork patched in, for builds with
# `av1-encoder`:
#
#   sh scripts/video-cargo.sh build --release --features h264-encoder,av1-encoder,av1-decoder
#
# Everything after the script name is passed to cargo unchanged.
#
# Why: phasm-core declares its AV1 encoder dependency as the crates.io
# package `rav1e` (`phasm-rav1e = { package = "rav1e", version = "=0.8.0" }`)
# so that an image-only build resolves from the registry index and fetches
# nothing. The stego hooks live only in the fork, github.com/cgaffga/phasm-rav1e,
# which is consumed from GitHub source and never published to crates.io. Video
# builds therefore patch `rav1e` to the fork at the commit below. The patch is
# passed on the command line rather than written into Cargo.toml because cargo
# resolves `[patch]` entries eagerly: in the manifest it would clone the fork
# for every build, image-only included.
#
# The Makefile's video targets and the CI workflows call this script; it is
# the ONLY place the pin is written for builds (BUILDING.md repeats it for
# people typing the commands by hand). sync-phasmcore.sh refuses to publish
# the mirror when PHASM_RAV1E_REV here (or in BUILDING.md) differs from the
# monorepo's vendor/phasm-rav1e gitlink, so the mirror builds exactly the
# fork commit the apps build.
#
# Overrides (environment):
#   PHASM_RAV1E_REV   another fork commit (full SHA), e.g. for fork-side CI
#   PHASM_RAV1E_GIT   another fork URL (e.g. a local file:// clone)
#   CARGO             the cargo binary (default: cargo)
#
# In the phasm monorepo the repo-root .cargo/config.toml already patches
# `rav1e` to the vendor/phasm-rav1e submodule checkout; there this script
# runs cargo unchanged, so local fork edits are what gets built.

set -eu

PHASM_RAV1E_DEFAULT_GIT="https://github.com/cgaffga/phasm-rav1e"
PHASM_RAV1E_DEFAULT_REV="abe19cd8b369a051bb08ca2c53aceabb920bc00c"

git_url=${PHASM_RAV1E_GIT:-$PHASM_RAV1E_DEFAULT_GIT}
rev=${PHASM_RAV1E_REV:-$PHASM_RAV1E_DEFAULT_REV}
cargo=${CARGO:-cargo}

root=$(cd "$(dirname "$0")/.." && pwd)
mono_config="$root/../.cargo/config.toml"
if [ -f "$mono_config" ] && grep -Eq '^[[:space:]]*rav1e[[:space:]]*=.*path[[:space:]]*=' "$mono_config"; then
    if [ -n "${PHASM_RAV1E_REV:-}${PHASM_RAV1E_GIT:-}" ]; then
        echo "video-cargo.sh: $mono_config patches rav1e to the submodule checkout;" >&2
        echo "  PHASM_RAV1E_REV / PHASM_RAV1E_GIT are ignored here (check out the commit in vendor/phasm-rav1e)." >&2
    fi
    exec "$cargo" "$@"
fi

case $rev in
    *[!0-9a-f]*|"") echo "video-cargo.sh: PHASM_RAV1E_REV must be a full commit SHA, got '$rev'" >&2; exit 2 ;;
esac
if [ ${#rev} -ne 40 ]; then
    echo "video-cargo.sh: PHASM_RAV1E_REV must be a full 40-character commit SHA, got '$rev'" >&2
    exit 2
fi

exec "$cargo" \
    --config "patch.crates-io.rav1e.git=\"$git_url\"" \
    --config "patch.crates-io.rav1e.rev=\"$rev\"" \
    "$@"
