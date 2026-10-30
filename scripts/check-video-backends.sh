#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026, Christoph Gaffga
#
# Fail if a video build linked a fail-fast STUB instead of the real codec.
#
# Without the vendor/ submodules, core-openh264-sys and core-dav1d-sys do not
# fail: they print a cargo:warning, compile a stub whose every call returns an
# error, and set `cfg(phasm_openh264_stub)` / `cfg(phasm_dav1d_stub)`. The
# build then succeeds and the binary fails at runtime (AV1 encode too, since
# it verifies through dav1d). The warning is easy to miss and is not
# replayed on a cached build, so this script reads the build-script output
# cargo keeps instead.
#
#   scripts/check-video-backends.sh [--profile release|debug] [h264] [av1]
#
# With no codec named it checks both. Run it after a build with the matching
# features (`make build-video` does). Exit 0 when every named backend was
# built for real, 1 when one is a stub or was not built at all.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROFILE=release
WANT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE=$2; shift 2 ;;
        h264|av1) WANT="$WANT $1"; shift ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--profile release|debug] [h264] [av1]" >&2; exit 2 ;;
    esac
done
[ -z "$WANT" ] && WANT="h264 av1"

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
STATUS=0

check() {  # check <label> <crate-dir-prefix> <stub-cfg>
    label=$1; crate=$2; cfg=$3
    # Newest build-script output for this crate under the profile dir
    # (host builds: target/<profile>, cross builds: target/<triple>/<profile>).
    newest=$(ls -t "$TARGET_DIR"/"$PROFILE"/build/"$crate"-*/output \
                   "$TARGET_DIR"/*/"$PROFILE"/build/"$crate"-*/output 2>/dev/null | head -n 1)
    if [ -z "$newest" ]; then
        echo "  FAIL  $label: $crate was not built under $TARGET_DIR/$PROFILE"
        STATUS=1
    elif grep -q "rustc-cfg=$cfg" "$newest"; then
        echo "  FAIL  $label: $crate built the fail-fast STUB ($newest)"
        echo "        fetch the sources: git submodule update --init --recursive"
        echo "        then rebuild (cargo re-runs the build script by itself)."
        STATUS=1
    else
        echo "  ok    $label: $crate linked the real codec"
    fi
}

echo "video backends ($PROFILE):"
for w in $WANT; do
    case "$w" in
        h264) check "H.264 encoder (OpenH264)" core-openh264-sys phasm_openh264_stub ;;
        av1)  check "AV1 decoder (dav1d)" core-dav1d-sys phasm_dav1d_stub ;;
    esac
done
exit $STATUS
