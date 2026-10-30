#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026, Christoph Gaffga
#
# Report the build tools phasm-core needs, and how to install the missing
# ones on this OS. Read-only: it never installs anything.
#
#   scripts/check-deps.sh            # everything (image + video features)
#   scripts/check-deps.sh base       # image stego only (default features)
#   scripts/check-deps.sh video      # what --features h264-encoder,av1-encoder needs
#
# Exit status: 0 when every REQUIRED tool for the selected set is present,
# 1 otherwise. Optional tools (the CLI's video runtime helpers) only warn.
#
# Which feature needs what (see BUILDING.md for the full matrix):
#   default / parallel / video / h264-decoder   Rust only
#   h264-encoder   OpenH264 fork: meson, ninja, C + C++ compiler,
#                  nasm on x86/x86_64
#   av1-decoder    dav1d fork: meson, ninja, C compiler, nasm >= 2.14 on x86/x86_64
#   av1-encoder    rav1e fork (its `asm` feature): nasm on x86_64,
#                  C compiler on aarch64; implies av1-decoder. cargo fetches
#                  the fork from github.com (scripts/video-cargo.sh), so it
#                  is not a submodule.
#   h264-encoder / av1-decoder also need the vendor/ git submodules
#   (OpenH264, dav1d) checked out.

set -u

MODE="${1:-all}"
case "$MODE" in
    all|base|video) ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: $0 [all|base|video]" >&2; exit 2 ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)

# ── platform ──────────────────────────────────────────────────────────
OS=$(uname -s 2>/dev/null || echo unknown)
ARCH=$(uname -m 2>/dev/null || echo unknown)
DISTRO=""
case "$OS" in
    Linux)
        if grep -qi microsoft /proc/version 2>/dev/null; then WSL=1; else WSL=0; fi
        if [ -r /etc/os-release ]; then
            # shellcheck disable=SC1091
            DISTRO=$(. /etc/os-release; echo "${ID:-} ${ID_LIKE:-}")
        fi
        case " $DISTRO " in
            *" debian "*|*" ubuntu "*) PLATFORM=debian ;;
            *" fedora "*|*" rhel "*|*" centos "*) PLATFORM=fedora ;;
            *" arch "*) PLATFORM=arch ;;
            *) PLATFORM=linux ;;
        esac
        ;;
    Darwin) PLATFORM=macos; WSL=0 ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM=windows; WSL=0 ;;
    *) PLATFORM=unknown; WSL=0 ;;
esac
case "$ARCH" in
    x86_64|amd64|i386|i686|x86) X86=1 ;;
    *) X86=0 ;;
esac

hint() {  # hint <tool>
    case "$PLATFORM:$1" in
        macos:cc|macos:c++) echo "xcode-select --install" ;;
        macos:ffmpeg)       echo "brew install ffmpeg" ;;
        macos:*)            echo "brew install $1" ;;
        debian:cc|debian:c++) echo "sudo apt install build-essential" ;;
        debian:ninja)       echo "sudo apt install ninja-build" ;;
        debian:*)           echo "sudo apt install $1" ;;
        fedora:cc)          echo "sudo dnf install gcc" ;;
        fedora:c++)         echo "sudo dnf install gcc-c++" ;;
        fedora:ninja)       echo "sudo dnf install ninja-build" ;;
        fedora:ffmpeg)      echo "sudo dnf install ffmpeg-free  (or ffmpeg from RPM Fusion)" ;;
        fedora:*)           echo "sudo dnf install $1" ;;
        arch:cc|arch:c++)   echo "sudo pacman -S base-devel" ;;
        arch:*)             echo "sudo pacman -S $1" ;;
        windows:cc|windows:c++) echo "Visual Studio Build Tools, workload 'Desktop development with C++' (or use WSL2)" ;;
        windows:meson|windows:ninja) echo "pip install meson ninja  (or winget install mesonbuild.meson)" ;;
        windows:nasm)       echo "winget install NASM.NASM  (https://nasm.us)" ;;
        windows:ffmpeg)     echo "winget install Gyan.FFmpeg" ;;
        *:cc|*:c++)         echo "install a C/C++ toolchain (gcc/g++ or clang) with your package manager" ;;
        *:ninja)            echo "install ninja (often packaged as ninja-build)" ;;
        *)                  echo "install $1 with your package manager" ;;
    esac
}

MISSING=0
WARNED=0
row() {  # row <status> <tool> <detail>
    printf '  %-8s %-15s %s\n' "$1" "$2" "$3"
}
have() { command -v "$1" >/dev/null 2>&1; }
first_of() {  # first_of <candidates...> → prints the first found
    for c in "$@"; do if have "$c"; then echo "$c"; return 0; fi; done
    return 1
}
require() {  # require <label> <why> <candidates...>
    label=$1; why=$2; shift 2
    if found=$(first_of "$@"); then
        ver=$("$found" --version 2>/dev/null | head -n 1)
        [ -z "$ver" ] && ver=$("$found" -v 2>/dev/null | head -n 1)
        row ok "$label" "$found ${ver:+($ver)}"
    else
        row MISSING "$label" "needed for $why → $(hint "$label")"
        MISSING=$((MISSING + 1))
    fi
}
optional() {  # optional <label> <why> <candidates...>
    label=$1; why=$2; shift 2
    if found=$(first_of "$@"); then
        row ok "$label" "$found (optional: $why)"
    else
        row warn "$label" "optional, $why → $(hint "$label")"
        WARNED=$((WARNED + 1))
    fi
}

WSL_TAG=""; [ "$WSL" = 1 ] && WSL_TAG=", WSL"
echo "phasm-core build dependencies ($OS $ARCH${DISTRO:+, $DISTRO}$WSL_TAG)"
echo

# ── base: Rust ────────────────────────────────────────────────────────
echo "Rust (all builds):"
if have cargo && have rustc; then
    row ok cargo "$(cargo --version 2>/dev/null)"
    if [ -f "$ROOT/rust-toolchain.toml" ]; then
        pin=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$ROOT/rust-toolchain.toml")
        if have rustup; then
            row ok rustup "toolchain $pin is pinned in rust-toolchain.toml; rustup fetches it on first build"
        else
            row warn rustup "not found; rust-toolchain.toml pins $pin, install it by hand or use rustup"
            WARNED=$((WARNED + 1))
        fi
    fi
else
    row MISSING cargo "install Rust via https://rustup.rs"
    MISSING=$((MISSING + 1))
fi
have git || { row MISSING git "needed to fetch the vendor/ submodules → $(hint git)"; MISSING=$((MISSING + 1)); }

if [ "$MODE" = base ]; then
    echo
    [ "$MISSING" -eq 0 ] && echo "OK: everything for the default (image) build is present." \
        || echo "$MISSING required tool(s) missing."
    [ "$MISSING" -eq 0 ]; exit $?
fi

# ── video: C toolchain + meson/ninja/nasm ─────────────────────────────
echo
echo "Video features (h264-encoder, av1-decoder, av1-encoder):"
if [ "$PLATFORM" = windows ]; then
    if have cl; then row ok cc "cl (MSVC)"; else require cc "the C/C++ shims and vendored C" cc gcc clang; fi
else
    require cc  "the vendored C (OpenH264, dav1d) and the shims" cc gcc clang
    require c++ "the OpenH264 C++ shim (h264-encoder)" c++ g++ clang++
fi
require meson "building OpenH264 (h264-encoder) and dav1d (av1-decoder)" meson
require ninja "building OpenH264 (h264-encoder) and dav1d (av1-decoder)" ninja ninja-build
if [ "$X86" = 1 ]; then
    require nasm "x86 assembly in OpenH264, dav1d (>= 2.14) and rav1e asm" nasm
else
    row skip nasm "not needed on $ARCH"
fi

# Submodules: vendor/ sits at the repo root of the phasmcore mirror, or one
# level above core/ in the monorepo.
echo
echo "Vendored sources (git submodules):"
VENDOR=""
for v in "$ROOT/vendor" "$ROOT/../vendor"; do
    if [ -d "$v/phasm-openh264" ] || [ -d "$v/phasm-dav1d" ]; then
        VENDOR=$v; break
    fi
done
for sub in phasm-openh264 phasm-dav1d; do
    if [ -n "$VENDOR" ] && [ -f "$VENDOR/$sub/meson.build" ]; then
        sha=$(git -C "$VENDOR/$sub" rev-parse --short HEAD 2>/dev/null || echo "?")
        row ok "$sub" "vendor/$sub @ $sha"
    else
        row MISSING "$sub" "run: git submodule update --init --recursive"
        MISSING=$((MISSING + 1))
    fi
done

echo
echo "CLI video runtime (phasm video-encode / video-capacity, not needed to build):"
optional ffmpeg  "the CLI demuxes and decodes the cover video with it" ffmpeg
optional ffprobe "the CLI probes the cover video with it" ffprobe

echo
if [ "$PLATFORM" = windows ]; then
    echo "Note: native Windows builds of the video features are untested; WSL2"
    echo "      (Ubuntu) is the supported route. Image-only builds work natively."
    echo
fi
if [ "$MISSING" -eq 0 ]; then
    if [ "$WARNED" -gt 0 ]; then
        echo "OK: everything required is present ($WARNED optional warning(s))."
    else
        echo "OK: everything required is present."
    fi
    exit 0
fi
echo "$MISSING required item(s) missing; see the hints above and BUILDING.md."
exit 1
