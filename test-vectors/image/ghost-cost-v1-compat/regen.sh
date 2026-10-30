#!/bin/bash
# Regenerate the Ghost cost-v1 compatibility fixtures (see README.md here).
#
#   bash regen.sh <phasm-0.4.x-binary> [--large]
#
# <phasm-0.4.x-binary> must be a `phasm` CLI with the v1 J-UNIWARD costs.
# The committed fixtures come from the RELEASED phasm-core 0.4.3 (tag
# v0.4.3 = the phasmcore core/ tree). Its shadow verify predates the
# 2026-10 decoder-exact fix and still accepts tight cost pools: these are
# the shadows that exist in the wild.
#   mkdir -p "$PHASM_TMP/v043/core" && git archive v0.4.3 | tar -x -C "$PHASM_TMP/v043/core"
#   ln -s <a checkout's vendor/> "$PHASM_TMP/v043/vendor"
#   (cd "$PHASM_TMP/v043/core" && cargo build --release -p phasm-cli --bin phasm)
#
# Small fixtures are written next to this script (committed). With --large,
# the 12 MP / real-photo fixtures go to core/tests/real_photos/ghost-cost-v1-compat/
# (gitignored, like their sources); the test skips them when absent.
set -euo pipefail
OLD="$1"
LARGE="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CORE="$(cd "$HERE/../../.." && pwd)"
TV="$CORE/test-vectors/image"
PHASM_TMP="${PHASM_TMP:-/tmp}"
TMP="$PHASM_TMP/ghost-cost-v1-regen"
mkdir -p "$TMP"

# name | cover | primary passphrase | primary message | shadow pass:msg ...
enc() {
  local out="$1" cover="$2" ppass="$3" pmsg="$4"; shift 4
  local args=() i=2
  for s in "$@"; do
    args+=("--shadow-passphrase-$i" "${s%%:*}" "--shadow-message-$i" "${s#*:}")
    i=$((i + 1))
  done
  "$OLD" encode "$cover" --mode ghost -m "$pmsg" -p "$ppass" "${args[@]}" -o "$out" --quiet >/dev/null
}

# Incompressible-ish deterministic text: <n> hex chars from SHA-256 of <seed>.<i>.
hexmsg() { local i=0 out=""; while [ ${#out} -lt "$2" ]; do out="$out$(printf '%s.%d' "$1" $i | shasum -a 256 | cut -c1-64)"; i=$((i + 1)); done; printf '%s' "${out:0:$2}"; }

P1="The quick brown fox jumps over the lazy dog, primary layer one."
P2="Primary message for the cost-v1 compatibility fixture, a little longer than the shadows."

# A PNG source (SI / Deep Cover path), derived from a committed JPEG.
sips -s format png "$TV/photo_640x480_q75_420.jpg" --out "$TMP/photo_640x480.png" >/dev/null

enc "$HERE/v1_320x240_s1.jpg"    "$TV/photo_320x240_q75_420.jpg" pp-a "$P1" "sp-a1:first shadow"
enc "$HERE/v1_640x480_s2.jpg"    "$TV/photo_640x480_q75_420.jpg" pp-b "$P2" "sp-b1:shadow one, b" "sp-b2:shadow two"
enc "$HERE/v1_640x480_s4.jpg"    "$TV/photo_640x480_q75_420.jpg" pp-c "$P2" \
  "sp-c1:s1" "sp-c2:second shadow" "sp-c3:third shadow layer" "sp-c4:the fourth shadow, longest"
enc "$HERE/v1_640x480_si_s1.jpg" "$TMP/photo_640x480.png"        pp-d "$P1" "sp-d1:si shadow 1"
# "Tight" shadows: sized to nearly fill the cheapest 5 % cost pool, so the
# v1 and v2 pools differ in many of the positions the shadow uses and only
# the legacy costs recover it (exercises the decoder's fallback).
enc "$HERE/v1_640x480_tight_s1.jpg" "$TV/photo_640x480_q75_420.jpg" pp-t "$(hexmsg prim-t 1200)" "sp-t1:$(hexmsg tight-1 300)"
enc "$HERE/v1_640x480_si_tight_s2.jpg" "$TMP/photo_640x480.png" pp-u "$(hexmsg prim-u 1500)" "sp-u1:$(hexmsg tight-u1 280)" "sp-u2:short one"
enc "$HERE/v1_640x480_si_s3.jpg" "$TMP/photo_640x480.png"        pp-e "$P2" "sp-e1:si one" "sp-e2:si two" "sp-e3:si three!"

if [ "$LARGE" = "--large" ]; then
  L="$CORE/tests/real_photos/ghost-cost-v1-compat"
  mkdir -p "$L"
  BIG="$TV/photo_original_3557.jpg"
  RP="$CORE/tests/real_photos"
  enc "$L/v1_12mp_s1.jpg" "$BIG" pp-f "$P2" "sp-f1:12mp shadow"
  enc "$L/v1_12mp_s2.jpg" "$BIG" pp-g "$P2" "sp-g1:12mp one" "sp-g2:12mp two, b"
  enc "$L/v1_12mp_s4.jpg" "$BIG" pp-h "$P2" "sp-h1:a" "sp-h2:bb shadow" "sp-h3:ccc shadow three" "sp-h4:dddd shadow four"
  enc "$L/v1_whatsapp_s3.jpg" "$TV/progressive_whatsapp_1200x1600.jpg" pp-i "$P2" "sp-i1:wa one" "sp-i2:wa two" "sp-i3:wa three"
  enc "$L/v1_12mp_tight_s2.jpg" "$BIG" pp-l "$(hexmsg prim-l 30000)" "sp-l1:$(hexmsg tight-l1 18000)" "sp-l2:$(hexmsg tight-l2 2000)"
  enc "$L/v1_png_si_s2.jpg" "$RP/IMG_3315.PNG" pp-j "$P2" "sp-j1:png si one" "sp-j2:png si two"
  enc "$L/v1_png_si_s4.jpg" "$RP/91ZhFnr13LL.png" pp-k "$P2" "sp-k1:k1" "sp-k2:k2 shadow" "sp-k3:k3 shadow 3" "sp-k4:k4 shadow four"
fi

# Large-frame fixture (gitignored, tests/real_photos/ghost-cost-v1-compat/v1_12mp_bigframe.jpg):
#   phasm encode photo_original_3557.jpg --mode ghost -p pp-big -m "$(hexmsg bigframe 600000)"
# A primary frame over 256 KB, which 0.4.3 wrote but could not decode itself.
