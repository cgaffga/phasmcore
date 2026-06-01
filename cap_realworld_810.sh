#!/bin/bash
# #810 — real-world capacity data collection driver.
# Fans out one process per corpus video at its NATIVE 16-aligned
# resolution and FULL length, concurrency 8 (the OH264 encoder has
# C-global state → cross-process is the only safe parallelism).
# Each process emits one `RESULT ...` line (reported vs true max + ratio).
set -u
DIR=/Users/cgaffga/Development/phasm/core/test-vectors/video/h264/real-world/source
BIN=$(ls -t /Users/cgaffga/Development/phasm/core/target/release/deps/cap_realworld_810-* 2>/dev/null | grep -v '\.d$' | head -1)
OUT=/tmp/cap810_results.txt
GOP=30
: > "$OUT"

PAIRS=(
  "iphone5:iphone5_1080p_30fps_h264_high.mov"
  "iphone7:iphone7_1080p_30fps_h264_high.mov"
  "lumix:lumix_g9_1080p_30fps_h264_high.mp4"
  "dji:dji_mini2_2_7k_24fps_h264_high.mp4"
  "img4138:IMG_4138.MOV"
  "img4273:IMG_4273.MOV"
  "asia_bottle:Artlist_AsiaBottle.mp4"
  "carplane:Artlist_CarPlane.mp4"
  "handbag:Artlist_Handbag.mp4"
  "horse_flag:Artlist_HorseFlag.mp4"
  "phone_booth:Artlist_PhoneBooth.mp4"
  "pirate_battle:Artlist_PirateBattle.mp4"
  "school_fight:Artlist_SchoolFight.mp4"
  "woman_subway:Artlist_WomanSubway.mp4"
)

run_one() {
  local tag=${1%%:*} file=${1#*:}
  local src="$DIR/$file"
  [ -f "$src" ] || { echo "RESULT $tag SKIP (missing $file)" | tee -a "$OUT"; return; }
  local W H N
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of default=nw=1:nk=1 "$src")
  H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "$src")
  N=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nw=1:nk=1 "$src")
  local AW=$(( (W/16)*16 )) AH=$(( (H/16)*16 ))
  PHASM_CAP_SRC="$file" PHASM_CAP_TAG="$tag" PHASM_CAP_W=$AW PHASM_CAP_H=$AH PHASM_CAP_N=$N PHASM_CAP_GOP=$GOP \
    "$BIN" --exact cap_realworld_one --ignored --nocapture 2>&1 | grep "^RESULT" | tee -a "$OUT"
}
export -f run_one; export DIR BIN OUT GOP

echo "=== #810 real-world capacity sweep START $(date) (bin=$(basename "$BIN")) ==="
printf '%s\n' "${PAIRS[@]}" | xargs -P 8 -I{} bash -c 'run_one "{}"'
echo "=== DONE $(date) ==="
echo "--- sorted by ratio ---"
sort -t= -k4 -n "$OUT" 2>/dev/null || sort "$OUT"
