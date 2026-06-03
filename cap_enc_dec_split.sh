#!/bin/bash
# #814 / §15 — corpus-wide enc/dec split. For each fixture: probe reported
# capacity, sweep sizes around it (0.85..1.25x) with K random salts,
# classify MsgTooLarge (graceful) / DECODE-FAIL (SILENT LOSS) / OK. The
# SILENT_MAX column is the headline: any nonzero anywhere = a CASCADE.V2
# residual (encode>decode) = correctness bug. All-zero confirms over-reports
# are always graceful. 8-way cross-process (OH264 C-global state).
set -u
DIR=/Users/cgaffga/Development/phasm/core/test-vectors/video/h264/real-world/source
BIN=$(ls -t /Users/cgaffga/Development/phasm/core/target/release/deps/cap_realworld_810-* 2>/dev/null | grep -v '\.d$' | head -1)
OUT=/tmp/cap814_encdec_corpus.txt
GOP=30
: > "$OUT"

PAIRS=(
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
    "$BIN" --exact cap_enc_dec_split_one --ignored --nocapture 2>&1 | grep "^RESULT" | tee -a "$OUT"
}
export -f run_one; export DIR BIN OUT GOP

echo "=== #814 corpus enc/dec split START $(date) (bin=$(basename "$BIN")) ==="
printf '%s\n' "${PAIRS[@]}" | xargs -P 8 -I{} bash -c 'run_one "{}"'
echo "=== DONE $(date) ==="
echo "--- any SILENT_MAX>0 (silent data loss = correctness bug)? ---"
grep -E "SILENT_MAX=[1-9]" "$OUT" || echo "NONE — all over-reports graceful (MessageTooLarge), zero silent loss."
