// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// Implementation of phasm_encoder_shim.h. See header for design notes.

#include "phasm_encoder_shim.h"

#include "codec_api.h"
#include "codec_app_def.h"
#include "codec_def.h"
#include "wels_stego.h"  // phasm: C.9.0 phasm_set_dual_recon_enabled (#482)

#include <cstring>
#include <new>

struct PhasmEncoderHandle {
  ISVCEncoder* enc;
};

// Pick H.264 Level Annex A based on macroblocks/second + frame size.
// Reference: ITU-T H.264 Table A-1 (Level limits). We pick the smallest
// level whose MaxMBPS and MaxFS (Max Frame Size in MBs) admit the
// requested (width, height, fps). Bitrate limits are not checked
// because phasm runs RC_OFF — actual bitrate is QP-dependent and
// typically well under any level's cap.
//
// Returns LEVEL_4_0 by default (covers 1080p × 30, the common ceiling
// in phasm's wall-measurement fixtures). Falls through to LEVEL_5_1
// for cases beyond 1080p × 60.
static ELevelIdc pick_level(int32_t width, int32_t height, float fps) {
  if (width <= 0 || height <= 0 || fps <= 0.0f) {
    return LEVEL_3_0;
  }
  const int32_t mbs_per_frame =
      ((width + 15) / 16) * ((height + 15) / 16);
  const int32_t mbps = static_cast<int32_t>(
      static_cast<float>(mbs_per_frame) * fps);

  // (MaxMBPS, MaxFS, Level). First row whose both limits accommodate
  // (mbps, mbs_per_frame) wins. Numbers from H.264 Annex A Table A-1.
  struct LevelEntry { int32_t max_mbps; int32_t max_fs; ELevelIdc level; };
  const LevelEntry table[] = {
      {  1485,    99, LEVEL_1_0 },  // QCIF @ 15
      {  3000,   396, LEVEL_1_1 },  // CIF  @ 7.5
      {  6000,   396, LEVEL_1_2 },  // CIF  @ 15
      { 11880,   396, LEVEL_1_3 },  // CIF  @ 30
      { 11880,   396, LEVEL_2_0 },  // CIF  @ 30 (more bitrate)
      { 19800,   792, LEVEL_2_1 },
      { 20250,  1620, LEVEL_2_2 },
      { 40500,  1620, LEVEL_3_0 },  // 720×480 @ 30 / 480p
      {108000,  3600, LEVEL_3_1 },  // 1280×720 @ 30
      {216000,  5120, LEVEL_3_2 },  // 1280×720 @ 60
      {245760,  8192, LEVEL_4_0 },  // 1920×1080 @ 30
      {245760,  8192, LEVEL_4_1 },  // same as 4.0, more bitrate
      {522240,  8704, LEVEL_4_2 },  // 1920×1080 @ 60
      {589824, 22080, LEVEL_5_0 },
      {983040, 36864, LEVEL_5_1 },
  };
  const int n = sizeof(table) / sizeof(table[0]);
  for (int i = 0; i < n; ++i) {
    if (mbps <= table[i].max_mbps && mbs_per_frame <= table[i].max_fs) {
      return table[i].level;
    }
  }
  return LEVEL_5_2;
}

extern "C" PhasmEncoderHandle* phasm_encoder_create(void) {
  PhasmEncoderHandle* h = new (std::nothrow) PhasmEncoderHandle;
  if (!h) return nullptr;
  h->enc = nullptr;
  if (WelsCreateSVCEncoder(&h->enc) != 0 || !h->enc) {
    delete h;
    return nullptr;
  }
  return h;
}

extern "C" void phasm_encoder_destroy(PhasmEncoderHandle* h) {
  if (!h) return;
  if (h->enc) {
    h->enc->Uninitialize();
    WelsDestroySVCEncoder(h->enc);
    h->enc = nullptr;
  }
  delete h;
}

extern "C" int32_t phasm_encoder_initialize(
    PhasmEncoderHandle* h,
    int32_t width,
    int32_t height,
    int32_t qp,
    int32_t gop_size) {
  if (!h || !h->enc) return -1;

  SEncParamExt param;
  h->enc->GetDefaultParams(&param);

  // Phasm deterministic defaults (matches Phase A.5 validation harness).
  param.iUsageType                 = CAMERA_VIDEO_REAL_TIME;
  param.iPicWidth                  = width;
  param.iPicHeight                 = height;
  param.fMaxFrameRate              = 30.0f;
  param.iTargetBitrate             = 1000000;
  param.iRCMode                    = RC_OFF_MODE;
  param.iTemporalLayerNum          = 1;
  param.iSpatialLayerNum           = 1;
  param.uiIntraPeriod              = static_cast<uint32_t>(gop_size);
  // V0.4.A reverted 2026-05-23 — OH264 cannot deliver real multi-ref P.
  //
  // Original V0.4.A.1 (commit ad648e3b) bumped this to 2 expecting per-MB
  // `ref_idx_l0` bins. V0.4.A.2 measurement revealed OH264 silently rewrote
  // the runtime cap back to 1 in `WelsCheckNumRefSetting` (au_set.cpp:130)
  // while leaving `iMaxNumRefFrame = 2` for the SPS wire field. Net effect:
  // SPS said max_ref=2 but per-MB encoding was unchanged (still zero
  // ref_idx_l0 bins). Even patching au_set.cpp's rewrite wouldn't help —
  // `encoder_ext.cpp:2869` comments `"always get item 0 due to reordering
  // done"`; the ME pipeline is hardwired single-ref by design. Real
  // multi-ref P would require porting the cross-reference motion-search
  // loop (200–500 LOC, comparable scope to B-frames — same WONTFIX).
  //
  // Empirical real-world corpus survey (V0.4.A.2): 7/7 fixtures emit zero
  // per-MB ref_idx_l0 bins. The dominant phone-camera cohort (iPhone5/7,
  // DJI) uses SPS max_ref=1, matching pre-A.1 behavior. Setting
  // iNumRefFrame=2 here shifts phasm into the rarer Lumix-style cohort
  // (max_ref=2 but per-MB still single-ref) — a net stealth LOSS.
  //
  // Stays at 1 until a backend switch (VideoToolbox / MediaCodec / x264)
  // makes real multi-ref P available. See:
  //   - docs/design/video/h264/v1_1-bframe-stego-roadmap.md (now covers
  //     B-frames + multi-ref P; both gated on backend switch)
  //   - memory/h264_v04a_multi_ref_p_negative.md (full investigation log)
  param.iNumRefFrame               = 1;
  param.iEntropyCodingModeFlag     = 1;  // CABAC
  // Task #339 (2026-05-12) — restore OpenH264's default
  // multi-threading. iMultipleThreadIdc=0 means "auto" (encoder
  // picks worker count from cpuid / DynamicDetectCpuCores).
  //
  // Hook ordering safety: all stego hooks (Phase A.5 HOOK-A..H7,
  // B.9.2 decoder hooks) fire from inside WelsCodeOneSlice ->
  // WelsCodeOneMb. With uiSliceMode=SM_SINGLE_SLICE there is
  // exactly one slice per frame, so per-MB encoding stays on a
  // single worker thread regardless of iMultipleThreadIdc. Other
  // worker threads (parallel deblock filter, lookahead background)
  // never fire stego hooks. The trampoline's user_data pointer is
  // accessed from one thread only; no Mutex needed.
  //
  // #778 (2026-05-29): VERIFIED deterministic — with a fixed crypto
  // seed (PHASM_DETERMINISTIC_SEED), two encodes of the same input are
  // byte-identical at idc=0. The Mode-B decode failures are NOT
  // encoder non-determinism (that was the random AES salt/nonce per
  // encode, by design). They are a real payload-dependent intra-call
  // wire_only cascade. So multi-threading is safe to keep; do NOT
  // pessimize to single-thread.
  //
  // bUseLoadBalancing stays false — codec_app_def.h documents it
  // as "result of each run may be different", which would break
  // deterministic stego output. It's also only consulted when
  // uiSliceMode==1 or 3; SM_SINGLE_SLICE is uiSliceMode==0, so
  // the flag is a no-op for us either way.
  param.iMultipleThreadIdc         = 0;
  param.bEnableSceneChangeDetect   = false;
  param.bEnableBackgroundDetection = false;
  param.bEnableAdaptiveQuant       = false;
  param.bUseLoadBalancing          = false;
  param.bEnableDenoise             = false;
  param.bEnableFrameSkip           = false;
  param.bEnableLongTermReference   = false;
  param.iComplexityMode            = MEDIUM_COMPLEXITY;
  // V0.4.E (2026-05-23) — pin to CONSTANT_ID. OH264's `FillDefault`
  // sets INCREASING_ID (param_svc.h:187) which rotates pic_parameter_set_id
  // per IDR (0,1,2,...,N). Two problems with that for phasm:
  //
  //   1. **Mux data-integrity bug**: AVAssetWriter on iOS only carries
  //      ONE PPS in its CMVideoFormatDescription (built from the first
  //      PPS NAL parsed in PhasmStegoTranscoder). Slices in later GOPs
  //      reference pps_id=1..N which no longer exist in the muxed MP4,
  //      producing ffmpeg's "non-existing PPS N referenced" warning on
  //      every IDR after the first.
  //   2. **Layer 3 stealth signal**: smartphone capture cohorts
  //      (iPhone / DJI / Lumix / phone-recording x264) all emit a single
  //      PPS at the start and never rotate. INCREASING_ID is a fingerprint
  //      seen mostly in higher-end ABR-streaming encoders, not consumer
  //      camera output. CONSTANT_ID matches the dominant real-world cohort
  //      already targeted by V0.4.C (PRO_HIGH) and V0.4.D (constraint_set
  //      4/5 = 0, POC type 0).
  //
  // CONSTANT_ID makes every PPS NAL identical (pps_id=0, sps_id=0) and
  // every slice header reference pps_id=0. OH264 still re-emits an SPS+PPS
  // pair at each IDR, but they're byte-identical now, so the mux's
  // first-PPS-only capture becomes correct by construction.
  param.eSpsPpsIdStrategy          = CONSTANT_ID;

  param.sSpatialLayers[0].iVideoWidth     = width;
  param.sSpatialLayers[0].iVideoHeight    = height;
  param.sSpatialLayers[0].fFrameRate      = 30.0f;
  param.sSpatialLayers[0].iSpatialBitrate = 1000000;
  // V0.4.C (2026-05-23) — High profile (=100) matches the dominant
  // real-world cohort: iPhone5/7, DJI Mini2, Lumix G9, professional
  // x264 content (Artlist SchoolFight/WomanSubway) all emit profile=100.
  // Previous PRO_MAIN (=77) matched only the older-Artlist-Main cohort
  // (~2/7 of corpus). High enables 8x8 transform + adds the SPS
  // chroma_format_idc/bit_depth fields that the High-profile cohort
  // already emits. CABAC stays valid (Main + High both permit CABAC).
  param.sSpatialLayers[0].uiProfileIdc    = PRO_HIGH;
  // #691 fix (2026-05-22): was hardcoded LEVEL_3_0, which caps at
  // 720p × 30. phasm encodes 1080p × 30 in production fixtures —
  // technically a Level violation. Lenient decoders (AVFoundation,
  // browser <video>) accept; strict decoders may reject. Now picked
  // dynamically from (width, height, fps).
  param.sSpatialLayers[0].uiLevelIdc      = pick_level(width, height, 30.0f);
  param.sSpatialLayers[0].iDLayerQp       = qp;
  param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SM_SINGLE_SLICE;

  return h->enc->InitializeExt(&param);
}

extern "C" int32_t phasm_encoder_uninitialize(PhasmEncoderHandle* h) {
  if (!h || !h->enc) return -1;
  return h->enc->Uninitialize();
}

extern "C" int32_t phasm_encoder_encode_frame(
    PhasmEncoderHandle* h,
    const uint8_t* y_plane, int32_t y_stride,
    const uint8_t* u_plane, int32_t u_stride,
    const uint8_t* v_plane, int32_t v_stride,
    int32_t width, int32_t height,
    int64_t timestamp_msec,
    uint8_t* out_buf,
    int32_t out_buf_capacity,
    int32_t* out_bytes_written) {
  if (!h || !h->enc) return -1;
  if (!out_bytes_written) return -2;
  *out_bytes_written = 0;

  SSourcePicture pic;
  std::memset(&pic, 0, sizeof(pic));
  pic.iColorFormat = videoFormatI420;
  pic.iPicWidth    = width;
  pic.iPicHeight   = height;
  pic.iStride[0]   = y_stride;
  pic.iStride[1]   = u_stride;
  pic.iStride[2]   = v_stride;
  // The OpenH264 SSourcePicture::pData is non-const internally but only
  // read during encode. const_cast is safe here under the encoder's
  // documented read-only contract on input frames.
  pic.pData[0]     = const_cast<uint8_t*>(y_plane);
  pic.pData[1]     = const_cast<uint8_t*>(u_plane);
  pic.pData[2]     = const_cast<uint8_t*>(v_plane);
  pic.uiTimeStamp  = timestamp_msec;

  SFrameBSInfo info;
  std::memset(&info, 0, sizeof(info));
  int rv = h->enc->EncodeFrame(&pic, &info);
  if (rv != cmResultSuccess) {
    // Encode the upstream error code into the return so caller can debug
    // without losing detail. Map to negative range starting at -4.
    return -4 - rv;
  }

  int32_t total = 0;
  for (int li = 0; li < info.iLayerNum; ++li) {
    SLayerBSInfo& layer = info.sLayerInfo[li];
    int32_t layer_bytes = 0;
    for (int n = 0; n < layer.iNalCount; ++n) {
      layer_bytes += layer.pNalLengthInByte[n];
    }
    if (total + layer_bytes > out_buf_capacity) {
      *out_bytes_written = total;  // partial write — caller can detect via -3
      return -3;
    }
    std::memcpy(out_buf + total, layer.pBsBuf, static_cast<size_t>(layer_bytes));
    total += layer_bytes;
  }
  *out_bytes_written = total;
  return info.eFrameType;
}

extern "C" void phasm_encoder_set_dual_recon_enabled(int32_t enabled) {
  /* Direct passthrough to the fork's libcommon global. No handle needed —
   * the flag is a process-wide setting consulted by InitDqLayers on the
   * NEXT encoder init. Caller must set it BEFORE phasm_encoder_initialize.
   */
  phasm_set_dual_recon_enabled(enabled);
}

extern "C" int32_t phasm_encoder_get_dec_pic_y(
    PhasmEncoderHandle* h,
    uint8_t** y_ptr,
    int32_t* y_stride) {
  if (!h || !h->enc || !y_ptr || !y_stride) return -1;
  /* Access the encoder's current reconstruction picture via the
   * ISVCEncoder vtable. GetEncDecPic is a phasm fork addition
   * (wels_stego.h) that returns a pointer to the most recently
   * reconstructed frame's Y plane (= pDecPic->pData[0]) + stride.
   * Returns false if no frame has been encoded yet. */
  uint8_t* p = nullptr;
  int32_t s = 0;
  if (!phasm_encoder_get_enc_dec_pic(h->enc, &p, &s)) return -1;
  *y_ptr = p;
  *y_stride = s;
  return 0;
}
