// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// Implementation of phasm_decoder_shim.h. See header for design notes.

#include "phasm_decoder_shim.h"

#include "codec_api.h"
#include "codec_app_def.h"
#include "codec_def.h"

#include <cstring>
#include <new>

struct PhasmDecoderHandle {
  ISVCDecoder* dec;
};

extern "C" PhasmDecoderHandle* phasm_decoder_create(void) {
  PhasmDecoderHandle* h = new (std::nothrow) PhasmDecoderHandle;
  if (!h) return nullptr;
  h->dec = nullptr;
  if (WelsCreateDecoder(&h->dec) != 0 || !h->dec) {
    delete h;
    return nullptr;
  }
  return h;
}

extern "C" void phasm_decoder_destroy(PhasmDecoderHandle* h) {
  if (!h) return;
  if (h->dec) {
    h->dec->Uninitialize();
    WelsDestroyDecoder(h->dec);
    h->dec = nullptr;
  }
  delete h;
}

extern "C" int32_t phasm_decoder_initialize(PhasmDecoderHandle* h) {
  return phasm_decoder_initialize_with_options(h, 0 /* parse_only off */);
}

extern "C" int32_t phasm_decoder_initialize_with_options(
    PhasmDecoderHandle* h,
    int32_t parse_only) {
  if (!h || !h->dec) return -1;

  SDecodingParam param;
  std::memset(&param, 0, sizeof(param));
  // Phasm deterministic defaults: no error concealment (clean fork
  // output, concealment would mask divergences we want to see).
  param.uiTargetDqLayer = 0xff;            // decode all layers
  param.eEcActiveIdc    = ERROR_CON_DISABLE;
  // B.9.2.6: parse-only path. WARNING — bParseOnly is NOT compatible
  // with phasm stego hooks. OpenH264 exits before residual decoding
  // (decoder_core.cpp:89), so dec_post_read callbacks never fire.
  // See header comment + the negative-result test
  // `b9_2_6_parse_only_skips_residual_parse` for empirical proof.
  // This option is retained only for SPS-metadata-only callers.
  param.bParseOnly      = (parse_only != 0);
  param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

  return h->dec->Initialize(&param);
}

extern "C" int32_t phasm_decoder_uninitialize(PhasmDecoderHandle* h) {
  if (!h || !h->dec) return -1;
  return h->dec->Uninitialize();
}

extern "C" int32_t phasm_decoder_decode_frame(
    PhasmDecoderHandle* h,
    const uint8_t* bytes,
    int32_t len,
    int32_t* out_buffer_status,
    int32_t* out_width,
    int32_t* out_height,
    int32_t* out_y_stride,
    int32_t* out_uv_stride,
    const uint8_t** out_y_plane,
    const uint8_t** out_u_plane,
    const uint8_t** out_v_plane) {
  if (!h || !h->dec) return -1;
  if (!out_buffer_status || !out_width || !out_height ||
      !out_y_stride || !out_uv_stride ||
      !out_y_plane || !out_u_plane || !out_v_plane) {
    return -2;
  }

  *out_buffer_status = 0;
  *out_width = 0;
  *out_height = 0;
  *out_y_stride = 0;
  *out_uv_stride = 0;
  *out_y_plane = nullptr;
  *out_u_plane = nullptr;
  *out_v_plane = nullptr;

  unsigned char* dst[3] = {nullptr, nullptr, nullptr};
  SBufferInfo info;
  std::memset(&info, 0, sizeof(info));

  DECODING_STATE state =
      h->dec->DecodeFrameNoDelay(bytes, len, dst, &info);
  // Most non-zero DECODING_STATE values are NOT errors — they're
  // status bits like dsFramePending (the decoder is still buffering
  // and needs more NALs to produce a frame). The OpenH264 reference
  // h264dec.cpp ignores `state` entirely and just checks
  // `info.iBufferStatus`. We only treat the "logic-level" errors
  // (>= 0x1000: invalid argument, init expected, OOM, dst-buf-need-
  // expand) as fatal; below-0x1000 bits are bitstream-parse signals
  // that the caller handles via iBufferStatus.
  const int32_t kFatalMask = 0xF000;
  if ((static_cast<int32_t>(state) & kFatalMask) != 0) {
    return -4 - static_cast<int32_t>(state);
  }

  *out_buffer_status = info.iBufferStatus;
  if (info.iBufferStatus == 1) {
    *out_width      = info.UsrData.sSystemBuffer.iWidth;
    *out_height     = info.UsrData.sSystemBuffer.iHeight;
    *out_y_stride   = info.UsrData.sSystemBuffer.iStride[0];
    *out_uv_stride  = info.UsrData.sSystemBuffer.iStride[1];
    *out_y_plane    = dst[0];
    *out_u_plane    = dst[1];
    *out_v_plane    = dst[2];
  }
  return 0;
}

extern "C" int32_t phasm_decoder_flush_frame(
    PhasmDecoderHandle* h,
    int32_t* out_buffer_status,
    int32_t* out_width,
    int32_t* out_height,
    int32_t* out_y_stride,
    int32_t* out_uv_stride,
    const uint8_t** out_y_plane,
    const uint8_t** out_u_plane,
    const uint8_t** out_v_plane) {
  if (!h || !h->dec) return -1;
  if (!out_buffer_status || !out_width || !out_height ||
      !out_y_stride || !out_uv_stride ||
      !out_y_plane || !out_u_plane || !out_v_plane) {
    return -2;
  }

  *out_buffer_status = 0;
  *out_width = 0; *out_height = 0;
  *out_y_stride = 0; *out_uv_stride = 0;
  *out_y_plane = nullptr; *out_u_plane = nullptr; *out_v_plane = nullptr;

  unsigned char* dst[3] = {nullptr, nullptr, nullptr};
  SBufferInfo info;
  std::memset(&info, 0, sizeof(info));

  DECODING_STATE state = h->dec->FlushFrame(dst, &info);
  const int32_t kFatalMask = 0xF000;
  if ((static_cast<int32_t>(state) & kFatalMask) != 0) {
    return -4 - static_cast<int32_t>(state);
  }

  *out_buffer_status = info.iBufferStatus;
  if (info.iBufferStatus == 1) {
    *out_width      = info.UsrData.sSystemBuffer.iWidth;
    *out_height     = info.UsrData.sSystemBuffer.iHeight;
    *out_y_stride   = info.UsrData.sSystemBuffer.iStride[0];
    *out_uv_stride  = info.UsrData.sSystemBuffer.iStride[1];
    *out_y_plane    = dst[0];
    *out_u_plane    = dst[1];
    *out_v_plane    = dst[2];
  }
  return 0;
}

extern "C" int32_t phasm_decoder_frames_remaining(
    PhasmDecoderHandle* h,
    int32_t* out_count) {
  if (!h || !h->dec) return -1;
  if (!out_count) return -2;
  int32_t n = 0;
  long rv = h->dec->GetOption(DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER, &n);
  if (rv != 0) {
    return -4 - static_cast<int32_t>(rv);
  }
  *out_count = n;
  return 0;
}
