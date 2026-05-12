// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// Implementation of phasm_encoder_shim.h. See header for design notes.

#include "phasm_encoder_shim.h"

#include "codec_api.h"
#include "codec_app_def.h"
#include "codec_def.h"

#include <cstring>
#include <new>

struct PhasmEncoderHandle {
  ISVCEncoder* enc;
};

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
  param.iNumRefFrame               = 1;
  param.iEntropyCodingModeFlag     = 1;  // CABAC
  param.iMultipleThreadIdc         = 1;  // single-threaded for determinism
  param.bEnableSceneChangeDetect   = false;
  param.bEnableBackgroundDetection = false;
  param.bEnableAdaptiveQuant       = false;
  param.bUseLoadBalancing          = false;
  param.bEnableDenoise             = false;
  param.bEnableFrameSkip           = false;
  param.bEnableLongTermReference   = false;
  param.iComplexityMode            = MEDIUM_COMPLEXITY;

  param.sSpatialLayers[0].iVideoWidth     = width;
  param.sSpatialLayers[0].iVideoHeight    = height;
  param.sSpatialLayers[0].fFrameRate      = 30.0f;
  param.sSpatialLayers[0].iSpatialBitrate = 1000000;
  param.sSpatialLayers[0].uiProfileIdc    = PRO_MAIN;
  param.sSpatialLayers[0].uiLevelIdc      = LEVEL_3_0;
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
