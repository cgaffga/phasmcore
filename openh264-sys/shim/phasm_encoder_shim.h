// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// Minimal C-callable shim over OpenH264's C++ ISVCEncoder. Phasm-core's
// Rust code calls these entry points instead of the C++ class directly,
// avoiding the need to model the C++ vtable in Rust.
//
// Phase B.5: only the surface required for the Rust smoke test. Phase
// B.6+ may add a few more entry points (e.g. dynamic parameter updates,
// long-GOP control) as the orchestrator wires up STC + cost vectors.
//
// All entry points are opinionated for phasm's deterministic defaults:
// single-threaded encoder, CABAC entropy, Main profile, RC_OFF_MODE
// with caller-specified QP, no scene-change detection, no adaptive QP,
// no frame skip. Mirrors the Phase A.5 validation harness exactly.

#ifndef PHASM_ENCODER_SHIM_H
#define PHASM_ENCODER_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque encoder handle. Caller treats this as an arbitrary pointer. */
typedef struct PhasmEncoderHandle PhasmEncoderHandle;

/* Frame types (subset of OpenH264's EVideoFrameType for the Rust side
 * to know which frames are IDR vs P). Values match codec_def.h. */
#define PHASM_FRAME_INVALID 0
#define PHASM_FRAME_IDR     1
#define PHASM_FRAME_I       2
#define PHASM_FRAME_P       3
#define PHASM_FRAME_SKIP    4
#define PHASM_FRAME_IPMIXED 5

/* Create + destroy an encoder. Returns NULL on failure. */
PhasmEncoderHandle* phasm_encoder_create(void);
void                phasm_encoder_destroy(PhasmEncoderHandle* h);

/* Initialize with phasm's deterministic defaults at the given geometry
 * and QP. Returns 0 on success, negative on error.
 *
 * gop_size: I-frame period (e.g. 30 = IDR every 30 frames; pass a
 *           value larger than your frame count to get one IDR followed
 *           by P-frames only). */
int32_t phasm_encoder_initialize(
    PhasmEncoderHandle* h,
    int32_t width,
    int32_t height,
    int32_t qp,
    int32_t gop_size);

/* Uninitialize the encoder (frees internal buffers). Called automatically
 * by phasm_encoder_destroy; only call explicitly if you want to re-init
 * with different parameters on the same handle. */
int32_t phasm_encoder_uninitialize(PhasmEncoderHandle* h);

/* Encode one YUV 4:2:0 frame. Inputs are caller-owned and read-only.
 * Output is copied into `out_buf` (capacity `out_buf_capacity` bytes);
 * `*out_bytes_written` receives the actual size on success.
 *
 * Returns one of PHASM_FRAME_* (>= 0) on success, or a negative error:
 *   -1: null handle or encoder
 *   -2: null out_bytes_written
 *   -3: out_buf too small for the encoded frame
 *   -4: EncodeFrame returned a non-success status; check the *out_bytes_written
 *       value (-4-N where N is the OpenH264 error code) for details.
 *
 * timestamp_msec is the presentation timestamp in milliseconds (passed
 * through to the encoder via SSourcePicture.uiTimeStamp). */
int32_t phasm_encoder_encode_frame(
    PhasmEncoderHandle* h,
    const uint8_t* y_plane, int32_t y_stride,
    const uint8_t* u_plane, int32_t u_stride,
    const uint8_t* v_plane, int32_t v_stride,
    int32_t width, int32_t height,
    int64_t timestamp_msec,
    uint8_t* out_buf,
    int32_t out_buf_capacity,
    int32_t* out_bytes_written);

/* C.9.0 (#482) — toggle the encoder fork's dual_recon (pVisualRef[] mirror
 * pool) before InitializeExt. enabled=0 skips ALL visual_recon allocation
 * and downstream mirror writes; the resulting bitstream is byte-identical
 * to the dual_recon-enabled run (mirror is internal-only — encoder
 * reference and bitstream emission both run off pDecPic). Used by the
 * Pass-1 cover probe (whose bitstream is walked then discarded). Pass-2
 * production encodes leave the flag at its default (enabled=1). The flag
 * is a process global — set it BEFORE phasm_encoder_initialize so the
 * fork's InitDqLayers reads the current value. */
void phasm_encoder_set_dual_recon_enabled(int32_t enabled);

/* P3.3a (2026-05-25) — post-frame DPB correction for inter-frame
 * cascade elimination. Returns a mutable pointer to pDecPic's Y plane
 * and its stride so the Rust side can apply per-pixel IDCT deltas
 * after each frame encode. Returns 0 on success, -1 if handle/encoder
 * is null or no pDecPic is available.
 *
 * Safety: the returned pointer is valid until the next encode_frame
 * call or encoder destruction. Caller must not hold it across those
 * boundaries. Width × height pixels are addressable at
 * y_ptr[row * *y_stride + col]. */
int32_t phasm_encoder_get_dec_pic_y(
    PhasmEncoderHandle* h,
    uint8_t** y_ptr,
    int32_t* y_stride);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PHASM_ENCODER_SHIM_H */
