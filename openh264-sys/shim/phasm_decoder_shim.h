// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// Minimal C-callable shim over OpenH264's C++ ISVCDecoder. Phasm-core's
// Rust code calls these entry points instead of the C++ class directly,
// avoiding the need to model the C++ vtable in Rust.
//
// Phase B.9.1: only the surface required to decode a phasm-fork-produced
// Annex-B byte stream into Y/U/V planes. The `dec_post_read` callback
// (declared in wels_stego.h and already plumbed through the Rust side)
// is NOT fired by this decoder yet — that requires fork-side patches
// inside parse_mb_syn_cabac.cpp (the residual sign-flag parse loop) and
// the MVD parse paths. Those patches are B.9.2, gated separately.
//
// All entry points are opinionated for phasm's deterministic defaults:
// error concealment disabled (we expect clean fork-encoder output), no
// parse-only mode (we want full reconstruction so the decoded YUV can
// be compared against the encoder's recon).

#ifndef PHASM_DECODER_SHIM_H
#define PHASM_DECODER_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque decoder handle. Caller treats this as an arbitrary pointer. */
typedef struct PhasmDecoderHandle PhasmDecoderHandle;

/* Create + destroy a decoder. Returns NULL on failure. */
PhasmDecoderHandle* phasm_decoder_create(void);
void                phasm_decoder_destroy(PhasmDecoderHandle* h);

/* Initialize with phasm's deterministic defaults. Returns 0 on success,
 * negative on error. After Initialize, the decoder is ready to consume
 * Annex-B bytes via phasm_decoder_decode_frame.
 *
 * Equivalent to phasm_decoder_initialize_with_options(h, 0). */
int32_t phasm_decoder_initialize(PhasmDecoderHandle* h);

/* Initialize with phasm's deterministic defaults plus per-instance
 * options. Returns 0 on success, negative on error.
 *
 * Options:
 *   parse_only: when 1, sets `SDecodingParam.bParseOnly = true`.
 *
 *               WARNING — NOT usable for phasm stego extract.
 *               OpenH264's bParseOnly is a metadata-only mode: it
 *               parses SPS / PPS / slice-headers and exits before
 *               any residual decoding via the SParserBsInfo
 *               early-out path at decoder_core.cpp:89. The CABAC
 *               residual parse never runs, so phasm's dec_post_read
 *               hooks (registered through wels_stego.h ABI) do NOT
 *               fire. Confirmed empirically by Rust test
 *               b9_2_6_parse_only_skips_residual_parse: full
 *               decode emits 228732 / 64 / 2 / 2 hooks across the 4
 *               stego domains; parse_only emits zero.
 *
 *               This option is retained for callers that only need
 *               SPS-level metadata (parsed frame dimensions,
 *               profile, etc.). Stego extract callers MUST use
 *               parse_only = 0.
 */
int32_t phasm_decoder_initialize_with_options(
    PhasmDecoderHandle* h,
    int32_t parse_only);

/* Uninitialize the decoder (frees internal buffers). Called automatically
 * by phasm_decoder_destroy; only call explicitly to re-init on the same
 * handle. */
int32_t phasm_decoder_uninitialize(PhasmDecoderHandle* h);

/* Decode an Annex-B chunk (one frame's worth of NAL units) into Y/U/V
 * planes. Output pointers point to decoder-internal buffers that remain
 * valid until the next decode_frame call. Caller copies as needed if it
 * wants to retain the data across decode calls.
 *
 * Inputs:
 *   bytes / len           - Annex-B byte stream to feed the decoder.
 *
 * Outputs (only valid when return value >= 0 AND *out_buffer_status == 1):
 *   *out_buffer_status    - 0 = decoder consumed bytes but no frame ready
 *                           yet (e.g. parsed SPS/PPS only); 1 = a full
 *                           frame's worth of YUV is now available.
 *   *out_width / height   - frame dimensions in pixels.
 *   *out_y_stride         - luma plane stride in bytes.
 *   *out_uv_stride        - chroma plane stride in bytes (both U and V
 *                           share this stride per OpenH264 conventions).
 *   *out_y/u/v_plane      - pointers to internal Y/U/V buffers.
 *
 * Returns:
 *   0     - success (buffer_status carries the ready/not-ready signal)
 *  -1     - null handle or decoder
 *  -2     - null required out-param
 *  -4-N   - OpenH264 DecodeFrameNoDelay returned negative DECODING_STATE
 *           = N; caller can map N back to OpenH264's enum if needed.
 */
int32_t phasm_decoder_decode_frame(
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
    const uint8_t** out_v_plane);

/* Drain one frame from the decoder's internal reorder buffer. Used
 * after the last input NAL when the decoder is for a non-Baseline
 * profile (Main, High) which buffers frames for display-order
 * reordering. Same output semantics as phasm_decoder_decode_frame. */
int32_t phasm_decoder_flush_frame(
    PhasmDecoderHandle* h,
    int32_t* out_buffer_status,
    int32_t* out_width,
    int32_t* out_height,
    int32_t* out_y_stride,
    int32_t* out_uv_stride,
    const uint8_t** out_y_plane,
    const uint8_t** out_u_plane,
    const uint8_t** out_v_plane);

/* Query how many frames remain in the decoder's reorder buffer after
 * input has been exhausted. Returns 0 on success and stores the count
 * in *out_count, or negative on error. */
int32_t phasm_decoder_frames_remaining(
    PhasmDecoderHandle* h,
    int32_t* out_count);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PHASM_DECODER_SHIM_H */
