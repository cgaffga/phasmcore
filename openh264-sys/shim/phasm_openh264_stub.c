// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Christoph Gaffga
// https://github.com/cgaffga/phasmcore
//
// Task #381 — Phase B.12: fail-fast stub library for the OpenH264
// backend when the `vendor/phasm-openh264` submodule is absent.
//
// When `core/openh264-sys/build.rs` cannot locate the submodule, it
// skips the meson + ninja invocation that would normally build
// libopenh264.a from the fork's C++ sources and compiles THIS file
// instead. Every C symbol the Rust FFI surface in `src/lib.rs`
// declares is provided here as a no-op that signals "OpenH264
// backend not available":
//
//   - int-returning functions return -1
//   - handle-returning functions return NULL
//   - void functions are no-ops
//   - WelsStegoAbiVersion returns 0 (≠ PHASM_STEGO_ABI_VERSION) so
//     callers can detect the missing backend at startup
//
// Higher-level Rust code (`StegoSession::new`, `Decoder::new`) sees
// the NULL/-1 returns, propagates them as `EncoderError` /
// `DecoderError`, and the caller falls back to the pure-Rust walker
// path or surfaces the configuration error to the user.
//
// This file is NEVER compiled when the submodule IS present —
// build.rs picks one path or the other, not both.

#include <stddef.h>
#include <stdint.h>

// ── wels_stego.h ABI ─────────────────────────────────────────────
// Real definitions live in the fork's
// codec/common/src/wels_stego_common.cpp (added by the
// 0002-wels-stego-api-header patch).

int32_t WelsRegisterPhasmStegoCallbacks(
    const void* cbs,
    void* user_data
) {
    (void)cbs;
    (void)user_data;
    return -1;
}

void WelsStegoSetFrameNum(uint32_t frame_num) {
    (void)frame_num;
}

uint32_t WelsStegoAbiVersion(void) {
    return 0;
}

// ── phasm_encoder_shim ABI ───────────────────────────────────────
// Real definitions: core/openh264-sys/shim/phasm_encoder_shim.cc.

void* phasm_encoder_create(void) {
    return NULL;
}

void phasm_encoder_destroy(void* h) {
    (void)h;
}

int32_t phasm_encoder_initialize(
    void* h,
    int32_t width,
    int32_t height,
    int32_t qp,
    int32_t gop_size
) {
    (void)h; (void)width; (void)height; (void)qp; (void)gop_size;
    return -1;
}

int32_t phasm_encoder_uninitialize(void* h) {
    (void)h;
    return -1;
}

int32_t phasm_encoder_encode_frame(
    void* h,
    const uint8_t* y_plane, int32_t y_stride,
    const uint8_t* u_plane, int32_t u_stride,
    const uint8_t* v_plane, int32_t v_stride,
    int32_t width, int32_t height,
    int64_t timestamp_msec,
    uint8_t* out_buf,
    int32_t out_buf_capacity,
    int32_t* out_bytes_written
) {
    (void)h;
    (void)y_plane; (void)y_stride;
    (void)u_plane; (void)u_stride;
    (void)v_plane; (void)v_stride;
    (void)width; (void)height;
    (void)timestamp_msec;
    (void)out_buf; (void)out_buf_capacity; (void)out_bytes_written;
    return -1;
}

// ── phasm_decoder_shim ABI ───────────────────────────────────────
// Real definitions: core/openh264-sys/shim/phasm_decoder_shim.cc.

void* phasm_decoder_create(void) {
    return NULL;
}

void phasm_decoder_destroy(void* h) {
    (void)h;
}

int32_t phasm_decoder_initialize(void* h) {
    (void)h;
    return -1;
}

int32_t phasm_decoder_initialize_with_options(
    void* h,
    int32_t parse_only
) {
    (void)h; (void)parse_only;
    return -1;
}

int32_t phasm_decoder_uninitialize(void* h) {
    (void)h;
    return -1;
}

int32_t phasm_decoder_decode_frame(
    void* h,
    const uint8_t* bytes,
    int32_t len,
    int32_t* out_buffer_status,
    int32_t* out_width,
    int32_t* out_height,
    int32_t* out_y_stride,
    int32_t* out_uv_stride,
    const uint8_t** out_y_plane,
    const uint8_t** out_u_plane,
    const uint8_t** out_v_plane
) {
    (void)h;
    (void)bytes; (void)len;
    (void)out_buffer_status;
    (void)out_width; (void)out_height;
    (void)out_y_stride; (void)out_uv_stride;
    (void)out_y_plane; (void)out_u_plane; (void)out_v_plane;
    return -1;
}

int32_t phasm_decoder_flush_frame(
    void* h,
    int32_t* out_buffer_status,
    int32_t* out_width,
    int32_t* out_height,
    int32_t* out_y_stride,
    int32_t* out_uv_stride,
    const uint8_t** out_y_plane,
    const uint8_t** out_u_plane,
    const uint8_t** out_v_plane
) {
    (void)h;
    (void)out_buffer_status;
    (void)out_width; (void)out_height;
    (void)out_y_stride; (void)out_uv_stride;
    (void)out_y_plane; (void)out_u_plane; (void)out_v_plane;
    return -1;
}

int32_t phasm_decoder_frames_remaining(
    void* h,
    int32_t* out_count
) {
    (void)h; (void)out_count;
    return -1;
}
