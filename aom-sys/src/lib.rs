// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga

//! FFI to `shim/phasm_aom.c`: one still image (8-bit 4:2:0, full range) encoded with libaom
//! (vendor/libaom, upstream, pinned by the submodule). The AVIF container and the fitting are in
//! phasm-core (`photo`).

#![no_std]

unsafe extern "C" {
    /// Encode one 8-bit 4:2:0 image (tight planes: `w` and `w / 2` bytes per row; `w`, `h` even) at
    /// libaom quantizer `quantizer` (0-63) and speed `speed`. On success (0) `*out` holds the AV1
    /// temporal unit (temporal delimiter, sequence header, frame), to be freed with
    /// [`phasm_aom_free`]. Negative: invalid input (-1) or a libaom failure (-2 .. -6).
    pub fn phasm_aom_encode_i420(
        y: *const u8,
        u: *const u8,
        v: *const u8,
        w: u32,
        h: u32,
        quantizer: i32,
        speed: i32,
        threads: i32,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;

    pub fn phasm_aom_free(p: *mut u8);
}
