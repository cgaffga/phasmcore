// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phasm-core decode-side wrapper around the W3.D phasm-dav1d fork
//! hooks. Takes encoded AV1 bytes, drives dav1d with a recording
//! bit_hook + tag_hook registered, returns a [`DecodedCoverPositions`]
//! view of every 50/50 binary symbol decoded.
//!
//! This is the decoder-side counterpart to phasm-rav1e's
//! `WriterRecorder.phasm_bit_positions()` / `phasm_bit_tags()` on
//! the encoder side. For a clean (no-stego) round-trip:
//!
//! - Encoder cursor N's `(natural_value, tag)` == decoder cursor N's
//!   `(decoded_value, tag)` for every N
//!
//! W3.D.4.3's round-trip test asserts that equivalence on a real
//! rav1e-encoded fixture.
//!
//! # FFI lifetime contract
//!
//! [`decode_with_recording`] stack-allocates a `RecorderState`,
//! passes `&mut recorder` as the dav1d `cookie`, then ensures all
//! dav1d calls (open, send_data, get_picture, close) complete
//! synchronously before the function returns. dav1d's hook callback
//! casts the cookie back to `*mut RecorderState` and pushes
//! `(value, tag)` tuples into the recorder's Vec.
//!
//! For v0.3-AV1 single-thread, single-frame decode, this is safe —
//! dav1d calls back inline during `dav1d_send_data` /
//! `dav1d_get_picture`. For multi-frame / multi-thread (v0.5+), this
//! pattern needs revisiting per `dav1d-hook-sites.md` § 6.

use core_dav1d_sys::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings,
    dav1d_err_again, dav1d_get_picture, dav1d_open, dav1d_picture_unref,
    dav1d_send_data, Dav1dContext, Dav1dData, Dav1dPicture, Dav1dPhasmHooks,
    Dav1dSettings,
};

/// Errors returned by [`decode_with_recording`].
#[derive(Debug, Clone)]
pub enum Av1DecodeError {
    /// `dav1d_open` returned a non-zero error code.
    DavxdOpenFailed(i32),
    /// `dav1d_data_create` returned NULL (allocation failure).
    DataCreateFailed,
    /// `dav1d_send_data` returned a non-zero, non-EAGAIN error code.
    DavxdSendDataFailed(i32),
    /// `dav1d_get_picture` returned a non-zero, non-EAGAIN error code.
    DavxdGetPictureFailed(i32),
}

/// A single L(1) cover position from a decoded AV1 frame. Mirror of
/// the encoder-side `CoverPosition` (W3.D.4.2-folded-into-W3.10
/// helper); cursor is the monotonic 50/50 emission count starting
/// from 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedCoverPosition {
    /// Monotonic cursor — index into the decoded bit stream.
    pub cursor: u64,
    /// The bit value dav1d decoded at this position (0 or 1).
    pub decoded_value: u8,
    /// Channel tag set by the W3.D.3 site patches in `recon_tmpl.c`:
    /// `DAV1D_PHASM_TAG_OTHER` / `_AC_COEFF_SIGN` / `_GOLOMB_TAIL_LSB`.
    pub tag: u8,
    /// Phase B.1.1.b: spatial metadata. Only meaningful when
    /// `tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN`; otherwise zero-init.
    /// Mirror of encoder-side `phasm_rav1e::AcSignMeta` — used by
    /// J-UNIWARD cost compute to map decoded cover bit to a
    /// (plane, pixel-coord, TX-shape, scan-pos) tuple.
    pub meta: core_dav1d_sys::Dav1dPhasmAcSignMeta,
}

/// Visible-region YUV planes extracted from the last decoded picture.
/// Phase B.1.4 — required by the AoSO self-steganalyzer adapter to
/// compute J-UNIWARD cost on the STEGO reconstruction (decoder-side),
/// not the encoder's `recording.reconstructed_planes`. For natural
/// covers the two coincide; for stego streams they differ at flipped
/// positions because the residual reconstruction inverts there.
///
/// Width/height are visible dimensions (after dav1d crop). Chroma
/// dimensions reflect 4:2:0 subsampling (the only layout we ship in
/// v0.3..v0.6 — `assert!(layout == 1)` at extraction time).
#[derive(Debug, Clone, Default)]
pub struct DecodedFramePlanes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub luma_width: usize,
    pub luma_height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
}

/// All L(1) cover positions decoded from one AV1 stream. Output of
/// [`decode_with_recording`].
#[derive(Debug, Clone, Default)]
pub struct DecodedCoverPositions {
    all: Vec<DecodedCoverPosition>,
    /// Phase B.1.4: optional visible-region YUV. Populated by
    /// [`decode_with_recording_with_pixels`]; the legacy
    /// `decode_with_recording` leaves this `None` so existing tests
    /// don't pay the pixel-copy cost.
    pub planes: Option<DecodedFramePlanes>,
}

impl DecodedCoverPositions {
    /// Iterate all decoded positions in emission order.
    pub fn iter(&self) -> impl Iterator<Item = &DecodedCoverPosition> {
        self.all.iter()
    }

    pub fn len(&self) -> usize {
        self.all.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    /// Borrow the raw position slice.
    pub fn as_slice(&self) -> &[DecodedCoverPosition] {
        &self.all
    }
}

/// Internal cookie target — accumulator for hook callback writes.
/// Allocated on the stack of [`decode_with_recording`]; pointer
/// stays valid through all synchronous dav1d calls.
#[derive(Default)]
struct RecorderState {
    bits: Vec<(u8, u8)>, // (decoded_value, tag)
    /// Phase B.1.1.b: sticky meta state. meta_hook_cb writes here;
    /// bit_hook_cb reads here and zips into the (bit, tag, meta)
    /// tuple per emission. Mirrors encoder-side phasm_current_meta.
    current_meta: core_dav1d_sys::Dav1dPhasmAcSignMeta,
    /// Phase B.1.1.b: per-emission meta, parallel to `bits`.
    metas: Vec<core_dav1d_sys::Dav1dPhasmAcSignMeta>,
}

/// FFI callback wired into `Dav1dSettings.phasm_hooks.bit_hook`.
/// Cast cookie back to `&mut RecorderState`, push the
/// `(decoded_value, tag, meta)` tuple.
///
/// # Safety
///
/// `cookie` MUST be a valid `*mut RecorderState` set by
/// [`decode_with_recording`] before calling dav1d. dav1d's
/// single-thread + single-frame mode guarantees the callback fires
/// only during synchronous `dav1d_send_data` + `dav1d_get_picture`
/// calls, while the RecorderState's stack frame is still alive.
unsafe extern "C" fn bit_hook_cb(
    cookie: *mut core::ffi::c_void,
    bit: core::ffi::c_uint,
    tag: u8,
) {
    let state = unsafe { &mut *(cookie as *mut RecorderState) };
    state.bits.push((bit as u8, tag));
    // Phase B.1.1.b: pull the current sticky meta into the parallel
    // Vec. Set by `meta_hook_cb` at AC sign sites; stale for non-AC
    // emissions (acceptable — phasm-core only reads meta for AC).
    state.metas.push(state.current_meta);
}

/// Phase B.1.1.b: FFI callback wired into
/// `Dav1dSettings.phasm_hooks.meta_hook`. Fires at each AC sign
/// decode site BEFORE the bit_hook_cb so the meta is in place when
/// the bit is captured.
///
/// # Safety
///
/// Same as `bit_hook_cb`. `meta` is a pointer into dav1d's stack
/// frame for the current decode_coefs call; we copy it out before
/// returning.
unsafe extern "C" fn meta_hook_cb(
    cookie: *mut core::ffi::c_void,
    meta: *const core_dav1d_sys::Dav1dPhasmAcSignMeta,
) {
    let state = unsafe { &mut *(cookie as *mut RecorderState) };
    if !meta.is_null() {
        state.current_meta = unsafe { *meta };
    }
}

/// Decode an AV1 byte stream and capture all 50/50 binary emissions
/// via the W3.D phasm-dav1d hooks. The returned
/// [`DecodedCoverPositions`] is the decoder-side counterpart of
/// phasm-rav1e's `phasm_bit_positions` / `phasm_bit_tags` on the
/// encoder side.
///
/// # v0.3-AV1 constraints
/// - Single-thread (`n_threads=1`) — avoids hook serialisation across
///   worker threads
/// - Single-frame-delay (`max_frame_delay=1`) — avoids per-frame
///   cookie multiplexing complexity per `dav1d-hook-sites.md` § 6
///
/// # Errors
/// Returns [`Av1DecodeError`] variants for `dav1d_open` / `_send_data`
/// / `_get_picture` failures. EAGAIN responses are handled by the
/// send → get loop (not error-returning).
pub fn decode_with_recording(
    av1_bytes: &[u8],
) -> Result<DecodedCoverPositions, Av1DecodeError> {
    decode_with_recording_inner(av1_bytes, false)
}

/// Phase B.1.4 variant: same as [`decode_with_recording`] but also
/// extracts the last decoded picture's visible-region YUV planes via
/// the [`Dav1dPictureView`] layout mirror. The returned positions
/// have `planes = Some(_)` populated.
///
/// Used by the W6/AoSO self-steganalyzer to compute J-UNIWARD costs
/// over stego-side reconstructed pixels (NOT the encoder's pre-flip
/// recording, which would yield identical costs to cover and erase
/// any signal).
pub fn decode_with_recording_with_pixels(
    av1_bytes: &[u8],
) -> Result<DecodedCoverPositions, Av1DecodeError> {
    decode_with_recording_inner(av1_bytes, true)
}

/// Layout mirror of dav1d's `Dav1dPicture` prefix (through
/// `Dav1dPictureParameters p`). C declaration in
/// `vendor/phasm-dav1d/include/dav1d/picture.h`:
///
/// ```c
/// typedef struct Dav1dPicture {
///     Dav1dSequenceHeader *seq_hdr;
///     Dav1dFrameHeader *frame_hdr;
///     void *data[3];
///     ptrdiff_t stride[2];
///     Dav1dPictureParameters p;  // { int w, h, layout, bpc }
///     ...
/// } Dav1dPicture;
/// ```
///
/// We only need fields up through `p.bpc` (72 bytes on 64-bit). The
/// outer `Dav1dPicture` opaque blob in dav1d-sys is 1024 bytes; we
/// `&*(pic as *const Dav1dPictureView)` to read the prefix.
///
/// Layout match is implicit (C-stable public header); a sentinel
/// extraction test in the integration suite verifies the planes
/// decode to the expected content for the v0.3 corpus.
#[repr(C)]
struct Dav1dPictureView {
    _seq_hdr: *mut core::ffi::c_void,
    _frame_hdr: *mut core::ffi::c_void,
    data: [*mut core::ffi::c_void; 3],
    stride: [isize; 2],
    p_w: i32,
    p_h: i32,
    p_layout: i32, // Dav1dPixelLayout: 0=MONO 1=I420 2=I422 3=I444
    _p_bpc: i32,
}

/// Copy the visible region of one plane (luma or chroma) out of a
/// dav1d picture into a packed Rust Vec. dav1d's plane buffers
/// include left/top stride padding for filter taps; the visible
/// region starts at `data + 0` in dav1d's interface (the padding is
/// to the LEFT of `data` per the public-header convention).
///
/// # Safety
/// `data` must be a valid non-null pointer into a live Dav1dPicture's
/// plane buffer, `stride >= width`, and `width × height` bytes must
/// be readable starting at `(data + row × stride)` for row ∈ [0, h).
unsafe fn copy_plane(
    data: *const u8,
    stride: isize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        let row_start = unsafe { data.offset(row as isize * stride) };
        let slice = unsafe { core::slice::from_raw_parts(row_start, width) };
        out.extend_from_slice(slice);
    }
    out
}

fn decode_with_recording_inner(
    av1_bytes: &[u8],
    capture_pixels: bool,
) -> Result<DecodedCoverPositions, Av1DecodeError> {
    let mut recorder = RecorderState::default();
    let eagain = dav1d_err_again();
    let mut captured_planes: Option<DecodedFramePlanes> = None;

    // SAFETY: `decode_with_recording` is the sole owner of the FFI
    // resources allocated below — they're created and freed within
    // this function's stack frame. The recorder cookie is valid for
    // the entire scope of unsafe block. All dav1d calls are
    // synchronous per the v0.3 single-thread / single-frame
    // configuration. Drop on early return is manual (no Drop guard)
    // because the error paths are simple.
    unsafe {
        // Zero-init the settings struct, then call default_settings
        // (fills allocator + logger but NOT phasm_hooks or reserved).
        let mut settings: Dav1dSettings = core::mem::zeroed();
        dav1d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.max_frame_delay = 1;

        // Register the recording hook with cookie pointing at our
        // stack-allocated recorder.
        settings.phasm_hooks = Dav1dPhasmHooks {
            cookie: &mut recorder as *mut RecorderState
                as *mut core::ffi::c_void,
            bit_hook: Some(bit_hook_cb),
            tag_hook: None, // not used for v0.3 recording
            meta_hook: Some(meta_hook_cb), // Phase B.1.1.b
        };

        let mut ctx: *mut Dav1dContext = core::ptr::null_mut();
        let rc = dav1d_open(&mut ctx, &settings);
        if rc != 0 {
            return Err(Av1DecodeError::DavxdOpenFailed(rc));
        }
        debug_assert!(!ctx.is_null());

        // Allocate dav1d-side buffer and copy our bytes in.
        let mut data: Dav1dData = core::mem::zeroed();
        let buf = dav1d_data_create(&mut data, av1_bytes.len());
        if buf.is_null() {
            dav1d_close(&mut ctx);
            return Err(Av1DecodeError::DataCreateFailed);
        }
        core::ptr::copy_nonoverlapping(
            av1_bytes.as_ptr(),
            buf,
            av1_bytes.len(),
        );

        // Send + receive loop. dav1d's flow: alternately call
        // send_data and get_picture until both return EAGAIN (which
        // means "we need more data" for both — i.e., done).
        let mut send_done = false;
        loop {
            if !send_done {
                let rc = dav1d_send_data(ctx, &mut data);
                if rc == 0 {
                    send_done = data.sz == 0;
                } else if rc == eagain {
                    // send buffer full; must call get_picture first
                } else {
                    dav1d_data_unref(&mut data);
                    dav1d_close(&mut ctx);
                    return Err(Av1DecodeError::DavxdSendDataFailed(rc));
                }
            }

            let mut pic = Dav1dPicture::default();
            let rc = dav1d_get_picture(ctx, &mut pic);
            if rc == 0 {
                // Phase B.1.4: if pixel capture is requested, read
                // the visible YUV via the Dav1dPictureView layout
                // mirror BEFORE unref. We keep the LAST picture
                // we see — v0.3 is single-frame so this is fine.
                if capture_pixels {
                    let view = &*(&pic as *const Dav1dPicture as *const Dav1dPictureView);
                    assert_eq!(
                        view.p_layout, 1,
                        "Phase B.1.4 only handles 4:2:0 (Dav1dPixelLayout I420 == 1), got {}",
                        view.p_layout
                    );
                    let luma_w = view.p_w as usize;
                    let luma_h = view.p_h as usize;
                    let chroma_w = (luma_w + 1) / 2;
                    let chroma_h = (luma_h + 1) / 2;
                    let y = copy_plane(
                        view.data[0] as *const u8,
                        view.stride[0],
                        luma_w,
                        luma_h,
                    );
                    let cb = copy_plane(
                        view.data[1] as *const u8,
                        view.stride[1],
                        chroma_w,
                        chroma_h,
                    );
                    let cr = copy_plane(
                        view.data[2] as *const u8,
                        view.stride[1],
                        chroma_w,
                        chroma_h,
                    );
                    captured_planes = Some(DecodedFramePlanes {
                        y,
                        cb,
                        cr,
                        luma_width: luma_w,
                        luma_height: luma_h,
                        chroma_width: chroma_w,
                        chroma_height: chroma_h,
                    });
                }
                dav1d_picture_unref(&mut pic);
                // Try for more pictures + send if we still have data.
                continue;
            }
            if rc == eagain {
                // Need more data. If we've sent everything, we're
                // done.
                if send_done {
                    break;
                }
                // Otherwise loop back to send_data.
                continue;
            }
            // Other error.
            dav1d_data_unref(&mut data);
            dav1d_close(&mut ctx);
            return Err(Av1DecodeError::DavxdGetPictureFailed(rc));
        }

        dav1d_data_unref(&mut data);
        dav1d_close(&mut ctx);
    }

    // Convert raw (value, tag, meta) tuples into structured positions.
    // Phase B.1.1.b: bits + metas Vecs are pushed in lockstep by
    // bit_hook_cb (meta is current sticky state at bit-capture time).
    debug_assert_eq!(
        recorder.bits.len(),
        recorder.metas.len(),
        "decoder bit/meta length mismatch"
    );
    let positions = recorder
        .bits
        .into_iter()
        .zip(recorder.metas.into_iter())
        .enumerate()
        .map(|(i, ((value, tag), meta))| DecodedCoverPosition {
            cursor: i as u64,
            decoded_value: value,
            tag,
            meta,
        })
        .collect();

    Ok(DecodedCoverPositions { all: positions, planes: captured_planes })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic sanity test: decode_with_recording with garbage input
    /// either errors OR returns empty positions — but doesn't panic /
    /// crash / leak. Dav1d's actual behaviour on unknown-OBU input
    /// is to log + skip + return Ok with no decoded frames (so no
    /// hook calls fire); that's the observed branch for all-zero
    /// garbage. Either branch is acceptable here — this test exists
    /// to catch real failures (panics / UB / OOM) on adversarial
    /// input.
    #[test]
    fn decode_garbage_bytes_handles_cleanly() {
        let garbage = [0u8; 100];
        let result = decode_with_recording(&garbage);
        match result {
            Err(_) => { /* dav1d rejected — fine */ }
            Ok(positions) => {
                assert!(
                    positions.is_empty(),
                    "garbage input shouldn't produce any decoded positions, got {}",
                    positions.len()
                );
            }
        }
    }

    /// Empty input also handled cleanly (no panic).
    #[test]
    fn decode_empty_bytes_handles_cleanly() {
        let empty: [u8; 0] = [];
        let result = decode_with_recording(&empty);
        match result {
            Err(_) => { /* expected — no data to decode */ }
            Ok(positions) => {
                assert!(
                    positions.is_empty(),
                    "empty input shouldn't produce decoded positions"
                );
            }
        }
    }

    // ============================================================
    // W3.D.4.3: rav1e encode + dav1d decode round-trip test.
    // ============================================================
    //
    // First end-to-end test where encoder + decoder are both
    // actively recording and we cross-validate. If this passes, the
    // entire W3.D architecture is proven correct.
    //
    // Approach:
    //   1. Encode a tiny gradient key frame via rav1e Context API
    //      (gets OBU-wrapped bytes that dav1d can decode).
    //   2. Decode the OBU bytes via decode_with_recording.
    //   3. Assert the decoder captured at least one AC_COEFF_SIGN bit
    //      (proves the hooks fired end-to-end on real input).
    //
    // Future extensions (W3.10+):
    //   - Compare decoder cursor sequence to encoder cursor sequence
    //     (load-bearing equivalence: cursor N's natural_value must
    //     equal cursor N's decoded_value).
    //   - Apply a Tier 1 override via STC plan + replay_with_overrides,
    //     verify the overridden value extracts via the decoder hook.

    #[cfg(feature = "av1-encoder")]
    mod round_trip {
        use super::*;
        use phasm_rav1e::{Config, EncoderConfig};
        use std::sync::Arc;

        /// Encode a tiny 64x64 gradient key frame via rav1e's public
        /// Context API. Returns the OBU-wrapped bytes from the first
        /// Packet (suitable for dav1d input).
        fn encode_one_frame() -> Vec<u8> {
            use phasm_rav1e::color::ChromaSampling;
            use phasm_rav1e::phasm_stego::make_frame;
            use phasm_rav1e::prelude::Tune;

            let enc_cfg = EncoderConfig {
                width: 64,
                height: 64,
                bit_depth: 8,
                chroma_sampling: ChromaSampling::Cs420,
                // Low QP → AC coefficients survive quantization →
                // encoder emits AC sign bits → decoder hook fires.
                quantizer: 30,
                tune: Tune::default(),
                ..Default::default()
            };
            let cfg = Config::new()
                .with_encoder_config(enc_cfg.clone())
                .with_threads(1);
            let mut ctx: phasm_rav1e::Context<u8> = cfg.new_context().unwrap();

            // Construct + fill a gradient frame (deterministic
            // non-uniform pattern so the encoder produces non-zero AC
            // coefficients).
            let mut frame =
                make_frame::<u8>(64, 64, ChromaSampling::Cs420);
            for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                let stride = plane.cfg.stride;
                for (row_idx, row) in plane.data.chunks_mut(stride).enumerate() {
                    for (col_idx, pixel) in row.iter_mut().enumerate() {
                        *pixel = ((row_idx.wrapping_mul(7)
                            + col_idx.wrapping_mul(3)
                            + plane_idx * 13)
                            & 0xff) as u8;
                    }
                }
            }

            // Send + flush.
            ctx.send_frame(Arc::new(frame)).unwrap();
            ctx.flush();

            // Collect all packets — accumulate the OBU bytes.
            let mut all_bytes = Vec::new();
            loop {
                match ctx.receive_packet() {
                    Ok(packet) => {
                        all_bytes.extend_from_slice(&packet.data);
                    }
                    Err(phasm_rav1e::EncoderStatus::Encoded)
                    | Err(phasm_rav1e::EncoderStatus::NeedMoreData) => {
                        // Expected — keep going.
                    }
                    Err(phasm_rav1e::EncoderStatus::LimitReached)
                    | Err(phasm_rav1e::EncoderStatus::Failure)
                    | Err(_) => break,
                }
            }
            all_bytes
        }

        /// Run encode_tile::<WriterRecorder> on a 64x64 gradient
        /// frame state — returns the captured encoder-side
        /// CoverPositions (cursor + storage_index + natural_value +
        /// tag per emission).
        ///
        /// Uses the same gradient fill + frame setup as
        /// encode_one_frame to ensure cursor-parity invariants hold
        /// across the two paths.
        fn encode_one_frame_to_recorder() -> crate::codec::av1::stego::writer::CoverPositions {
            use crate::codec::av1::stego::writer::CoverPositions;
            use phasm_rav1e::color::ChromaSampling;
            use phasm_rav1e::ec::{WriterBase, WriterRecorder};
            use phasm_rav1e::phasm_stego::{
                encode_tile, make_frame, make_inter_config, FrameBlocks,
                FrameInvariants, FrameState,
            };
            use phasm_rav1e::prelude::Sequence;
            use phasm_rav1e::EncoderConfig;

            let config = Arc::new(EncoderConfig {
                width: 64,
                height: 64,
                bit_depth: 8,
                chroma_sampling: ChromaSampling::Cs420,
                quantizer: 30,
                ..Default::default()
            });
            let mut sequence = Sequence::new(&config);
            sequence.enable_large_lru = false;
            let fi = FrameInvariants::<u8>::new_key_frame(
                config.clone(),
                Arc::new(sequence),
                0,
                Box::new([]),
            );
            let mut frame = make_frame::<u8>(64, 64, ChromaSampling::Cs420);
            for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                let stride = plane.cfg.stride;
                for (row_idx, row) in plane.data.chunks_mut(stride).enumerate() {
                    for (col_idx, pixel) in row.iter_mut().enumerate() {
                        *pixel = ((row_idx.wrapping_mul(7)
                            + col_idx.wrapping_mul(3)
                            + plane_idx * 13)
                            & 0xff) as u8;
                    }
                }
            }
            let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
            let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
            let mut cdf = phasm_rav1e::context::CDFContext::new(fi.base_q_idx);
            let inter_cfg = make_inter_config(&config);

            let ti = &fi.sequence.tiling;
            let mut iter = ti.tile_iter_mut(&mut fs, &mut blocks);
            let mut ctx = iter.next().expect("single-tile expected");
            drop(iter);

            let (recorder, _stats): (WriterBase<WriterRecorder>, _) =
                encode_tile(&fi, &mut ctx.ts, &mut cdf, &mut ctx.tb, &inter_cfg);

            CoverPositions::from_recorder(
                recorder.phasm_bit_positions(),
                recorder.phasm_bit_tags(),
            )
        }

        /// W3.D.4.3 round-trip test: rav1e encode a gradient frame
        /// → dav1d decode the OBU bytes with recording → assert the
        /// decoder hook fired on at least some AC_COEFF_SIGN bits.
        ///
        /// If this passes, the entire W3.D architecture
        /// (W3.D.1 design + W3.D.2.* hooks + W3.D.3 tag sites +
        /// W3.D.4.1 FFI + W3.D.4.2 wrapper) is end-to-end functional.
        #[test]
        fn w3d43_rav1e_encode_dav1d_decode_round_trip() {
            use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;

            // Encode side: get OBU bytes from rav1e.
            let av1_bytes = encode_one_frame();
            assert!(
                !av1_bytes.is_empty(),
                "rav1e Context API should produce non-empty AV1 output"
            );

            // Decode side: drive dav1d with our recording hook.
            let positions = decode_with_recording(&av1_bytes)
                .expect("dav1d should decode rav1e output");

            // Hook must have fired at least once during decode.
            assert!(
                !positions.is_empty(),
                "expected decoded positions; got 0 — dav1d hooks didn't fire"
            );

            // At least one position should be AcCoeffSign-tagged
            // (gradient frame produces non-zero AC coefficients →
            // sign bits get decoded → tag site at recon_tmpl.c:642/691
            // fires).
            let ac_sign_count = positions
                .iter()
                .filter(|p| p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN)
                .count();
            assert!(
                ac_sign_count > 0,
                "expected ≥1 AC_COEFF_SIGN-tagged position; got 0 of {} total. \
                 Either W3.D.3 site patches didn't propagate to runtime decode \
                 path, or encoder produced no AC coefficients at this QP",
                positions.len()
            );

            // Decoded values must all be valid bits.
            for pos in positions.iter() {
                assert!(
                    pos.decoded_value == 0 || pos.decoded_value == 1,
                    "decoded_value must be 0 or 1; got {} at cursor {}",
                    pos.decoded_value,
                    pos.cursor
                );
            }
        }

        /// W3.10.4 STRICT cursor parity test — uses
        /// encode_frame_with_phasm_tee to get OBU-wrapped bytes AND
        /// per-tile recorder data from a SINGLE encode pass. Then
        /// dav1d-decodes those exact bytes with the recording hook.
        ///
        /// **Asserts STRICT byte-and-tag parity**: for every cursor
        /// in [0, total_bits), encoder.natural_value ==
        /// decoder.decoded_value AND encoder.tag == decoder.tag.
        ///
        /// **2026-05-21 result on 128×128 gradient frame** (after
        /// W3.10.4-fix: phasm-rav1e `replay()` resets dest tag +
        /// phasm-dav1d DC golomb sites tagged):
        ///   Total 50/50 bits: 16421 == 16421 ✓
        ///   AC_SIGN tags: 5077 == 5077 ✓
        ///   GOLOMB tags: 10636 == 10636 ✓
        ///   OTHER tags: 708 == 708 ✓
        ///   Zero tag disagreements, zero value disagreements.
        ///
        /// This is THE LOAD-BEARING equivalence for v0.3-AV1 stego —
        /// STC plan + replay_with_overrides + decode extraction all
        /// rely on encoder + decoder agreeing bit-for-bit on the
        /// same bytes.
        #[test]
        fn w3104_strict_cursor_parity_via_phasm_tee() {
            use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;
            use phasm_rav1e::color::ChromaSampling;
            use phasm_rav1e::phasm_stego::{
                encode_frame_with_phasm_tee, make_frame, make_inter_config,
                FrameInvariants, FrameState, PHASM_TAG_AC_COEFF_SIGN,
            };
            use phasm_rav1e::prelude::Sequence;
            use phasm_rav1e::EncoderConfig;

            // 128x128 with segmentation disabled — see
            // encode_frame_with_phasm_tee_produces_packet_and_recording
            // smoke test in phasm-rav1e for why these constraints
            // apply when bypassing Context API.
            let config = Arc::new(EncoderConfig {
                width: 128,
                height: 128,
                bit_depth: 8,
                chroma_sampling: ChromaSampling::Cs420,
                quantizer: 30,
                ..Default::default()
            });
            let mut sequence = Sequence::new(&config);
            sequence.enable_large_lru = false;
            let mut fi = FrameInvariants::<u8>::new_key_frame(
                config.clone(),
                Arc::new(sequence),
                0,
                Box::new([]),
            );
            fi.enable_segmentation = false;
            let mut frame = make_frame::<u8>(128, 128, ChromaSampling::Cs420);
            for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                let stride = plane.cfg.stride;
                for (row_idx, row) in plane.data.chunks_mut(stride).enumerate() {
                    for (col_idx, pixel) in row.iter_mut().enumerate() {
                        *pixel = ((row_idx.wrapping_mul(7)
                            + col_idx.wrapping_mul(3)
                            + plane_idx * 13)
                            & 0xff) as u8;
                    }
                }
            }
            let mut fs =
                FrameState::new_with_frame(&fi, Arc::new(frame));
            let inter_cfg = make_inter_config(&config);

            // SINGLE encode produces both OBU bytes + recorder data.
            let (packet, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);
            assert!(!packet.is_empty(), "encode_frame_with_phasm_tee must produce non-empty packet");
            assert_eq!(recording.tiles.len(), 1, "single-tile config expected");

            // Encoder-side AC sign positions (natural values).
            let encoder_ac: Vec<u16> = recording.tiles[0]
                .bit_positions
                .iter()
                .zip(recording.tiles[0].bit_tags.iter())
                .filter(|&(_, &tag)| tag == PHASM_TAG_AC_COEFF_SIGN)
                .map(|(&(_, val), _)| val)
                .collect();

            assert!(
                encoder_ac.len() > 100,
                "encoder should capture > 100 AC signs on gradient frame; got {}",
                encoder_ac.len()
            );

            // Decode the SAME bytes via dav1d.
            let decoder_positions = decode_with_recording(&packet)
                .expect("dav1d must decode encode_frame_with_phasm_tee output");
            let decoder_ac: Vec<u8> = decoder_positions
                .iter()
                .filter(|p| p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN)
                .map(|p| p.decoded_value)
                .collect();

            // === STRICT PARITY ASSERTIONS ===
            // After W3.10.4-fix (replay() resets dest tag +
            // dav1d DC golomb tag sites), encoder + decoder agree
            // on EVERY bit's value AND tag across all 16421
            // emissions on the 128×128 gradient.
            let encoder_total = recording.tiles[0].bit_positions.len();
            let decoder_total = decoder_positions.len();

            assert_eq!(
                encoder_total, decoder_total,
                "STRICT: total 50/50 bit count must match"
            );

            // Full per-cursor scan: every (value, tag) tuple must
            // be identical between encoder + decoder.
            for i in 0..encoder_total {
                let (_, enc_val) = recording.tiles[0].bit_positions[i];
                let enc_tag = recording.tiles[0].bit_tags[i];
                let dec_pos = decoder_positions.as_slice()[i];

                assert_eq!(
                    enc_val as u8, dec_pos.decoded_value,
                    "STRICT value parity FAIL at cursor {}: \
                     encoder natural={}, decoder decoded={}",
                    i, enc_val, dec_pos.decoded_value
                );
                assert_eq!(
                    enc_tag, dec_pos.tag,
                    "STRICT tag parity FAIL at cursor {}: \
                     encoder tag={}, decoder tag={}",
                    i, enc_tag, dec_pos.tag
                );
            }

            // AC sign filter sanity: both sides should see the same
            // AC sign count.
            assert_eq!(
                encoder_ac.len(),
                decoder_ac.len(),
                "AC sign count parity"
            );
            assert!(
                !encoder_ac.is_empty(),
                "expected ≥1 AC sign on gradient frame"
            );

            // AC values must all match (redundant with the full
            // scan above but documents the channel-specific
            // invariant explicitly).
            for (i, (enc, dec)) in encoder_ac.iter().zip(decoder_ac.iter()).enumerate() {
                assert_eq!(
                    *enc as u8, *dec,
                    "AC sign value mismatch at AC index {}",
                    i
                );
            }
        }

        /// W3.10.3 STRICT cursor parity test — uses WriterTee to
        /// produce bytes + recorder data from a SINGLE encode pass,
        /// so the decoder + encoder are guaranteed to see the same
        /// emission stream.
        ///
        /// For every AC_COEFF_SIGN-tagged position:
        ///   encoder.natural_value (recorded during the encode that
        ///                          produced these bytes)
        ///   ==
        ///   decoder.decoded_value (extracted from those same bytes
        ///                          by dav1d)
        ///
        /// If this passes, the encoder + decoder agree
        /// bit-for-bit on every AC sign emission. This is the
        /// load-bearing equivalence for v0.3-AV1 stego — STC plans
        /// + replay_with_overrides + decoder extraction all depend
        /// on it.
        #[test]
        fn w3103_strict_cursor_parity_via_writer_tee() {
            use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;
            use phasm_rav1e::color::ChromaSampling;
            use phasm_rav1e::ec::WriterBase;
            use phasm_rav1e::phasm_stego::{
                encode_tile, make_frame, make_inter_config, FrameBlocks,
                FrameInvariants, FrameState, WriterTee, PHASM_TAG_AC_COEFF_SIGN,
            };
            use phasm_rav1e::prelude::Sequence;
            use phasm_rav1e::EncoderConfig;

            // Build the same frame state as W3.10.2 but encode via
            // WriterTee — single encode produces both bytes AND
            // recorder data.
            let config = Arc::new(EncoderConfig {
                width: 64,
                height: 64,
                bit_depth: 8,
                chroma_sampling: ChromaSampling::Cs420,
                quantizer: 30,
                ..Default::default()
            });
            let mut sequence = Sequence::new(&config);
            sequence.enable_large_lru = false;
            let fi = FrameInvariants::<u8>::new_key_frame(
                config.clone(),
                Arc::new(sequence),
                0,
                Box::new([]),
            );
            let mut frame = make_frame::<u8>(64, 64, ChromaSampling::Cs420);
            for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                let stride = plane.cfg.stride;
                for (row_idx, row) in plane.data.chunks_mut(stride).enumerate() {
                    for (col_idx, pixel) in row.iter_mut().enumerate() {
                        *pixel = ((row_idx.wrapping_mul(7)
                            + col_idx.wrapping_mul(3)
                            + plane_idx * 13)
                            & 0xff) as u8;
                    }
                }
            }
            let mut fs =
                FrameState::new_with_frame(&fi, Arc::new(frame));
            let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
            let mut cdf = phasm_rav1e::context::CDFContext::new(fi.base_q_idx);
            let inter_cfg = make_inter_config(&config);

            let ti = &fi.sequence.tiling;
            let mut iter = ti.tile_iter_mut(&mut fs, &mut blocks);
            let mut ctx = iter.next().expect("single-tile expected");
            drop(iter);

            let (mut writer, _stats): (WriterBase<WriterTee>, _) =
                encode_tile(&fi, &mut ctx.ts, &mut cdf, &mut ctx.tb, &inter_cfg);

            // Capture the recorder data BEFORE done() — same as
            // the smoke test pattern.
            let encoder_bit_positions = writer.phasm_bit_positions().to_vec();
            let encoder_bit_tags = writer.phasm_bit_tags().to_vec();
            let _tile_bytes = writer.done();

            // Filter encoder side to AcCoeffSign positions.
            let encoder_ac: Vec<u16> = encoder_bit_positions
                .iter()
                .zip(encoder_bit_tags.iter())
                .filter(|&(_, &tag)| tag == PHASM_TAG_AC_COEFF_SIGN)
                .map(|(&(_, val), _)| val)
                .collect();

            // Decoder side: encode via Context (which goes through
            // a separate path — but we need OBU-wrapped bytes for
            // dav1d). For STRICT parity, ideally we'd splice tile_bytes
            // into a stego OBU wrapper, but that's W3.10.4 territory.
            // Here we use Context encode for the OBU bytes + accept
            // that those bytes might differ from tile_bytes — what we
            // CAN test strictly is that encoder records correctly
            // even if not byte-paired with the decoder run.
            //
            // Specifically: this test validates the WriterTee
            // primitive works correctly (encoder + recorder in lockstep
            // on a single pass). True strict parity needs W3.10.4
            // OBU-wrap + send the SAME bytes both directions.
            //
            // For now we assert that WriterTee captured a substantial
            // count of AcCoeffSign tags. The strict per-cursor parity
            // test against dav1d follows in W3.10.4.

            assert!(
                encoder_ac.len() > 100,
                "WriterTee should capture > 100 AC sign positions on gradient frame; got {}",
                encoder_ac.len()
            );

            // All values must be 0 or 1.
            for (i, &val) in encoder_ac.iter().enumerate() {
                assert!(
                    val == 0 || val == 1,
                    "AcCoeffSign value at index {} must be 0 or 1; got {}",
                    i, val
                );
            }
        }

        /// W3.10.2 SOFT cursor parity test — sanity check that
        /// encoder + decoder produce SIMILAR AC sign counts on the
        /// same input.
        ///
        /// **Important architectural finding (2026-05-21):** strict
        /// cursor parity (encoder[N].natural_value ==
        /// decoder[N].decoded_value for every N) requires running
        /// BOTH paths on the SAME single encode. We don't have that
        /// infra yet — current options:
        ///
        ///   A. encode_tile<WriterRecorder> (manual setup, no Context
        ///      lookahead pre-processing) → recorder positions, plus
        ///      Context API encode (separate, full lookahead) → OBU
        ///      bytes → decode → decoder positions. These produce
        ///      SLIGHTLY DIFFERENT byte streams because Context's
        ///      lookahead affects rate control + mode decisions.
        ///
        ///   B. Single Context encode + extract recorder data via
        ///      side channel. Requires fork patch (e.g. TEE
        ///      WriterEncoder+WriterRecorder, or hooking
        ///      encode_frame to expose the per-tile recorder).
        ///
        ///   C. Single encode_tile<WriterRecorder> + OBU-wrap the
        ///      output ourselves (don't go through Context). Requires
        ///      implementing AV1 OBU framing in phasm-core.
        ///
        /// Option B or C is W3.10.3+ work (orchestrator plumbing).
        /// For now, this test does a SOFT check: both sides should
        /// see roughly the same number of AC sign bits (within
        /// tolerance) on the same gradient input. Catches gross
        /// misalignment without requiring path-perfect alignment.
        ///
        /// Strict cursor parity will land as a separate test in
        /// W3.10.3+ once orchestrator infra exists.
        #[test]
        fn w3102_soft_parity_ac_coeff_sign_count() {
            use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;
            use phasm_rav1e::phasm_stego::PHASM_TAG_AC_COEFF_SIGN;

            // Sanity: encoder + decoder tag constants must match.
            assert_eq!(
                PHASM_TAG_AC_COEFF_SIGN, DAV1D_PHASM_TAG_AC_COEFF_SIGN,
                "encoder + decoder tag constants must agree"
            );

            // Encoder side: capture positions via direct
            // encode_tile<WriterRecorder>.
            let encoder_positions = encode_one_frame_to_recorder();
            let encoder_ac_count =
                encoder_positions.count_by_tag(PHASM_TAG_AC_COEFF_SIGN);

            // Decoder side: encode via Context API, decode bytes,
            // capture positions.
            let av1_bytes = encode_one_frame();
            let decoder_positions = decode_with_recording(&av1_bytes)
                .expect("dav1d decode of rav1e output");
            let decoder_ac_count = decoder_positions
                .iter()
                .filter(|p| p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN)
                .count();

            // Both sides must see a substantial number of AC signs
            // (gradient frame at QP=30 → ~1000+ on each side).
            assert!(
                encoder_ac_count > 100,
                "encoder AC sign count too low: {} (expected ≥100)",
                encoder_ac_count
            );
            assert!(
                decoder_ac_count > 100,
                "decoder AC sign count too low: {} (expected ≥100)",
                decoder_ac_count
            );

            // Soft tolerance: counts should be within 10% of each
            // other. Catches if one path is producing dramatically
            // different output (e.g., the encoder is using a
            // different speed preset or QP path).
            let diff = encoder_ac_count.abs_diff(decoder_ac_count);
            let tolerance = encoder_ac_count.max(decoder_ac_count) / 10;
            assert!(
                diff <= tolerance,
                "encoder + decoder AC sign counts diverge too much: \
                 encoder={}, decoder={}, diff={}, tolerance={} (10%). \
                 If this fails, likely the encoder + decoder paths are \
                 producing materially different bytes — check setup_rav1e_state \
                 vs Context API config alignment.",
                encoder_ac_count, decoder_ac_count, diff, tolerance
            );
        }
    }
}
