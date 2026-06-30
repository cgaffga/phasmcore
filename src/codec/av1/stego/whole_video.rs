// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Whole-video shadow encode flow.
//!
//! Ported from production OH264's
//! `h264_encode_with_shadows` (see
//! `core/src/codec/h264/openh264_stego.rs`). AV1-simplified:
//! `WriterRecorder` + `replay_with_overrides` is wire-clean by
//! construction (encoder state never sees stego flips), so the
//! provisional Pass 2 the OH264 path needs is elided. Pass 1
//! produces both natural OBU bytes AND the cover the decoder will
//! see; the cascade loop runs once-through.
//!
//! Design: see
//! `docs/design/video/av1/phase-c-wv-whole-video-shadow.md` § 3.2.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::{PhasmFrameRecording, WriterEncoder};
use phasm_rav1e::phasm_stego::{
    encode_gop_with_phasm_tee, make_frame, AcSignMeta,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

/// Phasm AV1 production speed preset.
///
/// Default: **`speed = 9`** (locked in by the Layer-3 stealth audit
/// 2026-06-29 — see `docs/design/video/av1/stealth-audit-speed-preset-2026-06-29.md`).
/// rav1e's natural default is 6; phasm pins higher because the audit
/// showed speed=9 has strictly better Layer-3 fingerprint alignment
/// against rav1e CLI on partition + mode axes (and pays only a marginal
/// MV-KS uptick), AND buys ~1.9× wall-clock vs speed=6.
///
/// Override with `PHASM_AV1_SPEED=N` env var (0..=10) for experiments.
/// The CLI exposes `--av1-speed N` which sets this env var. Used by:
///   - `encode_gop_natural` (this file; Pass 1 + cascade re-encode)
///   - `session::encode_one_keyframe` (single-frame GOP, no-shadow path)
///   - `session::encode_one_gop_multi` (multi-frame GOP, no-shadow path)
///
/// `multiref` defaults off (kept off for stealth-profile stability +
/// the single-pass STC plan). The `PHASM_AV1_MULTIREF=1` env knob
/// flips it to true for the 2026-06-29 stealth pre-experiment (see
/// `av1-stealth-lookahead-plan-2026-06-29.md` § 2 / task #231) — not
/// for production use. Production default unchanged.
pub(super) fn apply_phasm_av1_speed(config: &mut EncoderConfig) {
    let speed = std::env::var("PHASM_AV1_SPEED")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(9)
        .min(10);
    config.speed_settings = phasm_rav1e::prelude::SpeedSettings::from_preset(speed);
    let multiref = std::env::var("PHASM_AV1_MULTIREF")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|n| n != 0)
        .unwrap_or(false);
    config.speed_settings.multiref = multiref;
}

/// Phasm AV1 production sequence-level settings.
///
/// `enable_large_lru` default: **false** (current production state).
/// The 2026-06-29 stealth audit identified `enable_large_lru = false`
/// as the only sequence-level knob that diverges from rav1e CLI without
/// a CLI flag to replicate — making it a likely contributor to phasm's
/// pre-existing s6-vs-CLI partition divergence. The flip to `true` is
/// being measured in a follow-on audit; keep default `false` until that
/// completes to preserve the existing stealth posture.
///
/// Override with `PHASM_AV1_LARGE_LRU=1` (or `0` for explicit-default
/// behaviour, useful inside experiment harnesses).
///
/// Called by every encoder construction site that builds a fresh
/// `Sequence`.
pub(super) fn apply_phasm_av1_sequence(sequence: &mut Sequence) {
    let enable = std::env::var("PHASM_AV1_LARGE_LRU")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|n| n != 0)
        .unwrap_or(false);
    sequence.enable_large_lru = enable;
}

use crate::codec::av1::stego::session::Av1StreamingEncodeParams;
use crate::codec::av1::stego::shadow::{
    finalize_shadow_state_av1, prepare_shadow_pre_selection_av1,
    Av1ShadowSelectionSweep, Av1ShadowState, Av1StreamingShadowVerify,
    AV1_SHADOW_PARITY_TIERS,
};
use crate::codec::av1::stego::writer::{replay_with_overrides, OverrideMap};
use crate::stego::chunk_frame::{
    build_chunk_frame_v3_1, build_first_chunk_frame_v3_1, split_message_into_chunks,
};
use crate::stego::cost::av1_uniward::{
    compute_av1_uniward_costs_with_state, Av1FramePosition, FramePlanes,
};
use crate::stego::stc::embed::stc_embed;
use crate::stego::stc::hhat;
use crate::stego::{frame, payload};

use super::orchestrator::{
    av1_stego_embed_payload_bits, av1_stego_encode_one_gop, pack_visible_planes,
    rebuild_obu_with_stego_tile_group, Av1StegoError,
};
use super::session::{encode_one_gop_multi, encode_one_keyframe};

const STC_H: usize = 7;

/// Encode a whole video with shadows spread across all GOPs
/// (whole-video shadow scope per `phase-c-shadows.md` § 2).
///
/// `yuv` is the packed tight I420 buffer for ALL `n_frames` frames
/// (frame-major). `primary_framed` is the pre-built primary frame
/// (encrypt + frame::build_frame already done by the caller — the
/// session does this at `create_whole_video_with_shadows`).
///
/// Returns the concatenated Annex-B-equivalent stego AV1 bytes for
/// every GOP, with the chunk_frame protocol per
/// `phase-c-streaming-session-v6.md` § 8 unchanged from per-GOP
/// scope.
///
/// Cascades through `AV1_SHADOW_PARITY_TIERS` `[4, 8, 16, 32, 64,
/// 128]` until the produced bytes round-trip every shadow under
/// dav1d walk + `av1_shadow_extract`. Returns
/// `Av1StegoError::Stego(StegoError::ShadowEmbedFailed)` if all
/// rungs exhaust.
// ────────────────────────────────────────────────────────────────
// WV.7.0 — forward-streaming source of decoded GOP YUV
// ────────────────────────────────────────────────────────────────

/// Forward-streaming source of decoded GOP YUV for the AV1 streaming
/// shadow encode (WV.7.0). Mirrors H.264's `GopYuvSource` trait — the
/// encoder pulls one GOP at a time, forward, so nothing wider than a
/// single GOP of decoded YUV is ever live in the streaming path.
///
/// Implementations (this commit ships [`Av1SliceYuvSource`] only):
/// - [`Av1SliceYuvSource`] — whole-buffer slicer for the byte-identical
///   gate and any caller that still has the full clip in RAM. Itself
///   O(clip); it's the gated-build stand-in until the bridges supply
///   real per-GOP re-decode sources (WV.7.10 / mobile pull).
///
/// **Contract.** `gop_yuv` is called with monotonically increasing
/// `gop_index` within a sweep; `rewind` restarts the forward stream at
/// GOP 0 between the encoder's two sweeps. The source is never seeked
/// to an arbitrary mid-clip GOP — a backward jump is only ever a full
/// rewind, which native decoders do by re-initialising their reader.
///
/// Returning an empty `Vec` for `gop_index >= n_gops` is the end-of-
/// stream signal (mirror of H.264's contract; lets the caller probe
/// the trailing-partial-GOP boundary without a separate `n_gops`
/// query).
pub trait Av1GopYuvSource {
    /// Tight-I420 YUV for `gop_index`: frames
    /// `[gop_index*gop_size .. min((gop_index+1)*gop_size, n_frames))`,
    /// where `gop_size` / `n_frames` match the encode call. Returned
    /// buffer is `gop_frames * width * height * 3/2` bytes. Empty
    /// `Vec` at end-of-stream.
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError>;
    /// Restart the forward stream at GOP 0 (called between encode
    /// sweeps — once Sweep B wiring lands in WV.7.7).
    fn rewind(&mut self) -> Result<(), Av1StegoError>;
}

/// Whole-buffer [`Av1GopYuvSource`]: slices a contiguous tight-I420
/// clip into per-GOP YUV. Used by the byte-identical gate (this
/// commit) and by callers that already have the whole clip in RAM
/// (the WV.6.h streaming-Pass-1 session for now). It holds the whole
/// clip, so it does NOT itself bound memory — it is the gated-build
/// stand-in that lets the streaming code path be proven byte-identical
/// before the bridge sources land.
pub struct Av1SliceYuvSource<'a> {
    yuv: &'a [u8],
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
    /// Enforces the trait contract: callers must invoke `rewind()`
    /// before the first `gop_yuv()`. Reads are inherently stateless on
    /// a borrowed slice, but the *contract* is shared with stateful
    /// sources (Av1CallbackYuvSource → AVAssetReader/MediaCodec) where
    /// rewind opens the underlying decoder. If a streaming entry like
    /// `av1_encode_streaming` forgets to call `rewind()` first, that
    /// bug must surface in slice-backed byte-identity tests too —
    /// otherwise the stateful FFI sources crash on first device run.
    has_rewound: bool,
}

impl<'a> Av1SliceYuvSource<'a> {
    pub fn new(
        yuv: &'a [u8],
        width: u32,
        height: u32,
        n_frames: u32,
        gop_size: u32,
    ) -> Self {
        Self {
            yuv,
            width,
            height,
            n_frames,
            gop_size: gop_size.max(1),
            has_rewound: false,
        }
    }
}

impl Av1GopYuvSource for Av1SliceYuvSource<'_> {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError> {
        if !self.has_rewound {
            return Err(Av1StegoError::InvalidPacket(
                "Av1SliceYuvSource: gop_yuv called before rewind() — \
                 contract violation that would crash stateful sources \
                 (AVAssetReader / MediaCodec / ffmpeg pipe)"
                    .into(),
            ));
        }
        let frame_bytes = expected_i420_size(self.width, self.height);
        let start_f = gop_index.saturating_mul(self.gop_size);
        if start_f >= self.n_frames {
            return Ok(Vec::new());
        }
        let end_f = ((gop_index + 1) * self.gop_size).min(self.n_frames);
        let start = (start_f as usize) * frame_bytes;
        let end = (end_f as usize) * frame_bytes;
        self.yuv.get(start..end).map(<[u8]>::to_vec).ok_or_else(|| {
            Av1StegoError::InvalidPacket(format!(
                "Av1SliceYuvSource: GOP {gop_index} range {start}..{end} \
                 out of {} bytes",
                self.yuv.len()
            ))
        })
    }

    fn rewind(&mut self) -> Result<(), Av1StegoError> {
        // Reads are stateless — slicing pulls from the borrowed buffer
        // directly. But we still flip `has_rewound` so gop_yuv()
        // enforces the trait contract that mirrors stateful sources.
        self.has_rewound = true;
        Ok(())
    }
}

/// WV.7.8 — closure-driven [`Av1GopYuvSource`] for the FFI bridges +
/// CLI demux callers.
///
/// Mirror of H.264's `CallbackYuvSource` (g.4.4a). The bridge wraps
/// its native decoder (iOS `AVAssetReader`, Android `MediaCodec`,
/// CLI ffmpeg) in two closures — one to decode a GOP on demand, one
/// to rewind to GOP 0 — and `Av1CallbackYuvSource` exposes them as a
/// trait the streaming encoder consumes.
///
/// **`Send + 'static` bounds.** Closures must outlive the Rust
/// caller's stack: the FFI extern-C entry receives raw C function
/// pointers + an opaque user-data pointer, builds an
/// `Av1CallbackYuvSource` around them, and passes a `&mut dyn
/// Av1GopYuvSource` into `av1_encode_with_shadows_streaming`. The
/// closure captures the user-data pointer (raw `*mut c_void`
/// wrapped in a `Send` newtype) and the C function pointer.
///
/// **No thread crossing.** The streaming encode runs sequentially;
/// closures are called from one thread at a time. `Send` is needed
/// only for the FFI's "no `Sync`-but-pinned-to-stack" idiom — the
/// closures can be safely re-bound across the FFI boundary.
///
/// **Lifetime.** The trait methods take `&mut self`, so callers can
/// mutate state inside the closures (e.g. advance a native
/// decoder's read cursor between GOPs).
pub struct Av1CallbackYuvSource {
    decode_gop_fn: Box<dyn FnMut(u32) -> Result<Vec<u8>, Av1StegoError> + 'static>,
    rewind_fn: Box<dyn FnMut() -> Result<(), Av1StegoError> + 'static>,
}

impl Av1CallbackYuvSource {
    /// Build a callback-driven source from two `FnMut` closures.
    ///
    /// **Why no `Send` bound** (relaxed in WV.7.9): the streaming
    /// encoder uses `source` sequentially on the calling thread — it
    /// never spawns an internal worker that takes `&mut source`. So
    /// `Send` would only be needed if a caller wanted to move the
    /// source itself across threads pre-encode, which no caller does
    /// in practice. Dropping the bound lets the FFI bridge capture raw
    /// C function pointers + `*mut c_void` user-data into the closure
    /// without `unsafe impl Send` wrapper dances (which RFC 2229
    /// disjoint capture defeats anyway by splitting `ctx.fn_ptr` and
    /// `ctx.user_ptr` into separate captures).
    pub fn new<D, R>(decode_gop_fn: D, rewind_fn: R) -> Self
    where
        D: FnMut(u32) -> Result<Vec<u8>, Av1StegoError> + 'static,
        R: FnMut() -> Result<(), Av1StegoError> + 'static,
    {
        Self {
            decode_gop_fn: Box::new(decode_gop_fn),
            rewind_fn: Box::new(rewind_fn),
        }
    }
}

impl Av1GopYuvSource for Av1CallbackYuvSource {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError> {
        (self.decode_gop_fn)(gop_index)
    }

    fn rewind(&mut self) -> Result<(), Av1StegoError> {
        (self.rewind_fn)()
    }
}

/// File-backed [`Av1GopYuvSource`] for desktop / CLI callers (WV.7.14
/// mirror of H.264's `FileYuvSource`). The CLI uses this to pipe an
/// ffmpeg-decoded tight-I420 YUV temp file directly into the streaming
/// AV1 shadow encode without ever loading the whole clip into RAM —
/// the long-clip-on-desktop OOM analogue of the mobile fix.
///
/// The file MUST be exactly `n_frames * width * height * 3/2` bytes of
/// tight-I420 (planar Y, then U, then V; no padding), frames in order.
/// `gop_yuv(g)` seeks absolutely on every call, so concurrent reads
/// from a single source aren't supported (and the encoder doesn't need
/// them — it pulls sequentially within each sweep, with `rewind()` in
/// between).
pub struct Av1FileYuvSource {
    file: std::fs::File,
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
}

impl Av1FileYuvSource {
    /// Open `path` as the tight-I420 YUV backing store.
    pub fn open(
        path: &std::path::Path,
        width: u32,
        height: u32,
        n_frames: u32,
        gop_size: u32,
    ) -> Result<Self, Av1StegoError> {
        let file = std::fs::File::open(path).map_err(|e| {
            Av1StegoError::InvalidPacket(format!(
                "Av1FileYuvSource: open {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self {
            file,
            width,
            height,
            n_frames,
            gop_size: gop_size.max(1),
        })
    }
}

impl Av1GopYuvSource for Av1FileYuvSource {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError> {
        use std::io::{Read, Seek, SeekFrom};
        let frame_bytes = (self.width as u64) * (self.height as u64) * 3 / 2;
        let start_f = (gop_index as u64).saturating_mul(self.gop_size as u64);
        if start_f >= self.n_frames as u64 {
            return Ok(Vec::new()); // past last GOP (matches Av1SliceYuvSource)
        }
        let end_f = (((gop_index as u64) + 1) * self.gop_size as u64)
            .min(self.n_frames as u64);
        let offset = start_f * frame_bytes;
        let len = ((end_f - start_f) * frame_bytes) as usize;
        self.file.seek(SeekFrom::Start(offset)).map_err(|e| {
            Av1StegoError::InvalidPacket(format!(
                "Av1FileYuvSource: seek {offset}: {e}"
            ))
        })?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf).map_err(|e| {
            Av1StegoError::InvalidPacket(format!(
                "Av1FileYuvSource: read {len} @ {offset}: {e}"
            ))
        })?;
        Ok(buf)
    }

    fn rewind(&mut self) -> Result<(), Av1StegoError> {
        use std::io::{Seek, SeekFrom};
        self.file
            .seek(SeekFrom::Start(0))
            .map(|_| ())
            .map_err(|e| {
                Av1StegoError::InvalidPacket(format!(
                    "Av1FileYuvSource: rewind: {e}"
                ))
            })
    }
}

/// WV.7.5 — streaming entry point for the AV1 whole-video shadow encode.
///
/// Production-shaped per-GOP streaming flow (no whole-clip
/// `per_gop_natural[]` accumulator). Design B from the WV.7.4→7.5
/// design analysis:
///
/// **Pass 1** (once, parity-independent): pull each GOP's YUV from
/// `source`, run `process_one_gop_pass1`, **keep** the harvest
/// (~60 KB/GOP), **drop** the per-frame natural OBU bytes + recording
/// (~5 MB/GOP) immediately. Total Pass-1 RAM: O(n_gops × 60 KB) ≈
/// 108 MB at 30 min.
///
/// **Cascade** (per parity tier, retried until verify passes):
/// 1. Re-run the shadow position sweep over the cached `cover_lens`
///    (no encode work — sweep is heap operations only). Yields
///    per-shadow `Av1ShadowState` for THIS parity tier.
/// 2. For each GOP: `source.gop_yuv(g)` → `process_one_gop_pass1`
///    → `cascade_one_gop` with the cached harvest + freshly-encoded
///    natural+recording → emit per-GOP bytes → DROP the per-GOP
///    natural+recording before advancing.
/// 3. Verify the assembled clip via `harvest_cover_bits_from_stego`
///    + per-shadow `av1_shadow_extract`; on success return; on
///    failure advance to the next parity tier (and re-run the
///    per-GOP encode + cascade loop from scratch).
///
/// **Memory at 30 min × 1080p × 8 shadows (Design B target):**
/// `current_gop_yuv` (~93 MB) + `per_gop_harvests` cache (~108 MB)
/// + rav1e encode transient (~500 MB peak) + cascade scratch
/// (~3 MB) + output Vec (~1.1 GB — O(clip)) ⇒ **~1.7 GB peak**.
/// Streaming the output to a sink (a follow-on after this commit)
/// drops the last term and gets us to ~625 MB — mobile-safe.
///
/// **Wall cost:** Pass 1 = 1× per-GOP encode total. Cascade = 1× per
/// per-GOP per parity tier. First-tier-success (common): 2×
/// per-GOP encode total. Worst case (5-tier cascade): 6×. Vs current
/// whole-clip path's 1× encode.
///
/// The shadow priority is position-local + parity-independent (only
/// `n_total` changes per tier — the priority order over the union
/// cover is fixed by `perm_seed`). So Pass 1's harvest cache is
/// reusable across all parity tiers; only the heap capacity changes.
pub fn av1_encode_with_shadows_streaming(
    source: &mut dyn Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    shadows: &[(&str, &[u8])],
    shadow_parity_len_floor: usize,
) -> Result<Vec<u8>, Av1StegoError> {
    if shadows.is_empty() {
        return Err(Av1StegoError::InvalidPacket(
            "av1_encode_with_shadows_streaming: empty shadows — caller \
             should route to per-GOP path instead"
                .into(),
        ));
    }
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_encode_with_shadows_streaming: n_frames == 0".into(),
        ));
    }
    let frame_size = expected_i420_size(params.width, params.height);
    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);
    let n_gops_usize = n_gops as usize;

    // ── Helper: per-GOP frame count, including the trailing partial.
    let frames_in_gop = |gop_idx: u32| -> u32 {
        let start = gop_idx * gop_size;
        let end = (start + gop_size).min(n_frames);
        end - start
    };

    // ── Helper: validate a GOP YUV pulled from `source`.
    let validate_gop_yuv =
        |gop_idx: u32, yuv: &[u8]| -> Result<(), Av1StegoError> {
            let expected = (frames_in_gop(gop_idx) as usize) * frame_size;
            if yuv.len() != expected {
                return Err(Av1StegoError::InvalidPacket(format!(
                    "av1_encode_with_shadows_streaming: GOP {gop_idx} source \
                     returned {} bytes, expected {} ({} frames × {} B)",
                    yuv.len(),
                    expected,
                    frames_in_gop(gop_idx),
                    frame_size,
                )));
            }
            Ok(())
        };

    // ── Pass 1: per-GOP encode → harvest only. Drops natural+recording
    // immediately. RAM held after Pass 1 = sum of GopHarvests (~60 KB
    // × n_gops). The natural+recording (~5 MB/GOP) is freed
    // gop-by-gop.
    //
    // `source.rewind()` is mandatory before the first `gop_yuv` call:
    // stateful sources (AVAssetReader, MediaCodec, ffmpeg pipe) need
    // it to open / position their underlying decoder. See the
    // matching note in `av1_encode_streaming` (T3.1).
    let prof = std::env::var("PHASM_AV1_PROFILE").map(|v| v == "1").unwrap_or(false);
    let t_total = std::time::Instant::now();
    if prof {
        eprintln!(
            "[AV1_PROF SHADOW] start: width={}, height={}, n_frames={}, n_gops={}, gop_size={}, n_shadows={}, parity_floor={}",
            params.width, params.height, n_frames, n_gops, gop_size,
            shadows.len(), shadow_parity_len_floor,
        );
    }
    let t_pass1 = std::time::Instant::now();
    source.rewind()?;
    let mut per_gop_harvests: Vec<GopHarvest> = Vec::with_capacity(n_gops_usize);
    let mut p1_sum_pull = std::time::Duration::ZERO;
    let mut p1_sum_proc = std::time::Duration::ZERO;
    for gop_idx in 0..n_gops {
        let t_pull = std::time::Instant::now();
        let gop_yuv = source.gop_yuv(gop_idx)?;
        let pull_dt = t_pull.elapsed();
        p1_sum_pull += pull_dt;
        validate_gop_yuv(gop_idx, &gop_yuv)?;
        let t_proc = std::time::Instant::now();
        let (per_frame, harvest) =
            process_one_gop_pass1(&gop_yuv, frames_in_gop(gop_idx), params)?;
        let proc_dt = t_proc.elapsed();
        p1_sum_proc += proc_dt;
        if prof {
            eprintln!(
                "[AV1_PROF SHADOW] Pass1 gop {}: pull={:?}, encode+harvest={:?}, cover_bits={}",
                gop_idx, pull_dt, proc_dt, harvest.cover_bits.len(),
            );
        }
        drop(per_frame);
        per_gop_harvests.push(harvest);
    }
    if prof {
        eprintln!(
            "[AV1_PROF SHADOW] Pass1 TOTAL: {:?}, sum_pull={:?}, sum_proc={:?}",
            t_pass1.elapsed(), p1_sum_pull, p1_sum_proc,
        );
    }

    // Per-GOP cover offsets into the union (= prefix sum of harvest
    // cover_bits.len()).
    let per_gop_cover_lens: Vec<usize> = per_gop_harvests
        .iter()
        .map(|h| h.cover_bits.len())
        .collect();
    let mut per_gop_cover_offsets: Vec<usize> = Vec::with_capacity(n_gops_usize);
    let mut offset = 0usize;
    for &n in &per_gop_cover_lens {
        per_gop_cover_offsets.push(offset);
        offset += n;
    }

    let total_chunks = u16::try_from(n_gops_usize).map_err(|_| {
        Av1StegoError::InvalidPacket(format!("n_gops {} exceeds u16::MAX", n_gops_usize))
    })?;
    if total_chunks == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_encode_with_shadows_streaming: n_gops == 0".into(),
        ));
    }
    if primary_framed.len() > u32::MAX as usize {
        return Err(Av1StegoError::InvalidPacket(format!(
            "primary_framed {} exceeds u32::MAX",
            primary_framed.len()
        )));
    }
    let total_message_bytes = primary_framed.len() as u32;
    let chunks = split_message_into_chunks(primary_framed, total_chunks)
        .map_err(Av1StegoError::Stego)?;

    let structural_key = crate::stego::crypto::derive_structural_key(primary_passphrase)
        .map_err(Av1StegoError::Stego)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    // Parity-tier cascade. Each tier re-runs the per-GOP re-encode
    // because the per-GOP natural+recording was dropped after Pass 1.
    let parity_tiers: Vec<usize> = AV1_SHADOW_PARITY_TIERS
        .iter()
        .copied()
        .filter(|&p| p >= shadow_parity_len_floor)
        .collect();
    if parity_tiers.is_empty() {
        return Err(Av1StegoError::InvalidPacket(format!(
            "shadow_parity_len_floor {} exceeds all SHADOW_PARITY_TIERS",
            shadow_parity_len_floor
        )));
    }

    for parity_len in parity_tiers {
        let t_tier = std::time::Instant::now();
        if prof {
            eprintln!("[AV1_PROF SHADOW] === Tier parity_len={} ===", parity_len);
        }
        // Cover-independent per-shadow state for THIS parity tier.
        let pre_selections: Vec<_> = match shadows
            .iter()
            .map(|(p, m)| prepare_shadow_pre_selection_av1(p, m, parity_len))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Drive the streaming sweep over cached per-GOP cover_lens.
        // No encode work — heap operations only.
        let specs: Vec<_> = pre_selections
            .iter()
            .map(|p| p.to_selection_spec())
            .collect();
        let mut sweep = Av1ShadowSelectionSweep::new(&specs);
        for &gop_n in &per_gop_cover_lens {
            sweep.push_gop(gop_n);
        }
        let per_shadow_positions = match sweep.finish() {
            Ok(v) => v,
            Err(_) => continue, // capacity exhausted → try next parity tier
        };

        // Clone pre_selections for the streaming verifier before
        // finalize_shadow_state_av1 consumes them. Cheap — each
        // pre_selection is ~80 B + n_total bits (~64 B for a 1 KB
        // shadow at parity 16). For ≤8 shadows total RAM is < 10 KB.
        let pre_selections_for_verify = pre_selections.clone();
        let passphrases_for_verify: Vec<String> =
            shadows.iter().map(|(p, _)| (*p).to_string()).collect();

        let shadow_states: Vec<Av1ShadowState> = pre_selections
            .into_iter()
            .zip(per_shadow_positions)
            .map(|(pre, positions)| finalize_shadow_state_av1(pre, positions))
            .collect();

        // WV.7.6 — streaming verify: as each GOP emits, walk its
        // bytes via dav1d, push the harvested cover bits (with global
        // cover_index) into per-shadow Av1ShadowBitTopN heaps. After
        // all GOPs, finish() drains heaps + runs RS-decode + AES auth
        // per shadow. No whole-clip cover_bits / priority_slots Vec
        // materialised — drops ~55 GB of verify-time RAM at 30 min
        // vs the legacy verify_shadows_round_trip path.
        let mut streaming_verify = Av1StreamingShadowVerify::new(
            pre_selections_for_verify,
            passphrases_for_verify,
        );

        // Per-GOP re-encode + cascade. Source rewind: the contract
        // is that `gop_yuv(g)` is callable with monotonically
        // increasing `g` within a sweep, and `rewind()` restarts the
        // forward stream for the next sweep.
        let t_cascade = std::time::Instant::now();
        source.rewind()?;
        let mut output: Vec<u8> = Vec::new();
        let mut encode_ok = true;
        let mut c_sum_pull = std::time::Duration::ZERO;
        let mut c_sum_encode = std::time::Duration::ZERO;
        let mut c_sum_cascade = std::time::Duration::ZERO;
        let mut c_sum_verify = std::time::Duration::ZERO;
        for gop_idx in 0..n_gops {
            let t_pull = std::time::Instant::now();
            let gop_yuv = source.gop_yuv(gop_idx)?;
            let pull_dt = t_pull.elapsed();
            c_sum_pull += pull_dt;
            validate_gop_yuv(gop_idx, &gop_yuv)?;
            // WV.7.7 — skip the J-UNIWARD harvest in Sweep B. Pass 1
            // already cached `per_gop_harvests[gop_idx]` (parity-
            // independent — perm_seed determines priority, parity_len
            // only changes heap capacity which we re-run via the
            // sweep). `cascade_one_gop` reads the cached harvest's
            // cover_bits + costs + combined_index, so the fresh
            // harvest would be discarded anyway. Calling
            // `encode_gop_natural` directly skips the per-frame
            // `compute_av1_uniward_costs_with_state` walk (~0.1-0.3 s
            // per 1080p GOP) — ~10-20% wall savings on the streaming
            // encode at long-clip scale.
            let t_enc = std::time::Instant::now();
            let per_frame = encode_gop_natural(
                &gop_yuv,
                frames_in_gop(gop_idx),
                params,
            )?;
            let enc_dt = t_enc.elapsed();
            c_sum_encode += enc_dt;
            let gop_idx_usize = gop_idx as usize;
            let inputs = Av1CascadeGopInputs {
                gop_idx: gop_idx_usize,
                gop_offset: per_gop_cover_offsets[gop_idx_usize],
                gop_n: per_gop_cover_lens[gop_idx_usize],
                harvest: &per_gop_harvests[gop_idx_usize],
                frames: &per_frame,
            };
            let shared = Av1CascadeGopShared {
                chunks: &chunks,
                total_message_bytes,
                shadow_states: &shadow_states,
                hhat_seed: &hhat_seed,
            };
            let t_casc = std::time::Instant::now();
            let cascade_result = cascade_one_gop(inputs, shared)?;
            let casc_dt = t_casc.elapsed();
            c_sum_cascade += casc_dt;
            match cascade_result {
                Some(per_gop_bytes) => {
                    let t_ver = std::time::Instant::now();
                    streaming_verify.push_gop_bytes(&per_gop_bytes)?;
                    let ver_dt = t_ver.elapsed();
                    c_sum_verify += ver_dt;
                    if prof {
                        eprintln!(
                            "[AV1_PROF SHADOW] Cascade gop {}: pull={:?}, encode={:?}, cascade={:?}, verify_push={:?}, packet={} B",
                            gop_idx, pull_dt, enc_dt, casc_dt, ver_dt, per_gop_bytes.len()
                        );
                    }
                    output.extend_from_slice(&per_gop_bytes);
                }
                None => {
                    if prof {
                        eprintln!(
                            "[AV1_PROF SHADOW] Cascade gop {}: cascade returned None — tier failed",
                            gop_idx
                        );
                    }
                    encode_ok = false;
                    break;
                }
            }
        }

        if !encode_ok {
            if prof {
                eprintln!(
                    "[AV1_PROF SHADOW] Tier parity_len={} FAILED in {:?}",
                    parity_len, t_tier.elapsed()
                );
            }
            continue;
        }
        let t_verfin = std::time::Instant::now();
        let ok = streaming_verify.finish();
        let verfin_dt = t_verfin.elapsed();
        if prof {
            eprintln!(
                "[AV1_PROF SHADOW] Tier parity_len={} cascade_total={:?}, sum_pull={:?}, sum_encode={:?}, sum_cascade={:?}, sum_verify_push={:?}, verify_finish={:?} → ok={}",
                parity_len, t_cascade.elapsed(),
                c_sum_pull, c_sum_encode, c_sum_cascade, c_sum_verify, verfin_dt, ok,
            );
        }
        if ok {
            if prof {
                eprintln!(
                    "[AV1_PROF SHADOW] TOTAL: {:?}, output={} B (succeeded at tier parity_len={})",
                    t_total.elapsed(), output.len(), parity_len
                );
            }
            return Ok(output);
        }
    }

    Err(Av1StegoError::Stego(
        crate::stego::error::StegoError::FrameCorrupted,
    ))
}

/// MOBOOM.T3.1 — streaming AV1 encode for the NO-SHADOW case.
///
/// Sibling of [`av1_encode_with_shadows_streaming`] for clips without
/// shadow embedding. Drives an [`Av1GopYuvSource`]: for each GOP the
/// source delivers tight-I420 YUV in a fast burst, the function runs
/// one rav1e encode + chunk_frame STC override, emits the GOP's stego
/// bytes, drops the YUV. **No cascade tier loop, no streaming verify,
/// no harvest cache** — none of that infrastructure is needed because
/// there are no shadow bits to embed.
///
/// **Why this exists.** The legacy no-shadow encode path
/// (`Av1StreamingEncodeSession::push_frame` → `drain_one_gop`) is
/// driven by the *caller* pulling one frame at a time from
/// `AVAssetReader` / `MediaCodec` and pushing it into a session that
/// only emits OBU bytes at GOP boundaries. AVFoundation reads happen
/// interleaved with multi-second rav1e calls on the same thread —
/// AVAssetReader marks the session interrupted under that read
/// cadence, surfacing as `MediaPlaneExtractorError.readerFailed("Operation Interrupted")`
/// on cgPhone testing. The shadow-WV streaming entry already uses the
/// pull-source pattern (reads happen in fast per-GOP bursts between
/// rav1e calls); this entry brings the no-shadow case onto the same
/// architecture so AVFoundation reads are no longer interleaved.
///
/// **Byte-identity with the legacy push-session.** For a given
/// (passphrase, primary_framed, params) tuple this function produces
/// **bit-for-bit identical output** to driving an
/// `Av1StreamingEncodeSession` push-frame loop over the same YUV at
/// the same `params.gop_size`. See `av1_no_shadow_streaming_byte_identical.rs`
/// for the gate. The per-GOP encode primitives (`encode_one_keyframe`,
/// `encode_one_gop_multi`) and the stego embed primitives
/// (`av1_stego_embed_payload_bits`, `av1_stego_encode_one_gop`) are
/// called in the same order with the same arguments.
///
/// `primary_framed` is the encrypt + `frame::build_frame` output for
/// the primary message (same contract as the shadow streaming entry —
/// the caller does crypto::encrypt + frame::build_frame upstream).
///
/// **Memory profile.** Per-GOP transient: one GOP's YUV (~90 MB at
/// 1080p × gop=30) + one rav1e encode state (~120 MB at 1080p) — both
/// dropped between GOPs. Persistent: the output Vec accumulating stego
/// OBU bytes across all GOPs (O(clip) — same as today). At 30 min ×
/// 1080p the output term is ~1 GB; dropping that to true streaming
/// requires a GOP-incremental muxer and is future work.
pub fn av1_encode_streaming(
    source: &mut dyn Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    // Optional 0.0–1.0 progress sink, ticked once per GOP after that
    // GOP's stego encode completes. Mirror of H.264's
    // `h264_encode_with_shadows_streaming` `progress` parameter. The
    // no-shadow streaming entry is a single forward pass, so the
    // emitted fraction is simply `(gop_idx + 1) / n_gops`.
    mut on_progress: Option<&mut dyn FnMut(f32)>,
) -> Result<Vec<u8>, Av1StegoError> {
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_encode_streaming: n_frames == 0".into(),
        ));
    }
    let frame_size = expected_i420_size(params.width, params.height);
    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);

    // Per-GOP frame count helper. The trailing GOP may be partial.
    let frames_in_gop = |gop_idx: u32| -> u32 {
        let start = gop_idx * gop_size;
        let end = (start + gop_size).min(n_frames);
        end - start
    };

    let validate_gop_yuv = |gop_idx: u32, yuv: &[u8]| -> Result<(), Av1StegoError> {
        let expected = (frames_in_gop(gop_idx) as usize) * frame_size;
        if yuv.len() != expected {
            return Err(Av1StegoError::InvalidPacket(format!(
                "av1_encode_streaming: GOP {gop_idx} source returned {} bytes, \
                 expected {} ({} frames × {} B)",
                yuv.len(),
                expected,
                frames_in_gop(gop_idx),
                frame_size,
            )));
        }
        Ok(())
    };

    // Bound + cap the chunk count. Mirror of the session's
    // `derived_total_gops`/`total_chunks` math, simplified: every GOP
    // is a stego GOP (no `plan_safe_balanced` concentrate-tail in v1
    // of this entry — if MessageTooLarge surfaces, threading per-GOP
    // allocation through is the follow-on).
    if n_gops == 0 || n_gops > u16::MAX as u32 {
        return Err(Av1StegoError::InvalidPacket(format!(
            "av1_encode_streaming: derived n_gops {n_gops} out of range [1..={}]",
            u16::MAX
        )));
    }
    if primary_framed.len() > u32::MAX as usize {
        return Err(Av1StegoError::InvalidPacket(format!(
            "av1_encode_streaming: primary_framed {} exceeds u32::MAX",
            primary_framed.len()
        )));
    }
    let total_message_bytes = primary_framed.len() as u32;
    let total_chunks = n_gops as u16;
    let chunks = split_message_into_chunks(primary_framed, total_chunks)
        .map_err(Av1StegoError::Stego)?;

    // Derive hhat seed from the primary passphrase — same path the
    // session uses for chunk_frame's STC override. Single derive for
    // the whole encode (hhat_seed is reused across GOPs).
    let structural_key = crate::stego::crypto::derive_structural_key(primary_passphrase)
        .map_err(Av1StegoError::Stego)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    // Output accumulator. Reserve ~1 KB/frame as a starting guess —
    // Vec doubles when full. Same reservation as the iOS session
    // path's `obus.reserveCapacity(totalFrames * 1024)`.
    let mut output: Vec<u8> = Vec::with_capacity((n_frames as usize).saturating_mul(1024));

    // Per-GOP encode loop. Bounded by `n_gops` not by what the source
    // yields — the source's contract is to deliver each GOP in order
    // (forward-only). Trailing partial GOPs are first-class.
    //
    // `source.rewind()` is mandatory before the first `gop_yuv` call:
    // stateful sources (Av1CallbackYuvSource backing the iOS
    // AVAssetReader / Android MediaCodec / CLI ffmpeg pipe) need it to
    // open / position their underlying decoder. Stateless sources
    // (Av1SliceYuvSource) make it a no-op. Skipping this surfaces as
    // `MediaPlaneGopSourceError.readerFailed("decode_gop before
    // rewind")` on iOS — see commit history for T3.1/T3.5 root cause.
    let prof = std::env::var("PHASM_AV1_PROFILE").map(|v| v == "1").unwrap_or(false);
    let t_total = std::time::Instant::now();
    if prof {
        eprintln!(
            "[AV1_PROF NO_SHADOW] start: width={}, height={}, n_frames={}, n_gops={}, gop_size={}, primary_framed_len={}",
            params.width, params.height, n_frames, n_gops, gop_size, primary_framed.len()
        );
    }
    let t_rewind = std::time::Instant::now();
    source.rewind()?;
    if prof {
        eprintln!("[AV1_PROF NO_SHADOW] rewind: {:?}", t_rewind.elapsed());
    }
    let mut sum_pull = std::time::Duration::ZERO;
    let mut sum_encode = std::time::Duration::ZERO;
    let mut sum_embed = std::time::Duration::ZERO;
    for gop_idx in 0..n_gops {
        let t_pull = std::time::Instant::now();
        let gop_yuv = source.gop_yuv(gop_idx)?;
        let pull_dt = t_pull.elapsed();
        sum_pull += pull_dt;
        validate_gop_yuv(gop_idx, &gop_yuv)?;
        let fig = frames_in_gop(gop_idx);

        // Build this GOP's chunk_frame slice. First GOP carries the
        // clip-level `total_message_bytes` header; subsequent GOPs
        // carry only their `payload_len`. Mirror of the session's
        // `drain_one_gop` chunk_frame logic.
        let chunk_payload = &chunks[gop_idx as usize];
        let framed = if gop_idx == 0 {
            build_first_chunk_frame_v3_1(total_message_bytes, chunk_payload)
        } else {
            build_chunk_frame_v3_1(chunk_payload)
        }
        .map_err(Av1StegoError::Stego)?;
        let payload_bits = frame::bytes_to_bits(&framed);

        // Encode + embed. Dispatches on frames_in_gop the same way the
        // session does — single-frame path uses `encode_one_keyframe`
        // + `av1_stego_embed_payload_bits` (bit-exact to the legacy
        // single-frame primitive); multi-frame uses `encode_one_gop_multi`
        // + `av1_stego_encode_one_gop` for the IDR + P-frame chain.
        let t_enc = std::time::Instant::now();
        let stego_packet = if fig == 1 {
            let (natural, recording) = encode_one_keyframe(&gop_yuv, params)?;
            let enc_dt = t_enc.elapsed();
            let t_embed = std::time::Instant::now();
            let packet =
                av1_stego_embed_payload_bits(natural, recording, &payload_bits, &hhat_seed)?;
            let embed_dt = t_embed.elapsed();
            sum_encode += enc_dt;
            sum_embed += embed_dt;
            if prof {
                eprintln!(
                    "[AV1_PROF NO_SHADOW] gop {}: fig=1, pull={:?}, encode={:?}, embed={:?}, packet={} B",
                    gop_idx, pull_dt, enc_dt, embed_dt, packet.len()
                );
            }
            packet
        } else {
            let per_frame = encode_one_gop_multi(&gop_yuv, fig, params)?;
            let enc_dt = t_enc.elapsed();
            let t_embed = std::time::Instant::now();
            let packet = av1_stego_encode_one_gop(per_frame, &payload_bits, &hhat_seed)?;
            let embed_dt = t_embed.elapsed();
            sum_encode += enc_dt;
            sum_embed += embed_dt;
            if prof {
                eprintln!(
                    "[AV1_PROF NO_SHADOW] gop {}: fig={}, pull={:?}, encode={:?}, embed={:?}, packet={} B",
                    gop_idx, fig, pull_dt, enc_dt, embed_dt, packet.len()
                );
            }
            packet
        };

        output.extend_from_slice(&stego_packet);

        if let Some(ref mut cb) = on_progress {
            cb((gop_idx + 1) as f32 / n_gops as f32);
        }
    }

    if prof {
        eprintln!(
            "[AV1_PROF NO_SHADOW] TOTAL: {:?}, sum_pull={:?}, sum_encode={:?}, sum_embed={:?}, output={} B",
            t_total.elapsed(), sum_pull, sum_encode, sum_embed, output.len()
        );
    }
    Ok(output)
}

/// Run Pass 1 on a single GOP: natural rav1e encode + dav1d harvest.
/// Returns the per-frame recordings (natural OBU + replay state) plus
/// the GOP-level cover/cost harvest. Used by the streaming session to
/// drive Pass 1 incrementally during `push_frame`, dropping YUV after
/// each GOP rather than holding the whole clip.
pub(super) fn process_one_gop_pass1(
    gop_yuv: &[u8],
    frames_in_gop: u32,
    params: Av1StreamingEncodeParams,
) -> Result<(Vec<(Vec<u8>, PhasmFrameRecording<u8>)>, GopHarvest), Av1StegoError> {
    let pf = encode_gop_natural(gop_yuv, frames_in_gop, params)?;
    let harvest = harvest_gop(&pf)?;
    Ok((pf, harvest))
}

pub fn av1_stego_encode_whole_video_with_shadows(
    yuv: &[u8],
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    shadows: &[(&str, &[u8])],
    shadow_parity_len_floor: usize,
) -> Result<Vec<u8>, Av1StegoError> {
    if shadows.is_empty() {
        return Err(Av1StegoError::InvalidPacket(
            "av1_stego_encode_whole_video_with_shadows: empty shadows — caller \
             should route to per-GOP path instead"
                .into(),
        ));
    }
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_stego_encode_whole_video_with_shadows: n_frames == 0".into(),
        ));
    }
    let frame_size = expected_i420_size(params.width, params.height);
    let expected_total = frame_size
        .checked_mul(n_frames as usize)
        .ok_or_else(|| {
            Av1StegoError::InvalidPacket(format!(
                "yuv size overflow ({} × {})",
                frame_size, n_frames
            ))
        })?;
    if yuv.len() != expected_total {
        return Err(Av1StegoError::InvalidPacket(format!(
            "yuv length {} != expected {} ({} frames × {} bytes)",
            yuv.len(),
            expected_total,
            n_frames,
            frame_size
        )));
    }

    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);

    // Pass 1: encode every GOP naturally + harvest per-GOP Tier-1 cover.
    // Buffer-all path: walks the full YUV in a single function call,
    // used by tests and the legacy whole-clip API. Production
    // `Av1StreamingEncodeSession` drives Pass 1 incrementally via
    // `process_one_gop_pass1` and calls the cascade-only path
    // (`encode_whole_video_with_shadows_from_prepared`) directly,
    // bounding peak memory to ~one GOP of YUV rather than the whole
    // clip.
    let mut per_gop_natural: Vec<Vec<(Vec<u8>, PhasmFrameRecording<u8>)>> = Vec::with_capacity(n_gops as usize);
    let mut per_gop_harvests: Vec<GopHarvest> = Vec::with_capacity(n_gops as usize);

    for gop_idx in 0..n_gops as usize {
        let start_frame = gop_idx as u32 * gop_size;
        let end_frame = (start_frame + gop_size).min(n_frames);
        let frames_in_gop = end_frame - start_frame;
        let gop_yuv = &yuv[(start_frame as usize * frame_size)
            ..(end_frame as usize * frame_size)];
        let (pf, harvest) = process_one_gop_pass1(gop_yuv, frames_in_gop, params)?;
        per_gop_natural.push(pf);
        per_gop_harvests.push(harvest);
    }

    encode_whole_video_with_shadows_from_prepared(
        &per_gop_natural,
        &per_gop_harvests,
        params,
        primary_framed,
        primary_passphrase,
        shadows,
        shadow_parity_len_floor,
    )
}

/// Cascade-only path: takes Pass 1 outputs and runs the cascade loop
/// (shadow position derivation over the union cover + per-GOP STC +
/// replay + verify) to produce the final stego AV1 bytes.
///
/// Called by `Av1StreamingEncodeSession::finish` in whole-video mode
/// once `push_frame` has streamed Pass 1 GOP-by-GOP. No YUV needed —
/// everything required for the cascade is in `per_gop_natural` (the
/// natural OBU bytes + per-frame `PhasmFrameRecording`s for replay) and
/// `per_gop_harvests` (per-GOP cover bits + costs).
pub(super) fn encode_whole_video_with_shadows_from_prepared(
    per_gop_natural: &[Vec<(Vec<u8>, PhasmFrameRecording<u8>)>],
    per_gop_harvests: &[GopHarvest],
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    shadows: &[(&str, &[u8])],
    shadow_parity_len_floor: usize,
) -> Result<Vec<u8>, Av1StegoError> {
    if shadows.is_empty() {
        return Err(Av1StegoError::InvalidPacket(
            "encode_whole_video_with_shadows_from_prepared: empty shadows".into(),
        ));
    }
    if per_gop_natural.len() != per_gop_harvests.len() {
        return Err(Av1StegoError::InvalidPacket(format!(
            "encode_whole_video_with_shadows_from_prepared: per_gop_natural ({}) \
             != per_gop_harvests ({})",
            per_gop_natural.len(),
            per_gop_harvests.len()
        )));
    }
    let n_gops_usize = per_gop_natural.len();
    let total_chunks = u16::try_from(n_gops_usize).map_err(|_| {
        Av1StegoError::InvalidPacket(format!("n_gops {} exceeds u16::MAX", n_gops_usize))
    })?;
    if total_chunks == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "encode_whole_video_with_shadows_from_prepared: n_gops == 0".into(),
        ));
    }
    // Clip-level total_bytes goes in the first chunk header (v3 wire format).
    if primary_framed.len() > u32::MAX as usize {
        return Err(Av1StegoError::InvalidPacket(format!(
            "primary_framed {} exceeds u32::MAX",
            primary_framed.len()
        )));
    }
    let total_message_bytes = primary_framed.len() as u32;
    let chunks = split_message_into_chunks(primary_framed, total_chunks)
        .map_err(Av1StegoError::Stego)?;

    let structural_key = crate::stego::crypto::derive_structural_key(primary_passphrase)
        .map_err(Av1StegoError::Stego)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    let n_gops = n_gops_usize as u32;
    let _ = n_gops; // value retained for future log/diagnostics if needed

    // Compute union cover length + per-GOP offsets.
    let per_gop_cover_lens: Vec<usize> = per_gop_harvests
        .iter()
        .map(|h| h.cover_bits.len())
        .collect();
    let mut per_gop_cover_offsets: Vec<usize> = Vec::with_capacity(n_gops as usize);
    let mut offset = 0usize;
    for &n in &per_gop_cover_lens {
        per_gop_cover_offsets.push(offset);
        offset += n;
    }
    let union_n = offset;

    // Cascade loop over SHADOW_PARITY_TIERS starting at the
    // caller-provided floor. Default floor matches `create_with_shadows`'s
    // 16 unless caller overrode.
    let parity_tiers: Vec<usize> = AV1_SHADOW_PARITY_TIERS
        .iter()
        .copied()
        .filter(|&p| p >= shadow_parity_len_floor)
        .collect();

    if parity_tiers.is_empty() {
        return Err(Av1StegoError::InvalidPacket(format!(
            "shadow_parity_len_floor {} exceeds all SHADOW_PARITY_TIERS",
            shadow_parity_len_floor
        )));
    }

    // `union_n` retained for diagnostics + the message-too-large guard
    // surfaced by the sweep's `finish()` (it returns
    // `StegoError::MessageTooLarge` iff the union doesn't hold a
    // shadow's n_total — same semantics as the pre-WV.7.3
    // `prepare_shadows(union_n, ...)` call).
    let _ = union_n;

    for parity_len in parity_tiers {
        // WV.7.3 — Prepare shadow states via streaming selection over
        // the per-GOP cover slices, NOT via the whole-clip
        // `priority_slots(union_n, ...).take(n_total)` materialisation
        // we used pre-WV.7.3.
        //
        // Step 1 (cover-independent): compute each shadow's `(perm_seed,
        // n_total, bits, parity_len, frame_data_len)`. ~µs per shadow.
        let pre_selections: Vec<_> = match shadows
            .iter()
            .map(|(p, m)| prepare_shadow_pre_selection_av1(p, m, parity_len))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(_) => continue, // crypto/RS failure at this parity → try next
        };

        // Step 2 (cover-dependent): drive the streaming sweep over
        // per-GOP cover sizes. Bit-identical to
        // `priority_slots(union_n, perm_seed).take(n_total)` by the
        // WV.7.1 (order-independence) + WV.7.2 (multi-shadow glue)
        // gates. The sweep holds O(n_total × n_shadows) entries — vs
        // the pre-WV.7.3 path's O(union_n × n_shadows) slot vector.
        let specs: Vec<_> = pre_selections
            .iter()
            .map(|p| p.to_selection_spec())
            .collect();
        let mut sweep = Av1ShadowSelectionSweep::new(&specs);
        for &gop_n in &per_gop_cover_lens {
            sweep.push_gop(gop_n);
        }
        let per_shadow_positions = match sweep.finish() {
            Ok(v) => v,
            Err(_) => continue, // capacity exhausted at this parity → try next
        };

        // Capture clones for the streaming verify (WV.7.6) before
        // pre_selections is consumed into shadow_states. The
        // streaming verifier needs n_total + parity_len +
        // frame_data_len + perm_seed per shadow at finish; cheaper
        // to clone the small pre_selections (n_total bits + ~80 B
        // overhead per shadow) than to plumb a second pass through
        // the shadow data.
        let pre_selections_for_verify = pre_selections.clone();
        let passphrases_for_verify: Vec<String> =
            shadows.iter().map(|(p, _)| (*p).to_string()).collect();

        // Step 3: combine into Av1ShadowStates. Byte-identical to
        // what `prepare_shadows(union_n, ...)` produced pre-WV.7.3
        // (both bits and positions identical).
        let shadow_states: Vec<Av1ShadowState> = pre_selections
            .into_iter()
            .zip(per_shadow_positions)
            .map(|(pre, positions)| finalize_shadow_state_av1(pre, positions))
            .collect();

        // WV.7.6 — streaming verify: as each GOP emits, walk its
        // bytes via dav1d, push the harvested cover bits (with global
        // cover_index) into per-shadow Av1ShadowBitTopN heaps. After
        // all GOPs, finish() drains heaps + runs RS-decode + AES auth
        // per shadow. No whole-clip cover_bits / priority_slots Vec
        // materialised — drops ~55 GB of verify-time RAM at 30 min
        // vs the legacy verify_shadows_round_trip path.
        let mut streaming_verify = Av1StreamingShadowVerify::new(
            pre_selections_for_verify,
            passphrases_for_verify,
        );

        // For each GOP, encode the GOP with the slice of shadow
        // positions that fall in that GOP's union range.
        let mut output: Vec<u8> = Vec::new();
        let mut encode_ok = true;
        for (gop_idx, harvest) in per_gop_harvests.iter().enumerate() {
            let inputs = Av1CascadeGopInputs {
                gop_idx,
                gop_offset: per_gop_cover_offsets[gop_idx],
                gop_n: per_gop_cover_lens[gop_idx],
                harvest,
                frames: &per_gop_natural[gop_idx],
            };
            let shared = Av1CascadeGopShared {
                chunks: &chunks,
                total_message_bytes,
                shadow_states: &shadow_states,
                hhat_seed: &hhat_seed,
            };
            match cascade_one_gop(inputs, shared)? {
                Some(per_gop_bytes) => {
                    // WV.7.6 — feed per-GOP bytes through the
                    // streaming verifier as they emit, instead of
                    // walking the assembled whole-clip output at the
                    // end. Drops the ~55 GB verify-time peak at
                    // 30 min (whole-clip cover_bits Vec + the
                    // whole-clip `priority_slots` Vec inside the
                    // legacy `av1_shadow_extract`). The output Vec
                    // itself is still O(clip) on this legacy path —
                    // its accumulator is the next term to cut, but
                    // requires a session-API change.
                    streaming_verify.push_gop_bytes(&per_gop_bytes)?;
                    output.extend_from_slice(&per_gop_bytes);
                }
                None => {
                    encode_ok = false;
                    break;
                }
            }
        }

        if !encode_ok {
            continue;
        }

        // WV.7.6 — drain per-shadow heaps + run RS-decode + AES
        // auth per shadow. Replaces the legacy
        // `verify_shadows_round_trip(&output, shadows)` whole-clip
        // walk. Same authenticated round-trip contract; ALL shadows
        // must verify or the parity tier is rejected.
        if streaming_verify.finish() {
            return Ok(output);
        }
    }

    Err(Av1StegoError::Stego(
        crate::stego::error::StegoError::FrameCorrupted,
    ))
}

// ────────────────────────────────────────────────────────────────
// WV.7.4 — per-GOP cascade body (the API for WV.7.5+ streaming)
// ────────────────────────────────────────────────────────────────

/// Per-GOP inputs to [`cascade_one_gop`]. Borrows from the existing
/// `per_gop_harvests` + `per_gop_natural` accumulators today; WV.7.5+
/// will populate this struct from a re-encode driven by a
/// [`Av1GopYuvSource`](super::Av1GopYuvSource) so the accumulators
/// can be dropped entirely.
pub(super) struct Av1CascadeGopInputs<'a> {
    pub gop_idx: usize,
    /// This GOP's first cover bit's offset into the union cover.
    /// Used to map shadow positions (which live in the union space)
    /// into GOP-local indices.
    pub gop_offset: usize,
    /// This GOP's cover bit count (= `harvest.cover_bits.len()`).
    pub gop_n: usize,
    pub harvest: &'a GopHarvest,
    pub frames: &'a [(Vec<u8>, PhasmFrameRecording<u8>)],
}

/// Whole-clip-scope inputs shared across every GOP within one parity
/// tier of the cascade loop. Borrowed for the duration of one
/// [`cascade_one_gop`] call.
pub(super) struct Av1CascadeGopShared<'a> {
    /// chunk_frame v3 per-GOP message bytes. `chunks[gop_idx]` is
    /// this GOP's slice; `chunks[0]` is the only one that carries
    /// the clip-level total_message_bytes header.
    pub chunks: &'a [Vec<u8>],
    /// Whole-clip total framed message length, for the chunk_frame
    /// v3 first-chunk header.
    pub total_message_bytes: u32,
    /// Per-shadow finalised state (positions + bits). The cascade
    /// filters each state's positions into this GOP's slice via
    /// `gop_offset`.
    pub shadow_states: &'a [Av1ShadowState],
    /// hhat_seed for STC, derived from primary passphrase.
    pub hhat_seed: &'a [u8; 32],
}

/// Run one GOP's body of the WV cascade: build chunk_frame v3 payload
/// bits, stamp shadow LSBs, ∞-cost-overlay the cost vector, run STC,
/// emit per-frame OverrideMaps, replay+splice each frame's natural
/// OBU bytes, return the GOP's stego output bytes.
///
/// `Ok(Some(bytes))` on success, `Ok(None)` if STC fails for this
/// GOP at this parity tier (caller advances to the next parity rung,
/// matching the pre-WV.7.4 `encode_ok = false; break;` semantics).
///
/// Extracted from `encode_whole_video_with_shadows_from_prepared` at
/// WV.7.4 — pure refactor, same per-GOP byte output. Sets up WV.7.5+
/// to drive the cascade from a streaming `Av1GopYuvSource` instead
/// of the `per_gop_natural[]` accumulator (which currently dominates
/// the O(clip) memory term at long clips, ~75-100 KB/frame at 1080p).
pub(super) fn cascade_one_gop(
    inputs: Av1CascadeGopInputs<'_>,
    shared: Av1CascadeGopShared<'_>,
) -> Result<Option<Vec<u8>>, Av1StegoError> {
    let Av1CascadeGopInputs {
        gop_idx,
        gop_offset,
        gop_n,
        harvest,
        frames,
    } = inputs;
    let Av1CascadeGopShared {
        chunks,
        total_message_bytes,
        shadow_states,
        hhat_seed,
    } = shared;

    // Build chunk_frame v3 payload bits for this GOP. First GOP
    // carries clip-level total_message_bytes; subsequent GOPs carry
    // only payload_len.
    let framed = if gop_idx == 0 {
        build_first_chunk_frame_v3_1(total_message_bytes, &chunks[gop_idx])
    } else {
        build_chunk_frame_v3_1(&chunks[gop_idx])
    }
    .map_err(Av1StegoError::Stego)?;
    let payload_bits = frame::bytes_to_bits(&framed);
    let m_bits = payload_bits.len();
    if m_bits == 0 || m_bits > gop_n {
        return Ok(None);
    }
    let w = (gop_n / m_bits).max(1);
    let n_used = m_bits * w;

    // Project per-GOP-local shadow slots (slot.cover_index -
    // gop_offset for the subset in [gop_offset, gop_offset + gop_n)).
    let mut gop_shadow_positions: Vec<Vec<(usize, u8)>> =
        vec![Vec::new(); shadow_states.len()];
    for (s_idx, state) in shadow_states.iter().enumerate() {
        for (bit_idx, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index >= gop_offset && slot.cover_index < gop_offset + gop_n {
                let local_idx = slot.cover_index - gop_offset;
                gop_shadow_positions[s_idx].push((local_idx, state.bits[bit_idx]));
            }
        }
    }

    // Stamp shadow LSBs into a clone of the GOP cover for STC's view.
    // Track positions for ∞-cost overlay + post-STC defensive stamp
    // + out-of-range overrides.
    let mut combined_cover: Vec<u8> = harvest.cover_bits.clone();
    let mut shadow_position_set: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for slots in &gop_shadow_positions {
        for &(local_idx, bit) in slots {
            if local_idx < n_used {
                combined_cover[local_idx] = bit;
            }
            shadow_position_set.insert(local_idx);
        }
    }
    // `harvest.cover_bits` is unchanged through this call (the shadow
    // stamps mutate `combined_cover`, never the harvest), so we read
    // the original cover directly off the borrowed harvest below —
    // no need for an explicit clone.

    // ∞-cost overlay.
    let mut costs_for_stc: Vec<f32> = harvest.costs[..n_used].to_vec();
    for &idx in &shadow_position_set {
        if idx < n_used {
            costs_for_stc[idx] = f32::INFINITY;
        }
    }

    let cover_used = &combined_cover[..n_used];
    let hhat_matrix = hhat::generate_hhat(STC_H, w, hhat_seed);
    let mut embed = match stc_embed(
        cover_used,
        &costs_for_stc,
        &payload_bits,
        &hhat_matrix,
        STC_H,
        w,
    ) {
        Some(p) => p,
        None => return Ok(None),
    };

    // Defensive shadow stamp on the STC plan.
    for slots in &gop_shadow_positions {
        for &(local_idx, bit) in slots {
            if local_idx < n_used {
                embed.stego_bits[local_idx] = bit;
            }
        }
    }

    // Split STC stego_bits into per-frame OverrideMaps.
    let mut per_frame_plans: Vec<OverrideMap> =
        (0..harvest.frame_starts.len())
            .map(|_| OverrideMap::new())
            .collect();
    for i in 0..n_used {
        if embed.stego_bits[i] != harvest.cover_bits[i] {
            let (frame_idx, cursor) = harvest.frame_cursor_at(i);
            per_frame_plans[frame_idx].set(cursor, embed.stego_bits[i] as u16);
        }
    }
    // Out-of-range shadow override entries.
    for slots in &gop_shadow_positions {
        for &(local_idx, bit) in slots {
            if local_idx >= n_used && bit != harvest.cover_bits[local_idx] {
                let (frame_idx, cursor) = harvest.frame_cursor_at(local_idx);
                per_frame_plans[frame_idx].set(cursor, bit as u16);
            }
        }
    }

    // Per-frame replay + splice.
    let mut output: Vec<u8> = Vec::new();
    for (frame_idx, (natural_packet, recording)) in frames.iter().enumerate() {
        let tile = &recording.tiles[0];
        let mut sink = WriterEncoder::new();
        replay_with_overrides(
            &tile.storage,
            &tile.bit_positions,
            &per_frame_plans[frame_idx],
            &mut sink,
        );
        let stego_tile_bytes = sink.done();

        let final_packet = if stego_tile_bytes.len() == recording.tile_group_len {
            let mut packet = natural_packet.clone();
            let dst = &mut packet[recording.tile_group_offset
                ..recording.tile_group_offset + recording.tile_group_len];
            dst.copy_from_slice(&stego_tile_bytes);
            packet
        } else {
            rebuild_obu_with_stego_tile_group(
                natural_packet,
                recording,
                &stego_tile_bytes,
            )
        };
        output.extend_from_slice(&final_packet);
    }
    Ok(Some(output))
}

// ────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────

fn expected_i420_size(width: u32, height: u32) -> usize {
    let y = (width as usize) * (height as usize);
    let uv = ((width as usize) / 2) * ((height as usize) / 2);
    y + 2 * uv
}

/// Encode one GOP naturally (mirror of session.rs::encode_one_gop_multi
/// but inline-d here to avoid an inter-module pub dependency).
///
/// Visibility raised to `pub(super)` at T2.4 so
/// `Av1StreamingProbeSession::flush_current_gop` can use it directly
/// (skipping `process_one_gop_pass1`'s J-UNIWARD harvest_gop pass —
/// the probe only needs the natural-encode bit_tags, not per-position
/// costs).
pub(super) fn encode_gop_natural(
    gop_yuv: &[u8],
    frames_in_gop: u32,
    params: Av1StreamingEncodeParams,
) -> Result<Vec<(Vec<u8>, PhasmFrameRecording<u8>)>, Av1StegoError> {
    let w = params.width as usize;
    let h = params.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let frame_size = y_size + 2 * uv_size;

    let mut config = EncoderConfig {
        width: w,
        height: h,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: params.quantizer,
        ..Default::default()
    };
    config.low_latency = true;
    apply_phasm_av1_speed(&mut config);
    let config = Arc::new(config);
    let mut sequence = Sequence::new(&config);
    apply_phasm_av1_sequence(&mut sequence);
    let sequence = Arc::new(sequence);

    let yuvs: Vec<Arc<phasm_rav1e::Frame<u8>>> = (0..frames_in_gop as usize)
        .map(|i| {
            let off = i * frame_size;
            let mut frame_in = make_frame::<u8>(w, h, ChromaSampling::Cs420);
            frame_in.planes[0].copy_from_raw_u8(&gop_yuv[off..off + y_size], w, 1);
            frame_in.planes[1].copy_from_raw_u8(
                &gop_yuv[off + y_size..off + y_size + uv_size],
                w / 2,
                1,
            );
            frame_in.planes[2].copy_from_raw_u8(
                &gop_yuv[off + y_size + uv_size..off + frame_size],
                w / 2,
                1,
            );
            Arc::new(frame_in)
        })
        .collect();

    Ok(encode_gop_with_phasm_tee::<u8>(&yuvs, config, sequence))
}

/// Harvested per-GOP Tier-1 cover + cost + per-bit back-index.
///
/// `pub(super)` so `Av1StreamingEncodeSession::Av1WholeVideoState` can
/// hold a `Vec<GopHarvest>` as a Pass-1 accumulator. Field access stays
/// internal to this module — sessions just push opaque entries.
pub(super) struct GopHarvest {
    cover_bits: Vec<u8>,
    costs: Vec<f32>,
    /// Per-bit (frame_idx, frame_cursor) — same shape as the
    /// `combined_index` in `av1_stego_encode_one_gop_with_shadows_parity`.
    combined_index: Vec<(usize, u64)>,
    /// Inclusive index where each frame's cover bits start (used to
    /// derive frame_cursor_at if combined_index becomes too large to
    /// stash on big resolutions; currently keeps the full
    /// per-bit vector for simplicity).
    frame_starts: Vec<usize>,
}

impl GopHarvest {
    fn frame_cursor_at(&self, idx: usize) -> (usize, u64) {
        self.combined_index[idx]
    }
}

/// Per-frame Pass-1 work for one frame's recording: walks the
/// bit_positions/tags/meta to pull AC_COEFF_SIGN + GOLOMB_TAIL_LSB
/// entries, packs the reconstructed planes, and runs the J-UNIWARD
/// cost compute. Returns the frame's cursors, cover bits, and per-
/// position costs in walk order. Pure function — no shared state —
/// so the outer harvest_gop can call it in parallel across the
/// frames of a GOP under `feature = "parallel"`.
fn process_one_frame_harvest(
    recording: &PhasmFrameRecording<u8>,
) -> Result<(Vec<u64>, Vec<u8>, Vec<f32>), Av1StegoError> {
    if recording.tiles.is_empty() {
        return Err(Av1StegoError::EmptyRecording);
    }
    let tile = &recording.tiles[0];

    let mut frame_cursors: Vec<u64> = Vec::new();
    let mut frame_bits: Vec<u8> = Vec::new();
    let mut frame_metas: Vec<AcSignMeta> = Vec::new();

    for (cursor, ((&(_, value), &tag), &meta)) in tile
        .bit_positions
        .iter()
        .zip(tile.bit_tags.iter())
        .zip(tile.bit_meta.iter())
        .enumerate()
    {
        if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
            frame_cursors.push(cursor as u64);
            frame_bits.push(value as u8);
            frame_metas.push(meta);
        }
    }

    let frame_planes = pack_visible_planes(&recording.reconstructed_planes);
    let av1_positions: Vec<Av1FramePosition> = frame_metas
        .iter()
        .map(|m| Av1FramePosition {
            plane: m.plane,
            plane_px_x: m.plane_px_x,
            plane_px_y: m.plane_px_y,
            tx_width_log2: m.tx_width_log2,
            tx_height_log2: m.tx_height_log2,
            tx_type: m.tx_type,
            scan_pos: m.scan_pos,
            coeff_magnitude: m.coeff_magnitude,
        })
        .collect();
    let frame_costs = compute_av1_uniward_costs_with_state(
        &frame_planes,
        &av1_positions,
        recording.frame_qindex,
        Some(recording.loop_filter_state),
    );

    Ok((frame_cursors, frame_bits, frame_costs))
}

fn harvest_gop(
    per_frame: &[(Vec<u8>, PhasmFrameRecording<u8>)],
) -> Result<GopHarvest, Av1StegoError> {
    // Per-frame Pass-1 work is independent across frames within the
    // same GOP — each frame has its own recording, planes, and qindex.
    // Run the expensive part (pack_visible_planes + J-UNIWARD cost
    // compute, audit P1) in parallel under `feature = "parallel"`;
    // fall back to sequential on no-parallel builds. Output order is
    // preserved by collect-into-Vec; the subsequent accumulation
    // stays sequential to keep the cover bit cursor monotonic.
    #[cfg(feature = "parallel")]
    let per_frame_outputs: Vec<(Vec<u64>, Vec<u8>, Vec<f32>)> = per_frame
        .par_iter()
        .map(|(_, recording)| process_one_frame_harvest(recording))
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let per_frame_outputs: Vec<(Vec<u64>, Vec<u8>, Vec<f32>)> = per_frame
        .iter()
        .map(|(_, recording)| process_one_frame_harvest(recording))
        .collect::<Result<Vec<_>, _>>()?;

    let mut cover_bits: Vec<u8> = Vec::new();
    let mut costs: Vec<f32> = Vec::new();
    let mut combined_index: Vec<(usize, u64)> = Vec::new();
    let mut frame_starts: Vec<usize> = Vec::with_capacity(per_frame.len());

    for (frame_idx, (frame_cursors, frame_bits, frame_costs)) in
        per_frame_outputs.into_iter().enumerate()
    {
        frame_starts.push(cover_bits.len());
        for c in &frame_cursors {
            combined_index.push((frame_idx, *c));
        }
        cover_bits.extend_from_slice(&frame_bits);
        costs.extend_from_slice(&frame_costs);
    }

    Ok(GopHarvest {
        cover_bits,
        costs,
        combined_index,
        frame_starts,
    })
}

// `verify_shadows_round_trip` (whole-clip dav1d walk + av1_shadow_extract
// per shadow) deleted at WV.7.6 — replaced by Av1StreamingShadowVerify
// in shadow.rs, which walks emitted GOP bytes per-GOP and feeds them
// into per-shadow Av1ShadowBitTopN heaps. Drops the whole-clip
// cover_bits Vec (~4 GB at 30 min) AND the whole-clip priority_slots
// Vec inside av1_shadow_extract (~50 GB at 30 min) from verify-time
// peak.
