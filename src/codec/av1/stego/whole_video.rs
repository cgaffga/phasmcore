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
//! see; the cascade loop runs once-through, rebuilding each GOP's
//! replay data from its cached natural packets (`rebuild`).
//!
//! Design: see
//! `docs/design/video/av1/phase-c-wv-whole-video-shadow.md` § 3.2.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::{PhasmFrameLoopFilterState, PhasmFrameRecording, WriterEncoder};
use phasm_rav1e::phasm_stego::{
    encode_gop_with_phasm_tee, make_frame, AcSignMeta,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

/// Pass 1 cache budget — RAM + disk thresholds and the scratch-dir parent
/// both encode paths use to hold each GOP's natural packets between Pass 1
/// and Pass 2 (1–2.4 MB per 1080p GOP; Pass 2 rebuilds the rest with a
/// dav1d decode).
///
/// Callers construct this explicitly (`Pass1CacheBudget::new`) when they
/// know device-aware budgets (iOS / Android bridges query device RAM +
/// cache dir), or use `from_env_or_default` for CLI / dev parity with
/// the env-var-driven behaviour that preceded 2026-10-19.
///
/// Semantics are unchanged from the preceding env-driven form:
///   - `ram_bytes = 0`   → disable the RAM cache
///   - `disk_bytes = 0`  → disable the disk spill (V1 behaviour)
///   - overflow past both tiers falls through to re-encoding the GOP from
///     the source in Pass 2, so clips of any length still finish.
#[derive(Debug, Clone)]
pub struct Pass1CacheBudget {
    pub ram_bytes: usize,
    pub disk_bytes: usize,
    pub scratch_parent: std::path::PathBuf,
}

impl Pass1CacheBudget {
    /// Default desktop budget: 500 MB RAM, 8 GB disk, scratch under
    /// `std::env::temp_dir()`. Baked here so no caller has to know
    /// about `$PHASM_TMP` or honour the env fall-through — it just
    /// gets the same answer the pre-2026-10-19 inline code produced.
    pub const DEFAULT_RAM_BYTES: usize = 500 * 1024 * 1024;
    pub const DEFAULT_DISK_BYTES: usize = 8 * 1024 * 1024 * 1024;

    /// Explicit budget. Mobile bridges use this with device-aware
    /// `ram_bytes` + a bridge-chosen cache dir.
    pub fn new(
        ram_bytes: usize,
        disk_bytes: usize,
        scratch_parent: std::path::PathBuf,
    ) -> Self {
        Self {
            ram_bytes,
            disk_bytes,
            scratch_parent,
        }
    }

    /// CLI / dev / test parity: respect the three
    /// `PHASM_AV1_PASS1_CACHE_*` env vars when set, otherwise fall
    /// back to the 500 MB / 8 GB / `temp_dir()` default. This is a
    /// pure function of the environment at call time — pass the
    /// resulting value into the encode fn once.
    pub fn from_env_or_default() -> Self {
        let ram_bytes = std::env::var("PHASM_AV1_PASS1_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(Self::DEFAULT_RAM_BYTES);
        let disk_bytes = std::env::var("PHASM_AV1_PASS1_CACHE_DISK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(Self::DEFAULT_DISK_BYTES);
        let scratch_parent = std::env::var("PHASM_AV1_PASS1_CACHE_DISK_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        Self {
            ram_bytes,
            disk_bytes,
            scratch_parent,
        }
    }
}

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
///   - `encode_gop_natural` (this file): every GOP of both encode paths,
///     1-frame GOPs included — byte-identical to the retired single-frame
///     keyframe primitive (checked 2026-10-24 at 64×64 … 1080p, noise and
///     real content, before it was deleted)
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
    // Perf #2 scaffold (2026-10-18) — override rdo_lookahead_frames so
    // the dedicated stealth audit can measure speed=9 w/ lookahead=1
    // without a rebuild. speed=9 inherits lookahead=10 from speed=8
    // (`vendor/phasm-rav1e/src/api/config/speedsettings.rs:169`); under
    // phasm's `low_latency=true` + `multiref=false` posture that state
    // is largely wasted — the hypothesis the audit tests.
    // Env unset → preserves the speed preset's shipped value. See
    // `docs/design/video/av1/stealth-audit-perf-levers-plan-2026-10-18.md`.
    if let Some(lookahead) = std::env::var("PHASM_AV1_LOOKAHEAD")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        config.speed_settings.rdo_lookahead_frames = lookahead;
    }
}

/// Phasm AV1 production sequence-level settings.
///
/// `enable_large_lru` default: **true** — matches rav1e's `Sequence`
/// constructor default (`vendor/phasm-rav1e/src/encoder.rs:315`).
///
/// Note: at phasm's shipping speed preset (speed=9), this field is
/// INERT. rav1e's preset sets `speed_settings.lrf = false` for
/// speed ≥ 8 (`vendor/phasm-rav1e/src/api/config/speedsettings.rs:170`),
/// which propagates to `sequence.enable_restoration = false`
/// (`vendor/phasm-rav1e/src/encoder.rs:313`). Every live consumer of
/// `enable_large_lru` (`RestorationState::new` in `lrf.rs:1329`,
/// `write_frame_lrf` in `header.rs:1121-1162`, per-SB `write_lrf` at
/// `encoder.rs:3443-3450`) is gated on `enable_restoration`. With LRF
/// off, nothing LRU-related reaches the bitstream regardless of this
/// value.
///
/// Flipped on 2026-10-20 anyway because: (a) the 2026-10-19 full-corpus
/// audit confirmed the forensic axes are invariant (byte-identical
/// BS/BL/Y/UV/INTER_MODE/MV_MAG histograms across all 4 fixtures —
/// unsurprising once we knew the knob is inert), (b) matching rav1e's
/// constructor default removes a divergent explicit override, (c)
/// future-proofs against any speed < 8 preset where LRF would turn on.
/// Investigation that caught the "likely alignment win" framing error:
/// `docs/design/video/av1/stealth-audit-perf-levers-verdict-2026-10-18.md`
/// § 6 (corrected 2026-10-20).
///
/// Override with `PHASM_AV1_LARGE_LRU=0` to force the pre-2026-10-20
/// explicit-false behaviour (useful inside experiment harnesses that
/// want to pin the historical posture). `=1` is a no-op against the
/// new default; and at speed=9 it is a no-op regardless.
///
/// Called by every encoder construction site that builds a fresh
/// `Sequence`.
pub(super) fn apply_phasm_av1_sequence(sequence: &mut Sequence) {
    let enable = std::env::var("PHASM_AV1_LARGE_LRU")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|n| n != 0)
        .unwrap_or(true);
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
    build_chunk_frame_v3_1, build_first_chunk_frame_v3_1,
};
use crate::stego::cost::av1_uniward::{
    compute_av1_uniward_costs_with_state, Av1FramePosition, FramePlanes,
};
use crate::stego::stc::embed::stc_embed;
use crate::stego::stc::hhat;
use crate::stego::{frame, payload};

use super::encode_progress::{
    Av1EncodePhase, Av1EncodeProgress, RecomputeForecast, WorkProgress, EMBED_WEIGHT,
};
use super::orchestrator::{
    pack_cost_planes, rebuild_obu_with_stego_tile_group, Av1StegoError,
};

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

/// Shadow-occupied cover positions per GOP — wet for the primary STC, so
/// the primary plan sizes each GOP net of them. `offsets` / `lens` are the
/// per-GOP cover offsets into the union cover and the cover counts.
fn shadow_wet_bits_per_gop(
    shadow_states: &[Av1ShadowState],
    offsets: &[usize],
    lens: &[usize],
) -> Vec<usize> {
    let mut wet = vec![0usize; lens.len()];
    for state in shadow_states {
        for slot in state.positions.iter().take(state.n_total) {
            // Last GOP starting at or before this position (later GOPs win
            // ties, so empty GOPs sharing an offset are skipped).
            let g = offsets.partition_point(|&o| o <= slot.cover_index).saturating_sub(1);
            if slot.cover_index < offsets[g] + lens[g] {
                wet[g] += 1;
            }
        }
    }
    wet
}

/// WV.7.5 — streaming entry point for the AV1 whole-video shadow encode.
///
/// Production-shaped per-GOP streaming flow (no whole-clip
/// `per_gop_natural[]` accumulator). Design B from the WV.7.4→7.5
/// design analysis:
///
/// **Pass 1** (once, parity-independent, [`run_pass1`]): pull each GOP's
/// YUV from `source`, natural-encode it, count its cover bits
/// and cache its natural packets (1–2.4 MB per 1080p GOP) in the bounded
/// Pass-1 cache.
///
/// **Cascade** (per parity tier, retried until verify passes):
/// 1. Re-run the shadow position sweep over the cached `cover_lens`
///    (no encode work — sweep is heap operations only). Yields
///    per-shadow `Av1ShadowState` for THIS parity tier.
/// 2. For each GOP: rebuild its recordings + harvest from the cached
///    packets with one dav1d decode ([`pass1_entry_for`]; a GOP over the
///    cache budget is re-encoded from `source` instead) →
///    `cascade_one_gop` → emit per-GOP bytes → drop the entry before
///    advancing.
/// 3. Verify the assembled clip per GOP (`Av1StreamingShadowVerify`); on
///    success return; on failure advance to the next parity tier.
///
/// **Memory at 30 min × 1080p × 8 shadows:** one GOP's YUV (~93 MB, Pass
/// 1 and re-encodes only) + rav1e encode transient (~500 MB peak, Pass 1)
/// + one GOP's rebuild + windowed harvest in Pass 2 (~470 MB peak at
/// 1080p, measured 2026-10-24) + the Pass-1 cache
/// (bounded by `Pass1CacheBudget`, ~1–2.4 MB per GOP) + output Vec
/// (~1.1 GB — O(clip)).
///
/// **Wall cost:** Pass 1 = 1× per-GOP encode total. Each cascade tier adds
/// ~4–5 % of that per cached GOP (decode + harvest + STC + replay; see
/// `encode_progress::EMBED_WEIGHT`) and a full encode per GOP the cache
/// could not hold.
///
/// The shadow priority is position-local + parity-independent (only
/// `n_total` changes per tier — the priority order over the union
/// cover is fixed by `perm_seed`). So Pass 1's packet cache serves every
/// parity tier; each tier rebuilds and harvests its GOPs again, and only
/// the heap capacity changes.
pub fn av1_encode_with_shadows_streaming(
    source: &mut dyn Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    shadows: &[(&str, &[u8])],
    shadow_parity_len_floor: usize,
    pass1_cache_budget: Pass1CacheBudget,
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
    // Refuse what no decoder will read back, before Pass 1 spends a full
    // natural encode on it.
    if primary_framed.len() > frame::VIDEO_MAX_FRAME_BYTES {
        return Err(Av1StegoError::Stego(
            crate::stego::error::StegoError::MessageTooLarge,
        ));
    }
    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);
    let n_gops_usize = n_gops as usize;

    // ── Helper: per-GOP frame count, including the trailing partial.
    let frames_in_gop = |gop_idx: u32| -> u32 {
        let start = gop_idx * gop_size;
        let end = (start + gop_size).min(n_frames);
        end - start
    };

    // ── Pass 1 (`run_pass1`): per-GOP natural encode; each GOP's natural
    // packets go to the bounded Pass-1 cache, and only the per-GOP cover
    // counts stay in RAM outside it.
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
    // Pass 2 rebuilds each tier's per-GOP recordings and harvest from the
    // cached packets (one dav1d decode per GOP) instead of re-encoding;
    // only a GOP over both budgets of `pass1_cache_budget` (RAM, then a
    // scratch-dir spill) is re-encoded from the source. See the "Pass 1 →
    // Pass 2" section below.
    let (mut pass1_cache, per_gop_cover_lens) =
        run_pass1(source, n_frames, params, &pass1_cache_budget, &mut |_, _| {})?;

    // Per-GOP cover offsets into the union (= prefix sum of cover lens).
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

    let structural_key = crate::stego::crypto::derive_structural_key(primary_passphrase)
        .map_err(Av1StegoError::Stego)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    // Parity-tier cascade. Each tier re-runs Pass 2 from the Pass-1
    // cache, with its own primary plan (the shadow positions it adds are
    // wet for the primary STC).
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

        // Primary plan for THIS tier: each GOP's capacity net of the
        // shadow positions it holds (wet for the primary STC). A higher
        // tier only adds wet positions, so a primary that does not fit
        // here fits no later tier either — `plan_primary`'s
        // `MessageTooLarge` ends the encode.
        let wet = shadow_wet_bits_per_gop(&shadow_states, &per_gop_cover_offsets, &per_gop_cover_lens);
        let mut caps = super::primary_plan::gop_primary_caps(&per_gop_cover_lens, &wet);
        let mut attempts = 0usize;
        let tier_output = loop {
            let mut plan = super::primary_plan::plan_primary(&caps, primary_framed.len())?;
            attempts += 1;

            // WV.7.6 — streaming verify: as each GOP emits, walk its
            // bytes via dav1d, push the harvested cover bits (with global
            // cover_index) into per-shadow Av1ShadowBitTopN heaps. After
            // all GOPs, finish() drains heaps + runs RS-decode + AES auth
            // per shadow. No whole-clip cover_bits / priority_slots Vec
            // materialised — drops ~55 GB of verify-time RAM at 30 min
            // vs the legacy verify_shadows_round_trip path. One verifier
            // per sweep: a re-planned sweep starts it afresh.
            let mut streaming_verify = Av1StreamingShadowVerify::new(
                pre_selections_for_verify.clone(),
                passphrases_for_verify.clone(),
            );

            // Per-GOP cascade. The source is only read for a GOP the
            // Pass-1 cache could not hold; a fresh puller per sweep restarts
            // it from GOP 0 (the source contract).
            let t_cascade = std::time::Instant::now();
            let mut puller = LazyGopPuller::new(&mut *source);
            let mut output: Vec<u8> = Vec::new();
            let mut sent = 0usize;
            let mut replan = false;
            let mut c_sum_entry = std::time::Duration::ZERO;
            let mut c_sum_cascade = std::time::Duration::ZERO;
            let mut c_sum_verify = std::time::Duration::ZERO;
            for gop_idx in 0..n_gops {
                // A cancel stops before the next GOP's rebuild or re-encode.
                crate::stego::progress::check_cancelled().map_err(Av1StegoError::Stego)?;
                // The GOP's entry — rebuilt from its cached natural packets,
                // or re-encoded on a miss — carries the frames and the
                // J-UNIWARD harvest. Both are parity-independent (perm_seed
                // fixes the shadow priority; parity_len only changes heap
                // capacity, re-run via the sweep), so every tier reuses
                // Pass 1's cache.
                let g = gop_idx as usize;
                let t_entry = std::time::Instant::now();
                let (entry, _) = pass1_entry_for(
                    &mut pass1_cache,
                    &mut puller,
                    frames_in_gop(gop_idx),
                    params,
                    gop_idx,
                    per_gop_cover_lens[g],
                )?;
                let entry_dt = t_entry.elapsed();
                c_sum_entry += entry_dt;

                // Every GOP runs the cascade — it also stamps the shadow
                // bits — so once the primary is complete the GOP carries a
                // header-only chunk the decoder never reaches.
                let planned = if sent == primary_framed.len() { 0 } else { plan[g] };
                let mut want = planned;
                let t_casc = std::time::Instant::now();
                let cascade_result = loop {
                    let inputs = Av1CascadeGopInputs {
                        gop_idx: g,
                        gop_offset: per_gop_cover_offsets[g],
                        gop_n: per_gop_cover_lens[g],
                        harvest: &entry.harvest,
                        frames: &entry.frames,
                        chunk: &primary_framed[sent..sent + want],
                    };
                    let shared = Av1CascadeGopShared {
                        total_message_bytes,
                        shadow_states: &shadow_states,
                        hhat_seed: &hhat_seed,
                    };
                    if let Some(bytes) = cascade_one_gop(inputs, shared)? {
                        break Some(bytes);
                    }
                    // `stc_embed` also returns `None` when the user cancelled;
                    // that is not a capacity signal, so never learn from it.
                    crate::stego::progress::check_cancelled().map_err(Av1StegoError::Stego)?;
                    if want == 0 {
                        break None;
                    }
                    want = super::primary_plan::shrink_step(want);
                };
                let casc_dt = t_casc.elapsed();
                c_sum_cascade += casc_dt;
                let Some(per_gop_bytes) = cascade_result else {
                    // Not even a header-only chunk fits. Every GOP of a
                    // shadow encode carries a chunk (the cascade also stamps
                    // its shadow bits), and higher parity tiers only add wet
                    // positions — no plan and no tier can succeed.
                    return Err(Av1StegoError::InvalidPacket(format!(
                        "GOP {gop_idx} has too little cover ({} bits) to carry a chunk frame; \
                         shadow messages need every GOP to hold one",
                        per_gop_cover_lens[g]
                    )));
                };
                let t_ver = std::time::Instant::now();
                streaming_verify.push_gop_bytes(&per_gop_bytes)?;
                let ver_dt = t_ver.elapsed();
                c_sum_verify += ver_dt;
                if prof {
                    eprintln!(
                        "[AV1_PROF SHADOW] Cascade gop {}: entry={:?}, cascade={:?}, verify_push={:?}, packet={} B, chunk={want}/{planned} B",
                        gop_idx, entry_dt, casc_dt, ver_dt, per_gop_bytes.len()
                    );
                }
                output.extend_from_slice(&per_gop_bytes);
                sent += want;
                if want < planned {
                    // Learned: this GOP carries less than planned. Carry the
                    // shortfall forward, or re-plan the whole message.
                    super::primary_plan::learn_cap(&mut caps, g, Some(want));
                    match super::primary_plan::replan_from(&caps, g + 1, primary_framed.len() - sent) {
                        Some(rest) => plan[g + 1..].copy_from_slice(&rest),
                        None => {
                            replan = true;
                            break;
                        }
                    }
                }
            }

            if replan || sent != primary_framed.len() {
                if attempts >= MAX_PRIMARY_PASSES {
                    return Err(super::primary_plan::message_too_large(&caps, primary_framed.len()));
                }
                if prof {
                    eprintln!("[AV1_PROF SHADOW] Tier parity_len={parity_len}: re-planning the primary");
                }
                continue;
            }
            let t_verfin = std::time::Instant::now();
            let ok = streaming_verify.finish();
            let verfin_dt = t_verfin.elapsed();
            if prof {
                eprintln!(
                    "[AV1_PROF SHADOW] Tier parity_len={} cascade_total={:?}, sum_entry={:?}, sum_cascade={:?}, sum_verify_push={:?}, verify_finish={:?} → ok={}",
                    parity_len, t_cascade.elapsed(),
                    c_sum_entry, c_sum_cascade, c_sum_verify, verfin_dt, ok,
                );
            }
            break ok.then_some(output);
        };

        match tier_output {
            Some(output) => {
                if prof {
                    let cs = pass1_cache.stats();
                    let hits = cs.ram_hits.saturating_add(cs.disk_hits);
                    let total_lookups = hits.saturating_add(cs.misses);
                    let hit_pct = if total_lookups > 0 {
                        (hits as f64 * 100.0) / (total_lookups as f64)
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[AV1_PROF SHADOW] TOTAL: {:?}, output={} B (succeeded at tier parity_len={}); \
                         pass1_cache ram_hits={} disk_hits={} misses={} ({:.1}% hit)",
                        t_total.elapsed(), output.len(), parity_len,
                        cs.ram_hits, cs.disk_hits, cs.misses, hit_pct,
                    );
                }
                return Ok(output);
            }
            None => {
                if prof {
                    eprintln!(
                        "[AV1_PROF SHADOW] Tier parity_len={} FAILED in {:?}",
                        parity_len, t_tier.elapsed()
                    );
                }
            }
        }
    }

    Err(Av1StegoError::Stego(
        crate::stego::error::StegoError::FrameCorrupted,
    ))
}

/// Most Pass-2 sweeps one [`av1_encode_streaming`] runs: the first, plus a
/// re-plan each time a sweep learns that a GOP carries less than planned
/// and the shortfall cannot be carried forward. Every re-plan lowers at
/// least one GOP's capacity, so this only bounds pathological clips —
/// they end in `MessageTooLarge`.
///
/// Each sweep rebuilds every cached GOP it embeds (a dav1d decode +
/// harvest, a few % of its encode) but re-encodes — a full rav1e pass — any
/// GOP the Pass-1 cache could not hold. On an overflowing cache the bound
/// therefore matters for time: up to this many re-encodes of those GOPs.
const MAX_PRIMARY_PASSES: usize = 8;

/// MOBOOM.T3.1 — streaming AV1 encode for the NO-SHADOW case.
///
/// Sibling of [`av1_encode_with_shadows_streaming`] for clips without
/// shadow embedding, driving an [`Av1GopYuvSource`] that delivers each
/// GOP's tight-I420 YUV in a fast burst between rav1e calls (so
/// AVAssetReader / MediaCodec reads never interleave with an encode —
/// the T3.1 "Operation Interrupted" fix).
///
/// **Two passes, proportional primary (2026-10-23).** Pass 1
/// ([`run_pass1`]) natural-encodes every GOP once, caching its natural
/// packets in the bounded Pass-1 cache and counting its exact cover. The primary is then planned per GOP in proportion to that
/// cover ([`primary_plan`](super::primary_plan)), and Pass 2 embeds each
/// GOP after rebuilding its replay data and harvest from the cached packets
/// with one dav1d decode ([`pass1_entry_for`], 2026-10-24). Until 2026-10-23 this entry split the primary
/// evenly in a single pass, so a GOP with little cover — almost always the
/// trailing partial GOP — was asked for a full share and the encode failed
/// there after every earlier GOP was done; measured in
/// `docs/design/video/av1/capacity-split-and-payload-limits-2026-10-23.md`.
/// The encode work is one natural encode per GOP plus a decode per GOP. A message larger than the measured capacity is
/// refused after Pass 1, before any embed.
///
/// If a GOP's STC is infeasible despite the plan (the payload ratio is an
/// estimate), its chunk shrinks step by step and the shortfall is
/// re-planned over the later GOPs; only a shortfall nothing can absorb is
/// `MessageTooLarge`. GOPs after the message is complete are emitted
/// natural — the decoder stops once it has `total_message_bytes`.
///
/// `primary_framed` is the encrypt + `frame::build_frame` output for the
/// primary message (the caller frames it, as for the shadow entry).
/// `on_progress` gets an [`Av1EncodeProgress`] after every GOP of each pass:
/// the phase, the GOP, and a whole-encode fraction proportional to the work
/// (`encode_progress`) — Pass 2 costs a few % of Pass 1 per cached GOP
/// (rebuild + embed) but a full re-encode per GOP the cache could not hold,
/// which is forecast while Pass 1 fills the cache. The fraction never decreases and is 1.0 only on
/// the final event.
///
/// **Memory profile.** Per-GOP transient: Pass 1 one GOP's YUV (~90 MB at
/// 1080p × gop=30) + one rav1e encode state (~120 MB); Pass 2 one GOP's
/// rebuild + windowed harvest (~470 MB peak at 1080p, ~1.0 GB at
/// 2720×1530; `HARVEST_WINDOW`). Between the passes: the Pass-1 cache of natural
/// packets, bounded by `pass1_cache_budget` (spills to disk, then
/// re-encodes). Pass 2 reads the source only for GOPs it re-encodes. Persistent: the output Vec of stego OBU bytes (O(clip);
/// ~1 GB at 30 min × 1080p — true streaming needs a GOP-incremental muxer).
pub fn av1_encode_streaming(
    source: &mut dyn Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    pass1_cache_budget: Pass1CacheBudget,
    mut on_progress: Option<&mut dyn FnMut(Av1EncodeProgress)>,
) -> Result<Vec<u8>, Av1StegoError> {
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_encode_streaming: n_frames == 0".into(),
        ));
    }
    // Refuse what no decoder will read back, before any GOP is encoded.
    if primary_framed.len() > frame::VIDEO_MAX_FRAME_BYTES {
        return Err(Av1StegoError::Stego(
            crate::stego::error::StegoError::MessageTooLarge,
        ));
    }
    let n_gops = n_frames.div_ceil(params.gop_size.max(1));
    if n_gops > u16::MAX as u32 {
        return Err(Av1StegoError::InvalidPacket(format!(
            "av1_encode_streaming: derived n_gops {n_gops} out of range [1..={}]",
            u16::MAX
        )));
    }

    // hhat seed from the primary passphrase — the same derive the decoder
    // runs; one per encode, reused across GOPs.
    let structural_key = crate::stego::crypto::derive_structural_key(primary_passphrase)
        .map_err(Av1StegoError::Stego)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    let prof = std::env::var("PHASM_AV1_PROFILE").map(|v| v == "1").unwrap_or(false);
    let t_total = std::time::Instant::now();

    // ── Progress, in frame-encode units: Pass 1 is one unit per frame; a
    // Pass-2 sweep is EMBED_WEIGHT per frame plus a full unit per frame of
    // every GOP the cache could not hold (re-encoded). The re-encode share is
    // forecast as Pass 1 fills the cache and exact once Pass 1 is done.
    let gop_size = params.gop_size.max(1);
    let gop_frames: Vec<u32> = (0..n_gops).map(|g| gop_frame_count(n_frames, gop_size, g)).collect();
    let all_frames: f64 = gop_frames.iter().map(|&f| f as f64).sum();
    let one_sweep = all_frames * EMBED_WEIGHT;
    let mut progress = WorkProgress::new(all_frames + one_sweep);
    let mut forecast = RecomputeForecast::new(
        pass1_cache_budget.ram_bytes,
        pass1_cache_budget.disk_bytes,
        gop_frames.clone(),
    );
    let mut emit = |fraction: f32, phase: Av1EncodePhase, gop: u32| {
        if let Some(cb) = on_progress.as_mut() {
            cb(Av1EncodeProgress { fraction, phase, gop, n_gops });
        }
    };

    // ── Pass 1: natural encode of every GOP; packets cached, cover counted.
    let (mut pass1_cache, cover_lens) = run_pass1(
        source,
        n_frames,
        params,
        &pass1_cache_budget,
        &mut |g, cache| {
            forecast.observe(cache.stats());
            progress.set_total(all_frames + one_sweep + forecast.predicted_recompute_frames());
            let fraction = progress.advance(gop_frames[g as usize] as f64);
            emit(fraction, Av1EncodePhase::Encode, g + 1);
        },
    )?;
    // Exact from here on: which GOPs Pass 2 re-encodes is known.
    progress.set_total(all_frames + one_sweep + forecast.unstored_frames());

    // ── Plan + Pass 2: per-GOP chunk sizes from the measured cover, then
    // embed each GOP from its Pass-1 entry. Over capacity is refused by
    // the plan, before any embed. A sweep that learns a GOP carries less
    // than planned and cannot carry the shortfall forward re-plans with
    // what it learned and sweeps again (from the cache — no re-encode).
    let mut caps =
        super::primary_plan::gop_primary_caps(&cover_lens, &vec![0; cover_lens.len()]);
    let mut shrinks = 0usize;
    let mut output = None;
    for sweep in 0..MAX_PRIMARY_PASSES {
        if sweep > 0 {
            // A re-plan runs one more full sweep.
            progress.set_total(progress.done() + one_sweep + forecast.unstored_frames());
        }
        let plan = super::primary_plan::plan_primary(&caps, primary_framed.len())?;
        output = embed_primary_pass(
            source,
            &mut pass1_cache,
            n_frames,
            params,
            &cover_lens,
            &mut caps,
            plan,
            primary_framed,
            &hhat_seed,
            &mut shrinks,
            &mut |g, recomputed| {
                let frames = gop_frames[g as usize] as f64;
                if recomputed && forecast.stored(g as usize) {
                    // A miss the forecast did not expect (unreadable spill).
                    progress.set_total(progress.total() + frames);
                }
                let fraction =
                    progress.advance(frames * EMBED_WEIGHT + if recomputed { frames } else { 0.0 });
                emit(fraction, Av1EncodePhase::Embed, g + 1);
            },
        )?;
        if output.is_some() {
            break;
        }
    }
    let output = output
        .ok_or_else(|| super::primary_plan::message_too_large(&caps, primary_framed.len()))?;
    emit(progress.finish(), Av1EncodePhase::Embed, n_gops);

    if prof {
        let s = pass1_cache.stats();
        eprintln!(
            "[AV1_PROF NO_SHADOW] TOTAL: {:?}, output={} B, {n_gops} GOPs, {shrinks} chunk shrinks; \
             pass1_cache ram_hits={} disk_hits={} misses={}",
            t_total.elapsed(),
            output.len(),
            s.ram_hits,
            s.disk_hits,
            s.misses,
        );
    }
    Ok(output)
}

/// One Pass-2 sweep of [`av1_encode_streaming`]: embed every GOP from its
/// Pass-1 entry following `plan`. An infeasible chunk shrinks
/// ([`shrink_step`](super::primary_plan::shrink_step)) until it fits, the
/// GOP's achieved size is recorded in `caps`
/// ([`learn_cap`](super::primary_plan::learn_cap)), and the shortfall is
/// re-planned over the GOPs still to come. `Ok(None)` means it could not
/// be — the caller re-plans the whole message with the learned caps and
/// sweeps again. GOPs after the message is complete are emitted natural.
/// `on_gop_done(g, recomputed)` fires after GOP `g`; `recomputed` is true
/// when the Pass-1 cache did not hold it and it was natural-encoded again
/// (otherwise it was rebuilt from its cached packets). The source is only
/// read for such GOPs ([`LazyGopPuller`]).
#[allow(clippy::too_many_arguments)]
fn embed_primary_pass(
    source: &mut dyn Av1GopYuvSource,
    cache: &mut crate::stego::gop_cache::BoundedGopCache<Pass1Packets>,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    cover_lens: &[usize],
    caps: &mut [super::primary_plan::GopPrimaryCap],
    mut plan: Vec<usize>,
    primary_framed: &[u8],
    hhat_seed: &[u8; 32],
    shrinks: &mut usize,
    on_gop_done: &mut dyn FnMut(u32, bool),
) -> Result<Option<Vec<u8>>, Av1StegoError> {
    use super::primary_plan::{learn_cap, replan_from, shrink_step};

    let gop_size = params.gop_size.max(1);
    // ≤ VIDEO_MAX_FRAME_BYTES (checked by the caller), so it fits the
    // first chunk's u32 header.
    let total_message_bytes = primary_framed.len() as u32;
    let mut output: Vec<u8> = Vec::with_capacity((n_frames as usize).saturating_mul(1024));
    let mut sent = 0usize;
    let mut puller = LazyGopPuller::new(source);
    for gop_idx in 0..n_frames.div_ceil(gop_size) {
        // A cancel stops before the next GOP's rebuild or re-encode.
        crate::stego::progress::check_cancelled().map_err(Av1StegoError::Stego)?;
        let g = gop_idx as usize;
        let fig = gop_frame_count(n_frames, gop_size, gop_idx);

        let recomputed = if sent == primary_framed.len() {
            // Message complete: natural tail, which the decoder stops before.
            // Only the packets are needed — no rebuild, no harvest.
            let (packets, recomputed) = pass1_packets_for(cache, &mut puller, fig, params, gop_idx)?;
            for natural in &packets {
                output.extend_from_slice(natural);
            }
            recomputed
        } else {
            let (entry, recomputed) =
                pass1_entry_for(cache, &mut puller, fig, params, gop_idx, cover_lens[g])?;
            let planned = plan[g];
            let mut want = planned;
            let packet = loop {
                let inputs = Av1CascadeGopInputs {
                    gop_idx: g,
                    gop_offset: 0,
                    gop_n: cover_lens[g],
                    harvest: &entry.harvest,
                    frames: &entry.frames,
                    chunk: &primary_framed[sent..sent + want],
                };
                let shared = Av1CascadeGopShared {
                    total_message_bytes,
                    shadow_states: &[],
                    hhat_seed,
                };
                if let Some(packet) = cascade_one_gop(inputs, shared)? {
                    break packet;
                }
                // `stc_embed` also returns `None` when the user cancelled;
                // that is not a capacity signal, so never learn from it.
                crate::stego::progress::check_cancelled().map_err(Av1StegoError::Stego)?;
                if want == 0 {
                    // Not even a header-only chunk fits, and the decoder
                    // needs a chunk in every GOP until the message is
                    // complete: the window has to end before this GOP.
                    learn_cap(caps, g, None);
                    return Ok(None);
                }
                want = shrink_step(want);
                *shrinks += 1;
            };
            output.extend_from_slice(&packet);
            sent += want;
            if want < planned {
                learn_cap(caps, g, Some(want));
                match replan_from(caps, g + 1, primary_framed.len() - sent) {
                    Some(rest) => plan[g + 1..].copy_from_slice(&rest),
                    None => return Ok(None),
                }
            }
            recomputed
        };
        on_gop_done(gop_idx, recomputed);
    }
    // The plan sums to the message, so this holds; never return a stream
    // the decoder cannot complete.
    Ok((sent == primary_framed.len()).then_some(output))
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
    let harvest = harvest_gop(&pf, params.width, params.height)?;
    Ok((pf, harvest))
}

// ────────────────────────────────────────────────────────────────
// WV.7.4 — per-GOP cascade body (the API for WV.7.5+ streaming)
// ────────────────────────────────────────────────────────────────

/// Per-GOP inputs to [`cascade_one_gop`]. Borrows from the GOP's
/// [`Pass1Entry`] (rebuilt from its cached packets, or re-encoded, by
/// `pass1_entry_for`) plus the
/// chunk the primary plan assigned it.
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
    /// This GOP's primary message bytes (may be empty: a header-only
    /// chunk). GOP 0's chunk also carries the clip-level
    /// `total_message_bytes` header.
    pub chunk: &'a [u8],
}

/// Whole-clip-scope inputs shared across every GOP within one parity
/// tier of the cascade loop. Borrowed for the duration of one
/// [`cascade_one_gop`] call.
pub(super) struct Av1CascadeGopShared<'a> {
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
/// Extracted at WV.7.4 as a pure refactor, same per-GOP byte output;
/// now drives the cascade from a streaming `Av1GopYuvSource` instead
/// of the `per_gop_natural[]` accumulator that previously dominated
/// the O(clip) memory term at long clips (~75-100 KB/frame at 1080p).
/// Pre-WV.7 whole-clip entry point was removed CR#5 (2026-10-18).
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
        chunk,
    } = inputs;
    let Av1CascadeGopShared {
        total_message_bytes,
        shadow_states,
        hhat_seed,
    } = shared;

    // Build chunk_frame v3 payload bits for this GOP. First GOP
    // carries clip-level total_message_bytes; subsequent GOPs carry
    // only payload_len.
    let framed = if gop_idx == 0 {
        build_first_chunk_frame_v3_1(total_message_bytes, chunk)
    } else {
        build_chunk_frame_v3_1(chunk)
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

// ────────────────────────────────────────────────────────────────
// Pass 1 → Pass 2: cached natural packets, rebuilt by decoding
// ────────────────────────────────────────────────────────────────
//
// Pass 1 natural-encodes every GOP and counts its cover; Pass 2 embeds each GOP
// by replaying its recorded range-coder symbols with overrides, which
// needs the full recording (storage tuples, bit positions, harvest) —
// 300–400 MB per detailed 1080p GOP. Pass 1 therefore keeps only each
// GOP's natural packets (1–2.4 MB per 1080p GOP) in the bounded
// `BoundedGopCache` (RAM, then a scratch-dir disk spill), and Pass 2
// rebuilds the recording from one dav1d decode of them
// (`rebuild::rebuild_gop_recordings`, ~0.3 s per 1080p GOP) plus the
// harvest. A GOP over both budgets (or whose spill fails to read back) is
// natural-encoded again from the source instead — deterministic and
// byte-identical to Pass 1, just ~45 s per 1080p GOP. Measurements and the
// decision: `docs/design/video/av1/capacity-split-and-payload-limits-
// 2026-10-23.md` § 4.6–4.7.

/// Scratch-directory tag for the AV1 Pass 1 cache. Scopes `sweep_orphans` so
/// the AV1 sweep never deletes the H.264 shadow cache's trees, or vice versa.
pub const AV1_PASS1_CACHE_TAG: &str = "av1-pass1";

/// What Pass 1 caches for a GOP: its natural packets, one per frame. Pass 2
/// rebuilds everything else from them ([`pass1_entry_for`]).
#[derive(Clone)]
pub(super) struct Pass1Packets(pub Vec<Vec<u8>>);

/// Flat-pack format of [`Pass1Packets`], little-endian:
/// `[3][u64 n][n × (u64 len, len bytes)]`. Versions 1 and 2 held the full
/// recordings; any other version byte is a cache miss, never a misparse.
const PASS1_CACHE_FORMAT_V3: u8 = 3;

impl crate::stego::gop_cache::GopCachePayload for Pass1Packets {
    fn cached_size(&self) -> usize {
        self.0.iter().map(|p| p.len() + 24).sum::<usize>() + 24
    }

    fn flat_pack(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(9 + self.0.iter().map(|p| 8 + p.len()).sum::<usize>());
        out.push(PASS1_CACHE_FORMAT_V3);
        out.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        for p in &self.0 {
            out.extend_from_slice(&(p.len() as u64).to_le_bytes());
            out.extend_from_slice(p);
        }
        out
    }

    fn flat_unpack(bytes: &[u8]) -> Result<Self, String> {
        let take = |cur: &mut usize, len: usize| -> Result<&[u8], String> {
            let end = cur.checked_add(len).filter(|&e| e <= bytes.len());
            let end = end.ok_or_else(|| "truncated Pass1Packets".to_string())?;
            let s = &bytes[*cur..end];
            *cur = end;
            Ok(s)
        };
        let read_len = |cur: &mut usize| -> Result<usize, String> {
            let v = u64::from_le_bytes(take(cur, 8)?.try_into().unwrap());
            usize::try_from(v).map_err(|_| "Pass1Packets length overflow".to_string())
        };
        let mut cur = 0usize;
        let version = take(&mut cur, 1)?[0];
        if version != PASS1_CACHE_FORMAT_V3 {
            return Err(format!("unsupported Pass1Packets format version {version}"));
        }
        let n = read_len(&mut cur)?;
        // Every packet costs at least its 8-byte length: bound `n` by the
        // blob before allocating for it.
        if n > (bytes.len() - cur) / 8 {
            return Err("truncated Pass1Packets".to_string());
        }
        let mut packets = Vec::with_capacity(n);
        for _ in 0..n {
            let len = read_len(&mut cur)?;
            packets.push(take(&mut cur, len)?.to_vec());
        }
        if cur != bytes.len() {
            return Err("trailing bytes after Pass1Packets".to_string());
        }
        Ok(Pass1Packets(packets))
    }
}

/// Pass 2's working set for one GOP: the natural packets with their
/// recordings (rebuilt, or re-encoded on a cache miss) and the harvest the
/// embed runs against. Lives for one GOP; never cached.
#[derive(Clone)]
pub(super) struct Pass1Entry {
    pub frames: Pass1Payload,
    pub harvest: GopHarvest,
}

/// Per-frame `(natural packet, recording)` of one GOP, as
/// [`encode_gop_natural`] and `rebuild::rebuild_gop_recordings` return it.
pub type Pass1Payload = Vec<(Vec<u8>, PhasmFrameRecording<u8>)>;

/// What a harvested frame keeps instead of its planes: one shared 2×2
/// frame (the recording type requires planes; the cascade reads none).
fn dummy_planes() -> Arc<phasm_rav1e::Frame<u8>> {
    Arc::new(make_frame::<u8>(2, 2, ChromaSampling::Cs420))
}

/// Once the harvest is taken the cascade reads no planes; dropping them
/// keeps ~3 MB per 1080p frame out of Pass 2's working set.
fn drop_planes(frames: &mut [(Vec<u8>, PhasmFrameRecording<u8>)]) {
    let dummy = dummy_planes();
    for (_, recording) in frames.iter_mut() {
        recording.reconstructed_planes = dummy.clone();
    }
}

/// Pass 2's access to the source: untouched while every GOP comes from the
/// Pass-1 cache — on a phone that saves a second full decode of the input
/// video. The first GOP that has to be re-encoded rewinds the source and
/// pulls forward to it, discarding the GOPs in between, so the source
/// contract (rewind, then increasing indices, no gaps) holds. One puller
/// per Pass-2 sweep or parity tier.
pub(super) struct LazyGopPuller<'a> {
    source: &'a mut dyn Av1GopYuvSource,
    /// Next GOP the source will deliver; `None` until the first pull.
    next: Option<u32>,
}

impl<'a> LazyGopPuller<'a> {
    pub(super) fn new(source: &'a mut dyn Av1GopYuvSource) -> Self {
        Self { source, next: None }
    }

    /// GOP `g`'s YUV. `g` must not precede a GOP already pulled.
    pub(super) fn gop(&mut self, g: u32) -> Result<Vec<u8>, Av1StegoError> {
        let mut next = match self.next {
            Some(next) => next,
            None => {
                self.source.rewind()?;
                0
            }
        };
        if g < next {
            return Err(Av1StegoError::InvalidPacket(format!(
                "Pass 2: GOP {g} requested after GOP {} was pulled",
                next - 1
            )));
        }
        loop {
            let yuv = self.source.gop_yuv(next)?;
            next += 1;
            self.next = Some(next);
            if next > g {
                return Ok(yuv);
            }
        }
    }
}

/// Frames in GOP `gop_idx` of an `n_frames` clip; the trailing GOP may be
/// partial.
fn gop_frame_count(n_frames: u32, gop_size: u32, gop_idx: u32) -> u32 {
    let start = gop_idx * gop_size;
    (start + gop_size).min(n_frames) - start
}

/// Pass 1 of both encode paths: rewind `source`, then natural-encode every
/// GOP once, counting its cover bits and caching its natural packets under
/// `budget`.
/// Returns the cache and each GOP's cover-bit count — what the primary plan
/// is sized from. `on_gop_done(g, cache)` fires after GOP `g` is inserted
/// (the cache's stats show whether it was stored).
///
/// `source.rewind()` comes first: stateful sources (AVAssetReader,
/// MediaCodec, ffmpeg pipe) need it to open their decoder.
fn run_pass1(
    source: &mut dyn Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    budget: &Pass1CacheBudget,
    on_gop_done: &mut dyn FnMut(u32, &crate::stego::gop_cache::BoundedGopCache<Pass1Packets>),
) -> Result<(crate::stego::gop_cache::BoundedGopCache<Pass1Packets>, Vec<usize>), Av1StegoError> {
    let prof = std::env::var("PHASM_AV1_PROFILE").map(|v| v == "1").unwrap_or(false);
    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);
    let frame_size = expected_i420_size(params.width, params.height);

    let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
        n_gops as usize,
        budget.ram_bytes,
        budget.disk_bytes,
        budget.scratch_parent.clone(),
        AV1_PASS1_CACHE_TAG,
    );
    // Reclaim debris from a run that was killed mid-encode: `Drop` never runs
    // under `panic = "abort"`, jetsam or SIGKILL. Own-pid exempt, tag-scoped,
    // age-guarded — see `gop_cache::sweep_orphans`.
    crate::stego::gop_cache::sweep_orphans(
        &budget.scratch_parent,
        AV1_PASS1_CACHE_TAG,
        crate::stego::gop_cache::ORPHAN_MAX_AGE,
    );

    let t_pass1 = std::time::Instant::now();
    source.rewind()?;
    let mut cover_lens: Vec<usize> = Vec::with_capacity(n_gops as usize);
    for gop_idx in 0..n_gops {
        // A user cancel stops before the next rav1e encode, not after Pass 1.
        crate::stego::progress::check_cancelled().map_err(Av1StegoError::Stego)?;
        let fig = gop_frame_count(n_frames, gop_size, gop_idx);
        let t_pull = std::time::Instant::now();
        let gop_yuv = source.gop_yuv(gop_idx)?;
        let pull_dt = t_pull.elapsed();
        if gop_yuv.len() != fig as usize * frame_size {
            return Err(Av1StegoError::InvalidPacket(format!(
                "Pass 1: GOP {gop_idx} source returned {} bytes, expected {} \
                 ({fig} frames × {frame_size} B)",
                gop_yuv.len(),
                fig as usize * frame_size,
            )));
        }
        let t_proc = std::time::Instant::now();
        let frames = encode_gop_natural(&gop_yuv, fig, params)?;
        if frames.iter().any(|(_, recording)| recording.tiles.is_empty()) {
            return Err(Av1StegoError::EmptyRecording);
        }
        // Only the cover count is kept: Pass 2 harvests each GOP from its
        // rebuild, so a Pass-1 harvest would be thrown away.
        let cover = count_cover_bits(&frames);
        if prof {
            eprintln!(
                "[AV1_PROF PASS1] gop {gop_idx}: pull={pull_dt:?}, encode={:?}, cover_bits={cover}",
                t_proc.elapsed(),
            );
        }
        cover_lens.push(cover);
        // Infallible: failing to cache must never fail an encode — a GOP
        // over both budgets is re-encoded in Pass 2.
        cache.insert(gop_idx as usize, Pass1Packets(frames.into_iter().map(|(p, _)| p).collect()));
        on_gop_done(gop_idx, &cache);
    }
    if prof {
        let s = cache.stats();
        eprintln!(
            "[AV1_PROF PASS1] TOTAL {:?}: {} RAM / {} disk / {} overflow (of {n_gops} GOPs); \
             {} MB RAM, {} MB disk used",
            t_pass1.elapsed(),
            s.cached_in_ram,
            s.spilled_to_disk,
            s.overflowed,
            s.ram_used / (1024 * 1024),
            s.disk_used / (1024 * 1024),
        );
    }
    Ok((cache, cover_lens))
}

/// Pass 2's [`Pass1Entry`] for GOP `gop_idx`. A GOP whose packets the cache
/// holds is rebuilt from them with one dav1d decode
/// (`rebuild::rebuild_gop_recordings`) and harvested; a miss (never cached,
/// over both budgets, unreadable spill) is natural-encoded again from the
/// source through `puller` — deterministic and byte-identical to Pass 1.
/// Either way the planes are dropped once harvested.
///
/// A rebuild that fails — a corrupt spill, a dav1d error, a failed
/// self-check, a cover count that disagrees with Pass 1 — is treated as a
/// miss too: logged and re-encoded (a cancel still ends the encode). A
/// systematic rebuild failure therefore costs time, not correctness;
/// `pull_no_shadow_pass2_leaves_source_alone_when_cached` is its canary
/// (it would see the source pulled).
///
/// Returns the entry and whether it had to be re-encoded.
///
/// `pass1_cover` is the cover count Pass 1 measured (the plan was sized
/// from it). A re-encode that disagrees is an error: the cascade slices the
/// harvest by that count, and an out-of-range slice would abort the app
/// (`panic = "abort"` on mobile) instead of failing the encode.
fn pass1_entry_for(
    cache: &mut crate::stego::gop_cache::BoundedGopCache<Pass1Packets>,
    puller: &mut LazyGopPuller<'_>,
    frames_in_gop: u32,
    params: Av1StreamingEncodeParams,
    gop_idx: u32,
    pass1_cover: usize,
) -> Result<(Pass1Entry, bool), Av1StegoError> {
    // Each frame's harvest job is taken the moment the decode hands the
    // frame over, and costed a window of frames at a time; the recording
    // keeps no planes. At most a window of pictures and cost working sets
    // is alive, instead of a whole GOP's (`rebuild::rebuild_gop_streamed`).
    let rebuild = |packets: &[Vec<u8>]| -> Result<(Pass1Payload, GopHarvest), Av1StegoError> {
        let (w, h) = (params.width, params.height);
        let dummy = dummy_planes();
        let window = harvest_window();
        let mut builder = GopHarvestBuilder::with_frames(packets.len());
        let mut pending: Vec<FrameHarvestJob> = Vec::with_capacity(window);
        let frames = super::rebuild::rebuild_gop_streamed(packets, w, h, &mut |_, (_, recording)| {
            // The job keeps the planes; the recording drops its reference.
            pending.push(FrameHarvestJob::new(recording)?);
            recording.reconstructed_planes = dummy.clone();
            if pending.len() == window {
                run_harvest_window(std::mem::take(&mut pending), w, h, &mut builder);
            }
            Ok(())
        })?;
        run_harvest_window(pending, w, h, &mut builder);
        let harvest = builder.finish();
        if harvest.cover_bits.len() != pass1_cover {
            return Err(Av1StegoError::InvalidPacket(format!(
                "rebuild sees {} cover bits, Pass 1 measured {pass1_cover}",
                harvest.cover_bits.len()
            )));
        }
        Ok((frames, harvest))
    };
    let rebuilt = match cache.fetch(gop_idx as usize) {
        Some(Pass1Packets(packets)) => match rebuild(&packets) {
            Ok(entry) => Some(entry),
            Err(Av1StegoError::Stego(crate::stego::error::StegoError::Cancelled)) => {
                return Err(Av1StegoError::Stego(crate::stego::error::StegoError::Cancelled));
            }
            Err(err) => {
                eprintln!("[phasm-av1] GOP {gop_idx}: rebuild failed ({err:?}); re-encoding it");
                None
            }
        },
        None => None,
    };
    let (mut frames, harvest, recomputed) = match rebuilt {
        Some((frames, harvest)) => (frames, harvest, false),
        None => {
            let gop_yuv = puller.gop(gop_idx)?;
            let expected =
                frames_in_gop as usize * expected_i420_size(params.width, params.height);
            if gop_yuv.len() != expected {
                return Err(Av1StegoError::InvalidPacket(format!(
                    "Pass 2: GOP {gop_idx} source returned {} bytes, expected {expected} \
                     ({frames_in_gop} frames)",
                    gop_yuv.len(),
                )));
            }
            let (frames, harvest) = process_one_gop_pass1(&gop_yuv, frames_in_gop, params)?;
            (frames, harvest, true)
        }
    };
    if harvest.cover_bits.len() != pass1_cover {
        return Err(Av1StegoError::InvalidPacket(format!(
            "GOP {gop_idx}: Pass 2 sees {} cover bits, Pass 1 measured {pass1_cover} \
             (non-deterministic re-encode)",
            harvest.cover_bits.len(),
        )));
    }
    drop_planes(&mut frames);
    Ok((Pass1Entry { frames, harvest }, recomputed))
}

/// The natural packets of GOP `gop_idx`, for a GOP Pass 2 emits unmodified
/// (the message is already complete): the cached packets as they are, or —
/// on a miss — a natural re-encode from the source through `puller`,
/// without the harvest. Neither path decodes or costs anything.
///
/// Returns the packets and whether the GOP had to be re-encoded.
fn pass1_packets_for(
    cache: &mut crate::stego::gop_cache::BoundedGopCache<Pass1Packets>,
    puller: &mut LazyGopPuller<'_>,
    frames_in_gop: u32,
    params: Av1StreamingEncodeParams,
    gop_idx: u32,
) -> Result<(Vec<Vec<u8>>, bool), Av1StegoError> {
    if let Some(Pass1Packets(packets)) = cache.fetch(gop_idx as usize) {
        return Ok((packets, false));
    }
    let gop_yuv = puller.gop(gop_idx)?;
    let expected = frames_in_gop as usize * expected_i420_size(params.width, params.height);
    if gop_yuv.len() != expected {
        return Err(Av1StegoError::InvalidPacket(format!(
            "Pass 2: GOP {gop_idx} source returned {} bytes, expected {expected} \
             ({frames_in_gop} frames)",
            gop_yuv.len(),
        )));
    }
    let frames = encode_gop_natural(&gop_yuv, frames_in_gop, params)?;
    Ok((frames.into_iter().map(|(packet, _)| packet).collect(), true))
}

/// Encode one GOP naturally: per frame the natural packet and the
/// encoder's recording (replay storage, cover positions + meta,
/// reconstructed planes, loop-filter state).
///
/// `pub(super)` so `Av1StreamingProbeSession::flush_current_gop` can use it
/// directly (the probe only needs the natural-encode bit tags, not the
/// J-UNIWARD harvest `process_one_gop_pass1` adds).
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
#[derive(Clone)]
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

#[cfg(test)]
impl GopHarvest {
    /// Same cover, bit-identical costs, same back-index.
    pub(super) fn bit_identical(&self, other: &GopHarvest) -> bool {
        self.cover_bits == other.cover_bits
            && self.costs.len() == other.costs.len()
            && self.costs.iter().zip(&other.costs).all(|(a, b)| a.to_bits() == b.to_bits())
            && self.combined_index == other.combined_index
            && self.frame_starts == other.frame_starts
    }

    pub(super) fn cover_len(&self) -> usize {
        self.cover_bits.len()
    }
}

/// A GOP's cover-bit count — what [`harvest_gop`] would return as
/// `cover_bits.len()` — read straight from the encoder's tags, without the
/// J-UNIWARD cost compute. Pass 1 only needs the count (the plan is sized
/// from it); Pass 2 harvests each GOP again anyway, from its rebuild.
fn count_cover_bits(per_frame: &[(Vec<u8>, PhasmFrameRecording<u8>)]) -> usize {
    per_frame
        .iter()
        .filter_map(|(_, recording)| recording.tiles.first())
        .map(|tile| {
            // Same walk as `FrameHarvestJob::new`: the parallel
            // vectors zipped, enrolled tags counted.
            tile.bit_positions
                .iter()
                .zip(&tile.bit_tags)
                .zip(&tile.bit_meta)
                .filter(|((_, tag), _)| {
                    **tag == PHASM_TAG_AC_COEFF_SIGN || **tag == PHASM_TAG_GOLOMB_TAIL_LSB
                })
                .count()
        })
        .sum()
}

/// One frame's harvest, split so the cheap walk runs as the frame arrives
/// and the J-UNIWARD cost compute can run later, a few frames in parallel:
/// the frame's AC_COEFF_SIGN + GOLOMB_TAIL_LSB entries (cursor, natural bit,
/// position) and what the costs need (planes, qindex, loop-filter state).
struct FrameHarvestJob {
    cursors: Vec<u64>,
    bits: Vec<u8>,
    positions: Vec<Av1FramePosition>,
    planes: Arc<phasm_rav1e::Frame<u8>>,
    qindex: u8,
    loop_filter: PhasmFrameLoopFilterState,
}

impl FrameHarvestJob {
    /// Walk `recording`'s 50/50 positions (a `planes` reference is taken,
    /// so the recording's own may be dropped right after).
    fn new(recording: &PhasmFrameRecording<u8>) -> Result<Self, Av1StegoError> {
        let tile = recording.tiles.first().ok_or(Av1StegoError::EmptyRecording)?;
        let mut job = FrameHarvestJob {
            cursors: Vec::new(),
            bits: Vec::new(),
            positions: Vec::new(),
            planes: Arc::clone(&recording.reconstructed_planes),
            qindex: recording.frame_qindex,
            loop_filter: recording.loop_filter_state,
        };
        for (cursor, ((&(_, value), &tag), m)) in tile
            .bit_positions
            .iter()
            .zip(tile.bit_tags.iter())
            .zip(tile.bit_meta.iter())
            .enumerate()
        {
            if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
                job.cursors.push(cursor as u64);
                job.bits.push(value as u8);
                job.positions.push(Av1FramePosition {
                    plane: m.plane,
                    plane_px_x: m.plane_px_x,
                    plane_px_y: m.plane_px_y,
                    tx_width_log2: m.tx_width_log2,
                    tx_height_log2: m.tx_height_log2,
                    tx_type: m.tx_type,
                    scan_pos: m.scan_pos,
                    coeff_magnitude: m.coeff_magnitude,
                });
            }
        }
        Ok(job)
    }

    /// The frame's `(cursors, cover bits, costs)`; the planes are released.
    fn run(self, width: u32, height: u32) -> (Vec<u64>, Vec<u8>, Vec<f32>) {
        let frame_planes = pack_cost_planes(&self.planes, width as usize, height as usize);
        drop(self.planes);
        let costs = compute_av1_uniward_costs_with_state(
            &frame_planes,
            &self.positions,
            self.qindex,
            Some(self.loop_filter),
        );
        (self.cursors, self.bits, costs)
    }
}

/// How many frames' J-UNIWARD costs are computed at once. The cost compute
/// is parallel within a frame only in its final per-position pass, so
/// costing frames one by one is ~4–5× slower; costing a whole GOP at once
/// held every frame's wavelet working set together (1.2–1.4 GB at 1080p,
/// ~2.5 GB at 2720×1530). A small window keeps most of the speed at a
/// bounded peak. `PHASM_AV1_HARVEST_WINDOW` overrides it (dev / measurement).
fn harvest_window() -> usize {
    std::env::var("PHASM_AV1_HARVEST_WINDOW")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(HARVEST_WINDOW)
}

const HARVEST_WINDOW: usize = 4;

/// Run `jobs` (one harvest window) and push their outputs in frame order.
fn run_harvest_window(
    jobs: Vec<FrameHarvestJob>,
    width: u32,
    height: u32,
    builder: &mut GopHarvestBuilder,
) {
    #[cfg(feature = "parallel")]
    let outputs: Vec<_> = jobs.into_par_iter().map(|job| job.run(width, height)).collect();
    #[cfg(not(feature = "parallel"))]
    let outputs: Vec<_> = jobs.into_iter().map(|job| job.run(width, height)).collect();
    for output in outputs {
        builder.push(output);
    }
}

/// `width`×`height` is the visible frame size: the cost input repeats its
/// edge into rav1e's alignment padding ([`pack_cost_planes`]), so a harvest
/// of the encoder's recording and of a decode of the same packets agree.
pub(super) fn harvest_gop(
    per_frame: &[(Vec<u8>, PhasmFrameRecording<u8>)],
    width: u32,
    height: u32,
) -> Result<GopHarvest, Av1StegoError> {
    // Frames are independent; their J-UNIWARD costs are computed a window
    // at a time ([`harvest_window`]), in parallel within the window, and
    // accumulated in frame order to keep the cover cursor monotonic.
    let mut builder = GopHarvestBuilder::with_frames(per_frame.len());
    for window in per_frame.chunks(harvest_window()) {
        let jobs = window
            .iter()
            .map(|(_, recording)| FrameHarvestJob::new(recording))
            .collect::<Result<Vec<_>, _>>()?;
        run_harvest_window(jobs, width, height, &mut builder);
    }
    Ok(builder.finish())
}

/// Assembles a [`GopHarvest`] from per-frame [`FrameHarvestJob::run`]
/// outputs pushed in frame order — the one place the GOP-wide cover cursor
/// and back-index are built, for both the whole-GOP [`harvest_gop`] and
/// Pass 2's frame-by-frame harvest of a streamed rebuild.
struct GopHarvestBuilder {
    harvest: GopHarvest,
}

impl GopHarvestBuilder {
    fn with_frames(n_frames: usize) -> Self {
        Self {
            harvest: GopHarvest {
                cover_bits: Vec::new(),
                costs: Vec::new(),
                combined_index: Vec::new(),
                frame_starts: Vec::with_capacity(n_frames),
            },
        }
    }

    /// The next frame's `(cursors, cover bits, costs)`.
    fn push(&mut self, (frame_cursors, frame_bits, frame_costs): (Vec<u64>, Vec<u8>, Vec<f32>)) {
        let h = &mut self.harvest;
        let frame_idx = h.frame_starts.len();
        h.frame_starts.push(h.cover_bits.len());
        h.combined_index.extend(frame_cursors.iter().map(|&c| (frame_idx, c)));
        h.cover_bits.extend_from_slice(&frame_bits);
        h.costs.extend_from_slice(&frame_costs);
    }

    fn finish(self) -> GopHarvest {
        self.harvest
    }
}

// `verify_shadows_round_trip` (whole-clip dav1d walk + av1_shadow_extract
// per shadow) deleted at WV.7.6 — replaced by Av1StreamingShadowVerify
// in shadow.rs, which walks emitted GOP bytes per-GOP and feeds them
// into per-shadow Av1ShadowBitTopN heaps. Drops the whole-clip
// cover_bits Vec (~4 GB at 30 min) AND the whole-clip priority_slots
// Vec inside av1_shadow_extract (~50 GB at 30 min) from verify-time
// peak.

/// An `n`-frame `w`×`h` GOP over xorshift-noise luma, and its params.
#[cfg(test)]
pub(super) fn noise_gop_sized(
    w: u32,
    h: u32,
    n: u32,
) -> (Vec<u8>, Av1StreamingEncodeParams) {
    let frame = expected_i420_size(w, h);
    let luma = (w * h) as usize;
    let mut s: u32 = 0x1234_5678;
    let mut yuv = vec![128u8; frame * n as usize];
    for f in 0..n as usize {
        for px in &mut yuv[f * frame..f * frame + luma] {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *px = (s >> 24) as u8;
        }
    }
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 60,
        gop_size: n,
        total_frames_hint: n,
    };
    (yuv, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::av1::stego::orchestrator::pack_visible_planes;
    use crate::stego::gop_cache::GopCachePayload;

    /// Two-frame 64×64 GOP over xorshift-noise luma (flat grey can yield an
    /// empty cover) and its encode params.
    fn noise_gop() -> (Vec<u8>, Av1StreamingEncodeParams) {
        noise_gop_sized(64, 64, 2)
    }

    /// Pass 2's entry for a GOP — rebuilt from cached packets (hit) or
    /// re-encoded from the source (miss) — must carry the cover count Pass 1
    /// measured; one that does not is an error, never an out-of-range slice
    /// (= abort on mobile).
    #[test]
    fn pass2_entry_must_match_pass1_cover() {
        let (yuv, params) = noise_gop();
        let (frames, harvest) = process_one_gop_pass1(&yuv, 2, params).unwrap();
        let pass1_cover = harvest.cover_bits.len();
        let packets = Pass1Packets(frames.into_iter().map(|(p, _)| p).collect());
        let scratch = std::env::temp_dir();
        for cached in [true, false] {
            let ram = if cached { 64 << 20 } else { 0 };
            let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
                1, ram, 0, scratch.clone(), AV1_PASS1_CACHE_TAG,
            );
            cache.insert(0, packets.clone());
            for (cover, ok) in [(pass1_cover, true), (pass1_cover + 1, false)] {
                if cached {
                    cache.insert(0, packets.clone());
                }
                let mut src = Av1SliceYuvSource::new(&yuv, 64, 64, 2, 2);
                let mut puller = LazyGopPuller::new(&mut src);
                let got = pass1_entry_for(&mut cache, &mut puller, 2, params, 0, cover);
                match got {
                    Ok((_, recomputed)) => {
                        assert!(ok, "cached={cached}: cover {cover} accepted");
                        assert_eq!(recomputed, !cached, "cached={cached}");
                    }
                    Err(Av1StegoError::InvalidPacket(_)) => assert!(!ok, "cached={cached}"),
                    Err(e) => panic!("cached={cached}: {e:?}"),
                }
            }
        }
    }

    /// A cached GOP whose packets do not rebuild (a corrupt spill, a dav1d
    /// error, a failed self-check) is treated as a cache miss: re-encoded
    /// from the source, pulled in contract order, with Pass 1's cover.
    #[test]
    fn pass2_rebuild_failure_falls_back_to_reencode() {
        struct LoggingSource<'a> {
            inner: Av1SliceYuvSource<'a>,
            log: Vec<Option<u32>>,
        }
        impl Av1GopYuvSource for LoggingSource<'_> {
            fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError> {
                self.log.push(Some(gop_index));
                self.inner.gop_yuv(gop_index)
            }
            fn rewind(&mut self) -> Result<(), Av1StegoError> {
                self.log.push(None);
                self.inner.rewind()
            }
        }

        let (yuv, params) = noise_gop();
        let (frames, harvest) = process_one_gop_pass1(&yuv, 2, params).unwrap();
        let pass1_cover = harvest.cover_bits.len();
        // Each packet loses its last 16 bytes: the rebuild cannot reproduce
        // the tile bytes (or dav1d rejects it) — never a valid entry.
        let broken = Pass1Packets(
            frames.into_iter().map(|(p, _)| p[..p.len() - 16].to_vec()).collect(),
        );
        let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
            1,
            64 << 20,
            0,
            std::env::temp_dir(),
            AV1_PASS1_CACHE_TAG,
        );
        cache.insert(0, broken);
        let mut src = LoggingSource { inner: Av1SliceYuvSource::new(&yuv, 64, 64, 2, 2), log: Vec::new() };
        let mut puller = LazyGopPuller::new(&mut src);
        let (entry, recomputed) = pass1_entry_for(&mut cache, &mut puller, 2, params, 0, pass1_cover)
            .expect("a failed rebuild falls back to re-encoding");
        drop(puller);
        assert!(recomputed, "the GOP was re-encoded");
        assert_eq!(entry.harvest.cover_bits.len(), pass1_cover);
        assert_eq!(src.log, vec![None, Some(0)], "re-encoded from the source, in contract order");
    }

    /// Pass 1 counts a GOP's cover bits straight from the encoder's tags
    /// instead of running the J-UNIWARD harvest it would throw away; the
    /// count must be exactly the harvest's (the plan is sized from it and
    /// Pass 2 checks against it).
    #[test]
    fn pass1_cover_count_matches_harvest() {
        for (w, h) in [(64u32, 64u32), (250, 142)] {
            let (yuv, params) = noise_gop_sized(w, h, 2);
            let (frames, harvest) = process_one_gop_pass1(&yuv, 2, params).unwrap();
            assert!(!harvest.cover_bits.is_empty(), "{w}x{h}: fixture must have cover");
            assert_eq!(count_cover_bits(&frames), harvest.cover_bits.len(), "{w}x{h}");
        }
    }

    /// A GOP emitted unmodified after the message is complete needs only its
    /// natural packets: a cache hit hands them over as stored — no decode
    /// (these bytes are not AV1, so a decode would fail) — and the source
    /// stays untouched.
    #[test]
    fn pass1_packets_for_uses_cached_packets_without_decoding() {
        let (_, params) = noise_gop();
        let stored = Pass1Packets(vec![vec![1, 2, 3], vec![4]]);
        let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
            1, 64 << 20, 0, std::env::temp_dir(), AV1_PASS1_CACHE_TAG,
        );
        cache.insert(0, stored.clone());
        let mut src = MockSource::default();
        let mut puller = LazyGopPuller::new(&mut src);
        let (packets, recomputed) = pass1_packets_for(&mut cache, &mut puller, 2, params, 0).unwrap();
        drop(puller);
        assert_eq!(packets, stored.0);
        assert!(!recomputed);
        assert!(src.log.is_empty(), "a cache hit must not read the source");
    }

    /// On a miss the packets are natural-encoded again from the source —
    /// the same bytes Pass 1 produced, without the harvest.
    #[test]
    fn pass1_packets_for_reencodes_a_miss() {
        let (yuv, params) = noise_gop();
        let pass1: Vec<Vec<u8>> =
            process_one_gop_pass1(&yuv, 2, params).unwrap().0.into_iter().map(|(p, _)| p).collect();
        let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
            1, 0, 0, std::env::temp_dir(), AV1_PASS1_CACHE_TAG,
        );
        let mut src = Av1SliceYuvSource::new(&yuv, 64, 64, 2, 2);
        let mut puller = LazyGopPuller::new(&mut src);
        let (packets, recomputed) = pass1_packets_for(&mut cache, &mut puller, 2, params, 0).unwrap();
        assert_eq!(packets, pass1);
        assert!(recomputed);
    }

    #[test]
    fn pass1_packets_flat_roundtrip() {
        let packets = Pass1Packets(vec![Vec::new(), vec![7], (0..1000u32).map(|i| i as u8).collect()]);
        let blob = packets.flat_pack();
        assert_eq!(Pass1Packets::flat_unpack(&blob).expect("unpack").0, packets.0);
        assert!(Pass1Packets::flat_unpack(&blob[..blob.len() - 1]).is_err(), "truncated");
        let mut v2 = blob.clone();
        v2[0] = 2;
        assert!(Pass1Packets::flat_unpack(&v2).is_err(), "old format byte");
        assert_eq!(packets.cached_size(), 1001 + 24 * 3 + 24);
    }

    /// Logs `None` for a rewind and `Some(g)` per GOP; GOP `g`'s YUV is
    /// four bytes of `g`.
    #[derive(Default)]
    struct MockSource {
        log: Vec<Option<u32>>,
    }

    impl Av1GopYuvSource for MockSource {
        fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError> {
            self.log.push(Some(gop_index));
            Ok(vec![gop_index as u8; 4])
        }
        fn rewind(&mut self) -> Result<(), Av1StegoError> {
            self.log.push(None);
            Ok(())
        }
    }

    #[test]
    fn lazy_puller_untouched_until_first_call() {
        let mut src = MockSource::default();
        let puller = LazyGopPuller::new(&mut src);
        drop(puller);
        assert!(src.log.is_empty());
    }

    /// The source contract holds on a miss after hits: one rewind, then
    /// every GOP in order up to the one needed.
    #[test]
    fn lazy_puller_pulls_forward_in_order() {
        let mut src = MockSource::default();
        let mut puller = LazyGopPuller::new(&mut src);
        assert_eq!(puller.gop(2).unwrap(), vec![2; 4]);
        assert_eq!(puller.gop(3).unwrap(), vec![3; 4]);
        assert!(matches!(puller.gop(1), Err(Av1StegoError::InvalidPacket(_))));
        drop(puller);
        assert_eq!(src.log, vec![None, Some(0), Some(1), Some(2), Some(3)]);
    }

    /// Overwrite every pixel of `rec`'s planes past the visible `w`×`h`
    /// (luma; chroma at half), up to the aligned plane size, with `v`.
    fn fill_alignment_padding(rec: &mut PhasmFrameRecording<u8>, w: usize, h: usize, v: u8) {
        let mut f = (*rec.reconstructed_planes).clone();
        for (i, p) in f.planes.iter_mut().enumerate() {
            let (vw, vh) = if i == 0 { (w, h) } else { (w.div_ceil(2), h.div_ceil(2)) };
            let (pw, ph) = (p.cfg.width, p.cfg.height);
            let (stride, xo, yo) = (p.cfg.stride, p.cfg.xorigin, p.cfg.yorigin);
            for y in 0..ph {
                for x in 0..pw {
                    if x >= vw || y >= vh {
                        p.data[(yo + y) * stride + xo + x] = v;
                    }
                }
            }
        }
        rec.reconstructed_planes = Arc::new(f);
    }

    /// J-UNIWARD costs depend only on pixels a viewer sees: rav1e aligns
    /// plane width and height to 8, and what it reconstructs in that
    /// padding is cropped on output and not reproduced by a decoder.
    #[test]
    fn harvest_ignores_alignment_padding() {
        let (w, h) = (250u32, 142u32);
        let (yuv, params) = noise_gop_sized(w, h, 2);
        let mut frames = encode_gop_natural(&yuv, 2, params).expect("encode");
        assert_eq!(frames[0].1.reconstructed_planes.planes[0].cfg.width, 256, "rav1e aligns to 8");
        let before = harvest_gop(&frames, w, h).expect("harvest");
        for (_, rec) in &mut frames {
            fill_alignment_padding(rec, w as usize, h as usize, 255);
        }
        let after = harvest_gop(&frames, w, h).expect("harvest");
        assert_eq!(after.cover_bits, before.cover_bits);
        let changed = before
            .costs
            .iter()
            .zip(&after.costs)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(changed, 0, "{changed} of {} costs read the padding", before.costs.len());
    }

    /// The cost input repeats the last visible column and row into rav1e's
    /// alignment padding, and leaves the visible region alone.
    #[test]
    fn pack_cost_planes_replicates_visible_edge() {
        let (w, h) = (250usize, 142usize);
        let mut f = make_frame::<u8>(w, h, ChromaSampling::Cs420);
        for (i, p) in f.planes.iter_mut().enumerate() {
            let (stride, xo, yo) = (p.cfg.stride, p.cfg.xorigin, p.cfg.yorigin);
            for y in 0..p.cfg.height {
                for x in 0..p.cfg.width {
                    p.data[(yo + y) * stride + xo + x] = ((x * 7 + y * 13 + i * 29) % 251) as u8;
                }
            }
        }
        let rec = Arc::new(f);
        let raw = pack_visible_planes(&rec);
        let got = pack_cost_planes(&rec, w, h);
        assert_eq!((got.luma_width, got.luma_height), (256, 144));
        assert_eq!((got.chroma_width, got.chroma_height), (128, 72));
        let check = |name: &str, got: &[u8], raw: &[u8], pw: usize, ph: usize, vw: usize, vh: usize| {
            for y in 0..ph {
                for x in 0..pw {
                    let want = raw[y.min(vh - 1) * pw + x.min(vw - 1)];
                    assert_eq!(got[y * pw + x], want, "{name} ({x},{y})");
                }
            }
        };
        check("Y", &got.y, &raw.y, 256, 144, w, h);
        check("U", &got.cb, &raw.cb, 128, 72, 125, 71);
        check("V", &got.cr, &raw.cr, 128, 72, 125, 71);
    }

    /// The streamed rebuild hands each frame over in order, as soon as it
    /// is decoded, with its full planes; Pass 2's hit path harvests it there
    /// and drops the planes, and must end up with exactly the entry a whole-
    /// GOP rebuild + `harvest_gop` gives: same packets, same storage, a
    /// bit-identical harvest — and only the shared dummy planes kept.
    #[test]
    fn streamed_rebuild_hit_path_matches_rebuild_then_harvest() {
        for (w, h) in [(256u32, 144u32), (250, 142)] {
            let (yuv, params) = noise_gop_sized(w, h, 4);
            let frames = encode_gop_natural(&yuv, 4, params).unwrap();
            let cover = count_cover_bits(&frames);
            let packets: Vec<Vec<u8>> = frames.iter().map(|(p, _)| p.clone()).collect();
            let rebuilt = super::super::rebuild::rebuild_gop_recordings(&packets, w, h).unwrap();
            let reference = harvest_gop(&rebuilt, w, h).unwrap();

            let mut seen = Vec::new();
            let streamed = super::super::rebuild::rebuild_gop_streamed(&packets, w, h, &mut |k, entry| {
                assert!(entry.1.reconstructed_planes.planes[0].cfg.width >= w as usize, "full planes");
                seen.push(k);
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, vec![0, 1, 2, 3], "{w}x{h}: frames handed over in order");
            assert_eq!(streamed.len(), 4);

            let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
                1, 64 << 20, 0, std::env::temp_dir(), AV1_PASS1_CACHE_TAG,
            );
            cache.insert(0, Pass1Packets(packets.clone()));
            let mut src = MockSource::default();
            let (entry, recomputed) = {
                let mut puller = LazyGopPuller::new(&mut src);
                pass1_entry_for(&mut cache, &mut puller, 4, params, 0, cover).unwrap()
            };
            assert!(!recomputed && src.log.is_empty(), "{w}x{h}: hit path fell back");
            assert!(entry.harvest.bit_identical(&reference), "{w}x{h}: harvest differs");
            let dummy_width = dummy_planes().planes[0].cfg.width;
            for ((p, r), (rp, rr)) in entry.frames.iter().zip(&rebuilt) {
                assert_eq!(p, rp);
                assert_eq!(r.tiles[0].storage, rr.tiles[0].storage);
                assert_eq!(r.tiles[0].bit_positions, rr.tiles[0].bit_positions);
                assert_eq!(r.reconstructed_planes.planes[0].cfg.width, dummy_width, "planes dropped");
            }
        }
    }

    // ── Pass-2 peak-memory measurement (two ignored tests, separate
    // processes, so rav1e's encode peak does not mask the rebuild's).
    //
    //   PHASM_TMP=… cargo test … --lib pass2_rss_write_packets -- --ignored --nocapture
    //   PHASM_TMP=… cargo test … --lib pass2_rss_rebuild -- --ignored --nocapture
    //
    // Env: PHASM_O3_CLIP (default IMG_4138), PHASM_O3_W / _H (1920×1080),
    // PHASM_O3_FRAMES (30), PHASM_O3_Q (100).

    fn rss_env_u32(k: &str, d: u32) -> u32 {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }

    /// `$PHASM_TMP/phasm-pass2-rss/packets_<W>x<H>.bin` for the env size.
    fn rss_packets_path() -> std::path::PathBuf {
        let base = std::env::var("PHASM_TMP")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let (w, h) = (rss_env_u32("PHASM_O3_W", 1920), rss_env_u32("PHASM_O3_H", 1080));
        base.join("phasm-pass2-rss").join(format!("packets_{w}x{h}.bin"))
    }

    fn peak_rss_bytes() -> u64 {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
            return 0;
        }
        // macOS reports bytes, Linux KiB.
        if cfg!(target_os = "macos") { ru.ru_maxrss as u64 } else { ru.ru_maxrss as u64 * 1024 }
    }

    /// Encode one GOP and write `[w][h][frames][pass1_cover][Pass1Packets]`.
    #[test]
    #[ignore = "measurement: writes packets for pass2_rss_rebuild"]
    fn pass2_rss_write_packets() {
        let clip = std::env::var("PHASM_O3_CLIP").unwrap_or_else(|_| {
            format!("{}/test-vectors/video/h264/real-world/source/IMG_4138.MOV", env!("CARGO_MANIFEST_DIR"))
        });
        let (w, h) = (rss_env_u32("PHASM_O3_W", 1920), rss_env_u32("PHASM_O3_H", 1080));
        let n = rss_env_u32("PHASM_O3_FRAMES", 30);
        let q = rss_env_u32("PHASM_O3_Q", 100) as usize;
        let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
        let out = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i", &clip, "-frames:v", &n.to_string(), "-vf", &vf])
            .args(["-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
            .output()
            .expect("ffmpeg");
        assert!(out.status.success(), "ffmpeg failed on {clip}");
        let params = Av1StreamingEncodeParams { width: w, height: h, quantizer: q, gop_size: n, total_frames_hint: n };
        let frames = encode_gop_natural(&out.stdout, n, params).expect("encode");
        let cover = count_cover_bits(&frames);
        let packets = Pass1Packets(frames.into_iter().map(|(p, _)| p).collect());
        let mut blob = Vec::new();
        for v in [w, h, n] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&(q as u32).to_le_bytes());
        blob.extend_from_slice(&(cover as u64).to_le_bytes());
        blob.extend_from_slice(&packets.flat_pack());
        let path = rss_packets_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &blob).unwrap();
        eprintln!("[pass2_rss] wrote {} ({w}x{h} x{n} q{q}, cover {cover}, {} B)", path.display(), blob.len());
    }

    /// Run Pass 2's cache-hit path on the written GOP and report peak RSS.
    #[test]
    #[ignore = "measurement: run after pass2_rss_write_packets, in its own process"]
    fn pass2_rss_rebuild() {
        let blob = std::fs::read(rss_packets_path()).expect("run pass2_rss_write_packets first");
        let u32_at = |i: usize| u32::from_le_bytes(blob[i..i + 4].try_into().unwrap());
        let (w, h, n, q) = (u32_at(0), u32_at(4), u32_at(8), u32_at(12) as usize);
        let cover = u64::from_le_bytes(blob[16..24].try_into().unwrap()) as usize;
        let packets = Pass1Packets::flat_unpack(&blob[24..]).expect("packets");
        let params = Av1StreamingEncodeParams { width: w, height: h, quantizer: q, gop_size: n, total_frames_hint: n };
        let mut cache = crate::stego::gop_cache::BoundedGopCache::<Pass1Packets>::new(
            1, 1 << 30, 0, std::env::temp_dir(), AV1_PASS1_CACHE_TAG,
        );
        cache.insert(0, packets);
        let rss_before = peak_rss_bytes();
        if std::env::var("PHASM_RSS_REBUILD_ONLY").is_ok() {
            // Attribution: the decode-side rebuild alone, no harvest.
            let packets = cache.fetch(0).unwrap().0;
            let t = std::time::Instant::now();
            let frames = super::super::rebuild::rebuild_gop_recordings(&packets, w, h).unwrap();
            eprintln!(
                "[pass2_rss] {w}x{h} x{n}: rebuild only {:.3}s, {} frames, peak RSS {:.1} MB",
                t.elapsed().as_secs_f64(),
                frames.len(),
                peak_rss_bytes() as f64 / (1 << 20) as f64
            );
            return;
        }
        // A pull would mean the rebuild failed and fell back to a re-encode.
        let mut src = MockSource::default();
        let t = std::time::Instant::now();
        let (entry, recomputed) = {
            let mut puller = LazyGopPuller::new(&mut src);
            pass1_entry_for(&mut cache, &mut puller, n, params, 0, cover).expect("hit path")
        };
        let dt = t.elapsed();
        assert!(!recomputed && src.log.is_empty(), "the rebuild fell back to a re-encode");
        assert_eq!(entry.harvest.cover_bits.len(), cover);
        let rss = peak_rss_bytes();
        eprintln!(
            "[pass2_rss] {w}x{h} x{n}: hit path {:.3}s, peak RSS {:.1} MB (before the call {:.1} MB)",
            dt.as_secs_f64(),
            rss as f64 / (1 << 20) as f64,
            rss_before as f64 / (1 << 20) as f64
        );
    }
}
