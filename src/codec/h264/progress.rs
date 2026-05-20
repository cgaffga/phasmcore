// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #474 — video stego progress reporting.
//!
//! Engine-agnostic progress event vocabulary emitted by the streaming
//! encode + decode sessions. Mobile bridges marshal these events
//! across FFI into per-platform smoothing engines that drive the HUD
//! progress bar + ETA text.
//!
//! Design doc: `docs/design/video/h264/progress-indicator.md`.
//!
//! ## Why an event enum (not raw fractions)
//!
//! A single 0..1 fraction can't express *which* phase of the pipeline
//! is running, and different phases have very different per-unit wall-
//! clock cost (Pass 1 frame ≠ STC GOP ≠ mux byte). Encoding the phase
//! into the event lets the mobile-side smoothing engine apply per-
//! phase weights (defaults + adaptive refinement) and compute a
//! meaningful ETA. See design doc §3, §4.
//!
//! ## Throttling contract
//!
//! Emit-site authors MUST throttle to ≤30 events/sec per session.
//! At 1080p × 60 fps the UI doesn't benefit from finer resolution
//! and the FFI cost (atomic + thread marshalling on iOS/Android)
//! adds up. The streaming session helper takes care of this; ad-hoc
//! callers should check `last_emit_time` themselves.

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Encode-side progress phases. Fires in this order during a single
/// streaming encode session:
///
/// `Setup → Pass1Capture×N → StcPlan×G → Pass2Replay×N → Mux → Done`
///
/// where `N = total_frames` and `G = total_gops`. The per-unit phases
/// (Pass1Capture, StcPlan, Pass2Replay) emit one event per unit;
/// boundary phases (Setup, Mux, Done) emit once each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodePhase {
    /// Session created; encoder + STC tables initialised. One event.
    Setup,
    /// Capture pass — encoder runs with no overrides, walker records
    /// per-MB cover positions. One event per frame.
    Pass1Capture {
        frame_idx: u32,
        total_frames: u32,
    },
    /// STC Viterbi plan computed per-GOP over the captured cover.
    /// One event per GOP (so progress moves smoothly across GOP
    /// boundaries even for long clips).
    StcPlan {
        gop_idx: u32,
        total_gops: u32,
    },
    /// Replay pass — encoder runs with override map active, emits
    /// the final stego wire. One event per frame.
    Pass2Replay {
        frame_idx: u32,
        total_frames: u32,
    },
    /// HandBrake-style mp4 muxing + final file write. One event.
    Mux,
    /// Encode finished; bytes are in the caller's buffer. One event.
    Done,
}

/// Decode-side progress phases. Fires in this order:
///
/// `Demux → Walker×N → StcExtract → ShadowExtract×T → Decrypt → Done`
///
/// where `N = total_frames`, `T = parity tier count` (currently 6,
/// see `core/src/codec/h264/stego/shadow.rs`). `ShadowExtract`
/// dominates the wall clock on wrong-passphrase decode today; #532
/// will collapse it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePhase {
    /// MP4 demux + Annex-B accumulation. One event at start.
    Demux,
    /// Per-frame CABAC walker pass that extracts cover bits. One
    /// event per frame.
    Walker {
        frame_idx: u32,
        total_frames: u32,
    },
    /// Primary STC extract + RS / Reed-Solomon parity check. One
    /// event at start.
    StcExtract,
    /// Shadow extract — iterate the 6 parity tiers brute-forcing
    /// frame-data-length candidates. One event per tier. After #532
    /// ships this collapses to a CRC oracle and the events become
    /// near-instantaneous; the phase weight in the smoothing engine
    /// must be re-calibrated then.
    ShadowExtract {
        tier_idx: u8,
        total_tiers: u8,
    },
    /// AES-GCM-SIV decrypt + brotli decompress + UTF-8 decode. One
    /// event at start.
    Decrypt,
    /// Decode finished; plaintext is ready. One event.
    Done,
}

/// Boxed-closure callback for encode progress. `Arc<dyn Fn>` so the
/// callback can be cheaply cloned into background threads (e.g. the
/// streaming session may move work to a worker pool in v1.2+) and
/// `Send + Sync` so emit sites in any thread context can invoke it.
pub type EncodeProgressCallback = Arc<dyn Fn(EncodePhase) + Send + Sync>;

/// Boxed-closure callback for decode progress. See
/// [`EncodeProgressCallback`] for the design rationale.
pub type DecodeProgressCallback = Arc<dyn Fn(DecodePhase) + Send + Sync>;

/// #475 — capacity-probe progressive estimate. Fires once per GOP as
/// the probe accumulates cover-bit data, so the UI can show an
/// estimated capacity number (with refining "~123…" presentation)
/// long before the full probe finishes.
///
/// Arguments: (gops_done, gops_total, cover_bits_so_far). Callers
/// compute the estimated total cover_bits as
/// `cover_bits_so_far * gops_total / gops_done`. `gops_total` is the
/// caller's `total_frames_hint / gop_size` — the probe doesn't know
/// the true total until the very last partial GOP drains, so this
/// number is the best-known target throughout the probe.
pub type CapacityProbeCallback =
    Arc<dyn Fn(u32, u32, usize) + Send + Sync>;

/// 30 Hz throttle interval — emit sites should drop per-unit events
/// that arrive less than this after the previous emission in the
/// same phase. Boundary events (`Setup`, `Mux`, `Done`, …) ignore
/// the throttle and always fire.
pub const PROGRESS_MIN_INTERVAL: Duration = Duration::from_millis(33);

/// Tiny helper for emit sites: tracks the last emission time per
/// phase and decides whether the current event should be throttled.
/// Boundary phases (those with no per-unit dimension) bypass the
/// throttle.
///
/// Construct once per session, reset on phase change.
#[derive(Debug, Default)]
pub struct ProgressThrottle {
    last_emit: Option<Instant>,
}

impl ProgressThrottle {
    pub fn new() -> Self {
        Self { last_emit: None }
    }

    /// Returns `true` if the caller should emit the event now,
    /// `false` if the event should be dropped. Always returns `true`
    /// for boundary events (`is_boundary == true`).
    pub fn should_emit(&mut self, is_boundary: bool) -> bool {
        if is_boundary {
            self.last_emit = Some(Instant::now());
            return true;
        }
        let now = Instant::now();
        let allow = self
            .last_emit
            .map_or(true, |t| now.duration_since(t) >= PROGRESS_MIN_INTERVAL);
        if allow {
            self.last_emit = Some(now);
        }
        allow
    }

    /// Reset on phase transition so the first event of the new phase
    /// fires immediately, regardless of when the previous phase last
    /// emitted.
    pub fn reset(&mut self) {
        self.last_emit = None;
    }
}

/// Convenience: build a no-op callback. Used by call sites that need
/// a `&EncodeProgressCallback` to pass through but don't actually
/// want to receive events (e.g. internal benchmarks).
pub fn noop_encode_callback() -> EncodeProgressCallback {
    Arc::new(|_phase: EncodePhase| {})
}

/// See [`noop_encode_callback`].
pub fn noop_decode_callback() -> DecodeProgressCallback {
    Arc::new(|_phase: DecodePhase| {})
}

// ─────────────────────── FFI marshalling ──────────────────────────────
//
// #474.3 / #474.4 — stable phase-code → integer mapping shared by the
// iOS C FFI and the Android JNI. The bridges expose a C callback with
// signature `(phase_code: u32, a: u32, b: u32, ctx: *mut c_void)`; the
// (a, b) slots carry frame_idx/total_frames or analogous per-unit
// counters. Boundary phases pass (0, 0). The Rust↔C trampoline lives
// in the bridges themselves so we can keep `phasm-core` free of FFI
// types; the *constants* live here so bridges and the mobile-side
// smoothing engine can't drift on the wire format.

/// Encode phase codes. Stable across FFI; do not renumber.
pub mod encode_phase_codes {
    pub const SETUP: u32 = 0;
    pub const PASS1_CAPTURE: u32 = 1;
    pub const STC_PLAN: u32 = 2;
    pub const PASS2_REPLAY: u32 = 3;
    pub const MUX: u32 = 4;
    pub const DONE: u32 = 5;
}

/// Decode phase codes. Stable across FFI; do not renumber.
pub mod decode_phase_codes {
    pub const DEMUX: u32 = 0;
    pub const WALKER: u32 = 1;
    pub const STC_EXTRACT: u32 = 2;
    pub const SHADOW_EXTRACT: u32 = 3;
    pub const DECRYPT: u32 = 4;
    pub const DONE: u32 = 5;
}

impl EncodePhase {
    /// Project a phase into the `(code, a, b)` FFI tuple. Boundary
    /// phases use `(code, 0, 0)`.
    pub fn to_ffi(self) -> (u32, u32, u32) {
        match self {
            Self::Setup => (encode_phase_codes::SETUP, 0, 0),
            Self::Pass1Capture { frame_idx, total_frames } => {
                (encode_phase_codes::PASS1_CAPTURE, frame_idx, total_frames)
            }
            Self::StcPlan { gop_idx, total_gops } => {
                (encode_phase_codes::STC_PLAN, gop_idx, total_gops)
            }
            Self::Pass2Replay { frame_idx, total_frames } => {
                (encode_phase_codes::PASS2_REPLAY, frame_idx, total_frames)
            }
            Self::Mux => (encode_phase_codes::MUX, 0, 0),
            Self::Done => (encode_phase_codes::DONE, 0, 0),
        }
    }
}

impl DecodePhase {
    /// Project a phase into the `(code, a, b)` FFI tuple. Boundary
    /// phases use `(code, 0, 0)`. ShadowExtract widens its `u8` tier
    /// counters to `u32` for a uniform wire format.
    pub fn to_ffi(self) -> (u32, u32, u32) {
        match self {
            Self::Demux => (decode_phase_codes::DEMUX, 0, 0),
            Self::Walker { frame_idx, total_frames } => {
                (decode_phase_codes::WALKER, frame_idx, total_frames)
            }
            Self::StcExtract => (decode_phase_codes::STC_EXTRACT, 0, 0),
            Self::ShadowExtract { tier_idx, total_tiers } => {
                (decode_phase_codes::SHADOW_EXTRACT, tier_idx as u32, total_tiers as u32)
            }
            Self::Decrypt => (decode_phase_codes::DECRYPT, 0, 0),
            Self::Done => (decode_phase_codes::DONE, 0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn encode_callback_captures_events() {
        let recorded: Arc<Mutex<Vec<EncodePhase>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_clone = Arc::clone(&recorded);
        let cb: EncodeProgressCallback = Arc::new(move |phase| {
            recorded_clone.lock().unwrap().push(phase);
        });

        cb(EncodePhase::Setup);
        cb(EncodePhase::Pass1Capture {
            frame_idx: 0,
            total_frames: 5,
        });
        cb(EncodePhase::Done);

        let got = recorded.lock().unwrap().clone();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], EncodePhase::Setup);
        assert_eq!(
            got[1],
            EncodePhase::Pass1Capture {
                frame_idx: 0,
                total_frames: 5
            }
        );
        assert_eq!(got[2], EncodePhase::Done);
    }

    #[test]
    fn decode_callback_captures_events() {
        let recorded: Arc<Mutex<Vec<DecodePhase>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_clone = Arc::clone(&recorded);
        let cb: DecodeProgressCallback = Arc::new(move |phase| {
            recorded_clone.lock().unwrap().push(phase);
        });

        cb(DecodePhase::Demux);
        cb(DecodePhase::Walker {
            frame_idx: 0,
            total_frames: 3,
        });
        cb(DecodePhase::Done);

        let got = recorded.lock().unwrap().clone();
        assert_eq!(got.len(), 3);
        assert!(matches!(got[2], DecodePhase::Done));
    }

    #[test]
    fn throttle_lets_first_emit_through() {
        let mut t = ProgressThrottle::new();
        assert!(t.should_emit(false));
    }

    #[test]
    fn throttle_drops_rapid_emits() {
        let mut t = ProgressThrottle::new();
        assert!(t.should_emit(false));
        // Immediately after — should be dropped.
        assert!(!t.should_emit(false));
    }

    #[test]
    fn throttle_lets_boundary_through_always() {
        let mut t = ProgressThrottle::new();
        assert!(t.should_emit(true));
        // Boundary again immediately — also allowed.
        assert!(t.should_emit(true));
    }

    #[test]
    fn throttle_reset_lets_next_emit_through() {
        let mut t = ProgressThrottle::new();
        assert!(t.should_emit(false));
        assert!(!t.should_emit(false));
        t.reset();
        assert!(t.should_emit(false));
    }

    #[test]
    fn noop_callbacks_compile_and_run() {
        let enc = noop_encode_callback();
        let dec = noop_decode_callback();
        enc(EncodePhase::Setup);
        dec(DecodePhase::Demux);
    }

    #[test]
    fn encode_phase_to_ffi_covers_all_variants() {
        assert_eq!(EncodePhase::Setup.to_ffi(), (encode_phase_codes::SETUP, 0, 0));
        assert_eq!(
            EncodePhase::Pass1Capture { frame_idx: 7, total_frames: 30 }.to_ffi(),
            (encode_phase_codes::PASS1_CAPTURE, 7, 30)
        );
        assert_eq!(
            EncodePhase::StcPlan { gop_idx: 2, total_gops: 5 }.to_ffi(),
            (encode_phase_codes::STC_PLAN, 2, 5)
        );
        assert_eq!(
            EncodePhase::Pass2Replay { frame_idx: 30, total_frames: 30 }.to_ffi(),
            (encode_phase_codes::PASS2_REPLAY, 30, 30)
        );
        assert_eq!(EncodePhase::Mux.to_ffi(), (encode_phase_codes::MUX, 0, 0));
        assert_eq!(EncodePhase::Done.to_ffi(), (encode_phase_codes::DONE, 0, 0));
    }

    #[test]
    fn decode_phase_to_ffi_covers_all_variants() {
        assert_eq!(DecodePhase::Demux.to_ffi(), (decode_phase_codes::DEMUX, 0, 0));
        assert_eq!(
            DecodePhase::Walker { frame_idx: 4, total_frames: 12 }.to_ffi(),
            (decode_phase_codes::WALKER, 4, 12)
        );
        assert_eq!(DecodePhase::StcExtract.to_ffi(), (decode_phase_codes::STC_EXTRACT, 0, 0));
        assert_eq!(
            DecodePhase::ShadowExtract { tier_idx: 2, total_tiers: 6 }.to_ffi(),
            (decode_phase_codes::SHADOW_EXTRACT, 2, 6)
        );
        assert_eq!(DecodePhase::Decrypt.to_ffi(), (decode_phase_codes::DECRYPT, 0, 0));
        assert_eq!(DecodePhase::Done.to_ffi(), (decode_phase_codes::DONE, 0, 0));
    }
}
