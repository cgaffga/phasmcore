// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §6E-A6.0 — B-slice macroblock mode decision (encoder side).
//!
//! Driven by the algorithm note at
//! `docs/design/video/h264/encoder-algorithms/6E-A6-bslice-partitions.md`.
//!
//! ## Sub-phase activation
//!
//! Each B mb_type variant lights up in a specific sub-phase. This
//! module defines the [`BMbDecision`] type and the `mb_decision_b`
//! entry point as the dispatch surface; the body progressively
//! enlarges its candidate set across the sub-phases:
//!
//! | Sub-phase | mb_types added | `BMbDecision` variants |
//! |---|---|---|
//! | §6E-A4(c)-lite (shipped) | 0, B_Skip | `Skip`, `Direct16x16` |
//! | §6E-A6.1 | 1, 2, 3 | `L0_16x16`, `L1_16x16`, `Bi_16x16` |
//! | §6E-A6.2 | 4..21 | `Partitioned` (shape + parts via `b_partitioned`) |
//! | §6E-A6.3 | 22 (sub_mb_type 0..3) | `B8x8` |
//! | §6E-A6.4 | 22 (sub_mb_type 4..12) | `B8x8` extended |
//!
//! The §6E-A6.0 stub returns only `Skip` or `Direct16x16` —
//! preserving §6E-A4(c)-lite's emission shape exactly. As §6E-A6.1
//! / .2 / .3 land, they progressively widen the candidate set.
//!
//! ## Mode-decision strategy
//!
//! Cost-based selection over the candidate set with mode-preference
//! penalties from standard MPEG-style B-frame motion estimation
//! (codec-agnostic, allowed per `memory/h264_clean_room_audit.md`):
//!
//! - Direct  ×0 (free; chosen when neighbours agree on a clean
//!   predictor)
//! - Bi      ×1 (slight penalty)
//! - L1      ×2
//! - L0      ×3
//! - Skip    handled via a separate up-front check (CBP-zero shortcut)
//!
//! See `docs/design/video/h264/encoder-algorithms/6E-A6-bslice-partitions.md`
//! § "Cost-penalty rationale" for the rationale + the empirical
//! converter-pipeline-centroid distribution we calibrate against in
//! §6E-A6.5.

use super::motion_estimation::MotionVector;
use super::partition_state::EncoderMvGrid;

/// §B-instrument v1 (#251 follow-on, 2026-05-08) — per-MB record
/// emitted by `mb_decision_b_rdo` when env `PHASM_B_INSTRUMENT=1`.
/// Used by `phase_2_10_b_mb_artifact_dump` test to correlate
/// max-deviation MBs with their picked mode + MVs + RDO data so
/// the actual bug class becomes empirically visible.
#[derive(Debug, Clone, Copy)]
pub struct BMbRecord {
    pub frame_idx: u32,
    pub mb_x: u16,
    pub mb_y: u16,
    pub mode_id: u8,            // 1=Direct 2=L0 3=L1 4=Bi 0=Skip
    pub mv_l0_x: i16,
    pub mv_l0_y: i16,
    pub mv_l1_x: i16,
    pub mv_l1_y: i16,
    pub direct_mv_l0_x: i16,
    pub direct_mv_l0_y: i16,
    pub direct_mv_l1_x: i16,
    pub direct_mv_l1_y: i16,
    pub satd_skip_or_direct: u32,
    pub satd_l0: u32,
    pub satd_l1: u32,
    pub satd_bi: u32,
    pub at_boundary: bool,
    pub src_mean: u8,
    pub src_min: u8,
    pub src_max: u8,
}

static B_MB_RECORDS: std::sync::OnceLock<std::sync::Mutex<Vec<BMbRecord>>> =
    std::sync::OnceLock::new();

/// §B-instrument v1 — current B-frame's display index, set by
/// `Encoder::encode_b_frame` before processing each frame so
/// per-MB records know which frame they belong to.
pub static B_INSTRUMENT_FRAME_IDX: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

fn b_mb_records() -> &'static std::sync::Mutex<Vec<BMbRecord>> {
    B_MB_RECORDS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// §B-instrument v1 — drain accumulated per-MB records (test-side
/// reader). Returns all records collected since last drain.
pub fn drain_b_mb_records() -> Vec<BMbRecord> {
    let mut guard = b_mb_records().lock().unwrap();
    std::mem::take(&mut *guard)
}

/// §intra-in-B (#319) Phase 2-diagnostic — counter of B-MBs whose
/// best inter SATD exceeds [`INTRA_FALLBACK_SATD_THRESHOLD`].
/// Incremented by `mb_decision_b_rdo` regardless of any env-var gate
/// — pure observation, zero decision impact. Phase 3 (encoder emit)
/// will switch this from observation to action.
pub static B_INTRA_FALLBACK_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Read + reset the intra-fallback candidate counter. Tests call this
/// before encode + after to measure how many B-MBs in a fixture
/// would benefit from intra-in-B.
pub fn drain_b_intra_fallback_count() -> u64 {
    B_INTRA_FALLBACK_COUNT.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// §B-instrument v1 — internal pushd; gated by `PHASM_B_INSTRUMENT=1`.
fn push_b_mb_record(record: BMbRecord) {
    if std::env::var_os("PHASM_B_INSTRUMENT").is_none() {
        return;
    }
    if let Ok(mut guard) = b_mb_records().lock() {
        guard.push(record);
    }
}

/// §6E-A6.1 / §6E-A6.2 — process-wide Mutex for tests that
/// manipulate the `PHASM_B_FORCE_MODE` env var. Tests touching
/// the env var (force-mode round-trip + distribution-match)
/// must hold this lock to prevent parallel-test races.
#[cfg(test)]
pub(crate) static B_FORCE_MODE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Outcome of a B-slice mode decision for a single macroblock.
///
/// One variant per spec mb_type family. As §6E-A6.1..§6E-A6.3 land,
/// new variants get filled in with the partition-specific data the
/// encoder needs to emit + populate the MV grid.
///
/// Variant names mirror H.264 spec Table 7-14 (`B_L0_16x16`,
/// `B_L1_16x16`, `B_Bi_16x16`); the spec uses CamelCase prefixes +
/// underscored size suffixes which clippy reads as non-camel-case.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum BMbDecision {
    /// `B_Skip` — `mb_skip_flag = 1`, no further syntax. Decoder
    /// derives the MV via spatial-direct.
    Skip,

    /// `mb_type = 0` (`B_Direct_16x16`) — `mb_skip_flag = 0`,
    /// `mb_type = 0`, `cbp_value = 0`, no MVDs, no residual. Decoder
    /// derives the MV spatially.
    Direct16x16,

    /// `mb_type = 1` (`B_L0_16x16`). One L0 MV, no L1.
    ///
    /// v1.4 Phase 4.3 (#314): `ref_idx_l0` carries the L0 reference
    /// index. Default 0 = closest past anchor (= `dpb.past_anchor`).
    /// Path B post-pass `refine_b_choice_multi_ref` may upgrade it to
    /// 1 (= `dpb.pre_past_anchor`) when ref_1 wins the SATD
    /// comparison.
    L0_16x16 { mv: MotionVector, ref_idx_l0: u8 },

    /// `mb_type = 2` (`B_L1_16x16`). One L1 MV, no L0. `ref_idx_l1 = 0`
    /// (L1 stays single-ref per Q1 design lock).
    L1_16x16 { mv: MotionVector },

    /// `mb_type = 3` (`B_Bi_16x16`). One L0 MV + one L1 MV.
    ///
    /// v1.4 Phase 4.3 (#314): `ref_idx_l0` per Path B post-pass
    /// (default 0). `ref_idx_l1` stays implicit 0 (Q1 lock).
    Bi_16x16 { mv_l0: MotionVector, mv_l1: MotionVector, ref_idx_l0: u8 },

    /// §6E-A6.2 — partitioned B mb_type. Covers all 18 variants
    /// 4..21 (both 16x8 and 8x16 shapes; the shape is encoded in the
    /// looked-up `BPartitionedMeta`). `mb_type` is the spec value
    /// (4..=21); `parts` is the per-partition MV pair, where each
    /// partition's `mv_l0` / `mv_l1` is `Some` only when the
    /// partition's list usage from spec Table 7-14 includes that
    /// list (per `partitioned_b_meta(mb_type).part0/part1`).
    Partitioned {
        mb_type: u8,
        parts: [super::b_partitioned::BPartitionMv; 2],
    },

    /// §intra-in-B (#319) — intra-fallback variant. v1.6 Phase 2-decision
    /// scope: emitted when `min(satd_skip_or_direct, satd_l0, satd_l1,
    /// satd_bi) > 25000` indicating ME couldn't find a usable reference
    /// patch. Walker support shipped Phase 1 (commit 56d2d1b);
    /// counter shipped Phase 2-diagnostic (commit 2843d5e). Encoder
    /// emit path lands in Phase 3 — until then `mb_decision_b_rdo`
    /// does NOT produce this variant, and the encoder match arm
    /// returns an error if reached. See
    /// `memory/h264_v16_intra_in_b_plan_2026_05_10.md`.
    ///
    /// `i16x16_mode`: 0=V, 1=H, 2=DC, 3=Plane (spec § 8.3.3).
    /// `chroma_pred_mode`: 0=DC, 1=H, 2=V, 3=Plane (spec § 8.3.4).
    IntraI16x16 { i16x16_mode: u8, chroma_pred_mode: u8 },

    /// §6E-A6.3 — `mb_type = 22` (`B_8x8`). Four 8x8 sub-MBs, each
    /// with its own `sub_mb_type` (0..=3 — uniform 8x8 family per
    /// the converter-pipeline-centroid distribution finding):
    ///
    /// - 0 = `B_Direct_8x8` (no MVDs, decoder spatial-direct)
    /// - 1 = `B_L0_8x8` (one L0 MV)
    /// - 2 = `B_L1_8x8` (one L1 MV)
    /// - 3 = `B_Bi_8x8` (one L0 MV + one L1 MV)
    ///
    /// `parts[i]` carries the chosen MVs for sub-MB `i`. For Direct,
    /// both fields are `None`. For L0/L1/Bi the matching field(s) are
    /// `Some(mv)`. The encoder consumes `parts[i]` together with the
    /// matching `sub_mb_types[i]` value to emit MVDs in spec order.
    ///
    /// Sub_mb_types 4..=12 (sub-sub partitions — 8x4 / 4x8 / 4x4)
    /// are descoped per §6E-A6.4.
    B8x8 {
        sub_mb_types: [u8; 4],
        parts: [super::b_partitioned::BPartitionMv; 4],
    },
}

impl BMbDecision {
    /// v1.4 Phase 4.3 (#314, Path B) — uniform-per-MB ref_idx_l0
    /// accessor. Reads the ref_idx_l0 from L0_16x16 or Bi_16x16 (the
    /// only variants Path B post-pass currently upgrades). Returns 0
    /// for variants without explicit ref_idx_l0 (Skip, Direct, L1,
    /// Partitioned, B8x8 — those keep ref_idx=0 in v1.4-cut1).
    pub fn ref_idx_l0_uniform(&self) -> u8 {
        match self {
            BMbDecision::L0_16x16 { ref_idx_l0, .. }
            | BMbDecision::Bi_16x16 { ref_idx_l0, .. } => *ref_idx_l0,
            // Partitioned + B8x8 carry ref_idx_l0 inside each
            // BPartitionMv but Path B v1.4-cut1 doesn't upgrade them.
            // Other variants don't use L0 at all.
            _ => 0,
        }
    }
}

/// §6E-A6.0 — entry point for B-slice MB mode decision. The default
/// implementation matches §6E-A4(c)-lite output exactly: deterministic
/// ~50/50 hash-based mix of `Skip` and `Direct16x16`.
///
/// **§6E-A6.1+ widens this** progressively: L0/L1/Bi 16x16 (§6E-A6.1),
/// partitioned 16x8/8x16 (§6E-A6.2), B_8x8 (§6E-A6.3). Real cost-
/// based selection (MPEG-style ×0/×1/×2/×3 mode-preference penalties)
/// lands alongside ME extensions.
///
/// **Test override (§6E-A6.1)**: when the `PHASM_B_FORCE_MODE` env
/// var is set, all B-MBs in the frame return the forced decision
/// instead of the hash-mix. Recognized values:
/// - `skip`        → `Skip`
/// - `direct`      → `Direct16x16`
/// - `l0_16x16`    → `L0_16x16 { mv: (0, 0) }`
/// - `l1_16x16`    → `L1_16x16 { mv: (0, 0) }`
/// - `bi_16x16`    → `Bi_16x16 { mv_l0 = mv_l1 = (0, 0) }`
///
/// Test-only — production callers should never set the var. The
/// override path is what `walk_b_l0_16x16` / `walk_b_l1_16x16` /
/// `walk_b_bi_16x16` round-trip tests use to exercise non-direct
/// paths before real ME-based mode selection lands.
///
/// `_grid` / `_mb_x` / `_mb_y` are accepted now so the signature
/// is forward-stable across the §6E-A6.x sub-phases.
pub fn mb_decision_b(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    frame_num: u32,
    mb_addr: u32,
) -> BMbDecision {
    mb_decision_b_with_mvs(grid, mb_x, mb_y, frame_num, mb_addr, /* me_mvs */ None)
}

/// §6E-A6.1q.b (#151) — variant of [`mb_decision_b`] that consumes
/// pre-computed (L0, L1) ME results. When `me_mvs` is `Some`, the
/// L0/L1/Bi 16x16 buckets emit those MVs; when `None` (test path or
/// no references available — first B before second anchor encoded),
/// they fall back to spatial-direct-predicted MVs (= zero MVD) per
/// §6E-A6.1's original ship.
///
/// The bucket distribution is unchanged — real ME widens the wire
/// MVD range without touching mode-mix proportions. Cost-based RDO
/// (mode mix derived from real cost rather than hash buckets) is a
/// follow-on; today's scope keeps the §Stealth.L3.x calibrated mix
/// while paying the residual + MVD bits a real encoder would.
pub fn mb_decision_b_with_mvs(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    frame_num: u32,
    mb_addr: u32,
    me_mvs: Option<(MotionVector, MotionVector)>,
) -> BMbDecision {
    if let Some(forced) = forced_b_mode_from_env() {
        return forced;
    }

    let bucket = mb_decision_bucket(frame_num, mb_addr);
    // §6E-A6.5 calibration FINAL (#155+#156, 2026-05-03). After
    // FOUR rounds of empirical calibration experiments + a 1080p
    // 14-mode probe, the §6E-A.deploy.4 baseline (50/35/6/5/4)
    // remains the lowest-Σ|Δ| variant tested:
    //
    //   Distribution                              | Σ|Δ| from centroid |
    //   ──────────────────────────────────────────|────────────────────|
    //   Original (50/35/6/5/4)                    |    38.9pp ✓        |
    //   Calib v1 ( 8/5/39/37/4 + 3/2/2)           |   180.5pp          |
    //   Calib v2 (30/30/16/15/2 + 3/2/2)          |   137.2pp          |
    //   Calib v3 (50/30/6/5/1 + 4/3/1)            |   162.2pp          |
    //
    // Why every linear-math attempt fails: the reference decoder's
    // MV-exporter labelling cascades non-linearly through mode mixing.
    //
    // Key empirical findings from the 1080p probe:
    //   - Probe (each mode 100% forced) gives clean per-mode
    //     coefficients: 1.0/1.749/2.0/4.0 r/MB for
    //     Skip/Direct/Partitioned/B_8x8.
    //   - But in MIXED distributions, those coefficients DO NOT
    //     compose linearly. Adding 1% B_8x8 to a 80%-Skip-Direct
    //     mix yields ~28% 8x8 records (predicted 3.4%) — a 7×
    //     emission cascade.
    //   - Skip's direction in isolated probe = 100% Bi. In
    //     mixed-context (pre-cal) it's ~47%/46%/6% (matching
    //     Direct's spatial-direct distribution). Skip's behaviour
    //     SWITCHES based on neighbouring modes.
    //   - B_8x8 in any sub_mb_type form (0/1/2/3 or mixed) emits
    //     100% Bi labels in the reference decoder's MV exporter —
    //     a hard quirk of that exporter, not of the bitstream.
    //
    // Path to actually closing the gap (#156, post-#118 wider corpus):
    //   (a) Trace the reference-decoder MV exporter per-mb_type to
    //       understand cascade rules empirically, OR
    //   (b) Use an alternative reference parser to disambiguate
    //       exporter-quirk vs phasm-encoder, OR
    //   (c) Iterative gradient descent on bucket sizes (each iter
    //       ~50s at 1080p × 10f). Not done in this session.
    //
    // Probe + calibration history at
    // `core/tests/h264_stego_distribution_probe.rs`.
    // §B-direct-fix root cause 2026-05-03: the previous bucket
    // distribution (50/35/6/5/4 for Skip/Direct/L0/L1/Bi) emitted
    // ~15% of B-MBs as L0/L1/Bi based on a CONTENT-INDEPENDENT hash.
    // Those MBs' MVs (whether ME-derived or neighbor-predicted)
    // propagate into the grid and corrupt downstream Skip/Direct
    // MBs via spatial-direct neighbor-median chain → visible
    // MB-level scrambling on real 1080p video at IBPBP shape.
    //
    // The fix: when RDO is off (this fallback path), emit Skip /
    // Direct ONLY. RDO path still emits L0/L1/Bi/Partitioned/B_8x8
    // via cost-based decisions over real ME results, which do
    // converge on visually-correct MVs.
    //
    // This trades distribution-match (Σ|Δ| widens from 38.9pp →
    // ~67pp on the §6E-A6.5 gate) for visual correctness. Since
    // production output goes through default-IPPPP per the
    // b01afcb hotfix, IBPBP-stealth mode is opt-in and explicitly
    // calls the RDO path. v1.1 cleanup: remove the buckets entirely
    // once RDO is the unconditional default and PHASM_B_RDO env
    // gate is retired.
    let _ = me_mvs; // unused in the Skip/Direct-only path
    match bucket {
        // 0..=49 → Skip (50%, unchanged)
        0..=49 => BMbDecision::Skip,
        // 50..=99 → Direct (50%, was 35% Direct + 15% L0/L1/Bi).
        _ => BMbDecision::Direct16x16,
    }
}

/// Task #204 (v1.1) — typed B-frame RDO configuration. Replaces the
/// `PHASM_B_RDO` / `PHASM_B_RESIDUAL` env-var pair with a struct that
/// can be set on [`super::encoder::Encoder::b_rdo_config`] before
/// invoking `encode_b_frame`.
///
/// Defaults are off so legacy callers keep the bucket-fallback Skip /
/// Direct mode-mix path (visually clean on real 1080p video at IBPBP
/// shape). Production paths that need real ME + residual emission
/// (iPhone7 visual demo, future stealth-calibrated CLI/mobile flow)
/// opt in by setting the field to [`BRdoConfig::PRODUCTION_VISUAL`]
/// before encode.
///
/// Env vars `PHASM_B_RDO=1` / `PHASM_B_RESIDUAL=1` remain available
/// as **debug overrides**: when set they win over the config field.
/// This keeps the existing test corpus running unchanged while new
/// callers can use the typed API.
///
/// §B-RDO.proper.6 (2026-05-04 — note): when both flags default ON,
/// two shadow tests regress (`n_shadows_roundtrip_n_equals_2`,
/// `shadow_roundtrip_handles_longer_primary_via_cascade`). Tracked
/// separately under task #204's "fix shadow tests under
/// B_RDO+RESIDUAL" deliverable — for now this struct provides clean
/// opt-in without forcing the default flip.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BRdoConfig {
    /// Enable fast-RDO B-slice mode decision in [`mb_decision_b_rdo`].
    /// When `false`, falls back to the §B-direct-fix Skip/Direct-only
    /// bucket mix (visually clean default).
    pub enable_rdo: bool,
    /// Enable B-MB residual emission (CBP > 0 with luma 4×4 + chroma
    /// DC/AC residuals + post-residual recon). When `false`, emits
    /// CBP=0 with prediction-only recon. Pairs with `enable_rdo` —
    /// RDO selecting L0/L1/Bi/Partitioned/B_8x8 modes without
    /// residual produces visible artifacts on motion content.
    pub enable_residual: bool,
    /// **Phase F (#262, 2026-05-08, v1.0 ship path)** — when `true`,
    /// override mode decision to emit B_L0_16x16 with MV=(0,0) for
    /// EVERY B-MB, residual emitted normally. Bypasses spatial-direct
    /// MV derivation and ME entirely. Pre-empts the
    /// encoder/decoder divergence on derived MVs (Phase D bisect
    /// finding 2026-05-08): forced (0,0) MV makes encoder + decoder
    /// MC paths agree on reference position, eliminates ghost-image
    /// + blocky artifacts on motion content.
    ///
    /// Trade-off: B-frames don't get any motion-prediction benefit
    /// (essentially predict-from-past-anchor + residual). PSNR is
    /// ~equal-to-better than the broken RDO path on motion content
    /// (+0.4-0.7 dB on the 4 broken corpus fixtures), but ~1-3 dB
    /// below an IPPPP encode (no B-frames at all).
    ///
    /// Stealth: maintains the IBPBP container shape (B nal_ref_idc,
    /// B slice headers, B_L0_16x16 mb_types in bitstream). Layer 4
    /// container fingerprint matches IBPBP. Layer 3 mode mix is
    /// degenerate (100% L0_16x16 on B-MBs) — visible to a forensic
    /// analyst running a B-mode-distribution histogram, but L4
    /// container is what most stego tools surface first.
    ///
    /// v1.1 follow-on: align encoder spatial-direct + bipred MC code
    /// with decoder spec § 8.4.1.2 to remove the workaround.
    pub force_l0_zero_b_mode: bool,
}

impl BRdoConfig {
    /// Conservative default — no RDO, no residual. Bucket-fallback
    /// Skip/Direct only on B-frames. Visually clean on all content;
    /// stealth distribution gap is the price.
    pub const SAFE: Self = Self {
        enable_rdo: false,
        enable_residual: false,
        force_l0_zero_b_mode: false,
    };

    /// Production visual-quality preset — full RDO + residual
    /// emission. Required for stealth-calibrated paths (iPhone7
    /// visual, post-#206 mobile stealth).
    pub const PRODUCTION_VISUAL: Self = Self {
        enable_rdo: true,
        enable_residual: true,
        force_l0_zero_b_mode: false,
    };

    /// **Phase F v1.0 ship path** — RDO + residual ON, but every
    /// B-MB is overridden to L0_16x16+MV=(0,0). Bypasses derived-MV
    /// paths so encoder/decoder MC agrees. Closes the corpus
    /// visual-quality bug (Phase D bisect 2026-05-08).
    pub const SAFE_L0_ZERO: Self = Self {
        enable_rdo: true,
        enable_residual: true,
        force_l0_zero_b_mode: true,
    };
}

/// §6E-D.5 — predicate for whether fast-RDO B-slice mode decision
/// is active. Reads the config field plus a `PHASM_B_RDO` env-var
/// debug override (env wins for backward compat with the existing
/// test corpus).
///
/// Task #204: pass an explicit [`BRdoConfig`] from the encoder
/// instead of relying on env-var grossness. Existing callers can
/// pass `BRdoConfig::SAFE` for the legacy default behaviour.
pub fn b_rdo_enabled_with(config: BRdoConfig) -> bool {
    match std::env::var("PHASM_B_RDO") {
        Ok(v) if v == "1" => true,
        Ok(v) if v == "0" => false,
        _ => config.enable_rdo,
    }
}

/// Legacy shim — `b_rdo_enabled()` with no config arg falls back to
/// env var only (= `BRdoConfig::SAFE` as default). Kept for in-place
/// callers in `mb_decision_b.rs` that haven't been threaded the
/// config yet. New code should call [`b_rdo_enabled_with`].
pub fn b_rdo_enabled() -> bool {
    b_rdo_enabled_with(BRdoConfig::SAFE)
}

/// §6E-D.5b Track A — return `true` iff the spatial-direct
/// prediction's luma residual would quantize to all zeros at QP
/// (CBP_luma=0). Skip pre-check: when this returns true, the caller
/// emits `BMbDecision::Skip` without scoring the RDO candidate set.
///
/// Implementation: build the spatial-direct luma prediction, compute
/// per-4×4-block residual = src − pred, run forward DCT + plain quant
/// (no trellis — quant rounding under the dead-zone is what determines
/// CBP=0), and check if every coefficient quantizes to 0.
///
/// Chroma CBP is NOT checked here (the reference fast encoder checks
/// it separately, but a luma-zero MB nearly always implies chroma-zero
/// at the same QP for realistic content). If false-Skip emissions show
/// up in the gate, extend with chroma quantization.
fn skip_cbp_is_zero(
    direct: &super::b_direct_predictor::BDirectSpatialResult,
    src_y: &[[u8; 16]; 16],
    l0_ref: &super::reference_buffer::ReconFrame,
    l1_ref: &super::reference_buffer::ReconFrame,
    mb_x: usize,
    mb_y: usize,
    mb_qp: u8,
) -> bool {
    use super::motion_compensation::{apply_luma_mv_block, apply_luma_mv_block_bipred};
    use super::quantization::{forward_quantize_4x4, trellis_quantize_4x4, QuantParams, QuantSlice};
    use super::transform::forward_dct_4x4;
    use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

    if !direct.uses_l0() && !direct.uses_l1() {
        return false; // pathological — fall through to RDO
    }

    // Phase 2.12.f (#282, 2026-05-08) — when colZeroFlag override fires
    // on at least one sub-block, the actual Skip-emit MC uses per-8x8
    // MVs (Phase 2.12.e). The pre-check must mirror that: a uniform
    // single-MV prediction here would say "all zero" while per-8x8
    // emit produces a different (non-zero residual) prediction. Use
    // the per-8x8 builder when override fires; legacy single-MV
    // builder otherwise (preserves SIMD perf for non-static colMb).
    let pred = if direct.has_per_8x8_override_l0() || direct.has_per_8x8_override_l1() {
        super::b_inter_prediction::build_b_luma_prediction_per_8x8(
            direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
            direct.uses_l0(), direct.uses_l1(),
            l0_ref, l1_ref, mb_x, mb_y,
        )
    } else {
        let mut pred = [[0u8; 16]; 16];
        let pred_flat = pred.as_flattened_mut();
        let mb_px_x = (mb_x * 16) as u32;
        let mb_px_y = (mb_y * 16) as u32;
        match (direct.uses_l0(), direct.uses_l1()) {
            (true, true) => apply_luma_mv_block_bipred(
                l0_ref, direct.mv_l0, l1_ref, direct.mv_l1,
                mb_px_x, mb_px_y, 16, 16, pred_flat, 16,
            ),
            (true, false) => apply_luma_mv_block(
                l0_ref, mb_px_x, mb_px_y, 16, 16, direct.mv_l0, pred_flat, 16,
            ),
            (false, true) => apply_luma_mv_block(
                l1_ref, mb_px_x, mb_px_y, 16, 16, direct.mv_l1, pred_flat, 16,
            ),
            (false, false) => unreachable!(),
        }
        pred
    };

    // §B-Direct-distortion-validation v3 (2026-05-08, #251) —
    // raw-SAD sanity check before trusting the all-zero-coeff
    // indicator. At motion-boundary MBs the spatial-direct MV
    // points to wall background; source is jacket content;
    // residual is large in absolute terms BUT trellis-quant can
    // still zero it out at QP=26 because the rate-cost-zero
    // structural advantage of "zero coefficients" wins the
    // trellis decision. Net effect: Skip emits the wall-pixel
    // prediction with no residual correction → output shows wall
    // pixels in jacket region (visible blocky-grey artifact in
    // V14/V20 demos).
    //
    // Validate by computing total absolute prediction error (SAD).
    // 16×16 = 256 pixels. SAD threshold 3072 (= 256 × 12 avg dev)
    // empirically separates content-correct Skip predictions
    // (SAD ≤ ~1500 per V21 force-L0 noise floor) from
    // content-wrong Skip predictions (SAD ≥ 5000+ at jacket-vs-
    // wall mismatch). Disable via PHASM_B_SKIP_NO_SAD_CHECK=1.
    if std::env::var_os("PHASM_B_SKIP_NO_SAD_CHECK").is_none() {
        let mut sad: u32 = 0;
        for dy in 0..16 {
            for dx in 0..16 {
                sad += (src_y[dy][dx] as i32 - pred[dy][dx] as i32).unsigned_abs();
            }
        }
        if sad > 3072 {
            return false; // refuse early-Skip; let RDO decide.
        }
    }

    // Forward DCT + quant per 4×4 block; bail on first non-zero level.
    let inter = QuantParams { qp: mb_qp, slice: QuantSlice::Inter };
    for k in 0..16 {
        let (bx, by) = BLOCK_INDEX_TO_POS[k];
        let sby = by as usize;
        let sbx = bx as usize;
        let mut sub_res = [[0i32; 4]; 4];
        for dy in 0..4 {
            for dx in 0..4 {
                sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                    - pred[sby * 4 + dy][sbx * 4 + dx] as i32;
            }
        }
        let coeffs = forward_dct_4x4(&sub_res);
        // §6E-D.5(g): trellis-quant matches the rest of the encoder
        // pipeline (encoder.rs B-frame residual path) and is more
        // permissive at zeroing small coefficients than plain quant.
        // Without this, plain-quant retained near-deadzone coefficients
        // that trellis would have zeroed → Skip share artificially low.
        let levels = trellis_quantize_4x4(&coeffs, inter, true)
            .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter));
        for row in &levels {
            for &v in row {
                if v != 0 {
                    return false;
                }
            }
        }
    }
    true
}

/// §6E-D.5 — fast-RDO B-slice mode decision.
///
/// Builds a candidate set covering the 16x16 family (Skip, Direct,
/// L0_16x16, L1_16x16, Bi_16x16) and picks the lowest-cost mode via
/// `evaluate_b_mb_rdo`. Skip-vs-Direct disambiguation: both share the
/// same SATD (same spatial-direct prediction), so the rate term
/// determines the winner — Skip's r_bits=1 always beats Direct's
/// r_bits=6 when CBP would be zero in either case. The function
/// scores both wire forms internally and returns the cheaper.
///
/// Partitioned (mb_types 4..21) + B_8x8 (22) candidates are NOT
/// scored in this entry yet — they require per-partition ME which
/// the encoder loop doesn't currently compute. Phase 6E-D.5b adds
/// the partition ME hooks; for now the wider candidate set lights
/// up via `PHASM_B_RDO=1` without partitioned modes, narrowing the
/// 16x16 mix to content-driven cost rather than hash buckets.
///
/// Determinism: pure function of inputs + ME results. No RNG.
/// Required for Pass 1 / Pass 3 stego mode parity.
///
#[allow(clippy::too_many_arguments)]
pub fn mb_decision_b_rdo(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    src_y: &[[u8; 16]; 16],
    l0_ref: &super::reference_buffer::ReconFrame,
    l1_ref: &super::reference_buffer::ReconFrame,
    mb_qp: u8,
    me_mvs: (MotionVector, MotionVector),
) -> BMbDecision {
    use super::rdo_b::{evaluate_b_mb_rdo, BMbCandidate};

    // §6E-D.5(a) — derive spec-correct list usage for SkipOrDirect
    // from neighbour state (spec § 8.4.1.2.1 + § 8.4.1.2.2).
    // §B-direct-fix #196 — pass the L1 ref's collocated MV grid so
    // the colZeroFlag check fires for static-background MBs (forces
    // mv_l0/l1 = (0, 0) when colMb is static, preventing the moving
    // subject's MV from leaking via neighbour median).
    let direct = super::b_direct_predictor::derive_b_direct_spatial_with_col(
        grid, mb_x, mb_y, l1_ref.motion_grid.as_ref(),
    );

    // §6E-D.5b Track A — Skip-CBP-zero pre-check.
    //
    // Before running RDO over candidates, apply the standard fast
    // B-frame analysis path: if the spatial-direct prediction's
    // residual quantizes to ALL zeros at the current QP, emit Skip
    // immediately and skip the RDO sweep entirely. This is the
    // structural reason the converter-pipeline centroid emits ~50%
    // of B-MBs as Skip; without the pre-check, RDO almost never
    // picks Skip because the L0/L1/Bi candidates always score better
    // on raw SATD (they use ME-derived MVs vs Skip's spatial-direct
    // MVs). §6E-D.6 + §6E-D.5(a) measurements showed Skip share at
    // 0.21% without this gate (target 50%+). Pre-check is not
    // cosmetic calibration — it's the standard fast encoder pattern.
    // Diagnostic env-gate: PHASM_B_SKIP_NO_CBP_PRECHECK=1 disables the
    // early-Skip CBP-zero pre-check entirely, forcing every B-MB
    // through the full RDO sweep. Used to measure how much of the
    // visible drift on motion content is driven by Skip-without-
    // residual emission. Default OFF (production behaviour unchanged).
    if std::env::var_os("PHASM_B_SKIP_NO_CBP_PRECHECK").is_none()
        && skip_cbp_is_zero(&direct, src_y, l0_ref, l1_ref, mb_x, mb_y, mb_qp)
    {
        // Phase 2.7 (#272 follow-on) — Skip pre-check fires for many
        // MBs in default RDO mix. Without instrumentation, the
        // post-fix bisect harness shows them as "BMbRecord missing"
        // and assumes Partitioned/B_8x8. Push a Skip-flavoured
        // record so the harness can distinguish.
        if std::env::var_os("PHASM_B_INSTRUMENT").is_some() {
            let frame_idx = B_INSTRUMENT_FRAME_IDX
                .load(std::sync::atomic::Ordering::Relaxed);
            push_b_mb_record(BMbRecord {
                frame_idx,
                mb_x: mb_x as u16,
                mb_y: mb_y as u16,
                mode_id: 0, // Skip
                mv_l0_x: direct.mv_l0.mv_x,
                mv_l0_y: direct.mv_l0.mv_y,
                mv_l1_x: direct.mv_l1.mv_x,
                mv_l1_y: direct.mv_l1.mv_y,
                direct_mv_l0_x: direct.mv_l0.mv_x,
                direct_mv_l0_y: direct.mv_l0.mv_y,
                direct_mv_l1_x: direct.mv_l1.mv_x,
                direct_mv_l1_y: direct.mv_l1.mv_y,
                satd_skip_or_direct: 0,
                satd_l0: 0,
                satd_l1: 0,
                satd_bi: 0,
                at_boundary: false,
                src_mean: 0,
                src_min: 0,
                src_max: 0,
            });
        }
        return BMbDecision::Skip;
    }

    let skip_or_direct = BMbCandidate::SkipOrDirect {
        mv_l0: direct.mv_l0,
        mv_l1: direct.mv_l1,
        uses_l0: direct.uses_l0(),
        uses_l1: direct.uses_l1(),
        mv_l0_per_8x8: direct.mv_l0_per_8x8,
        mv_l1_per_8x8: direct.mv_l1_per_8x8,
    };
    // v1.4 Phase 2 (#305): single-ref default ref_idx_l0=0. Multi-ref
    // wiring lands in Phase 4 (ME) + Phase 5 (RDO ref selection).
    let l0 = BMbCandidate::L0_16x16 { mv_l0: me_mvs.0, ref_idx_l0: 0 };
    let l1 = BMbCandidate::L1_16x16 { mv_l1: me_mvs.1 };
    let bi = BMbCandidate::Bi_16x16 { mv_l0: me_mvs.0, mv_l1: me_mvs.1, ref_idx_l0: 0 };

    // Score the explicit-MV candidates with the SATD-domain fast path.
    // The §6E-D.5(m) HF-proportional Direct multiplier + §6E-D.5(o) Bi
    // discount inside `evaluate_b_mb_rdo` compensate for SATD being a
    // distortion proxy.
    let r_skip_or_direct = evaluate_b_mb_rdo(&skip_or_direct, src_y, l0_ref, l1_ref, mb_x, mb_y, mb_qp);
    let r_l0 = evaluate_b_mb_rdo(&l0, src_y, l0_ref, l1_ref, mb_x, mb_y, mb_qp);
    let r_l1 = evaluate_b_mb_rdo(&l1, src_y, l0_ref, l1_ref, mb_x, mb_y, mb_qp);
    let r_bi = evaluate_b_mb_rdo(&bi, src_y, l0_ref, l1_ref, mb_x, mb_y, mb_qp);

    // §6E-D.5(e) — Skip wire form is NOT a free RDO candidate.
    //
    // Earlier (commits up to 8045068) we computed
    //   `skip_cost = direct_satd + λ × 1`
    // and let it compete against L0/L1/Bi which charge ~λ × 18+.
    // The 17λ rate gap means Skip ALWAYS wins as long as direct's
    // spatial-direct prediction is within ~17λ SATD of the best
    // explicit-MV candidate — true for nearly every B-MB. §6E-D.5(d)
    // trace confirmed: Skip share was 99.1% before §6E-D.5(c), and
    // STILL 99% after §6E-D.5(c) bipred refinement, because the
    // artificial skip_cost dominated regardless of explicit-MV
    // SATD improvements.
    //
    // The fix: Skip wire form is ONLY emitted by the CBP-zero
    // pre-check (Track A above) — the standard fast B-frame
    // analysis pattern. If the residual genuinely quantizes to
    // zero, emit Skip. Otherwise, the encoder MUST emit a mode
    // that includes residual — and Skip is not in that race.
    // Removing the artificial `skip_cost` from the RDO comparison
    // forces explicit-MV candidates to compete only with Direct,
    // which is the correct rate-distortion tradeoff.
    //
    // Effect: when Track A returns false (CBP would be non-zero),
    // best is initialised to SkipOrDirect (Direct emit form, 6
    // bin overhead) and L0/L1/Bi can beat it on cost when their
    // SATD savings exceed the rate gap.

    // §6E-D.5(d) — Bi early-termination threshold.
    //
    // Standard fast-encoder Bi SATD-threshold mechanism. On the same
    // 10-frame IMG_4138 fixture, the converter-pipeline centroid
    // emits Bi for only 1.10% of B-MBs; phasm RDO without this gate
    // emits Bi 71.82% (see corresponding memory note on the same-fixture gate).
    // Root cause: bipred averaging on real motion content has a
    // 30-50% SATD advantage that beats single-list rate-cost most of
    // the time. The standard pattern short-circuits this by not even
    // considering Bi unless its SATD advantage over the BEST
    // single-list exceeds a meaningful margin. Below the margin,
    // single-list "won" and Bi gets dropped from the candidate set
    // entirely.
    //
    // §6E-D.5(k) — Bi early-termination threshold REMOVED.
    //
    // The threshold was added in §6E-D.5(d) when Bi share was
    // 71.82% (clearly broken). It treated the symptom: Bi was
    // over-emitting because of a reference-decoder MV-exporter
    // labelling artifact (Skip+spatial-direct synthesized as Bi
    // labels) and because the artificial skip_cost + missing Direct
    // distortion penalty were giving Direct/Skip an unfair cost
    // advantage.
    //
    // §6E-D.5(e) (drop artificial skip_cost) + §6E-D.5(i)+(j)+(k)
    // (Direct distortion penalty) fixed the structural causes. With
    // those in place, Bi cost-comparison is honest: Bi must save
    // ~12 bins of rate (≈12λ SATD ≈ 3300 at QP=34) over the best
    // single-list to win. That's a real margin, content-driven.
    //
    // Removing the threshold lets Bi emerge naturally where it
    // genuinely beats single-list by enough to overcome its rate
    // overhead. Per-MB defendable: each Bi emission means "bipred
    // SATD savings exceeded the 12-bin extra rate cost over single-
    // list". The converter-pipeline centroid emits 1.42% Bi on the
    // same fixture — small but non-zero, exactly this content-driven
    // pattern. Phasm should land in a similar low-single-digit range.
    //
    // Always-true now (kept as identity for the diagnostic logging).
    let best_single_satd = r_l0.satd.min(r_l1.satd);
    let bi_passes_threshold = true;
    let _ = best_single_satd; // referenced below by diagnostic only

    if std::env::var_os("PHASM_B_RDO_LOG_BI").is_some() {
        eprintln!(
            "B-MB ({mb_x},{mb_y}) qp={mb_qp} l0_satd={} l1_satd={} bi_satd={} ratio={:.3} pass={}",
            r_l0.satd, r_l1.satd, r_bi.satd,
            (r_bi.satd as f64) / (best_single_satd.max(1) as f64),
            bi_passes_threshold,
        );
    }

    // §6E-D.5(d) v2 diagnostic — emit decision-counter histogram.
    // PHASM_B_RDO_TRACE=1 dumps a per-decision count every 1000 MBs.
    // Note: counts ONLY MBs that fall through to RDO — not the
    // CBP-zero-Skip-pre-check path (those are counted upstream
    // by the encoder loop's BMbDecision::Skip emit branch).
    if std::env::var_os("PHASM_B_RDO_TRACE").is_some() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DIRECT_CNT: AtomicUsize = AtomicUsize::new(0);
        static L0_CNT: AtomicUsize = AtomicUsize::new(0);
        static L1_CNT: AtomicUsize = AtomicUsize::new(0);
        static BI_CNT: AtomicUsize = AtomicUsize::new(0);
        // Mirror the actual decision below (skip_cost path removed
        // in §6E-D.5(e); Skip only via CBP-zero pre-check).
        let dec_label;
        let mut best_for_trace = (r_skip_or_direct.cost, 1_u8);
        if r_l0.cost              < best_for_trace.0 { best_for_trace = (r_l0.cost, 2); }
        if r_l1.cost              < best_for_trace.0 { best_for_trace = (r_l1.cost, 3); }
        if bi_passes_threshold && r_bi.cost < best_for_trace.0 { best_for_trace = (r_bi.cost, 4); }
        match best_for_trace.1 {
            1 => { DIRECT_CNT.fetch_add(1, Ordering::Relaxed); dec_label = "Direct"; }
            2 => { L0_CNT.fetch_add(1, Ordering::Relaxed); dec_label = "L0"; }
            3 => { L1_CNT.fetch_add(1, Ordering::Relaxed); dec_label = "L1"; }
            4 => { BI_CNT.fetch_add(1, Ordering::Relaxed); dec_label = "Bi"; }
            _ => unreachable!(),
        }
        let total = DIRECT_CNT.load(Ordering::Relaxed)
            + L0_CNT.load(Ordering::Relaxed) + L1_CNT.load(Ordering::Relaxed)
            + BI_CNT.load(Ordering::Relaxed);
        if total % 1000 == 0 {
            eprintln!(
                "B-RDO trace (RDO-only) [{total}]: Direct={} L0={} L1={} Bi={} (this={dec_label})",
                DIRECT_CNT.load(Ordering::Relaxed),
                L0_CNT.load(Ordering::Relaxed), L1_CNT.load(Ordering::Relaxed),
                BI_CNT.load(Ordering::Relaxed),
            );
        }
    }

    // §B-direct-fix.v3 Path 4.1 — surgical motion-boundary penalty.
    //
    // Direct mode (spatial-direct § 8.4.1.2.2 OR temporal-direct
    // § 8.4.1.2.3) derives a SINGLE MV per MB from neighbour state,
    // either via median over A/B/C (spatial) or scaled colMb
    // (temporal). On motion BOUNDARIES — where neighbour MVs differ
    // sharply because the MB straddles two motion regions — neither
    // derivation strategy can express the bimodal motion. The result
    // is "wrong reference region" prediction → the user-visible
    // "pants pixels spilling onto wall" / "wall pixels invading
    // pants" artifact at object edges (V7 visual A/B 2026-05-07
    // confirmed: spatial puts artifact INSIDE pants, temporal puts
    // it OUTSIDE — neither hides it).
    //
    // Real-world consumer-grade encoders pick Direct ~55% of B-MBs
    // vs phasm's ~99% (memory: h264_phase6e_d_5c_5e_result.md). The
    // 44pp gap concentrates exactly on these boundary MBs because
    // their full RDO routes them to L0/L1/Bi where the explicit MV
    // is content-correct.
    //
    // This penalty detects a motion boundary by neighbour-MV span
    // (max - min on each axis over top-left, top, top-right, left).
    // Above a threshold, multiplies SkipOrDirect's cost so RDO
    // routes the MB to an explicit-MV mode. Per-MB defendable: each
    // bypassed-Direct emission corresponds to a real bimodal-MV
    // neighbourhood where Direct's single-MV output cannot encode
    // the actual motion. Threshold + ramp picked to fire only on
    // sharp boundaries (≥1 px MV span across neighbours), NOT on
    // ambient ME noise.
    //
    // Env-gated via `PHASM_B_BOUNDARY_PENALTY=1` until validated by
    // visual demo. Default off; flip to default on after V8 demo
    // shows the visible artifact closes.
    let direct_cost_adjusted = if std::env::var_os("PHASM_B_BOUNDARY_PENALTY").is_some() {
        let factor_q8 = compute_motion_boundary_factor_q8(grid, mb_x, mb_y);
        ((r_skip_or_direct.cost as u128 * factor_q8 as u128) >> 8) as u64
    } else {
        r_skip_or_direct.cost
    };

    // §B-Direct-distortion-validation v1.0 (#251 follow-on, 2026-05-08).
    //
    // Root cause of visible v1.0 BLOCKER (per V21 force-L0 demo
    // 2026-05-08): RDO picks Direct/Skip even when its derived MV
    // points to wrong content (e.g. spatial-direct median over wall
    // neighbours = (0,0) at jacket-edge MB → Direct predicts wall
    // pixels into jacket region). The rate-cost-zero advantage of
    // Direct's no-MVD emission lets it win even when its raw SATD
    // is significantly worse than an explicit mode's SATD.
    //
    // V21 (force PHASM_B_FORCE_MODE=l0_16x16) showed body=570 /
    // max_dev=51 vs default RDO V14 body=1005 / max_dev=164. With
    // Direct removed entirely, output is visually clean. With
    // Direct present, RDO picks it on motion boundaries and produces
    // the blocky-grey-on-jacket artifact. Phase 4.x ME tuning
    // (V18-V20) was symptom-treatment.
    //
    // Distortion validation: refuse Direct when its raw SATD is
    // > k × best-explicit-SATD. Threshold k=1.5 picked empirically:
    // - Uniform-motion MBs: Direct's MV ≈ explicit ME's MV → SATDs
    //   match (ratio ≈ 1.0) → no penalty.
    // - Motion-boundary MBs: Direct's spatial-derived MV is
    //   content-wrong; explicit ME finds correct MV; SATD ratio
    //   typically > 2.0 → Direct refused.
    //
    // Env-gateable via `PHASM_B_DIRECT_NO_VALIDATE=1` (default OFF
    // = validation enabled). Disable for ablation experiments.
    // §B-boundary-anchor v1 (#251) — refuse Direct at motion-
    // boundary MBs (neighbour-MV variance trigger).
    let at_boundary = is_motion_boundary(grid, mb_x, mb_y);

    // §B-direct-magnitude-clamp v1 (#251 follow-on, 2026-05-08) —
    // refuse Direct when its derived spatial-direct MV magnitude
    // exceeds plausible-motion threshold (32 qpel = 8 px).
    //
    // Phase 2.10 instrumentation finding 2026-05-08: top-10
    // worst-luma-deviation B-MBs in V14-baseline default RDO
    // are EXCLUSIVELY mode=Direct with derived MVs ranging from
    // 25-50 pixels (e.g. (-109, 28), (-23, 201), (-35, 199)).
    // These are not local boundaries — chain-propagated MVs
    // inherited via spatial-median from upstream MBs that picked
    // wrong huge ME results. Variance-based boundary detection
    // misses them (all neighbours have the same huge MV → low
    // variance). Absolute-magnitude check catches them.
    //
    // 32 qpel = 8 pixels is x4 typical camera-shake displacement
    // and x16 the MV range we'd expect at 30fps for hand-held
    // jacket-motion content. MVs > 8 px on this content are
    // chain-propagated nonsense, not real motion-tracking.
    //
    // Default ON. Disable via PHASM_B_NO_DIRECT_MAGCLAMP=1.
    const DIRECT_MV_QPEL_THRESHOLD: i16 = 32;
    let direct_mv_too_large = direct.mv_l0.mv_x.abs() > DIRECT_MV_QPEL_THRESHOLD
        || direct.mv_l0.mv_y.abs() > DIRECT_MV_QPEL_THRESHOLD
        || direct.mv_l1.mv_x.abs() > DIRECT_MV_QPEL_THRESHOLD
        || direct.mv_l1.mv_y.abs() > DIRECT_MV_QPEL_THRESHOLD;

    // Phase 2.19 (#288, 2026-05-09) — Path B' SATD-floor refusal of
    // Direct.
    //
    // Diagnosis: at high-motion B-frames (e.g., carplane display=5,
    // src delta 19 dB → MC error >> QP=26 noise floor), RDO picks
    // Direct for every MB because Direct's rate cost is ~6 bins vs
    // L0_16x16's 30+ bins, which dominates over the SATD difference
    // when both modes have similar (high) SATD. The result is a B-
    // frame with CBP=0 everywhere — bitstream collapses to ~1KB and
    // recon = bipred MC at spatial-direct MV ≈ 21 dB PSNR vs source.
    //
    // Mode-count instrumentation 2026-05-09 confirmed: at carplane
    // display=5, RDO picks 100% Skip+Direct (3936+4104=8040 MBs),
    // ZERO L0/L1/Bi. the converter-pipeline centroid on identical content/QP gets 43 dB
    // because it emits explicit-MV+residual when Direct's prediction
    // is poor (real-world distribution).
    //
    // This is a fast-RDO approximation flaw: SATD on prediction
    // error doesn't credit residual's distortion reduction. L0_16x16
    // computed cost ≈ Skip cost (same SATD) + higher rate → Skip
    // wins. But L0_16x16 EMITS residual that quantizes the error down
    // to QP-noise level (~64 SAD post-quant ≈ 38 dB). Skip emits no
    // residual → 21 dB PSNR.
    //
    // Fix: refuse Direct when its SATD exceeds the same SAD-quality
    // floor used by `skip_cbp_is_zero` (3072 = 256 × 12 avg dev).
    // Above the floor, Direct's no-residual emission cannot recover
    // the prediction error, so route the MB to an explicit-MV
    // candidate that includes residual emission. The residual
    // closes the prediction-vs-source gap.
    //
    // Per-MB defendable: each Direct refusal corresponds to a real
    // case where the spatial-direct prediction is too far from
    // source for CBP=0 emission to produce visually correct output.
    // Stealth-positive: real encoders also emit explicit-MV at high
    // motion (the converter-pipeline centroid 1.10% Bi + 41% L0/L1 on carplane).
    //
    // Default ON. Disable via PHASM_B_NO_DIRECT_SATD_FLOOR=1.
    const DIRECT_SATD_FLOOR: u32 = 3072;
    let direct_satd_too_high = r_skip_or_direct.satd > DIRECT_SATD_FLOOR
        && std::env::var_os("PHASM_B_NO_DIRECT_SATD_FLOOR").is_none();

    // v1.5 §B-fast-motion (#317) — RELATIVE SATD gate. The absolute
    // DIRECT_SATD_FLOOR (3072) misses MBs where Direct's SATD is
    // moderate but explicit-ME modes have much lower distortion.
    // Empirical observation on carplane road MBs (max|Δ|>20 cluster):
    // Direct picked even when L0/L1's SATD was 1.5–3× lower because
    // Direct's rate=0 dominates λ·R at QP=26. Result: visible block
    // artifacts on flat-ish moving content where Direct's stale
    // (0,0) prediction differs from source by a constant offset that
    // the dead-zone-quantized residual fails to correct.
    //
    // Spec-aligned behaviour: the converter-pipeline centroid picks
    // L0/L1 over Direct when SATD ratio >> 1, even at low rate cost.
    // The phasm RDO undervalues Direct's distortion at low-HF MBs
    // (multiplier ~1.0 there).
    //
    // Threshold 3/2 = 1.5×. Disable via PHASM_B_NO_DIRECT_SATD_RATIO=1.
    let best_alt_satd = r_l0.satd
        .min(r_l1.satd)
        .min(r_bi.satd)
        .max(1);
    let direct_satd_ratio_too_high = (r_skip_or_direct.satd as u64) * 2
        > (best_alt_satd as u64) * 3
        && std::env::var_os("PHASM_B_NO_DIRECT_SATD_RATIO").is_none();

    let direct_cost_validated = if (at_boundary
        && std::env::var_os("PHASM_B_NO_BOUNDARY_REFUSE").is_none())
        || (direct_mv_too_large
            && std::env::var_os("PHASM_B_NO_DIRECT_MAGCLAMP").is_none())
        || direct_satd_too_high
        || direct_satd_ratio_too_high
    {
        u64::MAX
    } else {
        direct_cost_adjusted
    };

    // §B-boundary-anchor v1 (#251) — also refuse Bi at boundary
    // MBs. Bi averages L0 + L1 predictions; at jacket-edge MBs,
    // L0 and L1 ME may find different MV directions (e.g., L0
    // tracks past-jacket position, L1 tracks future-wall area
    // because kid moved away). Averaging the two gives a grey
    // blend (jacket ~50 luma + wall ~230 luma → ~140 luma) —
    // visually wrong-color rectangular blocks in the user-visible
    // V14/V20/V23 demos. Single-list (L0 or L1) avoids the
    // averaging artifact.
    let bi_cost_validated = if at_boundary
        && std::env::var_os("PHASM_B_NO_BOUNDARY_REFUSE").is_none()
    {
        u64::MAX
    } else {
        r_bi.cost
    };

    // Lowest-cost decision. Skip is NOT in the candidate set —
    // see §6E-D.5(e) comment above; Skip is only emitted by the
    // CBP-zero pre-check at the top of this function.
    let mut best = (direct_cost_validated, 1_u8); // 1 = Direct
    if r_l0.cost              < best.0 { best = (r_l0.cost, 2); }
    if r_l1.cost              < best.0 { best = (r_l1.cost, 3); }
    if bi_passes_threshold && bi_cost_validated < best.0 { best = (bi_cost_validated, 4); }

    // §intra-in-B (#319) Phase 2-diagnostic — count B-MBs that would
    // be candidates for intra-fallback under the v1.6 plan. A B-MB is
    // a fallback candidate when ALL inter SATDs exceed the threshold,
    // meaning ME couldn't find any reference patch that matches the
    // source well. The dark-debris artifact in the user's screenshot
    // is exactly this case: small high-contrast feature on uniform
    // background, ME finds wrong-but-cheap match in the surrounding
    // texture, all candidates have high residual energy.
    //
    // This counter is incremented but no decision is made yet —
    // Phase 3 (encoder emit path) wires the actual intra emission.
    // Read via [`drain_b_intra_fallback_count`].
    //
    // Threshold 25000 is calibrated from the v1.5 artifact harness:
    // the worst MBs in the carplane Class B set had min(satd) in the
    // 20k–30k range; setting the gate at 25k catches genuine ME
    // failures without firing on normal high-residual content.
    // Env-tunable for Phase 5-6 calibration: PHASM_B_INTRA_FALLBACK_SATD_MIN.
    // Higher value → fewer intra emissions; lower → more. v1.6.0 default
    // 25000 lands at 0.53% of B-MBs on carplane, matching the converter-pipeline centroid
    // 0.1-2% range. Phase 5 visual A/B may revise this.
    let intra_fallback_threshold: u32 = std::env::var("PHASM_B_INTRA_FALLBACK_SATD_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25_000);
    let best_inter_satd = r_skip_or_direct.satd
        .min(r_l0.satd).min(r_l1.satd).min(r_bi.satd);
    let intra_fallback_eligible = best_inter_satd > intra_fallback_threshold;
    if intra_fallback_eligible {
        B_INTRA_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // §intra-in-B (#319) Phase 2-decision — emit IntraI16x16 when
    // PHASM_B_INTRA_FALLBACK=1 AND the threshold fires. The encoder
    // emit path is currently a stub (Phase 3 pending), so the env var
    // produces a clean error message rather than corrupted output.
    // Intentional: the decision-side wiring exists so Phase 3 can
    // light up the actual emit without further plumbing changes.
    if intra_fallback_eligible
        && std::env::var_os("PHASM_B_INTRA_FALLBACK").is_some()
    {
        return BMbDecision::IntraI16x16 {
            i16x16_mode: 2,        // DC (works without neighbour pixels)
            chroma_pred_mode: 0,   // DC chroma
        };
    }

    // §INVEST 2026-05-07 — dump per-MB decision when PHASM_INVEST_DUMP=1.
    // Only fires for source MBs that look like pure wall (uniform high
    // luma) AND the chosen prediction is non-trivial. Prints the
    // candidate's MV and the source/prediction luma summary so we can
    // see exactly what's happening on the artifact MBs.
    if std::env::var_os("PHASM_INVEST_DUMP").is_some() {
        let mut src_min = 255u8;
        let mut src_max = 0u8;
        let mut src_sum: u32 = 0;
        for row in src_y {
            for &v in row {
                src_sum += v as u32;
                if v < src_min { src_min = v; }
                if v > src_max { src_max = v; }
            }
        }
        let src_mean = (src_sum / 256) as u8;
        // Allow PHASM_INVEST_DUMP_ALL=1 to log all MBs, not just wall.
        let dump_all = std::env::var_os("PHASM_INVEST_DUMP_ALL").is_some();
        if dump_all || (src_mean >= 220 && src_max - src_min <= 10) {
            let mode_name = match best.1 {
                1 => "Direct",
                2 => "L0",
                3 => "L1",
                4 => "Bi",
                _ => "?",
            };
            let (mv_l0, mv_l1) = match best.1 {
                1 => (direct.mv_l0, direct.mv_l1),
                2 => (me_mvs.0, MotionVector::ZERO),
                3 => (MotionVector::ZERO, me_mvs.1),
                4 => (me_mvs.0, me_mvs.1),
                _ => (MotionVector::ZERO, MotionVector::ZERO),
            };
            eprintln!(
                "WALL-MB ({mb_x},{mb_y}) src=[{src_min},{src_max}]={src_mean} mode={mode_name} \
                 mvL0=({},{}) mvL1=({},{}) costs: skip_or_direct={} (adj={}) l0={} l1={} bi={}",
                mv_l0.mv_x, mv_l0.mv_y, mv_l1.mv_x, mv_l1.mv_y,
                r_skip_or_direct.cost, direct_cost_adjusted, r_l0.cost, r_l1.cost, r_bi.cost,
            );
        }
    }

    // §B-instrument v1 (#251) — record per-MB decision data when
    // PHASM_B_INSTRUMENT=1. Test reads via `drain_b_mb_records()`
    // after encode + correlates with worst-deviation MBs.
    {
        let (chosen_mv_l0, chosen_mv_l1) = match best.1 {
            1 => (direct.mv_l0, direct.mv_l1),
            2 => (me_mvs.0, MotionVector::ZERO),
            3 => (MotionVector::ZERO, me_mvs.1),
            4 => (me_mvs.0, me_mvs.1),
            _ => (MotionVector::ZERO, MotionVector::ZERO),
        };
        let mut src_min = 255u8;
        let mut src_max = 0u8;
        let mut src_sum: u32 = 0;
        for row in src_y {
            for &v in row {
                src_sum += v as u32;
                if v < src_min { src_min = v; }
                if v > src_max { src_max = v; }
            }
        }
        let frame_idx = B_INSTRUMENT_FRAME_IDX
            .load(std::sync::atomic::Ordering::Relaxed);
        push_b_mb_record(BMbRecord {
            frame_idx,
            mb_x: mb_x as u16,
            mb_y: mb_y as u16,
            mode_id: best.1,
            mv_l0_x: chosen_mv_l0.mv_x,
            mv_l0_y: chosen_mv_l0.mv_y,
            mv_l1_x: chosen_mv_l1.mv_x,
            mv_l1_y: chosen_mv_l1.mv_y,
            direct_mv_l0_x: direct.mv_l0.mv_x,
            direct_mv_l0_y: direct.mv_l0.mv_y,
            direct_mv_l1_x: direct.mv_l1.mv_x,
            direct_mv_l1_y: direct.mv_l1.mv_y,
            satd_skip_or_direct: r_skip_or_direct.satd as u32,
            satd_l0: r_l0.satd as u32,
            satd_l1: r_l1.satd as u32,
            satd_bi: r_bi.satd as u32,
            at_boundary,
            src_mean: (src_sum / 256) as u8,
            src_min,
            src_max,
        });
    }

    match best.1 {
        1 => BMbDecision::Direct16x16,
        2 => BMbDecision::L0_16x16 { mv: me_mvs.0, ref_idx_l0: 0 },
        3 => BMbDecision::L1_16x16 { mv: me_mvs.1 },
        4 => BMbDecision::Bi_16x16 {
            mv_l0: me_mvs.0,
            mv_l1: me_mvs.1,
            ref_idx_l0: 0,
        },
        _ => unreachable!(),
    }
}

/// Hash-based bucket [0, 100) for §6E-A6.1 mode-mix selection.
/// Same hash family as `mb_skip_or_direct_decision` for stability,
/// just spread across 100 buckets instead of 2. Determinism: pure
/// function of (frame_num, mb_addr) so encoder + walker (via the
/// same call) agree on the chosen mode.
fn mb_decision_bucket(frame_num: u32, mb_addr: u32) -> u32 {
    let mut x = (frame_num as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = x.wrapping_add(mb_addr as u64);
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    (x % 100) as u32
}

/// §B-direct-fix.v3 Path 4.1 — motion-boundary detector for B-MB
/// SkipOrDirect cost penalty. Returns a Q.8 factor:
/// - homogeneous neighbours (no boundary) → 256 (= 1.0×, no change)
/// - boundary detected (bimodal MVs) → up to 768 (= 3.0× penalty)
///
/// Samples L0 MVs at top-left / top / top-right / left of the
/// current MB (these are the available decoded neighbours under
/// raster scan). Computes max-minus-min span on each MV axis. Span
/// in quarter-pel units:
///   ≤ 4 (1 px or less)   → 256       (homogeneous; Direct trustworthy)
///   ≥ 32 (8 px or more)  → 768       (sharp boundary; trust L0/L1/Bi)
///   in between           → linear ramp
///
/// Why these thresholds: ME noise in homogeneous regions typically
/// stays within ±1 px (≤ 4 quarter-pel) so the lower bound avoids
/// false-positives on flat content. 8 px is a typical motion delta
/// across an object silhouette in 30 fps handheld video, captured
/// over consecutive frames in IBPBP M=2 — past the ramp's saturation
/// point Direct is reliably wrong on these MBs.
///
/// Single-list considered: even in B-frames most explicit-MV
/// activity is on L0 (B-frame's L0 = previous P/I; usually the
/// dominant motion direction). Sampling only L0 keeps the detector
/// cheap; if the L0 grid for a neighbour is empty (intra MB or
/// pre-decode boundary), the entry is skipped.
fn compute_motion_boundary_factor_q8(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
) -> u32 {
    let bx = (mb_x as isize) * 4;
    let by = (mb_y as isize) * 4;
    let neighbour_offsets: [(isize, isize); 4] = [
        (bx - 1, by - 1), // top-left
        (bx, by - 1),     // top
        (bx + 4, by - 1), // top-right
        (bx - 1, by),     // left
    ];
    let mut neighbours: [(i32, i32); 4] = [(0, 0); 4];
    let mut count = 0usize;
    for (nx, ny) in neighbour_offsets {
        if let Some((mv, _)) = grid.get_l0(nx, ny) {
            neighbours[count] = (mv.mv_x as i32, mv.mv_y as i32);
            count += 1;
        }
    }
    if count < 2 {
        return 256;
    }
    let active = &neighbours[..count];
    let (mut min_x, mut max_x) = (active[0].0, active[0].0);
    let (mut min_y, mut max_y) = (active[0].1, active[0].1);
    for &(x, y) in &active[1..] {
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }
    let span = ((max_x - min_x).max(max_y - min_y)).max(0) as u32;
    if span <= 4 {
        256
    } else if span >= 32 {
        768
    } else {
        256 + ((span - 4) * 512) / 28
    }
}

/// §B-boundary-anchor v1 (#251) — binary motion-boundary
/// detector. Returns true when neighbour L0 MVs span more than
/// 1 pixel (4 qpel) on any axis — indicating bimodal motion in
/// the MB's neighbourhood. Used by:
///   - encoder ME path: at boundary MBs, override `me_anchor_l0/l1`
///     to (0,0) so ME's rate-cost preference doesn't pull toward
///     the (likely wrong-direction) spatial-median predictor.
///   - mb_decision_b_rdo: refuse Direct/Skip at boundary MBs so
///     the bitstream never emits a wall-direction implicit MV.
///
/// Threshold = 4 qpel = 1 pixel: ME noise is typically <1 px on
/// camera-shake content, real motion-boundary spans are 4 px+
/// when content (jacket) moves against static (wall).
pub fn is_motion_boundary(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> bool {
    compute_motion_boundary_factor_q8(grid, mb_x, mb_y) > 256
}

/// Predicted L0 MV for a B-MB's 16x16 partition — used as the
/// chosen MV for §6E-A6.1 zero-MVD mode emission AND as the
/// median bit-cost anchor for §6E-A6.1q.b real ME (`predicted_mv`
/// arg to `MotionEstimator::search_block`).
fn predict_b_partition_mv_l0(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> MotionVector {
    super::partition_state::predict_mv_for_mb_partition(
        grid, mb_x * 4, mb_y * 4,
        /* part_w_4x4 */ 4, /* part_h_4x4 */ 4,
        /* mb_part_idx */ 0, /* current_ref_idx */ 0,
    )
}

/// §6E-A6.1q.b (#151) — public mirror of [`predict_b_partition_mv_l0`]
/// for the encoder's ME pipeline.
pub fn predict_b_partition_mv_l0_pub(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> MotionVector {
    predict_b_partition_mv_l0(grid, mb_x, mb_y)
}

/// Mirror of [`predict_b_partition_mv_l0`] for List 1.
fn predict_b_partition_mv_l1(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> MotionVector {
    super::b_direct_predictor::predict_mv_for_partition_l1_pub(
        grid, mb_x * 4, mb_y * 4, /* current_ref_idx */ 0,
    )
}

/// §6E-A6.1q.b (#151) — public mirror of [`predict_b_partition_mv_l1`]
/// for the encoder's ME pipeline.
pub fn predict_b_partition_mv_l1_pub(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> MotionVector {
    predict_b_partition_mv_l1(grid, mb_x, mb_y)
}

/// §6E-A6.1 — read `PHASM_B_FORCE_MODE` env var and translate to a
/// `BMbDecision`. Returns `None` if the var is unset / unrecognized
/// (production path falls through to the §6E-A4(c)-lite hash mix).
pub fn forced_b_mode_from_env() -> Option<BMbDecision> {
    let var = std::env::var("PHASM_B_FORCE_MODE").ok()?;
    let zero = MotionVector { mv_x: 0, mv_y: 0 };
    // §B-RDO.debug.6 — optional non-zero MV override via PHASM_B_FORCE_MV
    // (=qpel_x,qpel_y in 1/4-luma-pel units). Used to probe sub-pel MC.
    let forced_mv = std::env::var("PHASM_B_FORCE_MV")
        .ok()
        .and_then(|s| {
            let mut parts = s.splitn(2, ',');
            let x: i16 = parts.next()?.trim().parse().ok()?;
            let y: i16 = parts.next()?.trim().parse().ok()?;
            Some(MotionVector { mv_x: x, mv_y: y })
        })
        .unwrap_or(zero);
    // §B-RDO.debug.7 — optional asymmetric L1 MV for bi_16x16 to
    // probe asymmetric bipred MC (mv_l0 ≠ mv_l1).
    let forced_mv_l1 = std::env::var("PHASM_B_FORCE_MV_L1")
        .ok()
        .and_then(|s| {
            let mut parts = s.splitn(2, ',');
            let x: i16 = parts.next()?.trim().parse().ok()?;
            let y: i16 = parts.next()?.trim().parse().ok()?;
            Some(MotionVector { mv_x: x, mv_y: y })
        })
        .unwrap_or(forced_mv);
    match var.to_ascii_lowercase().as_str() {
        "skip" => Some(BMbDecision::Skip),
        "direct" => Some(BMbDecision::Direct16x16),
        "l0_16x16" => Some(BMbDecision::L0_16x16 { mv: forced_mv, ref_idx_l0: 0 }),
        "l1_16x16" => Some(BMbDecision::L1_16x16 { mv: forced_mv }),
        "bi_16x16" => Some(BMbDecision::Bi_16x16 {
            mv_l0: forced_mv,
            mv_l1: forced_mv_l1,
            ref_idx_l0: 0,
        }),
        // §6E-A6.2 — partitioned variants. `partitioned_<mb_type>`
        // forces a specific 4..21 mb_type for round-trip testing.
        // §B-Partitioned-Residual (#206) — also honors PHASM_B_FORCE_MV
        // / PHASM_B_FORCE_MV_L1 for non-zero MV residual round-trips.
        s if s.starts_with("partitioned_") => {
            let mb_type: u8 = s["partitioned_".len()..].parse().ok()?;
            forced_partitioned_decision(mb_type, forced_mv, forced_mv_l1)
        }
        // §6E-A6.3 — uniform B_8x8 variants. `b_8x8_uniform_<sub>`
        // forces all 4 sub-MBs to the same sub_mb_type 0..=3:
        //   b_8x8_uniform_direct → all four = B_Direct_8x8 (0)
        //   b_8x8_uniform_l0     → all four = B_L0_8x8 (1)
        //   b_8x8_uniform_l1     → all four = B_L1_8x8 (2)
        //   b_8x8_uniform_bi     → all four = B_Bi_8x8 (3)
        // `b_8x8_mixed` exercises a different sub_mb_type per sub-MB
        // (Direct, L0, L1, Bi in raster order) — round-trip stress for
        // the 5-bin Bi tail. §B-Partitioned-Residual (#206) also
        // honors PHASM_B_FORCE_MV / PHASM_B_FORCE_MV_L1.
        "b_8x8_uniform_direct" => Some(forced_b_8x8_uniform(0, forced_mv, forced_mv_l1)),
        "b_8x8_uniform_l0" => Some(forced_b_8x8_uniform(1, forced_mv, forced_mv_l1)),
        "b_8x8_uniform_l1" => Some(forced_b_8x8_uniform(2, forced_mv, forced_mv_l1)),
        "b_8x8_uniform_bi" => Some(forced_b_8x8_uniform(3, forced_mv, forced_mv_l1)),
        "b_8x8_mixed" => Some(forced_b_8x8_mixed(forced_mv, forced_mv_l1)),
        _ => None,
    }
}

/// §6E-A6.3 — build a uniform `B_8x8` decision where all four
/// sub-MBs share the same `sub_mb_type` (0..=3). `mv_l0` is used for
/// L0/Bi sub-MBs; `mv_l1` is used for L1/Bi sub-MBs.
fn forced_b_8x8_uniform(sub: u8, mv_l0: MotionVector, mv_l1: MotionVector) -> BMbDecision {
    let part = b_8x8_part_for_subtype(sub, mv_l0, mv_l1);
    BMbDecision::B8x8 {
        sub_mb_types: [sub; 4],
        parts: [part; 4],
    }
}

/// §6E-A6.3 — `b_8x8_mixed` test mode: one of each sub_mb_type
/// (Direct, L0, L1, Bi) across the four sub-MBs. Stresses the 5-bin
/// Bi tail + the L0/L1 3-bin path in the same MB.
fn forced_b_8x8_mixed(mv_l0: MotionVector, mv_l1: MotionVector) -> BMbDecision {
    let sub_mb_types = [0u8, 1, 2, 3];
    let parts = [
        b_8x8_part_for_subtype(0, mv_l0, mv_l1),
        b_8x8_part_for_subtype(1, mv_l0, mv_l1),
        b_8x8_part_for_subtype(2, mv_l0, mv_l1),
        b_8x8_part_for_subtype(3, mv_l0, mv_l1),
    ];
    BMbDecision::B8x8 { sub_mb_types, parts }
}

/// §6E-A6.3 — translate a `sub_mb_type` (0..=3) + per-list MVs into
/// the matching [`BPartitionMv`]:
///
/// | sub_mb_type | mv_l0 | mv_l1 |
/// |---|---|---|
/// | 0 (Direct) | None | None |
/// | 1 (L0)     | Some(mv_l0) | None |
/// | 2 (L1)     | None | Some(mv_l1) |
/// | 3 (Bi)     | Some(mv_l0) | Some(mv_l1) |
fn b_8x8_part_for_subtype(
    sub: u8,
    mv_l0: MotionVector,
    mv_l1: MotionVector,
) -> super::b_partitioned::BPartitionMv {
    use super::b_partitioned::BPartitionMv;
    match sub {
        0 => BPartitionMv { mv_l0: None, mv_l1: None, ref_idx_l0: 0 },
        1 => BPartitionMv { mv_l0: Some(mv_l0), mv_l1: None, ref_idx_l0: 0 },
        2 => BPartitionMv { mv_l0: None, mv_l1: Some(mv_l1), ref_idx_l0: 0 },
        3 => BPartitionMv { mv_l0: Some(mv_l0), mv_l1: Some(mv_l1), ref_idx_l0: 0 },
        _ => {
            debug_assert!(false, "B_8x8 sub_mb_type {sub} out of §6E-A6.3 scope");
            BPartitionMv { mv_l0: None, mv_l1: None, ref_idx_l0: 0 }
        }
    }
}

/// §6E-A6.2 — build a `Partitioned` decision for a given mb_type
/// (4..=21). `mv_l0` is used for L0/Bi partitions; `mv_l1` for L1/Bi.
/// Used by the env-var override + tests.
fn forced_partitioned_decision(
    mb_type: u8,
    mv_l0: MotionVector,
    mv_l1: MotionVector,
) -> Option<BMbDecision> {
    use super::b_partitioned::{partitioned_b_meta, BListUse, BPartitionMv};
    let meta = partitioned_b_meta(mb_type as u32)?;
    let mv_for = |usage: BListUse| -> BPartitionMv {
        let (mv_l0_o, mv_l1_o) = match usage {
            BListUse::L0 => (Some(mv_l0), None),
            BListUse::L1 => (None, Some(mv_l1)),
            BListUse::Bi => (Some(mv_l0), Some(mv_l1)),
        };
        BPartitionMv { mv_l0: mv_l0_o, mv_l1: mv_l1_o, ref_idx_l0: 0 }
    };
    Some(BMbDecision::Partitioned {
        mb_type,
        parts: [mv_for(meta.part0), mv_for(meta.part1)],
    })
}

/// v1.4 Phase 4.3 (#314, Path B) — B-side post-pass that re-searches
/// the chosen B-MB's L0 MV against `ref_1` and upgrades `ref_idx_l0`
/// when ref_1 wins by more than λ. Mirrors P-side
/// `refine_p_choice_multi_ref` but for B-frames; only L0_16x16 +
/// Bi_16x16 are upgraded in v1.4-cut1 (Partitioned + B8x8 deferred,
/// plus Skip/Direct/L1 variants don't carry an L0 MV to upgrade).
///
/// Per-MB uniform: when an upgrade fires the whole BMbDecision gets
/// `ref_idx_l0 = 1`. For Bi_16x16 only the L0 MV is re-searched; L1
/// MV stays unchanged (Q1 lock — L1 is single-ref).
///
/// Cost comparison: same as P-side. SATD against ref_0 vs SATD
/// against ref_1, plus λ for the +1 ref_idx bin delta. ref_0
/// baseline is computed by re-running ME from the existing MV
/// (converges fast since existing MV is already optimal there).
///
/// Returns true iff the MB was upgraded.
pub fn refine_b_choice_multi_ref(
    _src_y: &[[u8; 16]; 16],
    _me: &mut super::motion_estimation::MotionEstimator,
    _mb_x: usize,
    _mb_y: usize,
    _ref_0: &super::reference_buffer::ReconFrame,
    _ref_1: &super::reference_buffer::ReconFrame,
    _choice: &mut BMbDecision,
) -> bool {
    // v1.4 Phase 4.5 (#316) STRUCTURAL DISABLE — B-slice refine
    // permanently off until RPLM (Reference Picture List Modification)
    // is implemented in v1.5+.
    //
    // Spec § 8.2.4.2.3 says B-slice RefPicList0 = (past sorted POC
    // descending) ++ (future sorted POC ascending). For IBPBP M=2:
    // L0[0] = past_anchor, L0[1] = last_ref (= L1's picture). Phasm's
    // `b_l0_ref_1 = pre_past_anchor` (2nd-closest past) does NOT
    // match the spec's L0[1] without RPLM commands in the slice
    // header. Since v1.4 doesn't emit RPLM, picking ref_idx_l0=1
    // means the encoder reconstructs against pre_past_anchor while
    // a spec-compliant decoder reconstructs against last_ref →
    // catastrophic block-level visual divergence (cf. carplane demo
    // 2026-05-10, -8.7 dB Y mean and -14.4 dB at frame 26).
    //
    // P-slice refine remains active because P-slice L0[1] = 2nd-
    // closest past = `past_anchor` field, which matches phasm's
    // `ref_1` mapping (P_8x8 deferred to v1.4-cut2).
    //
    // To re-enable B-side refine: emit RPLM in
    // `build_b_slice_rbsp_cabac` to remap L0[1] from last_ref to
    // pre_past_anchor, mirror the modification in walker
    // `parse_b_slice_header`, then revert this stub. Tracking on
    // task #318.
    false
    /*
    use super::motion_estimation::me_lambda_pub;

    // Path B v1.4-cut1: only 16x16 family L0/Bi gets refine.
    let l0_mv = match choice {
        BMbDecision::L0_16x16 { mv, .. } => *mv,
        BMbDecision::Bi_16x16 { mv_l0, .. } => *mv_l0,
        // Skip / Direct / L1 / Partitioned / B8x8: no upgrade.
        _ => return false,
    };

    let lambda = me_lambda_pub();
    let mb_px_x = (mb_x * 16) as u32;
    let mb_px_y = (mb_y * 16) as u32;
    let src_flat = src_y.as_flattened();

    let r0 = me.search_block(
        src_flat, 16, ref_0, mb_px_x, mb_px_y, 16, 16, l0_mv,
    );
    let r1 = me.search_block(
        src_flat, 16, ref_1, mb_px_x, mb_px_y, 16, 16, l0_mv,
    );

    // Per-partition ref_idx_bits delta = 1 bin for 16x16. n_partitions=1.
    let penalty = lambda;

    if r1.cost.saturating_add(penalty) < r0.cost {
        match choice {
            BMbDecision::L0_16x16 { mv, ref_idx_l0 } => {
                *mv = r1.mv;
                *ref_idx_l0 = 1;
            }
            BMbDecision::Bi_16x16 { mv_l0, ref_idx_l0, .. } => {
                *mv_l0 = r1.mv;
                *ref_idx_l0 = 1;
                // L1 MV unchanged — stays single-ref per Q1 lock.
            }
            _ => unreachable!("filtered above"),
        }
        return true;
    }

    false
    }
    */
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_distribution_skip_direct_only() {
        // §B-direct-fix: when RDO is off, the no-RDO bucket fallback
        // emits Skip + Direct ONLY (~50/50). The previous 50/35/6/5/4
        // mix introduced bucket-derived L0/L1/Bi MBs whose MVs
        // polluted the spatial-direct grid for downstream Skip/Direct
        // neighbours → visible MB-level scrambling on real 1080p video
        // (root cause of the 2026-05-03 visual bug).
        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
        let grid = EncoderMvGrid::new(2, 2);
        let mut counts = [0u32; 5];
        let n = 10_000u32;
        for mb_addr in 0..n {
            let d = mb_decision_b(&grid, 0, 0, 0, mb_addr);
            let idx = match d {
                BMbDecision::Skip => 0,
                BMbDecision::Direct16x16 => 1,
                BMbDecision::L0_16x16 { .. } => 2,
                BMbDecision::L1_16x16 { .. } => 3,
                BMbDecision::Bi_16x16 { .. } => 4,
                _ => panic!("unexpected variant from no-RDO fallback"),
            };
            counts[idx] += 1;
        }
        let pct = |c: u32| (c as f32 / n as f32) * 100.0;
        let (skip, direct, l0, l1, bi) =
            (pct(counts[0]), pct(counts[1]), pct(counts[2]), pct(counts[3]), pct(counts[4]));
        eprintln!(
            "no-RDO fallback mix: skip={skip:.1}% direct={direct:.1}% \
             L0={l0:.1}% L1={l1:.1}% Bi={bi:.1}%"
        );
        // Skip 50% / Direct 50%, no L0/L1/Bi from fallback.
        assert!((skip - 50.0).abs() < 3.0, "skip {skip:.1}%");
        assert!((direct - 50.0).abs() < 3.0, "direct {direct:.1}%");
        assert_eq!(counts[2], 0, "L0 must be 0 (RDO-only); got {l0:.1}%");
        assert_eq!(counts[3], 0, "L1 must be 0 (RDO-only); got {l1:.1}%");
        assert_eq!(counts[4], 0, "Bi must be 0 (RDO-only); got {bi:.1}%");
    }

    #[test]
    fn deterministic_output_for_same_input() {
        // Critical for round-trip correctness: encoder + walker
        // both call mb_decision_b at the same (frame_num, mb_addr)
        // and must agree on the result.
        let grid = EncoderMvGrid::new(2, 2);
        for mb_addr in 0..256 {
            let a = mb_decision_b(&grid, 0, 0, 7, mb_addr);
            let b = mb_decision_b(&grid, 0, 0, 7, mb_addr);
            // BMbDecision doesn't impl PartialEq (some variants carry
            // MV data), but the discriminant must match.
            let disc = |d: &BMbDecision| std::mem::discriminant(d);
            assert_eq!(disc(&a), disc(&b),
                "non-deterministic mb_decision at mb={mb_addr}");
        }
    }

    // -- §6E-D.5 RDO mode-decision tests -------------------------------------

    fn make_recon_b(width: u32, height: u32, y_fill: u8) -> super::super::reference_buffer::ReconFrame {
        use super::super::reconstruction::ReconBuffer;
        let mut buf = ReconBuffer::new(width, height).unwrap();
        for v in buf.y.iter_mut() { *v = y_fill; }
        for v in buf.cb.iter_mut() { *v = 128; }
        for v in buf.cr.iter_mut() { *v = 128; }
        super::super::reference_buffer::ReconFrame::snapshot(&buf)
    }

    #[test]
    fn rdo_picks_skip_when_l0_matches_exactly() {
        // src = L0 ref = 100 → all candidates produce SATD=0; Skip
        // wire form has the lowest rate (1 bin) → wins.
        let mut src = [[0u8; 16]; 16];
        for row in &mut src { for px in row { *px = 100; } }
        let l0 = make_recon_b(64, 64, 100);
        let l1 = make_recon_b(64, 64, 100);
        let grid = EncoderMvGrid::new(4, 4);
        let zero = MotionVector { mv_x: 0, mv_y: 0 };
        let d = mb_decision_b_rdo(&grid, 0, 0, &src, &l0, &l1, 30, (zero, zero));
        assert!(matches!(d, BMbDecision::Skip),
            "Skip should win when prediction is exact, got {:?}", d);
    }

    #[test]
    fn rdo_picks_l0_when_l0_matches_l1_doesnt() {
        // src = L0 = 100, L1 = 50 → L0_16x16 with zero MV gets SATD=0,
        // L1_16x16 + Bi_16x16 get non-zero SATD. Skip uses spatial-direct
        // (zero MV from grid) which reads BIPRED (L0+L1) → mid value 75.
        // Source = 100 → bipred SATD ≠ 0. So Skip/Direct loses.
        let mut src = [[0u8; 16]; 16];
        for row in &mut src { for px in row { *px = 100; } }
        let l0 = make_recon_b(64, 64, 100);
        let l1 = make_recon_b(64, 64, 50);
        let grid = EncoderMvGrid::new(4, 4);
        let zero = MotionVector { mv_x: 0, mv_y: 0 };
        let d = mb_decision_b_rdo(&grid, 0, 0, &src, &l0, &l1, 30, (zero, zero));
        assert!(matches!(d, BMbDecision::L0_16x16 { .. }),
            "L0_16x16 should win when only L0 matches, got {:?}", d);
    }

    #[test]
    fn rdo_picks_l1_when_l1_matches_l0_doesnt() {
        let mut src = [[0u8; 16]; 16];
        for row in &mut src { for px in row { *px = 200; } }
        let l0 = make_recon_b(64, 64, 50);
        let l1 = make_recon_b(64, 64, 200);
        let grid = EncoderMvGrid::new(4, 4);
        let zero = MotionVector { mv_x: 0, mv_y: 0 };
        let d = mb_decision_b_rdo(&grid, 0, 0, &src, &l0, &l1, 30, (zero, zero));
        assert!(matches!(d, BMbDecision::L1_16x16 { .. }),
            "L1_16x16 should win when only L1 matches, got {:?}", d);
    }

    #[test]
    fn rdo_deterministic() {
        // Same inputs → same decision (required for stego Pass1/Pass3).
        let mut src = [[0u8; 16]; 16];
        for (y, row) in src.iter_mut().enumerate() {
            for (x, px) in row.iter_mut().enumerate() {
                *px = ((x * 7 + y * 11) & 0xFF) as u8;
            }
        }
        let l0 = make_recon_b(64, 64, 80);
        let l1 = make_recon_b(64, 64, 90);
        let grid = EncoderMvGrid::new(4, 4);
        let zero = MotionVector { mv_x: 0, mv_y: 0 };
        let d_a = mb_decision_b_rdo(&grid, 0, 0, &src, &l0, &l1, 30, (zero, zero));
        let d_b = mb_decision_b_rdo(&grid, 0, 0, &src, &l0, &l1, 30, (zero, zero));
        assert_eq!(std::mem::discriminant(&d_a), std::mem::discriminant(&d_b));
    }

    #[test]
    fn b_rdo_enabled_reads_env() {
        // Default off.
        unsafe { std::env::remove_var("PHASM_B_RDO"); }
        assert!(!b_rdo_enabled());
        unsafe { std::env::set_var("PHASM_B_RDO", "1"); }
        assert!(b_rdo_enabled());
        unsafe { std::env::set_var("PHASM_B_RDO", "0"); }
        assert!(!b_rdo_enabled());
        unsafe { std::env::remove_var("PHASM_B_RDO"); }
    }
}
