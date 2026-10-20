// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// ── Visual damage as a function of payload FILL (OH264 Ghost video) ──────
//
// WHY THIS EXISTS, and why it is not `openh264_visual_psnr_regression.rs`.
//
// That gate embeds a 25-byte message (`"C.2 visual gate — iPhone 7"`) and
// asserts `PSNR_FLOOR_DB = 24.0`. Two independent blind spots:
//
//   1. **Payload size.** 25 bytes is ~3 orders of magnitude below what the
//      app's capacity HUD offers the user on the same clip. Stego damage
//      scales with how many cover symbols get flipped, so a 25-byte probe
//      measures almost nothing about a full video.
//
//   2. **Bundled transcode loss.** It compares the *source* YUV against the
//      ffmpeg-decoded *stego* stream, so the number contains both the plain
//      OH264 transcode loss and the stego loss. Hand measurement on an 18 s
//      1080p clip (2026-10-20, `docs/notes/2026-10-20-video-encode-ux-defects.md`
//      "Round 2") split them: transcode alone was 42.1 dB (essentially
//      transparent) while stego cost 13 dB. Bundling hides exactly the term
//      we care about, and it hides it *asymmetrically* — a quieter transcode
//      makes the bundled number look worse, a noisier one makes stego damage
//      look better.
//
// This file is an INSTRUMENT first and a gate second. Per fixture it:
//
//   1. Probes the offered capacity with the production `StreamingProbeSession`
//      → `finish_with_per_gop()` → `achievable_max_message_bytes` — the exact
//      number the mobile HUD shows. No local capacity formula; if the probe
//      and the encoder disagree, that disagreement is a finding, not something
//      to paper over.
//   2. Encodes a CLEAN REFERENCE: the same OH264 encoder at the same QP over
//      the same pixels with **no stego handlers registered at all**
//      (`oh264_stream::encode_clean`).
//
//      This started out as "a stego encode carrying a 1-character message",
//      which is what the brief asked for, and that turned out to be wrong:
//      `FRAME_OVERHEAD` frames 50 bytes before a single payload byte, so on a
//      clip whose entire capacity is a few hundred bytes the 1-char encode is
//      already an ~18% fill. Measured on the 480×256×6 smoke, it sat anywhere
//      between 24.8 dB and 34.3 dB from source depending only on the
//      passphrase. A reference that moves 10 dB with the passphrase is not a
//      reference. The 1-char encode is still made — it is a genuinely
//      interesting data point, the *floor* cost of the envelope — but it is
//      reported, not subtracted.
//   3. Encodes at each fill level of that capacity, with an *incompressible*
//      blob carried as a FILE ATTACHMENT (see `blob_file` below for why not
//      repeated text), and round-trips the payload back out so a flattering
//      number can't come from an encode that quietly carried less.
//   4. ffmpeg-decodes everything, and measures each stego encode against the
//      **decoded clean reference** — so the reported numbers are STEGO-ONLY.
//   5. Reports source-vs-clean once per fixture (the transcode-only term), so
//      the split stays visible and a drifting transcode can't be mistaken for
//      a stego regression.
//
// Metric kernels come from `common::visual_metrics` — the same code the
// existing gate uses. Do not fork them; a previous drift between two metric
// copies already cost a session.
//
// Run the fast smoke (default `cargo test`):
//   cargo test -p phasm-core --features h264-encoder \
//     --test oh264_quality_vs_fill -- --test-threads=1 --nocapture
//
// Run one production fixture (minutes of OH264 encoding — use --release):
//   cargo test --release -p phasm-core --features h264-encoder \
//     --test oh264_quality_vs_fill -- --ignored --test-threads=1 --nocapture \
//     fill_iphone7_1080p
//
// `--test-threads=1` is mandatory: the OH264 FFI is serialized behind
// `common::video_fixture::session_guard()`, and running fixtures concurrently
// just queues on that mutex while multiplying peak RSS.
//
// Knobs (all optional, all documented at their definitions):
//   PHASM_FILL_REPEATS=<n>      salt draws per fill level (default 1). Raise it
//                               to see the J2 jitter in the `spread` column —
//                               without it a table is ONE draw, not a curve.
//   PHASM_FILL_PASSPHRASE=<s>   re-seed the STC to resample which positions
//                               carry the payload.
//
// Every encode pins `PHASM_DETERMINISTIC_SEED` per (fixture, fill, draw), so a
// run is byte-reproducible — verified: two consecutive runs of the smoke give
// identical metrics to the last printed digit. Without that pin the random
// per-encode salt re-rolls the jitter and no threshold is defensible.

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;
use common::video_fixture::{
    ensure_yuv, ffmpeg_decode_annex_b, probe_aligned_dims, session_guard, VideoFixture,
};
use common::visual_metrics::{count_bad_pixels, psnr_y, ssim_y, worst_mb_psnr_y};

use phasm_core::stego::payload::{compressed_payload_size, FileEntry};

// ───────────────────────────── thresholds ────────────────────────────────
//
// SET WHERE THEY PASS TODAY, DELIBERATELY. This instrument lands green so the
// measurement and the tightening are two separate, reviewable steps. The
// numbers below are floors on *today's* behaviour, not statements about what
// is acceptable. Nothing here should be read as "28 dB is fine".
//
// ── What was measured, 2026-10-20, to pick them ──────────────────────────
//
// All of it on commit `c8b0ea18` (`content_costs.rs` md5 521905b9), K=1 salt
// draw, stego-only vs a NO-STEGO OH264 reference. Re-deriving these after a
// cost-model change is the point of the file — rerun with
// `--ignored --nocapture` and compare tables, don't trust the numbers below to
// still hold.
//
//   fixture              transcode-only   worst stego-only across the ladder
//   asia_bottle_1080p       39.0 dB        22.47 dB (@75%)  SSIM 0.956  wMB 13.6
//   carplane_1080p          42.1 dB        23.98 dB (@6%)   SSIM 0.953  wMB 17.5
//   dji_mini2_1080p         39.2 dB        29.89 dB (@25%)  SSIM 0.978  wMB 17.1
//   iphone7_1080p           43.5 dB        23.40 dB (@100%) SSIM 0.971  wMB 14.4
//   schoolfight (720p)      46.2 dB        27.56 dB (@50%)  SSIM 0.981  wMB 21.2
//   iphone7 smoke (480p)    39.0 dB        22.81 dB (@50%)  SSIM 0.895  wMB 14.6
//
// Then re-measured with several salt draws per fill level
// (`PHASM_FILL_REPEATS`), same commit, which is what sizes the margins:
//
//   smoke, K=8:        worst 20.70 dB  SSIM 0.804  wMB 12.65  spread 1.1-5.7 dB
//   asia_bottle, K=4:  worst 21.41 dB  SSIM 0.940  wMB 10.40  spread 2.9-9.2 dB
//   iphone7, K=4:      worst 23.78 dB  SSIM 0.966  wMB 13.15  spread 0.1-1.5 dB
//
// **The margins are sized by that spread, not by taste.** Salt jitter at a
// FIXED payload size reached 9.2 dB — larger than the whole fill-level effect
// on some fixtures. Each floor therefore sits ~5 dB (≈ half the worst measured
// spread) below the worst value ever observed, so a different salt draw cannot
// flip the gate red on its own. A tighter floor would be a flaky gate, not a
// stricter one.
//
// ── WHAT "VISUALLY INVISIBLE" WOULD BE ──────────────────────────────────
//
// Stego-only PSNR-Y ≥ ~45 dB, SSIM ≥ ~0.995, worst-MB ≥ ~35 dB at 100% fill —
// i.e. the stego term QUIETER than the transcode term (39-46 dB above), rather
// than 15-20 dB louder than it. The user's hand measurement called 38 dB
// visible and 28 dB "lots of glitches". So the distance from these constants to
// invisible is ~25 dB, not 1-2 dB of polish. See `VISUALLY_INVISIBLE_*`.

/// Stego-only mean Y-PSNR floor **at any fill level**.
///
/// Worst ever observed: 20.70 dB (smoke, K=8, 50% fill). Floor 5.7 dB under it.
const STEGO_ONLY_PSNR_MEAN_FLOOR_DB: f64 = 15.0;

/// Stego-only *per-frame minimum* Y-PSNR floor at any fill level.
///
/// Worst ever observed: 20.60 dB (smoke, K=8, 50% fill). Worst frames run
/// ~1-2 dB under the mean here (the damage is spread across frames rather than
/// concentrated), so this sits only slightly below the mean floor.
const STEGO_ONLY_PSNR_MIN_FLOOR_DB: f64 = 13.0;

/// Stego-only mean Y-SSIM floor at any fill level.
///
/// Worst ever observed: 0.8042 (smoke, K=8, 50% fill); 1080p fixtures stay
/// ≥0.94 even at 22 dB PSNR. That gap is itself a finding: **SSIM is a weak
/// detector for this damage class.** Whole-16×16-macroblock displacement
/// preserves local mean, variance and structure *within* the moved block, which
/// is most of what SSIM measures. Do not promote SSIM to the primary gate.
const STEGO_ONLY_SSIM_MEAN_FLOOR: f64 = 0.70;

/// Stego-only worst-16×16-MB Y-PSNR floor at any fill level. Catches localized
/// displaced-block damage that a whole-frame average washes out.
///
/// Worst ever observed: 10.40 dB (asia_bottle, K=4, 100% fill).
const STEGO_ONLY_WORST_MB_PSNR_FLOOR_DB: f64 = 5.0;

/// Per-pixel delta threshold for the "bad pixel" metric. Kept identical to
/// `openh264_visual_psnr_regression.rs` so the two gates' bad-pixel numbers are
/// comparable: at >80 the metric is bug-discriminating, at ≤50 normal
/// high-frequency content floods it.
const BAD_PIXEL_DELTA: u8 = 80;

/// Stego-only bad-pixel ceiling per frame at `BAD_PIXEL_DELTA`.
///
/// The 25-byte gate next door uses 100. A *full* payload legitimately blows
/// past that — worst observed here is 889 (iphone7, K=4, 100% fill) — which is
/// exactly what that gate cannot see. 20 000 is ~1% of a 1080p luma frame: far
/// above anything measured, but still a hard stop on a cascade blow-up that
/// corrupts whole frames.
const STEGO_ONLY_BAD_PIXELS_CEILING_PER_FRAME: u32 = 20_000;

/// Sanity floor on the TRANSCODE-ONLY term (source YUV vs the NO-STEGO
/// reference, mean Y-PSNR). The split this file exists to expose is only
/// meaningful while the transcode stays near-transparent. If this trips, the
/// reference itself moved and every stego-only number in the table is suspect.
///
/// Measured 39.0-46.2 dB across the corpus — and 39.0 dB on the 480×256 smoke
/// too, now that the reference is a true no-stego encode. (With the earlier
/// 1-char-message "baseline" the same smoke read 24.8-34.3 dB depending on the
/// passphrase, which is what exposed that baseline as unusable.) Floor 6 dB
/// under the worst observed.
const TRANSCODE_ONLY_PSNR_MEAN_FLOOR_DB: f64 = 33.0;

/// PER-FIXTURE floor overrides, where that fixture's own salt jitter has been
/// measured.
///
/// The global floors above take their margin from the WORST jitter seen
/// anywhere (9.2 dB on asia_bottle). That is the right call for a fixture whose
/// jitter is unknown, but applied everywhere it makes the gate blunt: iphone7's
/// measured spread is 0.1-1.5 dB and its worst observation 23.78 dB, so the
/// global 15.0 floor sits **8.8 dB** under it — a 10 dB regression on iphone7
/// would pass silently. A regression gate should be as tight as that fixture's
/// own noise allows, and no tighter.
///
/// Each entry is `(name, mean_floor, min_floor, worst_mb_floor)`, sized as
/// `worst observed − that fixture's worst measured spread`, rounded down.
/// Fixtures absent here fall back to the global constants, which is the honest
/// default: no jitter measurement, no tightening.
///
/// To add a fixture: run it with `PHASM_FILL_REPEATS=4` (or more), take the
/// worst value and the widest spread from the printed table, and subtract.
const PER_FIXTURE_FLOORS: &[(&str, f64, f64, f64)] = &[
    // worst 23.78 dB, spread 0.1-1.5 dB (K=4)  ->  23.78 - 1.5 = 22.2
    ("iphone7_1080p", 22.0, 20.0, 11.0),
    // worst 21.41 dB, spread 2.9-9.2 dB (K=4). Widest jitter in the corpus,
    // so this barely tightens on the global — correctly so.
    ("asia_bottle_1080p", 12.0, 10.0, 5.0),
    // worst 20.70 dB, spread 1.1-5.7 dB (K=8)  ->  20.70 - 5.7 = 15.0
    ("iphone7_fill_smoke", 15.0, 13.0, 7.0),
];

/// Floors for `name`: its own measured ones, else the global defaults.
fn floors_for(name: &str) -> (f64, f64, f64) {
    PER_FIXTURE_FLOORS
        .iter()
        .find(|(n, ..)| *n == name)
        .map(|&(_, mean, min, wmb)| (mean, min, wmb))
        .unwrap_or((
            STEGO_ONLY_PSNR_MEAN_FLOOR_DB,
            STEGO_ONLY_PSNR_MIN_FLOOR_DB,
            STEGO_ONLY_WORST_MB_PSNR_FLOOR_DB,
        ))
}

/// The targets, for the workstream that owns the cost model. Not asserted.
/// Documented as `const` so the tightening step is a one-line edit of which
/// constant the assertions read.
#[allow(dead_code)]
const VISUALLY_INVISIBLE_PSNR_MEAN_DB: f64 = 45.0;
#[allow(dead_code)]
const VISUALLY_INVISIBLE_SSIM_MEAN: f64 = 0.995;
#[allow(dead_code)]
const VISUALLY_INVISIBLE_WORST_MB_PSNR_DB: f64 = 35.0;

/// Fill levels, as a percentage of the probe's offered capacity.
const FILL_PERCENTS: &[u32] = &[6, 25, 50, 75, 100];

/// Default passphrase for every encode in this file. Fixed so runs are
/// comparable: the STC seeds derive from it, so changing it reshuffles which
/// cover symbols get flipped.
const DEFAULT_PASSPHRASE: &str = "quality-vs-fill";

/// Passphrase actually used, overridable with `PHASM_FILL_PASSPHRASE`.
///
/// This override is an instrument feature, not a convenience. The passphrase
/// seeds the STC, so re-running a fixture under a second passphrase resamples
/// *which* cover positions carry the message at the same payload size. That is
/// the ablation that separates two explanations for a non-monotonic
/// quality-vs-fill curve:
///
///   - **structure** — something about payload size changes the embedding
///     strategy (tier selection, domain mix, allocation); the shape then
///     repeats under a new passphrase.
///   - **variance** — damage is dominated by a handful of very expensive
///     flips (an MVD sign flip displaces a whole 16×16 macroblock), so at low
///     flip counts the outcome depends on which positions the STC happened to
///     pick; the shape then moves.
fn passphrase() -> String {
    std::env::var("PHASM_FILL_PASSPHRASE").unwrap_or_else(|_| DEFAULT_PASSPHRASE.to_string())
}

/// The 1-char data point's message. Not empty, because an empty payload is a
/// different code path; the point is "the smallest real message there is".
const BASELINE_MESSAGE: &str = "x";

/// Salt draws per fill level (`PHASM_FILL_REPEATS`, default 1).
///
/// **Why this knob is not optional reading.** `crypto::encrypt` draws a fresh
/// random salt + nonce per encode, so the ciphertext bits — and therefore the
/// STC's min-cost path through the cover — differ on every run. The capacity
/// work already named this axis: "J2 message/salt jitter … random whether the
/// min-cost embedding path is forced through an ∞-cost position"
/// (`docs/design/video/h264/capacity-spreading-plan.md` §"What it actually
/// is"), and documented that *feasibility* is non-monotonic in payload size
/// because of it. Visual damage inherits the same jitter, and at these payload
/// sizes (hundreds of bytes into a 1080p clip) a handful of expensive MVD flips
/// dominate the dB — so a SINGLE sample per fill level does not measure a
/// function of fill, it measures one draw.
///
/// Default 1 keeps the gate cheap. Raise it (`PHASM_FILL_REPEATS=5`) to see the
/// spread, which is what the `spread` column reports.
fn repeats() -> usize {
    std::env::var("PHASM_FILL_REPEATS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 32)
}

/// Deterministic salt/nonce seed for one (fixture, fill level, repeat) cell.
///
/// Pinning the salt is what makes this gate *reproducible*: without it the
/// random salt re-rolls the J2 jitter on every run and no threshold is
/// defensible. FNV-1a over the fixture name, mixed with the fill percent and
/// the repeat index, so each cell gets its own fixed draw and repeats sample
/// genuinely different ones.
fn salt_seed(fixture: &str, pct: u32, rep: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in fixture.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= (pct as u64) << 32;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    h ^= rep as u64;
    h.wrapping_mul(0x0000_0100_0000_01b3)
}

/// Pin `crypto::encrypt`'s salt + nonce for the next encode.
///
/// `PHASM_DETERMINISTIC_SEED` is phasm-core's own diagnostic hook (see
/// `stego::crypto::encrypt`), and `unsafe { set_var }` here follows the
/// established idiom in this test suite (`wire_format_cross_platform_decode`,
/// `av1_shadow_capacity_calibration`). Safe in practice because every caller
/// holds `session_guard()` and no encode is in flight at the call site.
fn pin_salt(seed: u64) {
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", seed.to_string()) };
}

// ───────────────────────────── payload ───────────────────────────────────

/// Deterministic, incompressible filler.
///
/// SplitMix64 keystream. Repeated text would be the obvious thing to reach
/// for and would be WRONG here: `encode_payload` runs Brotli and keeps it when
/// it helps, so a text payload of N bytes frames to ~0.83·N (measured ~17%
/// saving on English) or far less on repeated text — the harness would silently
/// under-fill and report flattering numbers at the high fill levels. A PRNG
/// keystream never compresses, so the requested fill is the delivered fill.
fn incompressible_blob(len: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut state = seed;
    while out.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Build a one-file attachment list whose **framed payload size** lands on
/// `target_payload_bytes` as closely as the envelope allows.
///
/// `encode_payload` wraps the blob in `[flags][text][0x00][namelen][name]
/// [u32 content_len][content]`, so the blob must be shorter than the target by
/// that envelope. Rather than hardcode the arithmetic (which would rot the
/// moment the envelope changes), this converges by asking
/// `compressed_payload_size` — the exact function the session will use — and
/// correcting.
fn blob_file(target_payload_bytes: usize, seed: u64) -> (Vec<FileEntry>, usize) {
    let name = "p.bin";
    let mut content_len = target_payload_bytes;
    for _ in 0..8 {
        let files = vec![FileEntry {
            filename: name.to_string(),
            content: incompressible_blob(content_len, seed),
        }];
        let actual = compressed_payload_size("", &files);
        if actual == target_payload_bytes {
            return (files, actual);
        }
        if actual > target_payload_bytes {
            let over = actual - target_payload_bytes;
            if over >= content_len {
                // Target is smaller than the bare envelope — can't hit it.
                return (files, actual);
            }
            content_len -= over;
        } else {
            content_len += target_payload_bytes - actual;
        }
    }
    let files = vec![FileEntry {
        filename: name.to_string(),
        content: incompressible_blob(content_len, seed),
    }];
    let actual = compressed_payload_size("", &files);
    (files, actual)
}

// ───────────────────────────── measurement ──────────────────────────────

#[derive(Clone, Copy)]
struct Quality {
    psnr_mean: f64,
    psnr_min: f64,
    ssim_mean: f64,
    worst_mb_min: f64,
    worst_mb_frame: usize,
    worst_mb_xy: (u32, u32),
    bad_pixels_max: u32,
}

/// Compare two flat I420 clips over the luma plane only.
///
/// Luma-only is deliberate: the hand measurement found chroma untouched
/// (U/V 48 dB) — **all** of this damage class is in Y.
fn measure(a: &[u8], b: &[u8], w: u32, h: u32, n_frames: u32) -> Quality {
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let mut psnrs = Vec::with_capacity(n_frames as usize);
    let mut ssims = Vec::with_capacity(n_frames as usize);
    let mut worst_mbs = Vec::with_capacity(n_frames as usize);
    let mut worst_locs = Vec::with_capacity(n_frames as usize);
    let mut bad = Vec::with_capacity(n_frames as usize);
    for f in 0..n_frames as usize {
        let off = f * frame_size;
        let ay = &a[off..off + y_size];
        let by = &b[off..off + y_size];
        psnrs.push(psnr_y(ay, by));
        ssims.push(ssim_y(ay, by, w, h));
        let (wmb, wx, wy) = worst_mb_psnr_y(ay, by, w, h);
        worst_mbs.push(wmb);
        worst_locs.push((wx, wy));
        bad.push(count_bad_pixels(ay, by, BAD_PIXEL_DELTA));
    }
    let n = n_frames as f64;
    let (worst_mb_frame, worst_mb_min) = worst_mbs
        .iter()
        .copied()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, f64::INFINITY));
    Quality {
        psnr_mean: psnrs.iter().sum::<f64>() / n,
        psnr_min: psnrs.iter().copied().fold(f64::INFINITY, f64::min),
        ssim_mean: ssims.iter().sum::<f64>() / n,
        worst_mb_min,
        worst_mb_frame,
        worst_mb_xy: worst_locs.get(worst_mb_frame).copied().unwrap_or((0, 0)),
        bad_pixels_max: bad.iter().copied().max().unwrap_or(0),
    }
}

struct FillRow {
    fill_pct: u32,
    /// Payload bytes requested (`fill% × offered capacity`).
    requested_payload: usize,
    /// Payload bytes actually built (post-envelope, post-Brotli-decision).
    actual_payload: usize,
    /// `StreamingEncodeSession::framed_message_len()` — what the session framed.
    framed: usize,
    /// Fill actually delivered, as a percentage of the OFFERED capacity, read
    /// back off the session's own framed length (`framed − FRAME_OVERHEAD`)
    /// rather than off the blob size we asked for. This is the column that
    /// catches a silent under-fill — e.g. Brotli having compressed the payload.
    delivered_pct: f64,
    enc_ms: f64,
    /// Wall time of the payload round-trip (pure-Rust CABAC walker).
    dec_ms: f64,
    /// Which cascade-safety tier `CascadeTier::Auto` resolved to. Printed
    /// because it is the main confounder in this table: tier selection is a
    /// function of payload size, and the tier filter only restricts the
    /// *coefficient* domains, so a tier step changes which domains carry the
    /// payload. Without this column a non-monotonic PSNR curve looks like noise.
    tier: Option<u8>,
    /// The WORST sample of `n_samples` salt draws (lowest mean PSNR). A gate
    /// asserts on the worst case it saw, not on an average that hides it.
    stego_only: Quality,
    /// How many salt draws produced this row (`PHASM_FILL_REPEATS`).
    n_samples: usize,
    /// max − min of mean Y-PSNR across those draws, in dB. 0 when
    /// `n_samples == 1`. This is the J2 jitter, and it is the number that says
    /// how much of the fill curve is signal.
    spread_db: f64,
    /// Source-vs-stego, i.e. the bundled number the OLD gate would report.
    /// Carried so the bundling effect is visible in one table.
    total_psnr_mean: f64,
    n_conceal: usize,
    /// `Some(err)` when the encode refused this fill level outright.
    failure: Option<String>,
}

fn run_fixture(spec: &VideoFixture) {
    // Recover from poisoning: one failing fixture is an independent
    // measurement and must not cascade-fail the rest of the corpus.
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let pass = passphrase();

    let (w, h) = probe_aligned_dims(spec);
    let src = ensure_yuv(spec, w, h);
    let frame_size = (w * h * 3 / 2) as usize;
    assert!(
        src.len() >= frame_size * spec.n_frames as usize,
        "[{}] source clip short: {} bytes for {} frames of {}",
        spec.name,
        src.len(),
        spec.n_frames,
        frame_size
    );

    // ── 1. offered capacity, from the production probe ──────────────────
    // Single GOP (gop_size = n_frames), matching `oh264_stream::encode*`,
    // which is what this harness encodes with. Probing one shape and
    // encoding another would make the fill percentages fiction.
    let t = std::time::Instant::now();
    let (probe, per_gop) = oh264_stream::probe_capacity(&src, w, h, spec.n_frames, spec.qp, spec.n_frames)
        .expect("capacity probe");
    let probe_ms = t.elapsed().as_secs_f64() * 1000.0;
    let capacity = probe.achievable_max_message_bytes;
    // `achievable_max_message_bytes = framed_ceiling − FRAME_OVERHEAD`, so the
    // framed length a session reports maps back to a payload-basis fill by
    // subtracting the same constant. Needed to turn `framed_message_len()`
    // into the delivered-fill column.
    let frame_overhead = frame_overhead();
    let framed_ceiling = capacity + frame_overhead;

    eprintln!();
    eprintln!(
        "┌─ [{name}] {w}×{h}×{n} qp={qp} pass=\"{pass}\" ──────────────────────",
        name = spec.name,
        w = w,
        h = h,
        n = spec.n_frames,
        qp = spec.qp,
        pass = pass,
    );
    eprintln!(
        "│ probe: {probe_ms:.0}ms  offered capacity = {capacity} B \
         (achievable_max_message_bytes; framed ceiling {framed_ceiling} B)"
    );
    eprintln!(
        "│        cover_bits={cb}  shadow_pool_bits={sp}  n_gops={ng}  \
         per_gop_tier0={pg:?}",
        cb = probe.cover_bits,
        sp = probe.shadow_pool_bits,
        ng = probe.n_gops,
        pg = per_gop,
    );
    eprintln!(
        "│        Σ per-tier payload (B): t0={} t1={} t2={} t3={} t4={}  \
         ← the capacity cost of each cascade-safety tier",
        probe.per_tier_sum_payload[0],
        probe.per_tier_sum_payload[1],
        probe.per_tier_sum_payload[2],
        probe.per_tier_sum_payload[3],
        probe.per_tier_sum_payload[4],
    );
    assert!(
        capacity > 0,
        "[{}] probe offered 0 bytes — nothing to measure",
        spec.name
    );

    // ── 2a. the CLEAN reference: same encoder, same pixels, NO stego ────
    //
    // Not a 1-character stego encode. `FRAME_OVERHEAD` frames 50 bytes before
    // a single payload byte, so on a clip whose whole capacity is a few hundred
    // bytes the "1-char baseline" is already a double-digit-percent fill and
    // costs real dB — measured 24.8-34.3 dB from source on the 480×256×6 smoke
    // depending only on the passphrase. Measuring against that would fold an
    // uncontrolled stego term into the reference and make every "stego-only"
    // number in the table meaningless. The 1-char encode is still made below,
    // but as a *data point* (the floor cost of the envelope), not as the
    // reference.
    let t = std::time::Instant::now();
    let clean = oh264_stream::encode_clean(&src, w, h, spec.n_frames, spec.qp, spec.n_frames);
    let clean_ms = t.elapsed().as_secs_f64() * 1000.0;
    let clean_clip = ffmpeg_decode_annex_b(&clean, &format!("fill_{}_clean", spec.name));
    assert_eq!(
        clean_clip.yuv.len(),
        frame_size * spec.n_frames as usize,
        "[{}] clean-reference decode size mismatch",
        spec.name
    );
    assert_eq!(
        clean_clip.n_conceal, 0,
        "[{}] clean reference bitstream is broken: {} concealment events (see {})",
        spec.name, clean_clip.n_conceal, clean_clip.log_path
    );

    // ── 2b. the 1-char stego encode — the envelope's floor cost ─────────
    // Salt pinned (fill level 0) so this data point is reproducible too.
    pin_salt(salt_seed(spec.name, 0, 0));
    let t = std::time::Instant::now();
    let baseline = oh264_stream::encode_with_files_reported(
        &src,
        w,
        h,
        spec.n_frames,
        spec.qp,
        BASELINE_MESSAGE,
        &[],
        &pass,
    )
    .expect("baseline encode");
    let base_ms = t.elapsed().as_secs_f64() * 1000.0;
    let base_clip = ffmpeg_decode_annex_b(&baseline.annex_b, &format!("fill_{}_base", spec.name));
    assert_eq!(
        base_clip.yuv.len(),
        frame_size * spec.n_frames as usize,
        "[{}] baseline decode size mismatch",
        spec.name
    );
    assert_eq!(
        base_clip.n_conceal, 0,
        "[{}] baseline bitstream is broken: {} concealment events (see {})",
        spec.name, base_clip.n_conceal, base_clip.log_path
    );

    // ── 3. the two reference terms, reported separately ─────────────────
    let transcode = measure(&src, &clean_clip.yuv, w, h, spec.n_frames);
    let floor = measure(&clean_clip.yuv, &base_clip.yuv, w, h, spec.n_frames);
    let floor_fill_pct = 100.0 * baseline.framed_len.saturating_sub(frame_overhead) as f64
        / capacity.max(1) as f64;
    eprintln!(
        "│ clean ref  enc={clean_ms:.0}ms            TRANSCODE-ONLY \
         (src vs NO-STEGO OH264): PSNR-Y avg={pa:.2} min={pm:.2}  SSIM={s:.4}  \
         worst-MB={mb:.2}dB",
        pa = transcode.psnr_mean,
        pm = transcode.psnr_min,
        s = transcode.ssim_mean,
        mb = transcode.worst_mb_min,
    );
    eprintln!(
        "│ 1-char     enc={base_ms:.0}ms framed={bf}B tier={bt}  STEGO FLOOR \
         (clean vs 1-char stego, ={ff:.1}% of payload capacity): \
         PSNR-Y avg={fa:.2} min={fm:.2}  SSIM={fs:.4}  worst-MB={fmb:.2}dB",
        bf = baseline.framed_len,
        bt = baseline
            .resolved_tier
            .map(|t| t.to_string())
            .unwrap_or_else(|| "?".into()),
        ff = floor_fill_pct,
        fa = floor.psnr_mean,
        fm = floor.psnr_min,
        fs = floor.ssim_mean,
        fmb = floor.worst_mb_min,
    );
    base_clip.discard_decoded_yuv();

    // ── 4. fill sweep ───────────────────────────────────────────────────
    let k = repeats();
    let mut rows: Vec<FillRow> = Vec::new();
    for (i, &pct) in FILL_PERCENTS.iter().enumerate() {
        let requested = ((capacity as u128 * pct as u128) / 100) as usize;
        if requested == 0 {
            eprintln!("│ fill {pct:>3}%: requested 0 B at capacity {capacity} B — skipped");
            continue;
        }
        // The blob content is fixed per fill level (seeded by the level index)
        // so repeats vary ONLY the crypto salt — the J2 axis — and not the
        // plaintext.
        let (files, actual_payload) = blob_file(requested, 0xC0FF_EE00 + i as u64);

        let mut samples: Vec<Quality> = Vec::with_capacity(k);
        let mut totals: Vec<f64> = Vec::with_capacity(k);
        let mut enc_total = 0.0f64;
        let mut dec_total = 0.0f64;
        let mut framed_len = 0usize;
        let mut tier = None;
        let mut conceal_max = 0usize;
        let mut refusal: Option<String> = None;

        for rep in 0..k {
            pin_salt(salt_seed(spec.name, pct, rep));
            let t = std::time::Instant::now();
            let encoded = oh264_stream::encode_with_files_reported(
                &src,
                w,
                h,
                spec.n_frames,
                spec.qp,
                "",
                &files,
                &pass,
            );
            enc_total += t.elapsed().as_secs_f64() * 1000.0;

            let report = match encoded {
                Ok(v) => v,
                Err(e) => {
                    // A refusal is a legitimate observation about the probe's
                    // documented ~2% over-report and the known non-monotonic
                    // feasibility, not a harness bug. Record and stop this level.
                    eprintln!(
                        "│ fill {pct:>3}% rep {rep}: requested {requested} B payload \
                         → ENCODE REFUSED: {e:?}"
                    );
                    refusal = Some(format!("{e:?}"));
                    break;
                }
            };
            framed_len = report.framed_len;
            tier = report.resolved_tier;

            // Round-trip the payload before trusting the pixels. A "100% fill"
            // measurement is meaningless if the encoder quietly embedded less
            // than it was handed — the PSNR would look flattering and the gate
            // would pass. Decode is the pure-Rust walker, i.e. the App Clip path.
            let t = std::time::Instant::now();
            let decoded = oh264_stream::decode_full(&report.annex_b, &pass).unwrap_or_else(|e| {
                panic!("[{}] fill {}% rep {}: stego decode failed: {e:?}", spec.name, pct, rep)
            });
            dec_total += t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                decoded.files.len(),
                1,
                "[{}] fill {}% rep {}: expected 1 attachment back, got {}",
                spec.name,
                pct,
                rep,
                decoded.files.len()
            );
            assert_eq!(
                decoded.files[0].content, files[0].content,
                "[{}] fill {}% rep {}: attachment came back altered ({} B out vs {} B in) — \
                 the quality numbers below describe an encode that did NOT carry the payload",
                spec.name,
                pct,
                rep,
                decoded.files[0].content.len(),
                files[0].content.len()
            );

            let clip =
                ffmpeg_decode_annex_b(&report.annex_b, &format!("fill_{}_{}", spec.name, pct));
            assert_eq!(
                clip.yuv.len(),
                frame_size * spec.n_frames as usize,
                "[{}] fill {}% rep {}: decode size mismatch",
                spec.name,
                pct,
                rep
            );
            samples.push(measure(&clean_clip.yuv, &clip.yuv, w, h, spec.n_frames));
            totals.push(measure(&src, &clip.yuv, w, h, spec.n_frames).psnr_mean);
            conceal_max = conceal_max.max(clip.n_conceal);
            // ~93 MB per sample at 1080p — drop it now that it is measured.
            // The Annex-B (a few MB) stays for eyeballing a bad level.
            clip.discard_decoded_yuv();
        }

        if samples.is_empty() {
            rows.push(FillRow {
                fill_pct: pct,
                requested_payload: requested,
                actual_payload,
                framed: 0,
                delivered_pct: 0.0,
                enc_ms: enc_total,
                dec_ms: 0.0,
                tier: None,
                n_samples: 0,
                spread_db: 0.0,
                stego_only: Quality {
                    psnr_mean: f64::NAN,
                    psnr_min: f64::NAN,
                    ssim_mean: f64::NAN,
                    worst_mb_min: f64::NAN,
                    worst_mb_frame: 0,
                    worst_mb_xy: (0, 0),
                    bad_pixels_max: 0,
                },
                total_psnr_mean: f64::NAN,
                n_conceal: 0,
                failure: refusal.or_else(|| Some("no sample".into())),
            });
            continue;
        }

        let n = samples.len();
        let worst = *samples
            .iter()
            .min_by(|a, b| a.psnr_mean.partial_cmp(&b.psnr_mean).unwrap())
            .unwrap();
        let hi = samples.iter().map(|q| q.psnr_mean).fold(f64::NEG_INFINITY, f64::max);
        let lo = samples.iter().map(|q| q.psnr_mean).fold(f64::INFINITY, f64::min);
        rows.push(FillRow {
            fill_pct: pct,
            requested_payload: requested,
            actual_payload,
            framed: framed_len,
            delivered_pct: 100.0 * framed_len.saturating_sub(frame_overhead) as f64
                / capacity as f64,
            enc_ms: enc_total / n as f64,
            dec_ms: dec_total / n as f64,
            tier,
            n_samples: n,
            spread_db: hi - lo,
            stego_only: worst,
            total_psnr_mean: totals.iter().sum::<f64>() / n as f64,
            n_conceal: conceal_max,
            // A refusal on SOME repeat but not others is exactly the known
            // non-monotonic feasibility; keep the measured samples and say so.
            failure: refusal.map(|e| format!("{e} (on a later salt draw)")),
        });
    }

    // ── 5. the table (this instrument's product) ────────────────────────
    eprintln!("│");
    eprintln!(
        "│ fill  req.payl  framed  deliv%  tier  enc(ms)  dec(ms) │ WORST-OF-{k}  PSNR avg  \
         PSNR min    SSIM  wMB(dB)   badpx/f  spread │ total PSNR  conceal",
        k = k
    );
    eprintln!(
        "│ ────  ────────  ──────  ──────  ────  ───────  ─────── │ ─────────────────────\
         ──────────────────────────────────────────── │ ──────────────────"
    );
    for r in &rows {
        if r.n_samples == 0 {
            let err = r.failure.as_deref().unwrap_or("no sample");
            eprintln!(
                "│ {f:>3}%  {rp:>8}  {dash:>6}  {dash:>6}  {dash:>4}  {e:>7.0}  {dash:>7} │ \
                 ALL {k} SALT DRAWS REFUSED: {err}",
                k = k.max(1),
                f = r.fill_pct,
                rp = r.requested_payload,
                dash = "—",
                e = r.enc_ms,
            );
            continue;
        }
        eprintln!(
            "│ {f:>3}%  {rp:>8}  {fr:>6}  {dp:>5.1}%  {ti:>4}  {e:>7.0}  {d:>7.0} │ \
             {pa:>21.2} {pm:>9.2} {ss:>7.4} {mb:>8.2} {bp:>9} {sp:>7.2} │ {tp:>10.2} {nc:>8}{warn}",
            f = r.fill_pct,
            rp = r.requested_payload,
            fr = r.framed,
            dp = r.delivered_pct,
            ti = r.tier.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
            e = r.enc_ms,
            d = r.dec_ms,
            pa = r.stego_only.psnr_mean,
            pm = r.stego_only.psnr_min,
            ss = r.stego_only.ssim_mean,
            mb = r.stego_only.worst_mb_min,
            bp = r.stego_only.bad_pixels_max,
            sp = r.spread_db,
            tp = r.total_psnr_mean,
            nc = r.n_conceal,
            warn = if r.failure.is_some() { "  ← REFUSED on a later salt draw" } else { "" },
        );
    }
    eprintln!(
        "│ transcode-only (src vs NO-STEGO reference): PSNR-Y avg={pa:.2} min={pm:.2}  \
         ← must stay ≫ the stego-only column",
        pa = transcode.psnr_mean,
        pm = transcode.psnr_min,
    );
    // Shape diagnostics — printed, never asserted. The expected shape is
    // "stego-only PSNR falls as fill rises"; a violation is a FINDING about the
    // encoder (most likely the auto-tier staircase, since the tier column moves
    // with payload size and the tier filter only restricts the coefficient
    // domains), not a reason to re-tune the harness.
    let measured: Vec<&FillRow> = rows.iter().filter(|r| r.n_samples > 0).collect();
    let mut inversions = Vec::new();
    for pair in measured.windows(2) {
        if pair[1].stego_only.psnr_mean > pair[0].stego_only.psnr_mean + 0.5 {
            inversions.push(format!(
                "{}%→{}% ({:.2}→{:.2} dB, tier {}→{})",
                pair[0].fill_pct,
                pair[1].fill_pct,
                pair[0].stego_only.psnr_mean,
                pair[1].stego_only.psnr_mean,
                pair[0].tier.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
                pair[1].tier.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
            ));
        }
    }
    if inversions.is_empty() {
        eprintln!("│ shape: stego-only PSNR falls monotonically with fill (as expected)");
    } else {
        eprintln!(
            "│ shape: NON-MONOTONIC — more payload measured QUIETER at: {}",
            inversions.join(", ")
        );
        eprintln!(
            "│        more payload should not be less damage. Check the tier column: \
             `auto_select_tier` picks the HIGHEST tier whose CSB+CSL capacity covers \
             the message, and the tier filter ∞-costs coefficient positions while \
             leaving MVD unfiltered — so a small payload can be pushed onto the \
             (louder) MVD domains."
        );
    }
    let tiers: Vec<String> = measured
        .iter()
        .map(|r| r.tier.map(|t| t.to_string()).unwrap_or_else(|| "?".into()))
        .collect();
    eprintln!("│ tiers across the ladder: [{}]", tiers.join(", "));
    for r in &rows {
        if r.n_samples > 0 && (r.actual_payload as f64 - r.requested_payload as f64).abs() > 2.0
        {
            eprintln!(
                "│ NOTE fill {}%: payload envelope moved the target \
                 ({} requested → {} delivered) — check `blob_file`",
                r.fill_pct, r.requested_payload, r.actual_payload
            );
        }
    }
    eprintln!("└──────────────────────────────────────────────────────────────────");
    clean_clip.discard_decoded_yuv();

    // ── 6. assertions ───────────────────────────────────────────────────
    assert!(
        transcode.psnr_mean >= TRANSCODE_ONLY_PSNR_MEAN_FLOOR_DB,
        "[{}] TRANSCODE-only mean Y-PSNR {:.2} dB < floor {} dB — the clean \
         baseline itself regressed, so every stego-only number above is suspect",
        spec.name,
        transcode.psnr_mean,
        TRANSCODE_ONLY_PSNR_MEAN_FLOOR_DB,
    );
    assert!(
        !rows.is_empty(),
        "[{}] no fill level produced a measurement",
        spec.name
    );
    for r in &rows {
        if r.n_samples == 0 {
            // EVERY salt draw was refused at this level. Only the top fill level
            // is allowed to do that (the probe's documented ~2% over-report). A
            // total refusal lower down means the capacity the probe OFFERS is not
            // encodable at all, which is a capacity-contract bug rather than the
            // known "interior infeasibility" jitter.
            //
            // A PARTIAL refusal (some draws fit, some don't) is NOT asserted on:
            // that is the already-accepted non-monotonic feasibility
            // (`docs/notes/2026-10-20-video-encode-ux-defects.md`,
            // "Known-and-accepted"). It is printed instead.
            assert_eq!(
                r.fill_pct,
                *FILL_PERCENTS.last().unwrap(),
                "[{}] fill {}% ({} B of {} B offered) was refused on every salt draw: \
                 {}. The capacity the probe OFFERS must be encodable — only the top \
                 level may graze the documented ~2% STC over-report.",
                spec.name,
                r.fill_pct,
                r.requested_payload,
                capacity,
                r.failure.as_deref().unwrap_or("?"),
            );
            continue;
        }
        assert_eq!(
            r.n_conceal, 0,
            "[{}] fill {}%: ffmpeg reported {} concealment events — bitstream broken",
            spec.name, r.fill_pct, r.n_conceal,
        );
        let (mean_floor, min_floor, wmb_floor) = floors_for(spec.name);
        assert!(
            r.stego_only.psnr_mean >= mean_floor,
            "[{}] fill {}%: stego-only mean Y-PSNR {:.2} dB < floor {} dB",
            spec.name,
            r.fill_pct,
            r.stego_only.psnr_mean,
            mean_floor,
        );
        assert!(
            r.stego_only.psnr_min >= min_floor,
            "[{}] fill {}%: stego-only min-frame Y-PSNR {:.2} dB < floor {} dB",
            spec.name,
            r.fill_pct,
            r.stego_only.psnr_min,
            min_floor,
        );
        assert!(
            r.stego_only.ssim_mean >= STEGO_ONLY_SSIM_MEAN_FLOOR,
            "[{}] fill {}%: stego-only mean Y-SSIM {:.4} < floor {:.4}",
            spec.name,
            r.fill_pct,
            r.stego_only.ssim_mean,
            STEGO_ONLY_SSIM_MEAN_FLOOR,
        );
        assert!(
            r.stego_only.worst_mb_min >= wmb_floor,
            "[{}] fill {}%: stego-only worst-MB Y-PSNR {:.2} dB < floor {} dB \
             (frame {}, MB ({}, {}))",
            spec.name,
            r.fill_pct,
            r.stego_only.worst_mb_min,
            wmb_floor,
            r.stego_only.worst_mb_frame,
            r.stego_only.worst_mb_xy.0,
            r.stego_only.worst_mb_xy.1,
        );
        assert!(
            r.stego_only.bad_pixels_max < STEGO_ONLY_BAD_PIXELS_CEILING_PER_FRAME,
            "[{}] fill {}%: peak {} pixels with |Δ|>{} on a single frame (limit {})",
            spec.name,
            r.fill_pct,
            r.stego_only.bad_pixels_max,
            BAD_PIXEL_DELTA,
            STEGO_ONLY_BAD_PIXELS_CEILING_PER_FRAME,
        );
    }
}

/// The one-time crypto-frame envelope, read straight from phasm-core.
///
/// `CapacityProbeResult::achievable_max_message_bytes` is defined as
/// `framed_ceiling − FRAME_OVERHEAD`, so this is what converts a session's
/// reported `framed_message_len()` back onto the payload basis the fill
/// percentages are expressed in. Imported rather than hardcoded as 50 so the
/// delivered-fill column cannot silently shift if the envelope changes;
/// `frame_overhead_matches_core` pins the current value so such a change is
/// noticed rather than absorbed.
fn frame_overhead() -> usize {
    phasm_core::stego::frame::FRAME_OVERHEAD
}

// ───────────────────────────── smoke ────────────────────────────────────

/// Default-`cargo test` smoke. One small fixture, the full fill sweep.
///
/// Small on purpose: it runs SEVEN OH264 encodes (clean reference + 1-char +
/// five fill levels) plus a capacity probe, so it is inherently ~7× the cost of
/// the single-encode gate next door. 480×256×6 keeps the whole test binary
/// (smoke + three unit checks) at ~20 s in an unoptimized `cargo test` build
/// and ~1.5 s in `--release`, while still leaving 6% of the offered capacity a
/// non-zero payload.
///
/// Production-resolution behaviour lives in the `#[ignore]`d cases — do NOT
/// read the smoke's absolute dB as representative of 1080p. At this size the
/// whole clip holds 282 payload bytes, so the per-flip damage is concentrated
/// into far fewer macroblocks than at 1080p and SSIM in particular reads
/// ~0.09 lower for the same PSNR.
#[test]
fn fill_smoke_iphone7() {
    run_fixture(&VideoFixture {
        name: "iphone7_fill_smoke",
        source_mp4: "IMG_4138.MOV",
        long_side_cap: Some(480),
        n_frames: 6,
        qp: 26,
    });
}

// ───────────────────────────── production ───────────────────────────────
//
// Same corpus clips the existing visual gate lists, at the same working
// resolution / frame count / QP, so the two gates' numbers are directly
// comparable. Each is minutes of OH264 encoding (6 encodes + a probe), hence
// `#[ignore]`.

const PROD_LONG: u32 = 1920;
const PROD_N: u32 = 30;
const PROD_QP: i32 = 26;

#[test]
#[ignore]
fn fill_iphone7_1080p() {
    run_fixture(&VideoFixture {
        name: "iphone7_1080p",
        source_mp4: "IMG_4138.MOV",
        long_side_cap: Some(PROD_LONG),
        n_frames: PROD_N,
        qp: PROD_QP,
    });
}

#[test]
#[ignore]
fn fill_carplane_1080p() {
    run_fixture(&VideoFixture {
        name: "carplane_1080p",
        source_mp4: "Artlist_CarPlane.mp4",
        long_side_cap: Some(PROD_LONG),
        n_frames: PROD_N,
        qp: PROD_QP,
    });
}

#[test]
#[ignore]
fn fill_schoolfight_1080p() {
    run_fixture(&VideoFixture {
        name: "schoolfight_1080p",
        source_mp4: "Artlist_SchoolFight.mp4",
        long_side_cap: Some(PROD_LONG),
        n_frames: PROD_N,
        qp: PROD_QP,
    });
}

#[test]
#[ignore]
fn fill_asia_bottle_1080p() {
    run_fixture(&VideoFixture {
        name: "asia_bottle_1080p",
        source_mp4: "Artlist_AsiaBottle.mp4",
        long_side_cap: Some(PROD_LONG),
        n_frames: PROD_N,
        qp: PROD_QP,
    });
}

#[test]
#[ignore]
fn fill_dji_mini2_1080p() {
    run_fixture(&VideoFixture {
        name: "dji_mini2_1080p",
        source_mp4: "dji_mini2_2_7k_24fps_h264_high.mp4",
        long_side_cap: Some(PROD_LONG),
        n_frames: PROD_N,
        qp: PROD_QP,
    });
}

// ───────────────────────────── unit checks ──────────────────────────────

/// The blob must not be compressible — otherwise every high fill level
/// silently under-fills and the whole table is optimistic.
#[test]
fn blob_is_incompressible() {
    for target in [64usize, 1024, 8192, 65536] {
        let (files, actual) = blob_file(target, 7);
        assert_eq!(
            actual, target,
            "blob_file missed its target payload size ({} → {})",
            target, actual
        );
        // Brotli must have been declined: the framed payload is the raw
        // envelope + raw content, so content_len == target − envelope.
        assert!(
            files[0].content.len() + 16 >= target,
            "payload shrank — Brotli compressed the blob ({} content for {} payload)",
            files[0].content.len(),
            target
        );
    }
}

/// Guard the fill ladder itself: a typo here would silently change what every
/// table in this file means.
#[test]
fn fill_ladder_is_the_documented_one() {
    assert_eq!(FILL_PERCENTS, &[6, 25, 50, 75, 100]);
}

/// `blob_file` needs the real envelope size to converge; if `FRAME_OVERHEAD`
/// ever changes, the delivered-fill column silently shifts. Pin it.
#[test]
fn frame_overhead_matches_core() {
    let fo = frame_overhead();
    assert_eq!(
        fo, 50,
        "FRAME_OVERHEAD changed ({fo}); the delivered-fill column in this gate's \
         tables is expressed on that basis — re-read the probe contract"
    );
}
