// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.13 (#446) — production stego orchestrator on top of the
// OpenH264 backend. Single-domain (CoeffSign) STC encode + brute-force
// decode over walker-aligned cover, with passphrase-derived seeds.
//
// **STATUS: v1.0 ALPHA** — API surface + capacity primitive + encode/decode
// shape are shipped, but the round-trip is **not yet robust** against the
// 0.003 % residual cascade-leak rate observed at higher flip counts
// (~100 + flips). For small messages (≤ 10-20 flips, ~10 message bytes) the
// round-trip is essentially deterministic; above that, occasional 1-2 bit
// wire divergences from cascade leak break STC syndrome and fail decode.
// C.8.13 ship gate is documented in
// `docs/design/video/h264/phase-c8-visual-recon-plan.md` §C.8.13.
//
// Two follow-ons clear the alpha tag:
//   * C.8.13(b) — cascade-break gap audit: localise which domain
//     (CoeffSign / chroma DC/AC / deblock / …) leaks 2 / 76 384 wire
//     bits on 124-flip plans and fix structurally.
//   * C.8.13(c) — fallback: WET-INFINITY cost at empirically-detected
//     unsafe positions per b10_cascade_safe_roundtrip pattern (slow but
//     always correct), used as the ship-safe path until (b) lands.
//
// **Single-domain v1.0**: only CoeffSign positions carry the message.
// CoeffSuffixLsb / MvdSign / MvdSuffixLsb are reserved for v1.1+ once
// cascade-safety analysis (per-domain `cascade_safety.rs` equivalents)
// is wired through the OpenH264 path.
//
// **Cascade-safety**: relies on the C.8.3-11 dual-recon (`pVisualRecPic`)
// to keep mode-decision identical between baseline and stego encodes.
// On the 4-fixture C.8.12 corpus this holds for the small flip set in
// that test (≤ 3 flips); higher-flip-count plans intermittently observe
// the residual leak described above.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use core_openh264_sys::{
    encoder_pos_to_phasm_position_key, PhasmStegoDomain, PHASM_MB_TYPE_OTHER,
};

use super::cabac::bin_decoder::slice::walk_annex_b_for_cover;
use super::openh264::{
    set_frame_num, Encoder, EncoderError, StegoHandlers, StegoSession,
};
use super::stego::hook::EmbedDomain;
use super::stego::keys::CabacStegoMasterKeys;
use crate::stego::{crypto, frame, payload};
use crate::stego::error::StegoError;
use crate::stego::stc::embed::stc_embed;
use crate::stego::stc::extract::stc_extract;
use crate::stego::stc::hhat::generate_hhat;

/// STC constraint length. Must match between encode and decode (decoder
/// brute-forces `m_total` but treats `STC_H` as fixed).
const STC_H: usize = 4;

/// Encoder configuration knobs for the production stego path. The
/// `qp` / `intra_period` defaults track the OpenH264 fork's recommended
/// values for visual quality + reasonable IDR cadence.
#[derive(Debug, Clone, Copy)]
pub struct EncodeOpts {
    /// Quantization parameter (initial). Lower = higher quality + larger
    /// bitstream + more cover bits. Range 0..=51; typical 22..32.
    pub qp: i32,
    /// IDR period in frames. The encoder emits IDR at every N-th frame
    /// boundary. 60 = once per second at 60 fps.
    pub intra_period: i32,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        Self {
            qp: 26,
            intra_period: 60,
        }
    }
}

/// Encode `message` (plus optional `files`) into a stego H.264 Annex-B
/// stream using the OpenH264 backend with C.8.3-11 visual_recon cascade
/// safety.
///
/// The encode flow:
/// 1. Encrypt the payload with the passphrase (AES-256-GCM-SIV +
///    Argon2id key derive).
/// 2. Wrap in the standard phasm v1/v2 frame format.
/// 3. Baseline encode → walk → walker-aligned `PositionKey` cover.
/// 4. STC plan over `cover[0..m_total * w]` with `w = n_cover / m_total`.
/// 5. Re-encode with an `enc_pre_emit` hook that translates encoder hook
///    fires to canonical `PositionKey` and applies the planned flips.
///
/// Returns the raw Annex-B bytes ready to mux into MP4 (or store raw).
///
/// # Arguments
/// * `yuv` — raw YUV420p bytes, `width * height * 3 / 2 * n_frames` long.
/// * `width`, `height` — 16-aligned encode dimensions.
/// * `n_frames` — number of frames in `yuv`.
/// * `opts` — encoder knobs (QP, intra_period).
/// * `message` — UTF-8 text to embed (typically short; <1 KB).
/// * `files` — file attachments embedded alongside the text.
/// * `passphrase` — derives the AES key + STC hhat seed.
///
/// # Errors
/// * `StegoError::InvalidVideo` — dims not 16-aligned, yuv length wrong.
/// * `StegoError::MessageTooLarge` — cover capacity insufficient.
/// * Encryption / encoding failures bubble up as their respective
///   `StegoError` variants.
pub fn openh264_stego_encode_yuv_string(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    message: &str,
    files: &[payload::FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;

    // 1. Build the encrypted, framed payload bits.
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);
    let frame_bits = bytes_to_bits_msb_first(&frame_bytes);

    // 2. Derive the per-domain hhat seed from passphrase.
    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    // 3-6. Per-chunk 2-pass: baseline encode → walk → STC → stego encode.
    encode_yuv_with_pre_framed_bits(yuv, width, height, n_frames, opts, &frame_bits, &hhat_seed)
}

/// D.0.7.2 — exposed core of the OH264 stego encode for the streaming
/// session. Takes already-framed-and-encrypted payload bits + the
/// passphrase-derived STC seed, runs the 2-pass (baseline encode →
/// walker → STC plan → stego encode), returns the stego Annex-B.
///
/// The one-shot wrapper [`openh264_stego_encode_yuv_string`] computes
/// `frame_bits` from a UTF-8 message + AES-256-GCM-SIV encryption +
/// phasm v1 frame format. The streaming session (per-GOP) computes
/// the same bits but with an additional `chunk_frame` header
/// (`stego::chunk_frame`) wrapping the chunk-of-payload-bytes.
///
/// # Errors
/// * [`StegoError::InvalidVideo`] — dims not 16-aligned, yuv length
///   wrong, cover empty, or encoder failure.
/// * [`StegoError::MessageTooLarge`] — `n_cover / m_total < 1`
///   (chunk too big for this carrier).
pub(crate) fn encode_yuv_with_pre_framed_bits(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
) -> Result<Vec<u8>, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;
    let m_total = frame_bits.len();

    // Baseline encode + walker for cover capture.
    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;
    let _ = mb_height; // dim sanity already done in validate_dims
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    let baseline_bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        override_map.clone(),
        mb_type_table.clone(),
        applied.clone(),
        mb_width,
        mb_per_frame,
    )?;
    let baseline_walk = walk_annex_b_for_cover(&baseline_bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let baseline_positions = baseline_walk.cover.coeff_sign_bypass.positions;
    let baseline_bits = baseline_walk.cover.coeff_sign_bypass.bits;
    let n_cover = baseline_bits.len();

    if n_cover == 0 {
        return Err(StegoError::InvalidVideo("openh264 cover empty".into()));
    }
    if m_total == 0 {
        return Err(StegoError::InvalidVideo("empty frame bits".into()));
    }

    // Compute STC params: w = n_cover / m_total (must be >= 2 in practice
    // for the h=4 Viterbi to find non-trivial plans). The cover slice
    // used is m_total * w bits; remaining cover stays untouched.
    let w = n_cover / m_total;
    if w == 0 {
        return Err(StegoError::MessageTooLarge);
    }
    let used_cover = m_total * w;
    let cover_slice: Vec<u8> = baseline_bits[..used_cover].to_vec();

    // STC plan.
    let costs: Vec<f32> = vec![1.0; used_cover];
    let hhat = generate_hhat(STC_H, w, hhat_seed);
    let plan = stc_embed(&cover_slice, &costs, frame_bits, &hhat, STC_H, w)
        .ok_or(StegoError::MessageTooLarge)?;

    // Build the override map keyed by canonical PositionKey::raw().
    let mut overrides_map: HashMap<u64, u8> = HashMap::new();
    for i in 0..used_cover {
        if plan.stego_bits[i] != cover_slice[i] {
            overrides_map.insert(baseline_positions[i].raw(), plan.stego_bits[i]);
        }
    }
    {
        let mut map = override_map.lock().expect("override map lock");
        map.clear();
        for (k, v) in overrides_map.iter() {
            map.insert(*k, *v);
        }
    }

    // Re-encode with overrides.
    encode_once(
        yuv, width, height, n_frames, opts,
        override_map, mb_type_table, applied,
        mb_width, mb_per_frame,
    )
}

/// D.0.7.2 — expose the MSB-first byte→bit helper for the streaming
/// session to convert its per-chunk `chunk_frame` + inner-stego-frame
/// concatenation into bits before calling
/// [`encode_yuv_with_pre_framed_bits`].
pub(crate) fn bytes_to_bits_msb_first_pub(bytes: &[u8]) -> Vec<u8> {
    bytes_to_bits_msb_first(bytes)
}

/// Convenience entry: encode a text-only message (no file attachments).
pub fn openh264_stego_encode_yuv_text(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    openh264_stego_encode_yuv_string(yuv, width, height, n_frames, opts, message, &[], passphrase)
}

/// Decode a stego Annex-B stream produced by `openh264_stego_encode_yuv_string`.
///
/// Walks the bitstream with the phasm CABAC walker, brute-forces
/// `m_total` over byte-aligned increments, and on each candidate runs
/// STC extract → frame parse (CRC oracle) → decrypt → payload decode.
///
/// Returns the recovered `PayloadData` (text + file attachments).
pub fn openh264_stego_decode_yuv(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<payload::PayloadData, StegoError> {
    let walk = walk_annex_b_for_cover(annex_b)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    let cover_bits = walk.cover.coeff_sign_bypass.bits;
    let n_cover = cover_bits.len();
    if n_cover == 0 {
        return Err(StegoError::InvalidVideo("empty cover".into()));
    }

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    let min_m = frame::FRAME_OVERHEAD * 8;
    let max_m = frame::MAX_FRAME_BITS.min(n_cover);

    let mut m_total = min_m;
    while m_total <= max_m {
        if let Some(plaintext) = try_decode_at(&cover_bits, &hhat_seed, m_total, passphrase) {
            return Ok(plaintext);
        }
        m_total += 8;
    }
    Err(StegoError::FrameCorrupted)
}

/// Text-only convenience wrapper around [`openh264_stego_decode_yuv`].
pub fn openh264_stego_decode_yuv_string(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    openh264_stego_decode_yuv(annex_b, passphrase).map(|p| p.text)
}

/// Capacity primitive (#424) — encode the YUV once into a baseline H.264
/// stream, walk it, and report the maximum CoeffSign cover bits + the
/// max embeddable message length (in bytes, after framing overhead).
///
/// The reported byte count is the *theoretical* max with `w = 1`; STC
/// at `w = 1` is degenerate, so practical capacity is roughly half this.
/// A caller building a UI capacity meter should display `max / 2` as
/// the safe budget.
#[derive(Debug, Clone, Copy)]
pub struct OpenH264StegoCapacity {
    /// Total CoeffSign cover bits the walker recovers from a baseline
    /// encode of this YUV with these opts.
    pub cover_bits: usize,
    /// `(cover_bits / 8) - FRAME_OVERHEAD`. Upper bound on message
    /// bytes; practical limit is roughly half (STC needs `w >= 2`).
    pub max_message_bytes: usize,
}

/// Predict the cover capacity of a YUV stream by running one baseline
/// encode + walker. No stego overrides applied.
pub fn openh264_stego_capacity_yuv(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<OpenH264StegoCapacity, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;

    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;

    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    let bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
    )?;
    let walk = walk_annex_b_for_cover(&bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    let cover_bits = walk.cover.coeff_sign_bypass.bits.len();
    let max_message_bytes = (cover_bits / 8).saturating_sub(frame::FRAME_OVERHEAD);
    Ok(OpenH264StegoCapacity {
        cover_bits,
        max_message_bytes,
    })
}

/// #424 D.0.6 — per-GOP cover-bits probe for the streaming capacity
/// session. Encodes the given YUV chunk through the OH264 fork in
/// baseline mode (no overrides applied) and walks the emitted Annex-B
/// for CoeffSign cover positions. Returns the cover-bit count for the
/// chunk.
///
/// Used by `StreamingProbeSession::push_frame` once per GOP. The
/// emitted bitstream is discarded — only the position count matters.
/// Cost is ~equal to the actual stego encode's first pass (the STC
/// plan + override application is what makes the second pass slow,
/// not the OH264 encode itself).
pub(crate) fn count_cover_bits_for_gop(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<usize, StegoError> {
    validate_dims(gop_yuv, width, height, n_frames)?;
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let bitstream = encode_once(
        gop_yuv, width, height, n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
    )?;
    let walk = walk_annex_b_for_cover(&bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    Ok(walk.cover.coeff_sign_bypass.bits.len())
}

// ---------- internals ----------

fn validate_dims(yuv: &[u8], width: u32, height: u32, n_frames: u32) -> Result<(), StegoError> {
    if width % 16 != 0 || height % 16 != 0 {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    let expected = frame_size * n_frames as usize;
    if yuv.len() != expected {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {} ({}x{} x {} frames)",
            yuv.len(), expected, width, height, n_frames
        )));
    }
    Ok(())
}

fn encode_once(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    override_map: Arc<Mutex<HashMap<u64, u8>>>,
    mb_type_table: Arc<Mutex<Vec<u8>>>,
    applied: Arc<Mutex<u32>>,
    mb_width: u32,
    mb_per_frame: usize,
) -> Result<Vec<u8>, StegoError> {
    // Reset per-encode state.
    {
        let mut t = mb_type_table.lock().expect("mb_type lock");
        for x in t.iter_mut() {
            *x = 0xff;
        }
    }
    *applied.lock().expect("applied lock") = 0;

    let map_for_hook = override_map.clone();
    let applied_for_hook = applied;
    let mb_type_for_md = mb_type_table;
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, _orig| {
            if pos.domain != PhasmStegoDomain::CoeffSign as u8 {
                return None;
            }
            let map = map_for_hook.lock().ok()?;
            if map.is_empty() {
                return None;
            }
            let mb_addr = (pos.mb_y as usize) * (mb_width as usize) + (pos.mb_x as usize);
            if mb_addr >= mb_per_frame {
                return None;
            }
            // C.8.13(b) root-cause fix 2026-05-13: always pass
            // `PHASM_MB_TYPE_OTHER` here. The mb_type filter inside
            // `encoder_pos_to_phasm_position_key` would otherwise drop
            // valid overrides because `md_cost` fires AFTER the hook
            // (svc_encode_slice.cpp:2076 vs the hook in WelsEncInterY).
            // For frame N+1, the table still holds frame N's mb_type;
            // if that was I_16x16, `block_cat_matches_mb_type(I_16x16,
            // BC=2)` returns false and silently filters out frame N+1's
            // P-frame Luma 4×4 overrides. The walker's key set already
            // gates overrides to actual wire positions, so the mb_type
            // filter is redundant. See `audit_b_single_flip_probe` +
            // `audit_b_single_flip_probe_filter_bypassed` in
            // `core/tests/openh264_cascade_gap_audit.rs`.
            let key = encoder_pos_to_phasm_position_key(pos, PHASM_MB_TYPE_OTHER, mb_width)?;
            map.get(&key).map(|&t| {
                if let Ok(mut a) = applied_for_hook.lock() {
                    *a += 1;
                }
                t as i32
            })
        })),
        md_cost: Some(Box::new(move |cost| {
            let mb_addr = (cost.mb_y as usize) * (mb_width as usize) + (cost.mb_x as usize);
            if mb_addr < mb_per_frame {
                if let Ok(mut t) = mb_type_for_md.lock() {
                    t[mb_addr] = cost.mb_type;
                }
            }
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers)
        .map_err(|e| StegoError::InvalidVideo(format!("openh264 session: {e}")))?;

    let mut encoder =
        Encoder::new(width as i32, height as i32, opts.qp, opts.intra_period)
            .map_err(encoder_err_to_stego)?;

    let frame_y = (width * height) as usize;
    let frame_uv = (width * height / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;

    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut bitstream = Vec::with_capacity(2 * 1024 * 1024);
    for frame in 0..n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        let y = &yuv[base..base + frame_y];
        let u = &yuv[base + frame_y..base + frame_y + frame_uv];
        let v = &yuv[base + frame_y + frame_uv..base + frame_total];
        let (_, n) = encoder
            .encode_frame(y, u, v, (frame as i64) * 33, &mut out)
            .map_err(encoder_err_to_stego)?;
        bitstream.extend_from_slice(&out[..n]);
    }
    Ok(bitstream)
}

fn encoder_err_to_stego(e: EncoderError) -> StegoError {
    StegoError::InvalidVideo(format!("openh264 encoder: {e}"))
}

fn try_decode_at(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    m_total: usize,
    passphrase: &str,
) -> Option<payload::PayloadData> {
    if m_total == 0 {
        return None;
    }
    let n_cover = cover_bits.len();
    if m_total > n_cover {
        return None;
    }
    let w = n_cover / m_total;
    if w == 0 {
        return None;
    }
    let used = m_total * w;
    let hhat = generate_hhat(STC_H, w, hhat_seed);
    let extracted = stc_extract(&cover_bits[..used], &hhat, w);
    let frame_bits = &extracted[..m_total.min(extracted.len())];
    let frame_bytes = bits_to_bytes_msb_first(frame_bits);
    let parsed = frame::parse_frame(&frame_bytes).ok()?;
    let plaintext =
        crypto::decrypt(&parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce).ok()?;
    payload::decode_payload(&plaintext).ok()
}

fn bytes_to_bits_msb_first(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in 0..8 {
            out.push((b >> (7 - i)) & 1);
        }
    }
    out
}

fn bits_to_bytes_msb_first(bits: &[u8]) -> Vec<u8> {
    let n_bytes = bits.len() / 8;
    let mut out = Vec::with_capacity(n_bytes);
    for byte_idx in 0..n_bytes {
        let mut byte = 0u8;
        for i in 0..8 {
            byte |= (bits[byte_idx * 8 + i] & 1) << (7 - i);
        }
        out.push(byte);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::openh264::SESSION_TEST_MUTEX;

    fn synth_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
        let frame_size = (w * h * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames as usize);
        let mut s: u32 = 0xCAFE_F00D;
        for frame in 0..n_frames {
            for j in 0..h {
                for i in 0..w {
                    let v = ((i + frame * 2) ^ (j + frame * 3)) as u8;
                    out.push(v);
                }
            }
            for _plane in 0..2 {
                for j in 0..(h / 2) {
                    for i in 0..(w / 2) {
                        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                        let texture = (s >> 16) as u8;
                        let pos = (i + j + frame) as u8;
                        out.push(texture.wrapping_add(pos));
                    }
                }
            }
        }
        out
    }

    // The full round-trip lives in the `openh264_stego_roundtrip`
    // integration-test binary (`core/tests/openh264_stego_roundtrip.rs`).
    // Lib tests can't host it: C.8.7's `phasm_set_mv_override_active`
    // fork-side global is process-wide, and the 1300+ other openh264-
    // touching lib tests share state with it under cargo's single test
    // binary, producing 1-2 wire-flip cascade leaks. In an isolated
    // integration-test binary the state is pristine and the round-trip
    // is byte-exact (C.8.12 corpus suite is the structural proof).

    #[test]
    fn capacity_reports_positive_bits() {
        let _g = SESSION_TEST_MUTEX.lock().unwrap();
        let yuv = synth_yuv(320, 240, 2);
        let opts = EncodeOpts { qp: 22, intra_period: 60 };
        let cap = openh264_stego_capacity_yuv(&yuv, 320, 240, 2, opts).expect("capacity");
        assert!(cap.cover_bits > 1000, "expect non-trivial cover, got {}", cap.cover_bits);
        assert!(cap.max_message_bytes > 0);
    }
}
