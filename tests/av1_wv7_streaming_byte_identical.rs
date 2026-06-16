#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

//! WV.7.0 — byte-identical gate for the AV1 streaming shadow encode.
//!
//! Mirrors `oh264_g4_streaming_byte_identical.rs` for AV1. At WV.7.0
//! the streaming entry (`av1_encode_with_shadows_streaming`) is a
//! SCAFFOLD: it pulls GOPs forward from a `Av1GopYuvSource`, reassembles
//! the whole-clip YUV, and delegates to the existing
//! `av1_stego_encode_whole_video_with_shadows`. So the gate at WV.7.0
//! is trivially green — but the SAME gate stays green as WV.7.1+
//! progressively replace the scaffold with real per-GOP 2-sweep
//! streaming, byte-identical at every step.
//!
//! ## Two findings inherited from H.264's gate (apply unchanged to AV1)
//!
//! 1. **Shadow encode is non-deterministic by design** — `crypto::encrypt`
//!    draws a random AES salt+nonce per call. The byte-identical gate
//!    MUST pin `PHASM_DETERMINISTIC_SEED` (read by `crypto::encrypt`)
//!    so both sides draw identical salt+nonce. Without it any two
//!    encodes differ.
//! 2. **`frame_idx` will need local→global remapping** in WV.7.1+ once
//!    the scaffold's whole-clip reassembly is replaced by per-GOP
//!    standalone encodes. The scaffold avoids this because the
//!    reassembled YUV is fed to the whole-clip path verbatim. This
//!    finding lands the moment WV.7.1a ships the per-GOP clean-cover
//!    primitive.
//!
//! Fast tests (slice source) are NOT `#[ignore]`. The encode gate is
//! `#[ignore]` because AV1 shadow encode runs ~10-30 s even on tiny
//! fixtures.

use phasm_core::codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_with_shadows_streaming, Av1CallbackYuvSource, Av1FileYuvSource,
    Av1GopYuvSource, Av1SliceYuvSource,
};
use phasm_core::stego::crypto;
use phasm_core::stego::frame;
use phasm_core::stego::payload;

/// Deterministic gradient + hash tight-I420 frame — same pattern the
/// `av1_whole_video_memory` regression test uses. Gradient + small
/// position hash gives non-trivial AC energy so the shadow allocator
/// has real cover positions to work with. A pure XOR pattern (the
/// shape `oh264_g4_streaming_byte_identical`'s `synth_yuv` uses)
/// works for H.264's CABAC-domain cover but starves AV1's RDO of
/// Tier 1 (AC_COEFF_SIGN + GOLOMB_TAIL_LSB) positions and the encode
/// fails verify at every parity tier.
fn gradient_yuv_frame(width: u32, height: u32, frame_idx: u32) -> Vec<u8> {
    let y_size = (width as usize) * (height as usize);
    let uv_size = ((width / 2) as usize) * ((height / 2) as usize);
    let mut buf = vec![0u8; y_size + 2 * uv_size];
    for y in 0..(height as usize) {
        for x in 0..(width as usize) {
            let hash = ((x * 37 + y * 53 + (frame_idx as usize) * 11) % 19) as i32 - 9;
            let val = ((x + y + (frame_idx as usize) * 3) % 256) as i32 + hash;
            buf[y * (width as usize) + x] = val.clamp(0, 255) as u8;
        }
    }
    let cb_off = y_size;
    let cr_off = y_size + uv_size;
    for y in 0..(height as usize) / 2 {
        for x in 0..(width as usize) / 2 {
            let idx = y * ((width / 2) as usize) + x;
            buf[cb_off + idx] = (128 + ((x + frame_idx as usize) % 32) as i32 - 16) as u8;
            buf[cr_off + idx] = (128 + ((y + frame_idx as usize) % 32) as i32 - 16) as u8;
        }
    }
    buf
}

fn synth_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3 / 2 * n_frames) as usize);
    for f in 0..n_frames {
        out.extend_from_slice(&gradient_yuv_frame(w, h, f));
    }
    out
}

fn frame_size(w: u32, h: u32) -> usize {
    (w as usize) * (h as usize) * 3 / 2
}

// ---------- Fast tests (no encode) ----------

#[test]
fn slice_yuv_source_per_gop_reassembles_to_whole_clip() {
    let (w, h, gop, n) = (64u32, 48u32, 4u32, 11u32); // 11 frames, gop=4 → 3 GOPs (4+4+3)
    let yuv = synth_yuv(w, h, n);
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let mut reassembled: Vec<u8> = Vec::new();
    let mut gop_idx = 0u32;
    loop {
        let g = src.gop_yuv(gop_idx).expect("gop_yuv");
        if g.is_empty() {
            break; // end-of-stream signal
        }
        reassembled.extend_from_slice(&g);
        gop_idx += 1;
    }
    assert_eq!(gop_idx, 3, "expected 3 GOPs (4+4+3 frames)");
    assert_eq!(reassembled.len(), yuv.len(), "reassembled bytes != source bytes");
    assert_eq!(reassembled, yuv, "reassembled YUV diverged from source");
}

#[test]
fn slice_yuv_source_trailing_partial_gop_correct_size() {
    let (w, h, gop, n) = (32u32, 24u32, 5u32, 7u32); // 7 frames, gop=5 → 5+2
    let yuv = synth_yuv(w, h, n);
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let fs = frame_size(w, h);
    let g0 = src.gop_yuv(0).expect("g0");
    assert_eq!(g0.len(), 5 * fs, "GOP 0 should hold 5 frames");
    let g1 = src.gop_yuv(1).expect("g1");
    assert_eq!(g1.len(), 2 * fs, "GOP 1 (trailing) should hold 2 frames");
    let g2 = src.gop_yuv(2).expect("g2");
    assert!(g2.is_empty(), "GOP 2 past EOS should be empty");
}

#[test]
fn slice_yuv_source_rewind_is_stateless() {
    let (w, h, gop, n) = (32u32, 24u32, 3u32, 6u32);
    let yuv = synth_yuv(w, h, n);
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let first = src.gop_yuv(0).expect("g0");
    src.rewind().expect("rewind");
    let again = src.gop_yuv(0).expect("g0 again");
    assert_eq!(first, again, "rewind+re-read must be byte-identical");
}

// ---------- WV.7.8 — Av1CallbackYuvSource fast tests ----------

/// Build an `Av1CallbackYuvSource` backed by an `Arc<Vec<u8>>` of
/// whole-clip YUV — the simplest pull driver. Mirrors what a real
/// FFI bridge looks like: closures capture an external state handle
/// (here an `Arc<Vec<u8>>` standing in for the bridge's native
/// decoder context) and slice per-GOP YUV out of it.
fn make_callback_source(
    yuv: std::sync::Arc<Vec<u8>>,
    w: u32,
    h: u32,
    n_frames: u32,
    gop_size: u32,
) -> Av1CallbackYuvSource {
    let frame_bytes = (w * h * 3 / 2) as usize;
    let yuv_decode = yuv.clone();
    let decode_gop = move |gop_index: u32| -> Result<Vec<u8>, _> {
        let start_f = gop_index.saturating_mul(gop_size);
        if start_f >= n_frames {
            return Ok(Vec::new());
        }
        let end_f = ((gop_index + 1) * gop_size).min(n_frames);
        let start = (start_f as usize) * frame_bytes;
        let end = (end_f as usize) * frame_bytes;
        Ok(yuv_decode[start..end].to_vec())
    };
    let rewind = move || -> Result<(), _> { Ok(()) };
    Av1CallbackYuvSource::new(decode_gop, rewind)
}

#[test]
fn callback_yuv_source_per_gop_matches_slice_source() {
    let (w, h, gop, n) = (64u32, 48u32, 4u32, 11u32);
    let yuv = synth_yuv(w, h, n);
    let mut slice = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let mut cb = make_callback_source(std::sync::Arc::new(yuv.clone()), w, h, n, gop);
    for gop_idx in 0..4 {
        // gop_idx 0..3 → 3 GOPs of (4 + 4 + 3) frames + EOS
        let a = slice.gop_yuv(gop_idx).expect("slice");
        let b = cb.gop_yuv(gop_idx).expect("cb");
        assert_eq!(
            a, b,
            "callback source diverged from slice source at gop_idx={gop_idx}"
        );
    }
}

// ---------- WV.7.14 — Av1FileYuvSource fast tests ----------

#[test]
fn file_yuv_source_per_gop_matches_slice_source() {
    let (w, h, gop, n) = (64u32, 48u32, 4u32, 11u32);
    let yuv = synth_yuv(w, h, n);
    let tmp = std::env::temp_dir().join("phasm_wv7_14_file_yuv_match.yuv");
    std::fs::write(&tmp, &yuv).expect("write tmp yuv");
    let mut slice = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let mut file = Av1FileYuvSource::open(&tmp, w, h, n, gop).expect("open file source");
    for gop_idx in 0..4 {
        let a = slice.gop_yuv(gop_idx).expect("slice");
        let b = file.gop_yuv(gop_idx).expect("file");
        assert_eq!(
            a, b,
            "file source diverged from slice source at gop_idx={gop_idx}"
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn file_yuv_source_rewind_re_reads_same_bytes() {
    let (w, h, gop, n) = (32u32, 24u32, 3u32, 6u32);
    let yuv = synth_yuv(w, h, n);
    let tmp = std::env::temp_dir().join("phasm_wv7_14_file_yuv_rewind.yuv");
    std::fs::write(&tmp, &yuv).expect("write tmp yuv");
    let mut file = Av1FileYuvSource::open(&tmp, w, h, n, gop).expect("open file source");
    let first = file.gop_yuv(0).expect("g0");
    file.rewind().expect("rewind");
    let again = file.gop_yuv(0).expect("g0 again");
    assert_eq!(first, again, "file rewind+re-read must be byte-identical");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn callback_yuv_source_rewind_is_callable() {
    let (w, h, gop, n) = (32u32, 24u32, 3u32, 6u32);
    let yuv = synth_yuv(w, h, n);
    let mut cb = make_callback_source(std::sync::Arc::new(yuv.clone()), w, h, n, gop);
    let first = cb.gop_yuv(0).expect("g0");
    cb.rewind().expect("rewind");
    let again = cb.gop_yuv(0).expect("g0 again");
    assert_eq!(first, again, "callback rewind+re-read must be byte-identical");
}

/// Counter side-effect inside a `FnMut` closure — proves the
/// `&mut self` flow through the trait method correctly mutates the
/// underlying closure state (e.g. a native decoder advancing its
/// read cursor).
#[test]
fn callback_yuv_source_closure_can_mutate_state() {
    use std::sync::{Arc, Mutex};
    let calls = Arc::new(Mutex::new(0u32));
    let calls_for_decode = calls.clone();
    let calls_for_rewind = calls.clone();
    let mut cb = Av1CallbackYuvSource::new(
        move |_: u32| -> Result<Vec<u8>, _> {
            *calls_for_decode.lock().unwrap() += 1;
            Ok(Vec::new())
        },
        move || -> Result<(), _> {
            *calls_for_rewind.lock().unwrap() += 100;
            Ok(())
        },
    );
    cb.gop_yuv(0).unwrap();
    cb.gop_yuv(1).unwrap();
    cb.rewind().unwrap();
    cb.gop_yuv(0).unwrap();
    assert_eq!(*calls.lock().unwrap(), 103);
}

// ---------- Encode gate (slow; ignored by default) ----------

/// Extract `n` YUV420 frames from the real-content test-vector corpus
/// (`IMG_4138.MOV`). Same fixture the WV mass measurement test uses —
/// real DCT energy so AV1's RDO produces enough Tier 1 cover positions
/// for the shadow to round-trip. Synthetic patterns (XOR, gradient)
/// starve the cover at small fixtures.
fn corpus_yuv_concat(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    assert!(p.exists(), "corpus fixture missing: {}", p.display());
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&p)
        .args([
            "-frames:v",
            &n.to_string(),
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    assert_eq!(
        out.stdout.len(),
        (w * h * 3 / 2 * n) as usize,
        "ffmpeg produced unexpected byte count"
    );
    out.stdout
}

/// The ship gate: streaming == whole-clip, byte-for-byte. At WV.7.0
/// the scaffold delegates verbatim, so this gate is trivial — it
/// becomes meaningful at WV.7.1+ when increments start replacing
/// scaffold pieces.
///
/// Fixture: 256×144 × 12 frames (gop=4 → 3 GOPs) from the test-vector
/// corpus via ffmpeg. Same shape as `av1_wv_shadow_mass_measurement`.
#[test]
#[ignore = "AV1 shadow encode + ffmpeg corpus pull ~30-60 s; run with --ignored \
            --test-threads=1"]
fn streaming_shadow_byte_identical_to_whole_clip() {
    // Pin the diagnostic crypto seed so both encodes draw identical AES
    // salt+nonce — without it the random per-encrypt salt/nonce alone
    // makes any two encodes differ. Same fix H.264's g.4 gate uses.
    // SAFETY: single-threaded gate (--test-threads=1).
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    // Quantizer 30 (low — high quality, lots of cover bits) matches the
    // calibration baseline (`AllocationCalibration::AV1_1080P_QP30`) and
    // matches the WV mass-measurement test. The bridges run quantizer
    // 100 in production but the gate needs cover headroom, not
    // production-realistic bitrate.
    // Same shape as the WV mass measurement test: 256×144 × 12f
    // (3 GOPs of 4) at quantizer=30, real-content corpus YUV. Real
    // DCT energy → enough Tier 1 cover for a substantive shadow at
    // the production parity floor (16).
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);

    // Mirror the WV mass measurement test's call shape exactly:
    // payload-encode the message text (compression + flags byte) THEN
    // hand the resulting bytes to the session. The session does its
    // own crypto::encrypt + frame::build_frame internally. The shadow
    // message goes through the same payload-encode.
    let primary_text = "wv7.0 primary message — substantive";
    let shadow_text = "wv7.0 shadow — also substantive";
    let primary_pass = "wv7-primary";
    let shadow_pass = "wv7-shadow";
    let parity_floor = 16usize;

    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary payload");
    let shadow_payload =
        payload::encode_payload(shadow_text, &[]).expect("encode shadow payload");

    eprintln!("=== WV.7.0 gate: whole-clip reference encode (session API) ===");
    let frame_size = (w as usize) * (h as usize) * 3 / 2;
    let shadows_session = vec![Av1ShadowSpec {
        passphrase: shadow_pass.to_string(),
        message: shadow_payload.clone(),
    }];
    let mut session_ref = Av1StreamingEncodeSession::create_whole_video_with_shadows(
        primary_pass,
        &primary_payload,
        params,
        shadows_session,
        parity_floor,
    )
    .expect("create_whole_video_with_shadows");
    let mut reference: Vec<u8> = Vec::new();
    for f in 0..(n as usize) {
        let frame_yuv = &yuv[f * frame_size..(f + 1) * frame_size];
        session_ref
            .push_frame(frame_yuv, &mut reference)
            .expect("push_frame ref");
    }
    session_ref.finish(&mut reference).expect("finish ref");

    // ── Streamed: SAME inputs through `av1_encode_with_shadows_streaming`
    // over an `Av1SliceYuvSource`. The scaffold reassembles GOPs forward
    // and delegates to `av1_stego_encode_whole_video_with_shadows` —
    // which goes through the SAME `encode_whole_video_with_shadows_from_prepared`
    // as the session path (both pre-compute per_gop_natural +
    // per_gop_harvests and pass them in identically). So the gate
    // exercises:
    //   (1) the slice source slices forward correctly (already proven
    //       by the fast tests above)
    //   (2) the scaffold reassembly matches the session's incremental
    //       Pass-1 outputs byte-for-byte downstream
    //
    // Both encodes need the SAME framed primary bytes — pinning
    // PHASM_DETERMINISTIC_SEED makes the two crypto::encrypt calls
    // (one inside the session, one here) produce identical salt+nonce.
    let (ciphertext, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &ciphertext);

    let shadows: Vec<(&str, &[u8])> = vec![(shadow_pass, &shadow_payload)];

    eprintln!("=== WV.7.0 gate: streaming encode (Av1SliceYuvSource scaffold) ===");
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let streamed = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &primary_framed,
        primary_pass,
        &shadows,
        parity_floor,
    )
    .expect("streaming encode");

    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let first_diff = reference
        .iter()
        .zip(streamed.iter())
        .position(|(a, b)| a != b);
    eprintln!(
        "WV.7.0 GATE — {w}x{h} x{n}f gop={gop} primary+1shadow: \
         reference={} B, streamed={} B, first_diff={first_diff:?} \
         (None + equal len ⇒ byte-identical)",
        reference.len(),
        streamed.len(),
    );
    assert_eq!(
        reference, streamed,
        "WV.7.0 streaming shadow encode diverged from the whole-clip path \
         (first_diff={first_diff:?}) — wire-compat / correctness gate FAILED"
    );
}

/// WV.7.8 — full end-to-end encode driven by `Av1CallbackYuvSource`
/// instead of `Av1SliceYuvSource`. Same fixture, same inputs, same
/// expected bytes. Proves the FFI-shaped closure source path is
/// byte-identical to the in-RAM slicing source.
#[test]
#[ignore = "AV1 shadow encode ~100 s; run with --ignored --test-threads=1"]
fn callback_yuv_source_streaming_byte_identical_to_slice() {
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);

    let primary_text = "wv7.8 callback gate primary";
    let shadow_text = "wv7.8 callback shadow";
    let primary_pass = "wv78-primary";
    let shadow_pass = "wv78-shadow";
    let parity_floor = 16usize;

    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary payload");
    let shadow_payload =
        payload::encode_payload(shadow_text, &[]).expect("encode shadow payload");

    let (ciphertext, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &ciphertext);
    let shadows: Vec<(&str, &[u8])> = vec![(shadow_pass, &shadow_payload)];

    eprintln!("=== WV.7.8 gate: encode via Av1SliceYuvSource (reference) ===");
    let mut slice_src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let bytes_slice = av1_encode_with_shadows_streaming(
        &mut slice_src,
        n,
        params,
        &primary_framed,
        primary_pass,
        &shadows,
        parity_floor,
    )
    .expect("slice encode");

    eprintln!("=== WV.7.8 gate: encode via Av1CallbackYuvSource ===");
    let mut cb_src = make_callback_source(std::sync::Arc::new(yuv.clone()), w, h, n, gop);
    let bytes_cb = av1_encode_with_shadows_streaming(
        &mut cb_src,
        n,
        params,
        &primary_framed,
        primary_pass,
        &shadows,
        parity_floor,
    )
    .expect("callback encode");

    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let first_diff = bytes_slice
        .iter()
        .zip(bytes_cb.iter())
        .position(|(a, b)| a != b);
    eprintln!(
        "WV.7.8 GATE — slice={} B, callback={} B, first_diff={first_diff:?}",
        bytes_slice.len(),
        bytes_cb.len(),
    );
    assert_eq!(
        bytes_slice, bytes_cb,
        "Av1CallbackYuvSource diverged from Av1SliceYuvSource over the same buffered \
         YUV (first_diff={first_diff:?})"
    );
}
