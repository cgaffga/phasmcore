// STEGO.A.10 diagnostic — primary-only Scheme A (no shadows) round-trip
// on the same corpus that failed for primary+shadow. Isolates whether
// the cascade leak is in the shadow logic or already in primary-only
// Scheme A + Tier 3 content-adaptive costs.

#![cfg(all(
    feature = "h264-encoder",
    feature = "openh264-backend",
    feature = "cabac-stego",
))]

use phasm_core::codec::h264::openh264_stego::{
    encode_yuv_with_pre_framed_bits_4domain, EncodeOpts,
};
use phasm_core::codec::h264::stego::cost_weights::CostWeights;
use phasm_core::codec::h264::stego::keys::CabacStegoMasterKeys;
use phasm_core::codec::h264::stego::hook::EmbedDomain;
use phasm_core::h264_stego_smart_decode_video;
use phasm_core::stego::{crypto, frame, payload};
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn try_scale_yuv(source_mp4: &str, tag: &str, w: u32, h: u32, n_frames: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source_mp4);
    if !src.exists() {
        return None;
    }
    let yuv_path = std::env::temp_dir().join(format!("stegoa_diag_{tag}_{w}x{h}_{n_frames}f.yuv"));
    if yuv_path.exists() && std::fs::metadata(&yuv_path).map(|m| m.len()).unwrap_or(0)
        == (w as u64 * h as u64 * 3 / 2) * n_frames as u64
    {
        return std::fs::read(&yuv_path).ok();
    }
    let vf = format!("scale={}:{}", w, h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    std::fs::read(&yuv_path).ok()
}

fn build_primary_frame_bits(message: &str, passphrase: &str) -> (Vec<u8>, [u8; 32]) {
    let primary_bytes = payload::encode_payload(message, &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&primary_bytes, passphrase).unwrap();
    let frame_bytes = frame::build_frame(primary_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let keys = CabacStegoMasterKeys::derive(passphrase).unwrap();
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;
    (frame_bits, hhat_seed)
}

const CORPUS: &[(&str, &str)] = &[
    ("iphone5",       "iphone5_1080p_30fps_h264_high.mov"),
    ("iphone7",       "iphone7_1080p_30fps_h264_high.mov"),
    ("lumix",         "lumix_g9_1080p_30fps_h264_high.mp4"),
    ("dji",           "dji_mini2_2_7k_24fps_h264_high.mp4"),
    ("img4138",       "IMG_4138.MOV"),
    ("asia_bottle",   "Artlist_AsiaBottle.mp4"),
    ("carplane",      "Artlist_CarPlane.mp4"),
    ("handbag",       "Artlist_Handbag.mp4"),
    ("horse_flag",    "Artlist_HorseFlag.mp4"),
    ("phone_booth",   "Artlist_PhoneBooth.mp4"),
    ("pirate_battle", "Artlist_PirateBattle.mp4"),
    ("school_fight",  "Artlist_SchoolFight.mp4"),
    ("woman_subway",  "Artlist_WomanSubway.mp4"),
];

/// Primary-only Scheme A + Tier 3 on the same corpus. If a fixture
/// fails HERE, the cascade leak is in Scheme A + Tier 3 itself, not
/// in shadow logic. If it passes here but failed in
/// stego_a_shadow_corpus_roundtrip, the bug is in the shadow
/// orchestrator's interaction with primary.
#[test]
fn stego_a_primary_only_scheme_a_corpus_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 8;
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let mut tested = 0usize;
    let mut skipped = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    for (tag, source_mp4) in CORPUS {
        let Some(yuv) = try_scale_yuv(source_mp4, tag, W, H, N) else {
            skipped.push(*tag);
            continue;
        };
        let primary_msg = format!("p {}", tag);
        let primary_pass = format!("ppass-diag-{}", tag);
        let (frame_bits, hhat_seed) = build_primary_frame_bits(&primary_msg, &primary_pass);

        let encode_res = encode_yuv_with_pre_framed_bits_4domain(
            &yuv, W, H, N, opts,
            &frame_bits, &hhat_seed, &weights,
        );
        match encode_res {
            Err(e) => failures.push((tag.to_string(), format!("encode: {e}"))),
            Ok(annex_b) => {
                match h264_stego_smart_decode_video(&annex_b, &primary_pass) {
                    Err(e) => failures.push((tag.to_string(), format!("decode: {e}"))),
                    Ok(recovered) if recovered != primary_msg => failures.push((
                        tag.to_string(),
                        format!("mismatch: got {recovered:?} expected {primary_msg:?}"),
                    )),
                    Ok(_) => {
                        eprintln!("DIAG primary-only SchemeA OK: {} (480×272×8, {} bytes)",
                            tag, annex_b.len());
                    }
                }
            }
        }
        tested += 1;
    }

    eprintln!(
        "DIAG primary-only SchemeA summary: tested={}, skipped={} ({:?}), failures={}",
        tested, skipped.len(), skipped, failures.len(),
    );
    if !failures.is_empty() {
        eprintln!("DIAG failures: {:?}", failures);
    }

    assert!(tested >= 1, "no fixtures present");
    assert!(
        failures.is_empty(),
        "DIAG primary-only SchemeA FAILED — {} failures: {:?}",
        failures.len(), failures,
    );
}
