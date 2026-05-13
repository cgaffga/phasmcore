// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.12 (#445) — OpenH264-backend visual_recon round-trip on
// real-world corpus fixtures.
//
// End-to-end pipeline (mirrors `b10_cascade_safe_roundtrip` minus the
// empirical safety scan — C.8.3-11 cascade-break makes structural what
// b10's scan validates empirically):
//
//   1. Load real iPhone/Artlist YUV (cached at /tmp via ffmpeg scale).
//   2. Baseline encode + walk → canonical PositionKey + cover bits.
//   3. STC plan over baseline_bits[0..STC_N].
//   4. Override map keyed by PositionKey::raw().
//   5. Pass 2 encode with key-translating hook (md_cost feeds mb_type).
//   6. Walk pass 2 → assert position sequence stable in [0, STC_N) and
//      extract message via stc_extract.
//
// Default smoke: iphone7_smoke (640x368 / 8 frames, intra_period=60)
// runs unignored. Larger fixtures stay #[ignore] for local invocation:
//   cargo test --release --features "h264-encoder openh264-backend" \
//       --test openh264_corpus_visual_recon -- --ignored --nocapture

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::cabac::bin_decoder::slice::walk_annex_b_for_cover;
use phasm_core::codec::h264::openh264::{
    set_frame_num, Encoder, StegoHandlers, StegoSession,
};
use phasm_core::stego::stc::embed::stc_embed;
use phasm_core::stego::stc::extract::stc_extract;
use phasm_core::stego::stc::hhat::generate_hhat;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use core_openh264_sys::{
    encoder_pos_to_phasm_position_key, PhasmStegoDomain, PHASM_MB_TYPE_OTHER,
};

/// Per-fixture spec mirroring h264_corpus_v26_validation but
/// targeting smaller / faster runs for the visual_recon round-trip.
struct FixtureSpec {
    name: &'static str,
    source_mp4: &'static str,
    encode_w: u32,
    encode_h: u32,
    n_frames: u32,
    qp: i32,
    intra_period: i32,
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

/// Scale source mp4 to raw YUV at the target encode dim, cache at /tmp.
fn ensure_yuv(spec: &FixtureSpec) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/phasm_c812_{}_{}x{}_f{}.yuv",
        spec.name, spec.encode_w, spec.encode_h, spec.n_frames
    );
    let frame_size = (spec.encode_w * spec.encode_h * 3 / 2) as usize;
    let need_bytes = frame_size * (spec.n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need_bytes {
            return data;
        }
    }
    let src = corpus_root().join(spec.source_mp4);
    if !src.exists() {
        panic!(
            "corpus fixture missing: {}\nexpected at {} (check test-vectors/video/h264/real-world/)",
            spec.source_mp4,
            src.display()
        );
    }
    let vf = format!("scale={}:{}", spec.encode_w, spec.encode_h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &spec.n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale -> yuv failed for {}", spec.source_mp4);
    std::fs::read(&yuv_path).expect("read yuv after regen")
}

/// Encode the YUV stream once. The hook reads `override_map` dynamically
/// — caller mutates the map between baseline and pass-2 invocations.
/// `md_cost` populates `mb_type_table` so the hook can translate
/// `PhasmStegoPos` → canonical phasm `PositionKey` for the right mode.
fn encode_with_overrides(
    yuv: &[u8],
    spec: &FixtureSpec,
    override_map: Arc<Mutex<HashMap<u64, u8>>>,
    mb_type_table: Arc<Mutex<Vec<u8>>>,
    applied_count: Arc<Mutex<u32>>,
    mb_width: u32,
    mb_count: usize,
) -> Vec<u8> {
    // Reset per-frame state.
    {
        let mut t = mb_type_table.lock().unwrap();
        for x in t.iter_mut() {
            *x = 0xff;
        }
    }
    *applied_count.lock().unwrap() = 0;

    let map_for_hook = override_map.clone();
    let mb_type_for_hook = mb_type_table.clone();
    let applied_for_hook = applied_count.clone();
    let mb_type_for_md = mb_type_table.clone();
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, _orig| {
            if pos.domain != PhasmStegoDomain::CoeffSign as u8 {
                return None;
            }
            let map = map_for_hook.lock().unwrap();
            if map.is_empty() {
                return None;
            }
            let mb_addr = (pos.mb_y as usize) * (mb_width as usize) + (pos.mb_x as usize);
            if mb_addr >= mb_count {
                return None;
            }
            let mt = mb_type_for_hook.lock().unwrap()[mb_addr];
            let mt_for_key = if mt == 0xff { PHASM_MB_TYPE_OTHER } else { mt };
            let key = encoder_pos_to_phasm_position_key(pos, mt_for_key, mb_width)?;
            map.get(&key).map(|&t| {
                *applied_for_hook.lock().unwrap() += 1;
                t as i32
            })
        })),
        md_cost: Some(Box::new(move |cost| {
            let mb_addr = (cost.mb_y as usize) * (mb_width as usize) + (cost.mb_x as usize);
            if mb_addr < mb_count {
                mb_type_for_md.lock().unwrap()[mb_addr] = cost.mb_type;
            }
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers).expect("register session");
    let mut encoder = Encoder::new(
        spec.encode_w as i32,
        spec.encode_h as i32,
        spec.qp,
        spec.intra_period,
    )
    .expect("encoder init");

    let frame_y = (spec.encode_w * spec.encode_h) as usize;
    let frame_uv = (spec.encode_w * spec.encode_h / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;

    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut bitstream = Vec::with_capacity(2 * 1024 * 1024);
    for frame in 0..spec.n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        let y = &yuv[base..base + frame_y];
        let u = &yuv[base + frame_y..base + frame_y + frame_uv];
        let v = &yuv[base + frame_y + frame_uv..base + frame_total];
        let (_, n) = encoder
            .encode_frame(y, u, v, (frame as i64) * 33, &mut out)
            .unwrap_or_else(|e| panic!("encode frame {}: {:?}", frame, e));
        bitstream.extend_from_slice(&out[..n]);
    }
    bitstream
}

fn run_visual_recon_roundtrip(spec: &FixtureSpec) {
    // SESSION_ALIVE inside the OpenH264 backend is process-global. Tests
    // in this integration target serialise through SESSION_GUARD.
    let _guard = session_guard().lock().unwrap();

    const STC_N: usize = 4096;
    const STC_M: usize = 16;
    const STC_H: usize = 7;
    let w = STC_N / STC_M;

    let mb_width = (spec.encode_w / 16) as u32;
    let mb_height = (spec.encode_h / 16) as u32;
    let mb_count = (mb_width * mb_height) as usize * (spec.n_frames as usize);

    eprintln!(
        "c812 fixture={} {}x{} frames={} qp={} ip={} mb_count_total={}",
        spec.name, spec.encode_w, spec.encode_h, spec.n_frames, spec.qp,
        spec.intra_period, mb_count
    );

    let yuv = ensure_yuv(spec);
    eprintln!("c812 loaded {} bytes of raw YUV", yuv.len());

    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_count]));
    let applied_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // ---------------- Baseline encode + walker ----------------
    let baseline_bitstream = encode_with_overrides(
        &yuv,
        spec,
        override_map.clone(),
        mb_type_table.clone(),
        applied_count.clone(),
        mb_width,
        mb_count,
    );
    eprintln!("c812 baseline encoded {} bytes", baseline_bitstream.len());

    let baseline_walk =
        walk_annex_b_for_cover(&baseline_bitstream).expect("baseline walker");
    let baseline_positions = baseline_walk.cover.coeff_sign_bypass.positions.clone();
    let baseline_bits = baseline_walk.cover.coeff_sign_bypass.bits.clone();
    assert!(
        baseline_positions.len() >= STC_N,
        "fixture {}: baseline walker emitted {} CoeffSign positions, need >= {}",
        spec.name,
        baseline_positions.len(),
        STC_N
    );
    eprintln!(
        "c812 baseline walker: {} CoeffSign positions",
        baseline_positions.len()
    );

    // ---------------- STC plan ----------------
    let cover: Vec<u8> = baseline_bits.iter().take(STC_N).copied().collect();
    let costs: Vec<f32> = vec![1.0; STC_N];
    let message: Vec<u8> = (0..STC_M).map(|i| ((i * 37 + 13) % 2) as u8).collect();
    let hhat = generate_hhat(STC_H, w, &[0xceu8; 32]);
    let plan = stc_embed(&cover, &costs, &message, &hhat, STC_H, w)
        .expect("stc_embed returned None");
    assert_eq!(plan.stego_bits.len(), STC_N);

    let mut overrides: HashMap<u64, u8> = HashMap::new();
    for i in 0..STC_N {
        if plan.stego_bits[i] != cover[i] {
            overrides.insert(baseline_positions[i].raw(), plan.stego_bits[i]);
        }
    }
    let planned_flips = overrides.len();
    assert!(
        planned_flips > 0,
        "fixture {}: STC produced zero flips",
        spec.name
    );
    eprintln!("c812 stc plan: {} flips over {} cover bits", planned_flips, STC_N);

    // ---------------- Pass 2 encode with overrides ----------------
    {
        let mut map = override_map.lock().unwrap();
        map.clear();
        for (k, v) in overrides.iter() {
            map.insert(*k, *v);
        }
    }
    let stego_bitstream = encode_with_overrides(
        &yuv,
        spec,
        override_map.clone(),
        mb_type_table.clone(),
        applied_count.clone(),
        mb_width,
        mb_count,
    );
    let p2_applied = *applied_count.lock().unwrap();
    eprintln!(
        "c812 pass2 encoded {} bytes; hook applied={} (>= planned={} since sim-fires share key)",
        stego_bitstream.len(),
        p2_applied,
        planned_flips
    );

    // ---------------- Walk pass 2 + verify position sequence ----------------
    let p2_walk = walk_annex_b_for_cover(&stego_bitstream).expect("pass2 walker");
    let p2_positions = &p2_walk.cover.coeff_sign_bypass.positions;
    let p2_bits = &p2_walk.cover.coeff_sign_bypass.bits;
    assert!(
        p2_positions.len() >= STC_N,
        "fixture {}: pass2 walker emitted {} positions, need >= {}",
        spec.name,
        p2_positions.len(),
        STC_N
    );

    // C.8.3-11 cascade-break should keep mode-decision identical →
    // walker position sequence in [0, STC_N) must be byte-identical
    // between baseline and pass 2. If this breaks, the cascade-break
    // failed at the recon-pixel level for this fixture.
    let mut p2_seq_breaks = 0u32;
    for i in 0..STC_N {
        if p2_positions[i].raw() != baseline_positions[i].raw() {
            p2_seq_breaks += 1;
        }
    }
    assert_eq!(
        p2_seq_breaks, 0,
        "fixture {}: {} of {} message-region positions diverged in pass 2 — \
         C.8.3-11 cascade-break leaked at this content",
        spec.name, p2_seq_breaks, STC_N
    );

    // ---------------- STC extract on pass 2 walker bits ----------------
    let cover_p2: Vec<u8> = p2_bits.iter().take(STC_N).copied().collect();
    let extracted = stc_extract(&cover_p2, &hhat, w);
    let recovered: Vec<u8> = extracted.into_iter().take(STC_M).collect();

    assert_eq!(
        recovered, message,
        "c812 round-trip failed for fixture {}:\n  message  = {:?}\n  recovered = {:?}",
        spec.name, message, recovered
    );
    eprintln!(
        "c812 fixture={} ROUND-TRIP GREEN: {} bits recovered, {} flips on wire, {} hook fires",
        spec.name, STC_M, planned_flips, p2_applied
    );

    // Sanity: cover_p2 differs from baseline at the planned flip
    // positions (proves the overrides reached the wire).
    let mut wire_flips_observed = 0u32;
    for i in 0..STC_N {
        if cover_p2[i] != cover[i] {
            wire_flips_observed += 1;
        }
    }
    assert_eq!(
        wire_flips_observed as usize, planned_flips,
        "fixture {}: observed {} wire flips, planned {} — overrides didn't land cleanly",
        spec.name, wire_flips_observed, planned_flips
    );
}

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

// ============== Fixtures ==============

const SPEC_IPHONE7_SMOKE: FixtureSpec = FixtureSpec {
    name: "iphone7_smoke",
    source_mp4: "IMG_4138.MOV",
    encode_w: 640,
    encode_h: 368,
    n_frames: 8,
    qp: 22,
    intra_period: 60,
};

const SPEC_IPHONE7_FULL: FixtureSpec = FixtureSpec {
    name: "iphone7_landscape",
    source_mp4: "IMG_4138.MOV",
    encode_w: 1920,
    encode_h: 1072,
    n_frames: 12,
    qp: 26,
    intra_period: 60,
};

const SPEC_ARTLIST_SCHOOLFIGHT: FixtureSpec = FixtureSpec {
    name: "artlist_schoolfight",
    source_mp4: "Artlist_SchoolFight.mp4",
    encode_w: 1280,
    encode_h: 720,
    n_frames: 12,
    qp: 26,
    intra_period: 60,
};

const SPEC_ARTLIST_CARPLANE: FixtureSpec = FixtureSpec {
    name: "artlist_carplane",
    source_mp4: "Artlist_CarPlane.mp4",
    encode_w: 1072,
    encode_h: 1920,
    n_frames: 12,
    qp: 26,
    intra_period: 60,
};

#[test]
fn c812_iphone7_smoke() {
    run_visual_recon_roundtrip(&SPEC_IPHONE7_SMOKE);
}

#[test]
#[ignore]
fn c812_iphone7_full() {
    run_visual_recon_roundtrip(&SPEC_IPHONE7_FULL);
}

#[test]
#[ignore]
fn c812_artlist_schoolfight() {
    run_visual_recon_roundtrip(&SPEC_ARTLIST_SCHOOLFIGHT);
}

#[test]
#[ignore]
fn c812_artlist_carplane() {
    run_visual_recon_roundtrip(&SPEC_ARTLIST_CARPLANE);
}
