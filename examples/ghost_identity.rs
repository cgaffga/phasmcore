// SHA-256 of Ghost encodes over a matrix of covers, modes, shadows, message
// sizes, memory budgets and thread counts: the byte-identity check of an
// encoder change that must not change the output (run before and after,
// compare the files).
//
// Usage (deterministic salt and nonce: PHASM_DETERMINISTIC_SEED is set here):
//   cargo run --release --features parallel --example ghost_identity -- <out.tsv> <cover>...
//   cover: jpeg=<path> | avif=<path> (needs `av1-decoder`)
// Environment (comma lists):
//   GI_MODES    plain,si,si-approx       (si: a Deep Cover written by core from
//               pixels that are not a JPEG decoded on its grid; si-approx: the
//               same cover with pixels it was not written from)
//   GI_SHADOWS  0,1,2                    (shadows of 100 and 60 characters; one
//               shadow larger than a short primary swaps them)
//   GI_MSGS     short,medium,max         (40 characters, max/8, about 0.9 max;
//               c<N>: N characters)
//   GI_BUDGETS  none,1,64,256,1674       (MB, set_memory_budget)
//   GI_THREADS  1,4                      (rayon pool of the encode)
//   GI_SPILL    <dir>                    (keyed order on disk, the spilled-order rung forced)
//   GI_STREAM   <dir>                    (the streamed encode, the lowest rung forced)
// One line per encode: cover, mode, shadows, message, budget, threads, the
// ladder rung the encode chose, the output's SHA-256 (or the error).

#[cfg(not(feature = "parallel"))]
fn main() {
    panic!("ghost_identity needs --features parallel");
}

#[cfg(feature = "parallel")]
fn main() {
    use sha2::{Digest, Sha256};
    use std::io::Write;
    use std::sync::atomic::{AtomicU8, Ordering};

    // SAFETY: set before any other thread exists.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "20261121") };
    // GI_SPILL=<dir>: every encode with its keyed order on disk (the lowest
    // memory rung forced, `PHASM_GHOST_FORCE_SPILL`), spilled to <dir>.
    fn plenty(_: &std::path::Path) -> Option<u64> {
        Some(u64::MAX / 2)
    }
    if let Some(dir) = std::env::var_os("GI_SPILL") {
        // SAFETY: as above.
        unsafe { std::env::set_var("PHASM_GHOST_FORCE_SPILL", "1") };
        phasm_core::set_spill_dir(Some(dir.into()), plenty);
    }
    // GI_STREAM=<dir>: every encode streamed (the lowest rung forced,
    // `PHASM_GHOST_FORCE_STREAM`), its files in <dir>.
    if let Some(dir) = std::env::var_os("GI_STREAM") {
        // SAFETY: as above.
        unsafe { std::env::set_var("PHASM_GHOST_FORCE_STREAM", "1") };
        phasm_core::set_spill_dir(Some(dir.into()), plenty);
    }
    static RUNG: AtomicU8 = AtomicU8::new(255);
    phasm_core::memory::set_telemetry_hook(Some(Box::new(|e| {
        if let phasm_core::memory::TelemetryEvent::EncodeStarted { ladder_rung, .. } = e {
            RUNG.store(*ladder_rung, Ordering::Relaxed);
        }
    })));

    let args: Vec<String> = std::env::args().collect();
    let out_path = &args[1];
    let list = |var: &str, default: &str| -> Vec<String> {
        std::env::var(var).unwrap_or_else(|_| default.into()).split(',').map(|s| s.trim().to_string()).collect()
    };
    let modes = list("GI_MODES", "plain,si,si-approx");
    let shadow_counts: Vec<usize> = list("GI_SHADOWS", "0,1,2").iter().map(|s| s.parse().unwrap()).collect();
    let msgs = list("GI_MSGS", "short,medium,max");
    let budgets: Vec<Option<usize>> =
        list("GI_BUDGETS", "none,1,64,256,1674").iter().map(|s| if s == "none" { None } else { Some(s.parse::<usize>().unwrap() << 20) }).collect();
    let threads: Vec<usize> = list("GI_THREADS", "1,4").iter().map(|s| s.parse().unwrap()).collect();
    let pools: Vec<(usize, rayon::ThreadPool)> =
        threads.iter().map(|&t| (t, rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap())).collect();

    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(out_path).unwrap();
    for spec in &args[2..] {
        let (kind, path) = spec.split_once('=').expect("cover: jpeg=<path> | avif=<path>");
        let file = std::fs::read(path).unwrap();
        let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
        // Source pixels that are not a JPEG decoded on a block grid: a JPEG's
        // decode cropped by (3, 5) pixels, or the AVIF's decode.
        let (src, w, h, jpeg) = match kind {
            "jpeg" => {
                let rgb = image::load_from_memory(&file).unwrap().to_rgb8();
                let (w0, h0) = rgb.dimensions();
                let (w, h) = (w0 - 3, h0 - 5);
                let mut px = Vec::with_capacity((w * h * 3) as usize);
                for y in 5..h0 {
                    for x in 3..w0 {
                        px.extend_from_slice(&rgb.get_pixel(x, y).0);
                    }
                }
                (px, w, h, Some(file))
            }
            "avif" => avif_rgb(&file),
            _ => panic!("cover kind {kind}"),
        };
        let deep = phasm_core::deep_cover_jpeg(&src, w, h, 92).unwrap();
        // Pixels the Deep Cover was not written from (approximate SI).
        let nudged: Vec<u8> = src.iter().enumerate().map(|(i, &v)| if i % 7 == 0 { v.saturating_add(1) } else { v }).collect();

        for mode in &modes {
            let (cover, pixels): (&[u8], Option<&[u8]>) = match mode.as_str() {
                "plain" => (jpeg.as_deref().unwrap_or(&deep), None),
                "si" => (&deep, Some(&src)),
                "si-approx" => (&deep, Some(&nudged)),
                m => panic!("mode {m}"),
            };
            let img = phasm_core::JpegImage::from_bytes(cover).unwrap();
            for &ns in &shadow_counts {
                let shadows: Vec<phasm_core::ShadowLayer> = (0..ns)
                    .map(|k| phasm_core::ShadowLayer {
                        message: text(100 - 40 * k, 77 + k as u32),
                        passphrase: format!("identity-shadow-{k}"),
                        files: vec![],
                    })
                    .collect();
                let shadow_total: usize = shadows.iter().map(|s| phasm_core::compressed_payload_size(&s.message, &[])).sum();
                let bands = match pixels {
                    Some(px) => phasm_core::ghost_bands_si(&img, px, w, h, ns, shadow_total),
                    None => phasm_core::ghost_bands(&img, ns, shadow_total),
                }
                .unwrap();
                for msg_kind in &msgs {
                    // Text of 6 bits per character: compresses to ~0.76.
                    let chars = match msg_kind.as_str() {
                        "short" => 40,
                        "medium" => bands.max_bytes / 8 * 100 / 76,
                        "max" => bands.max_bytes * 90 / 76,
                        m => m.strip_prefix('c').and_then(|n| n.parse().ok()).unwrap_or_else(|| panic!("message {m}")),
                    };
                    let msg = text(chars.max(1), 1);
                    for &budget in &budgets {
                        phasm_core::set_memory_budget(budget);
                        for (t, pool) in &pools {
                            RUNG.store(255, Ordering::Relaxed);
                            let t0 = std::time::Instant::now();
                            let r = pool.install(|| match (pixels, ns) {
                                (None, 0) => phasm_core::ghost_encode(cover, &msg, "identity-pass"),
                                (None, _) => phasm_core::ghost_encode_with_shadows(cover, &msg, &[], "identity-pass", &shadows, None),
                                (Some(px), 0) => phasm_core::ghost_encode_si(cover, px, w, h, &msg, "identity-pass"),
                                (Some(px), _) => phasm_core::ghost_encode_si_with_shadows(cover, px, w, h, &msg, &[], "identity-pass", &shadows),
                            });
                            let digest = match r {
                                Ok(bytes) => Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>(),
                                Err(e) => format!("ERR {e:?}"),
                            };
                            let b = budget.map_or("none".to_string(), |b| (b >> 20).to_string());
                            let line = format!(
                                "{name}\t{mode}\t{ns}\t{msg_kind}:{chars}\t{b}\t{t}\t{}\t{digest}",
                                RUNG.load(Ordering::Relaxed)
                            );
                            writeln!(out, "{line}").unwrap();
                            eprintln!("{line}\t{:.1}s", t0.elapsed().as_secs_f64());
                        }
                    }
                }
            }
        }
    }
    phasm_core::set_memory_budget(None);
}

/// `n` printable characters, 6 bits of entropy each (xorshift from `seed`).
#[cfg(feature = "parallel")]
fn text(n: usize, seed: u32) -> String {
    let mut s = 0x9E37_79B9u32 ^ seed.wrapping_mul(0x85EB_CA6B);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (b'!' + (s >> 26) as u8) as char
        })
        .collect()
}

#[cfg(all(feature = "parallel", feature = "av1-decoder"))]
fn avif_rgb(file: &[u8]) -> (Vec<u8>, u32, u32, Option<Vec<u8>>) {
    let d = phasm_core::photo::decode::decode_avif_rgb(file, phasm_core::MAX_PIXELS as u64).unwrap();
    (d.rgb, d.width, d.height, None)
}

#[cfg(all(feature = "parallel", not(feature = "av1-decoder")))]
fn avif_rgb(_file: &[u8]) -> (Vec<u8>, u32, u32, Option<Vec<u8>>) {
    panic!("avif covers need --features av1-decoder")
}
