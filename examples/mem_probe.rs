// Peak heap of phasm-core operations, measured with a counting allocator.
//
// Usage:
//   cargo run --release --example mem_probe -- <op> <image> [passphrase] [message-file]
//   op: parse | capacity | decode | ghost | shadow | encode | encode-budget | build-cover
//   op shape <jpeg>: `decode_peak_bytes` and the coefficient count it reads.
// `encode-budget` sets a 1.67 GB memory budget first (what cgPhone reports).
//   op encode-si <raw-rgb-file> <pass> <message-file> <width> <height>: Deep
//   Cover as the apps do it (cover written by `deep_cover_jpeg` at QF92, then
//   `ghost_encode_si` with the same pixels), at the app budget.
//   op to-rgb <jpeg|avif> <out.rgb>: raw RGB for encode-si (prints width height).
//   op model-check <cover.jpg>...: each encode's whole peak (from just before
//   the call) against `ghost_encode_peak_bytes` and the plan's own estimate
//   (see `model_check`: MC_CASES, MC_SI).
//   op peak-model <w>x<h>...: `ghost_encode_peak_bytes` (Deep Cover, 1 kB
//   message) at RAYON_NUM_THREADS workers and APP_BUDGET_MB.
//   op app-flow <avif|jpeg> [pass] [message-file]: the Android app's whole
//   image pipeline on a picked AVIF (or JPEG), stage by stage (see `app_flow`;
//   needs `--features av1-decoder`). APP_RELIEF=all runs
//   `malloc_zone_pressure_relief` after the decode, the Ghost cover and the
//   encode (`relief_after`). For the iOS footprint run it with
//   `MallocLargeCache=0`: macOS keeps freed huge blocks in a cache that iOS
//   does not have (docs/design/app/large-samples.md, "Memory").
// Reallocations count as a copying allocator holds them (the new block next
// to the old until the copy is done: Android's Scudo); MEM_PROBE_REALLOC=delta
// counts only the size change.
// MEM_PROBE_SAMPLE_S=<s>: the live heap and its peak every <s> seconds.
// MEM_PROBE_SPILL_DIR=<dir>: the Ghost encoder's spill directory (the lowest
// memory rung; PHASM_GHOST_FORCE_SPILL forces it).
// MEM_PROBE_OUT=<path> writes the encoded image (with PHASM_DETERMINISTIC_SEED
// for byte comparisons between builds). MEM_PROBE_REPS=<n> repeats the
// operation n times in the process (the device bench runs two encodes).
// On macOS the result line adds the process `phys_footprint` (what jetsam
// judges on iOS): its lifetime peak and the value at the start and end. It
// exceeds the heap peak by the freed large blocks libmalloc keeps resident
// (`MallocLargeCache=0` makes the two agree); CPU = user + system time.
// Page faults (minor + major) count the kernel's zero-fill work.

use std::alloc::{GlobalAlloc, Layout};
#[cfg(not(phasm_large_alloc))]
use std::alloc::System;
use std::sync::atomic::{AtomicUsize, Ordering};

// `--cfg phasm_large_alloc` (RUSTFLAGS) builds in the ios-bridge allocator
// (large blocks mapped from the kernel); MEM_PROBE_MMAP_MB=<1|2|4|8|16|32>
// picks its threshold, unset keeps the system allocator.
#[cfg(phasm_large_alloc)]
#[path = "../../ios-bridge/src/large_alloc.rs"]
#[allow(dead_code)]
mod large_alloc;

#[cfg(phasm_large_alloc)]
static LARGE: [large_alloc::LargeAlloc; 7] = [
    large_alloc::LargeAlloc::new(1 << 20),
    large_alloc::LargeAlloc::new(2 << 20),
    large_alloc::LargeAlloc::new(4 << 20),
    large_alloc::LargeAlloc::new(8 << 20),
    large_alloc::LargeAlloc::new(16 << 20),
    large_alloc::LargeAlloc::new(32 << 20),
    large_alloc::LargeAlloc::new(usize::MAX),
];
/// Index into LARGE; set once at the start of `main`, before any block of
/// 1 MiB or more exists.
#[cfg(phasm_large_alloc)]
static LARGE_IX: AtomicUsize = AtomicUsize::new(6);

#[cfg(phasm_large_alloc)]
fn inner() -> &'static large_alloc::LargeAlloc {
    &LARGE[LARGE_IX.load(Ordering::Relaxed)]
}
#[cfg(not(phasm_large_alloc))]
fn inner() -> &'static System {
    &System
}

struct Counting;
static REALLOC_DELTA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// MEM_PROBE_LOG=<path>: every allocation / free of 1 MiB or more, in order
// (op, size, address), written at the end. Fixed table, no allocation.
const LOG_CAP: usize = 1 << 20;
static LOG_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOG_N: AtomicUsize = AtomicUsize::new(0);
static mut LOG: [(usize, usize, usize); LOG_CAP] = [(0, 0, 0); LOG_CAP];
fn log(op: usize, size: usize, p: *mut u8) {
    if size >= 1 << 20 && LOG_ON.load(Ordering::Relaxed) {
        let i = LOG_N.fetch_add(1, Ordering::Relaxed);
        if i < LOG_CAP {
            unsafe { LOG[i] = (op, size, p as usize) };
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { inner().alloc(l) };
        log(1, l.size(), p);
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // Forward to calloc: the default (alloc + memset) dirties every page
        // of a fresh zeroed block, which the system allocator does not.
        let p = unsafe { inner().alloc_zeroed(l) };
        log(2, l.size(), p);
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        log(0, l.size(), p);
        unsafe { inner().dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        log(0, l.size(), p);
        // Counted as a copying allocator holds it (Android's Scudo, and
        // jemalloc between size classes): the new block next to the old one
        // until the copy is done. MEM_PROBE_REALLOC=delta counts only the
        // size change (an allocator growing the block in place).
        let copy = !REALLOC_DELTA.load(Ordering::Relaxed);
        if copy {
            let c = CUR.fetch_add(n, Ordering::Relaxed) + n;
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        let q = unsafe { inner().realloc(p, l, n) };
        log(1, n, q);
        if copy {
            CUR.fetch_sub(if q.is_null() { n } else { l.size() }, Ordering::Relaxed);
        } else if !q.is_null() {
            if n > l.size() {
                let c = CUR.fetch_add(n - l.size(), Ordering::Relaxed) + n - l.size();
                PEAK.fetch_max(c, Ordering::Relaxed);
            } else {
                CUR.fetch_sub(l.size() - n, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

fn mb(b: usize) -> usize {
    b >> 20
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    static mach_task_self_: u32;
    fn task_info(task: u32, flavor: u32, info: *mut u64, count: *mut u32) -> i32;
}

/// (phys_footprint, lifetime peak of phys_footprint) of this process, from
/// `task_info(TASK_VM_INFO)` (byte offsets 144 and 168 of `task_vm_info`).
#[cfg(target_os = "macos")]
fn footprint() -> (usize, usize) {
    let mut buf = [0u64; 64];
    let mut count: u32 = 128;
    let kr = unsafe { task_info(mach_task_self_, 22, buf.as_mut_ptr(), &mut count) };
    if kr != 0 {
        return (0, 0);
    }
    (buf[144 / 8] as usize, buf[168 / 8] as usize)
}

#[cfg(not(target_os = "macos"))]
fn footprint() -> (usize, usize) {
    (0, 0)
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
}

/// `APP_RELIEF=<stage,...|all>`: `malloc_zone_pressure_relief(NULL, 0)` after
/// those app-flow stages (decode, ghost-cover, encode), with what it returned
/// and the footprint after it. With xzone malloc (current libmalloc's default
/// on Apple silicon, macOS and iOS) the default zone's pressure relief is a
/// no-op returning 0 (`malloc_zone_pressure_relief_fallback`): the freed
/// picture macOS keeps in its huge cache stays; `MallocLargeCache=0` turns
/// that cache off, which is the iOS default
/// (`docs/design/app/large-samples.md`, "Memory").
fn relief_after(stage: &str) -> String {
    let on = std::env::var("APP_RELIEF").is_ok_and(|v| v == "all" || v.split(',').any(|s| s == stage));
    if !on {
        return String::new();
    }
    #[cfg(target_os = "macos")]
    let released = unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    #[cfg(not(target_os = "macos"))]
    let released = 0usize;
    format!("  relief after {stage}: returned {} MB, footprint {} MB\n", mb(released), mb(footprint().0))
}

/// (user + system CPU seconds, page faults) of this process.
fn cpu_seconds() -> (f64, i64) {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    (tv(ru.ru_utime) + tv(ru.ru_stime), ru.ru_minflt as i64 + ru.ru_majflt as i64)
}

fn main() {
    #[cfg(phasm_large_alloc)]
    if let Ok(v) = std::env::var("MEM_PROBE_MMAP_MB") {
        let ix = [1, 2, 4, 8, 16, 32].iter().position(|&m| v == m.to_string()).expect("MEM_PROBE_MMAP_MB: 1, 2, 4, 8, 16 or 32");
        LARGE_IX.store(ix, Ordering::Relaxed);
    }
    // MEM_PROBE_SPILL_DIR=<dir>: the Ghost encoder may spill its keyed order
    // there (the lowest memory rung; free space from statvfs), as the apps
    // set it; with PHASM_GHOST_FORCE_SPILL every encode does.
    if let Some(dir) = std::env::var_os("MEM_PROBE_SPILL_DIR") {
        fn free(dir: &std::path::Path) -> Option<u64> {
            use std::os::unix::ffi::OsStrExt;
            let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
            let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
            (unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0).then(|| s.f_bavail as u64 * s.f_frsize as u64)
        }
        phasm_core::set_spill_dir(Some(dir.into()), free);
    }
    if std::env::var("MEM_PROBE_REALLOC").is_ok_and(|v| v == "delta") {
        REALLOC_DELTA.store(true, Ordering::Relaxed);
    }
    // MEM_PROBE_SAMPLE_S=<s>: the live heap and its peak so far every <s>
    // seconds on stderr (a time series, as the device test's sampler).
    if let Some(s) = std::env::var("MEM_PROBE_SAMPLE_S").ok().and_then(|v| v.parse::<f64>().ok()) {
        let t0 = std::time::Instant::now();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs_f64(s));
            let (cur, peak) = (CUR.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed));
            eprintln!("sample t={:.0}s live={} MB peak={} MB", t0.elapsed().as_secs_f64(), cur >> 20, peak >> 20);
        });
    }
    let args: Vec<String> = std::env::args().collect();
    let op = args[1].as_str();
    if op == "count-check" {
        // `jpeg::count::coefficient_shape` against a parse, for each JPEG.
        for f in &args[2..] {
            let bytes = std::fs::read(f).unwrap();
            let Ok(img) = phasm_core::JpegImage::from_bytes(&bytes) else {
                println!("{f}: no parse");
                continue;
            };
            let parsed = phasm_core::stego::capacity::count_nonzero_ac(img.dct_grid(0));
            match phasm_core::jpeg::count::coefficient_shape(&bytes) {
                Ok(s) => println!("{f}: {} {} (parse {parsed})", if s.luma_nonzero_ac == parsed { "ok" } else { "MISMATCH" }, s.luma_nonzero_ac),
                Err(e) => println!("{f}: count error {e:?} (parse {parsed})"),
            }
        }
        return;
    }
    if op == "peak-model" {
        // `ghost_encode_peak_bytes` for the apps' checks: Deep Cover, a 1 kB
        // message (31 B compressed), at RAYON_NUM_THREADS workers and the
        // APP_BUDGET_MB budget (default 1500), for each <w>x<h> argument.
        let budget_mb: usize = std::env::var("APP_BUDGET_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
        phasm_core::set_memory_budget(Some(budget_mb << 20));
        for a in &args[2..] {
            let (w, h) = a.split_once('x').expect("<w>x<h>");
            let (w, h) = (w.parse().unwrap(), h.parse().unwrap());
            let peak = phasm_core::ghost_encode_peak_bytes(w, h, true, 0, 31);
            println!("{a}: {} MB ({} workers, budget {budget_mb} MB)", mb(peak), phasm_core::stego::memory::encode_workers());
        }
        return;
    }
    let bytes = std::fs::read(&args[2]).unwrap();
    let pass = args.get(3).map(String::as_str).unwrap_or("wrongpass");
    let reps: usize = std::env::var("MEM_PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let log_path = std::env::var_os("MEM_PROBE_LOG");
    LOG_ON.store(log_path.is_some(), Ordering::Relaxed);
    let base = CUR.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let fp0 = footprint();
    let cpu0 = cpu_seconds();
    let t = std::time::Instant::now();
    if std::env::var_os("MEM_PROBE_TRACE").is_some() {
        // Heap and footprint every MEM_PROBE_TRACE ms (default 250; pair with
        // PHASM_GHOST_V3_TIMING phases).
        let ms = std::env::var("MEM_PROBE_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(ms));
                let heap = mb(CUR.load(Ordering::Relaxed));
                eprintln!("trace {:.2}s heap {heap} MB footprint {} MB", t.elapsed().as_secs_f64(), mb(footprint().0));
            }
        });
    }
    let mut r = String::new();
    for _ in 0..reps {
        r = match op {
            "parse" => {
                let img = phasm_core::JpegImage::from_bytes(&bytes).unwrap();
                format!("{}x{}", img.frame_info().width, img.frame_info().height)
            }
            "capacity" => {
                let img = phasm_core::JpegImage::from_bytes(&bytes).unwrap();
                format!("ghost capacity {} B", phasm_core::ghost_capacity(&img).unwrap())
            }
            "decode" => format!("{:?}", phasm_core::smart_decode(&bytes, pass).map(|_| ()).err()),
            "shape" => {
                // `decode_peak_bytes` and the count it reads (without a parse).
                let s = phasm_core::jpeg::count::coefficient_shape(&bytes).unwrap();
                format!(
                    "{}x{}: {} coefficients, luma {} ({} non-zero AC); decode_peak_bytes {} MB",
                    s.frame.width,
                    s.frame.height,
                    s.coefficients,
                    s.luma_coefficients,
                    s.luma_nonzero_ac,
                    mb(phasm_core::decode_peak_bytes(&bytes))
                )
            }
            "ghost" => format!("{:?}", phasm_core::ghost_decode(&bytes, pass).map(|_| ()).err()),
            "armor" => format!("{:?}", phasm_core::armor_decode(&bytes, pass).map(|_| ()).err()),
            "shadow" => format!("{:?}", phasm_core::ghost_shadow_decode(&bytes, pass).map(|_| ()).err()),
            "cost-strips" => {
                // The separable cost pass alone (strip working set).
                let img = phasm_core::JpegImage::from_bytes(&bytes).unwrap();
                let qt = img.quant_table(img.frame_info().components[0].quant_table_id as usize).unwrap();
                let grid = img.dct_grid(0);
                let rows = phasm_core::stego::cost::uniward::cost_strip_rows(grid.blocks_wide());
                let base = CUR.load(Ordering::Relaxed);
                PEAK.store(base, Ordering::Relaxed);
                phasm_core::stego::cost::uniward::ac_cost_strips(grid, qt, rows, |_, _| Ok(())).unwrap();
                format!("{rows} block rows per strip, strip working set {} MB", mb(PEAK.load(Ordering::Relaxed) - base))
            }
            "shadow-v3" | "shadow-legacy" | "legacy-costs" => {
                let img = phasm_core::JpegImage::from_bytes(&bytes).unwrap();
                use phasm_core::stego::shadow::{shadow_extract_legacy, shadow_extract_v3};
                match op {
                    "shadow-v3" => format!("{:?}", shadow_extract_v3(&img, pass).map(|_| ()).err()),
                    "shadow-legacy" => format!("{:?}", shadow_extract_legacy(&img, pass).map(|_| ()).err()),
                    _ => {
                        let qt = img.quant_table(img.frame_info().components[0].quant_table_id as usize).unwrap();
                        let mut p = phasm_core::stego::cost::uniward::compute_positions_streaming_legacy_v1(img.dct_grid(0), qt, None).unwrap();
                        p.sort_by(|a, b| a.cost.total_cmp(&b.cost));
                        format!("{} positions", p.len())
                    }
                }
            }
            "build-cover" => {
                // `build_cover` as the bridges call it: <raw-rgb-file> as the
                // source pixels (a PNG, so core writes the cover), Ghost,
                // original size. args: width height. Measures core's peak
                // with the caller's pixels already resident (`bytes`).
                let (w, h): (u32, u32) = (args[4].parse().unwrap(), args[5].parse().unwrap());
                let req = phasm_core::CoverRequest {
                    mode: phasm_core::CoverMode::Ghost,
                    wand: true,
                    ghost_quality: None,
                    ghost_max_side: None,
                    armor_preset: phasm_core::ArmorCoverPreset::Standard,
                };
                let png = b"\x89PNG\r\n\x1a\n".to_vec();
                let out = phasm_core::build_cover(&png, Some((&bytes, w, h)), &req).unwrap();
                format!("cover {} B, returned si copy {} B, deep_cover {}", out.jpeg.len(), out.si_pixels.as_ref().map_or(0, |v| v.len()), out.deep_cover)
            }
            "app-flow" => app_flow(&bytes, pass, args.get(4)),
            "model-check" => model_check(&args[2..]),
            "to-rgb" => {
                // <jpeg|avif> <out.rgb>: raw RGB of the image for `encode-si`; a
                // JPEG's decode is cropped by (3, 5) pixels so that it is not a
                // JPEG decoded on its block grid (which gets no side information).
                let (rgb, w, h) = if bytes.starts_with(&[0xFF, 0xD8]) {
                    let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
                    let (w0, h0) = img.dimensions();
                    let mut px = Vec::with_capacity(((w0 - 3) * (h0 - 5) * 3) as usize);
                    for y in 5..h0 {
                        for x in 3..w0 {
                            px.extend_from_slice(&img.get_pixel(x, y).0);
                        }
                    }
                    (px, w0 - 3, h0 - 5)
                } else {
                    avif_rgb(&bytes)
                };
                std::fs::write(&args[3], &rgb).unwrap();
                format!("{w} {h}")
            }
            "encode-si" => {
                phasm_core::set_memory_budget(Some(1674 << 20));
                let msg = std::fs::read_to_string(&args[4]).unwrap();
                let (w, h): (u32, u32) = (args[5].parse().unwrap(), args[6].parse().unwrap());
                let cover = phasm_core::deep_cover_jpeg(&bytes, w, h, 92).unwrap();
                let out = phasm_core::ghost_encode_si(&cover, &bytes, w, h, &msg, pass).unwrap();
                if let Some(path) = std::env::var_os("MEM_PROBE_OUT") {
                    std::fs::write(path, &out).unwrap();
                }
                format!("{} B", out.len())
            }
            "encode" | "encode-budget" | "encode-shadows" => {
                if op != "encode" {
                    phasm_core::set_memory_budget(Some(1674 << 20));
                }
                let msg = std::fs::read_to_string(&args[4]).unwrap();
                let out = if op == "encode-shadows" {
                    // Two shadows of 60 and 100 bytes, as the device bench does.
                    let shadows: Vec<phasm_core::ShadowLayer> = (0..2)
                        .map(|k| phasm_core::ShadowLayer {
                            message: format!("shadow {k} {}", "s".repeat(60 + 40 * k)),
                            passphrase: format!("phasm-shadow-{k}"),
                            files: vec![],
                        })
                        .collect();
                    phasm_core::ghost_encode_with_shadows(&bytes, &msg, &[], pass, &shadows, None).unwrap()
                } else {
                    phasm_core::ghost_encode(&bytes, &msg, pass).unwrap()
                };
                if let Some(path) = std::env::var_os("MEM_PROBE_OUT") {
                    std::fs::write(path, &out).unwrap();
                }
                format!("{} B", out.len())
            }
            _ => panic!("op"),
        };
    }
    let cpu1 = cpu_seconds();
    let (wall, cpu, faults) = (t.elapsed().as_secs_f64(), cpu1.0 - cpu0.0, cpu1.1 - cpu0.1);
    let fp1 = footprint();
    if let Some(path) = log_path {
        LOG_ON.store(false, Ordering::Relaxed);
        let n = LOG_N.load(Ordering::Relaxed).min(LOG_CAP);
        let mut txt = String::new();
        for i in 0..n {
            let (op, size, p) = unsafe { LOG[i] };
            txt.push_str(&format!("{op} {size} {p:x}\n"));
        }
        std::fs::write(path, txt).unwrap();
    }
    println!(
        "{op}: peak heap {} MB (start {} MB), footprint peak {} MB (start {}, end {}), {:.2} s, CPU {:.2} s, {}k page faults, {r}",
        mb(PEAK.load(Ordering::Relaxed) - base),
        mb(base),
        mb(fp1.1),
        mb(fp0.0),
        mb(fp1.0),
        wall,
        cpu,
        faults / 1000
    );
}

/// `model-check`: Ghost encodes of each cover, their whole peak heap (from
/// just before the call: parse, side information, payload, keys, embedding,
/// write) against `ghost_encode_peak_bytes` (public: any subsampling) and
/// the plan's own estimate (telemetry). `MC_CASES`: comma list of
/// `shadows:message:budget-MB` (shadows `n` = n shadows of 20-100
/// characters, `bigN` = N shadows near the primary's size; message `short`
/// = 40 characters, `medium` = max/8, `max` = 0.9 max, `c<N>`; budget
/// `none` or MB); default the reviewer's under-prediction shapes.
/// `MC_SI=<rgb-file>:<w>x<h>`: Deep Cover encodes with those pixels.
fn model_check(covers: &[String]) -> String {
    use std::sync::atomic::AtomicU32;
    static PRED_MB: AtomicU32 = AtomicU32::new(0);
    phasm_core::memory::set_telemetry_hook(Some(Box::new(|e| {
        if let phasm_core::memory::TelemetryEvent::EncodeStarted { predicted_peak_mb, .. } = e {
            PRED_MB.store(*predicted_peak_mb, Ordering::Relaxed);
        }
    })));
    let text = |n: usize, seed: u32| -> String {
        let mut s = 0x9E37_79B9u32 ^ seed.wrapping_mul(0x85EB_CA6B);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (b'!' + (s >> 26) as u8) as char
            })
            .collect()
    };
    let cases: Vec<String> = std::env::var("MC_CASES")
        .unwrap_or_else(|_| "0:short:none,0:medium:1,0:max:1,0:max:none,2:short:none,big2:medium:1".into())
        .split(',')
        .map(String::from)
        .collect();
    let si: Option<(Vec<u8>, u32, u32)> = std::env::var("MC_SI").ok().map(|v| {
        let (f, d) = v.split_once(':').expect("MC_SI=<rgb>:<w>x<h>");
        let (w, h) = d.split_once('x').expect("MC_SI=<rgb>:<w>x<h>");
        (std::fs::read(f).unwrap(), w.parse().unwrap(), h.parse().unwrap())
    });
    let mut out = String::new();
    for path in covers {
        let cover = std::fs::read(path).unwrap();
        let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
        let img = phasm_core::JpegImage::from_bytes(&cover).unwrap();
        let (w, h) = (img.frame_info().width as u32, img.frame_info().height as u32);
        let bands = |img: &phasm_core::JpegImage, ns: usize, st: usize| match &si {
            Some((px, pw, ph)) => phasm_core::ghost_bands_si(img, px, *pw, *ph, ns, st),
            None => phasm_core::ghost_bands(img, ns, st),
        };
        let max0 = bands(&img, 0, 0).map(|b| b.max_bytes).unwrap_or(100);
        for case in &cases {
            let p: Vec<&str> = case.split(':').collect();
            let (big, ns) = match p[0].strip_prefix("big") {
                Some(n) => (true, n.parse::<usize>().unwrap_or(1)),
                None => (false, p[0].parse::<usize>().unwrap()),
            };
            let shadows: Vec<phasm_core::ShadowLayer> = (0..ns)
                .map(|k| phasm_core::ShadowLayer {
                    message: if big { text(max0 / (2 * ns + 4) * 100 / 76 + 50, 99 + k as u32) } else { text(100 - 80 * k / ns.max(1), 77 + k as u32) },
                    passphrase: format!("mc-shadow-{k}"),
                    files: vec![],
                })
                .collect();
            let st: usize = shadows.iter().map(|s| phasm_core::compressed_payload_size(&s.message, &[])).sum();
            let mb = bands(&img, ns, st).map(|b| b.max_bytes).unwrap_or(100);
            let chars = match p[1] {
                "short" => 40,
                "medium" => mb / 8 * 100 / 76,
                "max" => mb * 90 / 76,
                m => m.strip_prefix('c').and_then(|n| n.parse().ok()).expect("message"),
            };
            let msg = text(chars.max(1), 1);
            let payload = phasm_core::compressed_payload_size(&msg, &[]) + st;
            phasm_core::set_memory_budget(if p[2] == "none" { None } else { Some(p[2].parse::<usize>().unwrap() << 20) });
            let public = phasm_core::ghost_encode_peak_bytes(w, h, si.is_some(), ns, payload);
            PRED_MB.store(0, Ordering::Relaxed);
            let base = CUR.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            let r = match (&si, ns) {
                (None, 0) => phasm_core::ghost_encode(&cover, &msg, "mc-pass"),
                (None, _) => phasm_core::ghost_encode_with_shadows(&cover, &msg, &[], "mc-pass", &shadows, None),
                (Some((px, pw, ph)), 0) => phasm_core::ghost_encode_si(&cover, px, *pw, *ph, &msg, "mc-pass"),
                (Some((px, pw, ph)), _) => phasm_core::ghost_encode_si_with_shadows(&cover, px, *pw, *ph, &msg, &[], "mc-pass", &shadows),
            };
            let peak = PEAK.load(Ordering::Relaxed) - base;
            let pred = PRED_MB.load(Ordering::Relaxed) as usize;
            let mib = |b: usize| b as f64 / 1048576.0;
            out += &format!(
                "{name} {w}x{h} {case} payload {payload} B: peak {:.1} MiB, public {:.1} ({:+.1} %), plan {pred} ({:+.1} %){}\n",
                mib(peak),
                mib(public),
                (public as f64 / peak as f64 - 1.0) * 100.0,
                (pred as f64 * 1048576.0 / peak as f64 - 1.0) * 100.0,
                r.err().map_or(String::new(), |e| format!(" {e:?}"))
            );
        }
    }
    phasm_core::set_memory_budget(None);
    out
}

/// One stage's numbers: the Rust heap's peak during it (with the part that
/// stands for Java `ByteArray`s), what stays live after it, the footprint.
fn stage_line(name: &str, base_java: usize, peak: usize, java_at_end: usize) -> String {
    let fp = footprint();
    format!(
        "  {name:<14} heap peak {:>5} MB (java {:>4}, native {:>5}) | live after {:>5} MB (java {:>4}, native {:>5}) | footprint {:>5} MB, peak {:>5} MB\n",
        mb(peak),
        mb(base_java),
        mb(peak.saturating_sub(base_java)),
        mb(CUR.load(Ordering::Relaxed)),
        mb(java_at_end),
        mb(CUR.load(Ordering::Relaxed).saturating_sub(java_at_end)),
        mb(fp.0),
        mb(fp.1),
    )
}

/// The Android app's image pipeline on one picked AVIF, stage by stage
/// (`docs/design/app/large-samples.md`, "Memory"): core's decode, the HUD
/// preview, the default Ghost cover (realigned, new SI pixels), the Armor
/// cover, the capacity bar's bands and the Deep Cover encode, at the app's
/// memory budget (`APP_BUDGET_MB`, default 1500 = `availMem / 2` of a phone
/// with 3 GB free).
///
/// A JPEG `file` is the platform-decode path: its decode stands for the
/// platform's (`image`, whose buffers are not the app's), then core's covers
/// (the Ghost cover passes the JPEG through, no side information), the bands
/// and a plain Ghost encode.
///
/// `APP_FLOW=heap` replays the app before 2026-11-21: the pixels are Java
/// `ByteArray`s (simulated here with the same allocator and reported as
/// "java"), every JNI call copies them into native memory and the SI pixels
/// come back as a Java copy. `APP_FLOW=native` (default): one native buffer
/// per image, borrowed by every call, the SI pixels stay native. dav1d's own
/// picture buffers (C allocations) show in the footprint, not the heap.
#[cfg(feature = "av1-decoder")]
fn app_flow(file: &[u8], pass: &str, msg_file: Option<&String>) -> String {
    let heap_flow = std::env::var("APP_FLOW").is_ok_and(|v| v == "heap");
    let budget_mb: usize = std::env::var("APP_BUDGET_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
    phasm_core::set_memory_budget(Some(budget_mb << 20));
    // APP_MSG_BYTES=<n>: a message of n incompressible characters (a long
    // message: more lanes); APP_SHADOWS=<k>: k shadow messages of 100 B;
    // APP_PLAIN=1: the encode without the side information (plain Ghost).
    let msg = match (msg_file, std::env::var("APP_MSG_BYTES").ok().and_then(|v| v.parse::<usize>().ok())) {
        (Some(p), _) => std::fs::read_to_string(p).unwrap(),
        (None, Some(n)) => {
            let mut s = 0x9E37_79B9u32;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (b'!' + (s >> 26) as u8) as char
                })
                .collect()
        }
        _ => "phasm native pixels ".repeat(50),
    };
    let n_shadows: usize = std::env::var("APP_SHADOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let shadows: Vec<phasm_core::ShadowLayer> = (0..n_shadows)
        .map(|k| phasm_core::ShadowLayer {
            message: format!("shadow {k} {}", "s".repeat(100)),
            passphrase: format!("phasm-shadow-{k}"),
            files: vec![],
        })
        .collect();
    let mut out = format!(
        "app-flow {} (budget {budget_mb} MB, {} B message, {n_shadows} shadows)\n",
        if heap_flow { "heap" } else { "native" },
        msg.len()
    );
    let begin = || {
        let c = CUR.load(Ordering::Relaxed);
        PEAK.store(c, Ordering::Relaxed);
    };
    let peak = || PEAK.load(Ordering::Relaxed);
    let mut overall = 0usize;

    // 1. Decode (ImageEngine.decodeAvifWithCore). APP_RGB=<w>x<h>: `file` is
    // raw RGB of that size instead (`to-rgb`), standing for a PNG source.
    begin();
    let raw = std::env::var("APP_RGB").ok().map(|v| {
        let (a, b) = v.split_once('x').expect("APP_RGB=<w>x<h>");
        (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())
    });
    let png_tag = b"\x89PNG\r\n\x1a\n".to_vec();
    let (d, file) = match raw {
        Some((w, h)) => (phasm_core::photo::decode::DecodedPhoto { rgb: file.to_vec(), width: w, height: h }, &png_tag[..]),
        None if file.starts_with(&[0xFF, 0xD8]) => {
            let img = image::load_from_memory(file).unwrap().to_rgb8();
            let (w, h) = img.dimensions();
            (phasm_core::photo::decode::DecodedPhoto { rgb: img.into_raw(), width: w, height: h }, file)
        }
        None => (phasm_core::photo::decode::decode_avif_rgb(file, phasm_core::MAX_PIXELS as u64).unwrap(), file),
    };
    let (w, h) = (d.width, d.height);
    let mut java = 0usize;
    // heap: byte_array_from_slice, then the Rust Vec is dropped
    let src: Vec<u8> = if heap_flow {
        let j = d.rgb.clone();
        java += j.len();
        drop(d.rgb);
        j
    } else {
        d.rgb
    };
    overall = overall.max(peak());
    out += &format!("  {w}x{h} = {:.1} MP\n", (w as f64 * h as f64) / 1e6);
    out += &stage_line("decode", 0, peak(), java);
    out += &relief_after("decode");

    // 2. HUD preview (ImageEngine.previewFromRgb): core's resampler to 1600 px.
    begin();
    let s = (1600.0 / w.max(h) as f64).min(1.0);
    let (pw, ph) = (((w as f64 * s) as u32).max(1), ((h as f64 * s) as u32).max(1));
    {
        let copy = heap_flow.then(|| src.clone()); // JNI convert_byte_array
        let px = copy.as_deref().unwrap_or(&src);
        let small = phasm_core::resample_rgb(px, w, h, pw, ph).unwrap();
        std::hint::black_box(&small);
    }
    overall = overall.max(peak());
    out += &stage_line("preview", java, peak(), java);

    // 3. Ghost cover (CoverBuilder.build, default key: wand on).
    let req = |mode| phasm_core::CoverRequest {
        mode,
        wand: true,
        ghost_quality: None,
        ghost_max_side: None,
        armor_preset: phasm_core::ArmorCoverPreset::Standard,
    };
    begin();
    let ghost = {
        let copy = heap_flow.then(|| src.clone());
        let px = copy.as_deref().unwrap_or(&src);
        let mut c = phasm_core::build_cover(file, Some((px, w, h)), &req(phasm_core::CoverMode::Ghost)).unwrap();
        // heap: the SI pixels come back as a Java copy while core's Vec (and
        // the input copy) are still alive
        if heap_flow {
            if let Some(si) = c.si_pixels.take() {
                let j = si.clone();
                java += j.len();
                c.si_pixels = Some(j);
            }
        }
        c
    };
    let java_ghost = java; // the Java SI copy is counted from here on
    overall = overall.max(peak());
    out += &stage_line("ghost-cover", java_ghost, peak(), java);
    out += &relief_after("ghost-cover");
    out += &format!(
        "  ghost cover {} MB JPEG, SI pixels {} (si_is_input {}), deep_cover {}\n",
        mb(ghost.jpeg.len()),
        ghost.si_pixels.as_ref().map_or("none".into(), |v| format!("{} MB", mb(v.len()))),
        ghost.si_is_input,
        ghost.deep_cover
    );
    // None: the JPEG passed through (plain Ghost, no side information).
    let si: Option<&[u8]> = ghost.si_pixels_of(Some(&src));

    // 4. Armor cover (prepare builds it too).
    begin();
    let armor = {
        let copy = heap_flow.then(|| src.clone());
        let px = copy.as_deref().unwrap_or(&src);
        phasm_core::build_cover(file, Some((px, w, h)), &req(phasm_core::CoverMode::Armor)).unwrap()
    };
    std::hint::black_box(&armor);
    overall = overall.max(peak());
    out += &stage_line("armor-cover", java, peak(), java);

    // 5. Capacity bar (ghostBandsAndShadowCapacity).
    begin();
    let bands = {
        let copy = heap_flow.then(|| si.map(<[u8]>::to_vec)).flatten();
        let px = copy.as_deref().or(si);
        let img = phasm_core::JpegImage::from_bytes(&ghost.jpeg).unwrap();
        phasm_core::ghost_bands_and_shadow_capacity(&img, px.map(|p| (p, w, h)), 0, 0).unwrap()
    };
    overall = overall.max(peak());
    out += &stage_line("bands", java, peak(), java);
    out += &format!("  bands max {} B, comfort {} B, shadow max {} B\n", bands.0.max_bytes, bands.0.comfort_bytes, bands.1);

    // 6. Deep Cover encode (ImageEngine.encodeCover -> encodeSi).
    begin();
    let live_before = CUR.load(Ordering::Relaxed);
    let stego = {
        let copy = heap_flow.then(|| si.map(<[u8]>::to_vec)).flatten();
        let px = copy.as_deref().or(si).unwrap_or_default();
        // APP_PLAIN=1: the same cover without side information (plain Ghost).
        let plain = std::env::var_os("APP_PLAIN").is_some() || si.is_none();
        match (plain, shadows.is_empty()) {
            (false, true) => phasm_core::ghost_encode_si_with_quality(&ghost.jpeg, px, w, h, &msg, pass).unwrap().0,
            (false, false) => {
                phasm_core::ghost_encode_si_with_shadows_quality(&ghost.jpeg, px, w, h, &msg, &[], pass, &shadows).unwrap().0
            }
            (true, true) => phasm_core::ghost_encode_with_quality(&ghost.jpeg, &msg, pass).unwrap().0,
            (true, false) => {
                phasm_core::ghost_encode_with_shadows_quality(&ghost.jpeg, &msg, &[], pass, &shadows, None).unwrap().0
            }
        }
    };
    let encode_peak = peak();
    overall = overall.max(encode_peak);
    out += &stage_line("encode", java, encode_peak, java);
    out += &relief_after("encode");
    out += &format!(
        "  encode: core's own peak {} MB = {:.2} B/px (above the {} MB live before it)\n",
        mb(encode_peak - live_before),
        (encode_peak - live_before) as f64 / (w as f64 * h as f64),
        mb(live_before)
    );
    {
        // The model's estimate of this encode (`ghost_encode_peak_bytes`,
        // the cover's own size, the layers' compressed payloads).
        let plain = std::env::var_os("APP_PLAIN").is_some() || si.is_none();
        let (cw, ch) = sof_dims(&ghost.jpeg);
        let payload = phasm_core::compressed_payload_size(&msg, &[])
            + shadows.iter().map(|s| phasm_core::compressed_payload_size(&s.message, &[])).sum::<usize>();
        let predicted = phasm_core::ghost_encode_peak_bytes(cw, ch, !plain, shadows.len(), payload);
        let measured = encode_peak - live_before;
        out += &format!(
            "  model: ghost_encode_peak_bytes({cw}, {ch}, si {}, {} shadows, {payload} B) = {} MB, {:+.1} % vs measured\n",
            !plain,
            shadows.len(),
            mb(predicted),
            (predicted as f64 / measured as f64 - 1.0) * 100.0
        );
    }
    {
        use sha2::{Digest, Sha256};
        let hex: String = Sha256::digest(&stego).iter().map(|b| format!("{b:02x}")).collect();
        out += &format!("  stego {} MB, sha256 {hex}\n", mb(stego.len()));
    }
    if let Some(path) = std::env::var_os("MEM_PROBE_OUT") {
        std::fs::write(path, &stego).unwrap();
    }
    out += &format!("  overall heap peak {} MB = {:.2} B/px", mb(overall), overall as f64 / (w as f64 * h as f64));
    out
}

/// Width and height from a JPEG's SOF segment.
#[cfg(feature = "av1-decoder")]
fn sof_dims(jpeg: &[u8]) -> (u32, u32) {
    let mut i = 2;
    while i + 9 < jpeg.len() {
        let (m, len) = (jpeg[i + 1], u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize);
        if (0xC0..=0xC2).contains(&m) {
            let h = u16::from_be_bytes([jpeg[i + 5], jpeg[i + 6]]) as u32;
            let w = u16::from_be_bytes([jpeg[i + 7], jpeg[i + 8]]) as u32;
            return (w, h);
        }
        i += 2 + len;
    }
    panic!("no SOF")
}

#[cfg(feature = "av1-decoder")]
fn avif_rgb(file: &[u8]) -> (Vec<u8>, u32, u32) {
    let d = phasm_core::photo::decode::decode_avif_rgb(file, phasm_core::MAX_PIXELS as u64).unwrap();
    (d.rgb, d.width, d.height)
}

#[cfg(not(feature = "av1-decoder"))]
fn avif_rgb(_file: &[u8]) -> (Vec<u8>, u32, u32) {
    panic!("AVIF needs --features av1-decoder")
}

#[cfg(not(feature = "av1-decoder"))]
fn app_flow(_file: &[u8], _pass: &str, _msg_file: Option<&String>) -> String {
    panic!("app-flow needs --features av1-decoder")
}
