// Peak heap of phasm-core operations, measured with a counting allocator.
//
// Usage:
//   cargo run --release --example mem_probe -- <op> <image> [passphrase] [message-file]
//   op: parse | capacity | decode | ghost | shadow | encode | encode-budget
// `encode-budget` sets a 1.67 GB memory budget first (what cgPhone reports).
//   op encode-si <raw-rgb-file> <pass> <message-file> <width> <height>: Deep
//   Cover as the apps do it (cover written by `deep_cover_jpeg` at QF92, then
//   `ghost_encode_si` with the same pixels), at the app budget.
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
        let q = unsafe { inner().realloc(p, l, n) };
        log(1, n, q);
        if !q.is_null() {
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
    let args: Vec<String> = std::env::args().collect();
    let op = args[1].as_str();
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
