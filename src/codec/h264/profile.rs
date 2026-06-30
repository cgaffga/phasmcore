//! `PHASM_PROFILE`-gated wall-clock profiling for the H.264 encode paths.
//!
//! Zero effect on output bytes — it only reads the monotonic clock and
//! accumulates `(call_count, total_duration)` per static label. When the
//! `PHASM_PROFILE` env var is unset (or `0`/`false`) every entry point is a
//! cheap no-op. `dump()` prints the accumulated table to stderr sorted by
//! total time, then clears it so a second encode in the same process starts
//! fresh.
//!
//! Labels follow a convention so the summary is readable:
//!   - `phase.*`  — a top-level pass / phase (INCLUSIVE: contains primitives)
//!   - everything else — a cumulative primitive across ALL passes
//!     (`oh264.encode_once` = the OpenH264 C encoder, `cover_walk` = the
//!     pure-Rust CABAC cover walker, `cost.combine_cover` = cost build).
//!
//! Purely a profiling aid; never gate production logic on it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

fn enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("PHASM_PROFILE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

#[allow(clippy::type_complexity)]
fn store() -> &'static Mutex<HashMap<&'static str, (u64, Duration)>> {
    static S: OnceLock<Mutex<HashMap<&'static str, (u64, Duration)>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Add one timing sample under `label`. No-op when profiling is disabled.
pub fn record(label: &'static str, dur: Duration) {
    if !enabled() {
        return;
    }
    if let Ok(mut m) = store().lock() {
        let e = m.entry(label).or_insert((0, Duration::ZERO));
        e.0 += 1;
        e.1 += dur;
    }
}

/// RAII timer: records elapsed under `label` when it drops. Use
/// `let _p = profile::scope("…");` at the top of a block.
pub struct Scope {
    label: &'static str,
    start: Instant,
}

#[must_use]
pub fn scope(label: &'static str) -> Scope {
    Scope { label, start: Instant::now() }
}

impl Drop for Scope {
    fn drop(&mut self) {
        record(self.label, self.start.elapsed());
    }
}

/// Print the accumulated table to stderr (sorted by total time, desc) and
/// reset it. No-op when profiling is disabled.
pub fn dump(header: &str) {
    if !enabled() {
        return;
    }
    let mut m = match store().lock() {
        Ok(m) => m,
        Err(_) => return,
    };
    let mut rows: Vec<(&'static str, u64, Duration)> =
        m.iter().map(|(k, (c, d))| (*k, *c, *d)).collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    eprintln!("\n=== PHASM_PROFILE: {header} ===");
    eprintln!(
        "  {:<26} {:>7} {:>11} {:>11}",
        "label", "calls", "total_ms", "ms/call"
    );
    for (k, c, d) in &rows {
        let ms = d.as_secs_f64() * 1000.0;
        eprintln!(
            "  {:<26} {:>7} {:>11.1} {:>11.3}",
            k,
            c,
            ms,
            ms / (*c as f64).max(1.0)
        );
    }
    eprintln!("  (phase.* = inclusive pass time; others = cumulative primitive across all passes)\n");
    m.clear();
}
