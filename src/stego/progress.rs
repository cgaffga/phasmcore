// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Progress tracking and cancellation, one set per [`Channel`].
//!
//! Uses atomics so it is safe to call from rayon worker threads.
//! When the `wasm` feature is enabled, an optional JS callback is invoked
//! on each `advance()` to drive a real-time progress bar via Web Worker
//! `postMessage`.

use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use super::error::StegoError;

/// The operations that can run at the same time, each with its own progress
/// counters and cancel flag.
///
/// The mobile apps run an encode (or a capacity probe) and a decode at once,
/// one per tab. With one global set, the decode tab's Cancel stopped the
/// encode tab's STC searches, which read a set flag as "no path" — the encode
/// was refused as too large, the probe under-reported — while a decode
/// starting ([`init`]) silently dropped the encode's pending cancel, and each
/// overlay showed the other's progress.
///
/// Code belongs to [`Channel::Encode`] unless it runs inside [`run_decode`],
/// so every caller that never asks — the CLI, WASM, the tests, every encode
/// and probe — behaves exactly as before. The module-level functions below
/// all act on the calling code's channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// Encodes, capacity probes, and everything not run through [`run_decode`].
    Encode = 0,
    /// Decodes the mobile bridges run through [`run_decode`].
    Decode = 1,
}

struct State {
    step: AtomicU32,
    total: AtomicU32,
    cancelled: AtomicBool,
    /// The cancel flag belongs to the caller; see [`Channel::hand_over`].
    handed_over: AtomicBool,
}

impl State {
    const fn new() -> Self {
        Self {
            step: AtomicU32::new(0),
            total: AtomicU32::new(0),
            cancelled: AtomicBool::new(false),
            handed_over: AtomicBool::new(false),
        }
    }
}

static STATES: [State; 2] = [State::new(), State::new()];

thread_local! {
    /// The channel of code running on this thread outside the decode pool:
    /// [`run_decode`]'s inline fallback sets it for the duration of `f`.
    static INLINE_CHANNEL: Cell<Channel> = const { Cell::new(Channel::Encode) };
}

/// The pool decode work runs on (`None` if it could not be built).
///
/// Its own pool, not a marker on the calling thread, because decode code
/// fans out with rayon — `smart_decode`'s parallel arms, the Armor and Ghost
/// candidate searches, the shadow brute force — and a worker of a SHARED pool
/// would run encode and decode jobs interleaved (work stealing), with no way
/// to tell which operation it is serving. Workers of this pool only ever run
/// decode jobs, so membership IS the channel, for every parallel region
/// present and future.
#[cfg(feature = "parallel")]
static DECODE_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();

impl Channel {
    /// The channel of the calling code.
    pub fn current() -> Channel {
        #[cfg(feature = "parallel")]
        if let Some(Some(pool)) = DECODE_POOL.get()
            && pool.current_thread_index().is_some()
        {
            return Channel::Decode;
        }
        INLINE_CHANNEL.with(Cell::get)
    }

    fn state(self) -> &'static State {
        &STATES[self as usize]
    }

    /// Request cancellation of this channel's operation.
    pub fn cancel(self) {
        self.state().cancelled.store(true, Ordering::Relaxed);
    }

    /// Clear this channel's cancellation request.
    pub fn clear_cancel(self) {
        self.state().cancelled.store(false, Ordering::Relaxed);
    }

    /// Whether this channel's operation was asked to stop.
    pub fn is_cancelled(self) -> bool {
        self.state().cancelled.load(Ordering::Relaxed)
    }

    /// This channel's `(step, total)`.
    pub fn get(self) -> (u32, u32) {
        let st = self.state();
        (st.step.load(Ordering::Relaxed), st.total.load(Ordering::Relaxed))
    }

    /// Hand this channel's cancel flag to the caller: from now on [`init`]
    /// leaves it alone, and only [`Channel::clear_cancel`] clears it.
    ///
    /// `init` clearing the flag suits a caller that never cancels mid-way
    /// (the CLI) or starts each operation fresh (WASM). It drops a cancel
    /// that lands between an app starting an operation and the pipeline
    /// reaching `init`, and one that lands before a pipeline re-inits for a
    /// retry (the Ghost shadow memory rungs). The mobile apps clear the flag
    /// themselves when each operation starts and when it ends, so their
    /// bridges hand both channels over on the first cancel / clear call.
    pub fn hand_over(self) {
        self.state().handed_over.store(true, Ordering::Relaxed);
    }
}

/// Run `f` as decode work: its progress and cancel flag are the
/// [`Channel::Decode`] ones, apart from whatever encode or probe runs
/// meanwhile.
///
/// `f` runs on the decode pool (the calling thread blocks until it returns),
/// so every parallel region inside it stays on that pool. Without the
/// `parallel` feature, or if the pool could not be built, `f` runs inline
/// with the calling thread marked as decode — correct for everything `f`
/// does on this thread, which without rayon is all of it.
pub fn run_decode<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    {
        let pool = DECODE_POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .thread_name(|i| format!("phasm-decode-{i}"))
                .build()
                .ok()
        });
        if let Some(pool) = pool {
            return pool.install(f);
        }
    }
    struct Restore(Channel);
    impl Drop for Restore {
        fn drop(&mut self) {
            INLINE_CHANNEL.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(INLINE_CHANNEL.with(|c| c.replace(Channel::Decode)));
    f()
}

fn state() -> &'static State {
    Channel::current().state()
}

/// Reset progress to 0 and set the total step count.
/// Also resets the cancellation flag so a fresh operation starts clean —
/// unless the channel was handed over ([`Channel::hand_over`]).
pub fn init(total: u32) {
    let st = state();
    if !st.handed_over.load(Ordering::Relaxed) {
        st.cancelled.store(false, Ordering::Relaxed);
    }
    st.step.store(0, Ordering::Relaxed);
    st.total.store(total, Ordering::Relaxed);
    notify();
}

/// Set (or update) the total without resetting the current step.
/// Used by pipeline code that discovers the real total mid-flight
/// (e.g. after counting delta candidates).
pub fn set_total(total: u32) {
    state().total.store(total, Ordering::Relaxed);
    notify();
}

/// Request cancellation of the current operation (this channel's).
///
/// The pipelines check this flag at natural loop boundaries and return
/// `Err(StegoError::Cancelled)` when set.
pub fn cancel() {
    Channel::current().cancel();
}

/// Clear a cancellation request without touching the progress counters.
///
/// [`init`] clears the flag, but only the IMAGE pipelines call it. The flag
/// is also read by code that video encode, probe and decode share —
/// `stc_embed` / `stc_is_feasible` treat a set flag as "no path" — and
/// nothing on those paths ever cleared it. So one cancelled image operation,
/// or a cancelled video encode, left the flag set, and every later video
/// operation found no STC path for any GOP: refused as too large, or a zero
/// capacity, until the next image operation happened to reset it. Callers
/// starting an operation that does not go through [`init`], and callers
/// whose cancelled operation has finished unwinding, call this.
pub fn clear_cancel() {
    Channel::current().clear_cancel();
}

/// Returns `true` if cancellation has been requested.
pub fn is_cancelled() -> bool {
    Channel::current().is_cancelled()
}

/// Check for cancellation and return an error if requested.
///
/// Call this at natural loop boundaries in the pipelines to allow early
/// termination without waiting for the full operation to complete.
pub fn check_cancelled() -> Result<(), StegoError> {
    if is_cancelled() {
        Err(StegoError::Cancelled)
    } else {
        Ok(())
    }
}

/// Advance progress by one step and notify the callback (if set).
/// Step is capped at total to avoid displaying values like "84/15".
/// When total is 0 (indeterminate), step still advances freely so that
/// the UI can show activity; `init()` with the real total will follow.
pub fn advance() {
    let st = state();
    let total = st.total.load(Ordering::Relaxed);
    if total == 0 {
        // Indeterminate phase — advance freely, UI should not display X/Y yet.
        st.step.fetch_add(1, Ordering::Relaxed);
    } else {
        // Cap at total-1 so the bar never hits 100% before finish().
        let _ = st.step.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
            if s + 1 < total { Some(s + 1) } else { Some(s) }
        });
    }
    notify();
}

/// Read the current (step, total) progress.
pub fn get() -> (u32, u32) {
    Channel::current().get()
}

/// Advance progress by `n` steps.  Convenience wrapper that calls
/// [`advance`] in a loop.
pub fn advance_by(n: u32) {
    for _ in 0..n {
        advance();
    }
}

/// Mark progress as complete (step = total) and notify.
pub fn finish() {
    let st = state();
    st.step.store(st.total.load(Ordering::Relaxed), Ordering::Relaxed);
    notify();
}

// ---------------------------------------------------------------------------
// WASM callback (only compiled with the `wasm` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
mod wasm_cb {
    use std::cell::RefCell;

    thread_local! {
        static CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
    }

    pub fn set(cb: Option<js_sys::Function>) {
        CALLBACK.with(|c: &RefCell<Option<js_sys::Function>>| *c.borrow_mut() = cb);
    }

    pub fn notify(step: u32, total: u32) {
        CALLBACK.with(|c: &RefCell<Option<js_sys::Function>>| {
            if let Some(ref f) = *c.borrow() {
                let _ = f.call2(
                    &wasm_bindgen::JsValue::NULL,
                    &wasm_bindgen::JsValue::from(step),
                    &wasm_bindgen::JsValue::from(total),
                );
            }
        });
    }
}

/// Set (or clear) the WASM progress callback. Only available with the `wasm` feature.
#[cfg(feature = "wasm")]
pub fn set_wasm_callback(cb: Option<js_sys::Function>) {
    wasm_cb::set(cb);
}

#[cfg(feature = "wasm")]
fn notify() {
    let (s, t) = get();
    wasm_cb::notify(s, t);
}

#[cfg(not(feature = "wasm"))]
fn notify() {
    // No-op on native — iOS/Android poll via FFI.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_flag_propagates() {
        // On the decode channel: the lib's other unit tests run in parallel
        // with this one on the encode channel, and its STC searches would
        // read this test's cancel as "no path".
        run_decode(cancel_flag_propagates_body);
    }

    fn cancel_flag_propagates_body() {
        // Reset state
        init(10);
        assert!(!is_cancelled());
        assert!(check_cancelled().is_ok());

        // Request cancellation
        cancel();
        assert!(is_cancelled());

        // check_cancelled should return Err(Cancelled)
        let err = check_cancelled().unwrap_err();
        assert!(
            matches!(err, StegoError::Cancelled),
            "expected Cancelled, got {err:?}"
        );

        // Reset clears the cancel flag
        init(5);
        assert!(!is_cancelled());
        assert!(check_cancelled().is_ok());
    }
}
