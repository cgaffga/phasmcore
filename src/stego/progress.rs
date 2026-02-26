//! Global decode progress tracking.
//!
//! Uses atomics so it is safe to call from rayon worker threads.
//! When the `wasm` feature is enabled, an optional JS callback is invoked
//! on each `advance()` to drive a real-time progress bar via Web Worker
//! `postMessage`.

use core::sync::atomic::{AtomicU32, Ordering};

static STEP: AtomicU32 = AtomicU32::new(0);
static TOTAL: AtomicU32 = AtomicU32::new(0);

/// Reset progress to 0 and set the total step count.
pub fn init(total: u32) {
    STEP.store(0, Ordering::Relaxed);
    TOTAL.store(total, Ordering::Relaxed);
    notify();
}

/// Advance progress by one step and notify the callback (if set).
pub fn advance() {
    STEP.fetch_add(1, Ordering::Relaxed);
    notify();
}

/// Read the current (step, total) progress.
pub fn get() -> (u32, u32) {
    (STEP.load(Ordering::Relaxed), TOTAL.load(Ordering::Relaxed))
}

/// Mark progress as complete (step = total) and notify.
pub fn finish() {
    let t = TOTAL.load(Ordering::Relaxed);
    STEP.store(t, Ordering::Relaxed);
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
