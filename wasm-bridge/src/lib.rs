// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WASM bridge for phasm-core.
//!
//! This crate exposes phasm-core's steganographic encode, decode, and capacity
//! functions to JavaScript via wasm-bindgen. It is the simplest of the three
//! platform bridges because wasm-bindgen handles all memory marshalling
//! automatically: `&[u8]` maps to `Uint8Array`, `String` maps to JS string,
//! and `Result<T, JsError>` maps to a thrown exception on the JS side.
//!
//! # Error convention
//!
//! All errors are returned as `JsError` with a message in the format
//! `"ERROR_CODE:Human-readable detail"`. The JavaScript caller can split on
//! the first colon to obtain a machine-readable code and a display message.
//!
//! # Mode values
//!
//! - `1` (or any value other than 2) = Ghost (stealth, resists steganalysis)
//! - `2` = Armor (robust, survives recompression)

use wasm_bindgen::prelude::*;

use phasm_core::{
    armor_capacity, armor_capacity_info, armor_decode, estimate_shadow_capacity_si,
    ghost_capacity, ghost_capacity_si, ghost_capacity_with_shadows, ghost_decode, ghost_shadow_decode,
    ghost_encode, ghost_encode_si, ghost_encode_si_with_files,
    ghost_encode_with_files,
    deep_cover_jpeg as core_deep_cover_jpeg,
    smart_decode, DecodeQuality, FileEntry, JpegImage, PayloadData, StegoError,
    GHOST_DECODE_STEPS,
};

const ALLOWED_HOSTS: &[&str] = &["phasm.app", "phasm.link", "localhost", "127.0.0.1"];

/// Browser tab memory ceiling for the rung selector. 800 MB chosen as
/// a compromise: Safari mobile gets the most aggressive degradation
/// (~1.5 GB tab limit on iPhone with browser overhead); Chrome/Firefox
/// desktop have more headroom but extreme-MP cases still benefit
/// from the streaming ladder. See
/// `docs/design/image/memory-budget-2026-05.md` § 8 Phase 3.
const WASM_DEFAULT_MEMORY_BUDGET_BYTES: usize = 800 * 1024 * 1024;

/// Initialization hook — wasm-bindgen invokes this automatically on
/// module load. Sets the default 800 MB memory budget so the rung
/// selector kicks in on big images without the JS caller having to
/// remember.
#[wasm_bindgen(start)]
pub fn wasm_init() {
    phasm_core::set_memory_budget(Some(WASM_DEFAULT_MEMORY_BUDGET_BYTES));
}

/// Override the default memory budget. Pass `0` to disable.
///
/// JS callers can use this when running on a host they know has
/// different memory characteristics — but the default 800 MB is
/// suitable for production browser tabs.
#[wasm_bindgen]
pub fn set_memory_budget(bytes: u64) {
    let opt = if bytes == 0 { None } else { Some(bytes as usize) };
    phasm_core::set_memory_budget(opt);
}

fn check_domain() -> Result<(), JsError> {
    let hostname = js_sys::eval("globalThis.location?.hostname || ''")
        .unwrap_or(JsValue::from_str(""))
        .as_string()
        .unwrap_or_default();
    if hostname.is_empty() || ALLOWED_HOSTS.iter().any(|h| *h == hostname) {
        Ok(())
    } else {
        Err(JsError::new("UNAUTHORIZED:This build is licensed to phasm.app"))
    }
}

/// Format a `StegoError` as `"CODE:detail"` for cross-FFI error reporting.
///
/// The error code is a stable, machine-readable identifier (e.g. `INVALID_JPEG`,
/// `MESSAGE_TOO_LARGE`). The detail string is a human-readable explanation.
fn stego_error_message(err: &StegoError) -> String {
    let code = match err {
        StegoError::InvalidJpeg(_) | StegoError::NoLuminanceChannel => "INVALID_JPEG",
        StegoError::ImageTooSmall => "IMAGE_TOO_SMALL",
        StegoError::ImageTooLarge => "IMAGE_TOO_LARGE",
        StegoError::MessageTooLarge => "MESSAGE_TOO_LARGE",
        StegoError::FrameCorrupted => "FRAME_CORRUPTED",
        StegoError::DecryptionFailed => "DECRYPTION_FAILED",
        StegoError::InvalidUtf8 => "INVALID_UTF8",
        StegoError::Cancelled => "CANCELLED",
        StegoError::KeyDerivationFailed => "KEY_DERIVATION_FAILED",
        StegoError::DuplicatePassphrase => "DUPLICATE_PASSPHRASE",
        StegoError::InvalidVideo(_) => "INVALID_VIDEO",
        StegoError::ShadowEmbedFailed => "SHADOW_EMBED_FAILED",
    };
    let detail = match err {
        StegoError::InvalidJpeg(e) => format!("{e:?}"),
        StegoError::ImageTooSmall => "Image too small for steganography".into(),
        StegoError::ImageTooLarge => "Image too large (max 16384px / 200MP)".into(),
        StegoError::MessageTooLarge => "Message too large for this image".into(),
        StegoError::FrameCorrupted => "No hidden message found or data corrupted".into(),
        StegoError::DecryptionFailed => "Wrong passphrase or no hidden message".into(),
        StegoError::InvalidUtf8 => "Decoded text is not valid UTF-8".into(),
        StegoError::NoLuminanceChannel => "Image has no luminance channel".into(),
        StegoError::Cancelled => "Operation cancelled".into(),
        StegoError::KeyDerivationFailed => "Key derivation failed".into(),
        StegoError::DuplicatePassphrase => {
            "Each layer must use a unique passphrase".into()
        }
        StegoError::InvalidVideo(s) => format!("Invalid video: {s}"),
        StegoError::ShadowEmbedFailed => {
            "Shadow embed failed: try fewer/shorter shadow messages \
             or a different passphrase".into()
        }
    };
    format!("{code}:{detail}")
}

// ---------------------------------------------------------------------------
// Base64 encode/decode (standard alphabet, with padding)
// ---------------------------------------------------------------------------

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, JsError> {
    fn val(c: u8) -> Result<u32, JsError> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(JsError::new("INVALID_BASE64:Invalid character in base64 input")),
        }
    }

    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return Err(JsError::new("INVALID_BASE64:Base64 input length not a multiple of 4"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let triple = (a << 18) | (b << 12) | (c << 6) | d;
        out.push(((triple >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            out.push((triple & 0xFF) as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Minimal JSON parsing helpers (no serde)
// ---------------------------------------------------------------------------

/// Parse a JSON array of file objects: `[{"filename":"...", "content":"<base64>"}]`
///
/// Returns a Vec<FileEntry> or a JsError. Handles only the exact shape we need;
/// no generic JSON parser.
fn parse_files_json(json: &str) -> Result<Vec<FileEntry>, JsError> {
    let trimmed = json.trim();
    if trimmed.is_empty() || trimmed == "[]" || trimmed == "null" {
        return Ok(Vec::new());
    }
    // Strip outer []
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(JsError::new("INVALID_INPUT:files_json must be a JSON array"));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut files = Vec::new();

    // Split by top-level objects: find matching { }
    let mut depth = 0i32;
    let mut obj_start: Option<usize> = None;
    for (i, ch) in inner.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        let obj_str = &inner[start..=i];
                        files.push(parse_single_file_object(obj_str)?);
                        obj_start = None;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(files)
}

/// Parse a single JSON object like `{"filename":"test.txt","content":"SGVsbG8="}`.
fn parse_single_file_object(obj: &str) -> Result<FileEntry, JsError> {
    let filename = extract_json_string_field(obj, "filename")?
        .ok_or_else(|| JsError::new("INVALID_INPUT:File object missing 'filename' field"))?;
    let content_b64 = extract_json_string_field(obj, "content")?
        .ok_or_else(|| JsError::new("INVALID_INPUT:File object missing 'content' field"))?;
    let content = base64_decode(&content_b64)?;
    Ok(FileEntry { filename, content })
}

/// Extract the value of a string field from a JSON object substring.
/// Handles JSON string escapes (\", \\, \n, etc.) within the value.
fn extract_json_string_field(obj: &str, field: &str) -> Result<Option<String>, JsError> {
    // Look for "field" : "value"
    let needle = format!("\"{}\"", field);
    let field_pos = match obj.find(&needle) {
        Some(p) => p,
        None => return Ok(None),
    };
    // Find the colon after the field name
    let after_field = field_pos + needle.len();
    let rest = &obj[after_field..];
    let colon_pos = match rest.find(':') {
        Some(p) => p,
        None => return Err(JsError::new("INVALID_INPUT:Malformed JSON object")),
    };
    let after_colon = &rest[colon_pos + 1..];
    // Find opening quote of the value
    let quote_pos = match after_colon.find('"') {
        Some(p) => p,
        None => return Err(JsError::new("INVALID_INPUT:Expected string value")),
    };
    let value_start = &after_colon[quote_pos + 1..];
    // Read until unescaped closing quote
    let mut value = String::new();
    let mut chars = value_start.chars();
    loop {
        match chars.next() {
            None => return Err(JsError::new("INVALID_INPUT:Unterminated string")),
            Some('"') => break,
            Some('\\') => {
                match chars.next() {
                    Some('"') => value.push('"'),
                    Some('\\') => value.push('\\'),
                    Some('/') => value.push('/'),
                    Some('n') => value.push('\n'),
                    Some('r') => value.push('\r'),
                    Some('t') => value.push('\t'),
                    Some('u') => {
                        // \uXXXX
                        let mut hex = String::with_capacity(4);
                        for _ in 0..4 {
                            match chars.next() {
                                Some(c) => hex.push(c),
                                None => return Err(JsError::new("INVALID_INPUT:Truncated \\u escape")),
                            }
                        }
                        let cp = u32::from_str_radix(&hex, 16)
                            .map_err(|_| JsError::new("INVALID_INPUT:Invalid \\u escape"))?;
                        if let Some(c) = char::from_u32(cp) {
                            value.push(c);
                        }
                    }
                    Some(c) => { value.push('\\'); value.push(c); }
                    None => return Err(JsError::new("INVALID_INPUT:Truncated escape")),
                }
            }
            Some(c) => value.push(c),
        }
    }
    Ok(Some(value))
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a secret message into a JPEG image.
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the cover image.
/// - `message`: The secret plaintext message to embed.
/// - `passphrase`: Passphrase for AES-256-GCM encryption (Argon2id KDF).
/// - `mode`: `1` = Ghost (stealth), `2` = Armor (robust; Fortress when it
///   fits), `3` = Armor forced to Standard. Any other value defaults to Ghost.
///   After an Armor encode, [`last_encode_fortress`] says which one core wrote.
///
/// # Returns
/// The stego JPEG as a `Uint8Array` (via wasm-bindgen), or throws a JS
/// exception with `"CODE:detail"` on error.
#[wasm_bindgen]
pub fn encode(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
    mode: u8,
) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    LAST_ENCODE_FORTRESS.with(|c| c.set(None));
    let stego = match mode {
        // 3 = Armor with the user's Standard switch (Fortress off).
        2 | 3 => phasm_core::armor_encode_with_options(image_bytes, message, passphrase, mode == 3).map(|(b, q)| {
            LAST_ENCODE_FORTRESS.with(|c| c.set(Some(q.fortress)));
            b
        }),
        _ => ghost_encode(image_bytes, message, passphrase),
    }
    .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(stego)
}

thread_local! {
    static LAST_ENCODE_FORTRESS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether the last [`encode`] wrote Fortress (`undefined` when it was not an
/// Armor encode or failed), so the page shows the sub-mode core actually used.
#[wasm_bindgen]
pub fn last_encode_fortress() -> Option<bool> {
    LAST_ENCODE_FORTRESS.with(|c| c.get())
}

/// Encode a secret message with file attachments into a JPEG image (Ghost mode only).
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the cover image.
/// - `message`: The secret plaintext message to embed.
/// - `files_json`: JSON array of file objects, each with `filename` (string)
///   and `content` (base64-encoded string). Example:
///   `[{"filename":"test.txt","content":"SGVsbG8="}]`
///   Pass `"[]"` or `""` for no files.
/// - `passphrase`: Passphrase for AES-256-GCM encryption (Argon2id KDF).
///
/// # Returns
/// The stego JPEG as a `Uint8Array` (via wasm-bindgen), or throws a JS
/// exception with `"CODE:detail"` on error.
#[wasm_bindgen]
pub fn encode_with_files(
    image_bytes: &[u8],
    message: &str,
    files_json: &str,
    passphrase: &str,
) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    let files = parse_files_json(files_json)?;
    let stego = ghost_encode_with_files(image_bytes, message, &files, passphrase)
        .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(stego)
}

/// Encode using Ghost mode with side information (SI-UNIWARD / "Deep Cover").
///
/// When the original uncompressed pixels are available (non-JPEG input like
/// PNG or BMP), the quantization rounding errors enable more efficient
/// embedding — roughly 1.5-2× capacity at the same detection risk.
///
/// # Parameters
/// - `image_bytes`: Cover JPEG bytes (as compressed from the raw pixels).
/// - `raw_pixels_rgb`: Original RGB pixels, row-major, 3 bytes per pixel (R,G,B,R,G,B,...).
/// - `pixel_width`: Width of the raw pixel buffer.
/// - `pixel_height`: Height of the raw pixel buffer.
/// - `message`: The secret plaintext message to embed.
/// - `passphrase`: Passphrase for AES-256-GCM encryption.
///
/// # Returns
/// The stego JPEG as a `Uint8Array`, or throws a JS exception.
#[wasm_bindgen]
pub fn encode_si(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    let stego = ghost_encode_si(
        image_bytes, raw_pixels_rgb, pixel_width, pixel_height, message, passphrase,
    )
    .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(stego)
}

/// Encode with SI-UNIWARD and file attachments (Ghost mode only).
///
/// Combines side-informed embedding with file attachment support.
///
/// # Parameters
/// - `image_bytes`: Cover JPEG bytes.
/// - `raw_pixels_rgb`: Original RGB pixels, row-major, 3 bytes per pixel.
/// - `pixel_width`, `pixel_height`: Dimensions of the raw pixel buffer.
/// - `message`: Plaintext message to embed.
/// - `files_json`: JSON array of file objects. Pass `""` or `"[]"` for no files.
/// - `passphrase`: Passphrase for encryption.
#[wasm_bindgen]
pub fn encode_si_with_files(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    files_json: &str,
    passphrase: &str,
) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    let files = parse_files_json(files_json)?;
    let stego = ghost_encode_si_with_files(
        image_bytes, raw_pixels_rgb, pixel_width, pixel_height, message, &files, passphrase,
    )
    .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(stego)
}

/// Encode with real-time progress reporting via a JS callback.
///
/// The `on_progress` callback receives `(step, total)` as arguments
/// each time the encoder advances. Only useful when called from a Web Worker;
/// on the main thread, the callback fires only after the encode completes.
///
/// # Parameters
/// Same as [`encode`], plus `on_progress` callback.
#[wasm_bindgen]
pub fn encode_with_progress(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
    mode: u8,
    on_progress: &js_sys::Function,
) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    phasm_core::progress::set_wasm_callback(Some(on_progress.clone()));
    let result = match mode {
        2 | 3 => phasm_core::armor_encode_with_options(image_bytes, message, passphrase, mode == 3).map(|(b, _)| b),
        _ => ghost_encode(image_bytes, message, passphrase),
    }
    .map_err(|e| JsError::new(&stego_error_message(&e)));
    phasm_core::progress::set_wasm_callback(None);
    result
}

// ---------------------------------------------------------------------------
// Parallel Ghost encode (Web Workers)
// ---------------------------------------------------------------------------

/// The coordinator of a Ghost encode split over Web Workers
/// (`phasm_core::GhostEncodeSession`): it hands out tasks for helper
/// workers (`ghost_run_task`) and takes their results. Same output bytes as
/// `encode_with_files` / `encode_si_with_files`. Every worker has its own
/// WASM instance; tasks and results are self-contained byte arrays, so no
/// `SharedArrayBuffer` (and no COOP/COEP headers) is needed.
#[wasm_bindgen]
pub struct GhostEncodeJob {
    inner: Option<phasm_core::GhostEncodeSession>,
}

fn job_err(e: StegoError) -> JsError {
    JsError::new(&stego_error_message(&e))
}

#[wasm_bindgen]
impl GhostEncodeJob {
    /// Start an encode. `raw_pixels_rgb` empty = JPEG cover; otherwise the
    /// Deep Cover pixels (as for `encode_si_with_files`). `helpers`: number of
    /// helper workers that will run the tasks.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        image_bytes: &[u8],
        raw_pixels_rgb: &[u8],
        pixel_width: u32,
        pixel_height: u32,
        message: &str,
        files_json: &str,
        passphrase: &str,
        helpers: u32,
    ) -> Result<GhostEncodeJob, JsError> {
        check_domain()?;
        let files = parse_files_json(files_json)?;
        let pixels = (!raw_pixels_rgb.is_empty()).then_some((raw_pixels_rgb, pixel_width, pixel_height));
        let inner = phasm_core::GhostEncodeSession::new(image_bytes, message, &files, passphrase, pixels, helpers as usize)
            .map_err(job_err)?;
        Ok(GhostEncodeJob { inner: Some(inner) })
    }

    fn session(&mut self) -> Result<&mut phasm_core::GhostEncodeSession, JsError> {
        self.inner.as_mut().ok_or_else(|| JsError::new("INVALID_INPUT:encode job already finished"))
    }

    /// Tasks ready for helpers: an array of `Uint8Array`. A task's int32
    /// (little-endian) at `ghost_task_target_offset()` is the helper it must
    /// run on, or −1 for any.
    pub fn take_tasks(&mut self) -> Result<js_sys::Array, JsError> {
        let tasks = self.session()?.take_tasks().map_err(job_err)?;
        let arr = js_sys::Array::new();
        for t in tasks {
            arr.push(&js_sys::Uint8Array::from(&t[..]));
        }
        Ok(arr)
    }

    /// The coordinator's own work (the keyed order); call right after the
    /// first `take_tasks`, while the helpers compute the costs.
    pub fn run_local(&mut self) -> Result<(), JsError> {
        self.session()?.run_local().map_err(job_err)
    }

    /// A helper's result; `helper` = index of the helper that ran it.
    pub fn submit(&mut self, result: &[u8], helper: i32) -> Result<(), JsError> {
        self.session()?.submit(result, helper).map_err(job_err)
    }

    pub fn is_done(&self) -> bool {
        self.inner.as_ref().is_some_and(|s| s.is_done())
    }

    /// Fraction of the work done, 0..1.
    pub fn progress(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |s| s.progress())
    }

    /// The stego JPEG (after `is_done()`).
    pub fn finish(&mut self) -> Result<Vec<u8>, JsError> {
        let s = self.inner.take().ok_or_else(|| JsError::new("INVALID_INPUT:encode job already finished"))?;
        s.finish().map(|(bytes, _)| bytes).map_err(job_err)
    }
}

thread_local! {
    static TASK_RUNNER: std::cell::RefCell<phasm_core::GhostTaskRunner> =
        std::cell::RefCell::new(phasm_core::GhostTaskRunner::new());
}

/// Run one task of a `GhostEncodeJob` (helper workers). The helper keeps
/// split-trellis back pointers between a segment task and its trace task.
#[wasm_bindgen]
pub fn ghost_run_task(task: Vec<u8>) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    TASK_RUNNER.with(|r| r.borrow_mut().run(task)).map_err(job_err)
}

/// Byte offset of a task's target helper (int32 LE, −1 = any helper).
#[wasm_bindgen]
pub fn ghost_task_target_offset() -> u32 {
    phasm_core::GHOST_TASK_TARGET_OFFSET as u32
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a hidden message from a stego JPEG image.
///
/// Auto-detects the embedding mode (Ghost or Armor) by trying both.
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the stego image.
/// - `passphrase`: The passphrase used during encoding.
///
/// # Returns
/// The decoded plaintext as a JS string, or throws a JS exception with
/// `"CODE:detail"` on error.
#[wasm_bindgen]
pub fn decode(image_bytes: &[u8], passphrase: &str) -> Result<String, JsError> {
    check_domain()?;
    let (payload, _quality) =
        smart_decode(image_bytes, passphrase).map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(payload.text)
}

/// Decode a hidden message and return quality information.
///
/// Returns a JSON string with fields: `message`, `mode` (1=Ghost, 2=Armor),
/// `integrity` (0-100), `rsErrors`, `rsCapacity`, `fortressUsed`, and `files`
/// (array of `{filename, size, content}` where content is base64-encoded).
#[wasm_bindgen]
pub fn decode_with_quality(image_bytes: &[u8], passphrase: &str) -> Result<String, JsError> {
    check_domain()?;
    let (payload, quality) =
        smart_decode(image_bytes, passphrase).map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(format_decode_result(&payload, &quality))
}

/// Decode with real-time progress reporting via a JS callback.
///
/// The `on_progress` callback receives `(step, total)` as arguments
/// each time the decoder advances to the next candidate.
/// Returns the same JSON string as `decode_with_quality`.
#[wasm_bindgen]
pub fn decode_with_progress(
    image_bytes: &[u8],
    passphrase: &str,
    on_progress: &js_sys::Function,
) -> Result<String, JsError> {
    check_domain()?;
    phasm_core::progress::set_wasm_callback(Some(on_progress.clone()));
    let result = smart_decode(image_bytes, passphrase)
        .map(|(p, q)| format_decode_result(&p, &q))
        .map_err(|e| JsError::new(&stego_error_message(&e)));
    phasm_core::progress::set_wasm_callback(None);
    result
}

/// Decode Ghost mode only (no Armor attempt).
///
/// Accepts an `on_progress` callback for real-time progress reporting.
/// Returns the same JSON format as `decode_with_quality`. Intended for use in
/// a parallel Web Worker alongside `armor_decode_only` — whichever mode
/// succeeds first wins.
#[wasm_bindgen]
pub fn ghost_decode_only(
    image_bytes: &[u8],
    passphrase: &str,
    on_progress: &js_sys::Function,
) -> Result<String, JsError> {
    check_domain()?;
    phasm_core::progress::set_wasm_callback(Some(on_progress.clone()));
    phasm_core::progress::init(GHOST_DECODE_STEPS);
    // Try primary Ghost decode first, then shadow decode as fallback
    let result = match ghost_decode(image_bytes, passphrase) {
        Ok(payload) => {
            let quality = DecodeQuality::ghost();
            Ok(format_decode_result(&payload, &quality))
        }
        Err(_primary_err) => {
            match ghost_shadow_decode(image_bytes, passphrase) {
                Ok(payload) => {
                    let quality = DecodeQuality::ghost();
                    Ok(format_decode_result(&payload, &quality))
                }
                Err(_) => Err(JsError::new(&stego_error_message(&_primary_err))),
            }
        }
    };
    phasm_core::progress::set_wasm_callback(None);
    result
}

/// Decode Armor mode only (no Ghost attempt).
///
/// Accepts an `on_progress` callback for real-time progress reporting.
/// Returns the same JSON format as `decode_with_quality`. Intended for use in
/// a parallel Web Worker alongside `ghost_decode_only`.
#[wasm_bindgen]
pub fn armor_decode_only(
    image_bytes: &[u8],
    passphrase: &str,
    on_progress: &js_sys::Function,
) -> Result<String, JsError> {
    check_domain()?;
    phasm_core::progress::set_wasm_callback(Some(on_progress.clone()));
    let result = armor_decode(image_bytes, passphrase)
        .map(|(payload, quality)| format_decode_result(&payload, &quality))
        .map_err(|e| JsError::new(&stego_error_message(&e)));
    phasm_core::progress::set_wasm_callback(None);
    result
}

/// Cancel a running decode operation.
///
/// Sets the global cancellation flag. The next `check_cancelled()` call inside
/// the decode pipeline will return `StegoError::Cancelled`, aborting the
/// operation early.
#[wasm_bindgen]
pub fn decode_cancel() {
    phasm_core::progress::cancel();
}

// ---------------------------------------------------------------------------
// JSON formatting
// ---------------------------------------------------------------------------

/// Escape a string for safe embedding in a JSON string value.
///
/// Handles backslash, double-quote, and all control characters U+0000-U+001F
/// per RFC 8259 S7.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Control characters U+0000-U+001F (excluding already-handled \n \r \t)
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

fn format_files_json(files: &[phasm_core::FileEntry]) -> String {
    let mut out = String::from("[");
    for (i, file) in files.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let escaped_name = json_escape(&file.filename);
        let b64_content = base64_encode(&file.content);
        out.push_str(&format!(
            r#"{{"filename":"{}","size":{},"content":"{}"}}"#,
            escaped_name,
            file.content.len(),
            b64_content,
        ));
    }
    out.push(']');
    out
}

fn format_decode_result(payload: &PayloadData, quality: &DecodeQuality) -> String {
    let mode_name = if quality.mode == 0x01 { "Ghost" } else { "Armor" };
    let escaped = json_escape(&payload.text);
    let files_json = format_files_json(&payload.files);
    format!(
        r#"{{"message":"{}","mode":"{}","modeId":{},"integrity":{},"rsErrors":{},"rsCapacity":{},"fortressUsed":{},"files":{}}}"#,
        escaped,
        mode_name,
        quality.mode,
        quality.integrity_percent,
        quality.rs_errors_corrected,
        quality.rs_error_capacity,
        quality.fortress_used,
        files_json,
    )
}

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

/// Estimate the maximum message capacity (in bytes) for a JPEG image.
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the cover image.
/// - `mode`: `1` = Ghost, `2` = Armor. Any other value defaults to Ghost.
///
/// # Returns
/// The estimated capacity in bytes as a `usize`, or throws a JS exception
/// with `"CODE:detail"` on error.
#[wasm_bindgen]
pub fn capacity(image_bytes: &[u8], mode: u8) -> Result<usize, JsError> {
    check_domain()?;
    let img = JpegImage::from_bytes(image_bytes)
        .map_err(|e| JsError::new(&format!("INVALID_JPEG:{e:?}")))?;
    let cap = match mode {
        2 => armor_capacity(&img),
        _ => ghost_capacity(&img),
    }
    .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(cap)
}

/// Ghost capacity of a Deep Cover embed: the cover JPEG plus the raw RGB
/// pixels that will be passed to `encode_si` / `encode_si_with_files`. With
/// exact side information (the cover is [`deep_cover_jpeg`]'s output for
/// these pixels, checked by core) the maximum is nzAC / 4 frame bits,
/// otherwise the plain Ghost [`capacity`].
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the cover image.
/// - `raw_pixels`: RGB (3 bytes per pixel, row-major), at least
///   `pixel_width * pixel_height * 3` bytes.
///
/// # Returns
/// The capacity in bytes as `usize`.
#[wasm_bindgen]
pub fn capacity_si(image_bytes: &[u8], raw_pixels: &[u8], pixel_width: u32, pixel_height: u32) -> Result<usize, JsError> {
    check_domain()?;
    let img = si_capacity_cover(image_bytes, raw_pixels)?;
    ghost_capacity_si(&img, raw_pixels, pixel_width, pixel_height).map_err(|e| JsError::new(&stego_error_message(&e)))
}

/// Shadow message capacity of a Deep Cover embed (cover JPEG + its raw RGB
/// pixels): with exact side information a shadow may hold half the Deep
/// Cover maximum, otherwise half the plain Ghost maximum.
#[wasm_bindgen]
pub fn shadow_capacity_si(
    image_bytes: &[u8],
    raw_pixels: &[u8],
    pixel_width: u32,
    pixel_height: u32,
) -> Result<usize, JsError> {
    check_domain()?;
    let img = si_capacity_cover(image_bytes, raw_pixels)?;
    estimate_shadow_capacity_si(&img, raw_pixels, pixel_width, pixel_height)
        .map_err(|e| JsError::new(&stego_error_message(&e)))
}

/// Parse the cover of a pixel-aware (`*_si`) capacity call.
fn si_capacity_cover(image_bytes: &[u8], raw_pixels: &[u8]) -> Result<JpegImage, JsError> {
    if raw_pixels.is_empty() {
        return Err(JsError::new("INTERNAL:No raw pixel data provided"));
    }
    JpegImage::from_bytes(image_bytes).map_err(|e| JsError::new(&format!("INVALID_JPEG:{e:?}")))
}

/// Write the Deep Cover cover JPEG for raw RGB pixels with phasm's cover
/// writer (byte-identical to libjpeg-turbo `cjpeg -quality Q`).
///
/// Pass the result to `encode_si` / `encode_si_with_files` together with the
/// SAME pixel buffer so the core uses exact side information
/// (`docs/design/image/deep-cover-exact-si.md`); compute capacity / bands on
/// these bytes too.
///
/// # Parameters
/// - `pixels`: raw RGB (3 bytes per pixel, row-major), `width * height * 3` bytes.
/// - `width`, `height`: dimensions in pixels.
/// - `quality`: IJG quality 1..=100 (92 = the apps' Ghost quality).
///
/// # Returns
/// The cover JPEG as `Uint8Array`.
#[wasm_bindgen]
pub fn deep_cover_jpeg(pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3));
    if expected != Some(pixels.len()) {
        return Err(JsError::new("INTERNAL:Pixel buffer size mismatch"));
    }
    if !(1..=100).contains(&quality) {
        return Err(JsError::new(&format!("INTERNAL:Quality must be 1..=100, got {quality}")));
    }
    core_deep_cover_jpeg(pixels, width, height, quality).map_err(|e| JsError::new(&stego_error_message(&e)))
}

/// The JPEG made upright (EXIF orientation applied losslessly in the DCT
/// domain, edge MCU trimmed) and stripped of metadata, without re-encoding:
/// the magic wand on a JPEG source (fix list 2d). Mirrors iOS
/// `phasm_jpeg_upright_and_clean`.
#[wasm_bindgen]
pub fn jpeg_upright_and_clean(jpeg: &[u8], keep_icc: bool) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    phasm_core::jpeg_upright_and_clean(jpeg, keep_icc).map_err(|e| JsError::new(&format!("INVALID_JPEG:{e}")))
}

/// The IJG quality (1..100) a JPEG's quantization tables correspond to
/// (exact for IJG tables, the closest IJG quality for others), -1 when the
/// bytes are no readable JPEG. Reads the header only. Mirrors iOS
/// `phasm_jpeg_quality_estimate`.
#[wasm_bindgen]
pub fn jpeg_quality_estimate(jpeg: &[u8]) -> Result<i32, JsError> {
    check_domain()?;
    Ok(phasm_core::estimate_jpeg_quality(jpeg).map_or(-1, |e| e.quality as i32))
}

/// Whether RGB pixels are a JPEG decoded on the cover's own 8x8 grid (no Deep
/// Cover side information; bands at nzAC/10). False on a bad buffer.
#[wasm_bindgen]
pub fn is_aligned_jpeg_decode(pixels: &[u8], width: u32, height: u32) -> Result<bool, JsError> {
    check_domain()?;
    let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3));
    Ok(expected == Some(pixels.len()) && phasm_core::is_aligned_jpeg_decode(pixels, width, height))
}

/// RGB pixels resampled with core's deterministic Lanczos-3 cover resampler
/// (the same bytes as the apps; fix list 2.3).
#[wasm_bindgen]
pub fn resample_rgb(pixels: &[u8], width: u32, height: u32, dst_width: u32, dst_height: u32) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    phasm_core::resample_rgb(pixels, width, height, dst_width, dst_height).map_err(|e| JsError::new(&stego_error_message(&e)))
}

fn cover_request(mode: u32, wand: bool, ghost_quality: u32, ghost_max_side: u32, armor_preset: u32) -> Result<phasm_core::CoverRequest, JsError> {
    Ok(phasm_core::CoverRequest {
        mode: match mode {
            1 => phasm_core::CoverMode::Ghost,
            2 => phasm_core::CoverMode::Armor,
            _ => return Err(JsError::new("INTERNAL:Unknown mode")),
        },
        wand,
        ghost_quality: (1..=100).contains(&ghost_quality).then_some(ghost_quality as u8),
        ghost_max_side: (ghost_max_side > 0).then_some(ghost_max_side),
        armor_preset: phasm_core::ArmorCoverPreset::from_id(armor_preset).ok_or_else(|| JsError::new("INTERNAL:Unknown preset"))?,
    })
}

/// Whether `build_cover` needs the source's upright sRGB pixels. `mode` 1 =
/// Ghost, 2 = Armor; `ghost_quality` 0 = default; `ghost_max_side` 0 =
/// original size; `armor_preset` 0 / 1 / 2 = WhatsApp 1600 / Telegram 1280 /
/// Instagram 1080.
#[wasm_bindgen]
pub fn cover_needs_pixels(original: &[u8], mode: u32, wand: bool, ghost_quality: u32, ghost_max_side: u32, armor_preset: u32) -> Result<bool, JsError> {
    check_domain()?;
    Ok(phasm_core::cover_needs_pixels(original, &cover_request(mode, wand, ghost_quality, ghost_max_side, armor_preset)?))
}

/// The cover to encode (`build_cover`).
#[wasm_bindgen]
pub struct CoverResult {
    jpeg: Vec<u8>,
    si_pixels: Option<Vec<u8>>,
    si_is_input: bool,
    info: [u32; 8],
    deep_cover: bool,
}

#[wasm_bindgen]
impl CoverResult {
    /// The cover JPEG.
    #[wasm_bindgen(getter)]
    pub fn jpeg(&self) -> Vec<u8> {
        self.jpeg.clone()
    }
    /// Pixels for the `*_si` calls when core made them (a resample), else
    /// `undefined`: the plain calls, or `si_is_input`.
    #[wasm_bindgen(getter)]
    pub fn si_pixels(&self) -> Option<Vec<u8>> {
        self.si_pixels.clone()
    }
    /// The `*_si` pixels are exactly the pixels passed to `build_cover`
    /// (a same-size Ghost cover): reuse that buffer, core returns no copy.
    #[wasm_bindgen(getter)]
    pub fn si_is_input(&self) -> bool {
        self.si_is_input
    }
    /// width, height, passThrough, hasCrop, cropX, cropY, cropW, cropH.
    #[wasm_bindgen(getter)]
    pub fn info(&self) -> Vec<u32> {
        self.info.to_vec()
    }
    /// Core found exact side information: the only case a "Deep Cover"
    /// label may show (`si_pixels` alone are not enough).
    #[wasm_bindgen(getter)]
    pub fn deep_cover(&self) -> bool {
        self.deep_cover
    }
}

/// The cover to encode for the picked file and the user's settings (rules in
/// core: docs/design/app/encode-cover-pipeline.md). `pixels` = the source's
/// upright sRGB RGB (empty when `cover_needs_pixels` is false).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn build_cover(
    original: &[u8],
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: u32,
    wand: bool,
    ghost_quality: u32,
    ghost_max_side: u32,
    armor_preset: u32,
) -> Result<CoverResult, JsError> {
    check_domain()?;
    let req = cover_request(mode, wand, ghost_quality, ghost_max_side, armor_preset)?;
    let px = (!pixels.is_empty()).then_some((pixels, width, height));
    let c = phasm_core::build_cover(original, px, &req).map_err(|e| JsError::new(&stego_error_message(&e)))?;
    let crop = c.crop.unwrap_or([0; 4]);
    Ok(CoverResult {
        info: [c.width, c.height, c.pass_through as u32, c.crop.is_some() as u32, crop[0], crop[1], crop[2], crop[3]],
        jpeg: c.jpeg,
        si_pixels: c.si_pixels,
        si_is_input: c.si_is_input,
        deep_cover: c.deep_cover,
    })
}

/// The cover JPEG with the source file's metadata, cleaned (the magic wand
/// off; fix list 1.2).
#[wasm_bindgen]
pub fn cover_with_source_metadata(cover: &[u8], source: &[u8]) -> Result<Vec<u8>, JsError> {
    check_domain()?;
    phasm_core::cover_with_source_metadata(cover, source).map_err(|e| JsError::new(&format!("INVALID_JPEG:{e}")))
}

/// Estimate Ghost primary capacity accounting for shadow position overhead.
///
/// Returns accurate primary capacity when shadows are present, subtracting
/// the positions consumed by shadow RS-encoded data from the usable pool.
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the cover image.
/// - `shadow_count`: Number of shadow layers.
/// - `shadow_total_bytes`: Total plaintext bytes across all shadow messages.
/// - `is_si`: ignored — no side information; the Deep Cover maximum needs the
///   pixels ([`ghost_bands_si`]).
///
/// # Returns
/// The estimated primary capacity in bytes as `usize`.
#[wasm_bindgen]
pub fn capacity_with_shadows(
    image_bytes: &[u8],
    shadow_count: usize,
    shadow_total_bytes: usize,
    is_si: bool,
) -> Result<usize, JsError> {
    check_domain()?;
    let img = JpegImage::from_bytes(image_bytes)
        .map_err(|e| JsError::new(&format!("INVALID_JPEG:{e:?}")))?;
    let cap = ghost_capacity_with_shadows(&img, shadow_count, shadow_total_bytes, is_si)
        .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(cap)
}

/// Estimate dual-tier capacity info for an Armor-mode JPEG image.
///
/// Returns a JSON string: `{"fortress": N, "stdm": N}` where N is max
/// plaintext bytes for each sub-mode.
#[wasm_bindgen]
pub fn capacity_info(image_bytes: &[u8]) -> Result<String, JsError> {
    check_domain()?;
    let info = armor_capacity_info(image_bytes)
        .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(format!(
        r#"{{"fortress":{},"fortressCompact":{},"stdm":{}}}"#,
        info.fortress_capacity,
        info.fortress_capacity_compact,
        info.stdm_capacity,
    ))
}
/// Ghost stealth bands of a cover (the capacity bar) as JSON
/// `{"max":…,"comfort":…,"caution":…}`, in plaintext bytes of the primary
/// payload next to `shadow_count` shadows of `shadow_total_bytes` together
/// (0, 0 = no shadows), for an embed without side information (JPEG input);
/// Deep Cover covers: [`ghost_bands_si`].
#[wasm_bindgen]
pub fn ghost_bands(image_bytes: &[u8], shadow_count: u32, shadow_total_bytes: u32) -> Result<String, JsError> {
    check_domain()?;
    let img = JpegImage::from_bytes(image_bytes)
        .map_err(|e| JsError::new(&format!("INVALID_JPEG:{e:?}")))?;
    let b = phasm_core::ghost_bands(&img, shadow_count as usize, shadow_total_bytes as usize)
        .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(bands_json(&b))
}

/// [`ghost_bands`] for a Deep Cover embed: the cover JPEG plus the raw RGB
/// pixels that will be passed to `encode_si*`. With exact side information
/// (checked by core) every band derives from the Deep Cover maximum, nzAC /
/// 4 frame bits; otherwise the same as [`ghost_bands`].
#[wasm_bindgen]
pub fn ghost_bands_si(
    image_bytes: &[u8],
    raw_pixels: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    shadow_count: u32,
    shadow_total_bytes: u32,
) -> Result<String, JsError> {
    check_domain()?;
    let img = si_capacity_cover(image_bytes, raw_pixels)?;
    let b = phasm_core::ghost_bands_si(
        &img,
        raw_pixels,
        pixel_width,
        pixel_height,
        shadow_count as usize,
        shadow_total_bytes as usize,
    )
    .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(bands_json(&b))
}

/// The Ghost capacity bar's numbers in one call (A#31): [`ghost_bands`] /
/// [`ghost_bands_si`] and the largest single shadow message
/// ([`shadow_capacity_si`]), as JSON `{"max","comfort","caution","shadowMax"}`.
/// `raw_pixels` empty = JPEG input (the plain figures); else the Deep Cover
/// pixels, as for [`ghost_bands_si`].
#[wasm_bindgen]
pub fn ghost_bands_and_shadow_capacity(
    image_bytes: &[u8],
    raw_pixels: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    shadow_count: u32,
    shadow_total_bytes: u32,
) -> Result<String, JsError> {
    check_domain()?;
    let img = JpegImage::from_bytes(image_bytes).map_err(|e| JsError::new(&format!("INVALID_JPEG:{e:?}")))?;
    let si = (!raw_pixels.is_empty()).then_some((raw_pixels, pixel_width, pixel_height));
    let (b, shadow) = phasm_core::ghost_bands_and_shadow_capacity(&img, si, shadow_count as usize, shadow_total_bytes as usize)
        .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(format!(
        "{{\"max\":{},\"comfort\":{},\"caution\":{},\"shadowMax\":{}}}",
        b.max_bytes, b.comfort_bytes, b.caution_bytes, shadow
    ))
}

fn bands_json(b: &phasm_core::GhostBands) -> String {
    format!("{{\"max\":{},\"comfort\":{},\"caution\":{}}}", b.max_bytes, b.comfort_bytes, b.caution_bytes)
}

/// Compute the compressed payload size for the given message and optional files.
///
/// Returns the number of bytes the payload would occupy after Brotli compression
/// (including the 1-byte flags prefix). This is the "used" value for the capacity
/// bar, reflecting real compression savings.
///
/// # Parameters
/// - `message`: The plaintext message.
/// - `files_json`: JSON array of file objects (`[{"filename":"...","content":"<base64>"}]`).
///   Pass `""` or `"[]"` for no files.
///
/// # Returns
/// Payload size in bytes as `usize`.
#[wasm_bindgen]
pub fn payload_size(message: &str, files_json: &str) -> usize {
    let files = parse_files_json(files_json).unwrap_or_default();
    phasm_core::compressed_payload_size(message, &files)
}

/// Maximum pixel dimension (width or height) for encode input.
#[wasm_bindgen]
pub fn max_dimension() -> u32 {
    phasm_core::MAX_DIMENSION
}

/// Maximum total pixel count (width x height) for encode input.
#[wasm_bindgen]
pub fn max_pixels() -> u32 {
    phasm_core::MAX_PIXELS
}

/// Minimum pixel dimension (width or height) for encode input.
#[wasm_bindgen]
pub fn min_encode_dimension() -> u32 {
    phasm_core::MIN_ENCODE_DIMENSION
}

/// Target pixel dimension (longest side) for Armor/Fortress pre-resize.
#[wasm_bindgen]
pub fn armor_target_dimension() -> u32 {
    phasm_core::ARMOR_TARGET_DIMENSION
}

/// The Armor cover size for an upright `src_w` x `src_h` source and a channel
/// preset (0 = Standard 1600 px, 1 = Telegram 1280 px, 2 = Instagram 1080 px
/// wide), as `[width, height, scaled_width, scaled_height, crop_x, crop_y]`
/// (a `Uint32Array`). Draw the source at scaled_width x scaled_height, offset
/// by (-crop_x, -crop_y), on a width x height canvas.
#[wasm_bindgen]
pub fn armor_cover_dimensions(src_w: u32, src_h: u32, preset: u32) -> Result<Vec<u32>, JsError> {
    let p = phasm_core::ArmorCoverPreset::from_id(preset)
        .ok_or_else(|| JsError::new(&format!("INVALID_ARGUMENT:unknown Armor cover preset {preset}")))?;
    Ok(phasm_core::armor_cover_dimensions(src_w, src_h, p).to_array().to_vec())
}

/// Cross-platform empirical STDM LLR byte-equivalence hook.
///
/// Returns the lowercase-hex SHA256 of 10,000 deterministic LCG-seeded
/// `spread_dot_product` calls. Used to verify WASM SIMD128 produces
/// bit-identical f64 dot products to the native NEON/SSE paths.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_spread_dot_hash() -> String {
    phasm_core::stego::armor::embedding_simd::spread_dot_test_hash_hex()
}

/// Cross-platform empirical FFT byte-equivalence hook.
///
/// Returns the lowercase-hex SHA256 of the deterministic 256×256 FFT
/// output. The expected value (recorded on aarch64 NEON) is checked
/// from Node/V8 to verify WASM SIMD128 produces bit-identical FFT
/// output to the native NEON path. Not part of the public API — only
/// exists so the test harness can call into the WASM bundle.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_fft2d_hash() -> String {
    phasm_core::stego::armor::fft2d::fft2d_test_hash_hex()
}

/// Cross-platform empirical AAN DCT/IDCT byte-equivalence
/// hook. Returns the lowercase-hex SHA256 of the deterministic 12-block
/// LL&M DCT/IDCT output. Expected value (recorded on aarch64 NEON):
/// `8affcdb98021d23b1c1ef536f782610f229fac187360f02722bb3470db8446d5`.
/// Verifies WASM SIMD128 produces bit-identical output to NEON + AVX2 +
/// scalar paths.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_aan_hash() -> String {
    phasm_core::codec::jpeg::pixels_aan::aan_test_hash_hex()
}

/// Test builds only: fixed salt/nonce seed (`PHASM_DETERMINISTIC_SEED`), so
/// an encode can be compared byte for byte with a native one.
#[cfg(feature = "test-hooks")]
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_set_deterministic_seed(seed: u64) {
    phasm_core::stego::crypto::test_hooks::set_deterministic_seed(Some(seed));
}

/// Test builds only: one STC trellis (`m` message bits, width `w`, height
/// `h`) on random input; returns the plan's total cost. For timing the
/// kernel from JS.
#[cfg(feature = "test-hooks")]
#[wasm_bindgen]
#[doc(hidden)]
pub fn __bench_trellis(m: usize, w: usize, h: usize, inline: bool) -> f64 {
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let n = m * w;
    let cover: Vec<u8> = (0..n).map(|_| (next() & 1) as u8).collect();
    let costs: Vec<f32> = (0..n).map(|_| (next() % 10_000) as f32 / 100.0).collect();
    let msg: Vec<u8> = (0..m).map(|_| (next() & 1) as u8).collect();
    let hh = phasm_core::stego::stc::hhat::generate_hhat_end_set(h, w, &[7u8; 32]);
    let cols: Vec<u32> = (0..w).map(|c| phasm_core::stego::stc::hhat::column_packed(&hh, c)).collect();
    let limit = if inline { usize::MAX } else { 0 };
    phasm_core::stego::stc::trellis::embed(&cover, &costs, &msg, &cols, h, limit, 1, &|_| {})
        .ok()
        .flatten()
        .map_or(-1.0, |p| p.total_cost)
}

/// Test builds only: time-able split-trellis pieces on random input
/// (`which`: 0 = pass A over `m` blocks, 1 = one segment's back pointers of
/// `m` blocks, 2 = `embed` inline).
#[cfg(feature = "test-hooks")]
#[wasm_bindgen]
#[doc(hidden)]
pub fn __bench_split(m: usize, w: usize, h: usize, which: u8) -> f64 {
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let n = m * w;
    let cover: Vec<u8> = (0..n).map(|_| (next() & 1) as u8).collect();
    let costs: Vec<f32> = (0..n).map(|_| (next() % 10_000) as f32 / 100.0).collect();
    let msg: Vec<u8> = (0..m).map(|_| (next() & 1) as u8).collect();
    let hh = phasm_core::stego::stc::hhat::generate_hhat_end_set(h, w, &[7u8; 32]);
    let cols: Vec<u32> = (0..w).map(|c| phasm_core::stego::stc::hhat::column_packed(&hh, c)).collect();
    use phasm_core::stego::stc::trellis as t;
    let code = t::Code::uniform(&cols, m);
    match which {
        0 => t::pass_a(&cover, &costs, &msg, &code, h, m, &|_| {}).ok().flatten().map_or(-1.0, |p| p.cost),
        1 => {
            let mut cp = vec![f64::INFINITY; 1 << h];
            cp[0] = 0.0;
            t::segment_rows(&cover, &costs, &msg, &code, 0, h, &cp).map_or(-1.0, |r| r[r.len() / 2] as f64)
        }
        _ => t::embed(&cover, &costs, &msg, &cols, h, usize::MAX, 1, &|_| {}).ok().flatten().map_or(-1.0, |p| p.total_cost),
    }
}

/// Cross-platform STC trellis byte-equivalence hook: the SHA-256 pinned in
/// `stc::trellis::tests::trellis_hash_pinned`. Verifies the WASM SIMD128
/// trellis kernel plans exactly what the NEON / scalar kernels plan.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_trellis_hash() -> String {
    phasm_core::stego::stc::trellis::trellis_test_hash_hex()
}

/// Cross-platform ChaCha20 stream hook: the WASM SIMD128 keystream used by
/// the keyed shuffle must equal `rand_chacha::ChaCha20Rng`'s (pinned in
/// `stego::chacha_stream::tests`).
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_chacha_hash() -> String {
    phasm_core::stego::chacha_stream::chacha_stream_test_hash_hex()
}

/// Cross-platform J-UNIWARD cost-kernel byte-equivalence hook: the
/// kernel hash pinned in `core/tests/cost_kernel_cross_platform.rs`
/// (`14156ec6…`, Ghost v3.1) and the legacy v1 one (`059a4a47…`). Verifies WASM
/// SIMD128 produces bit-identical costs to NEON / SSE2 / scalar.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_cost_kernel_hashes() -> String {
    format!(
        "{} {}",
        phasm_core::stego::cost::uniward::cost_kernel_test_hash_hex(),
        phasm_core::stego::cost::uniward::cost_kernel_test_hash_hex_legacy_v1(),
    )
}

/// SHA256 over `(flat_idx, cost bits)` of every Y AC coefficient of a JPEG
/// under the Ghost encoder's costs (`compute_ac_costs_separable`), for comparing
/// a whole real image's costs against a native build
/// (`core/examples/cost_hash_check.rs <image>`).
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_cost_positions_hash(image_bytes: &[u8]) -> Result<String, JsError> {
    use sha2::{Digest, Sha256};
    let img = phasm_core::JpegImage::from_bytes(image_bytes).map_err(|e| JsError::new(&format!("{e:?}")))?;
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or_else(|| JsError::new("no qt"))?;
    let pos = phasm_core::stego::cost::uniward::compute_ac_costs_separable(img.dct_grid(0), qt)
        .map_err(|e| JsError::new(&format!("{e:?}")))?;
    let mut h = Sha256::new();
    let mut n_pos = 0usize;
    for (fi, c) in pos.iter().enumerate().filter(|(_, c)| c.is_finite()) {
        h.update((fi as u32).to_le_bytes());
        h.update(c.to_bits().to_le_bytes());
        n_pos += 1;
    }
    let _ = n_pos;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Returns the SHA256 of the SCALAR-explicit path (bypassing
/// SIMD dispatch). Used to confirm the SIMD path is being exercised (the
/// returned hash should match the canonical aarch64 NEON pin
/// `8affcdb9…`); if `__test_aan_hash` differs from this scalar value, the
/// WASM SIMD kernel has diverged.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_aan_hash_scalar() -> String {
    use sha2::{Digest, Sha256};
    let bytes = phasm_core::codec::jpeg::pixels_aan::aan_test_deterministic_bytes_scalar();
    let mut h = Sha256::new();
    h.update(&bytes);
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}
