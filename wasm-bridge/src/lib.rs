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
    armor_capacity, armor_capacity_info, armor_decode, armor_encode, ghost_capacity,
    ghost_capacity_si, ghost_capacity_with_shadows, ghost_decode, ghost_shadow_decode,
    ghost_encode, ghost_encode_si, ghost_encode_si_with_files,
    ghost_encode_with_files, optimize_cover, OptimizerConfig, OptimizerMode,
    smart_decode, DecodeQuality, FileEntry, JpegImage, PayloadData, StegoError,
    GHOST_DECODE_STEPS,
};

const ALLOWED_HOSTS: &[&str] = &["phasm.app", "phasm.link", "localhost", "127.0.0.1"];

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
/// - `mode`: `1` = Ghost (stealth), `2` = Armor (robust). Any other value
///   defaults to Ghost.
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
    let stego = match mode {
        2 => armor_encode(image_bytes, message, passphrase),
        _ => ghost_encode(image_bytes, message, passphrase),
    }
    .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(stego)
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
        2 => armor_encode(image_bytes, message, passphrase),
        _ => ghost_encode(image_bytes, message, passphrase),
    }
    .map_err(|e| JsError::new(&stego_error_message(&e)));
    phasm_core::progress::set_wasm_callback(None);
    result
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

/// Estimate Ghost mode capacity with SI-UNIWARD ("Deep Cover").
///
/// Returns ~43% more capacity than standard Ghost mode, reflecting the
/// lower per-bit distortion cost when raw pixels are available.
///
/// # Parameters
/// - `image_bytes`: Raw JPEG bytes of the cover image.
///
/// # Returns
/// The estimated SI capacity in bytes as `usize`.
#[wasm_bindgen]
pub fn capacity_si(image_bytes: &[u8]) -> Result<usize, JsError> {
    check_domain()?;
    let img = JpegImage::from_bytes(image_bytes)
        .map_err(|e| JsError::new(&format!("INVALID_JPEG:{e:?}")))?;
    let cap = ghost_capacity_si(&img)
        .map_err(|e| JsError::new(&stego_error_message(&e)))?;
    Ok(cap)
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
/// - `is_si`: `true` for SI-UNIWARD (Deep Cover), `false` for standard J-UNIWARD.
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
        r#"{{"fortress":{},"stdm":{}}}"#,
        info.fortress_capacity,
        info.stdm_capacity,
    ))
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

/// Optimize raw RGB pixels for steganographic embedding.
///
/// Preprocesses the pixel buffer to improve embedding quality and capacity.
/// The optimizer is mode-aware: Ghost (1) gets a full 4-stage pipeline,
/// Armor (2) gets block-boundary smoothing + DC stabilization,
/// Fortress (3) gets block-boundary smoothing only.
///
/// # Parameters
/// - `pixels`: Raw RGB pixel data (3 bytes per pixel, row-major).
/// - `width`, `height`: Image dimensions in pixels.
/// - `mode`: `1` = Ghost, `2` = Armor, `3` = Fortress.
/// - `seed`: 32-byte ChaCha20 seed for deterministic noise.
///
/// # Returns
/// Optimized pixel buffer as `Uint8Array` (same dimensions/format).
#[wasm_bindgen]
pub fn optimize_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: u8,
    seed: &[u8],
) -> Result<Vec<u8>, JsError> {
    check_domain()?;

    if seed.len() < 32 {
        return Err(JsError::new("INTERNAL:Seed must be 32 bytes"));
    }
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(&seed[..32]);

    let expected_len = width as usize * height as usize * 3;
    if pixels.len() != expected_len {
        return Err(JsError::new("INTERNAL:Pixel buffer size mismatch"));
    }

    let optimizer_mode = match mode {
        2 => OptimizerMode::Armor,
        3 => OptimizerMode::Fortress,
        _ => OptimizerMode::Ghost,
    };

    let config = OptimizerConfig {
        strength: 0.85,
        seed: seed_arr,
        mode: optimizer_mode,
    };

    Ok(optimize_cover(pixels, width, height, &config))
}

/// T2.4 — Cross-platform empirical STDM LLR byte-equivalence hook.
///
/// Returns the lowercase-hex SHA256 of 10,000 deterministic LCG-seeded
/// `spread_dot_product` calls. Used to verify WASM SIMD128 produces
/// bit-identical f64 dot products to the native NEON/SSE paths.
#[wasm_bindgen]
#[doc(hidden)]
pub fn __test_spread_dot_hash() -> String {
    phasm_core::stego::armor::embedding_simd::spread_dot_test_hash_hex()
}

/// T2.3 — Cross-platform empirical FFT byte-equivalence hook.
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
