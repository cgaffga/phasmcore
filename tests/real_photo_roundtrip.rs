/// Real-world photo integration tests.
///
/// These tests are marked `#[ignore]` because they are slow (3-4 minutes).
/// They are skipped by default when running `cargo test` or `./build.sh test`.
///
/// To run them explicitly:
///
///     ./build.sh test-all
///     # or:
///     cargo test -p phasm-core -- --include-ignored --nocapture
///     # or just the real photo tests:
///     cargo test -p phasm-core --test real_photo_roundtrip -- --ignored --nocapture
///
/// Drop ANY photos into `core/tests/real_photos/` before running.
///
/// Supported formats: JPEG, PNG, HEIC, WebP, TIFF, GIF, BMP — anything macOS
/// can read. Non-baseline JPEGs (progressive, etc.) and non-JPEG formats are
/// automatically converted to baseline JPEG via `sips` (macOS built-in), just
/// like the frontends convert via UIImage/BitmapFactory/canvas.
///
/// Each photo is tested with both Ghost and Armor modes, with and without a
/// passphrase. Photos too small for a given mode are skipped (not failures).

use phasm_core::{
    armor_capacity, armor_decode, armor_encode, ghost_capacity, ghost_decode, ghost_encode,
    validate_encode_dimensions, JpegImage,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn real_photos_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("real_photos")
}

/// Collect all image files from the real_photos directory.
/// Accepts any common image extension — conversion happens later.
fn collect_image_files() -> Vec<PathBuf> {
    let dir = real_photos_dir();
    if !dir.exists() {
        return vec![];
    }
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let ext = path.extension()?.to_str()?.to_lowercase();
            match ext.as_str() {
                "jpg" | "jpeg" | "jfif" | "png" | "heic" | "heif" | "webp" | "tiff"
                | "tif" | "gif" | "bmp" => Some(path),
                _ => None,
            }
        })
        .collect();
    files.sort();
    files
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

/// Load an image file as baseline JPEG bytes ready for the Rust engine.
///
/// 1. Reads the raw file bytes
/// 2. Tries to parse directly with `JpegImage::from_bytes`
/// 3. If that fails (progressive JPEG, non-JPEG format), uses macOS `sips`
///    to convert to baseline JPEG — mirroring what the frontends do with
///    their native APIs (UIImage, BitmapFactory, canvas)
///
/// Returns `(jpeg_bytes, was_converted)` or an error message.
fn load_as_baseline_jpeg(path: &Path) -> Result<(Vec<u8>, bool), String> {
    let data = std::fs::read(path).map_err(|e| format!("read error: {e}"))?;

    // Try direct parse first
    if JpegImage::from_bytes(&data).is_ok() {
        return Ok((data, false));
    }

    // Convert via Python PIL to baseline JPEG
    let tmp = std::env::temp_dir().join(format!(
        "phasm_test_{}_{:?}.jpg",
        path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .replace(' ', "_"),
        std::thread::current().id(),
    ));

    // Use Python PIL to convert any image to baseline JPEG, mirroring what
    // the frontends do with their native image APIs.
    // PIL's Image.save with progressive=False guarantees baseline JPEG output.
    let script = format!(
        "from pillow_heif import register_heif_opener; register_heif_opener(); from PIL import Image; Image.open('{}').convert('RGB').save('{}', 'JPEG', quality=92, progressive=False)",
        path.to_str().unwrap().replace('\'', "\\'"),
        tmp.to_str().unwrap().replace('\'', "\\'"),
    );

    let output = Command::new("python3")
        .args(["-c", &script])
        .output()
        .map_err(|e| format!("python3 failed to launch: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("PIL conversion failed: {stderr}"));
    }

    let converted = std::fs::read(&tmp).map_err(|e| format!("read converted: {e}"))?;
    let _ = std::fs::remove_file(&tmp); // cleanup

    // Verify the converted file parses
    JpegImage::from_bytes(&converted)
        .map_err(|e| format!("converted JPEG still fails to parse: {e}"))?;

    Ok((converted, true))
}

/// Short test message that fits in any reasonable image.
const SHORT_MSG: &str = "Phasm test";
/// Passphrase for tests that use one.
const TEST_PASS: &str = "correct-horse-battery-staple";

// ---------------------------------------------------------------------------
// Ghost mode
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Slow: runs on real photos in tests/real_photos/. Use `./build.sh test-all` to include.
fn ghost_with_passphrase_real_photos() {
    let files = collect_image_files();
    if files.is_empty() {
        eprintln!("No photos in real_photos/ — drop some images there to test");
        return;
    }

    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for path in &files {
        let label = file_label(path);

        let (data, converted) = match load_as_baseline_jpeg(path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  SKIP {label}: {e}");
                skipped += 1;
                continue;
            }
        };

        let tag = if converted { " (converted)" } else { "" };

        let img = JpegImage::from_bytes(&data).unwrap();
        let fi = img.frame_info();
        if validate_encode_dimensions(fi.width as u32, fi.height as u32).is_err() {
            eprintln!("  SKIP {label}{tag}: dimensions {}x{} outside valid range", fi.width, fi.height);
            skipped += 1;
            continue;
        }
        let cap = match ghost_capacity(&img) {
            Ok(c) if c >= SHORT_MSG.len() + 100 => c,
            Ok(c) => {
                eprintln!("  SKIP {label}{tag}: Ghost capacity too low ({c} bytes)");
                skipped += 1;
                continue;
            }
            Err(e) => {
                eprintln!("  SKIP {label}{tag}: capacity error: {e}");
                skipped += 1;
                continue;
            }
        };

        let stego = match ghost_encode(&data, SHORT_MSG, TEST_PASS) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: ghost_encode: {e}");
                failed += 1;
                continue;
            }
        };

        let decoded = match ghost_decode(&stego, TEST_PASS) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: ghost_decode: {e}");
                failed += 1;
                continue;
            }
        };

        assert_eq!(decoded, SHORT_MSG, "FAIL {label}: message mismatch");
        eprintln!("  PASS {label}{tag} (Ghost+pass, cap={cap})");
        passed += 1;
    }

    eprintln!("\nGhost+passphrase: {passed} passed, {skipped} skipped, {failed} failed out of {} photos", files.len());
    assert_eq!(failed, 0, "{failed} photos failed Ghost+passphrase encode/decode");
}

#[test]
#[ignore] // Slow: runs on real photos in tests/real_photos/. Use `./build.sh test-all` to include.
fn ghost_without_passphrase_real_photos() {
    let files = collect_image_files();
    if files.is_empty() {
        return;
    }

    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for path in &files {
        let label = file_label(path);

        let (data, converted) = match load_as_baseline_jpeg(path) {
            Ok(v) => v,
            Err(_) => { skipped += 1; continue; }
        };
        let tag = if converted { " (converted)" } else { "" };

        let img = JpegImage::from_bytes(&data).unwrap();
        let fi = img.frame_info();
        if validate_encode_dimensions(fi.width as u32, fi.height as u32).is_err() { skipped += 1; continue; }
        let cap = ghost_capacity(&img).unwrap_or(0);
        if cap < SHORT_MSG.len() + 100 { skipped += 1; continue; }

        let stego = match ghost_encode(&data, SHORT_MSG, "") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: ghost_encode (no pass): {e}");
                failed += 1;
                continue;
            }
        };

        let decoded = match ghost_decode(&stego, "") {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: ghost_decode (no pass): {e}");
                failed += 1;
                continue;
            }
        };

        assert_eq!(decoded, SHORT_MSG, "FAIL {label}: message mismatch (no pass)");
        eprintln!("  PASS {label}{tag} (Ghost, no pass)");
        passed += 1;
    }

    eprintln!("Ghost (no passphrase): {passed} passed, {skipped} skipped, {failed} failed");
    assert_eq!(failed, 0, "{failed} photos failed Ghost (no pass) encode/decode");
}

// ---------------------------------------------------------------------------
// Armor mode
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Slow: runs on real photos in tests/real_photos/. Use `./build.sh test-all` to include.
fn armor_with_passphrase_real_photos() {
    let files = collect_image_files();
    if files.is_empty() {
        return;
    }

    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for path in &files {
        let label = file_label(path);

        let (data, converted) = match load_as_baseline_jpeg(path) {
            Ok(v) => v,
            Err(_) => { skipped += 1; continue; }
        };
        let tag = if converted { " (converted)" } else { "" };

        let img = JpegImage::from_bytes(&data).unwrap();
        let fi = img.frame_info();
        if validate_encode_dimensions(fi.width as u32, fi.height as u32).is_err() { skipped += 1; continue; }
        let cap = match armor_capacity(&img) {
            Ok(c) if c >= SHORT_MSG.len() + 100 => c,
            Ok(c) => {
                eprintln!("  SKIP {label}{tag}: Armor capacity too low ({c} bytes)");
                skipped += 1;
                continue;
            }
            Err(_) => { skipped += 1; continue; }
        };

        let stego = match armor_encode(&data, SHORT_MSG, TEST_PASS) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: armor_encode: {e}");
                failed += 1;
                continue;
            }
        };

        let (decoded, quality) = match armor_decode(&stego, TEST_PASS) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: armor_decode: {e}");
                failed += 1;
                continue;
            }
        };

        assert_eq!(decoded, SHORT_MSG, "FAIL {label}: message mismatch");
        assert_eq!(quality.integrity_percent, 100, "FAIL {label}: integrity not 100%");
        eprintln!("  PASS {label}{tag} (Armor+pass, cap={cap})");
        passed += 1;
    }

    eprintln!("Armor+passphrase: {passed} passed, {skipped} skipped, {failed} failed");
    assert_eq!(failed, 0, "{failed} photos failed Armor+passphrase encode/decode");
}

#[test]
#[ignore] // Slow: runs on real photos in tests/real_photos/. Use `./build.sh test-all` to include.
fn armor_without_passphrase_real_photos() {
    let files = collect_image_files();
    if files.is_empty() {
        return;
    }

    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for path in &files {
        let label = file_label(path);

        let (data, converted) = match load_as_baseline_jpeg(path) {
            Ok(v) => v,
            Err(_) => { skipped += 1; continue; }
        };
        let tag = if converted { " (converted)" } else { "" };

        let img = JpegImage::from_bytes(&data).unwrap();
        let fi = img.frame_info();
        if validate_encode_dimensions(fi.width as u32, fi.height as u32).is_err() { skipped += 1; continue; }
        let cap = armor_capacity(&img).unwrap_or(0);
        if cap < SHORT_MSG.len() + 100 { skipped += 1; continue; }

        let stego = match armor_encode(&data, SHORT_MSG, "") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: armor_encode (no pass): {e}");
                failed += 1;
                continue;
            }
        };

        let (decoded, _) = match armor_decode(&stego, "") {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  FAIL {label}{tag}: armor_decode (no pass): {e}");
                failed += 1;
                continue;
            }
        };

        assert_eq!(decoded, SHORT_MSG, "FAIL {label}: message mismatch (no pass)");
        eprintln!("  PASS {label}{tag} (Armor, no pass)");
        passed += 1;
    }

    eprintln!("Armor (no passphrase): {passed} passed, {skipped} skipped, {failed} failed");
    assert_eq!(failed, 0, "{failed} photos failed Armor (no pass) encode/decode");
}

// ---------------------------------------------------------------------------
// Cross-mode rejection
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Slow: runs on real photos in tests/real_photos/. Use `./build.sh test-all` to include.
fn cross_mode_rejection_real_photos() {
    let files = collect_image_files();
    if files.is_empty() {
        return;
    }

    for path in &files {
        let (data, _) = match load_as_baseline_jpeg(path) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let img = JpegImage::from_bytes(&data).unwrap();
        let fi = img.frame_info();
        if validate_encode_dimensions(fi.width as u32, fi.height as u32).is_err() { continue; }
        let ghost_cap = ghost_capacity(&img).unwrap_or(0);
        let armor_cap = armor_capacity(&img).unwrap_or(0);

        if ghost_cap < SHORT_MSG.len() + 100 || armor_cap < SHORT_MSG.len() + 100 {
            continue;
        }

        let label = file_label(path);

        // Ghost-encoded -> Armor decode should fail
        let ghost_stego = ghost_encode(&data, SHORT_MSG, TEST_PASS).unwrap();
        let armor_result = armor_decode(&ghost_stego, TEST_PASS);
        assert!(
            armor_result.is_err() || armor_result.as_ref().unwrap().0 != SHORT_MSG,
            "{label}: Armor should not decode a Ghost message"
        );

        // Armor-encoded -> Ghost decode should fail
        let armor_stego = armor_encode(&data, SHORT_MSG, TEST_PASS).unwrap();
        let ghost_result = ghost_decode(&armor_stego, TEST_PASS);
        assert!(
            ghost_result.is_err() || ghost_result.as_ref().unwrap() != SHORT_MSG,
            "{label}: Ghost should not decode an Armor message"
        );

        eprintln!("  PASS {label}: cross-mode rejection verified");
        return;
    }

    eprintln!("  SKIP: no photo with sufficient capacity for both modes");
}
