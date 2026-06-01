// Diagnostic: decode the user's stego MP4 (extracted to Annex-B) with
// the shadow passphrase "s" and report what's recoverable.

#![cfg(all(
    feature = "h264-encoder",
    feature = "openh264-backend",
    feature = "cabac-stego",
))]

use phasm_core::h264_stego_smart_decode_video;

#[test]
fn decode_user_stego_video_with_shadow_pass_s() {
    let annex_b = match std::fs::read("/tmp/stego.h264") {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: /tmp/stego.h264 missing ({e})");
            return;
        }
    };
    eprintln!("Annex-B size: {} bytes", annex_b.len());

    let start = std::time::Instant::now();
    let result_shadow = h264_stego_smart_decode_video(&annex_b, "s");
    let dt_shadow = start.elapsed();
    eprintln!("decode (pass='s'): {:?} (took {:?})", result_shadow, dt_shadow);

    let start = std::time::Instant::now();
    let result_empty = h264_stego_smart_decode_video(&annex_b, "");
    let dt_empty = start.elapsed();
    eprintln!("decode (pass=''): {:?} (took {:?})", result_empty, dt_empty);
}
