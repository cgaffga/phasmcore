use phasm_core::{ghost_encode, ghost_decode, ghost_capacity, JpegImage};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("../test-vectors/{name}")).unwrap()
}

#[test]
fn ghost_roundtrip_basic() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Hello, steganography!";
    let passphrase = "test-passphrase-123";

    let stego = ghost_encode(&cover, message, passphrase).unwrap();
    let decoded = ghost_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
}

#[test]
fn ghost_wrong_key_fails() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = ghost_encode(&cover, "secret msg", "correct-pass").unwrap();

    let result = ghost_decode(&stego, "wrong-pass");
    assert!(
        result.is_err(),
        "decoding with wrong passphrase should fail"
    );
}

#[test]
fn ghost_roundtrip_empty_message() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = ghost_encode(&cover, "", "pass").unwrap();
    let decoded = ghost_decode(&stego, "pass").unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn ghost_message_too_large() {
    // tiny_8x8 has only 1 block → 63 AC positions. Way too small.
    let cover = load_test_image("tiny_8x8_q95.jpg");
    let big_message = "x".repeat(2000);

    let result = ghost_encode(&cover, &big_message, "pass");
    assert!(
        result.is_err(),
        "encoding a huge message in a tiny image should fail"
    );
}

#[test]
fn ghost_small_image_too_small() {
    // 64x64 grayscale: 64 blocks × 63 AC = 4,032 positions.
    // MAX_FRAME_BITS = 8,600. w = floor(4032/8600) = 0 → ImageTooSmall.
    let cover = load_test_image("gray_64x64_q75.jpg");
    let result = ghost_encode(&cover, "test", "pass");
    assert!(result.is_err(), "64x64 grayscale should be too small");
}

#[test]
fn ghost_stego_is_valid_jpeg() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = ghost_encode(&cover, "test", "pass").unwrap();

    // Stego output must be parseable as a valid JPEG.
    let img = JpegImage::from_bytes(&stego).unwrap();
    let frame = img.frame_info();
    assert_eq!(frame.width, 320);
    assert_eq!(frame.height, 240);
}

#[test]
fn ghost_capacity_reasonable() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = ghost_capacity(&img).unwrap();

    // 320×240 should have substantial capacity.
    assert!(cap > 50, "capacity {cap} too low");
    assert!(cap < 5000, "capacity {cap} suspiciously high");
}

#[test]
fn ghost_roundtrip_various_lengths() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let passphrase = "multi-test";

    for len in [1, 10, 50, 100] {
        let message: String = (0..len).map(|i| (b'A' + (i % 26) as u8) as char).collect();
        let stego = ghost_encode(&cover, &message, passphrase).unwrap();
        let decoded = ghost_decode(&stego, passphrase).unwrap();
        assert_eq!(decoded, message, "failed for message length {len}");
    }
}

#[test]
fn ghost_roundtrip_unicode() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Héllo wörld! 日本語テスト 🔐";
    let passphrase = "unicode-key";

    let stego = ghost_encode(&cover, message, passphrase).unwrap();
    let decoded = ghost_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
}
