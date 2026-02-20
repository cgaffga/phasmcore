use phasm_core::{armor_encode, armor_decode, armor_capacity, smart_decode, JpegImage};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("../test-vectors/{name}")).unwrap()
}

#[test]
fn armor_roundtrip_basic() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Hello, Armor mode!";
    let passphrase = "test-passphrase-123";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(quality.mode, 0x02, "should be Armor mode");
    assert_eq!(quality.rs_errors_corrected, 0, "no recompression = no errors");
    assert_eq!(quality.integrity_percent, 100);
}

#[test]
fn armor_wrong_passphrase_fails() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = armor_encode(&cover, "secret", "correct-pass").unwrap();

    let result = armor_decode(&stego, "wrong-pass");
    assert!(result.is_err(), "decoding with wrong passphrase should fail");
}

#[test]
fn armor_capacity_positive() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = armor_capacity(&img).unwrap();

    assert!(cap > 0, "capacity should be positive for 320x240");
    assert!(cap < 3000, "capacity {cap} suspiciously high for Armor");
}

#[test]
fn armor_stego_is_valid_jpeg() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = armor_encode(&cover, "test", "pass").unwrap();

    let img = JpegImage::from_bytes(&stego).unwrap();
    let frame = img.frame_info();
    assert_eq!(frame.width, 320);
    assert_eq!(frame.height, 240);
}

#[test]
fn armor_roundtrip_unicode() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Héllo wörld! 🔐";
    let passphrase = "unicode-key";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(quality.integrity_percent, 100);
}

#[test]
fn smart_decode_detects_ghost() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "ghost message";
    let passphrase = "shared-pass";

    let stego = phasm_core::ghost_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(quality.mode, 0x01, "should detect Ghost mode");
    assert_eq!(quality.integrity_percent, 100, "Ghost is always 100%");
    assert_eq!(quality.rs_errors_corrected, 0, "Ghost has no RS");
}

#[test]
fn smart_decode_detects_armor() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "armor message";
    let passphrase = "shared-pass";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(quality.mode, 0x02, "should detect Armor mode");
    assert_eq!(quality.integrity_percent, 100);
}

#[test]
fn both_modes_have_positive_capacity() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let armor_cap = armor_capacity(&img).unwrap();
    let ghost_cap = phasm_core::ghost_capacity(&img).unwrap();

    assert!(armor_cap > 0, "Armor capacity should be positive");
    assert!(ghost_cap > 0, "Ghost capacity should be positive");
}
