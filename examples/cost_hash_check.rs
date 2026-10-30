// One-off binary that prints the T3.3 cost-kernel cross-platform
// SHA256 hashes (current J-UNIWARD kernel, then legacy v1). Compile to wasm32-wasip1 and run
// via `wasmtime` to confirm the WASM SIMD128 path produces the same
// hashes as native aarch64-NEON and x86_64-SSE2 builds.
//
// With an image path argument it instead prints the SHA256 over
// `(flat_idx, cost bits)` of every Y AC coefficient of that JPEG under
// the Ghost encoder's costs (`compute_ac_costs_separable`) — the native
// counterpart of the wasm bridge's `__test_cost_positions_hash`.
//
// Usage:
//   cargo run --release --example cost_hash_check [image.jpg]
//   cargo build --release --target wasm32-wasip1 --example cost_hash_check
//   wasmtime target/wasm32-wasip1/release/examples/cost_hash_check.wasm

fn main() {
    if let Some(path) = std::env::args().nth(1) {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(&path).expect("read image");
        let img = phasm_core::JpegImage::from_bytes(&bytes).expect("parse JPEG");
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).expect("quant table");
        let pos = phasm_core::stego::cost::uniward::compute_ac_costs_separable(img.dct_grid(0), qt)
            .expect("positions");
        let mut h = Sha256::new();
        let mut n_pos = 0usize;
        for (fi, c) in pos.iter().enumerate().filter(|(_, c)| c.is_finite()) {
            h.update((fi as u32).to_le_bytes());
            h.update(c.to_bits().to_le_bytes());
            n_pos += 1;
        }
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        println!("{hex}  {n_pos} positions  {path}");
        return;
    }
    println!("{}", phasm_core::stego::cost::uniward::cost_kernel_test_hash_hex());
    println!("{}", phasm_core::stego::cost::uniward::cost_kernel_test_hash_hex_legacy_v1());
}
