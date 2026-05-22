// One-off binary that prints the T3.3 cost-kernel cross-platform
// SHA256 hash. Compile to wasm32-wasip1 and run via `wasmtime` to
// confirm the WASM SIMD128 path produces the same hash as native
// aarch64-NEON and x86_64-SSE2 builds.
//
// Usage:
//   cargo build --release --target wasm32-wasip1 \
//       --example cost_hash_check
//   wasmtime target/wasm32-wasip1/release/examples/cost_hash_check.wasm

fn main() {
    let hash = phasm_core::stego::cost::uniward::cost_kernel_test_hash_hex();
    println!("{}", hash);
}
