// T3.6.E cross-platform hash check. Build for wasm32-wasip1, run
// via wasmtime to confirm the WASM SIMD128 (and scalar-fallback)
// path produces the same Viterbi hash as native aarch64-NEON and
// x86_64-SSE2 builds.
//
//   cargo build --release --target wasm32-wasip1 \
//       --example viterbi_hash_check
//   wasmtime target/wasm32-wasip1/release/examples/viterbi_hash_check.wasm

fn main() {
    println!("{}", phasm_core::stego::stc::embed::viterbi_test_hash_hex());
}
