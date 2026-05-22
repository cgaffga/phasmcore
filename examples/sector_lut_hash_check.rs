// T3.4.D cross-platform hash check. Build for wasm32-wasip1, run
// via wasmtime to confirm the SectorLut classification produces the
// same hash on every platform.
//
//   cargo build --release --target wasm32-wasip1 \
//       --example sector_lut_hash_check
//   wasmtime target/wasm32-wasip1/release/examples/sector_lut_hash_check.wasm

fn main() {
    println!("{}", phasm_core::stego::armor::dft_payload::lut_cross_platform_hash_hex());
}
