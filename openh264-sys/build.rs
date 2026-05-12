// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// build.rs for core-openh264-sys.
//
// Drives a meson + ninja static build of the phasm-openh264 submodule
// at ../../vendor/phasm-openh264 and emits Cargo link directives so
// downstream Rust code can call into wels_stego.h via extern "C".
//
// Prerequisites checked at build time:
//   - vendor/phasm-openh264 submodule initialized (meson.build present)
//   - meson + ninja on PATH
//
// Behaviour:
//   - First build: meson setup _build_static --default-library=static
//                  --buildtype=release ; ninja -C _build_static
//   - Incremental: ninja -C _build_static (meson setup is idempotent
//     and skipped when the build dir already has build.ninja)
//   - Re-run trigger: any change under vendor/phasm-openh264/codec/

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // CARGO_MANIFEST_DIR = core/openh264-sys
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Locate submodule at <project root>/vendor/phasm-openh264.
    // From core/openh264-sys/, the project root is two levels up.
    let submodule_path = manifest_dir
        .join("..")
        .join("..")
        .join("vendor")
        .join("phasm-openh264");
    let submodule_path = submodule_path.canonicalize().unwrap_or_else(|e| {
        panic!(
            "phasm-openh264 submodule not found at {}: {}\n\
             Run `git submodule update --init --recursive` from the project root.",
            submodule_path.display(),
            e
        );
    });

    if !submodule_path.join("meson.build").exists() {
        panic!(
            "phasm-openh264 submodule directory exists at {} but contains no meson.build.\n\
             The submodule may not be checked out. Run\n\
             `git submodule update --init --recursive` from the project root.",
            submodule_path.display()
        );
    }

    // Phase B.8 step 5 — SHA pin. The Rust-side scan-table + hook-key
    // translation in `core-openh264-sys` depends on Phase A.5's hook
    // firing convention (raster within block for HOOK-A/B/E, POST-scan
    // for HOOK-F/G, partition_idx-as-plane for HOOK-C/D, etc.). If the
    // fork's hook insertion sites shift in a future Phase A.5 stage,
    // those tables silently go wrong. Pin to a known-good SHA so any
    // future fork update fails the build loudly and the maintainer
    // can re-validate empirically.
    //
    // To unpin during fork development on a feature branch, export
    //   PHASM_OPENH264_SHA_PIN_OVERRIDE=<sha>
    // or
    //   PHASM_OPENH264_SHA_PIN_OVERRIDE=*  (accept any SHA)
    {
        const PINNED_SHAS: &[&str] = &[
            // 2026-05-11: A.5(j) wired md_cost_capture (ABI 0x010100).
            "acf7db39843183f30771ffba2830f4861d73088c",
            // 2026-05-12: B.9.2.0 + .1 decoder-side emit helpers
            // (codec/common/inc/wels_stego_dec_helpers.h + impls in
            // wels_stego.cpp). No ABI bump; no encoder behavior change.
            "3ff049e3335b8f998341671b01cd38cab960e206",
            // 2026-05-12: B.9.2.2 CoeffSign decoder hook +
            // wels_stego_common.cpp split (state + accessors + dec
            // helpers move to libcommon so libdecoder TU resolves
            // them). Encoder helpers slimmed; direct g_phasm_callbacks
            // access via accessor functions. dec_post_read now fires
            // 106,725× on the 320×240 IDR fixture — exact walker match.
            "c266256d5ce2ab121fca0d0fcf4b74e74c11f7ba",
            // 2026-05-12: B.9.2.3 MvdSignBypass decoder hook at
            // ParseMvdInfoCabac (post-DecodeBypassCabac, inside the
            // |MVD|>0 branch). Verified by b9_2_3_dec_mvd_sign_
            // fires_match_walker: decoder fires 2 vs walker 2 on a
            // 3-frame IDR+P+P shifted-noise fixture.
            "a9102398a208a00711b324a2a843c84468a662d9",
            // 2026-05-12: B.9.2.4 MvdSuffixLsb decoder hook at
            // ParseMvdInfoCabac (post-sign-flip, |MVD|>=9 only).
            "e262a6cd65051d1c2fb7ad2a05014e571c7cfc26",
            // 2026-05-12: B.9.2.5 CoeffSuffixLsb decoder hook in
            // ParseSignificantCoeffCabac (|coeff|>=16 — phasm walker's
            // WET-eligible threshold). Closes B.9.2 fork-side: all 4
            // stego domains now fire decoder hooks bit-exact vs walker.
            "715ac507da3dcb37f222aaadce4e8e1c9b795744",
        ];
        let head_output = Command::new("git")
            .arg("-C")
            .arg(&submodule_path)
            .arg("rev-parse")
            .arg("HEAD")
            .output();
        if let Ok(out) = head_output {
            if out.status.success() {
                let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let override_env = std::env::var("PHASM_OPENH264_SHA_PIN_OVERRIDE").ok();
                let accept = override_env.as_deref() == Some("*")
                    || override_env.as_deref() == Some(&sha)
                    || PINNED_SHAS.iter().any(|s| *s == sha);
                if !accept {
                    panic!(
                        "phasm-openh264 submodule HEAD {} is not in the\n\
                         known-good SHA pin list {:?}.\n\
                         Phase B.8's Rust-side scan-table translation in\n\
                         core-openh264-sys depends on Phase A.5's hook firing\n\
                         convention. Verify empirically by re-running:\n\
                         \n\
                         cargo test --all-features --lib openh264::tests::b8_full_translation\n\
                         \n\
                         If it passes, add this SHA to PINNED_SHAS in\n\
                         core/openh264-sys/build.rs. If it fails, the fork's\n\
                         hook firing convention has shifted and the\n\
                         translation tables need updating.\n\
                         \n\
                         Override (for fork-side dev branches):\n\
                         PHASM_OPENH264_SHA_PIN_OVERRIDE={}\n\
                         PHASM_OPENH264_SHA_PIN_OVERRIDE=* (accept any)",
                        sha, PINNED_SHAS, sha
                    );
                }
                // Re-run build.rs if the submodule HEAD changes.
                println!(
                    "cargo:rerun-if-changed={}",
                    submodule_path.join(".git").display()
                );
            }
            // If `git rev-parse` fails (e.g. submodule isn't a git
            // checkout), skip the pin silently — the build can still
            // proceed, just without the safety check.
        }
        println!("cargo:rerun-if-env-changed=PHASM_OPENH264_SHA_PIN_OVERRIDE");
    }

    check_tool_available("meson", "Install via `brew install meson` (macOS) or `pip install meson` (Linux).");
    check_tool_available("ninja", "Install via `brew install ninja` (macOS) or `apt install ninja-build` (Linux).");

    let build_dir = out_dir.join("openh264-static");

    // Meson setup (skip if already configured).
    let needs_setup = !build_dir.join("build.ninja").exists();
    if needs_setup {
        std::fs::create_dir_all(&build_dir).expect("create build dir");
        let status = Command::new("meson")
            .arg("setup")
            .arg("--default-library=static")
            .arg("--buildtype=release")
            .arg("-Db_lto=false")  // LTO clashes with cargo's own LTO downstream
            .arg(&build_dir)
            .arg(&submodule_path)
            .status()
            .expect("failed to spawn meson");
        if !status.success() {
            panic!("meson setup failed (exit {:?}); check output above", status.code());
        }
    }

    // Ninja build. We deliberately build all default targets — Meson's
    // default_library=static still produces an aggregate libopenh264.a
    // plus per-component .a files, but the individual archive layout
    // varies by meson version, so the simplest portable thing is to
    // just build the whole project. The C++ unit tests are a sub-target;
    // they get built too but ignored for linking. (TODO follow-on: pass
    // --target libopenh264.a once we verify the artifact name is stable
    // across meson versions.)
    let status = Command::new("ninja")
        .arg("-C")
        .arg(&build_dir)
        .status()
        .expect("failed to spawn ninja");
    if !status.success() {
        panic!("ninja build failed (exit {:?}); check output above", status.code());
    }

    // OpenH264 with default-library=static produces an aggregate
    // libopenh264.a at the build root that bundles encoder + decoder +
    // common + processing. Linking that single archive avoids worrying
    // about cross-archive symbol resolution.
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=openh264");

    // Compile the C++ shims alongside libopenh264.a. Encoder shim wraps
    // ISVCEncoder, decoder shim wraps ISVCDecoder — both expose plain-C
    // entry points so the Rust side doesn't have to model the C++ vtable.
    // See core/openh264-sys/shim/phasm_{encoder,decoder}_shim.{h,cc}.
    //
    // Both shims compile into a single static library `phasm_shims` so the
    // cargo cc invocation runs once; the file order doesn't matter (both
    // TUs are independent).
    let shim_dir = manifest_dir.join("shim");
    cc::Build::new()
        .cpp(true)
        .file(shim_dir.join("phasm_encoder_shim.cc"))
        .file(shim_dir.join("phasm_decoder_shim.cc"))
        .include(&shim_dir)
        .include(submodule_path.join("codec/api/wels"))
        .flag_if_supported("-std=c++14")
        // Match OpenH264's release build flags closely.
        .flag_if_supported("-fno-rtti")
        .flag_if_supported("-fno-exceptions")
        .opt_level(2)
        .warnings(true)
        .compile("phasm_shims");
    println!("cargo:rerun-if-changed={}", shim_dir.display());

    // C++ stdlib link.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    } else if cfg!(target_os = "windows") {
        // MSVC bundles its stdlib; nothing to do.
    }

    // Expose include path so phasm-core can #include wels_stego.h via
    // bindgen later if we adopt it. For Phase B.3's hand-written
    // bindings, this is informational only.
    println!(
        "cargo:include={}",
        submodule_path.join("codec/api/wels").display()
    );

    // Trigger rebuild on any source/header change in the submodule.
    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("codec").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("meson.build").display()
    );
}

fn check_tool_available(tool: &str, install_hint: &str) {
    let status = Command::new(tool).arg("--version").output();
    match status {
        Ok(output) if output.status.success() => {}
        _ => panic!(
            "Required build tool `{}` not found on PATH.\n{}",
            tool, install_hint
        ),
    }
}
