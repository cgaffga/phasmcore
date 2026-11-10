// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// build.rs for core-aom-sys: a static, encoder-only, 8-bit libaom from vendor/libaom (upstream, pinned
// by the submodule; ../vendor in the phasmcore mirror, ../../vendor in the monorepo), built with CMake
// + Ninja into OUT_DIR, plus shim/phasm_aom.c.
//
// CMake: $PHASM_CMAKE, else tools/cmake/bin/cmake of the checkout (tools/fetch_cmake.sh), else `cmake`
// on PATH. Ninja on PATH. nasm on PATH for x86_64 targets (the Android emulator).
//
// Targets: the host (tests, CLI), aarch64-apple-ios(-sim) (CMake's iOS support, deployment 17.0),
// Android (the NDK's CMake toolchain file; API 26, like the app's minSdk).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const IOS_DEPLOYMENT: &str = "17.0";
const ANDROID_API: &str = "26";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target = env::var("TARGET").unwrap_or_default();

    let candidates = [
        manifest_dir.join("..").join("vendor").join("libaom"),
        manifest_dir.join("..").join("..").join("vendor").join("libaom"),
    ];
    let src = candidates
        .iter()
        .find_map(|c| c.canonicalize().ok().filter(|p| p.join("CMakeLists.txt").exists()))
        .unwrap_or_else(|| {
            panic!(
                "vendor/libaom is missing (searched {:?}). Run `git submodule update --init vendor/libaom`.",
                candidates
            )
        });
    let cmake = find_cmake(&manifest_dir);

    let build = out_dir.join("aom-build");
    std::fs::create_dir_all(&build).expect("create aom-build");
    let mut args: Vec<String> = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=0",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DCONFIG_AV1_DECODER=0",
            "-DCONFIG_AV1_HIGHBITDEPTH=0",
            "-DCONFIG_TUNE_VMAF=0",
            "-DCONFIG_TUNE_BUTTERAUGLI=0",
            "-DENABLE_DOCS=0",
            "-DENABLE_EXAMPLES=0",
            "-DENABLE_TESTS=0",
            "-DENABLE_TESTDATA=0",
            "-DENABLE_TOOLS=0",
    ]
    .iter()
    .map(|a| a.to_string())
    .collect();
    args.extend(target_args(&target));
    // configure again whenever the settings (or the CMake used) change, not only on a clean build
    let stamp = build.join("phasm-configure.txt");
    let wanted = format!("{}\n{}", cmake.display(), args.join("\n"));
    if !build.join("build.ninja").exists() || std::fs::read_to_string(&stamp).ok().as_deref() != Some(wanted.as_str()) {
        let mut cfg = Command::new(&cmake);
        cfg.current_dir(&build).arg("-G").arg("Ninja").arg(&src).args(&args);
        run(&mut cfg, "cmake configure");
        std::fs::write(&stamp, &wanted).expect("write configure stamp");
    }
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "4".into());
    run(
        Command::new(&cmake).arg("--build").arg(&build).args(["--target", "aom", "-j", &jobs]),
        "cmake --build",
    );

    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-lib=static=aom");

    cc::Build::new()
        .file(manifest_dir.join("shim").join("phasm_aom.c"))
        .include(&src)
        .warnings(true)
        .compile("phasm_aom_shim");

    println!("cargo:rerun-if-changed=shim/phasm_aom.c");
    println!("cargo:rerun-if-changed={}", src.join("CMakeLists.txt").display());
    // the pinned libaom: a submodule move changes its HEAD
    println!("cargo:rerun-if-changed={}", src.join(".git").display());
    println!("cargo:rerun-if-env-changed=PHASM_CMAKE");
    for v in ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
}

fn find_cmake(manifest_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("PHASM_CMAKE") {
        return PathBuf::from(p);
    }
    for root in [manifest_dir.join("..").join(".."), manifest_dir.join("..")] {
        let c = root.join("tools").join("cmake").join("bin").join("cmake");
        if c.exists() {
            return c;
        }
    }
    if Command::new("cmake").arg("--version").output().is_ok() {
        return PathBuf::from("cmake");
    }
    panic!("CMake not found: run tools/fetch_cmake.sh (fills tools/cmake/), or set PHASM_CMAKE, or put cmake on PATH");
}

/// The cross-compile settings per Rust target; none for the host.
fn target_args(target: &str) -> Vec<String> {
    let ios = |sysroot: &str| {
        vec![
            "-DCMAKE_SYSTEM_NAME=iOS".into(),
            format!("-DCMAKE_OSX_SYSROOT={sysroot}"),
            "-DCMAKE_OSX_ARCHITECTURES=arm64".into(),
            format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={IOS_DEPLOYMENT}"),
            "-DAOM_TARGET_CPU=arm64".into(),
            // CMake's iOS mode makes every executable an app bundle, which libaom's install rules
            // (for aomenc, never built here) reject
            "-DCMAKE_MACOSX_BUNDLE=OFF".into(),
        ]
    };
    match target {
        // a Mac build for the other architecture (the host needs nothing)
        "aarch64-apple-darwin" | "x86_64-apple-darwin" if !env::var("HOST").unwrap_or_default().starts_with(target.split('-').next().unwrap_or("")) => {
            let arch = if target.starts_with("aarch64") { "arm64" } else { "x86_64" };
            vec![format!("-DCMAKE_OSX_ARCHITECTURES={arch}"), format!("-DAOM_TARGET_CPU={arch}")]
        }
        "aarch64-apple-ios" => ios("iphoneos"),
        "aarch64-apple-ios-sim" => ios("iphonesimulator"),
        t if t.contains("android") => {
            let abi = match t {
                "aarch64-linux-android" => "arm64-v8a",
                "x86_64-linux-android" => "x86_64",
                "armv7-linux-androideabi" => "armeabi-v7a",
                "i686-linux-android" => "x86",
                _ => panic!("core-aom-sys: unsupported Android target {t}"),
            };
            let ndk = ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME"]
                .iter()
                .find_map(|v| env::var(v).ok().filter(|s| !s.is_empty()))
                .expect("Android build needs ANDROID_NDK_HOME (or ANDROID_NDK_ROOT / NDK_HOME)");
            vec![
                format!("-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake"),
                format!("-DANDROID_ABI={abi}"),
                format!("-DANDROID_PLATFORM=android-{ANDROID_API}"),
            ]
        }
        _ => Vec::new(),
    }
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|e| panic!("{what}: cannot start: {e}"));
    assert!(status.success(), "{what} failed ({status})");
}
