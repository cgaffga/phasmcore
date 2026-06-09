# Phasm paper reproducibility image (W5).
#
# Builds the runtime environment for reproducing every number in the
# Phasm Ghost arXiv paper:
#   - Rust 1.93.1 (pinned via rust-toolchain.toml) + phasm CLI binary
#     built image-stego-only (patent safety, see PATENT SAFETY block below)
#   - Python 3.12 + uv-managed venv from eval/pyproject.toml + eval/uv.lock
#     (PyTorch 2.x CPU, conseal, jpeglib, xgboost, scipy, Pillow, ...)
#   - System tooling: libjpeg-dev, p7zip-full (for BOSSbase 1.01.7z),
#     build-essential, git, curl
#
# Datasets (BOSSbase, ALASKA-II) are NOT baked into the image. They are
# fetched on first repro run via prep/download_*.py (or manually per
# anc/DATASET_HASHES.md) into a host-mounted volume.
#
# Build context = phasmcore root:
#   docker build -f eval/Dockerfile -t phasm-repro:v1.0 .
#
# Run interactively with a host data dir mounted:
#   docker run --rm -it \
#     -v "$PWD/data:/work/eval/data" \
#     -v "$PWD/runs:/work/eval/runs" \
#     phasm-repro:v1.0
#
# Inside the container:
#   phasm --version                              # CLI smoke test
#   cargo test --release --test cross_platform   # re-verify §6.6 bit-exact
#   cd eval && uv run python scripts/run_e10_shadow_pe.py
#
# Image size: ~5-7 GB (Rust toolchain ~2 GB + PyTorch CPU + Python deps
# ~3 GB + source ~50 MB). NO datasets, NO model checkpoints, NO stegos.

# ===========================================================
# PATENT SAFETY — Via LA AVC obligations
# ===========================================================
# The phasm CLI in this image is built with default features only —
# the SAME invocation cargo-dist uses for the public GitHub-release
# binary. ALL H.264 code paths — pure-Rust encoder, CABAC stego, the
# Baseline CAVLC parser/walker, OpenH264 FFI — are gated behind opt-in
# Cargo features (`video`, `h264-encoder`, `cabac-stego`,
# `h264-encoder`) and are OFF in this image.
#
# v0.2.9 (2026-05-19) tightened the posture: the `video` feature is
# now opt-in at the CLI level. Default `cargo build -p phasm-cli`
# compiles zero H.264 code, links zero H.264 dependencies (no
# memmap2 from phasm-core's `video`, no `core-openh264-sys`), and
# the binary's `phasm --help` does not list `video-encode` /
# `video-capacity` subcommands. Verify with `cargo tree -p phasm-cli`
# (post-build) — phasm-core appears, openh264-sys does not.
#
# DO NOT add `--features video`, `--features h264-encoder`,
# `--features cabac-stego`, or `--features h264-encoder` to the
# cargo build invocation below. If video stego repro is ever needed
# in v2, build from the monorepo with explicit feature flags — not
# from this image.
# ===========================================================

# Pin to linux/amd64 — most reviewers run x86_64, and several Python deps
# (notably jpeglib) only ship x86_64 manylinux wheels. On Apple Silicon
# the build runs under Rosetta emulation; slower but functional.
FROM --platform=linux/amd64 python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="phasm-repro"
LABEL org.opencontainers.image.description="Phasm Ghost arXiv paper reproducibility (image stego only)"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/cgaffga/phasmcore"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CARGO_TERM_COLOR=always

# ----------------------------------------------------------------
# System dependencies
# ----------------------------------------------------------------
# libjpeg-dev:   Pillow + Rust jpeg crates link against it
# p7zip-full:    extracts BOSSbase_1.01.7z (the canonical 1.3 GB archive)
# build-essential / pkg-config:  native extension builds (PyTorch, scipy)
# git / curl / ca-certificates:  rustup + uv installers, optional repo cloning
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libjpeg-dev \
        p7zip-full \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------
# Rust toolchain (rustup) — honors rust-toolchain.toml at WORKDIR
# ----------------------------------------------------------------
# Install rustup but DON'T pre-pick a channel; rust-toolchain.toml
# (copied below) will trigger the correct 1.93.1 download on first
# `cargo` invocation. Pinning happens in the project, not the image.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain none --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# ----------------------------------------------------------------
# uv (Astral) — fast Python package + venv manager
# ----------------------------------------------------------------
# Pins lockfile via `uv sync --frozen` below.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /work

# ----------------------------------------------------------------
# Rust source — copy before src/ so the toolchain pin takes effect
# before any cargo invocation
# ----------------------------------------------------------------
COPY rust-toolchain.toml ./
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY cli/ ./cli/
COPY examples/ ./examples/
COPY test-vectors/ ./test-vectors/
# openh264-sys is a phasm-core workspace member, so cargo needs its
# Cargo.toml to parse the workspace even when we build only -p phasm-cli.
# We deliberately do NOT copy vendor/phasm-openh264/ — openh264-sys's
# build.rs has a graceful stub-compile fallback when the submodule is
# absent, which is the same posture a reviewer fresh-cloning phasmcore
# sees (phasmcore intentionally does not mirror vendor/).
COPY openh264-sys/ ./openh264-sys/

# Trigger toolchain download via a no-op (gives clearer error if pin
# is unsatisfiable than discovering it during the build below).
RUN cargo --version && rustc --version

# ----------------------------------------------------------------
# Build phasm CLI — DEFAULT FEATURES ONLY.
# ----------------------------------------------------------------
# Same invocation as cargo-dist's GitHub-release CLI build. With v0.2.9
# the resulting binary is *strictly* image-stego-only: zero H.264 code
# compiled in, no `video-encode` / `video-capacity` subcommands in
# `phasm --help`, no `core-openh264-sys` linkage.
#
# DO NOT add `--features video`, `--features h264-encoder`,
# `--features cabac-stego`, or `--features h264-encoder` here — any
# of those would re-introduce H.264 code and diverge from the public
# cargo-dist GitHub-release binary.
RUN cargo build --release -p phasm-cli && \
    cp target/release/phasm /usr/local/bin/phasm && \
    phasm --version

# ----------------------------------------------------------------
# Python eval/ environment
# ----------------------------------------------------------------
# Copy lockfile + README first (hatchling validates readme path during
# the editable build of phasm-eval). Cache `uv sync` between source-only
# changes.
COPY eval/pyproject.toml eval/uv.lock eval/README.md ./eval/
# Install the [classifier] extra (scikit-learn + xgboost) — needed for
# §6.4 Phase 4 stat-aware attack reproduction. Skip [notebook] (jupyterlab,
# not needed for paper repro) and [tf-inference] (tensorflow, only for
# aletheia interop which is not in v1 repro scope).
RUN cd eval && uv sync --frozen --no-dev --extra classifier

# Now copy the rest of eval/ (sources change more often than deps)
COPY eval/scripts/   ./eval/scripts/
COPY eval/detectors/ ./eval/detectors/
COPY eval/prep/      ./eval/prep/
COPY eval/.gitignore ./eval/
COPY eval/runs/REPRO.md ./eval/runs/REPRO.md
COPY eval/bin/MANIFEST.md ./eval/bin/MANIFEST.md

# Smoke-test the Python env (import the heavy deps so any cold-start
# import error surfaces during image build, not on reviewer's first
# `docker run`).
RUN cd eval && uv run python -c \
    "import torch, torchvision, numpy, scipy, PIL, xgboost, conseal, jpeglib; \
     print(f'torch {torch.__version__}, numpy {numpy.__version__}, conseal {conseal.__version__}')"

# ----------------------------------------------------------------
# Bit-exact CI smoke test — verifies the build is consistent with
# the §6.6 reproducibility claim. Optional; runs in ~5 s.
# ----------------------------------------------------------------
RUN cargo test --release --test cross_platform 2>&1 | tail -20

# ----------------------------------------------------------------
# Default: drop into bash so reviewers can run interactively.
# ----------------------------------------------------------------
ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
