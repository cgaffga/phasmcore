# phasm-core: convenience targets for building from source.
#
# Everything here is a thin wrapper around plain cargo commands; BUILDING.md
# lists the same commands for use without make (Windows, CI, scripts).
#
#   make                 # this help
#   make check-deps      # which build tools are present / missing
#   make build-video     # library with every video codec
#   make cli-video       # the `phasm` CLI with H.264 + AV1
#
# Variables: JOBS=4 (cargo -j), CARGO=cargo, PROFILE=release.
#
# Every target that builds `av1-encoder` runs cargo through
# scripts/video-cargo.sh, which patches the crates.io package `rav1e` to the
# phasm rav1e fork on GitHub at the pinned commit (BUILDING.md, "The rav1e
# fork"). Image-only targets run plain cargo and never fetch rav1e.

CARGO   ?= cargo
PROFILE ?= release
JOBS    ?=

VIDEO_FEATURES     := h264-encoder,av1-encoder,av1-decoder
VIDEO_CARGO        := CARGO="$(CARGO)" sh scripts/video-cargo.sh
CLI_VIDEO_FEATURES := h264-encoder,av1-encoder

ifeq ($(PROFILE),release)
PROFILE_FLAG := --release
else
PROFILE_FLAG := --profile $(PROFILE)
endif
ifneq ($(strip $(JOBS)),)
JOBS_FLAG := -j $(JOBS)
endif
CARGO_FLAGS := $(PROFILE_FLAG) $(JOBS_FLAG)
TARGET_DIR  ?= $(if $(CARGO_TARGET_DIR),$(CARGO_TARGET_DIR),target)
BIN         := $(TARGET_DIR)/$(PROFILE)/phasm

SMOKE_CLIP  := test-vectors/video/h264/encoder-corpus/mandelbrot_x264_crf23_fast.mp4
SMOKE_DIR   := $(TARGET_DIR)/smoke-video

.DEFAULT_GOAL := help
.PHONY: help check-deps check-deps-video submodules build build-video \
        check-video-backends cli cli-video install-cli install-cli-video \
        test test-video smoke-video publish-dry-run clean

help:
	@echo "phasm-core build targets (plain cargo equivalents in BUILDING.md)"
	@echo ""
	@echo "  make check-deps          report missing build tools, with install hints"
	@echo "  make submodules          fetch the vendored C codec sources (git submodules)"
	@echo ""
	@echo "  make build               library, default features (image stego only)"
	@echo "  make build-video         library + H.264 encoder + AV1 encoder/decoder"
	@echo "  make cli                 the phasm CLI, image stego  -> $(BIN)"
	@echo "  make cli-video           the phasm CLI with H.264 + AV1 video"
	@echo "  make install-cli[-video] cargo install the CLI into ~/.cargo/bin"
	@echo ""
	@echo "  make test                library tests, default features"
	@echo "  make test-video          H.264 + AV1 encode/decode round-trip tests"
	@echo "  make smoke-video         CLI video-encode + decode on a sample clip"
	@echo "  make publish-dry-run     package the crates.io flavour in a temp copy"
	@echo "  make clean               cargo clean"
	@echo ""
	@echo "Variables: JOBS=N (cargo -j N), PROFILE=release|dev, CARGO=cargo"
	@echo "Video builds need meson, ninja, a C/C++ compiler and nasm on x86_64,"
	@echo "the submodules, and network access to github.com for the rav1e fork."

check-deps:
	@sh scripts/check-deps.sh all

check-deps-video:
	@sh scripts/check-deps.sh video

# Works in a phasmcore clone. In the monorepo the submodules live one level
# up, and `git submodule` from core/ handles that too.
submodules:
	git submodule update --init --recursive

build:
	$(CARGO) build $(CARGO_FLAGS)

build-video: check-deps-video
	$(VIDEO_CARGO) build $(CARGO_FLAGS) --features $(VIDEO_FEATURES)
	@sh scripts/check-video-backends.sh --profile $(PROFILE) h264 av1

# Fails if OpenH264 or dav1d was built as a fail-fast stub (missing submodule).
check-video-backends:
	@sh scripts/check-video-backends.sh --profile $(PROFILE) h264 av1

cli:
	$(CARGO) build $(CARGO_FLAGS) -p phasm-cli
	@echo "built $(BIN)"

cli-video: check-deps-video
	$(VIDEO_CARGO) build $(CARGO_FLAGS) -p phasm-cli --features $(CLI_VIDEO_FEATURES)
	@sh scripts/check-video-backends.sh --profile $(PROFILE) h264 av1
	@echo "built $(BIN) (H.264 + AV1)"

install-cli:
	$(CARGO) install $(JOBS_FLAG) --path cli

install-cli-video: check-deps-video
	$(VIDEO_CARGO) install $(JOBS_FLAG) --path cli --features $(CLI_VIDEO_FEATURES)

test:
	$(CARGO) test $(CARGO_FLAGS) --lib

# Real round trips through the linked codecs: a stub build fails here even
# if the backend check above were skipped.
test-video: build-video
	$(VIDEO_CARGO) test $(CARGO_FLAGS) --features $(VIDEO_FEATURES) --lib streaming_session_oh264_
	$(VIDEO_CARGO) test $(CARGO_FLAGS) --features $(VIDEO_FEATURES) --test av1_stratified_probe

# End-to-end through the CLI. `phasm video-encode` reads the cover video with
# the ffmpeg/ffprobe programs at runtime, so they must be on PATH.
smoke-video: cli-video
	@mkdir -p $(SMOKE_DIR)
	$(BIN) video-encode $(SMOKE_CLIP) --codec h264 -m "phasm h264 smoke" -p smoke-pass -o $(SMOKE_DIR)/h264.mp4 --quiet
	@out=$$($(BIN) decode $(SMOKE_DIR)/h264.mp4 -p smoke-pass --quiet); \
	  if [ "$$out" = "phasm h264 smoke" ]; then echo "ok: H.264 round trip"; \
	  else echo "FAIL: H.264 decode returned: $$out"; exit 1; fi
	$(BIN) video-encode $(SMOKE_CLIP) --codec av1 -m "phasm av1 smoke" -p smoke-pass -o $(SMOKE_DIR)/av1.mp4 --quiet
	@out=$$($(BIN) decode $(SMOKE_DIR)/av1.mp4 -p smoke-pass --quiet); \
	  if [ "$$out" = "phasm av1 smoke" ]; then echo "ok: AV1 round trip"; \
	  else echo "FAIL: AV1 decode returned: $$out"; exit 1; fi

# Builds the crates.io package exactly as publish-crates.yml does, in a
# throwaway copy so this checkout stays untouched: strip, then
# `cargo publish --dry-run` (verifies the package builds; uploads nothing),
# then a build of the packaged manifest with `h264-decoder`.
publish-dry-run:
	@tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/phasm-publish.XXXXXX"); \
	  echo "copying to $$tmp"; \
	  tar --exclude=./target --exclude=./vendor --exclude=./.git -cf - . | tar -xf - -C "$$tmp" && \
	  python3 scripts/strip-for-crates-io.py "$$tmp" && \
	  ( cd "$$tmp" && $(CARGO) publish -p phasm-core --dry-run --allow-dirty $(JOBS_FLAG) && \
	    $(CARGO) build --release $(JOBS_FLAG) --features h264-decoder ) && \
	  echo "publish dry-run OK ($$tmp; delete when done)"

clean:
	$(CARGO) clean
