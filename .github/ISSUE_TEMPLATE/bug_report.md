---
name: Bug report
about: Something doesn't work as documented or as expected
title: "[bug] "
labels: bug
---

## What you tried

<!-- A clear, minimal description of what you were trying to do. -->

## What happened

<!-- The actual behavior. Paste error messages verbatim, including
the stack trace if any. -->

## What you expected

<!-- The behavior you expected instead. -->

## Reproduction

<!--
Minimum steps to reproduce the issue. If a fixture is needed,
attach it (small synthetic fixture preferred — large real-world
samples will be requested separately under the test-vectors
policy).

Example:
1. `cargo run --release --features cabac-stego -- video-encode input.mp4 -m "x" -p "test" -o out.mp4`
2. `cargo run --release --features cabac-stego -- decode out.mp4 -p "test"`
3. Decode returns "..." instead of "x".
-->

## Medium affected

<!-- Tick the relevant medium. -->

- [ ] Image stego (JPEG / PNG-via-SI / Ghost / Armor)
- [ ] Video stego (H.264 / CABAC v2)
- [ ] Shadow messages
- [ ] CLI tool
- [ ] iOS / Android / WASM bridge
- [ ] Build / tooling

## Environment

- phasm version (commit SHA or tag):
- OS + architecture (e.g. `macOS 15 arm64`, `Ubuntu 24.04 x86_64`):
- Rust version (`rustc --version`):
- Feature flags used (`cargo build --features ...`):
- ffmpeg version (if relevant):

## Additional context

<!-- Logs, screenshots, suspected root cause, anything else. -->

<!--
Please do NOT report security findings here. Use the channel in
SECURITY.md (security@phasm.app or the GitHub "Report a
vulnerability" button on the Security tab).
-->
