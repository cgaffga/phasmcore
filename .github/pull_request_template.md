<!--
Thanks for opening a pull request!

Please read CONTRIBUTING.md if you haven't already — it
explains the mirror-not-source flow, the medium-by-medium
contribution policy, and the cross-cutting determinism /
clean-room / crypto invariants.
-->

## Summary

<!-- 1-3 sentences explaining what this PR changes and why. -->

## Medium affected

<!-- Tick all that apply. -->

- [ ] Image stego (JPEG / Ghost / Armor / SI-UNIWARD / Shadow)
- [ ] Video stego (H.264 CABAC v2 / streaming orchestrator)
- [ ] New media type (please describe)
- [ ] Cross-cutting (det_math, crypto, stc, codec parser, FFI)
- [ ] Tooling (CLI, examples, build scripts)
- [ ] Documentation only

## Linked issue

<!-- e.g., "Closes #123" or "Refs #45". -->

## Test plan

<!--
What did you run to validate the change? Be specific. Examples:
  - `cargo test --features cabac-stego`
  - `cargo test --release --features cabac-stego --test h264_stego_streaming_orchestrator_v2`
  - `./build.sh test-all`
  - Manual: encoded a 1080p clip, decoded round-trip, payload preserved
-->

- [ ] Added a regression test (required for behavior changes).
- [ ] All existing tests pass with the new code.
- [ ] If perf-sensitive: ran a benchmark; results in PR description.

## Cross-cutting checklist

<!-- Reviewers will check these too — please confirm before requesting review. -->

- [ ] No `f64::sin/cos/atan2/hypot` introduced (use `det_math`).
- [ ] No FMA / non-deterministic SIMD in code paths affecting
      output bytes.
- [ ] No `HashMap` iteration in output-affecting paths.
- [ ] No new `unwrap()` / `expect()` on production paths.
- [ ] No new C/C++ FFI in `core/`.
- [ ] No large binary fixtures committed.

## Codec / clean-room attestation

<!--
Required only if you touched files under core/src/codec/<format>/
where <format> has a clean-room audit (currently: H.264 — see
docs/design/h264-encoder-algorithms/).

If applicable, confirm:
-->

- [ ] N/A — this PR does not touch a clean-room codec subtree.
- [ ] I followed the spec-as-reference workflow: studied the
      spec / upstream-as-reference, wrote an algorithm note,
      implemented from my own note. No copy-paste from third-
      party encoder source.
- [ ] I have NOT read x264 / libx264 source. (Or: I have, and
      I'm flagging it here so the maintainer can decide whether
      to accept.)

## DCO sign-off

By submitting this PR, I confirm that:

- [ ] I have signed every commit with `git commit --signoff`.
- [ ] I have the right to contribute this code under
      GPL-3.0-only.
