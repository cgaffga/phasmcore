# H.264 reference distributions for §6E-A6.5 stealth gate

Empirical mb_type / partition / direction histograms from real-world
H.264 sources. Used by §6E-A6.5 (#126) to verify phasm's encoder
output falls within the convex hull of plausible "in-the-wild"
distributions.

Sources are gitignored MP4s (per `feedback_test_vectors_no_big_commits.md`);
the histograms below are the committed artefacts. Regenerate via
`scripts/h264_mb_partition_histogram.py <source.h264>`.

## At-a-glance comparison

| Encoder | GOP | I% | P% | B% |
|---|---|---:|---:|---:|
| x264 medium-CRF23 (HandBrake default) | IBPBP M=2 | 3% | 53% | 43% |
| Lumix G9 1080p / 30fps | **IBBP M=3** | 7% | 27% | **67%** |
| DJI Mini2 2.7K / 24fps | **IPPPP** | 3% | 97% | **0%** |
| iPhone 6s 4K / 30fps (H.264 mode) | IPPPP | 3% | 97% | 0% |
| iPhone 5 1080p / 30fps (H.264) | IPPPP | 4% | 96% | 0% |
| iPhone 7 1080p / 30fps (H.264) | IPPPP | 3% | 97% | 0% |

External corroboration: Sharabayko & Markov 2017 surveyed nine
2016-flagship smartphones (iPhone 6s, Galaxy S7, Lumia 950XL, Xperia Z5
Premium, LG G5, Nexus 6P, HTC 10, Moto Z Force, Xiaomi Mi 5) and found
**all nine skip B-frames AND use only one reference frame in the DPB**.
That generalises our iPhone IPPPP observation across vendors and SoCs.
Full breakdown + per-vendor coding-tool matrix (CABAC/CAVLC, AQ, ME
region, intra/inter block sizes) lives in
`docs/research/sharabayko-2017-h264-smartphones.md` — read it before
designing the §6E-A6.5 stealth gate (#126), especially when picking a
single-encoder target vs. content-aware ε.

External adversary model: Xiang/Bestagini/Tubaro/Delp 2023 (H4VDM,
arXiv:2210.11549) is the L3 fingerprint adversary made concrete — an
8-layer transformer that takes mb_type-per-pixel + luma-QP-per-pixel +
frame-types + decoded I/DF frames per GOP and achieves **AUC=85.2**
open-set device matching on 35 devices, distinguishing even same-model
phones with different iOS versions. Direct implication for #126: the
gate must match per-pixel mb_type / QP maps + frame-type sequence
against the claimed-source device, not just aggregate %. Full details
+ phasm-specific reading in
`docs/research/h4vdm-2023-h264-device-matching.md`.

## Per-clip notes (content-dependent variation)

The committed Lumix histogram is from `P1024244.mp4` (9.5 s,
motion-heavy / panning content). For comparison, an earlier
26.8 s static-content Lumix clip (`P1024246.mp4`, NOT in the
committed source set) showed VERY different numbers:

| Metric | P1024244 (motion) | P1024246 (static) |
|---|---:|---:|
| B no-MV | 0.07% | 25.7% |
| B Bi rate | 37.6% | 1.7% |
| B L0/L1 | 22.9 / 39.5 | 43.2 / 55.1 |

This is a real signal: **content (not just encoder) drives the
distribution heavily**. A "match Lumix" stealth gate needs ε wide
enough to handle both static and motion-heavy content, OR phasm's
gate needs to be content-aware (different thresholds when source
has high motion). Tracked as a §6E-A6.5 design question for #126.

## B-frame partition shapes (only encoders with B-frames)

| Shape | x264-medium | Lumix G9 (motion) | Comment |
|---|---:|---:|---|
| 16x16 | 81.6% | **94.5%** | Lumix-motion uses MORE 16x16 (Bi-heavy at 16x16 covers it) |
| 16x8  | 5.7% | 3.0% | |
| 8x16  | 4.8% | 2.5% | |
| 8x8   | 7.9% | **0%** | **Lumix never uses 8x8** |
| sub-8x8 (4x4 etc) | 0% | 0% | Both encoders skip — `§6E-A6.4` confirmed unnecessary |

## B-frame directional usage

| Direction | x264-medium | Lumix G9 (motion) |
|---|---:|---:|
| L0 | 49.3% | 22.9% |
| L1 | 45.8% | 39.5% |
| Bi | **4.9%** | **37.6%** |

Lumix-on-motion picks bipred 7× more often than x264 medium —
the M=3 GOP shape (two B-frames between each P) means the middle
B has equally distant L0 + L1 anchors, making bipred near-optimal.

## P-frame partition shapes (all encoders)

| Shape | x264-medium | Lumix G9 (motion) | DJI Mini2 |
|---|---:|---:|---:|
| 16x16 | 49.9% | **84.4%** | 82.2% |
| 16x8  | 13.0% | 7.9% | **0%** |
| 8x16  | 11.9% | 6.7% | **0%** |
| 8x8   | 25.3% | **0%** | 17.8% |

DJI Mini2's "16x16 / 8x8 only" partition pattern is striking — the
drone encoder restricts itself to the two extremes, never picking
16x8 or 8x16. Possibly a fixed-mode-decision config in DJI's chip.

## Stealth implications for phasm §6E-A6.5

1. **One-target match would over-fit.** Distributions vary 2-4×
   between the three references. Any single ε=5pp gate against
   x264 alone would fail Lumix output (16x8/8x16 way off) and
   succeed only for x264-converted content.

2. **Phasm's chosen target should be the dominant in-the-wild H.264
   producer (= x264 medium).** That's the largest crowd. Footnote:
   if phasm output gets converted-via-HandBrake again it stays
   x264-shaped anyway, so x264 is the resilient target.

3. **§6E-A6.4 (sub-8x8) confirmed deferred.** All three references
   emit 0% sub-8x8.

4. **§6E-A6.3 (B_8x8) is borderline.** x264 uses 8x8 ~8% of B-MBs;
   Lumix uses **0%**. If phasm targets x264, §6E-A6.3 is needed; if
   targeting Lumix, §6E-A6.3 is over-engineering. Decision-pending
   for #126.

## File index

- `x264_reference_mb_histogram.txt` — IMG_4138 HEVC source → x264 medium-CRF23
- `x264_reference_mb_histogram_4273.txt` — IMG_4273 same pipeline
- `lumix_g9_1080p_reference_mb_histogram.txt` — Panasonic Lumix G9 P1024246 (real camera output)
- `dji_mini2_2_7k_reference_mb_histogram.txt` — DJI Mini2 DJI_0558 (real drone output)
