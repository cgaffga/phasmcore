# Eval Run Manifest — Reproducibility Index

**Purpose:** seed → config → result traceability for the arXiv paper's
reproducibility-badge eligibility. Each row below is a discrete experiment;
the linked `RESULTS.md` carries the full narrative and the
`results.json` (where present) carries machine-readable numbers.

**How to read:** rows are grouped by paper section. The "Seeds" column is
the seed used for fold-split + classifier-init (where applicable). For
multi-seed runs, the seeds are listed.

**Reproducibility contract:** every row should be re-runnable from
`config.json` + the linked driver script + `uv sync` against the
project's pinned `pyproject.toml`. Where no `config.json` exists yet, the
RESULTS.md "Reproduce" section gives the exact command.

---

## Paper §6.2 — Single-message Phasm Ghost vs J-UNIWARD detector

Detector-trained-on-J-UNIWARD evaluated on Phasm Ghost — does the canonical
J-UNIWARD adversary catch our impl?

| Run | Date | Detector | Payload | n covers | Seeds | Headline | RESULTS |
|---|---|---|---|---|---|---|---|
| `2026-05-10-phasm-ghost-vs-effnet` | 2026-05-10 | EffNet-B0 (aletheia variant A) | Ghost @ max (~0.19 bpnzAC) + J-UNI@{0.1,0.4} | 5 + 5 + 5 | n/a (inference) | Phasm Ghost stego_prob delta +0.09 vs cover (less than J-UNI@0.10 delta +0.13). Phasm at least as stealthy as cost-optimal J-UNIWARD at similar payload. | [link](2026-05-10-phasm-ghost-vs-effnet/RESULTS.md) |
| `2026-05-11-path2-alaska2-effnet` | 2026-05-11 | EffNet-B0 (aletheia) | mixed | 200 | n/a | Color cover replication. | [link](2026-05-11-path2-alaska2-effnet/RESULTS.md) |
| `2026-05-11-path2-color-effnet` | 2026-05-11 | EffNet-B0 | mixed | 100 | n/a | Color cover scaling test. | [link](2026-05-11-path2-color-effnet/RESULTS.md) |
| `2026-05-10-path1c-trained-srnet-pf040-eval` | 2026-05-10 | SRNet from-scratch | 0.40 bpnzAC | BOSSbase 100 | 42 | Phasm Ghost detector test PE 0.148. Δ +0.012 over J-UNI reference. | [link](2026-05-10-path1c-trained-srnet-pf040-eval/RESULTS.md) |
| `2026-05-11-path1c-trained-srnet-pf010-eval` | 2026-05-11 | SRNet from-scratch | 0.10 bpnzAC (no curriculum) | BOSSbase 100 | 42 | PE 0.41. Phasm Δ +0.027 vs J-UNI Δ +0.096 (3.5× less). | [link](2026-05-11-path1c-trained-srnet-pf010-eval/RESULTS.md) |
| `2026-05-11-path1c-curriculum-pf020-eval` | 2026-05-11 | SRNet curriculum | 0.20 bpnzAC | BOSSbase 100 | 42 | Curriculum SRNet payload-matched headline. PE 0.297. | [link](2026-05-11-path1c-curriculum-pf020-eval/RESULTS.md) |
| `2026-05-11-srnet-juniward-pf010-curriculum2-s042` | 2026-05-11→12 | SRNet curriculum (40 epochs) | 0.10 bpnzAC | BOSSbase 100 | 42 | **NEGATIVE.** Longer training didn't break PE 0.40 floor (best val_pe 0.4025). | [link](2026-05-11-srnet-juniward-pf010-curriculum2-s042/RESULTS.md) |

## Paper §6.3 — Shadow-message security (N = 1..4)

Does adding shadow layers degrade the primary-message security?

| Run | Date | Detector | N | n covers | Seeds | Headline | RESULTS |
|---|---|---|---|---|---|---|---|
| `2026-05-11-path3-shadow-ablation` | 2026-05-11 | SRNet curriculum @ 0.20 | 0..4 (universal subset) | BOSSbase 40 | 42 | **Detection delta goes NEGATIVE with shadow count** (N=4 → Δ −0.119). Shadow LSB obfuscates the J-UNI signature. | [link](2026-05-11-path3-shadow-ablation/RESULTS.md) |
| `2026-05-11-path3-shadow-crossdetector` | 2026-05-11 | EfficientNet-B0 | 0..4 | BOSSbase 40 | n/a (inference) | Confirms cross-detector. | [link](2026-05-11-path3-shadow-crossdetector/RESULTS.md) |

## Paper §6.4 — Shadow-count deniability (decoy detection)

"Is there another shadow beyond the one I extracted?" — adversary tries to
detect a SECOND shadow layer when shown an N=1 vs N=2 paired stego.

| Run | Date | Attacker | n paired | Seeds | Headline AUC | RESULTS |
|---|---|---|---|---|---|---|
| `2026-05-11-path4-decoy-detection` | 2026-05-11 | Phase 4a single-feature | BOSSbase 160 | 42 | 0.572 (Phase 4a) | [link](2026-05-11-path4-decoy-detection/RESULTS.md) |
| `2026-05-11-path4b-shadow-classifier` | 2026-05-11 | Phase 4b multi-feature LR | BOSSbase 160 | 42 | 0.481 (Phase 4b) | [link](2026-05-11-path4b-shadow-classifier/RESULTS.md) |
| `2026-05-11-shadow-classifier-1vs2-stable-s042` | 2026-05-11 | Phase 4c dedicated CNN | BOSSbase 900 | 42 | Phase 4c stable run. | [link](2026-05-11-shadow-classifier-1vs2-stable-s042/RESULTS.md) |
| `2026-05-11-shadow-classifier-1vs2-stable-s123` | 2026-05-11 | Phase 4c dedicated CNN | BOSSbase 900 | 123 | (no RESULTS.md — see 4d aggregate) | — |
| `2026-05-11-shadow-classifier-1vs2-stable-s999` | 2026-05-11 | Phase 4c dedicated CNN | BOSSbase 900 | 999 | (no RESULTS.md — see 4d aggregate) | — |
| `2026-05-11-path4d-3seed-variance` | 2026-05-11 | Phase 4d (3-seed CNN aggregate) | BOSSbase 900×3 | 42,123,999 | **0.556 ± 0.003** (paper-grade CNN baseline) | [link](2026-05-11-path4d-3seed-variance/RESULTS.md) |
| `2026-05-11-path4e-handcrafted-minimal` | 2026-05-11 | Phase 4e MINIMAL XGBoost | ALASKA-2 color 387 | 2026 | **0.685 ± 0.018** | [link](2026-05-11-path4e-handcrafted-minimal/RESULTS.md) |
| `2026-05-11-path4e-handcrafted-minimal-seed12345` | 2026-05-11 | Phase 4e MINIMAL XGBoost | ALASKA-2 color 387 | 12345 | (no RESULTS.md — variance sanity check) | — |
| `2026-05-11-path4e-handcrafted-full` | 2026-05-11 | Phase 4e FULL XGBoost (depth=4 lr=0.1 n=300) | ALASKA-2 color 387 | 2026 | **0.701 ± 0.022** | [link](2026-05-11-path4e-handcrafted-full/RESULTS.md) |
| `2026-05-11-path4e-handcrafted-gray` | 2026-05-11 | Phase 4e gray XGBoost | BOSSbase 1496 | 2026 | **0.803 ± 0.009** (XGBoost ceiling) | [link](2026-05-11-path4e-handcrafted-gray/RESULTS.md) |
| `2026-05-11-path4f-cost-pool` | 2026-05-11 | Phase 4f white-box cost-pool (1-seed) | ALASKA-2 color 387 | 2026 | **0.553 ± 0.027** | [link](2026-05-11-path4f-cost-pool/RESULTS.md) |
| `2026-05-14-path4f-cost-pool-3seed` | 2026-05-14 | Phase 4f white-box cost-pool (3-seed) | ALASKA-2 color 387 | 12345,67890,99 | **0.551 [0.540, 0.562]** pooled | [link](2026-05-14-path4f-cost-pool-3seed/RESULTS.md) |
| `2026-05-14-path4f-cost-pool-gray` | 2026-05-14 | Phase 4f white-box cost-pool (gray, 3-seed) | BOSSbase 79 | 12345,67890,99 | **0.517 [0.489, 0.545]** (at chance) | [link](2026-05-14-path4f-cost-pool-gray/RESULTS.md) |
| `2026-05-14-path4e-mlp-gray` | 2026-05-14 | E4 MLP-on-handcrafted (4-seed) | BOSSbase 1496 | 2026,12345,67890,99 | **0.870 [0.864, 0.877]** ← **NEW CEILING** | [link](2026-05-14-path4e-mlp-gray/RESULTS.md) |
| `2026-05-14-path4e-mlp-color` | 2026-05-14 | E4 MLP-on-handcrafted (color sanity) | ALASKA-2 color 387 | 2026 | 0.517 (high-dim collapse) | (no RESULTS.md — single comparison cell) |

## Paper §6.5 — Capacity sweep

Capacity vs source / QF / image dim.

| Run | Date | Buckets | n images | Headline | RESULTS |
|---|---|---|---|---|---|
| `2026-05-12-capacity-sweep` | 2026-05-12 | BOSSbase QF75 gray, ALASKA-2 color, real-world (incl. user 02445.jpg) | 212 | Ghost ≈ 0.19 bpnzAC, Shadow ≈ 0.97 bpnzAC (cover-distribution-invariant). Shadow ≈ 5× Ghost. Rules-of-thumb: ghost_bytes ≈ nzAC_Y/42, shadow_bytes ≈ nzAC_Y/8. | [link](2026-05-12-capacity-sweep/RESULTS.md) |

## Paper §6.6 — Bit-exact cross-platform CI

(Lives in `core/.github/workflows/bit-exact.yml`, not under `eval/runs/`.)

First green run 2026-05-12 on 5 platforms (4 native + WASM). See
`marketing/research/20260512-bit-exact-cross-platform-CI.md` for details.

## Audit / verification runs

| Run | Date | Purpose | Verdict | RESULTS |
|---|---|---|---|---|
| `marketing/research/20260514-juniward-fix-audit.md` | 2026-05-14 | E1 — verify shipped J-UNIWARD includes Butora 2023 off-by-one fix | 🟢 code-level pass; numerical comparison inconclusive (impl-level noise > off-by-one shift) | [link](../../marketing/research/20260514-juniward-fix-audit.md) |

## Smoke tests / harness validation (not paper-bound)

| Run | Date | Purpose |
|---|---|---|
| `2026-05-10-smoke-test` | 2026-05-10 | Path 1a EffNet pipeline validation |
| `2026-05-10-srnet-smoke-pf010-s042` | 2026-05-10 | Path 1c SRNet harness validation |
| `2026-05-10-path1a-n100-effnet` | 2026-05-10 | Path 1a scaling sanity check |
| `2026-05-10-path1b-srnet-suniward` | 2026-05-10 | Path 1b S-UNIWARD detector — **NULL RESULT** (transfer didn't work) |

## Runs without RESULTS.md (intermediate / superseded)

These are training-only run dirs that fed into a parent analysis run.
The parent run's RESULTS.md aggregates them; the per-run dir holds the
checkpoints + logs but not a narrative.

- `2026-05-10-srnet-juniward-pf010-s042` → feeds `2026-05-11-path1c-trained-srnet-pf010-eval`
- `2026-05-10-srnet-juniward-pf040-s042` → feeds `2026-05-10-path1c-trained-srnet-pf040-eval`
- `2026-05-11-path1c-curriculum-pf010-eval` → ANALYSIS of curriculum-trained model (single-seed)
- `2026-05-11-srnet-juniward-pf010-curriculum-s042` → 20-epoch curriculum, superseded by curriculum2 (40 epochs)
- `2026-05-11-srnet-juniward-pf020-curriculum-s042` → training-only dir, feeds curriculum-pf020-eval

## Reproducibility action items (W5)

These items extend this manifest into a TIFS-badge-eligible artifact:

- [ ] Add `config.json` per run dir (current docs cover most but not all).
- [ ] Compute SHA256 hash of each dataset slice and pin in `REPRO_DATASETS.md`.
- [ ] Pin `phasm-core` git SHA per run via `eval/bin/MANIFEST.md`.
- [ ] Build `eval/Dockerfile` with pinned Python + Rust + jpeglib + conseal.
- [ ] Tag `paper-v1.0` on the `phasm` repo at submission time.

See `marketing/research/20260514-arxiv-paper-submission-todo.md` §W5.

## Document History

- 2026-05-14: Initial manifest. 34 run dirs indexed. Phase 4 ceiling moved
  to AUC 0.870 (MLP) from 0.803 (XGBoost).
