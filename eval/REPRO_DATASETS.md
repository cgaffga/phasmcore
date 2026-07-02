# REPRO_DATASETS.md — Dataset integrity manifest

**Purpose.** Pin the SHA-256 of every *source* dataset behind the Phasm Ghost
arXiv paper, so a reviewer can verify their downloads match what we evaluated.
Closes the W5 action item in `runs/REPRO.md` ("compute SHA256 hash of each
dataset slice and pin it").

**What is pinned here:** the two source cover datasets (BOSSbase 1.01, ALASKA-II
covers). **What is *not*:** the derived slices (QF75/QF95 JPEG covers,
J-UNIWARD / Phasm stego, hand-crafted feature tables) — they regenerate
deterministically from the two sources via the `prep/` scripts + the pinned
`phasm` CLI (`bin/MANIFEST.md`) + the pinned Python env (`uv.lock`), inside
`eval/Dockerfile`.

Hashes computed 2026-06-29 on the eval host (`shasum -a 256`).

> **Canonical name.** This file is the dataset-hash manifest. The
> `eval/Dockerfile` header comment still points reviewers at
> `anc/DATASET_HASHES.md` — update that reference to `REPRO_DATASETS.md` when
> this lands (single source of truth).

---

## 1. BOSSbase 1.01 — primary grayscale cover source

| Field | Value |
|---|---|
| Archive | `data/bossbase/BOSSbase_1.01.7z` |
| Size | `1,405,584,456` bytes (1.31 GiB) |
| **SHA-256** | `5a772894316953edb440c25d3ccdd54690c377d616604fbedcf910a783248275` |
| Contents | 10,000 grayscale 512×512 PGM covers |
| Source | CTU mirror `http://agents.fel.cvut.cz/stegodata/PGMs.tgz`; Binghamton DDE Lab original `http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip` |
| Fetch script | `prep/download_bossbase.py` |

**Archive-format caveat (read this).** The SHA above is for the `.7z`
distribution we hold locally. `prep/download_bossbase.py` fetches the
`.tgz`/`.zip` mirrors, which have *different* archive-level SHAs but extract to
the **same 10,000-PGM cover set** — that PGM set is the load-bearing invariant,
not the container format. (`download_bossbase.py` shipped with
`EXPECTED_SHA256 = None`; this manifest supplies the pin for the `.7z` we used.)

**Derived slices used in the paper:** `data/bossbase/jpeg_qf75/` and
`jpeg_qf95/` (10,000 images each), produced from the PGMs by
`prep/encode_jpeg.py` (libjpeg pinned in the Docker image).

**Verify:**
```bash
shasum -a 256 data/bossbase/BOSSbase_1.01.7z
# -> 5a772894316953edb440c25d3ccdd54690c377d616604fbedcf910a783248275
```

---

## 2. ALASKA-II covers — color cross-source

| Field | Value |
|---|---|
| Slice | `data/alaska2/cover/` — the 484-image `Cover/` subset used in the paper |
| Files | 484 color JPEGs (`00001.jpg` … by Kaggle listing order) |
| **Slice digest** | `355cb1dff478a51e9c0abe85104e935a36803f9b623d0ac1c4977cf1ebaea6aa` |
| Digest method | content-only: `sha256` of the sorted list of per-file `sha256`s (path-independent) |
| Source | Kaggle competition `alaska2-image-steganalysis`, `Cover/` class |
| Fetch script | `prep/download_alaska_covers.py` (needs `~/.kaggle/kaggle.json`) |

**Derived slice:** `data/alaska2/jpeg_qf75/` via `prep/prepare_alaska_qf75.py`.

**Verify** (run from `eval/data/`):
```bash
find alaska2/cover -type f -iname '*.jpg' | sort \
  | xargs shasum -a 256 | awk '{print $1}' | shasum -a 256
# -> 355cb1dff478a51e9c0abe85104e935a36803f9b623d0ac1c4977cf1ebaea6aa
```

---

## 3. Out of scope (intentionally not pinned)

- **ImageNet / JIN pretrain** (`data/imagenet/`, `data/jin/`) — the JIN
  experiment (the §6.3 control) uses an ImageNet subset that is not
  redistributable; the JIN backbone reproduces from
  `prep/generate_jin_pretrain.py` against a disclosed substitute corpus (see
  that script and the JIN run's `RESULTS.md`).
- **Derived stego / feature tables** — regenerate from the §1–2 sources via the
  `prep/generate_*.py` scripts; deterministic given the pinned CLI + Python env.

---

## Document history

- 2026-06-29: Initial manifest — BOSSbase 1.01 `.7z` + ALASKA-II 484-cover
  subset pinned. Supplies the dataset-SHA item in the W5 repro checklist.
