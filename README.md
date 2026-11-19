# Building Change Detection in Multi-Temporal Satellite Images

**Course:** DATA 255 Deep Learning Vision · Fall 2026  
**Team:** BuildSafe Vision — Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

Compares a six-channel U-Net (**FC-EF**) with shared-encoder Siamese U-Nets on [LEVIR-CD](https://justchenhao.github.io/LEVIR/) for building construction and demolition.

---

## Results

| Method | Split | F1 | Notes |
| --- | --- | ---: | --- |
| Abs RGB difference | val | 0.129 | Non-learned reference |
| FC-EF (BCE+Dice, mean 3 seeds) | val | 0.718 | Principal comparison |
| FC-Siam-diff (BCE+Dice, mean 3 seeds) | val | 0.543 | H1 not supported |
| FC-Siam-conc (BCE+Dice, seed 1511) | val | 0.773 | Single seed |
| FC-EF BCE (locked) | val → **test** | 0.778 → **0.780** | Chosen after freeze |

**H1** (Siam-diff > FC-EF) and **H2** (BCE+Dice > BCE) were **not supported** under this `base=16` protocol. Absolute difference is weaker than every learned model.

Figures: [`results/report_figures/`](results/report_figures/) · Gallery: [`results/experiments/full_unet_bce_aug_seed1511_tile256/validation_gallery.png`](results/experiments/full_unet_bce_aug_seed1511_tile256/validation_gallery.png)

---

## Deliverables

| Document | Path |
| --- | --- |
| Proposal | [`docs/Data255_ProjectProposal.pdf`](docs/Data255_ProjectProposal.pdf) |
| Final report (PDF) | [`report/DATA255_BuildSafeVision_IEEE.pdf`](report/DATA255_BuildSafeVision_IEEE.pdf) |
| Final report (DOCX) | [`report/DATA255_BuildSafeVision_IEEE.docx`](report/DATA255_BuildSafeVision_IEEE.docx) |
| Week 3 dataset card | [`docs/progress-report-week-3-4/week3_dataset_card.pdf`](docs/progress-report-week-3-4/week3_dataset_card.pdf) |
| Week 4 progress report | [`docs/progress-report-week-3-4/week4_progress_report.pdf`](docs/progress-report-week-3-4/week4_progress_report.pdf) |
| Reproduction evidence | [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) |
| Presentation (PDF / PPTX) | [`presentation/LEVIR_Change_Detection_DATA255.pdf`](presentation/LEVIR_Change_Detection_DATA255.pdf) · [`presentation/LEVIR_Change_Detection_DATA255.pptx`](presentation/LEVIR_Change_Detection_DATA255.pptx) |
| Presentation outline | [`docs/presentation/slides.md`](docs/presentation/slides.md) |
| Team work division | [`docs/TEAM_WORK_DIVISION.md`](docs/TEAM_WORK_DIVISION.md) |
| Individual statements | [`individual-contribution/`](individual-contribution/) |

Individual statements: [Sneha](individual-contribution/sneha_singh.md) · [Ritika](individual-contribution/ritika_mukesh_neema.md) · [Danish](individual-contribution/danish_waseem.md) · [Rajesh](individual-contribution/rajesh_paruchuri.md)

---

## Team roles (equal split)

| Member | Owns |
| --- | --- |
| **Sneha Singh** | Data pipeline, leakage audit, abs-diff baseline, protocol, Week 3 dataset card |
| **Danish Waseem** | FC-EF / Siamese models, GPU 9-run matrix, locked test, H1/H2 |
| **Ritika Mukesh Neema** | Configs, Week 4 progress report, validation gallery, responsible-use |
| **Rajesh Paruchuri** | Metrics, comparison/latency tables, figures, environment, reproduction |

Full checklist and cross-review pairs: [`docs/TEAM_WORK_DIVISION.md`](docs/TEAM_WORK_DIVISION.md).

---

## Status

| Deliverable | Status |
| --- | --- |
| Code (Daudt triad, losses, metrics, abs-diff, audits, tests) | Done |
| GPU 9-run matrix + comparison table | Done |
| Locked test (FC-EF BCE, seed 1511) | Done — test F1 **0.780** |
| Leakage audit (stem overlap = 0) | Done |
| Report write-up + figures | Done |
| Assistance disclosure | Done (Claude noted in report) |
| Presentation decks (PDF/PPTX) | Done — [`presentation/`](presentation/) |
| Confidential peer assessment | Each member / Canvas (outside repo) |

No further training is required for submission. Numbers come only from `results/`.

---

## Dataset

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) — 637×1024² RGB pairs (Texas) with building change masks. Academic use only. Imagery is **not** in Git.

| Split | Pairs | 256² tiles |
| --- | ---: | ---: |
| Train | 445 | 7,120 |
| Val | 64 | 1,024 |
| Test | 128 | 2,048 |

Stem overlap across splits: **0** ([`results/levir_audit/stem_overlap_report.json`](results/levir_audit/stem_overlap_report.json)).

```bash
export LEVIR_DATA_ROOT="/path/to/levir-cd"   # train/val/test with A/, B/, label/
```

`results/levir_audit/split_manifest.csv` stores host-local image paths from the machine that built it. Stem-overlap counts are still valid. To rebuild paths on a new host after setting `LEVIR_DATA_ROOT`, re-run the dataset audit (`notebooks/01_dataset_audit.ipynb` or the audit helpers). Reported metric summaries under `results/` do **not** need that rebuild.

---

## Architecture

We follow the Daudt et al. (2018) change-detection triad and score with change-class F1 (Chen & Shi, 2020). Code: [`src/levir_cd/models.py`](src/levir_cd/models.py). Related work: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md). We do **not** reproduce STANet attention.

| Paper name | `MODEL` | What it does |
| --- | --- | --- |
| **FC-EF** | `unet` | Stack A and B as 6 channels → one U-Net |
| **FC-Siam-diff** | `siamese_diff` | Shared encoder; fuse with \|fA−fB\| |
| **FC-Siam-conc** | `siamese_conc` | Shared encoder; fuse by concatenating features |

All learned paths: logits → sigmoid → threshold **0.5** → change-class F1 / IoU / P / R. Width `base=16`. Checkpoint by validation F1.

![LEVIR-CD Building Change Detection Architecture — Daudt triad](docs/architecture.jpg)

```text
LEVIR pair (A, B, mask)
        │
        ▼
 256×256 tiles (+ optional synced flip/rot90)
        │
        ├── Absolute RGB difference (reference)
        └── FC-EF | FC-Siam-diff | FC-Siam-conc
                 │
                 ▼
        sigmoid → threshold 0.5 → change-class F1 / IoU / P / R
```

---

## Setup

```bash
git clone https://github.com/danwaseem/levir-change-detection.git
cd levir-change-detection

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Quick check (no LEVIR needed):

```bash
PYTHONPATH=src python scripts/run_smoke.py
PYTHONPATH=src python -m pytest -q
```

Do **not** commit smoke folders or `.pt` checkpoints.

---

## Reproduce experiments

| Goal | Command | Needs LEVIR? |
| --- | --- | --- |
| Smoke (code only) | `PYTHONPATH=src python scripts/run_smoke.py` | No |
| Unit tests | `PYTHONPATH=src python -m pytest -q` | No |
| Abs-diff baseline (val) | `PYTHONPATH=src python scripts/run_baseline.py` | Yes |
| Full 9-run matrix | `PYTHONPATH=src python scripts/run_plan.py --mode full` | Yes (GPU) |
| Build comparison table | `PYTHONPATH=src python scripts/run_plan.py --aggregate-only` | Summaries on disk |
| Locked test (once) | `PYTHONPATH=src python scripts/run_locked_test.py --checkpoint …/best_model.pt --i-confirm-choices-are-frozen` | Yes |

The full matrix already ran on the university GPU lab. Re-running is optional. Artifact check: [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). Reported configs: [`configs/INDEX.md`](configs/INDEX.md).

---

## Repository layout

```text
.
├── README.md
├── individual-contribution/   # per-member contribution statements
├── configs/                   # JSON configs for the nine reported GPU runs
├── docs/                      # proposal, related work, week3/4 Canvas, slide outline
├── presentation/              # final PPTX + PDF decks
├── report/                    # IEEE DOCX + PDF (+ build_ieee_report.py)
├── results/
│   ├── absdiff_baseline/
│   ├── levir_audit/           # stem overlap + manifest
│   ├── report_figures/        # loss / P-R / bar charts
│   ├── experiments/           # per-run summaries (+ locked gallery)
│   ├── locked_test/
│   └── comparison_table.md
├── src/levir_cd/
├── scripts/                   # run_smoke, run_plan, run_baseline, run_locked_test
├── notebooks/
└── tests/
```

---

## Credits

LEVIR-CD: Chen & Shi (2020). Academic use only.  
Team work details: [`docs/TEAM_WORK_DIVISION.md`](docs/TEAM_WORK_DIVISION.md).
