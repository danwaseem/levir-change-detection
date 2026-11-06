# Building Change Detection in Multi-Temporal Satellite Images

DATA 255 Deep Learning Vision, Fall 2026  
Team **BuildSafe Vision**: Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri

Compares a six-channel U-Net (FC-EF) with shared-encoder Siamese U-Nets on [LEVIR-CD](https://justchenhao.github.io/LEVIR/) for building construction/demolition. Proposal: [`docs/Data255_ProjectProposal.pdf`](docs/Data255_ProjectProposal.pdf). Report draft: [`report/report.md`](report/report.md).

---

## Status — what is done vs left

| Done | Still needed (teammate / GPU) |
| --- | --- |
| Full training + eval code (Daudt triad, losses, metrics, ablations) | Run full 9-job matrix on GPU |
| Dataset audit notebook + manifest builder + stem-overlap check | Paste GPU numbers into report tables |
| Abs-diff reference baseline scored on **val** (F1 **0.129**) | One locked **test** after freeze |
| Prior FC-EF U-Net val baseline (F1 **0.732**, seed 1511) | Mean ± seed spread for Siamese |
| Smoke tests + unit tests (14 passed) | Presentation / contribution statements |
| Report outline filled with real numbers; blanks for missing runs | — |
| Architecture figure (JPG main + SVG) | — |

Coding for the graded experiment plan is on `main`. **Do not invent Siamese / ablation / test scores.** Prefer university **CUDA GPU** for `run_plan.py --mode full` (Mac MPS is too slow for all 9 runs).

---

## Architecture overview

We follow the Daudt et al. (2018) change-detection triad and score with change-class F1 (Chen & Shi, 2020). Code: [`src/levir_cd/models.py`](src/levir_cd/models.py). Paper map: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md). We do **not** reproduce STANet attention.

**How data flows.** Each LEVIR sample is a pair of RGB images (A = earlier date, B = later date) plus a binary building-change mask. We cut 1024×1024 pairs into non-overlapping 256×256 tiles. Optional flips/rotations are applied the same way to A, B, and the mask. A non-learned absolute RGB difference baseline is the reference. Learned models are one of:

| Paper name | `MODEL` | What it does |
| --- | --- | --- |
| **FC-EF** | `unet` | Stack A and B as 6 channels → one U-Net |
| **FC-Siam-diff** | `siamese_diff` | Shared encoder on A and B; fuse with \|fA−fB\| |
| **FC-Siam-conc** | `siamese_conc` | Shared encoder; fuse by concatenating features |

All learned paths output 1-channel logits → sigmoid → threshold **0.5** → change-class F1 / IoU / precision / recall. Default width `base=16`. Checkpoint by validation F1.

![Architecture overview — FC-EF, FC-Siam-diff, FC-Siam-conc](docs/architecture-overview.jpg)

*Figure: Architecture overview.* Vector copy: [`docs/architecture.svg`](docs/architecture.svg).

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

## Local setup (do this first)

```bash
git clone https://github.com/danwaseem/levir-change-detection.git
cd levir-change-detection
git pull

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# optional Jupyter kernel
python -m ipykernel install --user --name levir-cd --display-name "Python (LEVIR-CD)"
```

**Dataset (LEVIR-CD, academic use only).** Not stored in Git (~2.3 GB zips).

```bash
export LEVIR_DATA_ROOT="/path/to/levir-cd"   # folder with train.zip, val.zip, test.zip
# optional: export LEVIR_PROJECT_ROOT="$PWD"
```

If `LEVIR_DATA_ROOT` is unset, `run_plan.py --mode full` looks under `../datasets/levir-cd` and can download once. Expected layout after extract: `A/`, `B/`, `label/` per split.

**Quick check (no LEVIR needed):**

```bash
PYTHONPATH=src python scripts/run_smoke.py
PYTHONPATH=src python -m pytest -q
```

Do **not** commit smoke folders, `.pt` checkpoints, or `results/smoke_summary.json`.

---

## How to run experiments

| Goal | Command | Needs LEVIR? |
| --- | --- | --- |
| Smoke (code only) | `PYTHONPATH=src python scripts/run_smoke.py` | No |
| Unit tests | `PYTHONPATH=src python -m pytest -q` | No |
| Abs-diff baseline (val) | `PYTHONPATH=src python scripts/run_baseline.py` | Yes |
| Full 9-run matrix | `PYTHONPATH=src python scripts/run_plan.py --mode full` | Yes |
| Build comparison table | `PYTHONPATH=src python scripts/run_plan.py --aggregate-only` | Summaries on disk |
| Locked test (once) | `PYTHONPATH=src python scripts/run_locked_test.py --checkpoint …/best_model.pt --i-confirm-choices-are-frozen` | Yes |

### University GPU lab (recommended for full matrix)

```bash
git pull
source .venv/bin/activate
# install CUDA torch for the lab, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .

export LEVIR_DATA_ROOT="/path/to/levir-cd"

# use tmux/screen — this takes hours
PYTHONPATH=src python scripts/run_smoke.py
PYTHONPATH=src python scripts/run_plan.py --mode full
PYTHONPATH=src python scripts/run_baseline.py
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
```

After freeze, one test pass, then copy back:

- `results/experiments/`
- `results/comparison_table.md`
- `results/absdiff_baseline/`
- `results/locked_test/`

Matrix contents: FC-EF + FC-Siam-diff at seeds 1511/42/123, one FC-Siam-conc, BCE-only ablation, no-aug ablation.

---

## Research question

Does a Siamese U-Net improve building-change F1 over a six-channel U-Net under the same protocol, and how do loss and paired augmentation affect accuracy and cost?

- **H1:** FC-Siam-diff beats FC-EF on change-class F1/IoU (maybe slower).
- **H2:** BCE+Dice beats BCE alone under class imbalance.

---

## Dataset

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) — 637×1024² RGB pairs, Texas, building change masks. Academic use only.

| Split | Pairs | 256² tiles |
| --- | ---: | ---: |
| Train | 445 | 7,120 |
| Val | 64 | 1,024 |
| Test | 128 | 2,048 |

Train/val only until freeze. Stem overlap across splits: **0** (`results/levir_audit/` after audit). Imagery is not in Git.

---

## Results so far (validation only)

| Method | Seed | F1 | IoU | P | R | Where |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Abs RGB difference | — | **0.129** | 0.069 | 0.096 | 0.193 | run locally → `results/absdiff_baseline/` |
| FC-EF U-Net | 1511 | **0.732** | 0.577 | 0.691 | 0.778 | [`results/unet_baseline/`](results/unet_baseline/) |

Siamese / ablations / test: **not run yet**.

---

## Repository layout

```text
.
├── README.md
├── requirements.txt / pyproject.toml
├── docs/          # proposal, related work, architecture figure
├── report/        # report.md
├── results/       # baselines + experiment summaries (no huge .pt in Git)
├── src/levir_cd/  # models, data, train, metrics, baseline, audit
├── scripts/       # run_smoke, run_plan, run_baseline, run_locked_test
├── notebooks/     # 01 audit, 02 experiments
└── tests/
```

---

## Credits

LEVIR-CD: Chen & Shi (2020). Academic use only. Team roles follow the proposal.
