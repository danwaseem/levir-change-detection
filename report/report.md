# Deep Learning for Building Change Detection in Multi-Temporal Satellite Imagery

**Course:** DATA 255 Deep Learning Vision, Fall 2026  
**Team:** BuildSafe Vision — Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri  
**Repository:** https://github.com/danwaseem/levir-change-detection  
**Report status:** Complete draft with verified citations. Filled cells use executed runs only. Empty cells (`—`) wait for the GPU `run_plan.py` matrix and the locked test. Do not invent those numbers.

---

## Abstract

Building change detection asks whether structures were constructed or removed between two satellite acquisitions of the same place. Simple image subtraction fails under illumination, seasonal, and acquisition shifts. This project studies binary building-change segmentation on LEVIR-CD under a matched protocol: non-overlapping 256×256 tiles, official train/val/test isolation, change-class F1 as the selection metric, and a fixed decision threshold of 0.5. We implement the Daudt triad—FC-EF (six-channel U-Net), FC-Siam-diff, and FC-Siam-conc—plus BCE versus BCE+Dice and paired geometric-augmentation ablations. A non-learned absolute RGB-difference baseline provides a reference floor.

**Measured so far (validation only):** absolute-difference F1 = **0.129** (threshold 0.4 selected on val); lightweight FC-EF U-Net F1 = **0.732** (seed 1511, 30 epochs, 483,169 parameters). Stem filenames do not overlap across splits. Siamese, loss/augmentation ablations, multi-seed means, and the locked test set remain **blank** until the GPU matrix finishes. We do not claim STANet attention results or published Open-CD / Reality-Check F1 numbers as this team’s scores.

---

## 1. Introduction and research question

Urban monitoring needs maps of *building* appearance and disappearance, not every pixel that looks different. Vegetation, shadows, and sensor conditions create large “visual change” without construction or demolition. A supervised change-detection model that outputs a dense binary mask is therefore more appropriate than global image differencing.

**Research question.** Under a matched training and evaluation protocol, does a shared-encoder Siamese U-Net improve building-change F1 over a conventional six-channel U-Net, and how do loss design and paired-image augmentation affect performance and computational cost?

**Hypotheses.**

- **H1.** Siamese feature extraction (FC-Siam-diff) improves change-class F1 and/or IoU relative to FC-EF by comparing before/after representations explicitly, possibly at higher inference latency.
- **H2.** BCE+Dice improves changed-pixel detection relative to BCE alone because changed pixels are rare on LEVIR-CD.

**Success** means producing credible, repeatable evidence for or against these claims—not promising a target score. This is an academic research prototype, not an operational monitoring or damage-assessment system.

---

## 2. Related work

We map three primary papers to concrete code choices. Full notes: [`docs/RELATED_WORK.md`](../docs/RELATED_WORK.md).

1. **Ronneberger et al. (2015)** introduced U-Net’s encoder–decoder with skip connections for dense prediction [1]. Our FC-EF path is a lightweight U-Net on concatenated bi-temporal RGB (`src/levir_cd/models.py`).

2. **Daudt et al. (2018)** proposed three fully convolutional change-detection networks: **FC-EF** (early fusion), **FC-Siam-diff** (shared encoder, absolute-difference skips), and **FC-Siam-conc** (shared encoder, concatenated skips) [2]. We implement that triad under one dataloader and metric API. Principal comparison: FC-EF vs FC-Siam-diff; FC-Siam-conc completes the triad.

3. **Chen and Shi (2020)** released LEVIR-CD and STANet, and emphasized change-class precision / recall / F1 under class imbalance [3]. We use LEVIR-CD and their metric spirit. We **do not** reproduce spatial–temporal attention (BAM/PAM) or batch-balanced contrastive loss; the proposal places STANet attention out of scope.

Later public tables (e.g. Open-CD; “A Change Detection Reality Check”) report FC-EF / FC-Siam test F1 roughly in the **0.81–0.86** range under heavier schedules and different widths [4,5]. Those numbers are **context only**. Our feasibility U-Net is narrower (`base=16`, ~0.48M parameters) and our locked test has not been run.

---

## 3. Dataset and evaluation protocol

### 3.1 Dataset

| Item | Value |
| --- | --- |
| Name | LEVIR-CD [3] |
| Source | https://justchenhao.github.io/LEVIR/ |
| Content | 637 bi-temporal RGB pairs, 1024×1024, ~0.5 m/px, Texas Google Earth scenes |
| Labels | Binary building construction / decline masks (~31k change instances reported by authors) |
| License / use | Academic use; respect Google Earth terms |
| Local access | Verified; archives under `../datasets/levir-cd` with MD5-checked download support in `audit.py` |

Official split → non-overlapping 256×256 tiles (16 tiles per pair):

| Split | Pairs | Tiles |
| --- | ---: | ---: |
| Train | 445 | 7,120 |
| Validation | 64 | 1,024 |
| Test | 128 | 2,048 |
| Total | 637 | 10,192 |

Masks are binarized at intensity ≥ 128. Training uses train/val only. Test stays isolated until a single locked evaluation after freezing model, loss, augmentation, and threshold.

### 3.2 Leakage and audit

Filename stem overlap across splits (`results/levir_audit/stem_overlap_report.json`):

| Comparison | Shared stems |
| --- | ---: |
| train vs val | **0** |
| train vs test | **0** |
| val vs test | **0** |

Zero stem overlap does **not** prove absence of nearby geographic scenes; residual spatial leakage remains a limitation.

### 3.3 Metrics

| Role | Metric |
| --- | --- |
| Primary | Change-class **F1** (Dice); selects checkpoints |
| Secondary | Change-class IoU, precision, recall; boundary F1; per-image summaries; small-building / empty-tile failure counts |
| Diagnostic only | Overall accuracy (not used for selection) |
| Decision threshold | Fixed **0.5** for learned models (no test tuning) |

---

## 4. Methods

![LEVIR-CD architecture: FC-EF, FC-Siam-diff, and FC-Siam-conc](../docs/architecture-overview.jpg)

**Figure 1.** Bi-temporal RGB inputs feed the Daudt triad. FC-EF concatenates A and B into one U-Net. FC-Siam-diff and FC-Siam-conc share one encoder and fuse with `|fA−fB|` or channel concatenation. All paths emit 1-channel logits scored at threshold 0.5 with change-class F1. Vector copy: [`docs/architecture.svg`](../docs/architecture.svg).

### 4.1 Reference baseline

Absolute mean RGB difference `|A−B|` with a threshold chosen by validation F1 sweep (`scripts/run_baseline.py`). Non-learned control for the “strong DL baseline must beat something simple” requirement.

### 4.2 Strong baseline — FC-EF

Six-channel U-Net: `concat(A,B)`. Width `base=16` → channels 16→32→64→128. Output: 1-channel logits. About **483,169** trainable parameters.

### 4.3 Team investigation — Siamese U-Nets

Tied encoder on A and B; fuse skips/bottleneck with absolute difference (`siamese_diff`) or concatenation (`siamese_conc`); shared decoder. Same tile interface (6-channel tensor split inside the network).

### 4.4 Losses and optimization

| Setting | Value |
| --- | --- |
| Losses | `bce` or `bce_dice` (= BCE + soft Dice) |
| Optimizer | Adam, lr `1e-3`, weight decay `1e-4` |
| Batch size | 4 |
| Full epochs | 30, early stop patience 8 after min 12 epochs |
| Augmentation | Synced horizontal/vertical flips and 90° rotations on (A, B, mask) |
| Checkpoint | Best **validation F1** |
| Seeds (main) | 1511, 42, 123 |

### 4.5 Out of scope

STANet attention, BCL loss, ImageNet/ResNet encoders, and claims of operational deployment.

---

## 5. Experimental setup

### 5.1 Experiment matrix

| Role | Model | Loss | Aug | Seeds |
| --- | --- | --- | --- | --- |
| Main | FC-EF (`unet`) | BCE+Dice | on | 1511, 42, 123 |
| Main | FC-Siam-diff | BCE+Dice | on | 1511, 42, 123 |
| Triad | FC-Siam-conc | BCE+Dice | on | 1511 |
| Loss ablation | FC-EF | BCE only | on | 1511 |
| Aug ablation | FC-EF | BCE+Dice | **off** | 1511 |
| Reference | Abs-diff | — | — | val threshold sweep |

Entry point: `PYTHONPATH=src python scripts/run_plan.py --mode full` then `--aggregate-only`.

### 5.2 Hardware / software (reported runs)

| Run | Device | Notes |
| --- | --- | --- |
| Prior FC-EF val baseline | Team machine (historical full 30-epoch run) | ~162 min; `results/unet_baseline/` |
| Abs-diff baseline | Mac (Nov 2026) | CPU/numpy over val tiles; ~1.2 min |
| Full 9-run matrix | **Planned: university CUDA GPU** | Mac MPS too slow for full matrix |

Software stack: Python 3.10+, PyTorch, NumPy, Pandas, Pillow (`requirements.txt`).

---

## 6. Results

### 6.1 Numbers from executed runs

| Method | Split | Seed | F1 | IoU | Precision | Recall | Params | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs RGB difference | val | — | **0.1286** | 0.0687 | 0.0965 | 0.1927 | 0 | `results/absdiff_baseline/` (thr=0.4) |
| FC-EF U-Net | val | 1511 | **0.7319** | 0.5772 | 0.6912 | 0.7778 | 483,169 | `results/unet_baseline/` (best epoch 24/30) |

FC-EF clearly beats the non-learned reference on validation (+0.60 F1). Test set was **not** used for these numbers.

Learning curves and a validation gallery: [`results/unet_baseline/learning_curves.png`](../results/unet_baseline/learning_curves.png), [`results/unet_baseline/validation_gallery.png`](../results/unet_baseline/validation_gallery.png).

### 6.2 Main comparison table (fill after GPU matrix)

| Model | Seed | Val F1 | Val IoU | Val P | Val R | Boundary F1 | Infer (s) | Synthetic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FC-EF | 1511 | 0.7319 | 0.5772 | 0.6912 | 0.7778 | — | — | no |
| FC-EF | 42 | — | — | — | — | — | — | — |
| FC-EF | 123 | — | — | — | — | — | — | — |
| FC-Siam-diff | 1511 | — | — | — | — | — | — | — |
| FC-Siam-diff | 42 | — | — | — | — | — | — | — |
| FC-Siam-diff | 123 | — | — | — | — | — | — | — |
| FC-Siam-conc | 1511 | — | — | — | — | — | — | — |

**Mean ± spread (main comparison, after three seeds):**

| Model | Mean Val F1 | Spread (max−min or ±std) |
| --- | ---: | ---: |
| FC-EF | — | — |
| FC-Siam-diff | — | — |

### 6.3 Ablation table (fill after GPU matrix)

| Setting | Model | Seed | Val F1 | Val IoU | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| BCE+Dice, aug on (reference row) | FC-EF | 1511 | 0.7319 | 0.5772 | from `unet_baseline` |
| BCE only, aug on | FC-EF | 1511 | — | — | H2 |
| BCE+Dice, aug **off** | FC-EF | 1511 | — | — | aug ablation |

### 6.4 Efficiency table (fill after GPU matrix)

| Model | Params | Train time | Checkpoint size | Batch-1 latency | Device |
| --- | ---: | ---: | ---: | ---: | --- |
| FC-EF | 483,169 | ~162 min (30 ep, historical) | — | — | prior run |
| FC-Siam-diff | — | — | — | — | — |
| FC-Siam-conc | — | — | — | — | — |

### 6.5 Qualitative examples

| Example type | Split | Status |
| --- | --- | --- |
| Clear hit (building appear/disappear) | val | See validation gallery; add captions after matrix |
| Missed small building | val | `failure_summary.json` after each full run |
| False positive on unchanged tile | val | same |
| Boundary error | val | boundary F1 column above |

**Caption rule:** always state split = validation, or test only after lock.

### 6.6 Locked test evaluation (once)

| Field | Value |
| --- | --- |
| Freeze date | — |
| Chosen model / loss / aug / seed | — |
| Checkpoint path | — |
| Test F1 / IoU / P / R | — / — / — / — |
| Retuned after seeing test? | **No** (required) |

Command (after freeze only):

```bash
PYTHONPATH=src python scripts/run_locked_test.py \
  --checkpoint results/experiments/<best_run>/best_model.pt \
  --i-confirm-choices-are-frozen
```

---

## 7. Discussion

**H1 (Siamese vs FC-EF).** Not yet answerable with this team’s multi-seed Siamese numbers (`—` in §6.2). The implemented comparison is fair under one protocol; evidence will be the mean±spread F1 gap and latency trade-off after the GPU matrix.

**H2 (BCE+Dice vs BCE).** Not yet answerable (`—` in §6.3). The loss ablation is coded and scheduled.

**What we can say now.** A lightweight FC-EF already outperforms absolute RGB difference by a large margin on validation (0.73 vs 0.13 F1), confirming that a learned change detector is necessary for this task under illumination/nuisance variation. Published ~0.81–0.86 test F1 figures use different widths and budgets; they are not our result.

**Limitations.** (1) Incomplete principal comparison until GPU runs finish. (2) Lightweight `base=16` vs wider public reimplementations. (3) Stem-overlap audit ≠ full geospatial leakage proof. (4) Texas-only LEVIR geography. (5) Abs-diff `accuracy` field in the JSON omits true negatives in the helper; change-class F1/IoU/P/R are the reported baseline metrics.

**Responsible use.** LEVIR-CD is for academic research. We do not claim safety-critical, legal, or surveillance deployment readiness.

---

## 8. Conclusion

We built a reproducible LEVIR-CD pipeline implementing the Daudt FC-EF / FC-Siam-diff / FC-Siam-conc triad, reference absolute-difference baseline, change-class F1 selection, and the proposed ablations. Executed evidence so far shows FC-EF validation F1 **0.7319** (seed 1511) and abs-diff F1 **0.129**, with no stem overlap across splits. The research question—whether Siamese fusion and loss/augmentation choices improve over FC-EF under a matched protocol—requires the remaining GPU matrix and one locked test. Claims will stay proportional to those completed runs.

---

## 9. Contributions and reproducibility

| Member | Primary technical ownership |
| --- | --- |
| Sneha Singh | Dataset integrity, geospatial/stem overlap checks, absolute-difference baseline |
| Ritika Mukesh Neema | U-Net / FC-EF pipeline, paired augmentation, seed reproducibility |
| Danish Waseem | Siamese U-Net, loss comparison, error analysis |
| Rajesh Paruchuri | Metrics, efficiency benchmarks, statistical summaries, reproduction check |

**Reproduce code (no LEVIR required):**

```bash
PYTHONPATH=src python scripts/run_smoke.py
PYTHONPATH=src python -m pytest -q
```

**Reproduce full experiments (LEVIR + GPU recommended):**

```bash
export LEVIR_DATA_ROOT=/path/to/levir-cd
PYTHONPATH=src python scripts/run_plan.py --mode full
PYTHONPATH=src python scripts/run_baseline.py
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
```

Cross-member reproduction: a teammate who did not author the component should regenerate one evaluation using the README and a saved `summary.json` (full retrain optional).

---

## 10. Assistance disclosure

- **Proposal stage:** OpenAI ChatGPT used for formatting and light structural organization; technical design and claims were team-authored and checked against sources.
- **Implementation / this report draft:** Cursor AI assisted with coding, citation checks, and drafting. All experimental numbers in filled cells come from executed artifacts under `results/`. Empty cells were left blank on purpose.
- No fabricated Siamese, ablation, or test scores.

---

## References

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351, Springer, 2015, pp. 234–241. doi: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).

[2] R. Caye Daudt, B. Le Saux, and A. Boulch, “Fully Convolutional Siamese Networks for Change Detection,” in *2018 25th IEEE International Conference on Image Processing (ICIP)*, 2018, pp. 4063–4067. doi: [10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652).

[3] H. Chen and Z. Shi, “A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection,” *Remote Sensing*, vol. 12, no. 10, art. 1662, 2020. doi: [10.3390/rs12101662](https://doi.org/10.3390/rs12101662). Dataset: https://justchenhao.github.io/LEVIR/.

[4] Open-CD project / toolbox documentation and LEVIR-CD benchmark tables (context only; not this team’s scores). See also https://github.com/likyoo/open-cd.

[5] I. Corley, C. Robinson, and A. Ortiz, “A Change Detection Reality Check,” arXiv:2402.06994, 2024. https://arxiv.org/abs/2402.06994 (context only).

---

## Appendix

### A. Artifact paths

| Artifact | Path |
| --- | --- |
| Proposal PDF | `docs/Data255_ProjectProposal.pdf` |
| Related-work map | `docs/RELATED_WORK.md` |
| Architecture figure | `docs/architecture-overview.jpg`, `docs/architecture.svg` |
| Split manifest | `results/levir_audit/split_manifest.csv` |
| Stem overlap | `results/levir_audit/stem_overlap_report.json` |
| Abs-diff summary | `results/absdiff_baseline/summary.json` |
| FC-EF baseline | `results/unet_baseline/` |
| Future matrix | `results/experiments/`, `results/comparison_table.md` |
| Locked test | `results/locked_test/` (after freeze) |

### B. Config snapshot (FC-EF historical baseline)

From `results/unet_baseline/summary.json`: 30 epochs, best epoch 24, threshold 0.5, 445/64 train/val pairs, 7120/1024 tiles, F1 selection only, `test_set_touched: false`.

### C. How to paste GPU outputs

After `run_plan.py --aggregate-only`, replace the `—` cells in §6.2–§6.4 with values from `results/comparison_table.md` and each run’s `summary.json` / `failure_summary.json`. Then run locked test once and fill §6.6.
