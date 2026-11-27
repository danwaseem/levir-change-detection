<!-- Build: pandoc docs/progress_report.md -o docs/progress_report.pdf --pdf-engine=xelatex
     Word:  pandoc docs/progress_report.md -o docs/progress_report.docx
     Built PDF/DOCX are gitignored: course writeups stay on disk only (e3a55c8). -->
---
title: "Progress Report — Deep Learning for Building Change Detection in Multi-Temporal Satellite Imagery"
author: "Team BuildSafe Vision — Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri"
date: "DATA 255 Deep Learning Vision, Fall 2026 · 26 November 2026"
documentclass: extarticle
fontsize: 9pt
geometry: margin=0.6in
colorlinks: true
urlcolor: blue
header-includes:
  - \usepackage{etoolbox}
  - \usepackage[compact]{titlesec}
  - \AtBeginEnvironment{longtable}{\scriptsize}
  - \titlespacing*{\section}{0pt}{4pt}{2pt}
  - \setlength{\parskip}{1.5pt}
  - \setlength{\abovecaptionskip}{2pt}
  - \setlength{\belowcaptionskip}{2pt}
  - \setlength{\textfloatsep}{4pt}
  - \setlength{\intextsep}{4pt}
---

Repo: <https://github.com/danwaseem/levir-change-detection>. Every number below was read back from an executed run under `results/`.

# 1. Updated Scope and Methods

**Research question (unchanged).** Under a matched training and evaluation protocol, does a shared-encoder Siamese U-Net improve building-change F1 over a six-channel early-fusion U-Net, and how do loss design and paired augmentation affect accuracy and compute? **H1:** Siamese feature comparison improves F1/IoU over FC-EF, possibly at higher inference cost. **H2:** BCE+Dice beats BCE alone on rare changed pixels.

**Changes since the approved proposal.**

| Change | Why |
| --- | --- |
| Widened from "a Siamese U-Net" to the full Daudt triad: FC-Siam-**diff** *and* FC-Siam-**conc** | With diff alone a negative H1 would wrongly indict every Siamese design; conc separates *difference fusion* from *shared encoding* |
| Locked model uses **BCE**, not the proposal's BCE+Dice principal setting | Ablation 1 (validation only) put BCE at 0.778 vs BCE+Dice 0.702; decided before the freeze |
| Geographic-leakage audit reduced to a **filename-stem** overlap check | We could not source per-pair coordinates for the released pairs, so the promised neighbourhood audit is still outstanding; residual risk disclosed, not resolved |
| FC-Siam-conc and both ablations stayed **single-seed** | Proposal allowed confirmation "if compute allows"; budget went to three seeds on the principal comparison |
| All nine runs moved to the **university CUDA lab**; Apple MPS kept for smoke tests only | The proposal's M5/MPS feasibility run (val F1 0.732) was too slow for nine runs |
| **Added:** locked-test error strata, 10,000-draw tile bootstrap CI, no-GPU artifact verifier | Strengthens reliability claims the proposal only sketched |

**Methods implemented.** Reference baseline: absolute RGB difference, validation-selected threshold 0.4. Strong baseline: FC-EF, six-channel early-fusion U-Net. Investigation: FC-Siam-diff (fuse |f_A − f_B|) and FC-Siam-conc (fuse by concatenation), both shared-encoder. **No pretrained weights anywhere** — all models train from scratch at width `base=16`; input is 256×256 tiles cut from 1024² pairs. Parameter counts and costs are in §3.

**Training setup.** Adam, lr 1e-3, weight decay 1e-4, batch 4, <= 30 epochs, early stopping (patience 8, minimum 12), checkpoint on best validation F1, decision threshold fixed at 0.5. Augmentation: synchronized flips and 90° rotations applied identically to A, B and mask. Seeds 1511, 42, 123.

**Status: the investigation is complete.** All nine GPU runs finished, settings were frozen 2026-11-08, and the locked test was evaluated once. §5 is confirmation and extension, not the main line.

![Pipeline: official LEVIR-CD pair split, then non-overlapping 256×256 tiling; absolute-difference reference plus FC-EF / FC-Siam-diff / FC-Siam-conc scored by one metric module.](report/figures_compressed/architecture.jpg){width=44%}

**Reused vs. team-written.** *Reused:* PyTorch, NumPy, pandas, Matplotlib, Pillow; the LEVIR-CD dataset; the U-Net and Daudt architecture *descriptions* — no third-party change-detection code, no pretrained checkpoints. *Written by us:* all three models, `losses.py`, `metrics.py` (incl. boundary F1, strata, bootstrap), the tiling dataloader, the dataset/leakage audit, the absolute-difference reference, the training loop, the locked-test harness and its freeze flag, the artifact verifier, and the test suite.

# 2. Data and Evaluation Protocol

The official pair split is applied **first**; each 1024² pair is then cut into sixteen non-overlapping 256² tiles, so no tile can straddle two splits.

| Split | Pairs | 256² tiles | Role |
| --- | ---: | ---: | --- |
| Train | 445 | 7,120 | Learn weights |
| Val | 64 | 1,024 | Select model, ablations, thresholds |
| Test | 128 | 2,048 | One locked evaluation |

**Target distribution.** Change is rare and tile-sparse: changed pixels are **4.20%** of validation and **5.09%** of test; only 935 of 2,048 test tiles (45.7%) contain any change, and 1,113 are completely empty. Pixel accuracy is recorded as a diagnostic but never used for selection — an all-negative predictor already scores ~95%.

**Leakage.** Filename-stem overlap across splits is **0** (`results/levir_audit/stem_overlap_report.json`). This blocks identity leakage, not geographic correlation between neighbouring scenes; as above, we could not source coordinates for the released pairs, so that risk stays disclosed and unresolved.

**Preprocessing and augmentation.** Masks binarized at intensity >= 128; images scaled to [0,1]; 256² tiling keeps batch 4 affordable. Augmentation is geometric only and synchronized across A, B and mask — photometric jitter was rejected because photometric nuisance *is* what the task must learn to ignore.

**Data-quality findings.** (i) Non-binary grayscale mask values, handled by the >= 128 rule. (ii) `split_manifest.csv` stores host-local absolute paths, so it must be rebuilt per machine — portability, not correctness. (iii) Severe class imbalance, quantified above. (iv) No corrupted, duplicate or unreadable pairs; all A/B/label filenames and dimensions match.

**Metrics.** Primary: change-class F1 at threshold 0.5, pooled (micro-averaged) over all tiles. Secondary: change-class IoU, precision, recall, boundary F1 (one-pixel dilation minus erosion), per-tile distributions. Only the non-learned reference gets a validation-selected threshold (0.4); every learned model is scored at a fixed 0.5.

**Test-set policy — disclosed.** Selection, ablations, checkpointing and thresholds used **validation only**; settings were frozen 2026-11-08. **We have evaluated on the test set, once.** The run records `test_set_touched: true` and `retuned_after_test: false`, the entry point requires an explicit `--i-confirm-choices-are-frozen` flag, and nothing has been changed since. The strata and bootstrap analyses re-read stored per-tile predictions — no retraining, no reselection.

# 3. Baseline and Preliminary Results

Validation unless marked. "Seeds" is the number of completed runs behind the figure.

| Method (config) | Val F1 | IoU | P | R | Seeds | Params | Train | Batch-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute RGB difference (th 0.4) | 0.129 | 0.069 | 0.096 | 0.193 | — | 0 | — | — |
| FC-EF, BCE+Dice, aug | 0.718 ±0.011 | — | — | — | 3 | 483,169 | ~51 min | 0.0014 s |
| FC-Siam-diff, BCE+Dice, aug | 0.543 ±0.045 | — | — | — | 3 | 482,737 | ~47 min | 0.0021 s |
| FC-Siam-conc, BCE+Dice, aug | 0.773 | 0.630 | 0.836 | 0.719 | 1 | 563,889 | ~48 min | 0.0028 s |
| FC-EF, **BCE**, aug (ablation 1) | **0.778** | 0.636 | 0.751 | 0.806 | 1 | 483,169 | ~51 min | 0.0014 s |
| FC-EF, BCE+Dice, **no aug** (ablation 2) | 0.745 | 0.594 | 0.764 | 0.728 | 1 | 483,169 | — | — |
| FC-EF, BCE, aug — **locked test** | **0.780** | 0.639 | 0.770 | 0.790 | 1 | 483,169 | — | — |

Locked-model boundary F1 is 0.356 against a pixel F1 of 0.780: the model finds the right buildings far better than it traces their outlines.

![Locked FC-EF+BCE run: train/validation loss and validation F1/IoU; best epoch 13.](report/figures_compressed/locked_learning_curves.jpg){width=43%}

**What the curves show.** Training loss falls monotonically 0.163 → 0.051 over 21 epochs; validation loss bottoms at epoch 17 (0.0501) then rises, so mild overfitting starts there — but the selected checkpoint is epoch 13, chosen on F1. Validation F1 is strikingly unstable (0.524, 0.476, 0.578 … **0.778** at 13 … 0.553 at 15 … 0.600 at 21): a precision/recall see-saw at the fixed 0.5 threshold, precision swinging 0.49–0.93 while recall moves in anti-phase 0.33–0.88. The decision boundary moves, not the representation. Early stopping fired at 21 = best (13) + patience (8).

**Evidence the pipeline is correct.**

- **Shapes and ranges.** The trainer asserts a `(1,1,256,256)` logit probe before epoch 1; loaders assert 6-channel inputs in [0,1] and binary masks.
- **Loss scale.** A constant base-rate predictor has cross-entropy 0.174–0.201 nats at the observed 4.20–5.09% change rate; mean epoch-1 training BCE was 0.163 — just past that floor, ruling out a mis-scaled or mis-signed loss.
- **Non-learned floor.** The absolute-difference reference reaches only 0.129 against 0.70–0.78 for learned models, so the metric is not rewarding a degenerate predictor.
- **Automated suite.** 15 unit/smoke tests (14 pass, 1 skips without LEVIR on the host) cover all three architectures, both losses, metrics against hand-computed tp/fp/fn, and a full synthetic training run.
- **Independent check.** `scripts/verify_reported_metrics.py` re-derives ten reported numbers from stored summaries with no GPU; a second member ran it on a clean checkout (2026-11-18).
- **Defect found and fixed.** A self-audit found two scoring bugs: post-training analyses (latency, per-tile scores, threshold sweep) ran on *final-epoch* weights instead of the reported checkpoint, and the absolute-difference baseline never counted true negatives, collapsing its accuracy onto IoU. Both fixed with a regression test (PR #1, merged 2026-11-23); no headline number changed, since the locked test always loaded `best_model.pt` explicitly.

# 4. Initial Error Analysis

**Stratified results (locked test, no retrain).** Stratifying the 2,048 test tiles by ground-truth change area is far more informative than the headline.

| Stratum | Tiles | Pooled F1 | Mean tile F1 | Mean recall | Signal |
|:--------|:------|:----------|:-------------|:------------|:-----------------------------------------------|
| Empty (0 px) | 1,113 | — | — | — | 212 fire anyway — 19.0% false-alarm rate |
| Small (1–256 px) | 61 | 0.088 | 0.078 | 0.083 | 56 of 61 have recall < 0.5 |
| Large (>256 px) | 874 | 0.790 | 0.691 | 0.702 | Carries essentially the whole headline |
| All | 2,048 | 0.780 | 0.297 | 0.302 | Bootstrap 95% CI [0.770, 0.789] |

Pooled F1 (0.780) and mean per-tile F1 (0.297) differ by 2.6× because 59.1% of tiles score exactly zero — mostly empty tiles that fired and small changes that were missed.

**Six annotated examples** (locked test; index, ground-truth and predicted changed pixels).

| # | Tile | GT px | Pred px | P | R | F1 | Reading |
|:--|:-----|:------|:--------|:--|:--|:--|:-------------------------------------------------------------------------------|
| 1 | 72 | 13,553 | 12,658 | 0.995 | 0.930 | **0.961** | **Success.** Large isolated construction; 60 false-positive pixels |
| 2 | 939 | 252 | 293 | 0.853 | 0.992 | **0.917** | **Success on a small object** — small change is not inherently unlearnable |
| 3 | 95 | 0 | 14,289 | 0.000 | — | 0.000 | **Worst false alarm.** Empty tile painted with 14k change pixels |
| 4 | 33 | 1,131 | 23 | 0.000 | 0.000 | 0.000 | **Worst large miss.** 23 scattered pixels, none overlapping |
| 5 | 762 | 249 | 717 | 0.000 | 0.000 | 0.000 | **Localization failure.** Right tile, wrong place: 717 px, zero overlap |
| 6 | 720 | 493 | 5,417 | 0.087 | 0.957 | 0.160 | **Over-segmentation.** Finds the change, floods 4,945 extra pixels |

![Locked FC-EF+BCE validation examples (before, after, ground truth, prediction @0.5). Row 1: empty-tile false alarm. Row 2: partial hit with spurious blobs. Row 3: small structures missed almost entirely. Row 4: large changes found but with displaced boundaries.](report/figures_compressed/validation_gallery.jpg){width=34%}

**Mode 1 — small-object recall collapse.** Small tiles reach pooled F1 0.088 against 0.790 for large. *Suggests:* at `base=16` with four downsampling stages a 16×16 building survives as roughly one bottleneck cell, leaving the decoder nothing to upsample; example 2 shows it is learnable, so this reads as capacity/resolution rather than an impossible target. *Test:* re-run FC-EF at `base=32`, and separately at tile size 128 — if width alone fixes it the cause is capacity, if only finer tiling does, it is resolution.

**Mode 2 — false alarms on unchanged tiles.** 212 of 1,113 empty tiles fire. *Suggests:* the model has partly learned "building-like texture is present" rather than "the built environment differs between A and B" — a shortcut early fusion makes easy, since a 6-channel stack can answer from one date alone. *Test:* feed identical pairs (A, A) from empty tiles; a true date-difference detector must return an empty mask, so any firing measures the shortcut directly. Compare against FC-Siam-conc, whose shared encoder must compare two encodings.

**Mode 3 — operating-point instability.** The precision/recall see-saw above means a fixed 0.5 threshold is not stable across checkpoints. *Suggests:* reported F1 partly reflects where the sigmoid happens to sit, not representation quality alone. *Test:* sweep the threshold on validation for the selected checkpoint and report a curve, not a point; the protocol threshold stays 0.5 everywhere so no retuning leaks in. Boundary F1 0.356 vs pixel F1 0.780 fits all three modes: the model localizes buildings but does not trace them.

# 5. Remaining Experiments, Risks, and Revised Plan

| Experiment | Question it answers | Status |
| --- | --- | --- |
| Abs-diff reference, val threshold sweep | Is a non-learned control sufficient? | Done — 0.129 |
| FC-EF vs FC-Siam-diff, BCE+Dice, 3 seeds | **H1**, main comparison | Done — 0.718 vs 0.543 |
| FC-Siam-conc, 1 seed | Is a negative H1 about *diff* or about Siamese generally? | Done — 0.773, single seed |
| Ablation 1: BCE vs BCE+Dice | **H2**, does Dice help the rare class? | Done — 0.778 vs 0.702 |
| Ablation 2: aug vs no-aug | Does synchronized geometric augmentation pay here? | Done — 0.702 vs 0.745 |
| Compute trade-off: params, train time, latency | Quality per unit of compute | Done — §3 |
| Locked test + strata + bootstrap CI | Does the frozen choice hold out of sample? | Done — 0.780, CI [0.770, 0.789] |
| FC-Siam-conc at seeds 42, 123 | Is conc's 0.773 real or a lucky seed? | **Planned — top priority** |
| FC-EF and FC-Siam-diff at `base=32` | Is the negative H1 a capacity artifact? | Planned |
| Identical-pair (A, A) probe on empty tiles | Failure mode 2 | Planned — no training needed |
| Gallery extended to >= 6 example rows | Wider qualitative coverage | Planned — needs checkpoint only |

**Variability.** Three seeds (1511, 42, 123) on the principal comparison, reported individually and as mean±std; FC-EF std is 0.011 against FC-Siam-diff's 0.045. Ablations and FC-Siam-conc are single-seed, labelled as such everywhere. We claim no significance from three seeds — it is a reliability check, and test-set uncertainty is reported separately as a bootstrap interval.

**Compute budget.** University CUDA lab for all nine reported runs; Apple M5/MPS for smoke tests and early feasibility only. Measured ~47–51 min per full run, so the completed matrix is roughly **7.5 GPU-hours**. Remaining: two FC-Siam-conc seeds (~1.6 h) plus two `base=32` runs (~2–2.5 h) = **~4 GPU-hours**; the (A, A) probe and gallery extension are forward passes only.

**Risks.** (1) *The negative H1 may be a capacity artifact* — at `base=16` we cannot separate "difference fusion is worse" from "it needs more width". *Mitigation:* the `base=32` pair; *fallback:* state the conclusion as explicitly width-bounded, as we already do. (2) *FC-Siam-conc rests on one seed*, 0.005 below the best FC-EF run, so a lucky seed could be carrying a headline. *Mitigation:* seeds 42 and 123; *fallback:* keep it labelled triad completion, excluded from every mean. (3) *Checkpoints are not in version control* — `best_model.pt` lives only on the team GPU store, so no one can re-score, regenerate the gallery or run the (A, A) probe from a clean checkout. *Mitigation:* publish the 2 MB checkpoint as a release asset; *fallback:* the committed per-tile CSVs already support every numeric claim.

**Timeline.** Done: freeze 2026-11-08 → reproduction check 11-18 → report and figures 11-21 → scoring-bug fix merged 11-23. Remaining: FC-Siam-conc seeds and the (A, A) probe in week 1; `base=32` pair and extended gallery in week 2; refreshed tables, figures and final submission in week 3.

**MVP complete; stretch goals outstanding.** Reference baseline, strong baseline, the Siamese investigation, two controlled ablations, a compute trade-off, three-seed reliability and one locked test with stratified error analysis are all archived. Stretch: multi-seed FC-Siam-conc, the `base=32` study, the shortcut probe, a boundary-aware loss, and a lat/long audit if coordinates can be obtained. We are not behind schedule; the honest caveat is that two of three headline comparisons still involve a single-seed arm.

# 6. Team Status

| Member | Completed | Next | Reviewed by |
| --- | --- | --- | --- |
| **Sneha Singh** | Tiling/split protocol; leakage audit (stem overlap 0); abs-diff reference; freeze-before-test procedure; Week 3 card | Lat/long audit if coordinates can be sourced; (A, A) probe design | Ritika — dataset card, stem overlap |
| **Danish Waseem** | All three model implementations; nine-run GPU matrix; locked-test execution; H1/H2 interpretation | FC-Siam-conc seeds 42, 123; `base=32` pair | Rajesh — locked-test F1/IoU, run summaries |
| **Ritika Mukesh Neema** | Config, seed and augmentation wiring plus the config index; `losses.py`; smoke packaging; validation gallery; responsible-use section; **found and fixed two scoring bugs (PR #1, merged)** | Gallery to >= 6 examples; boundary-aware loss prototype | Sneha — configs, loss/aug flags, gallery |
| **Rajesh Paruchuri** | `metrics.py` incl. boundary F1; comparison table; latency/compute; `environment.json`; figures; strata + bootstrap CI; artifact verifier; signed reproduction | Refresh tables/figures after the new runs; publish the locked checkpoint as a release asset | Danish — reproduction commands, table, figures |

Every member owns executed code or measurement, not only writing or coordination.

**Known evidence gap.** The committed gallery holds four image examples; the six required here are supplied numerically in §4 from the per-tile test metrics. Rendering six or more image rows needs `best_model.pt`, which is not in version control (risk 3) — a forward pass on the GPU host, not a retrain, assigned above.

# References

[1] H. Chen and Z. Shi, "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection," *Remote Sensing*, vol. 12, no. 10, art. 1662, 2020, doi: 10.3390/rs12101662. Dataset: <https://justchenhao.github.io/LEVIR/>.

[2] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, 2015, pp. 234–241, doi: 10.1007/978-3-319-24574-4_28.

[3] R. C. Daudt, B. Le Saux, and A. Boulch, "Fully convolutional Siamese networks for change detection," in *Proc. IEEE ICIP*, 2018, pp. 4063–4067, doi: 10.1109/ICIP.2018.8451652.

[4] H. Chen and Z. Shi, STANet, public research implementation. <https://github.com/justchenhao/STANet> (accessed Nov. 2026).

[5] Open-CD, LEVIR-CD benchmark tables (context only). <https://github.com/likyoo/open-cd> (accessed Nov. 2026).

[6] I. Corley, C. Robinson, and A. Ortiz, "A change detection reality check," arXiv:2402.06994, 2024 (context only).

[7] H. Chen, Z. Qi, and Z. Shi, "Remote sensing image change detection with transformers," *IEEE Trans. Geosci. Remote Sens.*, vol. 60, 2022, art. 5607514, doi: 10.1109/TGRS.2021.3095166 (context only).

[8] W. G. C. Bandara and V. M. Patel, "A transformer-based Siamese network for change detection," in *Proc. IEEE IGARSS*, 2022, pp. 207–210, doi: 10.1109/IGARSS46834.2022.9883686 (context only).

[9] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in *Proc. 3DV*, 2016, pp. 565–571, doi: 10.1109/3DV.2016.79.

Software: PyTorch 2.10.0, NumPy, pandas, Matplotlib, Pillow. No pretrained weights and no third-party change-detection implementations were used.

# Assistance Disclosure

Claude (Anthropic) was used as a reviewing and drafting aid for this progress report: it audited the committed result artifacts against the reported numbers, identified the two scoring defects described in §3 (fixed by the team in PR #1), extracted the six per-tile examples in §4 from `per_image_test_metrics.csv`, and drafted this document's prose. All experiments, training runs, measurements and figures are the team's own executed work, and every number here was read back from files under `results/` in the public repository.
