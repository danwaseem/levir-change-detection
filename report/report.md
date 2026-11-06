# Deep Learning for Building Change Detection in Multi-Temporal Satellite Imagery

DATA 255 Deep Learning Vision, Fall 2026  
Team **BuildSafe Vision**: Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri  
Repo: https://github.com/danwaseem/levir-change-detection

**Note on results.** Numbers in this draft come only from runs we already finished. Blank cells (`—`) are for the GPU experiment matrix and the locked test. We are not filling those in with made-up scores.

---

## Abstract

This project looks at building change detection: given two satellite images of the same location at different times, can a model mark where buildings were added or removed? Plain image subtraction does not work well because lighting, seasons, and vegetation also change. We use the LEVIR-CD dataset and treat the task as binary segmentation.

We compare three networks from Daudt et al. (2018): FC-EF (six-channel U-Net), FC-Siam-diff, and FC-Siam-conc. We also compare BCE vs BCE+Dice and training with vs without paired flips/rotations. For a simple reference, we run absolute RGB difference with a validation-chosen threshold.

What we have measured so far on **validation**:

- Absolute difference: F1 = **0.129** (threshold 0.4)
- Lightweight FC-EF U-Net: F1 = **0.732** (seed 1511, 30 epochs, 483,169 parameters)

Siamese results, ablations, and the test set are still empty. We are not claiming STANet numbers or published Open-CD / Reality Check scores as ours.

---

## 1. Introduction and research question

The practical need is to find building construction and demolition, not every pixel that looks different between two dates. Shadows, plants, and acquisition conditions create a lot of visual change that is not a building change. So we train a model that outputs a binary change mask.

**Research question.** With the same data splits, tiles, and metrics, does a shared-encoder Siamese U-Net get better building-change F1 than a normal six-channel U-Net? How much do the loss choice and paired augmentation matter, and what is the compute cost?

**Hypotheses.**

- **H1.** FC-Siam-diff should help F1/IoU over FC-EF because the two dates are compared in feature space, though inference may be slower.
- **H2.** BCE+Dice should help more than BCE alone when changed pixels are rare.

We treat success as answering these questions with fair experiments, even if some results are negative or mixed. This is a class research project, not a deployed monitoring product.

---

## 2. Related work

We mainly follow three papers. Longer notes are in [`docs/RELATED_WORK.md`](../docs/RELATED_WORK.md).

1. Ronneberger et al. (2015) proposed U-Net for dense prediction with skip connections [1]. Our FC-EF model is a small U-Net on stacked RGB pairs.

2. Daudt et al. (2018) compared FC-EF, FC-Siam-diff, and FC-Siam-conc for change detection [2]. We coded all three in `models.py` under one training loop. The main comparison in our proposal is FC-EF vs FC-Siam-diff. FC-Siam-conc is included so the full triad is covered.

3. Chen and Shi (2020) released LEVIR-CD and STANet, and used change-class precision, recall, and F1 [3]. We use their dataset and that metric style. We do **not** rebuild STANet attention or their contrastive loss; that was out of scope in the proposal.

Other benchmarks report FC-style models around test F1 0.81–0.86 under longer schedules and wider networks [4,5]. Those papers are useful context. They are not our scores. Our U-Net is narrow (`base=16`).

---

## 3. Dataset and evaluation protocol

### 3.1 Dataset

| Item | Value |
| --- | --- |
| Dataset | LEVIR-CD [3] |
| Link | https://justchenhao.github.io/LEVIR/ |
| Size | 637 RGB pairs, each 1024×1024, about 0.5 m/pixel, Texas scenes |
| Labels | Binary building change masks |
| Use | Academic only; follow Google Earth terms |
| Our copy | Downloaded and checked locally (`../datasets/levir-cd`) |

We keep the official split and cut each image into 16 non-overlapping 256×256 tiles:

| Split | Pairs | Tiles |
| --- | ---: | ---: |
| Train | 445 | 7,120 |
| Val | 64 | 1,024 |
| Test | 128 | 2,048 |

Masks are thresholded at intensity ≥ 128. We only use train/val while picking models. Test is for one final run after we freeze settings.

### 3.2 Split audit

We checked whether any filename stems appear in more than one split (`stem_overlap_report.json`):

| Pair of splits | Overlap count |
| --- | ---: |
| train / val | 0 |
| train / test | 0 |
| val / test | 0 |

No shared stems is good, but it does not prove that nearby geographic areas never leak across splits. That is still a limitation.

### 3.3 Metrics

| Use | Metric |
| --- | --- |
| Main | Change-class F1 (used to pick checkpoints) |
| Extra | IoU, precision, recall, boundary F1, per-image / failure summaries |
| Not for selection | Overall accuracy |
| Threshold | 0.5 for learned models (not tuned on test) |

---

## 4. Methods

![Architecture overview](../docs/architecture-overview.jpg)

**Figure 1.** Inputs A and B go into FC-EF, FC-Siam-diff, or FC-Siam-conc. FC-EF stacks the two RGB images. The Siamese models share an encoder and fuse with absolute difference or concatenation. Output is a one-channel change map. Vector copy: [`docs/architecture.svg`](../docs/architecture.svg).

### 4.1 Reference baseline

Mean absolute RGB difference `|A − B|`. We pick the threshold on validation by F1. Code: `scripts/run_baseline.py`.

### 4.2 Strong baseline (FC-EF)

Concatenate A and B to 6 channels and run a U-Net with base width 16 (16→32→64→128). About 483k parameters.

### 4.3 Team models (Siamese)

Same encoder weights for A and B. FC-Siam-diff uses `|fA − fB|`. FC-Siam-conc concatenates features. Shared decoder predicts the change map.

### 4.4 Training details

| Setting | Choice |
| --- | --- |
| Loss | BCE or BCE+Dice |
| Optimizer | Adam, lr 1e-3, weight decay 1e-4 |
| Batch size | 4 |
| Epochs | up to 30, early stopping |
| Augmentation | same flip / rot90 on A, B, and mask |
| Checkpoint | best validation F1 |
| Main seeds | 1511, 42, 123 |

---

## 5. Experimental setup

Planned matrix (`scripts/run_plan.py --mode full`):

| Role | Model | Loss | Aug | Seeds |
| --- | --- | --- | --- | --- |
| Main | FC-EF | BCE+Dice | on | 1511, 42, 123 |
| Main | FC-Siam-diff | BCE+Dice | on | 1511, 42, 123 |
| Triad | FC-Siam-conc | BCE+Dice | on | 1511 |
| Loss ablation | FC-EF | BCE | on | 1511 |
| Aug ablation | FC-EF | BCE+Dice | off | 1511 |

We started a full run on a Mac (MPS) but stopped it because one epoch was already ~16 minutes. The full nine runs will be done on a university GPU. Smoke tests and unit tests pass on the Mac.

---

## 6. Results

### 6.1 Finished runs

| Method | Split | Seed | F1 | IoU | P | R | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs RGB difference | val | — | 0.129 | 0.069 | 0.096 | 0.193 | 0 |
| FC-EF U-Net | val | 1511 | 0.732 | 0.577 | 0.691 | 0.778 | 483,169 |

So the U-Net is much better than raw differencing on validation. Test was not used here.

Plots from the U-Net run: [`learning_curves.png`](../results/unet_baseline/learning_curves.png), [`validation_gallery.png`](../results/unet_baseline/validation_gallery.png).

### 6.2 Main comparison (to fill after GPU)

| Model | Seed | Val F1 | IoU | P | R | Boundary F1 | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC-EF | 1511 | 0.732 | 0.577 | 0.691 | 0.778 | — | — |
| FC-EF | 42 | — | — | — | — | — | — |
| FC-EF | 123 | — | — | — | — | — | — |
| FC-Siam-diff | 1511 | — | — | — | — | — | — |
| FC-Siam-diff | 42 | — | — | — | — | — | — |
| FC-Siam-diff | 123 | — | — | — | — | — | — |
| FC-Siam-conc | 1511 | — | — | — | — | — | — |

Mean over seeds (after runs finish):

| Model | Mean Val F1 | Spread |
| --- | ---: | ---: |
| FC-EF | — | — |
| FC-Siam-diff | — | — |

### 6.3 Ablations (to fill)

| Setting | Val F1 | IoU |
| --- | ---: | ---: |
| FC-EF, BCE+Dice, aug on, seed 1511 | 0.732 | 0.577 |
| FC-EF, BCE only, aug on, seed 1511 | — | — |
| FC-EF, BCE+Dice, aug off, seed 1511 | — | — |

### 6.4 Compute (to fill)

| Model | Params | Train time | Checkpoint size | Batch-1 latency |
| --- | ---: | ---: | ---: | ---: |
| FC-EF | 483,169 | ~162 min (old 30-epoch run) | — | — |
| FC-Siam-diff | — | — | — | — |
| FC-Siam-conc | — | — | — | — |

### 6.5 Qualitative examples

We will caption validation examples for hits, missed small buildings, false alarms, and messy boundaries after the matrix writes `failure_summary.json` for each run. The current U-Net gallery is a start.

### 6.6 Locked test (once, later)

| Item | Value |
| --- | --- |
| Freeze date | — |
| Chosen setup | — |
| Test F1 / IoU / P / R | — |
| Changed model after seeing test? | No |

```bash
PYTHONPATH=src python scripts/run_locked_test.py \
  --checkpoint results/experiments/<best_run>/best_model.pt \
  --i-confirm-choices-are-frozen
```

---

## 7. Discussion

Right now we can only say that a trained FC-EF clearly beats absolute difference on validation (0.73 vs 0.13 F1). That matches the idea that building change needs more than pixel differencing.

H1 and H2 are still open until Siamese and ablation runs finish on GPU. We will report mean F1 across seeds and not over-claim if Siamese does not win.

Limits of this draft: missing matrix numbers, smaller network than many public LEVIR tables, stem check is not a full geo audit, and LEVIR is Texas-only. We also do not claim this system is ready for real monitoring use.

---

## 8. Future work

Beyond this course project, several extensions would strengthen the study:

1. **Capacity and training budget.** Our networks use a small width (`base=16`). A controlled scale-up (wider U-Net / longer schedule) under the same splits would show how much of the gap to published LEVIR F1 numbers is capacity versus architecture choice.
2. **Stronger geographic leakage checks.** Filename stems do not overlap across splits, but nearby scenes could still correlate. Using LEVIR’s released lat/long metadata to measure spatial distance between train and eval tiles would make the protocol tighter.
3. **Other datasets and domains.** Repeating the same FC-EF vs Siamese comparison on WHU-CD or LEVIR-CD+ would test whether the ranking holds outside Texas Google Earth imagery.
4. **Hard cases.** More work on small buildings, dense suburbs, and seasonal / illumination shifts—either with targeted sampling or harder augmentations—would address the failure modes change detectors usually hit.
5. **Efficiency.** Measuring and improving latency (e.g. lighter decoders or export to ONNX) would matter if the method were ever used for large map tiles, though deployment is outside our current claims.
6. **Attention / modern CD models.** Methods such as STANet or transformer-based change detectors are natural follow-ons. We left them out on purpose so the Daudt triad comparison stayed fair and feasible; they remain open for later work.

---

## 9. Conclusion

We set up a full LEVIR-CD training and evaluation pipeline with the Daudt triad, a simple difference baseline, and the planned ablations. So far, validation shows FC-EF F1 0.732 and abs-diff F1 0.129, with no filename stem overlap across splits. The main research comparison still needs the GPU runs and one locked test. We will keep claims matched to whatever those runs show.

---

## 10. Contributions and how to reproduce

| Member | Main work |
| --- | --- |
| Sneha | Dataset checks, stem overlap, abs-diff baseline |
| Ritika | U-Net / FC-EF path, paired aug, seeds |
| Danish | Siamese models, loss comparison, errors |
| Rajesh | Metrics, timing, summaries, reproduction |

Quick code check (no dataset):

```bash
PYTHONPATH=src python scripts/run_smoke.py
PYTHONPATH=src python -m pytest -q
```

Full runs (needs LEVIR; GPU preferred):

```bash
export LEVIR_DATA_ROOT=/path/to/levir-cd
PYTHONPATH=src python scripts/run_plan.py --mode full
PYTHONPATH=src python scripts/run_baseline.py
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
```

---

## 11. Assistance disclosure

*(Team to fill before final submission. List any AI / writing tools used and how claims were checked, or state that no generative AI assistance was used.)*

---

## References

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI 2015, LNCS 9351, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.

[2] R. Caye Daudt, B. Le Saux, and A. Boulch, “Fully Convolutional Siamese Networks for Change Detection,” IEEE ICIP 2018, pp. 4063–4067. doi: 10.1109/ICIP.2018.8451652.

[3] H. Chen and Z. Shi, “A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection,” *Remote Sensing*, 12(10), 1662, 2020. doi: 10.3390/rs12101662. https://justchenhao.github.io/LEVIR/

[4] Open-CD LEVIR-CD benchmark tables (context only). https://github.com/likyoo/open-cd

[5] I. Corley et al., “A Change Detection Reality Check,” arXiv:2402.06994, 2024 (context only).

---

## Appendix

| File | Path |
| --- | --- |
| Proposal | `docs/Data255_ProjectProposal.pdf` |
| Related work notes | `docs/RELATED_WORK.md` |
| Architecture figure (main) | `docs/architecture-overview.jpg` |
| Architecture figure (vector) | `docs/architecture.svg` |
| Manifest | `results/levir_audit/split_manifest.csv` |
| Stem overlap | `results/levir_audit/stem_overlap_report.json` |
| Abs-diff | `results/absdiff_baseline/summary.json` |
| U-Net baseline | `results/unet_baseline/` |

After GPU training, copy values from `results/comparison_table.md` into the blank tables above.
