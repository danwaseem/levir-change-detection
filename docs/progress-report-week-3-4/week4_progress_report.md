# Progress Report — Week 4  
**DATA 255 Deep Learning Vision, Fall 2026**  
**Team:** BuildSafe Vision — Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri  
**Repository:** https://github.com/danwaseem/levir-change-detection

## 1. Updated question and scope

**Research question.** Under matched splits, tiles, and metrics, does a shared-encoder Siamese U-Net improve building-change F1 over a six-channel early-fusion U-Net (FC-EF), and how do loss choice and paired augmentation affect accuracy and compute?

**Hypotheses.**  
(H1) FC-Siam-diff improves change-class F1/IoU over FC-EF by comparing dates in feature space (possibly at higher inference cost).  
(H2) BCE+Dice improves rare change-pixel detection relative to BCE alone.

**Scope (unchanged).** Binary building change segmentation on LEVIR-CD; Daudt triad (FC-EF, FC-Siam-diff, FC-Siam-conc); absolute RGB difference as reference; no STANet attention reproduction. Success means credible evidence for or against H1/H2 under a fair protocol, including negative results.

**Minimum viable project.** Reference baseline + FC-EF vs Siamese comparison + two controlled ablations (loss, augmentation) + documented validation analysis.

## 2. Data and evaluation protocol (implemented)

| Item | Status |
| --- | --- |
| Official split 445 / 64 / 128 | Done |
| Tile-after-split \(256\times256\) | Done |
| Stem-overlap audit = 0 | Done (`results/levir_audit/`) |
| Primary metric: change-class F1 @ 0.5 | Done |
| Secondary: IoU, precision, recall, boundary F1 | Done |
| Validation-only selection; test isolated | Done |
| Dataset card (Week 3) | Done (`docs/progress-report-week-3-4/`) |

Audit notebook: `notebooks/01_dataset_audit.ipynb`.

## 3. Methods (implemented)

| Role | Method | Implementation status |
| --- | --- | --- |
| Reference baseline | Absolute RGB difference + val-selected threshold | Runnable (`scripts/run_baseline.py`) |
| Strong baseline | FC-EF (6-channel U-Net, width `base=16`) | Trainable; feasibility run complete |
| Team investigation | FC-Siam-diff / FC-Siam-conc (shared encoder) | Implemented in `src/levir_cd/models.py` |

**Shared training protocol.** Adam (\(10^{-3}\), weight decay \(10^{-4}\)), batch size 4, up to 30 epochs with early stopping, checkpoint by **validation F1**, losses BCE vs BCE+Dice, synchronized flip/\(90^\circ\) augmentation when enabled. Seeds planned for the principal comparison: 1511, 42, 123.

## 4. Preliminary results (Week 4)

| Method | Split | F1 | Notes |
| --- | --- | ---: | --- |
| Absolute RGB difference | Val | **0.129** | Threshold 0.4 chosen on validation |
| FC-EF feasibility (Mac MPS) | Val | **0.732** | Best epoch 24 / 30; ~162 min; ~483k parameters |

Artifacts: `results/absdiff_baseline/summary.json`, `results/unet_baseline/`.

The feasibility FC-EF run already beats absolute difference by a large margin, which supports continuing the Siamese comparison and ablations on GPU rather than relying on pixel subtraction.

## 5. Training / validation diagnostic

On the FC-EF feasibility curves (`results/unet_baseline/learning_curves.png`):

- Training loss decreases smoothly across epochs.
- Validation F1 peaks at epoch **24** (F1 **0.732**) and is unstable if the last epoch is kept.
- **Implication:** checkpoint by best validation F1 (with early stopping), not by final epoch. This rule is locked into the full experiment plan.

## 6. Initial qualitative analysis

Human review of train pairs (notebook 01) shows masks generally align with building appearance/disappearance. The feasibility gallery (`results/unet_baseline/validation_gallery.png`) suggests:

- Larger new buildings are often recovered.
- Boundaries are softer than ground truth.
- Small structures are frequently weak or missed.
- Many tiles contain no change after \(256\times256\) tiling, so empty-tile false alarms are a risk to watch.

These patterns motivate reporting precision/recall and small-change failure counts in the final analysis—not only pooled F1.

## 7. Remaining experiments

1. Full GPU 9-run matrix: 3-seed FC-EF vs FC-Siam-diff (BCE+Dice), one FC-Siam-conc, loss ablation, augmentation ablation.  
2. Freeze the best validation setup; evaluate the held-out test set **once**.  
3. Complete error analysis, compute/latency reporting, and the final IEEE report.

Mac MPS is too slow for the full matrix (~minutes per epoch); university CUDA GPU is required for the graded runs.

## 8. Responsibilities

| Member | Focus |
| --- | --- |
| Sneha Singh | Dataset integrity, leakage audit, abs-diff baseline, protocol wording |
| Ritika Mukesh Neema | FC-EF path, paired augmentation, seed/config packaging |
| Danish Waseem | Siamese models, GPU matrix, loss comparison, error/gallery analysis |
| Rajesh Paruchuri | Metrics, timing summaries, reproduction checks |

Each member contributes to code and writing, reviews another component, and can explain at least one experiment in depth.
