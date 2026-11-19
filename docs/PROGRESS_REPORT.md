# Progress report — Week 4 (2–3 pages)

**Course:** DATA 255 Deep Learning Vision, Fall 2026  
**Team:** BuildSafe Vision — Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri  
**Repository:** https://github.com/danwaseem/levir-change-detection

## Updated question and scope

**RQ.** Under matched splits/tiles/metrics, does shared-encoder Siamese U-Net improve building-change F1 over FC-EF, and how do loss and paired augmentation affect accuracy and compute?

**Scope (unchanged).** LEVIR-CD binary change segmentation; Daudt triad; abs-diff reference; no STANet reproduction. MVP = reference + FC-EF vs Siamese + two ablations + documented validation analysis.

## Data and evaluation protocol (implemented)

- Official 445/64/128 split; \(256\times256\) tiles after split; stem overlap **0**.
- Primary metric: change-class F1 @ 0.5; secondary IoU / P / R / boundary F1.
- Validation-only selection; test frozen until final evaluation.
- Dataset card: `docs/DATASET_CARD.md`. Audit notebook: `notebooks/01_dataset_audit.ipynb`.

## Methods and preliminary results (at Week 4 stage)

| Role | Method | Status at progress checkpoint |
| --- | --- | --- |
| Reference | Absolute RGB difference | Runnable; val F1 **0.129** |
| Strong baseline | FC-EF (6-ch U-Net, `base=16`) | Feasibility Mac run val F1 ~0.73; full GPU matrix planned/completed later |
| Investigation | FC-Siam-diff / FC-Siam-conc | Implemented in `src/levir_cd/models.py`; GPU matrix later |

Training protocol: Adam \(10^{-3}\), batch 4, ≤30 epochs, checkpoint by val F1, BCE vs BCE+Dice, synced flip/rot90.

## Training / validation diagnostic

Early FC-EF feasibility curves (`results/unet_baseline/`) showed learning with best-checkpoint selection mattering (validation unstable if last epoch kept). This motivated early stopping / best val-F1 checkpointing for the full matrix.

## Initial qualitative analysis

Human audit of train pairs in notebook 01 confirmed masks align with building appearance/disappearance. Failure modes expected: small buildings, soft boundaries, empty-tile false alarms (confirmed later on locked test).

## Remaining experiments (then completed for final report)

1. Nine-run GPU matrix (3-seed principal + loss/aug ablations + Siam-conc).  
2. Freeze best validation setup; one locked test.  
3. Error analysis + compute table + final IEEE report.

*(Final outcomes are in `report/report.pdf`: locked test F1 **0.780**; H1/H2 not supported under `base=16`.)*

## Responsibilities (revised)

| Member | Focus |
| --- | --- |
| Sneha | Data audit, leakage, abs-diff |
| Ritika | FC-EF path, aug, seeds |
| Danish | Siamese, GPU matrix, failures |
| Rajesh | Metrics, timing, reproduction |

Each member contributes code, writing, and one cross-review (see `docs/TEAM_WORK_DIVISION.md`).
