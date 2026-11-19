# Individual contribution statement — Ritika Mukesh Neema

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- FC-EF / U-Net path and paired augmentation in training pipeline (`src/levir_cd/models.py`, `train.py`)
- Seed-controlled FC-EF experiment configs: `configs/full_unet_*.json`
- Progress report draft: `docs/PROGRESS_REPORT.md`
- Report methods / training-protocol wording

## Cross-review activity
Reviewed Danish’s locked-test package `results/locked_test/full_unet_bce_aug_seed1511_tile256/` (test F1 0.780 matches report; `retuned_after_test` false).

## Experiment I can explain in depth
FC-EF loss ablation (seed 1511): BCE-only val F1 **0.778** vs BCE+Dice **0.702** under matched aug — H2 not supported; BCE frozen for locked test.
