# Experiment configurations

Each JSON file mirrors `results/experiments/<run>/config.json` for the nine
reported full-matrix runs. Use with `scripts/run_plan.py` / training entry points
via the same field names (`MODEL`, `LOSS`, `AUGMENT`, `SEED`, …).

Locked evaluation setup: `full_unet_bce_aug_seed1511_tile256.json`
(FC-EF, BCE, augmentation on, seed 1511, threshold 0.5).
