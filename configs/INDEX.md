# Reported experiment configs

These JSON files mirror `results/experiments/*/config.json` for the nine GPU runs in the final report.

| Config | Model | Loss | Seed | Val F1 |
| --- | --- | --- | ---: | ---: |
| `configs/full_siamese_conc_bce_dice_aug_seed1511_tile256.json` | siamese_conc | bce_dice | 1511 | 0.7730 |
| `configs/full_siamese_diff_bce_dice_aug_seed123_tile256.json` | siamese_diff | bce_dice | 123 | 0.5208 |
| `configs/full_siamese_diff_bce_dice_aug_seed1511_tile256.json` | siamese_diff | bce_dice | 1511 | 0.5033 |
| `configs/full_siamese_diff_bce_dice_aug_seed42_tile256.json` | siamese_diff | bce_dice | 42 | 0.6063 |
| `configs/full_unet_bce_aug_seed1511_tile256.json` | unet | bce | 1511 | 0.7777 |
| `configs/full_unet_bce_dice_aug_seed123_tile256.json` | unet | bce_dice | 123 | 0.7261 |
| `configs/full_unet_bce_dice_aug_seed1511_tile256.json` | unet | bce_dice | 1511 | 0.7024 |
| `configs/full_unet_bce_dice_aug_seed42_tile256.json` | unet | bce_dice | 42 | 0.7270 |
| `configs/full_unet_bce_dice_noaug_seed1511_tile256.json` | unet | bce_dice | 1511 | 0.7455 |

Locked evaluation setup: `configs/full_unet_bce_aug_seed1511_tile256.json` (FC-EF, BCE, augmentation on, seed 1511, threshold 0.5).
