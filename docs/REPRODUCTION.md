# Cross-member reproduction evidence

**Verifier role (proposal):** Rajesh Paruchuri (metrics / summaries / reproduction checks).
**Author of locked GPU matrix:** Danish Waseem (and training pipeline co-authors).
**Date recorded:** 2026-11-18

This check does **not** require full retraining. It verifies that reported numbers match on-disk artifacts using README commands.

## Commands run

```bash
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
PYTHONPATH=src python -c "import json; from pathlib import Path; s=json.loads(Path('results/locked_test/full_unet_bce_aug_seed1511_tile256/summary.json').read_text()); print(round(s['test_f1'],3), round(s['test_iou'],3))"
PYTHONPATH=src python scripts/run_smoke.py
PYTHONPATH=src python -m pytest -q
```

## Reproduced values (from artifacts)

| Check | Expected (report) | Observed on disk | Match? |
| --- | ---: | ---: | --- |
| Locked test F1 | 0.780 | 0.780 | yes |
| Locked test IoU | 0.639 | 0.639 | yes |
| Abs-diff val F1 | 0.129 | 0.129 | yes |

### Per-run validation F1 (artifact read)

| Run directory | Model | Loss | Seed | Val F1 |
| --- | --- | --- | ---: | ---: |
| `full_siamese_conc_bce_dice_aug_seed1511_tile256` | FC-Siam-conc | bce_dice | 1511 | 0.7730 |
| `full_siamese_diff_bce_dice_aug_seed123_tile256` | FC-Siam-diff | bce_dice | 123 | 0.5208 |
| `full_siamese_diff_bce_dice_aug_seed1511_tile256` | FC-Siam-diff | bce_dice | 1511 | 0.5033 |
| `full_siamese_diff_bce_dice_aug_seed42_tile256` | FC-Siam-diff | bce_dice | 42 | 0.6063 |
| `full_unet_bce_aug_seed1511_tile256` | FC-EF (6-channel U-Net) | bce | 1511 | 0.7777 |
| `full_unet_bce_dice_aug_seed123_tile256` | FC-EF (6-channel U-Net) | bce_dice | 123 | 0.7261 |
| `full_unet_bce_dice_aug_seed1511_tile256` | FC-EF (6-channel U-Net) | bce_dice | 1511 | 0.7024 |
| `full_unet_bce_dice_aug_seed42_tile256` | FC-EF (6-channel U-Net) | bce_dice | 42 | 0.7270 |
| `full_unet_bce_dice_noaug_seed1511_tile256` | FC-EF (6-channel U-Net) | bce_dice | 1511 | 0.7455 |

## Discrepancies / fixes

None for the locked test and nine `full_*` summaries. Synthetic smoke runs are not LEVIR scores.

## Checkpoint note

`*.pt` weights are gitignored. Reproducing the locked test forward pass without weights requires either obtaining `best_model.pt` for `full_unet_bce_aug_seed1511_tile256` from the team GPU store, or retraining from `configs/full_unet_bce_aug_seed1511_tile256.json`. Metric summaries and figures under `results/` are sufficient for the non-retrain verification above.
