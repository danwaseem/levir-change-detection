# Cross-member reproduction evidence

**Verifier:** Rajesh Paruchuri (metrics / summaries / reproduction checks)  
**Authors of GPU matrix / locked test:** Danish Waseem (+ training pipeline co-authors)  
**Initial record:** 2026-11-18  
**Re-verified:** 2026-11-18 (Rajesh / artifact check on clean `main`)

This check does **not** require full retraining. It verifies that reported numbers match on-disk artifacts.

## Commands run

```bash
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
PYTHONPATH=src python -c "import json; from pathlib import Path; s=json.loads(Path('results/locked_test/full_unet_bce_aug_seed1511_tile256/summary.json').read_text()); print(round(s['test_f1'],3), round(s['test_iou'],3))"
PYTHONPATH=src python -c "import json; from pathlib import Path; s=json.loads(Path('results/absdiff_baseline/summary.json').read_text()); print(round(s['f1'],3))"
```

Optional smoke (`scripts/run_smoke.py`) exercises code paths without LEVIR; it is **not** used as a LEVIR score.

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
| `full_unet_bce_aug_seed1511_tile256` | FC-EF | bce | 1511 | 0.7777 |
| `full_unet_bce_dice_aug_seed123_tile256` | FC-EF | bce_dice | 123 | 0.7261 |
| `full_unet_bce_dice_aug_seed1511_tile256` | FC-EF | bce_dice | 1511 | 0.7024 |
| `full_unet_bce_dice_aug_seed42_tile256` | FC-EF | bce_dice | 42 | 0.7270 |
| `full_unet_bce_dice_noaug_seed1511_tile256` | FC-EF | bce_dice | 1511 | 0.7455 |

## Discrepancies / fixes

None for locked test, abs-diff, or the nine `full_*` summaries on re-verification.

## Environment note

See `results/environment.json`. Training: university CUDA lab (`PROJECT_ROOT=/app/dl`). Verification host: Darwin arm64, PyTorch **2.10.0**.

## Checkpoint note

`*.pt` weights are gitignored. Metric summaries under `results/` are sufficient for this non-retrain verification. Forward-pass re-score needs `best_model.pt` from the team GPU store or a retrain from `configs/full_unet_bce_aug_seed1511_tile256.json`.

## Sign-off

| Role | Name | Date | Result |
| --- | --- | --- | --- |
| Verifier | Rajesh Paruchuri | 2026-11-18 | Artifact numbers match report |
