# Cross-member reproduction evidence

**Data / abs-diff / audit (+ freeze support):** Sneha Singh  
**Models / GPU matrix / locked test (+ summary QA support):** Danish Waseem  
**Configs / losses / gallery (+ GPU matrix configs):** Ritika Mukesh Neema  
**Metrics / tables / reproduction (+ locked-test verify):** Rajesh Paruchuri  
**Initial record:** 2026-11-18  
**Re-verified:** 2026-11-18 (Rajesh on clean `main`)  
**Scoring-integrity follow-up:** 2026-11-23 (secondary-field labels + stronger verifier on `main`)

This check does **not** require full retraining. It verifies that reported numbers match on-disk artifacts.

## Commands run

```bash
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
python scripts/verify_reported_metrics.py
PYTHONPATH=src python -c "import json; from pathlib import Path; s=json.loads(Path('results/locked_test/full_unet_bce_aug_seed1511_tile256/summary.json').read_text()); print(round(s['test_f1'],3), round(s['test_iou'],3))"
PYTHONPATH=src python -c "import json; from pathlib import Path; s=json.loads(Path('results/absdiff_baseline/summary.json').read_text()); print(round(s['f1'],3))"
```

Locked-test strata / bootstrap CI (no GPU): `results/locked_test/full_unet_bce_aug_seed1511_tile256/error_strata.json`.

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

None for locked test F1/IoU, abs-diff **F1**, or the nine `full_*` **best_val_f1** values on re-verification.

### Secondary-field reading rules (do not invent numbers)

| Field in `results/experiments/full_*/summary.json` | Trust for report claims? | Note |
| --- | --- | --- |
| `best_val_f1`, `val_precision`, `val_recall`, `val_iou`, `val_boundary_f1` | Yes | Match best epoch in that run’s `metrics.csv` |
| `threshold_sweep`, embedded `per_image_val_summary`, embedded `failure_summary` | No (legacy runs) | Labeled `secondary_artifacts_source: last_epoch_weights` — computed after early stop, before the best-checkpoint reload fix |
| Locked-test `failure_summary` / `error_strata.json` | Yes | Scored from listed `best_model.pt` |
| Abs-diff `f1` / `iou` / `precision` / `recall` | Yes | Unchanged by the tn fix |
| Abs-diff `accuracy` / `tn` | Yes (after fix) | Older copies had `tn=0` so accuracy wrongly equaled IoU; current file has real `tn` |

`scripts/verify_reported_metrics.py` now checks headline locked/abs-diff numbers, abs-diff `tn`, and that each `full_*` `best_val_f1` matches `metrics.csv`, and that last-epoch sweeps are labeled.

## Environment note

See `results/environment.json`. Training: university CUDA lab (`PROJECT_ROOT=/app/dl`). Verification host: Darwin arm64, PyTorch **2.10.0**.

## Checkpoint note

`*.pt` weights are gitignored. Metric summaries under `results/` are sufficient for this non-retrain verification. Forward-pass re-score needs `best_model.pt` from the team GPU store or a retrain from `configs/full_unet_bce_aug_seed1511_tile256.json`.

## Manifest path note

`results/levir_audit/split_manifest.csv` contains absolute image paths from the host that built it. Stem-overlap counts in `stem_overlap_report.json` remain valid without those files. To train or re-score on a new machine, set `LEVIR_DATA_ROOT` and rebuild the manifest with the dataset audit before loading tiles.

## Sign-off

| Role | Name | Date | Result |
| --- | --- | --- | --- |
| Abs-diff / audit (+ freeze support) | Sneha Singh | 2026-11-18 | Reference + leakage artifacts |
| GPU / locked test (+ summary QA) | Danish Waseem | 2026-11-18 | Runs + locked test on disk |
| Configs / losses / gallery | Ritika Mukesh Neema | 2026-11-18 | Config index + loss/aug wiring + gallery |
| Metrics / reproduction (+ verify) | Rajesh Paruchuri | 2026-11-18 | Artifact numbers match report |
