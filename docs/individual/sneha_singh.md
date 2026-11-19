# Individual contribution statement — Sneha Singh

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- Dataset integrity / audit: `notebooks/01_dataset_audit.ipynb`
- Stem-overlap leakage check: `results/levir_audit/stem_overlap_report.json`, `split_manifest.csv`
- Reference baseline: abs-diff pipeline and `results/absdiff_baseline/summary.json` (val F1 0.129)
- Dataset card for Week 3: `docs/DATASET_CARD.md`
- Report sections on dataset / protocol / responsible-use wording review

## Cross-review activity
Reviewed Ritika’s FC-EF configs under `configs/full_unet_*` against `results/experiments/*/summary.json` for seed/loss/aug consistency.

## Experiment I can explain in depth
Absolute RGB difference baseline: validation threshold sweep → selected threshold **0.4** → change-class F1 **0.129**, showing non-learned differencing is far below learned models.
