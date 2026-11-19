# Individual contribution statement — Danish Waseem

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- Siamese FC-Siam-diff / FC-Siam-conc implementation and GPU 9-run matrix under `results/experiments/full_*`
- Locked test execution after freeze (2026-11-08): test F1 **0.780**
- Qualitative gallery and failure analysis figures
- Hardware/software note + `results/environment.json` (lab CUDA `/app/dl`; verification host torch 2.10.0)
- Future runs now log `torch_version` / `cuda_device_name` in `summary.json` via `train.py`

## Cross-review activity
Reviewed Rajesh’s `docs/REPRODUCTION.md` commands against on-disk locked-test and comparison-table numbers.

## Experiment I can explain in depth
Principal 3-seed comparison (BCE+Dice): FC-EF mean val F1 **0.718±0.011** vs FC-Siam-diff **0.543±0.045** — H1 not supported for difference fusion at `base=16`.
