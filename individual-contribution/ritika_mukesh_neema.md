# Individual contribution statement — Ritika Mukesh Neema

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- FC-EF / paired-augmentation config wiring and seed matrix: `configs/*.json`, `configs/INDEX.md`
- Loss implementations used in the matrix (BCE / BCE+Dice): `src/levir_cd/losses.py`
- Local feasibility + smoke packaging under `results/unet_baseline/`
- Week 4 progress-report packaging: `docs/progress-report-week-3-4/week4_progress_report.pdf`
- Qualitative validation gallery and failure-mode image selection for the report
- Responsible-use / limitation drafting for the final report
- Support: dataset-card / stem-overlap review with Sneha; full seed/loss/aug config matrix for Danish’s GPU runs

## Cross-review activity
Reviewed Sneha’s dataset card and stem-overlap report (counts 445/64/128; overlap **0**) for upload readiness.

## Experiment I can explain in depth
How `SEED` / `AUGMENT` / `LOSS` in `configs/` and `losses.py` map to `results/experiments/full_*` folders, plus gallery failure modes (small-building misses, empty-tile false alarms).
