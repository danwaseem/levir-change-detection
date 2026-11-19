# Individual contribution statement — Ritika Mukesh Neema

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- FC-EF / paired-augmentation config wiring: `configs/full_unet_*.json`
- Config index for reported runs: `configs/INDEX.md`
- Local feasibility packaging under `results/unet_baseline/`
- Week 4 progress-report packaging: `docs/progress-report-week-3-4/week4_progress_report.pdf`
- Qualitative validation gallery and failure-mode image selection for the report
- Responsible-use / limitation drafting for the final report

## Cross-review activity
Reviewed Sneha’s dataset card and stem-overlap report (counts 445/64/128; overlap **0**) for upload readiness.

## Experiment I can explain in depth
FC-EF seed/aug config layout for the GPU matrix (how `SEED` / `AUGMENT` / `LOSS` in `configs/` map to `results/experiments/full_unet_*` folders), plus gallery failure modes (small-building misses, empty-tile false alarms).
