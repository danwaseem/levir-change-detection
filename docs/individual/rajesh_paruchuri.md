# Individual contribution statement — Rajesh Paruchuri

**Team:** BuildSafe Vision · DATA 255 Fall 2026  
**Load:** Supporting (metrics tables, timing packaging, reproduction sign-off)

## Concrete artifacts
- Aggregate comparison table regeneration: `results/comparison_table.md`
- Signed cross-member reproduction log: `docs/REPRODUCTION.md`
- Packaging of compute / batch-1 latency rows for the report (from existing `summary.json` fields)
- Supporting check that reported F1/IoU match on-disk artifacts

## Cross-review activity
Reviewed Danish’s locked-test summary (`test_f1` **0.780**, `test_iou` **0.639**, no post-test retune) against the IEEE report tables.

## Experiment I can explain in depth
Artifact reproduction workflow: `run_plan.py --aggregate-only` + reading locked-test/abs-diff JSON to confirm report numbers without retraining.
