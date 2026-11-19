# Individual contribution statement — Rajesh Paruchuri

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- Metrics implementation and secondary scores (IoU, precision, recall, boundary F1): `src/levir_cd/metrics.py`
- Aggregate comparison table: `results/comparison_table.md`
- Compute / batch-1 latency packaging and trade-off rows in the final report
- Environment record for verification host + lab notes: `results/environment.json`
- Report figure packaging under `results/report_figures/` (bars, curves used in the IEEE report)
- Quantitative failure-count tables for locked-test analysis (empty-tile / small-change stats) pulled into the report
- Signed cross-member reproduction log: `docs/REPRODUCTION.md`

## Cross-review activity
Reviewed Danish’s locked-test summary (`test_f1` **0.780**, `test_iou` **0.639**, no post-test retune) against the IEEE report tables and figure set.

## Experiment I can explain in depth
Metric + compute package: change-class F1 selection vs secondary scores, latency warm-up/sync protocol, and reproducing Table numbers from `summary.json` without retraining.
