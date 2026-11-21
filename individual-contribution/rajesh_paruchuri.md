# Individual contribution statement — Rajesh Paruchuri

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- Metrics implementation and secondary scores (IoU, precision, recall, boundary F1): `src/levir_cd/metrics.py`
- Aggregate comparison table and plan-aggregation outputs: `results/comparison_table.md`
- Compute / batch-1 latency packaging and trade-off rows in the final report
- Environment record for verification host + lab notes: `results/environment.json`
- Report figure packaging under `results/report_figures/` (bars, curves used in the IEEE report)
- Quantitative failure-count / error-strata tables for locked-test analysis (report Table VIII)
- Locked-test strata + tile-bootstrap CI artifact: `results/locked_test/full_unet_bce_aug_seed1511_tile256/error_strata.json`
- No-GPU metric verifier: `scripts/verify_reported_metrics.py`
- Signed cross-member reproduction log: `docs/REPRODUCTION.md`
- Support: locked-test F1/IoU verification with Danish; compute/latency rows for Ritika’s progress packaging

## Cross-review activity
Reviewed Danish’s locked-test summary (`test_f1` **0.780**, `test_iou` **0.639**, no post-test retune) against the IEEE report tables and figure set.

## Experiment I can explain in depth
Metric + compute package: change-class F1 selection vs secondary scores, latency warm-up/sync protocol, and reproducing Table numbers from `summary.json` without retraining.
