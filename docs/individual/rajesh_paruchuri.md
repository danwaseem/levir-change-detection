# Individual contribution statement — Rajesh Paruchuri

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- Metrics / timing surfaces (`src/levir_cd/metrics.py`, batch-1 latency warm-up+sync in `train.py`)
- Aggregate comparison table: `results/comparison_table.md`
- Cross-member reproduction log: `docs/REPRODUCTION.md` (re-verified 2026-11-18)
- Compute / latency narration in the IEEE report

## Cross-review activity
Reviewed Sneha’s leakage audit (`stem_overlap_report.json`): all pairwise stem overlaps **0**; counts 445/64/128.

## Experiment I can explain in depth
Computational trade-off at seed 1511: FC-EF (~483k params, ~0.0014 s batch-1) vs Siam-diff/conc higher latency; FC-EF Pareto-strong on F1 and speed under this budget.
