# Individual contribution statement — Danish Waseem

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- Siamese FC-Siam-diff / FC-Siam-conc and GPU execution of the full Daudt triad matrix (`results/experiments/full_*`, nine runs)
- Principal 3-seed comparison and loss/architecture investigation (H1/H2)
- Freeze + locked test: test F1 **0.780** (`results/locked_test/...`)
- Qualitative validation gallery and failure-mode diagnosis for the report
- Report methods/results sections for model comparison and hypothesis outcomes

## Cross-review activity
Reviewed Rajesh’s `docs/REPRODUCTION.md`, comparison table, and `results/report_figures/` against locked-test and per-run `summary.json` values.

## Experiment I can explain in depth
Principal comparison under matched BCE+Dice: FC-EF mean val F1 **0.718±0.011** vs FC-Siam-diff **0.543±0.045**, then BCE ablation win (**0.778**) → locked test **0.780**.
