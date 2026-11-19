# Individual contribution statement — Danish Waseem

**Team:** BuildSafe Vision · DATA 255 Fall 2026  
**Load:** Hard technical lead (models, GPU matrix, locked test, diagnosis)

## Concrete artifacts
- Siamese FC-Siam-diff / FC-Siam-conc and GPU execution of the full Daudt triad matrix (`results/experiments/full_*`, nine runs)
- Principal 3-seed comparison and loss/architecture investigation (H1/H2)
- Freeze + locked test: test F1 **0.780** (`results/locked_test/...`)
- Failure analysis, validation gallery, learning/ablation figures used in the report
- Lab environment record: `results/environment.json`; `train.py` device/torch fields for future runs
- Report §§IV–VII: methods execution, results, hypothesis outcomes, error analysis

## Cross-review activity
Reviewed Rajesh’s `docs/REPRODUCTION.md` and `results/comparison_table.md` against locked-test and per-run `summary.json` values.

## Experiment I can explain in depth
Principal comparison under matched BCE+Dice: FC-EF mean val F1 **0.718±0.011** vs FC-Siam-diff **0.543±0.045**, then BCE ablation win (**0.778**) → locked test **0.780** with small-object failure diagnosis.
