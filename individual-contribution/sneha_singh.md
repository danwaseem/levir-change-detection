# Individual contribution statement — Sneha Singh

**Team:** BuildSafe Vision · DATA 255 Fall 2026

## Concrete artifacts
- End-to-end dataset audit and integrity pipeline: `notebooks/01_dataset_audit.ipynb`
- Leakage / stem-overlap science: `results/levir_audit/stem_overlap_report.json`, `split_manifest.csv`
- Non-learned reference baseline design and scoring: `results/absdiff_baseline/` (val F1 **0.129** @ threshold 0.4)
- Evaluation protocol: official splits, tile-after-split rule, freeze-before-test policy
- Week 3 dataset card (PDF removed from git)
- Support: freeze checklist with Danish before the locked test; protocol field checks with Ritika’s configs

## Cross-review activity
Reviewed Ritika’s progress report, `configs/`, `losses.py` flags, and gallery packaging for consistency with the frozen protocol (seeds, aug flags, loss keys).

## Experiment I can explain in depth
Abs-diff reference vs learned models under the same val tiles: threshold sweep → **0.4** → F1 **0.129**, establishing that the task needs learning beyond pixel subtraction.
