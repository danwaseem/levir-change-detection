# Schedule milestones → evidence

Index for the course Week 1–8 table. Artifacts stay in their normal folders; this file only maps them.

| Timing | Milestone | Evidence in this repo |
| --- | --- | --- |
| Week 1 | Feasibility and team agreement | [`Data255_ProjectProposal.pdf`](Data255_ProjectProposal.pdf) (question, dataset, compute, roles, fallback); [`TEAM_WORK_DIVISION.md`](TEAM_WORK_DIVISION.md) |
| Week 2 | Proposal | [`Data255_ProjectProposal.pdf`](Data255_ProjectProposal.pdf); related work notes in [`RELATED_WORK.md`](RELATED_WORK.md) |
| Week 3 | Dataset and baseline checkpoint | Splits/baseline under `results/` (`absdiff_baseline`, `levir_audit`); runnable `scripts/run_baseline.py`. Week 3 PDF is not in git. |
| Week 4 | Main model + progress report | Training/val via `scripts/run_plan.py` + `configs/`. Week 4 PDF is not in git. |
| Week 5 | Controlled experiments | `configs/` (9-run matrix); `results/experiments/`; [`results/comparison_table.md`](../results/comparison_table.md) (main compare + ablation) |
| Week 6 | Final evaluation and analysis | `results/locked_test/` (+ `error_strata.json` strata/bootstrap CI). IEEE PDF/DOCX stay on disk and are not in git. |
| Week 7 | Reproducibility and communication | [`REPRODUCTION.md`](REPRODUCTION.md); `scripts/verify_reported_metrics.py`; final configs + `results/report_figures/`; report draft→final; [`../presentation/`](../presentation/); contrib in report Table VIII + [`../individual-contribution/`](../individual-contribution/) |
| Week 8 | Final submission and presentation | IEEE PDF/DOCX generated locally by `report/build_ieee_report.py` and not committed; code package (`README`, `requirements.txt`, `configs/`, `src/`, `scripts/`, `results/`, `docs/`); slides; disclosures; presentation + peer assessment on Canvas |

**Outside the zip / repo:** Canvas upload, live presentation, confidential peer assessments.
