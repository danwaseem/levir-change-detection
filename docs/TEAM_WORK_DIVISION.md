# Team work division — BuildSafe Vision

**Members:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

Presentation is handled separately by the team.

---

## Owned technical areas

| Member | Owns | Key artifacts |
| --- | --- | --- |
| **Sneha Singh** | Data pipeline; leakage/geo audit; abs-diff reference baseline; split/tile protocol; Week 3 dataset card; responsible-use drafting | `notebooks/01_dataset_audit.ipynb`, `results/levir_audit/`, `results/absdiff_baseline/`, `docs/progress-report-week-3-4/week3_dataset_card.pdf` |
| **Danish Waseem** | Siamese + FC-EF GPU training; 9-run matrix; locked test execution; qualitative gallery; H1/H2 interpretation | `src/levir_cd/models.py`, `results/experiments/full_*`, `results/locked_test/`, validation gallery |
| **Ritika Mukesh Neema** | FC-EF/paired-aug config wiring; seed configs; local feasibility packaging; Week 4 progress report | `configs/full_unet_*`, `docs/progress-report-week-3-4/week4_progress_report.pdf`, `results/unet_baseline/` |
| **Rajesh Paruchuri** | Metrics + secondary scores (IoU/P/R/boundary F1); comparison table; latency/compute reporting; `results/environment.json`; report figure packaging (`results/report_figures/`); signed reproduction check; quantitative failure-count tables for the report | `src/levir_cd/metrics.py`, `results/comparison_table.md`, `results/environment.json`, `results/report_figures/`, `docs/REPRODUCTION.md` |

Individual statements: `individual-contribution/`.

---

## Checklist

| Item | Owner | Status |
| --- | --- | --- |
| Data + leakage + abs-diff | Sneha | Done |
| GPU matrix + locked test + gallery | Danish | Done |
| Configs + progress report | Ritika | Done |
| Metrics, figures, environment, reproduction | Rajesh | Done |
| IEEE report packaging | team | Done |
| Confidential peer assessment | each | Outside repo |
| Presentation | team | Separate |

---

## Cross-review pairs

| Reviewer | Reviews |
| --- | --- |
| Ritika → Sneha | Dataset card + stem-overlap + abs-diff summary |
| Rajesh → Danish | Locked-test F1/IoU + experiment summaries vs report |
| Sneha → Ritika | Progress report + U-Net configs consistency |
| Danish → Rajesh | Reproduction commands + comparison table + figure set |
