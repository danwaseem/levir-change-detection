# Team work division — BuildSafe Vision

**Members:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

Equal ownership across four technical areas each. Presentation is handled separately by the team.

---

## Owned technical areas

| Member | Owns | Key artifacts |
| --- | --- | --- |
| **Sneha Singh** | Data pipeline; leakage/geo audit; abs-diff reference baseline; split/tile + freeze-before-test protocol; Week 3 dataset card | `notebooks/01_dataset_audit.ipynb`, `results/levir_audit/`, `results/absdiff_baseline/`, `docs/progress-report-week-3-4/week3_dataset_card.pdf` |
| **Danish Waseem** | FC-EF / Siamese model code; 9-run GPU matrix; locked test execution; H1/H2 interpretation | `src/levir_cd/models.py`, `results/experiments/full_*`, `results/locked_test/` |
| **Ritika Mukesh Neema** | FC-EF/paired-aug config wiring + seed configs; local feasibility packaging; Week 4 progress report; qualitative validation gallery; responsible-use drafting | `configs/full_unet_*`, `configs/INDEX.md`, `results/unet_baseline/`, `docs/progress-report-week-3-4/week4_progress_report.pdf`, validation gallery |
| **Rajesh Paruchuri** | Metrics + secondary scores; comparison table + latency/compute; `results/environment.json` + `results/report_figures/`; signed reproduction + failure-count tables | `src/levir_cd/metrics.py`, `results/comparison_table.md`, `results/environment.json`, `results/report_figures/`, `docs/REPRODUCTION.md` |

Individual statements: `individual-contribution/`.

---

## Checklist

| Item | Owner | Status |
| --- | --- | --- |
| Data + leakage + abs-diff + Week 3 card | Sneha | Done |
| Models + GPU matrix + locked test + H1/H2 | Danish | Done |
| Configs + Week 4 progress + gallery + responsible-use | Ritika | Done |
| Metrics + figures + environment + reproduction | Rajesh | Done |
| IEEE report packaging | team | Done |
| Confidential peer assessment | each | Outside repo / Canvas |
| Presentation decks | team | Done (`presentation/LEVIR_Change_Detection_DATA255.{pptx,pdf}`) |

---

## Cross-review pairs

| Reviewer | Reviews |
| --- | --- |
| Ritika → Sneha | Dataset card + stem-overlap + abs-diff summary |
| Rajesh → Danish | Locked-test F1/IoU + experiment summaries vs report |
| Sneha → Ritika | Progress report + configs + gallery consistency |
| Danish → Rajesh | Reproduction commands + comparison table + figure set |
