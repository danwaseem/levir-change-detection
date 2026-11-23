# Team work division — BuildSafe Vision

**Members:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

Even split: each member has one **technical lead** block and one **support** block. Presentation decks are shared.

---

## Owned technical areas

| Member | Technical lead | Also supports | Key artifacts |
| --- | --- | --- | --- |
| **Sneha Singh** | Data pipeline; leakage/geo audit; abs-diff reference; split/tile + freeze-before-test protocol; Week 3 dataset card | Danish — freeze checklist before locked test; Ritika — protocol fields in configs | `notebooks/01_dataset_audit.ipynb`, `results/levir_audit/`, `results/absdiff_baseline/` |
| **Danish Waseem** | FC-EF / Siamese model code; 9-run GPU matrix; locked test execution; H1/H2 interpretation | Sneha — Siamese/dataloader integration for bi-temporal tiles; Rajesh — per-run `summary.json` QA vs report | `src/levir_cd/models.py`, `results/experiments/full_*`, `results/locked_test/` |
| **Ritika Mukesh Neema** | Config/seed/aug wiring; `losses.py` (BCE / BCE+Dice); local feasibility + smoke packaging; Week 4 progress; validation gallery; responsible-use drafting | Sneha — dataset-card / stem-overlap review; Danish — full config matrix for the GPU runs | `configs/`, `configs/INDEX.md`, `src/levir_cd/losses.py`, `results/unet_baseline/`, gallery |
| **Rajesh Paruchuri** | Metrics + secondary scores; comparison table; latency/compute; `environment.json`; `report_figures/`; failure-count / error-strata tables; signed reproduction; aggregate plan outputs; `verify_reported_metrics.py` | Danish — locked-test F1/IoU verification; Ritika — compute/latency rows for progress packaging | `src/levir_cd/metrics.py`, `results/comparison_table.md`, `results/environment.json`, `results/report_figures/`, `results/locked_test/.../error_strata.json`, `scripts/verify_reported_metrics.py`, `docs/REPRODUCTION.md` |

Individual statements: `individual-contribution/`.

---

## Checklist

| Item | Owner | Status |
| --- | --- | --- |
| Data + leakage + abs-diff + Week 3 (+ support freeze) | Sneha | Done |
| Models + GPU matrix + locked test + H1/H2 (+ support data/metrics QA) | Danish | Done |
| Configs + losses + feasibility + Week 4 + gallery (+ support audit/GPU configs) | Ritika | Done |
| Metrics + figures + environment + reproduction (+ support locked-test/compute) | Rajesh | Done |
| IEEE report packaging | team | Done |
| Confidential peer assessment | each | Outside repo / Canvas |
| Presentation decks | team | Done (`presentation/LEVIR_Change_Detection_DATA255.{pptx,pdf}`) |

---

## Cross-review pairs

| Reviewer | Reviews |
| --- | --- |
| Ritika → Sneha | Dataset card + stem-overlap + abs-diff summary |
| Rajesh → Danish | Locked-test F1/IoU + experiment summaries vs report |
| Sneha → Ritika | Progress report + configs + losses/aug flags + gallery |
| Danish → Rajesh | Reproduction commands + comparison table + figure set |
