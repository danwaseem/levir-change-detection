# Team work division — BuildSafe Vision

**Members:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

**Load split:** **Sneha** and **Danish** own the hard research/engineering core. **Ritika** and **Rajesh** own supporting implementation, packaging, and verification. Presentation is handled separately by the team.

---

## Owned technical areas

| Member | Load | Owns (hard vs supporting) | Key artifacts |
| --- | --- | --- | --- |
| **Sneha Singh** | **Hard** | Data pipeline end-to-end; leakage/geo audit; abs-diff reference science; eval protocol (splits, freeze rules, primary metric); dataset card; responsible-use / limitation drafting; deep review of experimental claims | `notebooks/01_dataset_audit.ipynb`, `results/levir_audit/`, `results/absdiff_baseline/`, `docs/DATASET_CARD.md`, report §§III & VII (protocol / responsible use) |
| **Danish Waseem** | **Hard** | Daudt triad model code on GPU; full 9-run matrix; loss/architecture investigation; locked test; failure analysis + gallery; environment logging; H1/H2 interpretation | `src/levir_cd/models.py` / train path on lab, `results/experiments/full_*`, `results/locked_test/`, `results/environment.json`, report §§IV–VII results/errors |
| **Ritika Mukesh Neema** | Supporting | FC-EF local feasibility / paired-aug wiring; seed config files; progress-report packaging; assist figure/config indexing | `configs/full_unet_*`, `docs/PROGRESS_REPORT.md`, feasibility `results/unet_baseline/` |
| **Rajesh Paruchuri** | Supporting | Metric aggregation / comparison table; latency numbers packaging; signed reproduction check; compute-table formatting | `results/comparison_table.md`, `docs/REPRODUCTION.md`, report compute rows |

Individual statements: `docs/individual/`.

---

## Non-presentation checklist

| Item | Owner | Status |
| --- | --- | --- |
| Hard science: data + protocol + abs-diff | Sneha | Done |
| Hard science: GPU matrix + locked test + failures | Danish | Done |
| Supporting: progress report / U-Net config packaging | Ritika | Done |
| Supporting: reproduction sign-off / tables | Rajesh | Done |
| IEEE report packaging | team (Sneha+Danish lead technical claims) | Done |
| Confidential peer assessment | each | Outside repo |
| Presentation | team | Separate |

---

## Cross-review pairs

Hard work is reviewed by supporters; supporters’ packaging is reviewed by leads.

| Reviewer | Reviews |
| --- | --- |
| Ritika → Sneha | Dataset card + stem-overlap + abs-diff summary |
| Rajesh → Danish | Locked-test F1/IoU + experiment summaries vs report |
| Sneha → Ritika | Progress report + U-Net configs consistency |
| Danish → Rajesh | Reproduction commands + comparison table |
