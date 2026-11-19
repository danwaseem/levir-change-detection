# Team work division — BuildSafe Vision

**Members:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Deadline:** Week 8 / Dec 16, 2026  
**Repo:** https://github.com/danwaseem/levir-change-detection

Roles match the proposal. Everyone still contributes to code review, report accuracy, and Q&A.

---

## 1. Owned technical areas (already done — use in contribution statements)

| Member | Owns | Key artifacts |
| --- | --- | --- |
| **Sneha Singh** | Data integrity, leakage audit, reference baseline | `notebooks/01_dataset_audit.ipynb`, `results/levir_audit/`, `scripts/run_baseline.py`, `results/absdiff_baseline/`, report §III A–B |
| **Ritika Mukesh Neema** | FC-EF / U-Net path, paired augmentation, seed reproducibility | `src/levir_cd/models.py` (U-Net), `src/levir_cd/train.py` (aug/seeds), FC-EF experiment configs under `configs/full_unet_*`, report §IV–V training protocol |
| **Danish Waseem** | Siamese models, loss comparison, GPU matrix, error/gallery | `siamese_*` configs, GPU `results/experiments/full_*`, locked test, `validation_gallery.png`, report §VI–VII H1/H2 + failures |
| **Rajesh Paruchuri** | Metrics, timing, summaries, reproduction | `src/levir_cd/metrics.py`, comparison table, latency protocol narration, `docs/REPRODUCTION.md`, report compute tables |

---

## 2. Remaining work before Dec 16 (must finish)

| Task | Owner | Due | Done when |
| --- | --- | --- | --- |
| Confirm Week 3 dataset-card / Week 4 progress report on **Canvas** | **Sneha** (lead) + all confirm | ASAP | Screenshots or “submitted” note in team chat |
| Build real slide deck from `docs/presentation/slides.md` (PPT/Google Slides) | **Ritika** (deck) | 3 days before talk | Link shared; figures embedded |
| Paste exact lab **GPU model + torch/CUDA versions** into report §VI.D (one line) | **Danish** | Before final PDF freeze | Report lists e.g. `torch x.y + CUDA z + GPU name` |
| Re-run reproduction commands on a clean pull; sign `docs/REPRODUCTION.md` | **Rajesh** | Before submission | Name + date + “re-verified” line |
| Individual contribution statement (use template) | **Each person** | With final submit | Private/course upload |
| Confidential peer assessment | **Each person** | With final submit | Per instructor form |
| Rehearse 8+3 presentation (all four speak) | **All** | Day before | Timed run-through |

### Presentation speaking map (8 min)

| Member | Slides | Time | Q&A depth topic |
| --- | --- | ---: | --- |
| **Sneha** | 1–2 Problem, data, protocol | ~1:45 | Splits, stem overlap, abs-diff threshold 0.4 |
| **Ritika** | 3 Methods (+ architecture fig) | ~1:15 | U-Net, aug sync, checkpoint by val F1 |
| **Danish** | 4–5 Design + main results | ~2:15 | Siamese fusion, why H1 failed, locked test 0.780 |
| **Rajesh** | 6–7 Ablations, cost, failures | ~2:00 | Metrics, latency warm-up, reproduction |
| **All** | 8 Conclusion | ~0:45 | — |

---

## 3. Cross-review pairs (required)

Each person reviews another’s area once before final submit:

| Reviewer | Reviews |
| --- | --- |
| Sneha → Ritika | U-Net / aug claims vs `configs/` and summaries |
| Ritika → Danish | Siamese / locked-test numbers vs `results/locked_test/` |
| Danish → Rajesh | Reproduction doc commands + F1 match |
| Rajesh → Sneha | Leakage audit JSON + abs-diff summary |

Write one sentence of what you checked into your individual contribution statement.

---

## 4. Suggested bullets for individual statements (edit to truth)

### Sneha
- Artifacts: dataset audit notebook; stem-overlap report; abs-diff baseline  
- Cross-review: Ritika’s FC-EF configs/metrics  
- Depth experiment: abs-diff threshold sweep on val (F1 0.129 @ 0.4)

### Ritika
- Artifacts: FC-EF training path; paired aug; seed-controlled U-Net runs  
- Cross-review: Danish’s locked-test summary  
- Depth experiment: FC-EF BCE vs BCE+Dice or aug on/off

### Danish
- Artifacts: Siamese models; 9-run GPU matrix; gallery; locked test  
- Cross-review: Rajesh’s reproduction log  
- Depth experiment: FC-EF vs FC-Siam-diff 3-seed comparison (H1)

### Rajesh
- Artifacts: metrics/timing; comparison table; `docs/REPRODUCTION.md`  
- Cross-review: Sneha’s leakage audit  
- Depth experiment: latency protocol / compute trade-off table

---

## 5. What is already shared (no single owner)

- Final IEEE report PDF: `report/report.pdf`  
- Proposal: `docs/Data255_ProjectProposal.pdf`  
- Package layout: `README.md`, `configs/`, `results/`, `src/`  
- Everyone may edit report for typos; **Danish** owns GPU number accuracy; **Rajesh** owns metric table accuracy; **Sneha** owns data-protocol wording; **Ritika** owns method wording.
