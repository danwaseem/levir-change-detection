# Team work division — BuildSafe Vision

**Members:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

Presentation / slides / rehearsal are **out of scope for this completion pass** (team handles separately).

---

## Owned technical areas

| Member | Owns | Key artifacts |
| --- | --- | --- |
| **Sneha Singh** | Data integrity, leakage, abs-diff | `docs/DATASET_CARD.md`, `results/levir_audit/`, abs-diff results |
| **Ritika Mukesh Neema** | FC-EF, aug, seeds | `configs/full_unet_*`, `docs/PROGRESS_REPORT.md` |
| **Danish Waseem** | Siamese, GPU matrix, locked test | `results/experiments/full_*`, locked test, `results/environment.json` |
| **Rajesh Paruchuri** | Metrics, timing, reproduction | `docs/REPRODUCTION.md`, comparison table |

Individual statements (drafts to personalize/submit): `docs/individual/`.

---

## Non-presentation checklist

| Item | Owner | Status |
| --- | --- | --- |
| IEEE report + packaging | team | Done |
| `configs/` | team | Done |
| Reproduction log signed | Rajesh | Done (`docs/REPRODUCTION.md`) |
| Dataset card (Week 3) | Sneha | Done (`docs/DATASET_CARD.md`) — upload to Canvas if required |
| Progress report (Week 4) | Ritika | Done (`docs/PROGRESS_REPORT.md`) — upload to Canvas if required |
| Env / torch strings in report | Danish | Done (`results/environment.json` + report §VI.D) |
| Individual contribution drafts | each | Done in `docs/individual/` — submit privately |
| Confidential peer assessment | each | **Outside repo** (course form only) |
| Presentation slides / talk | team | **Left to team** |

---

## Cross-review pairs

| Reviewer | Reviews |
| --- | --- |
| Sneha → Ritika | U-Net configs vs summaries |
| Ritika → Danish | Locked-test numbers |
| Danish → Rajesh | Reproduction doc |
| Rajesh → Sneha | Leakage audit JSON |
