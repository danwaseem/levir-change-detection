# Final Report — Building Change Detection on LEVIR-CD

**Course:** DATA 255 Deep Learning Vision, Fall 2026  
**Team:** BuildSafe Vision — Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri  
**Status:** Methods and the experiment matrix are implemented. The only LEVIR number in this repo is the validation U-Net baseline (F1 0.7319, seed 1511). Siamese runs, ablations, and the test set are not filled in yet. Do not invent those numbers.

Related files:
- Proposal: [`docs/Data255_ProjectProposal.pdf`](../docs/Data255_ProjectProposal.pdf)
- Metric artifacts (tables/plots from runs): [`results/`](../results/)
- Code: [`src/levir_cd/`](../src/levir_cd/)

---

## 1. Title and abstract

- Working title: Deep Learning for Building Change Detection in Multi-Temporal Satellite Imagery
- Abstract (≈150–200 words): problem, LEVIR-CD, U-Net vs Siamese U-Net, main validation finding, one limitation

## 2. Introduction and research question

- Why building change ≠ generic visual change (illumination, vegetation, acquisition)
- Research question (from proposal)
- Hypotheses:
  - **H1:** Siamese shared encoder improves changed-class F1/IoU vs six-channel U-Net (possibly at higher latency)
  - **H2:** BCE+Dice improves changed-pixel detection vs BCE alone under class imbalance

## 3. Related work

Detailed mapping: [`docs/RELATED_WORK.md`](../docs/RELATED_WORK.md).

- U-Net (Ronneberger et al., 2015) → our FC-EF encoder–decoder
- Fully convolutional Siamese CD (Daudt et al., 2018) → **FC-EF / FC-Siam-diff / FC-Siam-conc** triad in `models.py`
- LEVIR-CD / STANet (Chen & Shi, 2020) → dataset + change-class Pr/Re/F1; we do **not** reproduce STANet attention or BCL

## 4. Dataset and protocol

- LEVIR-CD: 637 pairs, 1024×1024, Texas; academic use / Google Earth terms
- Official split: 445 / 64 / 128 → 256×256 non-overlapping tiles
- Mask binarization at intensity ≥ 128
- Paired augmentation policy
- Train/val only until locked test
- Stem-overlap / residual geographic-leakage note

## 5. Methods / architecture

### 5.1 Reference baseline
- Absolute RGB difference + validation-selected threshold

### 5.2 Six-channel U-Net
- Input: concat(A,B) → 6 channels; base width 16; 1-channel logits

### 5.3 Shared-encoder Siamese U-Net
- Tied encoder on A and B; fuse with `|fA − fB|`; shared decoder

### 5.4 Losses and training
- BCE vs BCE+Dice
- Adam, lr, batch, early stopping, checkpoint by validation F1
- Fixed decision threshold 0.5
- Secondary: boundary F1 and small-change / false-alarm counts in `failure_summary.json`
- Overall accuracy is logged and is not the selection metric

### 5.5 What we do not claim
- STANet attention and batch-balanced contrastive loss are out of scope
- Published LEVIR test F1 numbers for FC-Siam (about 0.81–0.86) come from other training setups. They are not this team's score.

## 6. Experiments

| Experiment | What changes | What is held fixed |
| --- | --- | --- |
| Main comparison | U-Net vs Siamese | tiles, aug, BCE+Dice, budget, seeds 1511/42/123 |
| Loss ablation | BCE vs BCE+Dice | chosen architecture, aug, seed |
| Augmentation ablation | aug on vs off | chosen architecture, loss, seed |
| Efficiency | params, train time, checkpoint size, batch-1 latency | same hardware note |

## 6b. How to produce the missing numbers

On the project machine. If LEVIR-CD is not already on disk, this downloads the train, val, and test archives once (~2.3 GB) and writes the audit manifest:

```bash
PYTHONPATH=src python scripts/run_plan.py --mode full
PYTHONPATH=src python scripts/run_plan.py --aggregate-only
```

`scripts/run_plan.py` runs this matrix and writes `results/comparison_table.md`:

| Role | Settings |
| --- | --- |
| Main | FC-EF and FC-Siam-diff, BCE+Dice, augmentation on, seeds 1511, 42, 123 |
| Triad column | FC-Siam-conc, same loss and augmentation, seed 1511 |
| Loss | FC-EF, BCE only, seed 1511 |
| Augmentation | FC-EF, augmentation off, seed 1511 |

Each run also writes `failure_summary.json` and, for real data, `threshold_sweep.json`. The official threshold remains 0.5.

## 7. Results

### 7.1 Numbers we actually have

| Model | Split | Seed | F1 | IoU | Precision | Recall | Where |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| FC-EF lightweight U-Net | validation | 1511 | 0.7319 | 0.5772 | 0.6912 | 0.7778 | `results/unet_baseline/` |

No Siamese validation number, no ablation number, and no test number is available yet. After `run_plan.py`, paste `results/comparison_table.md` here and add mean ± spread across the three main seeds.

### 7.2 Qualitative examples
- Hits, missed small buildings, false positives, boundary errors
- Caption every figure with split = validation (or test only after lock)

### 7.3 Locked test evaluation (once)
- Date frozen, configs frozen, single pass
- No retuning after seeing test

## 8. Discussion

- Evidence for/against H1 and H2
- Compute trade-offs
- Failure modes and geographic generalization limits
- What we would change with more accelerator hours

## 9. Conclusion

- 3–5 sentences: answer the research question, main limitation, next step

## 10. Contributions and reproducibility

| Member | Primary ownership (proposal) |
| --- | --- |
| Sneha | Dataset integrity, geospatial/overlap checks, abs-diff baseline |
| Ritika | U-Net pipeline, paired augmentation, seed reproducibility |
| Danish | Siamese U-Net, loss comparison, error analysis |
| Rajesh | Metrics, efficiency benchmarks, statistical summaries, reproduction |

How to reproduce:
1. `notebooks/01_dataset_audit.ipynb`
2. `notebooks/02_experiments.ipynb` (`smoke` → `benchmark` → `full`)
3. Or `PYTHONPATH=src python scripts/run_smoke.py` for code-only checks

## 11. Assistance disclosure

- List tools used for writing/coding assistance (proposal already disclosed ChatGPT for formatting).
- Affirm experimental numbers come from executed runs.

## Appendix

- Exact configs (`config.json` per run)
- Software versions (`torch`, device: MPS/CUDA/CPU)
- Extra plots from [`results/unet_baseline/`](../results/unet_baseline/) and later experiment folders
