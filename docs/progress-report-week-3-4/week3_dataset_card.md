# Dataset Card — LEVIR-CD  
**DATA 255 Deep Learning Vision · Week 3 checkpoint**  
**Team:** BuildSafe Vision — Sneha Singh, Ritika Mukesh Neema, Danish Waseem, Rajesh Paruchuri  
**Repository:** https://github.com/danwaseem/levir-change-detection

## 1. Dataset identity

| Item | Detail |
| --- | --- |
| Name | LEVIR-CD (Building Change Detection) |
| Citation | Chen & Shi, *Remote Sensing*, 2020 |
| Source / access | https://justchenhao.github.io/LEVIR/ |
| License / use | Academic research; Google Earth terms apply |
| Intended use here | Course research prototype for binary building change segmentation |
| Not for | Operational monitoring, medical, surveillance, or biometric decisions |

Imagery is stored locally via `LEVIR_DATA_ROOT` and is **not** committed to Git.

## 2. Task and labels

- **Input:** Aligned bi-temporal RGB pair \(A\) (earlier date), \(B\) (later date), each \(1024\times1024\) (~0.5 m/pixel, Texas Google Earth scenes).
- **Target:** Binary mask of building **construction or demolition** (not every radiometric difference).
- **Preprocessing:** Masks binarized at intensity \(\ge 128\).

## 3. Split plan and sample counts

Official pair split is preserved. Each \(1024\times1024\) image is tiled into **16** non-overlapping \(256\times256\) patches **after** the pair is assigned to a split.

| Split | Pairs | \(256\times256\) tiles |
| --- | ---: | ---: |
| Train | 445 | 7,120 |
| Validation | 64 | 1,024 |
| Test | 128 | 2,048 |
| **Total** | **637** | **10,192** |

**Class imbalance.** Changed pixels are a small minority (~**4–5%**). Overall accuracy is misleading; we use **change-class F1** as the primary metric.

## 4. Leakage audit

Filename-stem overlap across official splits is **0**  
(`results/levir_audit/stem_overlap_report.json`; manifest `results/levir_audit/split_manifest.csv`).

This rules out trivial identity leakage via repeated filenames. It does **not** prove geographic independence between nearby scenes; residual geographic correlation remains a known risk.

## 5. Preprocessing, augmentation, and test isolation

- Fit / select using **train and validation only**.
- Test set stays untouched until a later freeze.
- When enabled, geometric augmentation (horizontal/vertical flip, \(90^\circ\) rotation) is applied **identically** to \(A\), \(B\), and the mask.

## 6. Initial baseline result (Week 3)

Runnable reference baseline (absolute RGB difference with validation-selected threshold):

```bash
export LEVIR_DATA_ROOT=/path/to/levir-cd
PYTHONPATH=src python scripts/run_baseline.py
```

| Method | Split | Threshold | Change-class F1 | IoU |
| --- | --- | ---: | ---: | ---: |
| Absolute RGB difference | Validation | 0.4 | **0.129** | 0.069 |

Artifact: `results/absdiff_baseline/summary.json`.

## 7. Representative examples

Visual audit of train pairs (before / after / mask) is in `notebooks/01_dataset_audit.ipynb`.  
Qualitative prediction examples from the early FC-EF feasibility run are in `results/unet_baseline/validation_gallery.png` (Week 3–4 checkpoint).

**Observation.** Larger building changes are usually visible in the mask; small structures and soft boundaries are harder. Empty tiles with no change are common after tiling.

## 8. Risks and limitations

1. Texas Google Earth coverage only — transfer to other regions/sensors is unproven.  
2. Severe class imbalance favors trivial “all unchanged” predictors.  
3. Stem overlap is only a partial geographic-leakage check.  
4. Absolute difference is weak under illumination/vegetation shift (F1 0.129), so a learned model is required for the main investigation.
