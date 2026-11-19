# Dataset card — LEVIR-CD (Week 3 checkpoint)

**Team:** BuildSafe Vision (DATA 255, Fall 2026)  
**Dataset:** LEVIR-CD (Chen & Shi, 2020)  
**Source:** https://justchenhao.github.io/LEVIR/  
**Use:** Academic research only; Google Earth terms apply. Not for operational deployment.

## Summary

| Item | Value |
| --- | --- |
| Task | Binary building change segmentation (construction / demolition) |
| Inputs | Paired RGB images \(A\) (earlier), \(B\) (later), \(1024\times1024\) |
| Labels | Binary change masks |
| Pairs | 637 total |
| Official split | Train 445 / Val 64 / Test 128 |
| Tiles | Non-overlapping \(256\times256\) **after** split (7,120 / 1,024 / 2,048) |
| Change-pixel rate | ~4–5% (rare positive class) |
| Access | Local `LEVIR_DATA_ROOT`; imagery **not** in Git |

## Split / leakage

- Preserve official pair IDs; tile only after split assignment.
- Stem-overlap audit: **0** shared stems across splits (`results/levir_audit/stem_overlap_report.json`).
- Manifest: `results/levir_audit/split_manifest.csv`.
- Masks binarized at intensity ≥ 128.

## Preprocessing and augmentation

- Train/val only for fitting and selection; test isolated until freeze.
- Paired geometric aug (flip / rot90) applied identically to \(A\), \(B\), mask when enabled.

## Representative examples

See training visual audit in `notebooks/01_dataset_audit.ipynb` and locked-model gallery  
`results/experiments/full_unet_bce_aug_seed1511_tile256/validation_gallery.png`.

## Risks / limitations

- Texas Google Earth coverage only; possible residual geographic correlation across splits.
- Class imbalance; small buildings hard.
- No medical / surveillance / biometric use.

## Runnable baseline (Week 3)

```bash
export LEVIR_DATA_ROOT=/path/to/levir-cd
PYTHONPATH=src python scripts/run_baseline.py
```

Initial reference result (validation): absolute RGB difference F1 **0.129** (`results/absdiff_baseline/summary.json`).
