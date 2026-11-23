# Results

Numbers come from `summary.json` files. Validation F1 selects each checkpoint.
Overall accuracy is not the selection metric. Synthetic smoke runs are not LEVIR scores.

## Summary

| Method | Split | F1 | Notes |
| --- | --- | ---: | --- |
| Abs RGB difference | val | 0.129 | Non-learned reference; threshold 0.4 |
| FC-EF (BCE+Dice, mean 3 seeds) | val | 0.718 | Principal comparison |
| FC-Siam-diff (BCE+Dice, mean 3 seeds) | val | 0.543 | H1 not supported |
| FC-Siam-conc (BCE+Dice, seed 1511) | val | 0.773 | Single seed |
| FC-EF BCE (locked) | val → test | 0.778 → 0.780 | Chosen after freeze; no retune |

Means are population standard deviation, rounded: FC-EF **0.718±0.011**, FC-Siam-diff **0.543±0.045**.

## Per-run validation

| Run | Paper name | Loss | Aug | Seed | Val F1 | IoU | P | R | Boundary F1 | Params | Infer (s) | Synthetic |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full | FC-Siam-conc | bce_dice | True | 1511 | 0.7730 | 0.6300 | 0.8357 | 0.7191 | 0.3518 | 563889 | 0.0028 | False |
| full | FC-Siam-diff | bce_dice | True | 123 | 0.5208 | 0.3521 | 0.9409 | 0.3601 | 0.2190 | 482737 | 0.0021 | False |
| full | FC-Siam-diff | bce_dice | True | 1511 | 0.5033 | 0.3363 | 0.6735 | 0.4018 | 0.2105 | 482737 | 0.0021 | False |
| full | FC-Siam-diff | bce_dice | True | 42 | 0.6063 | 0.4350 | 0.6083 | 0.6042 | 0.2265 | 482737 | 0.0021 | False |
| full | FC-EF (6-channel U-Net) | bce | True | 1511 | 0.7777 | 0.6363 | 0.7511 | 0.8063 | 0.3378 | 483169 | 0.0014 | False |
| full | FC-EF (6-channel U-Net) | bce_dice | True | 123 | 0.7261 | 0.5700 | 0.7529 | 0.7012 | 0.3276 | 483169 | 0.0014 | False |
| full | FC-EF (6-channel U-Net) | bce_dice | True | 1511 | 0.7024 | 0.5413 | 0.7339 | 0.6734 | 0.3108 | 483169 | 0.0015 | False |
| full | FC-EF (6-channel U-Net) | bce_dice | True | 42 | 0.7270 | 0.5711 | 0.7067 | 0.7485 | 0.3234 | 483169 | 0.0014 | False |
| full | FC-EF (6-channel U-Net) | bce_dice | False | 1511 | 0.7455 | 0.5942 | 0.7644 | 0.7275 | 0.3428 | 483169 | 0.0014 | False |

## Principal comparison (BCE+Dice, augmentation on)

| Model | Seed | Val F1 | IoU | P | R |
| --- | ---: | ---: | ---: | ---: | ---: |
| FC-EF | 1511 | 0.702 | 0.541 | 0.734 | 0.673 |
| FC-EF | 42 | 0.727 | 0.571 | 0.707 | 0.749 |
| FC-EF | 123 | 0.726 | 0.570 | 0.753 | 0.701 |
| FC-EF mean | 3 seeds | 0.718±0.011 | — | — | — |
| FC-Siam-diff | 1511 | 0.503 | 0.336 | 0.674 | 0.402 |
| FC-Siam-diff | 42 | 0.606 | 0.435 | 0.608 | 0.604 |
| FC-Siam-diff | 123 | 0.521 | 0.352 | 0.941 | 0.360 |
| FC-Siam-diff mean | 3 seeds | 0.543±0.045 | — | — | — |
| FC-Siam-conc | 1511 | 0.773 | 0.630 | 0.836 | 0.719 |

FC-Siam-conc is one seed under the same loss and augmentation. It is not in the FC-Siam-diff mean.

## FC-EF ablations (seed 1511)

| Setting | Val F1 | IoU |
| --- | ---: | ---: |
| BCE+Dice, aug on | 0.702 | 0.541 |
| BCE only, aug on | 0.778 | 0.636 |
| BCE+Dice, aug off | 0.745 | 0.594 |

## Locked test (FC-EF + BCE + aug, seed 1511)

Best validation epoch 13. Test was scored once. `retuned_after_test` is false.

| Split | F1 | IoU | P | R | Boundary F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val (best) | 0.778 | 0.636 | 0.751 | 0.806 | 0.338 |
| Test (once) | 0.780 | 0.639 | 0.770 | 0.790 | 0.356 |

## Locked-test error strata

Tile bootstrap of pooled test F1, 2,000 resamples, seed 0: 95% interval **[0.770, 0.790]**. No retrain.

| Stratum | N | Pooled F1 | Mean tile F1 | Key failure signal |
| --- | ---: | ---: | ---: | --- |
| Empty (0 px) | 1113 | — | — | False-alarm rate 19.0% (212) |
| Small (1–256) | 61 | 0.088 | 0.078 | Recall 0.083; 56/61 missed |
| Large (>256) | 874 | 0.790 | 0.691 | Recall 0.702 |
| All tiles | 2048 | 0.780 | 0.297 | CI 0.770–0.790 |

## Compute (seed 1511)

| Model | Params | Train | Batch-1 | Val F1 |
| --- | ---: | ---: | ---: | ---: |
| FC-EF (BCE) | 483169 | 50.7 min | 0.0014 s | 0.778 |
| FC-Siam-diff | 482737 | 46.5 min | 0.0021 s | 0.503 |
| FC-Siam-conc | 563889 | 48.4 min | 0.0028 s | 0.773 |
