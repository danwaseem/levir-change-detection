# Comparison table

Built from `summary.json` files. Validation F1 selects each checkpoint.
Overall accuracy is not the selection metric. Synthetic smoke runs are not LEVIR scores.

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
