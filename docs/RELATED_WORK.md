# Related work → what this repo implements

This note maps the proposal citations to concrete code choices. It is **not** a claim that we reproduce STANet.

## Papers we follow

### 1. Ronneberger et al., U-Net (MICCAI 2015)
- **Idea:** encoder–decoder with skip connections for dense prediction.
- **In this repo:** lightweight U-Net blocks in `src/levir_cd/models.py` (`ConvBlock`, `UNet`).
- **Adaptation:** 6-channel early-fusion input for bi-temporal RGB; base width 16 for MPS/CPU feasibility.

### 2. Daudt et al., Fully Convolutional Siamese Networks (ICIP 2018)
Daudt compares three FCN designs. We implement the same **family**:

| Daudt name | Our `MODEL` key | Fusion | Code |
| --- | --- | --- | --- |
| **FC-EF** | `unet` / `fc_ef` | Concatenate RGB_A∥RGB_B as 6 input channels | `UNet` |
| **FC-Siam-diff** | `siamese` / `siamese_diff` / `fc_siam_diff` | Shared encoder + `|fA−fB|` skips | `SiameseUNet(fusion="diff")` |
| **FC-Siam-conc** | `siamese_conc` / `fc_siam_conc` | Shared encoder + `concat(fA,fB)` skips | `SiameseUNet(fusion="conc")` |

Daudt reported strong results for the Siamese variants (often **diff**). Our proposal’s principal comparison is **FC-EF vs FC-Siam-diff**; **FC-Siam-conc** is included so the Daudt triad is complete under the same tile/loss/seed protocol.

### 3. Chen & Shi, STANet + LEVIR-CD (Remote Sensing 2020)
- **Dataset:** LEVIR-CD (637×1024² pairs, building construction/decline, Texas, 0.5 m/px).
- **Metrics we match:** change-class **precision / recall / F1** (plus IoU as a secondary score).
- **Protocol we match in spirit:** use official-style train/val/test isolation; crop to 256×256 tiles; do not tune on test.
- **What we deliberately do *not* reproduce:** spatial–temporal attention (BAM/PAM) or batch-balanced contrastive loss (BCL). The proposal states STANet attention is out of scope; student contribution is the controlled U-Net/Siamese comparison + ablations.

Official STANet code (reference only): https://github.com/justchenhao/STANet  
Dataset page: https://justchenhao.github.io/LEVIR/

## How this strengthens the project vs a thin baseline-only repo

1. Full Daudt triad under one dataloader/metrics API.
2. Loss ablation (BCE vs BCE+Dice) for class imbalance (~4–5% changed pixels).
3. Paired geometric augmentation ablation.
4. Non-learned absolute-difference control.
5. Pooled **and** per-image validation summaries for failure analysis.
6. Efficiency logging (params, train time, checkpoint size, batch-1 latency).
7. Dataset-free smoke tests so coding can be verified without LEVIR on disk.

## Gaps checked against the papers and later LEVIR benchmarks

Sources: Daudt et al. 2018, Chen & Shi 2020, and later public LEVIR tables (Open-CD; "A Change Detection Reality Check"). Those later tables are **context only**. They use heavier schedules and often ImageNet backbones. They are not our scores.

| Check | Status |
| --- | --- |
| FC-EF, FC-Siam-diff, FC-Siam-conc under one protocol | Implemented |
| 256×256 non-overlapping tiles, official split kept | Implemented |
| Change-class precision, recall, F1, IoU | Implemented; F1 selects the checkpoint |
| Overall accuracy | Recorded as `val_accuracy` and **not** used for selection (Chen & Shi and the proposal warn it is misleading) |
| Boundary errors | `val_boundary_f1` on a 3×3 boundary band |
| Missed small buildings / empty-tile false alarms | `failure_summary.json` (small = ≤256 changed pixels on a tile) |
| BCE vs BCE+Dice and paired flip/rot90 ablations | Switches exist; still need real LEVIR runs |
| STANet attention, BCL loss, ImageNet/ResNet encoders | Left out on purpose. Adding them would change the controlled comparison. |

Published LEVIR **test** F1 numbers other groups report for the Daudt models are about 0.81–0.86 under their own training budgets. Our feasibility number is validation F1 **0.7319** for the lightweight FC-EF, seed 1511. Do not present the published test numbers as this team's result.

## Citation stubs (for the final report)

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  pages={234--241},
  year={2015},
  doi={10.1007/978-3-319-24574-4_28}
}

@inproceedings{daudt2018fully,
  title={Fully Convolutional Siamese Networks for Change Detection},
  author={Caye Daudt, Rodrigo and Le Saux, Bertrand and Boulch, Alexandre},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  pages={4063--4067},
  year={2018},
  doi={10.1109/ICIP.2018.8451652}
}

@article{chen2020stanet,
  title={A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
  author={Chen, Hao and Shi, Zhenwei},
  journal={Remote Sensing},
  volume={12},
  number={10},
  pages={1662},
  year={2020},
  doi={10.3390/rs12101662}
}
```
