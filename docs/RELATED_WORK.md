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

## Citation stubs (for the final report)

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}

@inproceedings{daudt2018fully,
  title={Fully Convolutional Siamese Networks for Change Detection},
  author={Caye Daudt, Rodrigo and Le Saux, Bertrand and Boulch, Alexandre},
  booktitle={IEEE ICIP},
  year={2018}
}

@article{chen2020stanet,
  title={A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
  author={Chen, Hao and Shi, Zhenwei},
  journal={Remote Sensing},
  volume={12},
  number={10},
  pages={1662},
  year={2020}
}
```
