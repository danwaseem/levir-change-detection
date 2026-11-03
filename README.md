# Building Change Detection in Multi-Temporal Satellite Images

DATA 255 Deep Learning term project, Fall 2026

This project studies whether deep learning can identify building construction and demolition by comparing satellite images of the same location taken at different times. The primary comparison is a standard U-Net versus a Siamese U-Net. Experiments will evaluate segmentation accuracy, class imbalance, training stability, and computational cost.

**Status:** Dataset audit complete. The standard U-Net baseline has been trained and evaluated on the validation split. Siamese U-Net and ablation experiments are planned. No final test-set results are reported.

## Research question

Does a Siamese U-Net improve building change detection over a standard U-Net, and how do loss functions and computational constraints affect detection accuracy?

## Dataset

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) provides 637 pairs of 1024 x 1024 RGB satellite images, accompanied by binary building-change masks. The original images are from Texas and were captured at different times.

| Split | Original pairs | Non-overlapping 256 x 256 paired tiles |
| --- | ---: | ---: |
| Train | 445 | 7,120 |
| Validation | 64 | 1,024 |
| Test | 128 | 2,048 |
| Total | 637 | 10,192 |

Each sample has an image from time A, an image from time B, and a mask showing changed building pixels. The project uses **only train and validation** for model development. The test split is reserved for the final evaluation.

**Data use:** The dataset creators restrict LEVIR-CD to academic use. Follow the original dataset's usage terms and Google Earth imagery terms. This repository does not distribute the satellite images or labels.

**Source paper:** Hao Chen and Zhenwei Shi (2020), *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. DOI: [10.3390/rs12101662](https://doi.org/10.3390/rs12101662).

## Current results

The completed lightweight U-Net uses a six-channel input, with the before and after RGB images concatenated, and predicts one binary change mask. Training used all 445 training image pairs, divided into 256 x 256 tiles. The model was selected using validation F1.

| Measure | U-Net validation result |
| --- | ---: |
| Best F1 (Dice) | **0.7319** |
| IoU | 0.5772 |
| Precision | 0.6912 |
| Recall | 0.7778 |
| Best epoch | 24 of 30 |
| Trainable parameters | 483,169 |
| Total training time | 161.92 minutes |

The run used an Apple M5 MacBook Pro with 16 GB unified memory and PyTorch MPS, seed 1511, batch size 4, an Adam optimizer, learning rate 0.001, and BCE plus soft Dice loss. The prediction threshold was fixed at 0.5. These values are from **validation**, not the held-out test set, and are not a multi-seed estimate.

See [training metrics](reports/unet_baseline/metrics.csv), [run summary](reports/unet_baseline/summary.json), [learning curves](reports/unet_baseline/learning_curves.png), and [validation prediction gallery](reports/unet_baseline/validation_gallery.png).

![Validation predictions for the U-Net baseline](reports/unet_baseline/validation_gallery.png)

## Repository layout

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   └── 02_unet_baseline.ipynb
├── reports/
│   └── unet_baseline/
│       ├── summary.json
│       ├── metrics.csv
│       ├── learning_curves.png
│       └── validation_gallery.png
└── results/                       # created locally, excluded from Git
```

The notebooks are stored without embedded execution outputs to keep Git diffs readable. The actual baseline outputs are saved under `reports/unet_baseline/`. Model checkpoints and raw datasets are intentionally excluded from Git.

## Local setup

Python 3.11 is recommended for the existing notebook workflow.

```bash
git clone <YOUR_REPOSITORY_URL>
cd levir-change-detection
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name levir-cd --display-name "Python (LEVIR-CD)"
```

Download the original LEVIR-CD dataset from the [authors' dataset page](https://justchenhao.github.io/LEVIR/). Place `train.zip`, `val.zip`, and the extracted `test/` folder in a sibling dataset directory:

```text
SJSU/data255/
├── datasets/
│   └── levir-cd/
│       ├── train.zip
│       ├── val.zip
│       └── test/
│           ├── A/
│           ├── B/
│           └── label/
└── levir-change-detection/
    ├── notebooks/
    └── ...
```

The notebooks infer the repository root from their working directory and locate the dataset at `../datasets/levir-cd` by default. To use different locations, set environment variables before opening Jupyter:

```bash
export LEVIR_PROJECT_ROOT="$PWD"
export LEVIR_DATA_ROOT="/your/actual/path/to/levir-cd"
jupyter lab
```

## Reproduce the baseline

1. Open `notebooks/01_dataset_audit.ipynb` and select the `Python (LEVIR-CD)` kernel. Run the cells to locate and safely extract the dataset, verify the A/B/mask triplets, and generate `results/levir_audit/split_manifest.csv`. The local manifest contains absolute paths and must be regenerated on each machine. The dataset audit uses `CHECK_ALL = True` and the optional small pilot is disabled by default.
2. Open `notebooks/02_unet_baseline.ipynb`. Set `cfg.RUN_MODE = "benchmark"` to run one complete training and validation epoch, or set `cfg.RUN_MODE = "full"` for the 30-epoch experiment. The notebooks default to the benchmark to avoid unintentional full training.
3. For a new full run, execute the cells from a restarted kernel. Training writes to `results/unet_baseline/full_seed1511_tile256/`, including model checkpoints, metrics, configuration, and visualizations. Keep the `best_unet.pt` checkpoint locally. Checkpoints are excluded from Git.
4. Select checkpoints by validation F1 only. Do not evaluate the test data until all model choices and evaluation rules are frozen.

### Baseline configuration

| Setting | Value |
| --- | --- |
| Network | Lightweight standard U-Net with base channels 16 |
| Input | 6 channels: RGB at time A plus RGB at time B |
| Training/validation tiles | 256 x 256, non-overlapping |
| Batch size | 4 |
| Seed | 1511 |
| Optimizer | Adam, learning rate 0.001, weight decay 0.0001 |
| Loss | BCEWithLogits plus soft Dice |
| Epochs | Maximum 30, early-stopping patience 8 after epoch 12 |
| Mask processing | Pixels >= 128 are treated as changed |
| Checkpoint choice | Highest validation F1 |
| Metrics | Pooled changed-class F1, IoU, precision, recall |

All geometric augmentations are synchronized across both images and the target mask. Splitting is done at the original image-pair level, before generating tiles.

## Dataset audit and limitations

The audit verified that the 445 training, 64 validation, and 128 test entries have matching A/B/mask filenames. All 509 train/validation image triples passed dimension checks. Changed pixels occupy only a small fraction of the masks, around 4 to 5 percent on average in the audited splits.

Some masks contained grayscale values besides 0 and 255. This implementation consistently binarizes at 128; those cases warrant visual review. Matching filenames do not rule out geographic overlap across nearby satellite scenes, so a geography-aware leakage check is still planned.

The current U-Net results are from one seed. Validation performance fluctuated between epochs; multi-seed reliability analysis has not yet been completed. Performance on Texas building changes should not be interpreted as proof of generalization to other locations, sensors, or tasks.

## Planned work

- Implement and train Siamese U-Net using the same split, tile size, metrics, and comparable training budget.
- Compare standard U-Net and Siamese U-Net on validation F1, IoU, precision, recall, parameters, and inference latency.
- Run a controlled loss-function ablation, such as BCE alone versus BCE plus Dice.
- Repeat the main comparison using additional seeds where feasible.
- Analyze false positives, missed buildings, and inaccurate change boundaries.
- Audit possible geographic overlap, then freeze model selection and run the held-out test evaluation once.
- Provide reproducible evaluation instructions for a teammate to verify a saved checkpoint.

## Credits and responsible use

LEVIR-CD was introduced by Chen and Shi (2020). The project uses the public dataset strictly under the original academic-use conditions. Standard PyTorch components and research ideas are credited to their sources in the final report. The final submission should describe the team's individual contributions and disclose material AI assistance, including assistance used to draft or edit project documentation and code.
