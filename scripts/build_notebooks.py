#!/usr/bin/env python3
"""Rebuild project notebooks with clear markdown sections and comments."""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    # Keep a trailing newline on each line for Jupyter compatibility.
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


def build_audit() -> None:
    cells: list[dict] = []

    cells.append(
        md(
            """
# Notebook 01 — LEVIR-CD dataset audit

**Team:** BuildSafe Vision (DATA 255, Fall 2026)  
**Purpose:** Verify the downloaded LEVIR-CD dataset, build a reproducible train/val manifest, and leave model training to notebook 02.

## What this notebook does
1. Locate `train.zip` / `val.zip` (and an existing `test/` folder).
2. Extract archives safely (once).
3. Confirm matching `A` / `B` / `label` filenames.
4. Check dimensions, mask values, and changed-pixel balance on **train/val only**.
5. Visualize a few aligned before/after/mask examples.
6. Run a filename stem-overlap check across splits.
7. Write `results/levir_audit/split_manifest.csv` for notebook 02.

## What this notebook does *not* do
- It does **not** train the final U-Net or Siamese models (use `02_experiments.ipynb`).
- It does **not** tune on or visualize the held-out **test** split for model selection.
- Optional legacy pilot training stays **off** (`RUN_PILOT = False`).

## Prerequisites
- LEVIR-CD placed under `LEVIR_DATA_ROOT` (default: `../datasets/levir-cd`).
- Academic-use terms for LEVIR-CD / Google Earth imagery respected.
"""
        )
    )

    cells.append(
        md(
            """
## 1. Paths and switches

Set project/data roots and audit flags here. Absolute paths in the saved manifest are machine-specific, so re-run this notebook after moving the dataset.
"""
        )
    )
    cells.append(
        code(
            """
from pathlib import Path
import os

# Jupyter usually starts inside notebooks/; allow an env override for cloud/lab machines.
_here = Path.cwd().resolve()
_default_project = _here.parent if _here.name == "notebooks" else _here
PROJECT_ROOT = Path(os.environ.get("LEVIR_PROJECT_ROOT", str(_default_project))).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get("LEVIR_DATA_ROOT", str(PROJECT_ROOT.parent / "datasets" / "levir-cd"))
).expanduser().resolve()
OUTPUT_DIR = PROJECT_ROOT / "results" / "levir_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_ARCHIVES = True   # Extract train.zip / val.zip once; never modifies the originals
RUN_PILOT = False         # Legacy mini-train; keep False — use notebook 02 / scripts/run_smoke.py
SEED = 1511

print("Project root:", PROJECT_ROOT)
print("Dataset folder:", DATA_ROOT)
print("Dataset exists:", DATA_ROOT.exists())
print("Audit outputs:", OUTPUT_DIR)
assert DATA_ROOT.is_dir(), (
    "DATA_ROOT not found. Put train.zip, val.zip, and test/ there, or set LEVIR_DATA_ROOT."
)
"""
        )
    )

    cells.append(
        md(
            """
## 2. Locate ZIP archives and inspect contents

We only *read* archive member lists here. A CRC check runs for smaller archives; huge ZIPs skip the expensive full test to save time.
"""
        )
    )
    cells.append(
        code(
            """
from collections import Counter
import zipfile

archives = sorted(
    p for p in DATA_ROOT.rglob("*.zip")
    if p.is_file() and "_extracted" not in p.parts
)
print("Archives found:")
for p in archives:
    print(f"  {p.relative_to(DATA_ROOT)} ({p.stat().st_size / (1024**2):.1f} MiB)")

archive_map = {}
for split in ("train", "val"):
    choices = [p for p in archives if p.name.lower() == f"{split}.zip"]
    if len(choices) > 1:
        print(f"WARNING: multiple {split}.zip files; using {choices[0]}")
    archive_map[split] = choices[0] if choices else None

for split, zpath in archive_map.items():
    if zpath is None:
        print(f"{split}.zip: not found (maybe already extracted)")
        continue
    with zipfile.ZipFile(zpath) as z:
        # Full CRC on multi-hundred-MB archives is slow; spot-check structure instead.
        bad = z.testzip() if zpath.stat().st_size < 350_000_000 else None
        print(f"{split}.zip: {len(z.infolist())} entries; sample paths:")
        for name in z.namelist()[:8]:
            print("  ", name)
        if bad:
            raise RuntimeError(f"Damaged archive member: {bad}")
"""
        )
    )

    cells.append(
        md(
            """
## 3. Extract train/val archives safely

Extraction is idempotent: a `.extraction_complete` marker skips work on re-runs. Path traversal and symlink members are rejected.
"""
        )
    )
    cells.append(
        code(
            """
import shutil

EXTRACT_ROOT = DATA_ROOT / "_extracted"

def safe_extract(zpath: Path, destination: Path) -> None:
    \"\"\"Extract a ZIP while blocking path traversal and symlink members.\"\"\"
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zpath) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink not allowed: {member.filename}")
        z.extractall(destination)

for split in ("train", "val"):
    zpath = archive_map[split]
    out = EXTRACT_ROOT / split
    marker = out / ".extraction_complete"
    if zpath is None:
        print(f"{split}: no ZIP; will search for an existing extracted split")
    elif marker.exists():
        print(f"{split}: already extracted at {out}")
    elif EXTRACT_ARCHIVES:
        if out.exists() and any(out.iterdir()):
            print(f"{split}: removing incomplete extraction at {out}")
            shutil.rmtree(out)
        print(f"Extracting {zpath.name}...")
        safe_extract(zpath, out)
        marker.write_text(f"source={zpath}\\nsize={zpath.stat().st_size}\\n")
        print(f"Done: {out}")
    else:
        print(f"{split}: extraction disabled")
"""
        )
    )

    cells.append(
        md(
            """
## 4. Resolve `A` / `B` / `label` directories

Handles both flat extracted layouts and ZIPs that wrap an extra top-level folder.
"""
        )
    )
    cells.append(
        code(
            """
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def valid_split_dir(d: Path) -> bool:
    return d.is_dir() and all((d / part).is_dir() for part in ("A", "B", "label"))

def locate_split(split: str):
    candidates = [
        p for p in DATA_ROOT.rglob("*")
        if p.is_dir() and p.name == split and valid_split_dir(p)
    ]
    if not candidates:
        fallback_base = EXTRACT_ROOT / split
        candidates = (
            [p for p in fallback_base.rglob("*") if valid_split_dir(p)]
            if fallback_base.exists() else []
        )
        if valid_split_dir(fallback_base):
            candidates.append(fallback_base)
    if not candidates:
        return None
    # Prefer the safe `_extracted/<split>` tree when present.
    candidates.sort(
        key=lambda p: (0 if "_extracted" in p.parts and split in ("train", "val") else 1, len(p.parts))
    )
    return candidates[0]

SPLIT_DIRS = {s: locate_split(s) for s in ("train", "val", "test")}
for s, d in SPLIT_DIRS.items():
    print(f"{s:5s}: {d if d else 'NOT FOUND'}")

assert SPLIT_DIRS["train"] is not None, "Training folder missing."
assert SPLIT_DIRS["val"] is not None, "Validation folder missing."
"""
        )
    )

    cells.append(
        md(
            """
## 5. Pair images and masks; export the manifest

Every stem must exist in `A`, `B`, and `label`. The CSV written here is the contract between notebook 01 and notebook 02.
"""
        )
    )
    cells.append(
        code(
            """
import pandas as pd

def files_by_stem(folder: Path) -> dict:
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    stems = [p.stem for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate file stems in {folder}")
    return {p.stem: p for p in paths}

PAIRS = {}
records = []
for split, sd in SPLIT_DIRS.items():
    if sd is None:
        continue
    a, b, y = (files_by_stem(sd / k) for k in ("A", "B", "label"))
    names = sorted(set(a) & set(b) & set(y))
    unmatched = (set(a) | set(b) | set(y)) - set(names)
    print(
        f"{split}: A={len(a)}, B={len(b)}, label={len(y)}, "
        f"complete triplets={len(names)}, unmatched={len(unmatched)}"
    )
    if unmatched:
        print("  First unmatched stems:", sorted(unmatched)[:10])
    assert not unmatched, f"Mismatched file names in {split}; inspect before continuing."
    PAIRS[split] = [(a[name], b[name], y[name]) for name in names]
    for name in names:
        records.append(
            {
                "split": split,
                "stem": name,
                "image_A": str(a[name]),
                "image_B": str(b[name]),
                "mask": str(y[name]),
            }
        )

manifest = pd.DataFrame(records)
manifest_path = OUTPUT_DIR / "split_manifest.csv"
manifest.to_csv(manifest_path, index=False)
print("Saved:", manifest_path)
print(manifest.groupby("split").size())
"""
        )
    )

    cells.append(
        md(
            """
## 6. Dimension, mask-value, and class-balance checks (train/val only)

Proposal rule: binarize masks at intensity `>= 128`. Unusual grayscale values are recorded for review; they are not used as soft labels.
"""
        )
    )
    cells.append(
        code(
            """
import numpy as np
from PIL import Image
import random

random.seed(SEED)
CHECK_ALL = True  # Set False only for a quick subsample while debugging paths

quality_rows = []
for split in ("train", "val"):
    examples = PAIRS[split] if CHECK_ALL else random.sample(PAIRS[split], min(50, len(PAIRS[split])))
    for a_path, b_path, y_path in examples:
        with Image.open(a_path) as ia, Image.open(b_path) as ib, Image.open(y_path) as im:
            ashape, bshape, mshape = ia.size, ib.size, im.size
            mask = np.asarray(im.convert("L"))
        unique_values = np.unique(mask)
        quality_rows.append(
            {
                "split": split,
                "stem": a_path.stem,
                "width": ashape[0],
                "height": ashape[1],
                "sizes_match": ashape == bshape == mshape,
                "mask_values": ",".join(map(str, unique_values.tolist()[:12])),
                "changed_fraction": float((mask >= 128).mean()),
            }
        )

quality = pd.DataFrame(quality_rows)
quality.to_csv(OUTPUT_DIR / "image_mask_quality_sample.csv", index=False)
print(quality.groupby("split")["changed_fraction"].agg(["count", "mean", "min", "max"]))
print("Sizes all match:", bool(quality["sizes_match"].all()))
print("Observed mask-value examples:", quality["mask_values"].value_counts().head(8).to_dict())
assert quality["sizes_match"].all(), "Some before/after/mask dimensions differ."
"""
        )
    )

    cells.append(
        md(
            """
## 7. Visual audit — before / after / change mask

Only **training** examples are shown. This is a human check that masks align with building appearance/disappearance, not a quantitative metric.
"""
        )
    )
    cells.append(
        code(
            """
import matplotlib.pyplot as plt

train_pairs = PAIRS["train"]
selected = random.sample(train_pairs, min(35, len(train_pairs)))
scored = []
for row in selected:
    with Image.open(row[2]) as im:
        ratio = (np.asarray(im.convert("L")) >= 128).mean()
    scored.append((ratio, row))
scored.sort(key=lambda x: x[0], reverse=True)
examples = [pair for _, pair in scored[: min(3, len(scored))]]

fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)), squeeze=False)
for i, (a_path, b_path, m_path) in enumerate(examples):
    with Image.open(a_path) as im:
        a = np.asarray(im.convert("RGB"))
    with Image.open(b_path) as im:
        b = np.asarray(im.convert("RGB"))
    with Image.open(m_path) as im:
        mask = np.asarray(im.convert("L")) >= 128
    for j, arr in enumerate((a, b, mask)):
        axes[i, j].imshow(arr, cmap="gray" if j == 2 else None, vmin=0 if j == 2 else None, vmax=1 if j == 2 else None)
        axes[i, j].axis("off")
    for j, title in enumerate(("Before (A)", "After (B)", "Change mask (label)")):
        axes[i, j].set_title(f"{title} | {a_path.stem}" if j == 0 else title)
plt.tight_layout()
figure_path = OUTPUT_DIR / "training_examples.png"
plt.savefig(figure_path, dpi=130, bbox_inches="tight")
plt.show()
print("Saved:", figure_path)
"""
        )
    )

    cells.append(
        md(
            """
## 8. Split sanity check — shared filenames

Identical stems in two splits are a **warning**, not automatic leakage. Confirm whether the images/locations are truly the same before deciding on exclusions.
"""
        )
    )
    cells.append(
        code(
            """
from itertools import combinations

for a, b in combinations(PAIRS, 2):
    fa = {p[0].stem for p in PAIRS[a]}
    fb = {p[0].stem for p in PAIRS[b]}
    common = sorted(fa & fb)
    print(f"{a} vs {b}: {len(common)} shared stems", common[:8])

print("Filename pairing audit complete. Test images were not used for model selection.")
"""
        )
    )

    cells.append(
        md(
            """
## 9. Stem-overlap report (written JSON)

Writes a small machine-readable summary under `results/levir_audit/`. This is still only a filename proxy — a fuller geospatial metadata audit can be added later without changing training code.
"""
        )
    )
    cells.append(
        code(
            """
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from levir_cd.geo_audit import write_geo_audit

report_path = write_geo_audit(OUTPUT_DIR / "split_manifest.csv", OUTPUT_DIR)
print("Wrote", report_path)
print((OUTPUT_DIR / "stem_overlap_report.json").read_text()[:800])
"""
        )
    )

    cells.append(
        md(
            """
## 10. Legacy optional pilot (disabled)

The old inline one-epoch U-Net pilot is intentionally **not** the supported smoke path anymore.

| Goal | Use instead |
| --- | --- |
| Code smoke without LEVIR | `PYTHONPATH=src python scripts/run_smoke.py` |
| Interactive smoke / full runs | `notebooks/02_experiments.ipynb` |

Keep `RUN_PILOT = False` unless you are debugging this legacy cell on purpose.
"""
        )
    )
    cells.append(
        code(
            """
if not RUN_PILOT:
    print("Pilot skipped (recommended). Use notebook 02 or scripts/run_smoke.py for model checks.")
else:
    print("RUN_PILOT=True is legacy. Prefer notebook 02 so Siamese / ablations share one training path.")
"""
        )
    )

    cells.append(
        md(
            """
## Done — handoff checklist

After a clean run, share with the team:

1. `results/levir_audit/split_manifest.csv`
2. `results/levir_audit/image_mask_quality_sample.csv`
3. `results/levir_audit/training_examples.png`
4. `results/levir_audit/stem_overlap_report.json`
5. Notes on any unusual mask values or shared stems

**Next:** open `02_experiments.ipynb`.
- `RUN_MODE="smoke"` after coding (no dataset needed)
- `RUN_MODE="benchmark"` for one real epoch
- `RUN_MODE="full"` for the documented experiment runs
"""
        )
    )

    write_nb(Path("/workspace/notebooks/01_dataset_audit.ipynb"), cells)


def build_experiments() -> None:
    cells: list[dict] = []

    cells.append(
        md(
            """
# Notebook 02 — Experiments (U-Net, Siamese, ablations)

**Team:** BuildSafe Vision (DATA 255, Fall 2026)  
**Depends on:** notebook 01 manifest for `benchmark` / `full` modes (not required for `smoke`).

## What we implemented
| Piece | Location | Notes |
| --- | --- | --- |
| 6-channel U-Net baseline | `src/levir_cd/models.py` | Same lightweight width as the prior val F1 0.7319 run |
| Shared-encoder Siamese U-Net | `src/levir_cd/models.py` | Absolute-difference fusion at each scale |
| BCE and BCE+Dice losses | `src/levir_cd/losses.py` | Ablation 1 |
| Paired geometric augmentation | `src/levir_cd/data.py` | Ablation 2 (flip/rot synced across A, B, mask) |
| Non-learned abs-diff baseline | `src/levir_cd/baseline.py` | Validation-selected threshold |
| Train / val loop + latency log | `src/levir_cd/train.py` | Checkpoint by validation F1 only |
| Dataset-free smoke path | `scripts/run_smoke.py` | Synthetic tiles for coding checks |

## Run modes
| `RUN_MODE` | Data | Typical use |
| --- | --- | --- |
| `smoke` | Synthetic tiles | After coding, on CPU/cloud — no LEVIR download |
| `benchmark` | Full train/val tiles, 1 epoch | Timing / device sanity on Mac MPS or GPU lab |
| `full` | Full train/val, up to 30 epochs | Real comparison / ablation runs |

## Safety rules
- Primary metric: **changed-class F1** (not accuracy).
- Threshold fixed at **0.5** unless a validation-only study is predeclared.
- `ALLOW_TEST` stays `False` until model choices are frozen for the locked test evaluation.
"""
        )
    )

    cells.append(
        md(
            """
## 1. Imports and environment

Adds `src/` to `sys.path` so the notebook can import `levir_cd` whether you launched Jupyter from the repo root or from `notebooks/`.
"""
        )
    )
    cells.append(
        code(
            """
import os
import sys
from pathlib import Path

# Resolve project root, then expose the local package.
_here = Path.cwd().resolve()
_project = _here.parent if _here.name == "notebooks" else _here
_project = Path(os.environ.get("LEVIR_PROJECT_ROOT", str(_project))).expanduser().resolve()
sys.path.insert(0, str(_project / "src"))

from levir_cd.baseline import absolute_difference_baseline
from levir_cd.config import ExperimentConfig
from levir_cd.data import ChangeTiles, load_split_frames, make_loaders
from levir_cd.device import describe_runtime, pick_device, set_seed
from levir_cd.models import build_model, count_parameters
from levir_cd.train import maybe_copy_summary_to_reports, run_experiment

print("Project root:", _project)
print("Package import OK")
"""
        )
    )

    cells.append(
        md(
            """
## 2. Experiment configuration

Change the knobs below between runs. Defaults start in **`smoke`** so a fresh clone does not accidentally launch a multi-hour training job.
"""
        )
    )
    cells.append(
        code(
            """
cfg = ExperimentConfig(
    PROJECT_ROOT=str(_project),
    # --- choose one run mode ---
    RUN_MODE="smoke",          # "smoke" | "benchmark" | "full"
    # --- main comparison / ablations ---
    MODEL="unet",              # "unet" | "siamese"
    LOSS="bce_dice",           # "bce" | "bce_dice"
    AUGMENT=True,              # paired flips + 90° rotations
    SEED=1511,                 # also try 42 and 123 for the main comparison
    # --- training defaults for real LEVIR tiles ---
    TILE_SIZE=256,
    IMAGE_SIZE=1024,
    BATCH_SIZE=4,
    BASE_CHANNELS=16,
    LEARNING_RATE=1e-3,
    WEIGHT_DECAY=1e-4,
    FULL_EPOCHS=30,
    EARLY_STOP_PATIENCE=8,
    MIN_EPOCHS=12,
    THRESHOLD=0.5,
    # --- smoke-only size knobs (ignored in benchmark/full) ---
    SMOKE_TRAIN_TILES=8,
    SMOKE_VAL_TILES=4,
    SMOKE_EPOCHS=1,
    SMOKE_BATCH_SIZE=2,
)

# Faster CPU smoke defaults. Real LEVIR runs keep 256 / base-16 above.
if cfg.RUN_MODE == "smoke":
    cfg.TILE_SIZE = 64
    cfg.BASE_CHANNELS = 8

cfg.validate()
device = pick_device()
set_seed(cfg.SEED)

print(describe_runtime(device))
print("Run name :", cfg.run_name)
print("Manifest :", cfg.manifest_path)
print("Output   :", cfg.run_dir)
print("Epochs   :", cfg.effective_epochs(), "| batch:", cfg.effective_batch_size())
"""
        )
    )

    cells.append(
        md(
            """
## 3. Reference baseline — absolute RGB difference

Non-learned control from the proposal. On validation tiles we pick a threshold that maximizes changed-class F1. Skipped in `smoke` because there is no LEVIR manifest/imagery yet.
"""
        )
    )
    cells.append(
        code(
            """
baseline_result = None

if cfg.RUN_MODE == "smoke":
    print("Smoke mode: skip abs-diff baseline (synthetic data only).")
else:
    # Requires notebook 01 to have written split_manifest.csv on this machine.
    assert cfg.manifest_path.is_file(), (
        "Missing manifest. Run notebooks/01_dataset_audit.ipynb first."
    )
    train_df, val_df = load_split_frames(cfg.manifest_path)
    val_ds = ChangeTiles(
        val_df,
        size=cfg.TILE_SIZE,
        image_size=cfg.IMAGE_SIZE,
        augment=False,
    )
    # Reuse make_loaders helper; we only need the validation iterator.
    _, val_loader = make_loaders(val_ds, val_ds, batch_size=cfg.BATCH_SIZE)
    baseline_result = absolute_difference_baseline(val_loader)
    print("Absolute-difference baseline (validation-selected threshold):")
    for key, value in baseline_result.items():
        print(f"  {key}: {value}")
"""
        )
    )

    cells.append(
        md(
            """
## 4. Model summary (before training)

Quick parameter count and a dry forward pass shape check. Both models consume the same 6-channel dataloader tensor; Siamese splits channels inside the network.
"""
        )
    )
    cells.append(
        code(
            """
import torch

model_preview = build_model(cfg.MODEL, base_channels=cfg.BASE_CHANNELS).to(device)
n_params = count_parameters(model_preview)
print(f"Model: {cfg.MODEL}")
print(f"Trainable parameters: {n_params:,}")

with torch.no_grad():
    demo = model_preview(
        torch.zeros(1, 6, cfg.TILE_SIZE, cfg.TILE_SIZE, device=device)
    )
print("Forward shape:", tuple(demo.shape))
assert tuple(demo.shape) == (1, 1, cfg.TILE_SIZE, cfg.TILE_SIZE)
del model_preview, demo
"""
        )
    )

    cells.append(
        md(
            """
## 5. Train and validate

`run_experiment` owns the matched protocol: same tiles/metrics/checkpoint rule for U-Net and Siamese. Artifacts land under `results/experiments/<run_name>/`:

- `config.json` — frozen knobs for this run  
- `metrics.csv` — per-epoch train/val curves  
- `best_model.pt` / `last_model.pt` — checkpoints (gitignored)  
- `summary.json` — F1, params, timing, batch-1 latency  

Smoke/benchmark summaries are also copied under `reports/experiments/` for quick sharing.
"""
        )
    )
    cells.append(
        code(
            """
summary = run_experiment(cfg, device=device)
report_dir = maybe_copy_summary_to_reports(summary, Path(cfg.PROJECT_ROOT))

print("\\n=== Run summary ===")
for key in (
    "run_mode",
    "model",
    "loss",
    "augment",
    "seed",
    "best_epoch",
    "best_val_f1",
    "parameters",
    "batch_one_inference_seconds",
    "test_set_touched",
    "run_dir",
):
    print(f"{key}: {summary.get(key)}")
print("Report copy:", report_dir)
"""
        )
    )

    cells.append(
        md(
            """
## 6. Read learning curves (when metrics exist)

For `smoke`, the CSV has a single epoch — useful as a pipeline check, not a quality claim. For `full`, compare train/val loss and validation F1/IoU across epochs.
"""
        )
    )
    cells.append(
        code(
            """
import pandas as pd
import matplotlib.pyplot as plt

metrics_path = Path(summary["run_dir"]) / "metrics.csv"
history = pd.read_csv(metrics_path)
print(history.tail())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.epoch, history.train_loss, "o-", label="Train loss")
axes[0].plot(history.epoch, history.val_loss, "o-", label="Val loss")
axes[0].set(title=f"Loss ({cfg.LOSS})", xlabel="Epoch", ylabel="Loss")
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(history.epoch, history.val_f1, "o-", label="Val F1")
axes[1].plot(history.epoch, history.val_iou, "o-", label="Val IoU")
axes[1].set(title=f"{cfg.MODEL} change-pixel scores", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
axes[1].legend()
axes[1].grid(alpha=0.2)

fig.tight_layout()
curve_path = Path(summary["run_dir"]) / "learning_curves.png"
fig.savefig(curve_path, dpi=160, bbox_inches="tight")
plt.show()
print("Saved:", curve_path)
"""
        )
    )

    cells.append(
        md(
            """
## 7. Ablation and multi-seed checklist

Re-run **Section 2 → Section 5** with one change at a time. Do not peek at test data while selecting the winner.

### A. Main comparison (proposal H1)
1. `MODEL="unet"`, `LOSS="bce_dice"`, `AUGMENT=True`, seeds `1511`, `42`, `123`
2. `MODEL="siamese"` with the same loss/aug/seeds
3. Report mean ± std validation F1 / IoU / precision / recall, plus params and latency

### B. Loss ablation (proposal H2)
- Chosen architecture, `LOSS="bce"` vs `LOSS="bce_dice"`, other knobs fixed

### C. Paired augmentation ablation
- Chosen architecture, `AUGMENT=True` vs `AUGMENT=False`

### D. After freezing choices
- Locked **one-shot** test evaluation outside exploratory loops
- Error gallery: missed small buildings, false positives, weak boundaries
- Final write-up / presentation (report comes at the end)
"""
        )
    )
    cells.append(
        code(
            """
checklist = [
    ("Main", "unet vs siamese", "LOSS=bce_dice, AUGMENT=True, seeds 1511/42/123"),
    ("Loss", "bce vs bce_dice", "same model/aug/seed"),
    ("Augmentation", "AUGMENT True vs False", "same model/loss/seed"),
    ("Efficiency", "params / train time / batch-1 latency", "logged in summary.json"),
    ("Test", "single locked eval", "only after freezing all choices"),
]
print("Remaining experiment matrix:")
for name, change, hold in checklist:
    print(f"- {name}: change [{change}]; hold [{hold}]")

print("\\nCLI smoke (no Jupyter):")
print("  PYTHONPATH=src python scripts/run_smoke.py")
print("Unit tests:")
print("  PYTHONPATH=src python -m pytest -q")
"""
        )
    )

    cells.append(
        md(
            """
## 8. Interpreting results (short)

| Observation | Likely meaning |
| --- | --- |
| Smoke F1 near 0 | Expected on tiny synthetic 1-epoch runs — pipeline check only |
| Benchmark finishes one epoch | Device/memory OK; not a quality claim |
| Full val F1 competitive with 0.73 | Comparable to the earlier U-Net feasibility run |
| Siamese helps F1 but is slower | Supports H1 quality claim with a compute trade-off |
| BCE+Dice lifts recall/F1 vs BCE | Supports H2 under class imbalance |

Keep every published table tied to a `summary.json` + seed + `run_name`.
"""
        )
    )

    write_nb(Path("/workspace/notebooks/02_experiments.ipynb"), cells)


if __name__ == "__main__":
    build_audit()
    build_experiments()
