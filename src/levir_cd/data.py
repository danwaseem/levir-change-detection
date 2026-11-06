from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ChangeTiles(Dataset):
    """Non-overlapping paired tiles from original 1024x1024 LEVIR pairs."""

    def __init__(
        self,
        frame: pd.DataFrame,
        size: int = 256,
        image_size: int = 1024,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.size = int(size)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        if self.image_size % self.size != 0:
            raise ValueError("image_size must be divisible by tile size")
        self.tiles: list[tuple[int, int, int]] = []
        for row_i, _ in self.frame.iterrows():
            for y in range(0, self.image_size, self.size):
                for x in range(0, self.image_size, self.size):
                    self.tiles.append((int(row_i), x, y))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_i, x, y = self.tiles[idx]
        row = self.frame.iloc[row_i]
        rect = (x, y, x + self.size, y + self.size)
        with Image.open(row.image_A) as source:
            if source.size != (self.image_size, self.image_size):
                raise ValueError(f"Bad A size: {row.image_A}")
            a = np.array(source.convert("RGB").crop(rect), dtype=np.uint8)
        with Image.open(row.image_B) as source:
            if source.size != (self.image_size, self.image_size):
                raise ValueError(f"Bad B size: {row.image_B}")
            b = np.array(source.convert("RGB").crop(rect), dtype=np.uint8)
        with Image.open(row["mask"]) as source:
            if source.size != (self.image_size, self.image_size):
                raise ValueError(f"Bad mask size: {row['mask']}")
            mask = np.asarray(source.convert("L").crop(rect)) >= 128

        if self.augment:
            if random.random() < 0.5:
                a, b, mask = a[:, ::-1], b[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                a, b, mask = a[::-1], b[::-1], mask[::-1]
            k = random.randrange(4)
            if k:
                a, b, mask = np.rot90(a, k), np.rot90(b, k), np.rot90(mask, k)

        x_arr = np.concatenate([a, b], axis=2).transpose(2, 0, 1).copy()
        y_arr = mask.astype(np.float32, copy=False)[None].copy()
        x_tensor = torch.from_numpy(x_arr).float().div_(255.0)
        y_tensor = torch.from_numpy(y_arr)
        return x_tensor, y_tensor


class SyntheticChangeTiles(Dataset):
    """Tiny in-memory tiles for smoke tests without LEVIR-CD."""

    def __init__(self, n_tiles: int, size: int = 256, seed: int = 0, augment: bool = False) -> None:
        self.n_tiles = int(n_tiles)
        self.size = int(size)
        self.augment = bool(augment)
        rng = np.random.default_rng(seed)
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(self.n_tiles):
            a = rng.integers(40, 180, size=(self.size, self.size, 3), dtype=np.uint8)
            b = a.copy()
            mask = np.zeros((self.size, self.size), dtype=bool)
            if rng.random() < 0.75:
                margin = max(1, self.size // 8)
                max_origin = max(margin + 1, self.size - margin - 1)
                y0 = int(rng.integers(margin, max_origin))
                x0 = int(rng.integers(margin, max_origin))
                max_h = max(2, self.size - y0 - 1)
                max_w = max(2, self.size - x0 - 1)
                h = int(rng.integers(1, max_h))
                w = int(rng.integers(1, max_w))
                y1, x1 = min(self.size, y0 + h), min(self.size, x0 + w)
                color = rng.integers(160, 255, size=(3,), dtype=np.uint8)
                if rng.random() < 0.5:
                    # Construction: building appears in B.
                    b[y0:y1, x0:x1] = color
                else:
                    # Demolition: building disappears from A.
                    a[y0:y1, x0:x1] = color
                mask[y0:y1, x0:x1] = True
            self.samples.append((a, b, mask))

    def __len__(self) -> int:
        return self.n_tiles

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, mask = self.samples[idx]
        a, b, mask = a.copy(), b.copy(), mask.copy()
        if self.augment:
            if random.random() < 0.5:
                a, b, mask = a[:, ::-1], b[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                a, b, mask = a[::-1], b[::-1], mask[::-1]
            k = random.randrange(4)
            if k:
                a, b, mask = np.rot90(a, k), np.rot90(b, k), np.rot90(mask, k)
        x_arr = np.concatenate([a, b], axis=2).transpose(2, 0, 1).copy()
        y_arr = mask.astype(np.float32)[None].copy()
        return torch.from_numpy(x_arr).float().div_(255.0), torch.from_numpy(y_arr)


def _validate_split_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"Manifest has no rows for split={split_name!r}")
    if not frame.stem.is_unique:
        raise ValueError(f"Duplicate stems inside split={split_name!r}")
    for key in ("image_A", "image_B", "mask"):
        absent = [p for p in frame[key] if not Path(p).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {key} files, e.g. {absent[:3]}")
    return frame.reset_index(drop=True)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run notebook 01, or run "
            "`PYTHONPATH=src python scripts/run_plan.py --mode full` after LEVIR-CD is on disk."
        )
    manifest = pd.read_csv(manifest_path)
    required = {"split", "stem", "image_A", "image_B", "mask"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return manifest


def load_split_frame(manifest_path: Path, split: str) -> pd.DataFrame:
    """Load one split (train/val/test). Training code should not call this for test."""
    key = str(split).lower().strip()
    if key not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    manifest = load_manifest(manifest_path)
    frame = manifest.loc[manifest.split == key].sort_values("stem")
    return _validate_split_frame(frame, key)


def load_split_frames(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val only. Test rows may exist in the CSV but are ignored here."""
    train_df = load_split_frame(manifest_path, "train")
    val_df = load_split_frame(manifest_path, "val")
    if not set(train_df.stem).isdisjoint(set(val_df.stem)):
        raise ValueError("Train/val stems overlap")
    return train_df, val_df


def make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
