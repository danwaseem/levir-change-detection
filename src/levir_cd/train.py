from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import (
    ChangeTiles,
    SyntheticChangeTiles,
    load_split_frames,
    make_loaders,
)
from .device import describe_runtime, pick_device, set_seed, sync_device
from .losses import segmentation_loss
from .metrics import evaluate_loader
from .metrics import failure_summary, per_image_scores, summarize_per_image
from .models import build_model, count_parameters, paper_name


def _loss_fn_for(cfg: ExperimentConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return segmentation_loss(logits, target, kind=cfg.LOSS)

    return _fn


def fit_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_every: int = 250,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch_id, (x, y) in enumerate(loader, 1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss at batch {batch_id}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)
        if batch_id == 1 or batch_id % log_every == 0 or batch_id == len(loader):
            print(f"  train batch {batch_id}/{len(loader)} | loss {loss.item():.4f}", flush=True)
    return total_loss / max(total_n, 1)


def build_datasets(cfg: ExperimentConfig):
    if cfg.RUN_MODE == "smoke":
        train_ds = SyntheticChangeTiles(
            n_tiles=cfg.SMOKE_TRAIN_TILES,
            size=cfg.TILE_SIZE,
            seed=cfg.SEED,
            augment=cfg.AUGMENT,
        )
        val_ds = SyntheticChangeTiles(
            n_tiles=cfg.SMOKE_VAL_TILES,
            size=cfg.TILE_SIZE,
            seed=cfg.SEED + 1,
            augment=False,
        )
        return train_ds, val_ds, {"train_pairs": 0, "val_pairs": 0, "synthetic": True}

    train_df, val_df = load_split_frames(cfg.manifest_path)
    train_ds = ChangeTiles(
        train_df,
        size=cfg.TILE_SIZE,
        image_size=cfg.IMAGE_SIZE,
        augment=cfg.AUGMENT,
    )
    val_ds = ChangeTiles(
        val_df,
        size=cfg.TILE_SIZE,
        image_size=cfg.IMAGE_SIZE,
        augment=False,
    )
    return train_ds, val_ds, {
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "synthetic": False,
    }


def run_experiment(cfg: ExperimentConfig, device: torch.device | None = None) -> dict[str, Any]:
    cfg.validate()
    set_seed(cfg.SEED)
    device = device or pick_device()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    print(describe_runtime(device))
    print("Run directory:", run_dir)

    train_ds, val_ds, meta = build_datasets(cfg)
    batch_size = cfg.effective_batch_size()
    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.MODEL, base_channels=cfg.BASE_CHANNELS).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    loss_fn = _loss_fn_for(cfg)
    n_params = count_parameters(model)
    print(f"Model={cfg.MODEL} params={n_params:,} loss={cfg.LOSS} augment={cfg.AUGMENT}")

    with torch.no_grad():
        probe = model(torch.zeros(1, 6, cfg.TILE_SIZE, cfg.TILE_SIZE, device=device))
    assert tuple(probe.shape) == (1, 1, cfg.TILE_SIZE, cfg.TILE_SIZE)

    epochs = cfg.effective_epochs()
    best_f1 = -1.0
    best_epoch = None
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    inference_seconds = None

    for epoch in range(1, epochs + 1):
        sync_device(device)
        tick = time.perf_counter()
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        train_loss = fit_one_epoch(model, train_loader, optimizer, device, loss_fn, log_every=50 if cfg.RUN_MODE == "smoke" else 250)
        metrics = evaluate_loader(model, val_loader, device, loss_fn, threshold=cfg.THRESHOLD)
        sync_device(device)
        elapsed = time.perf_counter() - tick
        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "elapsed_seconds": elapsed}
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        print(
            f"  loss {train_loss:.4f} / val {metrics['val_loss']:.4f} | "
            f"F1 {metrics['val_f1']:.4f} | IoU {metrics['val_iou']:.4f} | "
            f"P {metrics['val_precision']:.4f} | R {metrics['val_recall']:.4f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "metrics": row,
            "torch_version": str(torch.__version__),
        }
        torch.save(state, run_dir / "last_model.pt")
        if metrics["val_f1"] > best_f1 + cfg.MIN_F1_IMPROVEMENT:
            best_f1 = metrics["val_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(state, run_dir / "best_model.pt")
            print("  saved new best checkpoint", flush=True)
        else:
            bad_epochs += 1
        if cfg.RUN_MODE == "full" and epoch >= cfg.MIN_EPOCHS and bad_epochs >= cfg.EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

    # Batch-one inference latency after a short warm-up.
    model.eval()
    sample_x, _ = next(iter(val_loader))
    sample_x = sample_x[:1].to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(sample_x)
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = model(sample_x)
        sync_device(device)
        inference_seconds = (time.perf_counter() - t0) / 10.0

    # Per-image validation summary (proposal secondary error analysis).
    per_image = per_image_scores(model, val_loader, device, threshold=cfg.THRESHOLD)
    per_image_path = run_dir / "per_image_val_metrics.csv"
    per_image.to_csv(per_image_path, index=False)
    per_image_summary = summarize_per_image(per_image)
    failures = failure_summary(per_image)
    (run_dir / "per_image_val_summary.json").write_text(
        json.dumps(per_image_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "failure_summary.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )

    total_seconds = time.perf_counter() - started
    summary = {
        "run_mode": cfg.RUN_MODE,
        "model": cfg.MODEL,
        "paper_name": paper_name(cfg.MODEL),
        "loss": cfg.LOSS,
        "augment": cfg.AUGMENT,
        "seed": cfg.SEED,
        "completed_epochs": len(rows),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "total_minutes": round(total_seconds / 60, 4),
        "mean_epoch_seconds": round(sum(r["elapsed_seconds"] for r in rows) / max(len(rows), 1), 3),
        "batch_one_inference_seconds": round(inference_seconds, 5) if inference_seconds is not None else None,
        "train_original_pairs": meta["train_pairs"],
        "val_original_pairs": meta["val_pairs"],
        "train_tiles": len(train_ds),
        "val_tiles": len(val_ds),
        "parameters": n_params,
        "threshold": cfg.THRESHOLD,
        "selected_using": "validation F1 only",
        "test_set_touched": False,
        "synthetic": meta["synthetic"],
        "checkpoint_bytes": (run_dir / "best_model.pt").stat().st_size if (run_dir / "best_model.pt").exists() else None,
        "per_image_val_summary": per_image_summary,
        "failure_summary": failures,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def maybe_copy_summary_to_reports(summary: dict[str, Any], project_root: Path) -> Path | None:
    """Copy smoke/benchmark summaries under results/experiments for sharing."""
    if summary.get("run_mode") not in {"smoke", "benchmark"}:
        return None
    out = project_root / "results" / "experiments" / Path(summary["run_dir"]).name
    out.mkdir(parents=True, exist_ok=True)
    src = Path(summary["run_dir"])
    for name in (
        "summary.json",
        "metrics.csv",
        "config.json",
        "per_image_val_summary.json",
        "per_image_val_metrics.csv",
        "failure_summary.json",
    ):
        p = src / name
        if p.exists():
            (out / name).write_bytes(p.read_bytes())
    return out
