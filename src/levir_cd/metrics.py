from __future__ import annotations

from typing import Any

import pandas as pd
import torch


def binary_counts(pred: torch.Tensor, truth: torch.Tensor) -> tuple[int, int, int]:
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    return tp, fp, fn


def scores_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, float]:
    """Change-class scores as in Chen & Shi (LEVIR-CD): Pr / Re / F1 (+ IoU).

    Overall accuracy is recorded only as a secondary diagnostic. Unchanged pixels
    dominate LEVIR-CD, so accuracy is not used to pick a checkpoint.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def boundary_mask(mask: torch.Tensor) -> torch.Tensor:
    """One-pixel-wide boundary of a boolean mask using a 3x3 dilation minus erosion."""
    value = mask.float()
    dilated = torch.nn.functional.max_pool2d(value, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - value, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded) > 0.5


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Pooled pixel metrics across the loader (primary protocol)."""
    model.eval()
    tp = fp = fn = tn = 0
    b_tp = b_fp = b_fn = 0
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)
        pred = torch.sigmoid(logits) >= threshold
        truth = y >= 0.5
        tpi, fpi, fni = binary_counts(pred, truth)
        pixels = int(pred.numel())
        tp += tpi
        fp += fpi
        fn += fni
        tn += pixels - tpi - fpi - fni
        pred_b = boundary_mask(pred)
        truth_b = boundary_mask(truth)
        bt, bf, bn = binary_counts(pred_b, truth_b)
        b_tp += bt
        b_fp += bf
        b_fn += bn
    metrics = scores_from_counts(tp, fp, fn, tn)
    boundary = scores_from_counts(b_tp, b_fp, b_fn)
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "val_iou": metrics["iou"],
        "val_accuracy": metrics["accuracy"],
        "val_boundary_f1": boundary["f1"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def per_image_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-sample change-class scores for failure analysis (proposal secondary summaries)."""
    model.eval()
    rows: list[dict[str, Any]] = []
    index = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = probs >= threshold
        truth = y >= 0.5
        for i in range(x.size(0)):
            tp, fp, fn = binary_counts(pred[i], truth[i])
            scores = scores_from_counts(tp, fp, fn)
            rows.append(
                {
                    "sample_index": index,
                    "changed_pixels": int(truth[i].sum().item()),
                    "pred_changed_pixels": int(pred[i].sum().item()),
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    "iou": scores["iou"],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def failure_summary(frame: pd.DataFrame, small_max_pixels: int = 256) -> dict[str, Any]:
    """Group validation tiles the way the proposal describes failures.

    Small-change tiles have at most ``small_max_pixels`` changed pixels
    (about a 16x16 building on a 256 tile). LEVIR buildings are often small
    relative to the scene (Chen & Shi report ~987 pixels on the full 1024 image).
    """
    if frame.empty:
        return {"n": 0, "small_change_max_pixels": small_max_pixels}
    changed = frame.loc[frame["changed_pixels"] > 0]
    small = changed.loc[changed["changed_pixels"] <= small_max_pixels]
    large = changed.loc[changed["changed_pixels"] > small_max_pixels]
    empty = frame.loc[frame["changed_pixels"] == 0]

    def _mean(series: pd.Series) -> float | None:
        if series.empty:
            return None
        return float(series.mean())

    return {
        "n": int(len(frame)),
        "small_change_max_pixels": int(small_max_pixels),
        "n_small_change_tiles": int(len(small)),
        "n_large_change_tiles": int(len(large)),
        "n_empty_tiles": int(len(empty)),
        "small_change_mean_recall": _mean(small["recall"]) if len(small) else None,
        "large_change_mean_recall": _mean(large["recall"]) if len(large) else None,
        "n_missed_small": int((small["recall"] < 0.5).sum()) if len(small) else 0,
        "n_false_alarm_empty": int((empty["pred_changed_pixels"] > 0).sum()) if len(empty) else 0,
        "n_low_precision_change": int((changed["precision"] < 0.5).sum()) if len(changed) else 0,
    }


def summarize_per_image(frame: pd.DataFrame) -> dict[str, Any]:
    """Compact distribution summary used in reports / error analysis."""
    if frame.empty:
        return {"n": 0}
    return {
        "n": int(len(frame)),
        "mean_f1": float(frame["f1"].mean()),
        "std_f1": float(frame["f1"].std(ddof=0)),
        "mean_iou": float(frame["iou"].mean()),
        "fraction_zero_f1": float((frame["f1"] == 0).mean()),
        "fraction_with_change": float((frame["changed_pixels"] > 0).mean()),
        "worst_f1_sample_index": int(frame.loc[frame["f1"].idxmin(), "sample_index"]),
        "best_f1_sample_index": int(frame.loc[frame["f1"].idxmax(), "sample_index"]),
    }
