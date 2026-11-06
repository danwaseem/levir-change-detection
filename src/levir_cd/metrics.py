from __future__ import annotations

from typing import Any

import pandas as pd
import torch


def binary_counts(pred: torch.Tensor, truth: torch.Tensor) -> tuple[int, int, int]:
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    return tp, fp, fn


def scores_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Change-class scores as in Chen & Shi (LEVIR-CD): Pr / Re / F1 (+ IoU)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


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
    tp = fp = fn = 0
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
        tp += tpi
        fp += fpi
        fn += fni
    metrics = scores_from_counts(tp, fp, fn)
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "val_iou": metrics["iou"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
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
