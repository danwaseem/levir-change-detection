from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import scores_from_counts


@torch.no_grad()
def absolute_difference_baseline(
    loader: DataLoader,
    threshold: float | None = None,
    threshold_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Non-learned RGB absolute-difference control.

    If threshold is None, choose the best F1 on the provided loader using
    threshold_grid (validation-only selection).
    """
    grid = threshold_grid or [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    # Accumulate soft difference maps once.
    diffs: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for x, y in loader:
        # x: N,6,H,W in [0,1]
        a = x[:, :3].numpy()
        b = x[:, 3:].numpy()
        diff = np.abs(a - b).mean(axis=1)  # N,H,W
        diffs.append(diff)
        truths.append((y[:, 0].numpy() >= 0.5))
    diff_all = np.concatenate([d.reshape(-1) for d in diffs])
    truth_all = np.concatenate([t.reshape(-1) for t in truths])

    def eval_at(th: float) -> dict[str, float]:
        pred = diff_all >= th
        tp = int(np.logical_and(pred, truth_all).sum())
        fp = int(np.logical_and(pred, ~truth_all).sum())
        fn = int(np.logical_and(~pred, truth_all).sum())
        return scores_from_counts(tp, fp, fn)

    if threshold is None:
        scored = [(th, eval_at(th)) for th in grid]
        best_th, best = max(scored, key=lambda item: item[1]["f1"])
        return {"threshold": best_th, "selected_on": "validation", **best}
    metrics = eval_at(float(threshold))
    return {"threshold": float(threshold), "selected_on": "fixed", **metrics}
