"""The proposal experiment matrix and a table built from saved summaries.

Training still happens in ``run_experiment``. This module only lists the runs
and turns finished ``summary.json`` files into one comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .models import paper_name

MAIN_SEEDS = (1511, 42, 123)
ABLATION_SEED = 1511


def _cfg(project_root: str, mode: str, model: str, loss: str, augment: bool, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        PROJECT_ROOT=project_root,
        RUN_MODE=mode,
        MODEL=model,
        LOSS=loss,
        AUGMENT=augment,
        SEED=seed,
    )
    if mode == "smoke":
        cfg.TILE_SIZE = 64
        cfg.BASE_CHANNELS = 8
        cfg.SMOKE_TRAIN_TILES = 4
        cfg.SMOKE_VAL_TILES = 2
        cfg.SMOKE_EPOCHS = 1
        cfg.SMOKE_BATCH_SIZE = 2
    cfg.validate()
    return cfg


def experiment_plan(project_root: str, mode: str = "full") -> list[dict[str, Any]]:
    """Runs required by the proposal, plus one FC-Siam-conc column.

    ``mode`` is ``smoke``, ``benchmark``, or ``full``. Smoke uses synthetic
    tiles and is only a code check. It is not a LEVIR result.
    """
    if mode not in {"smoke", "benchmark", "full"}:
        raise ValueError(f"Unknown mode: {mode}")
    rows: list[dict[str, Any]] = []

    def add(role: str, model: str, loss: str, augment: bool, seed: int) -> None:
        cfg = _cfg(project_root, mode, model, loss, augment, seed)
        rows.append({"role": role, "config": cfg})

    for seed in MAIN_SEEDS:
        add("main", "unet", "bce_dice", True, seed)
        add("main", "siamese_diff", "bce_dice", True, seed)
    add("triad", "siamese_conc", "bce_dice", True, ABLATION_SEED)
    add("loss", "unet", "bce", True, ABLATION_SEED)
    add("augmentation", "unet", "bce_dice", False, ABLATION_SEED)
    return rows


def load_summaries(results_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return summaries
    for path in sorted(results_root.rglob("summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "model" in payload:
            payload["_path"] = str(path)
            summaries.append(payload)
    return summaries


def comparison_markdown(summaries: list[dict[str, Any]]) -> str:
    """One table of finished runs. Empty when no summaries exist yet."""
    lines = [
        "# Comparison table",
        "",
        "Built from `summary.json` files. Validation F1 selects each checkpoint.",
        "Overall accuracy is not the selection metric. Synthetic smoke runs are not LEVIR scores.",
        "",
        "| Run | Paper name | Loss | Aug | Seed | Val F1 | IoU | P | R | Boundary F1 | Params | Infer (s) | Synthetic |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not summaries:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | no runs yet |")
        return "\n".join(lines) + "\n"
    for row in summaries:
        name = row.get("paper_name") or paper_name(str(row.get("model", "")))
        lines.append(
            "| {run} | {paper} | {loss} | {aug} | {seed} | {f1} | {iou} | {p} | {r} | {b} | {params} | {lat} | {syn} |".format(
                run=row.get("run_mode", ""),
                paper=name,
                loss=row.get("loss", ""),
                aug=row.get("augment", ""),
                seed=row.get("seed", ""),
                f1=_fmt(row.get("best_val_f1")),
                iou=_fmt(_nested(row, "val_iou")),
                p=_fmt(_nested(row, "val_precision")),
                r=_fmt(_nested(row, "val_recall")),
                b=_fmt(_nested(row, "val_boundary_f1")),
                params=row.get("parameters", ""),
                lat=_fmt(row.get("batch_one_inference_seconds")),
                syn=row.get("synthetic", ""),
            )
        )
    return "\n".join(lines) + "\n"


def write_comparison_table(results_root: Path, destination: Path | None = None) -> Path:
    destination = destination or (results_root / "comparison_table.md")
    text = comparison_markdown(load_summaries(results_root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def _nested(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    per_image = row.get("per_image_val_summary")
    if isinstance(per_image, dict) and key in per_image:
        return per_image[key]
    return ""
