#!/usr/bin/env python3
"""One-shot locked test evaluation. Do NOT run until models are frozen.

    PYTHONPATH=src python scripts/run_locked_test.py \\
      --checkpoint results/experiments/<run>/best_model.pt \\
      --i-confirm-choices-are-frozen

Writes results/locked_test/<run_name>/summary.json. Refuses without the
confirmation flag. Threshold stays 0.5. No retuning after seeing test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from levir_cd.config import default_project_root
from levir_cd.data import ChangeTiles, load_split_frame, make_loaders
from levir_cd.device import pick_device
from levir_cd.losses import segmentation_loss
from levir_cd.metrics import evaluate_loader, failure_summary, per_image_scores, summarize_per_image
from levir_cd.models import build_model, count_parameters, paper_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locked one-shot LEVIR test evaluation.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument(
        "--i-confirm-choices-are-frozen",
        action="store_true",
        help="Required. Affirms validation-based selection is frozen.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    if not args.i_confirm_choices_are_frozen:
        print(
            "Refusing to touch the test set. Re-run with "
            "--i-confirm-choices-are-frozen after freezing model/loss/aug/threshold.",
            file=sys.stderr,
        )
        return 2

    project_root = Path(args.project_root).resolve() if args.project_root else default_project_root()
    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("config") or {}
    # Locked test path is intentional; ExperimentConfig.validate() forbids ALLOW_TEST.
    model_name = str(cfg_dict.get("MODEL", "unet"))
    threshold = float(cfg_dict.get("THRESHOLD", 0.5))
    tile_size = int(cfg_dict.get("TILE_SIZE", 256))
    image_size = int(cfg_dict.get("IMAGE_SIZE", 1024))
    base = int(cfg_dict.get("BASE_CHANNELS", 16))
    loss_kind = str(cfg_dict.get("LOSS", "bce_dice"))

    device = pick_device()
    model = build_model(model_name, base_channels=base).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    test_df = load_split_frame(project_root / "results" / "levir_audit" / "split_manifest.csv", "test")
    test_ds = ChangeTiles(test_df, size=tile_size, image_size=image_size, augment=False)
    _, test_loader = make_loaders(test_ds, test_ds, batch_size=args.batch_size)

    def loss_fn(logits, target):
        return segmentation_loss(logits, target, kind=loss_kind)

    metrics = evaluate_loader(model, test_loader, device, loss_fn, threshold=threshold)
    per_image = per_image_scores(model, test_loader, device, threshold=threshold)
    run_name = ckpt_path.parent.name
    out_dir = project_root / "results" / "locked_test" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out_dir / "per_image_test_metrics.csv", index=False)

    summary = {
        "split": "test",
        "checkpoint": str(ckpt_path),
        "model": model_name,
        "paper_name": paper_name(model_name),
        "loss": loss_kind,
        "threshold": threshold,
        "official_threshold": 0.5,
        "parameters": count_parameters(model),
        "n_pairs": len(test_df),
        "n_tiles": len(test_ds),
        "test_f1": metrics["val_f1"],
        "test_iou": metrics["val_iou"],
        "test_precision": metrics["val_precision"],
        "test_recall": metrics["val_recall"],
        "test_boundary_f1": metrics["val_boundary_f1"],
        "test_accuracy": metrics["val_accuracy"],
        "per_image_test_summary": summarize_per_image(per_image),
        "failure_summary": failure_summary(per_image),
        "test_set_touched": True,
        "retuned_after_test": False,
        "note": "One-shot locked evaluation. Do not change models after this.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
