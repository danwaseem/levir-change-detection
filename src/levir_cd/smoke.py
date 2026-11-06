from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig, default_project_root
from .device import pick_device
from .models import build_model, count_parameters, paper_name
from .train import maybe_copy_summary_to_reports, run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test LEVIR change-detection models without the real dataset.",
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--model",
        choices=["unet", "siamese", "siamese_diff", "siamese_conc", "all"],
        default="all",
        help="Daudt triad: unet=FC-EF, siamese/siamese_diff=FC-Siam-diff, siamese_conc=FC-Siam-conc",
    )
    parser.add_argument("--seed", type=int, default=1511)
    parser.add_argument("--tile-size", type=int, default=64, help="Use 64 for fast CPU smoke; 256 for fidelity.")
    parser.add_argument("--train-tiles", type=int, default=4)
    parser.add_argument("--val-tiles", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=8)
    return parser.parse_args(argv)


def _model_list(choice: str) -> list[str]:
    if choice == "all":
        return ["unet", "siamese_diff", "siamese_conc"]
    if choice == "siamese":
        return ["siamese_diff"]
    return [choice]


def run_smoke(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).resolve() if args.project_root else default_project_root()
    device = pick_device()
    models = _model_list(args.model)
    summaries = {}

    for name in models:
        net = build_model(name, base_channels=args.base_channels).to(device)
        with torch.no_grad():
            y = net(torch.zeros(1, 6, args.tile_size, args.tile_size, device=device))
        assert tuple(y.shape) == (1, 1, args.tile_size, args.tile_size)
        print(
            f"Forward OK: {name} ({paper_name(name)}) "
            f"params={count_parameters(net):,} shape={tuple(y.shape)}"
        )

    for name in models:
        cfg = ExperimentConfig(
            PROJECT_ROOT=str(project_root),
            RUN_MODE="smoke",
            MODEL=name,
            LOSS="bce_dice",
            AUGMENT=True,
            SEED=args.seed,
            TILE_SIZE=args.tile_size,
            IMAGE_SIZE=args.tile_size,
            BASE_CHANNELS=args.base_channels,
            SMOKE_TRAIN_TILES=args.train_tiles,
            SMOKE_VAL_TILES=args.val_tiles,
            SMOKE_EPOCHS=args.epochs,
            SMOKE_BATCH_SIZE=args.batch_size,
        )
        summary = run_experiment(cfg, device=device)
        summary["paper_name"] = paper_name(name)
        maybe_copy_summary_to_reports(summary, project_root)
        summaries[name] = summary

    out = project_root / "results" / "smoke_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("Wrote", out)
    return summaries


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
