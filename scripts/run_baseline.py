#!/usr/bin/env python3
"""Score the absolute-difference reference baseline on validation tiles.

    PYTHONPATH=src python scripts/run_baseline.py

Uses the audit manifest. Does not touch the test split. Safe to run after the
main matrix, or on CPU while training finishes on MPS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from levir_cd.audit import ensure_manifest
from levir_cd.baseline import absolute_difference_baseline
from levir_cd.config import ExperimentConfig, default_project_root
from levir_cd.data import ChangeTiles, SyntheticChangeTiles, load_split_frame, make_loaders
from levir_cd.geo_audit import write_geo_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the absolute-difference baseline on validation.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve() if args.project_root else default_project_root()
    out_dir = project_root / "results" / "absdiff_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        ds = SyntheticChangeTiles(n_tiles=8, size=64, seed=0)
        loader, _ = make_loaders(ds, ds, batch_size=2)
        result = absolute_difference_baseline(loader)
        result.update({"split": "synthetic", "mode": "smoke", "n_tiles": len(ds)})
    else:
        manifest = ensure_manifest(project_root)
        write_geo_audit(manifest, project_root / "results" / "levir_audit")
        cfg = ExperimentConfig(PROJECT_ROOT=str(project_root), RUN_MODE="full")
        val_df = load_split_frame(cfg.manifest_path, "val")
        val_ds = ChangeTiles(val_df, size=cfg.TILE_SIZE, image_size=cfg.IMAGE_SIZE, augment=False)
        loader, _ = make_loaders(val_ds, val_ds, batch_size=args.batch_size)
        # absolute_difference_baseline only needs one loader; reuse val for both slots.
        result = absolute_difference_baseline(loader)
        result.update(
            {
                "split": "val",
                "mode": "full",
                "n_pairs": len(val_df),
                "n_tiles": len(val_ds),
                "selected_using": "validation F1 threshold sweep",
                "test_set_touched": False,
            }
        )

    path = out_dir / "summary.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
