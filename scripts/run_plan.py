#!/usr/bin/env python3
"""Run the proposal matrix, then refresh results/comparison_table.md.

    PYTHONPATH=src python scripts/run_plan.py --mode smoke
    PYTHONPATH=src python scripts/run_plan.py --mode full
    PYTHONPATH=src python scripts/run_plan.py --aggregate-only

``smoke`` does not use LEVIR and must not be reported as a result.
``full`` and ``benchmark`` need the audit manifest from notebook 01.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from levir_cd.audit import ensure_manifest
from levir_cd.geo_audit import write_geo_audit
from levir_cd.plan import experiment_plan, write_comparison_table
from levir_cd.train import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or summarize the LEVIR experiment plan.")
    parser.add_argument("--mode", choices=["smoke", "benchmark", "full"], default="full")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if not args.aggregate_only and args.mode in {"benchmark", "full"}:
        manifest = ensure_manifest(project_root)
        write_geo_audit(manifest, project_root / "results" / "levir_audit")
    if not args.aggregate_only:
        for item in experiment_plan(str(project_root), mode=args.mode):
            print(f"\n=== {item['role']} {item['config'].run_name} ===", flush=True)
            cfg = item["config"]
            summary_path = cfg.run_dir / "summary.json"
            config_path = cfg.run_dir / "config.json"
            checkpoint_path = cfg.run_dir / "best_model.pt"

            if cfg.RUN_MODE == "full" and summary_path.exists():
                import json

                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )

                if not config_path.exists() or not checkpoint_path.exists():
                    raise RuntimeError(
                        f"Incomplete saved run: {cfg.run_dir}"
                    )

                saved_config = json.loads(
                    config_path.read_text(encoding="utf-8")
                )

                if saved_config != cfg.to_dict():
                    raise RuntimeError(
                        f"Completed run configuration mismatch: {cfg.run_dir}"
                    )

                if (summary.get("run_mode") != "full"
                        or summary.get("synthetic") is not False
                        or summary.get("completed_epochs", 0) < 1):
                    raise RuntimeError(
                        f"Invalid completed-run summary: {cfg.run_dir}"
                    )

                print(
                    f"SKIPPING completed experiment: {cfg.run_name}",
                    flush=True,
                )
                continue

            run_experiment(cfg)
    table = write_comparison_table(project_root / "results")
    print("Wrote", table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
