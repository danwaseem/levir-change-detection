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

from levir_cd.plan import experiment_plan, write_comparison_table
from levir_cd.train import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or summarize the LEVIR experiment plan.")
    parser.add_argument("--mode", choices=["smoke", "benchmark", "full"], default="full")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if not args.aggregate_only:
        for item in experiment_plan(str(project_root), mode=args.mode):
            print(f"\n=== {item['role']} {item['config'].run_name} ===", flush=True)
            run_experiment(item["config"])
    table = write_comparison_table(project_root / "results")
    print("Wrote", table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
