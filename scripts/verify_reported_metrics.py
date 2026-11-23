#!/usr/bin/env python3
"""Verify key report numbers against on-disk result JSON (no GPU / no retrain)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def approx(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def check_experiment_summaries() -> list[tuple[str, bool, str]]:
    """Catch the last-epoch vs best-checkpoint secondary-artifact mismatch."""
    rows: list[tuple[str, bool, str]] = []
    for summary_path in sorted((ROOT / "results" / "experiments").glob("full_*/summary.json")):
        summary = load(summary_path)
        metrics_path = summary_path.parent / "metrics.csv"
        if not metrics_path.is_file():
            rows.append((summary_path.parent.name, False, "missing metrics.csv"))
            continue
        metrics = pd.read_csv(metrics_path)
        best_row = metrics.loc[metrics["val_f1"].idxmax()]
        last_row = metrics.iloc[-1]
        name = summary_path.parent.name

        ok_best = approx(float(summary["best_val_f1"]), float(best_row["val_f1"]), tol=1e-6)
        rows.append(
            (
                f"{name} best_val_f1 vs metrics.csv",
                ok_best,
                f"summary={summary['best_val_f1']} csv={best_row['val_f1']}",
            )
        )

        sweep = summary.get("threshold_sweep") or []
        sw05 = next((x for x in sweep if abs(float(x.get("threshold", -1)) - 0.5) < 1e-9), None)
        if sw05 is not None:
            matches_last = approx(float(sw05["f1"]), float(last_row["val_f1"]), tol=1e-4)
            matches_best = approx(float(sw05["f1"]), float(summary["best_val_f1"]), tol=1e-4)
            source = summary.get("secondary_artifacts_source")
            # Old runs: sweep matches last epoch, not best. Must be labeled.
            if matches_last and not matches_best:
                labeled = source == "last_epoch_weights"
                rows.append(
                    (
                        f"{name} threshold_sweep@0.5 is last-epoch",
                        labeled,
                        f"sweep={sw05['f1']} last={last_row['val_f1']} source={source}",
                    )
                )
            elif matches_best:
                rows.append(
                    (
                        f"{name} threshold_sweep@0.5 matches best",
                        True,
                        f"sweep={sw05['f1']}",
                    )
                )
    return rows


def main() -> int:
    checks: list[tuple[str, object, object]] = []

    locked = load(
        ROOT
        / "results/locked_test/full_unet_bce_aug_seed1511_tile256/summary.json"
    )
    checks.append(("locked test_f1", locked["test_f1"], 0.780))
    checks.append(("locked test_iou", locked["test_iou"], 0.639))
    fail = locked["failure_summary"]
    checks.append(("empty tiles", fail["n_empty_tiles"], 1113))
    checks.append(("small tiles", fail["n_small_change_tiles"], 61))
    checks.append(("missed small", fail["n_missed_small"], 56))
    checks.append(("false-alarm empty", fail["n_false_alarm_empty"], 212))

    absdiff = load(ROOT / "results/absdiff_baseline/summary.json")
    abs_f1 = absdiff.get("f1", absdiff.get("val_f1"))
    checks.append(("absdiff f1", abs_f1, 0.129))
    # Abs-diff accuracy must not collapse onto IoU (needs real tn).
    checks.append(("absdiff tn > 0", absdiff.get("tn", 0) > 0, True))
    checks.append(
        (
            "absdiff accuracy != iou",
            abs(float(absdiff["accuracy"]) - float(absdiff["iou"])) > 1e-3,
            True,
        )
    )

    strata_path = (
        ROOT
        / "results/locked_test/full_unet_bce_aug_seed1511_tile256/error_strata.json"
    )
    if strata_path.exists():
        strata = load(strata_path)
        ci = strata["bootstrap_pooled_f1"]["ci95"]
        checks.append(("bootstrap ci lo", ci[0], 0.7699))
        checks.append(("bootstrap ci hi", ci[1], 0.7895))
        checks.append(
            ("large pooled f1", strata["strata"]["large_gt_256"]["pooled_f1"], 0.7904)
        )

    ok = True
    for name, got, exp in checks:
        if isinstance(got, (int, float)) and isinstance(exp, float):
            good = approx(float(got), float(exp))
        else:
            good = got == exp
        status = "OK" if good else "FAIL"
        if not good:
            ok = False
        print(f"{status:4} {name}: got={got} expected={exp}")

    for name, good, detail in check_experiment_summaries():
        status = "OK" if good else "FAIL"
        if not good:
            ok = False
        print(f"{status:4} {name}: {detail}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
