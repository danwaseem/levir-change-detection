#!/usr/bin/env python3
"""Verify key report numbers against on-disk result JSON (no GPU / no retrain)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def approx(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    checks = []

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
    # abs-diff summary may use 'f1' or 'val_f1'
    abs_f1 = absdiff.get("f1", absdiff.get("val_f1"))
    checks.append(("absdiff f1", abs_f1, 0.129))

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

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
