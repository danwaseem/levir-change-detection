#!/usr/bin/env python3
"""CLI entry for smoke tests. Run from repo root:

    PYTHONPATH=src python scripts/run_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from levir_cd.smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
