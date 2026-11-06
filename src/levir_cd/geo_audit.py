from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def filename_overlap_report(manifest_path: Path) -> dict[str, Any]:
    """Report shared stems across splits.

    Matching filenames are a warning, not automatic geographic leakage.
    A fuller geospatial check needs external metadata (planned separately).
    """
    df = pd.read_csv(manifest_path)
    splits = sorted(df["split"].astype(str).unique())
    by_split = {s: set(df.loc[df.split == s, "stem"].astype(str)) for s in splits}
    overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            common = sorted(by_split[a] & by_split[b])
            overlaps[f"{a}_vs_{b}"] = {"count": len(common), "examples": common[:20]}
    report = {
        "splits": {s: len(by_split[s]) for s in splits},
        "stem_overlaps": overlaps,
        "note": (
            "Shared stems across splits are rare for LEVIR-CD. "
            "Absence of stem overlap does not prove absence of nearby/overlapping scenes."
        ),
    }
    return report


def write_geo_audit(manifest_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = filename_overlap_report(manifest_path)
    out = output_dir / "stem_overlap_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
