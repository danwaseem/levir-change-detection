"""Build results/levir_audit/split_manifest.csv from a local LEVIR-CD folder."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def candidate_data_roots(project_root: Path) -> list[Path]:
    """Places we look for train/val images before giving up."""
    env = os.environ.get("LEVIR_DATA_ROOT")
    roots = []
    if env:
        roots.append(Path(env).expanduser())
    home = Path.home()
    roots.extend(
        [
            project_root.parent / "datasets" / "levir-cd",
            project_root / "datasets" / "levir-cd",
            home / "datasets" / "levir-cd",
            home / "Desktop" / "datasets" / "levir-cd",
            home / "SJSU" / "data255" / "datasets" / "levir-cd",
            home / "data255" / "datasets" / "levir-cd",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_data_root(project_root: Path) -> Path:
    for root in candidate_data_roots(project_root):
        if root.is_dir() and _looks_like_levir(root):
            return root
    checked = "\n".join(f"  - {path}" for path in candidate_data_roots(project_root))
    raise FileNotFoundError(
        "LEVIR-CD was not found, so results/levir_audit/split_manifest.csv cannot be built.\n"
        "Put train.zip, val.zip, and test/ in one of these folders, or set LEVIR_DATA_ROOT:\n"
        f"{checked}"
    )


def ensure_manifest(project_root: Path) -> Path:
    """Write the train/val/test manifest. Reuses it when the file already exists."""
    project_root = project_root.expanduser().resolve()
    output = project_root / "results" / "levir_audit" / "split_manifest.csv"
    if output.is_file():
        print("Using existing manifest:", output)
        return output
    data_root = find_data_root(project_root)
    print("Dataset folder:", data_root)
    _extract_train_val(data_root)
    records = _pair_records(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print("Saved manifest:", output, f"({len(records)} rows)")
    return output


def _looks_like_levir(root: Path) -> bool:
    if _locate_split(root, "train") or _locate_split(root, "val"):
        return True
    names = {path.name.lower() for path in root.glob("*.zip")}
    return "train.zip" in names or "val.zip" in names


def _valid_split_dir(path: Path) -> bool:
    return path.is_dir() and all((path / part).is_dir() for part in ("A", "B", "label"))


def _locate_split(data_root: Path, split: str) -> Path | None:
    if not data_root.is_dir():
        return None
    candidates = [
        path
        for path in data_root.rglob("*")
        if path.is_dir() and path.name == split and _valid_split_dir(path)
    ]
    extracted = data_root / "_extracted" / split
    if _valid_split_dir(extracted):
        candidates.append(extracted)
    if extracted.is_dir():
        candidates.extend(path for path in extracted.rglob("*") if _valid_split_dir(path))
    if not candidates:
        return None
    candidates.sort(key=lambda path: (0 if "_extracted" in path.parts else 1, len(path.parts)))
    return candidates[0]


def _safe_extract(zpath: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zpath) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink not allowed: {member.filename}")
        archive.extractall(destination)


def _extract_train_val(data_root: Path) -> None:
    extracted = data_root / "_extracted"
    for split in ("train", "val"):
        zpath = data_root / f"{split}.zip"
        if not zpath.is_file():
            matches = [path for path in data_root.rglob(f"{split}.zip") if "_extracted" not in path.parts]
            zpath = matches[0] if matches else None
        out = extracted / split
        marker = out / ".extraction_complete"
        if zpath is None or not Path(zpath).is_file():
            continue
        if marker.exists():
            continue
        if out.exists() and any(out.iterdir()):
            shutil.rmtree(out)
        print(f"Extracting {zpath.name}...")
        _safe_extract(Path(zpath), out)
        marker.write_text(f"source={zpath}\n", encoding="utf-8")


def _files_by_stem(folder: Path) -> dict[str, Path]:
    paths = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate file stems in {folder}")
    return {path.stem: path for path in paths}


def _pair_records(data_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    found = {split: _locate_split(data_root, split) for split in ("train", "val", "test")}
    for split, folder in found.items():
        print(f"{split:5s}: {folder if folder else 'NOT FOUND'}")
    if found["train"] is None or found["val"] is None:
        raise FileNotFoundError(f"train/ and val/ with A, B, and label were not found under {data_root}")
    for split, folder in found.items():
        if folder is None:
            continue
        groups = [_files_by_stem(folder / name) for name in ("A", "B", "label")]
        names = sorted(set(groups[0]) & set(groups[1]) & set(groups[2]))
        unmatched = (set(groups[0]) | set(groups[1]) | set(groups[2])) - set(names)
        if unmatched:
            raise ValueError(f"Mismatched filenames in {split}: {sorted(unmatched)[:10]}")
        for name in names:
            records.append(
                {
                    "split": split,
                    "stem": name,
                    "image_A": str(groups[0][name]),
                    "image_B": str(groups[1][name]),
                    "mask": str(groups[2][name]),
                }
            )
    return records
