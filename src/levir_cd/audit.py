"""Build results/levir_audit/split_manifest.csv from a local LEVIR-CD folder."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Public mirror of the official train/val/test archives (torchgeo LEVIRCD checksums).
# Academic use only. This is not LEVIR-CD+.
_HF = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6"
LEVIR_ZIPS = {
    "train": {"url": f"{_HF}/train.zip", "md5": "a638e71f480628652dea78d8544307e4"},
    "val": {"url": f"{_HF}/val.zip", "md5": "f7b857978524f9aa8c3bf7f94e3047a4"},
    "test": {"url": f"{_HF}/test.zip", "md5": "07d5dd89e46f5c1359e2eca746989ed9"},
}


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


def locate_dataset(project_root: Path, extra_roots: list[Path] | None = None) -> Path | None:
    """Return a folder that already has train and val, or None."""
    for root in candidate_data_roots(project_root):
        if root.is_dir() and _looks_like_levir(root):
            return root
    home = Path.home()
    search_roots = [
        project_root,
        project_root.parent,
        home / "Desktop",
        home / "Downloads",
        home / "Documents",
        home / "datasets",
    ]
    if extra_roots:
        search_roots.extend(extra_roots)
    seen: set[Path] = set()
    for root in search_roots:
        found = _search_tree(root, seen)
        if found is not None:
            return found
    return None


def download_levir(dest: Path, splits: dict | None = None) -> Path:
    """Download train.zip, val.zip, and test.zip. Skip a file whose checksum already matches."""
    splits = LEVIR_ZIPS if splits is None else splits
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    need = 5 * 1024**3
    if free < need:
        raise OSError(
            f"Need about 5 GB free to download and unpack LEVIR-CD. {dest} has {free / 1024**3:.1f} GB free."
        )
    print("LEVIR-CD is not on this machine. Downloading train, val, and test once (~2.3 GB) into", dest)
    print("Academic use only. The original images stay zipped; extraction is a separate folder.")
    for split, meta in splits.items():
        target = dest / f"{split}.zip"
        if target.is_file() and _md5(target) == meta["md5"]:
            print(f"{split}.zip already present")
            continue
        partial = dest / f"{split}.zip.partial"
        _download(meta["url"], partial)
        digest = _md5(partial)
        if digest != meta["md5"]:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{split}.zip checksum mismatch after download ({digest})")
        partial.replace(target)
        print(f"Saved {target}")
    return dest


def ensure_manifest(project_root: Path) -> Path:
    """Write the train/val/test manifest. Reuses it when the file already exists."""
    project_root = project_root.expanduser().resolve()
    output = project_root / "results" / "levir_audit" / "split_manifest.csv"
    if output.is_file():
        print("Using existing manifest:", output)
        return output
    data_root = locate_dataset(project_root)
    if data_root is None:
        env = os.environ.get("LEVIR_DATA_ROOT")
        dest = Path(env).expanduser() if env else project_root.parent / "datasets" / "levir-cd"
        data_root = download_levir(dest)
    print("Dataset folder:", data_root)
    _extract_archives(data_root)
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


def _search_tree(root: Path, seen: set[Path]) -> Path | None:
    if not root.is_dir():
        return None
    root = root.resolve()
    if root in seen:
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name not in {"Library", "node_modules", "Applications", ".git"}
        ]
        if depth >= 3:
            dirnames.clear()
        if current in seen:
            continue
        seen.add(current)
        names = {name.lower() for name in filenames}
        if "train.zip" in names and "val.zip" in names:
            return current
        if current.name in {"train", "val"} and _valid_split_dir(current):
            parent = current.parent
            if _valid_split_dir(parent / "train") and _valid_split_dir(parent / "val"):
                return parent
    return None


def _download(url: str, dest: Path) -> None:
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "levir-cd-course/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response_cm = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and dest.exists():
            print(f"  {dest.name}: already downloaded", flush=True)
            return
        raise
    with response_cm as response:
        if response.status == 200:
            existing = 0
        mode = "ab" if existing and response.status == 206 else "wb"
        total_header = response.headers.get("Content-Length")
        total = int(total_header) + existing if total_header and response.status == 206 else (
            int(total_header) if total_header else None
        )
        got = existing
        next_mark = 0
        with dest.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                got += len(chunk)
                if total and got - next_mark >= max(total // 20, 1):
                    print(f"  {dest.name}: {got / total:.0%} ({got // (1024**2)} MiB)", flush=True)
                    next_mark = got
    print(f"  {dest.name}: download finished", flush=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archives(data_root: Path) -> None:
    extracted = data_root / "_extracted"
    for split in ("train", "val", "test"):
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
