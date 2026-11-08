from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Shared settings for U-Net / Siamese experiments and smoke tests."""

    PROJECT_ROOT: str = "."
    RUN_MODE: str = "smoke"  # smoke | benchmark | full
    # unet/fc_ef | siamese/siamese_diff/fc_siam_diff | siamese_conc/fc_siam_conc
    MODEL: str = "unet"
    LOSS: str = "bce_dice"  # bce | bce_dice
    AUGMENT: bool = True
    SEED: int = 1511
    TILE_SIZE: int = 256
    IMAGE_SIZE: int = 1024
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    BASE_CHANNELS: int = 16
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    FULL_EPOCHS: int = 30
    EARLY_STOP_PATIENCE: int = 8
    MIN_EPOCHS: int = 12
    MIN_F1_IMPROVEMENT: float = 1e-4
    THRESHOLD: float = 0.5
    # Smoke-only knobs (synthetic tiles; no LEVIR download required).
    SMOKE_TRAIN_TILES: int = 8
    SMOKE_VAL_TILES: int = 4
    SMOKE_EPOCHS: int = 1
    SMOKE_BATCH_SIZE: int = 2
    ALLOW_TEST: bool = False  # keep False until model choices are frozen

    def validate(self) -> None:
        if self.RUN_MODE not in {"smoke", "benchmark", "full"}:
            raise ValueError(f"Unknown RUN_MODE: {self.RUN_MODE}")
        from .models import MODEL_ALIASES

        if self.MODEL.lower().strip() not in MODEL_ALIASES:
            raise ValueError(
                f"Unknown MODEL: {self.MODEL}. Expected one of {sorted(MODEL_ALIASES)}"
            )
        if self.LOSS not in {"bce", "bce_dice"}:
            raise ValueError(f"Unknown LOSS: {self.LOSS}")
        if self.TILE_SIZE <= 0 or self.IMAGE_SIZE % self.TILE_SIZE != 0:
            raise ValueError("IMAGE_SIZE must be divisible by TILE_SIZE")
        if self.BATCH_SIZE < 1:
            raise ValueError("BATCH_SIZE must be >= 1")
        if self.ALLOW_TEST:
            raise ValueError("ALLOW_TEST must stay False until the locked final evaluation")

    @property
    def project_root(self) -> Path:
        return Path(self.PROJECT_ROOT).expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.project_root / "results" / "levir_audit" / "split_manifest.csv"

    @property
    def run_name(self) -> str:
        aug = "aug" if self.AUGMENT else "noaug"
        return f"{self.RUN_MODE}_{self.MODEL}_{self.LOSS}_{aug}_seed{self.SEED}_tile{self.TILE_SIZE}"

    @property
    def run_dir(self) -> Path:
        return self.project_root / "results" / "experiments" / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_batch_size(self) -> int:
        return self.SMOKE_BATCH_SIZE if self.RUN_MODE == "smoke" else self.BATCH_SIZE

    def effective_epochs(self) -> int:
        if self.RUN_MODE == "smoke":
            return self.SMOKE_EPOCHS
        if self.RUN_MODE == "benchmark":
            return 1
        return self.FULL_EPOCHS


def default_project_root(cwd: Path | None = None) -> Path:
    here = (cwd or Path.cwd()).resolve()
    if here.name == "notebooks":
        return here.parent
    if (here / "notebooks").is_dir() and (here / "src").is_dir():
        return here
    if here.name == "src":
        return here.parent
    return here
