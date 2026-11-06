from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from levir_cd.baseline import absolute_difference_baseline
from levir_cd.config import ExperimentConfig
from levir_cd.data import SyntheticChangeTiles, make_loaders
from levir_cd.losses import segmentation_loss
from levir_cd.metrics import failure_summary, per_image_scores, scores_from_counts, summarize_per_image
from levir_cd.models import build_model, count_parameters, paper_name
from levir_cd.train import run_experiment


def test_daudt_triad_shapes_and_params():
    for name in ("unet", "siamese_diff", "siamese_conc", "fc_ef", "fc_siam_diff", "fc_siam_conc"):
        model = build_model(name, base_channels=8)
        x = torch.zeros(2, 6, 64, 64)
        y = model(x)
        assert tuple(y.shape) == (2, 1, 64, 64)
        assert count_parameters(model) > 1000
        assert "FC-" in paper_name(name) or "U-Net" in paper_name(name)


def test_losses_finite():
    logits = torch.randn(2, 1, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()
    for kind in ("bce", "bce_dice"):
        loss = segmentation_loss(logits, target, kind=kind)
        assert torch.isfinite(loss)


def test_absdiff_baseline():
    ds = SyntheticChangeTiles(n_tiles=4, size=32, seed=0)
    loader, _ = make_loaders(ds, ds, batch_size=2)
    result = absolute_difference_baseline(loader)
    assert 0.0 <= result["f1"] <= 1.0
    assert "threshold" in result


def test_scores_include_accuracy_without_using_it_as_f1():
    scores = scores_from_counts(tp=10, fp=10, fn=0, tn=80)
    assert scores["f1"] == 2 * 10 / (20 + 10)
    assert abs(scores["accuracy"] - 0.9) < 1e-9


def test_per_image_metrics():
    ds = SyntheticChangeTiles(n_tiles=4, size=32, seed=1)
    loader, _ = make_loaders(ds, ds, batch_size=2)
    model = build_model("unet", base_channels=4)
    frame = per_image_scores(model, loader, torch.device("cpu"))
    assert len(frame) == 4
    summary = summarize_per_image(frame)
    assert summary["n"] == 4
    failures = failure_summary(frame)
    assert failures["n"] == 4
    assert "n_missed_small" in failures


def test_smoke_train_unet(tmp_path):
    cfg = ExperimentConfig(
        PROJECT_ROOT=str(tmp_path),
        RUN_MODE="smoke",
        MODEL="unet",
        LOSS="bce",
        AUGMENT=False,
        SEED=42,
        TILE_SIZE=32,
        IMAGE_SIZE=32,
        BASE_CHANNELS=4,
        SMOKE_TRAIN_TILES=4,
        SMOKE_VAL_TILES=2,
        SMOKE_EPOCHS=1,
        SMOKE_BATCH_SIZE=2,
    )
    summary = run_experiment(cfg, device=torch.device("cpu"))
    assert summary["completed_epochs"] == 1
    assert summary["paper_name"].startswith("FC-EF")
    assert Path(summary["run_dir"], "best_model.pt").is_file()
    assert Path(summary["run_dir"], "per_image_val_metrics.csv").is_file()


def test_smoke_train_siamese_variants(tmp_path):
    for model_name in ("siamese_diff", "siamese_conc"):
        cfg = ExperimentConfig(
            PROJECT_ROOT=str(tmp_path / model_name),
            RUN_MODE="smoke",
            MODEL=model_name,
            LOSS="bce_dice",
            AUGMENT=True,
            SEED=123,
            TILE_SIZE=32,
            IMAGE_SIZE=32,
            BASE_CHANNELS=4,
            SMOKE_TRAIN_TILES=4,
            SMOKE_VAL_TILES=2,
            SMOKE_EPOCHS=1,
            SMOKE_BATCH_SIZE=2,
        )
        summary = run_experiment(cfg, device=torch.device("cpu"))
        assert summary["model"] == model_name
        assert summary["test_set_touched"] is False
        assert "FC-Siam" in summary["paper_name"]
