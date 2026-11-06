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
from levir_cd.metrics import failure_summary, per_image_scores, scores_from_counts, summarize_per_image, threshold_sweep
from levir_cd.plan import comparison_markdown, experiment_plan
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


def test_manifest_builder_pairs_a_b_label(tmp_path):
    from PIL import Image

    from levir_cd.audit import ensure_manifest

    for split in ("train", "val"):
        for part in ("A", "B", "label"):
            folder = tmp_path / "levir-cd" / split / part
            folder.mkdir(parents=True)
            Image.new("RGB" if part != "label" else "L", (8, 8), color=0).save(folder / "sample.png")
    project = tmp_path / "project"
    project.mkdir()
    os_env = __import__("os").environ
    previous = os_env.get("LEVIR_DATA_ROOT")
    os_env["LEVIR_DATA_ROOT"] = str(tmp_path / "levir-cd")
    try:
        path = ensure_manifest(project)
    finally:
        if previous is None:
            os_env.pop("LEVIR_DATA_ROOT", None)
        else:
            os_env["LEVIR_DATA_ROOT"] = previous
    text = path.read_text(encoding="utf-8")
    assert "train" in text and "val" in text and "sample" in text


def test_locate_dataset_finds_nested_zips(tmp_path):
    from levir_cd.audit import locate_dataset

    folder = tmp_path / "Downloads" / "cd"
    folder.mkdir(parents=True)
    (folder / "train.zip").write_bytes(b"a")
    (folder / "val.zip").write_bytes(b"b")
    found = locate_dataset(tmp_path / "project", extra_roots=[tmp_path])
    assert found == folder.resolve()


def test_ensure_manifest_downloads_when_missing(tmp_path, monkeypatch):
    import hashlib
    import io
    import threading
    import zipfile
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    from levir_cd import audit

    payloads = {}
    for split in ("train", "val", "test"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for part, mode in (("A", "RGB"), ("B", "RGB"), ("label", "L")):
                raw = io.BytesIO()
                Image.new(mode, (8, 8), color=0).save(raw, format="PNG")
                archive.writestr(f"{part}/{split}_1.png", raw.getvalue())
        payloads[split] = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1].removesuffix(".zip")
            body = payloads[name]
            start = 0
            status = 200
            header = self.headers.get("Range")
            if header:
                start = int(header.split("=", 1)[1].split("-", 1)[0])
                if start >= len(body):
                    self.send_response(416)
                    self.end_headers()
                    return
                status = 206
            chunk = body[start:]
            self.send_response(status)
            self.send_header("Content-Length", str(len(chunk)))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    splits = {
        split: {
            "url": f"http://127.0.0.1:{port}/{split}.zip",
            "md5": hashlib.md5(payloads[split]).hexdigest(),
        }
        for split in payloads
    }
    monkeypatch.setattr(audit, "LEVIR_ZIPS", splits)
    monkeypatch.setenv("LEVIR_DATA_ROOT", str(tmp_path / "downloaded"))
    try:
        path = audit.ensure_manifest(tmp_path / "project")
    finally:
        server.shutdown()
    text = path.read_text(encoding="utf-8")
    assert "train_1" in text and "val_1" in text and "test_1" in text


def test_experiment_plan_covers_proposal():
    rows = experiment_plan("/tmp/levir-plan", mode="full")
    roles = [row["role"] for row in rows]
    assert roles.count("main") == 6
    assert "triad" in roles and "loss" in roles and "augmentation" in roles
    assert all(row["config"].ALLOW_TEST is False for row in rows)
    text = comparison_markdown([])
    assert "no runs yet" in text


def test_threshold_sweep_marks_official_half():
    ds = SyntheticChangeTiles(n_tiles=2, size=32, seed=3)
    loader, _ = make_loaders(ds, ds, batch_size=2)
    model = build_model("unet", base_channels=4)
    rows = threshold_sweep(model, loader, torch.device("cpu"), grid=[0.5])
    assert rows[0]["official"] == 1.0
    assert 0.0 <= rows[0]["f1"] <= 1.0


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
