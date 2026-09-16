"""End-to-end CPU runs of the training and evaluation entry points on tiny data."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("hydra")
pytest.importorskip("hydra_plugins.hydra_colorlog")  # configs/hydra/default.yaml
pytest.importorskip("tensorboard")

from hydra import compose, initialize_config_dir  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


@pytest.fixture
def glas_root(tmp_path):
    """A GlaS-style folder: RGB ``*.bmp`` images and ``*_anno.bmp`` instance masks."""
    rng = np.random.default_rng(0)
    root = tmp_path / "GlaS"
    root.mkdir()
    for prefix, count in (("train", 4), ("testA", 1), ("testB", 1)):
        for i in range(1, count + 1):
            image = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
            Image.fromarray(image).save(root / f"{prefix}_{i}.bmp")
            mask = np.zeros((40, 48), dtype=np.uint8)
            mask[5:15, 5:15], mask[20:32, 25:40] = 1, 2
            Image.fromarray(mask).save(root / f"{prefix}_{i}_anno.bmp")
    return root


def _compose(config_name, overrides):
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _overrides(glas_root, output_dir):
    return [
        f"path.dataset_dir={glas_root}",
        f"path.output_dir={output_dir}",
        "experiment=test",
        "dataset.crop_size=[32,32]",
        "dataset.batch_size=2",
        "dataset.num_workers=0",
        "model.encoder.model_name=resnet18",
        "model.encoder.pretrained=false",
        "model.encoder.out_indices=[-2,-1]",
        "model.modulator.combiner.1.rank=8",
    ]


@pytest.mark.parametrize("scheduler", ["cosine", "none"])
def test_train_then_evaluate(glas_root, tmp_path, monkeypatch, scheduler):
    from cinder.engine.eval import evaluate
    from cinder.engine.train import train

    # Run on the CPU even on a GPU machine, where ignite would pick CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_dir = tmp_path / "train"
    overrides = _overrides(glas_root, train_dir) + [
        "trainer.max_iters=4",
        "trainer.eval_interval=2",
        "trainer.log_interval=1",
        "trainer.eval_train_ratio=0.5",
    ]
    if scheduler == "none":
        overrides.append("trainer.lr_scheduler=null")
    torch.manual_seed(0)
    train(0, _compose("train", overrides))

    checkpoint = train_dir / "checkpoints" / "checkpoint_4.pt"
    saved = torch.load(checkpoint, map_location="cpu")
    expected = {"trainer", "model", "optimizer"} | (
        {"lr_scheduler"} if scheduler == "cosine" else set()
    )
    assert set(saved) == expected
    log = (train_dir / "train.log").read_text()
    assert log.count("] Train ") == 4 and log.count("] Eval ") == 2
    assert list((train_dir / "tensorboard").glob("events.*"))

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()  # Hydra creates the output directory in real runs
    evaluate(
        0,
        _compose(
            "eval",
            _overrides(glas_root, eval_dir)
            + [f"handler.checkpoint.load_from={checkpoint}", "metric=[dice,object_f1]"],
        ),
    )
    assert "Metrics    : dice=" in (eval_dir / "eval.log").read_text()
