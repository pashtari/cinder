"""Checkpointing and progress logging, attached to minimal Ignite runs."""

import re

import pytest
import torch
from ignite.engine import Engine, Events
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cinder.handlers import load_checkpoint, setup_checkpointing, setup_progress_logging


def _loader(num_samples=4, batch_size=2):
    return DataLoader(TensorDataset(torch.zeros(num_samples, 1)), batch_size=batch_size)


def _training_objects(tmp_path, lr_scheduler=None):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return {
        "model": model,
        "trainer": Engine(lambda engine, batch: 1.0),
        "train_evaluator": Engine(lambda engine, batch: None),
        "val_evaluator": Engine(lambda engine, batch: None),
        "val_loader": _loader(),
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "experiment": "test",
        "seed": 0,
        "output_dir": str(tmp_path),
        "max_iters": 4,
        "iters_per_pass": 2,
    }


def test_checkpointing_skips_a_missing_lr_scheduler(tmp_path):
    objects = _training_objects(tmp_path, lr_scheduler=None)
    setup_checkpointing(objects, dirname=tmp_path / "checkpoints", n_saved=1)
    objects["trainer"].run(_loader(), max_epochs=2)

    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "checkpoint_4.pt")
    assert set(checkpoint) == {"trainer", "model", "optimizer"}


def test_checkpointing_keeps_the_best_validation_score(tmp_path):
    objects = _training_objects(tmp_path)
    scores = iter([0.2, 0.9, 0.5])
    val_evaluator = objects["val_evaluator"]

    @val_evaluator.on(Events.COMPLETED)
    def score(engine):
        engine.state.metrics["dice"] = next(scores)

    trainer = objects["trainer"]
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: val_evaluator.run(_loader())
    )
    setup_checkpointing(
        objects, dirname=tmp_path, n_saved=1, score_metric="dice", require_empty=False
    )
    trainer.run(_loader(), max_epochs=3)

    assert [path.name for path in tmp_path.glob("*.pt")] == [
        "checkpoint_4_dice=0.9000.pt"
    ]


def test_evaluation_restores_the_model_weights(tmp_path):
    saved = nn.Linear(1, 1)
    torch.save({"model": saved.state_dict()}, tmp_path / "weights.pt")
    model = nn.Linear(1, 1)
    setup_checkpointing(
        {"evaluator": Engine(lambda e, b: None), "model": model},
        load_from=tmp_path / "weights.pt",
    )
    torch.testing.assert_close(model.state_dict(), saved.state_dict())

    with pytest.raises(ValueError, match="requires a checkpoint"):
        setup_checkpointing({"evaluator": Engine(lambda e, b: None), "model": model})


def test_training_times_exclude_evaluation(tmp_path, monkeypatch):
    """Each progress line times its training window, not the evaluation before it."""
    now = [0.0]
    monkeypatch.setattr("ignite.handlers.timing.perf_counter", lambda: now[0])

    objects = _training_objects(tmp_path)
    trainer = objects["trainer"]
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda _: now.__setitem__(0, now[0] + 1.0)
    )
    log_file = tmp_path / "train.log"
    setup_progress_logging(objects, log_interval=2, filepath=str(log_file))

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(_):
        now[0] += 100.0  # a slow evaluation
        objects["val_evaluator"].run(_loader())

    trainer.run(_loader(), max_epochs=2, epoch_length=2)

    log = log_file.read_text()
    assert re.findall(r"Train .*? time=(.+?) │", log) == ["2.00s", "2.00s"]
    assert re.findall(r"Eval .*? time=(.+?) │", log) == ["1m 40s", "1m 40s"]
    assert "Avg interval    : 2.00s (train) / 1m 40s (eval)" in log
