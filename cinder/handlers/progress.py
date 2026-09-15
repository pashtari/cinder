"""Console and file logs of run information, progress and summaries."""

import logging
import os
import time
from typing import Any

import ignite
import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.utils import setup_logger
from torch import nn

__all__ = ["quiet_library_loggers", "setup_progress_logging"]


def quiet_library_loggers() -> None:
    """Keep library messages from interrupting the progress display."""
    for name in (
        "timm",
        "httpx",
        "PIL",
        "ignite.engine.engine.Engine",
        "ignite.distributed",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _header(title: str) -> str:
    return f"═ {title} {'═' * (51 - len(title))}"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        minutes, remaining_seconds = divmod(int(seconds), 60)
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remainder = divmod(int(seconds), 3600)
    return f"{hours}h {remainder // 60:02d}m"


def _format_progress(iteration: int, max_iters: int, iters_per_pass: int) -> str:
    """Format the iteration and the passes over the training set.

    Passes are reported instead of Ignite epochs, which are evaluation intervals.
    """
    num_passes = max_iters / iters_per_pass
    fmt = ".1f" if num_passes < 10 else ".0f"  # show fractional passes in short runs
    iteration_width = len(str(max_iters))
    passes_width = len(f"{num_passes:{fmt}}")
    return (
        f"Iter {iteration:>{iteration_width}}/{max_iters} │ "
        f"Epoch {iteration / iters_per_pass:>{passes_width}{fmt}}/{num_passes:{fmt}}"
    )


def _num_scored_samples(engine: Engine) -> int:
    """Count the samples an evaluation run scored, allowing for shortened runs."""
    loader = engine.state.dataloader
    # The loader's batch size is per process, while its dataset is global.
    batch_size = getattr(loader, "batch_size", None) or 1
    num_samples = engine.state.epoch_length * batch_size * idist.get_world_size()
    return min(num_samples, len(loader.dataset))


def _log_run_info(logger: logging.Logger, objects: dict[str, Any]) -> None:
    logger.info(_header("Experiment"))
    logger.info(f"  Name       : {objects['experiment']}")
    logger.info(f"  Run ID     : {os.path.basename(objects['output_dir'])}")
    logger.info(f"  Seed       : {objects['seed']}")
    logger.info("")

    logger.info(_header("Environment"))
    frameworks = [f"PyTorch {torch.__version__}", f"Ignite {ignite.__version__}"]
    if torch.cuda.is_available():
        frameworks.append(f"CUDA {torch.version.cuda}")
    logger.info(f"  Framework : {' │ '.join(frameworks)}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device    : GPU {i} ({torch.cuda.get_device_name(i)})")
    else:
        logger.info("  Device    : CPU")
    if idist.get_world_size() > 1:
        logger.info(
            f"  Backend   : {idist.backend()} │ World size: {idist.get_world_size()}"
        )
    logger.info("")


def _log_dataset(logger: logging.Logger, dataset: Any, counts: dict[str, int]) -> None:
    logger.info(_header("Dataset"))
    logger.info(f"  Name       : {type(dataset).__name__}")
    if hasattr(dataset, "num_classes"):
        logger.info(f"  # Classes  : {dataset.num_classes}")
    for label, count in counts.items():
        logger.info(f"  # {label:<9}: {count}")


def _log_model(logger: logging.Logger, model: nn.Module) -> None:
    model = getattr(model, "module", model)  # unwrap DistributedDataParallel
    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info("")
    logger.info(_header("Model"))
    logger.info(f"  Name       : {type(model).__name__}")
    logger.info(f"  # Params   : {num_parameters / 1e6:.2f}M")
    logger.info("")


def setup_progress_logging(
    objects: dict[str, Any], log_interval: int = 50, **kwargs: Any
) -> None:
    """Log run information, progress lines and a final summary.

    Args:
        objects: Run objects from the entry point. Training passes ``trainer``,
            ``train_evaluator``, ``val_evaluator``, ``val_loader``, ``optimizer``,
            ``max_iters`` and ``iters_per_pass``; evaluation passes ``evaluator``.
            Both pass ``model``, ``experiment``, ``seed`` and ``output_dir``.
        log_interval: Iterations between progress lines; during evaluation,
            ``0`` disables them.
        **kwargs: Passed to :func:`ignite.utils.setup_logger`, e.g. ``filepath``.
    """
    # Only rank 0 logs; other ranks get a logger without output. Resetting lets a
    # later run in the same process write to its own file.
    logger = setup_logger(
        name="ProgressLogger", format="%(message)s", reset=True, **kwargs
    )
    _log_run_info(logger, objects)
    if "trainer" in objects:
        _attach_training_progress(logger, objects, log_interval)
    else:
        _attach_evaluation_progress(logger, objects, log_interval)


def _attach_training_progress(
    logger: logging.Logger, objects: dict[str, Any], log_interval: int
) -> None:
    trainer = objects["trainer"]
    train_evaluator = objects["train_evaluator"]
    val_evaluator = objects["val_evaluator"]

    def progress() -> str:
        return _format_progress(
            trainer.state.iteration, objects["max_iters"], objects["iters_per_pass"]
        )

    total_timer = Timer().attach(trainer, start=Events.STARTED, pause=Events.COMPLETED)
    # Paused while the evaluation hook runs, so training times exclude evaluation.
    train_timer = Timer().attach(
        trainer,
        start=Events.STARTED,
        pause=Events.EPOCH_COMPLETED,
        resume=Events.EPOCH_STARTED,
    )
    eval_timer = Timer()
    trainer.add_event_handler(Events.EPOCH_COMPLETED, eval_timer.reset)

    loss_sum, loss_count = 0.0, 0
    train_times: list[float] = []
    eval_times: list[float] = []
    eval_speeds: list[float] = []

    @trainer.on(Events.STARTED)
    def log_setup(engine: Engine) -> None:
        counts = {
            "Train": len(engine.state.dataloader.dataset),
            "Val": len(objects["val_loader"].dataset),
        }
        _log_dataset(logger, engine.state.dataloader.dataset, counts)
        _log_model(logger, objects["model"])
        logger.info(_header("Training"))

    @trainer.on(Events.ITERATION_COMPLETED)
    def accumulate_loss(engine: Engine) -> None:
        nonlocal loss_sum, loss_count
        loss_sum += engine.state.output
        loss_count += 1

    @trainer.on(
        Events.ITERATION_COMPLETED(every=max(log_interval, 1)) | Events.COMPLETED
    )
    def log_training(engine: Engine) -> None:
        nonlocal loss_sum, loss_count
        if loss_count == 0:  # the final iteration was already logged
            return
        elapsed = train_timer.value()
        train_times.append(elapsed)
        lr = objects["optimizer"].param_groups[0]["lr"]
        logger.info(
            f"[{_timestamp()}] Train  {progress()} │ "
            f"time={_format_duration(elapsed)} │ lr={lr:.1e} │ "
            f"loss={loss_sum / loss_count:.4f}"
        )
        loss_sum, loss_count = 0.0, 0
        train_timer.reset()

    @val_evaluator.on(Events.COMPLETED)
    def log_evaluation(engine: Engine) -> None:
        elapsed = eval_timer.value()
        eval_times.append(elapsed)
        # The training-set evaluation is optional and may cover only a subset.
        scored_train = train_evaluator.state.dataloader is not None
        num_samples = _num_scored_samples(engine)
        if scored_train:
            num_samples += _num_scored_samples(train_evaluator)
        speed = num_samples / max(elapsed, 1e-6)
        eval_speeds.append(speed)

        parts = []
        for name, value in engine.state.metrics.items():
            if scored_train:
                parts.append(f"train_{name}={train_evaluator.state.metrics[name]:.4f}")
            parts.append(f"val_{name}={value:.4f}")
        logger.info(
            f"[{_timestamp()}] Eval   {progress()} │ "
            f"time={_format_duration(elapsed)} │ speed={speed:.1f} samples/s │ "
            f"{' │ '.join(parts)}"
        )
        logger.info("")

    @trainer.on(Events.COMPLETED)
    def log_summary(_: Engine) -> None:
        average_train = sum(train_times) / max(len(train_times), 1)
        logger.info(_header("Summary"))
        logger.info(f"  Total time      : {_format_duration(total_timer.value())}")
        if eval_times:
            average_eval = sum(eval_times) / len(eval_times)
            logger.info(
                f"  Avg interval    : {_format_duration(average_train)} (train) / "
                f"{_format_duration(average_eval)} (eval)"
            )
            average_speed = sum(eval_speeds) / len(eval_speeds)
            logger.info(f"  Avg speed       : {average_speed:.1f} samples/s")
        else:
            logger.info(
                f"  Avg interval    : {_format_duration(average_train)} (train)"
            )
        logger.info("")


def _attach_evaluation_progress(
    logger: logging.Logger, objects: dict[str, Any], log_interval: int
) -> None:
    evaluator = objects["evaluator"]
    total_timer = Timer().attach(
        evaluator, start=Events.STARTED, pause=Events.COMPLETED
    )
    step_timer = Timer().attach(
        evaluator, start=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED
    )

    @evaluator.on(Events.STARTED)
    def log_setup(engine: Engine) -> None:
        dataset = engine.state.dataloader.dataset
        _log_dataset(logger, dataset, {"Samples": len(dataset)})
        _log_model(logger, objects["model"])
        logger.info(_header("Evaluation"))

    if log_interval > 0:

        @evaluator.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_step(engine: Engine) -> None:
            num_steps = engine.state.epoch_length
            logger.info(
                f"[{_timestamp()}] Eval   Step "
                f"{engine.state.iteration:>{len(str(num_steps))}}/{num_steps} │ "
                f"time={_format_duration(step_timer.value())}"
            )

    @evaluator.on(Events.COMPLETED)
    def log_summary(engine: Engine) -> None:
        metrics = " │ ".join(f"{k}={v:.4f}" for k, v in engine.state.metrics.items())
        logger.info("")
        logger.info(_header("Summary"))
        logger.info(f"  Total time : {_format_duration(total_timer.value())}")
        logger.info(f"  Metrics    : {metrics}")
        logger.info("")
