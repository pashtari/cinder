import logging
import os
import time

import torch
import ignite
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.engine import Events

# Silence noisy third-party and internal loggers
for _name in (
    "timm",
    "httpx",
    "PIL",
    "ignite.engine.engine.Engine",
    "ignite.distributed",
):
    logging.getLogger(_name).setLevel(logging.WARNING)


def make_header(title: str) -> str:
    """Create a section header with consistent alignment."""
    header_width = len("═" * 55)
    title_len = len(title)
    # Format: "═ Title ═════════════════════════════════"
    # Total: 2 (left) + 1 (space) + title_len + 1 (space) + (header_width - title_len - 4) (right)
    right_fill = header_width - title_len - 4
    return f"═ {title} {'═' * right_fill}"


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"


def _samples_seen(engine):
    """Number of samples an evaluation run actually processed.

    ``len(dataloader.dataset)`` is wrong whenever the run was truncated with
    ``epoch_length`` (as the sub-sampled train-set pass is), so derive it from
    the iterations that ran and fall back to the dataset size.
    """
    state = engine.state
    loader = state.dataloader
    iterations = state.epoch_length or 0
    if not iterations:
        return len(loader.dataset)
    # Loader length and batch size are both per rank under DDP, while the
    # dataset is the global one -- so scale up before comparing the two.
    batch_size = getattr(loader, "batch_size", None) or 1
    seen = iterations * batch_size * idist.get_world_size()
    return min(seen, len(loader.dataset))


def _progress(iteration, objects):
    """``Iter 2000/20000 │ Epoch 400/4000`` for a position in training.

    Training is budgeted in iterations, but "how many times the model has seen
    each image" is still the quantity the small-dataset literature reports, so
    both are shown. Epochs here are true passes over the training set, not the
    engine's epochs (which are evaluation intervals).
    """
    max_iters = objects.get("max_iters") or iteration
    per_pass = max(objects.get("iters_per_pass") or 1, 1)
    total_epochs = max_iters / per_pass
    # Real budgets are hundreds of passes and read best as integers; a short
    # debug run can be a fraction of one pass, where rounding to "0/1" would
    # hide all the progress there is.
    fmt = ".1f" if total_epochs < 10 else ".0f"
    wi = len(str(max_iters))
    we = len(f"{total_epochs:{fmt}}")
    return (
        f"Iter {iteration:>{wi}}/{max_iters} │ "
        f"Epoch {iteration / per_pass:>{we}{fmt}}/{total_epochs:{fmt}}"
    )


def proglogger(objects, log_interval=50, **kwargs):
    """
    Create a progress logger handler for Ignite trainers and evaluators.

    Args:
        objects: Dictionary containing trainer, train_evaluator, and val_evaluator
        log_interval: Iterations between training lines, shared with the
            TensorBoard curves. Independent of the evaluation interval, which is
            usually far longer; set to 1 to log every step.
        **kwargs: Additional arguments to pass to setup_logger

    Returns:
        None
    """

    rank = idist.get_rank()

    logger = setup_logger(
        name="ProgressLogger",
        format="%(message)s",
        distributed_rank=0,
        **kwargs,
    )

    # --- Header ---
    if rank == 0:
        # Experiment section
        experiment_name = objects.get("experiment", "unknown")
        run_id = os.path.basename(objects.get("output_dir", "unknown"))
        seed = objects.get("seed", 42)
        logger.info(make_header("Experiment"))
        logger.info(f"  Name       : {experiment_name}")
        logger.info(f"  Run ID     : {run_id}")
        logger.info(f"  Seed       : {seed}")
        logger.info("")

        # Environment section
        logger.info(make_header("Environment"))
        fw_parts = [f"PyTorch {torch.__version__}", f"Ignite {ignite.__version__}"]
        if torch.cuda.is_available():
            fw_parts.append(f"CUDA {torch.version.cuda}")
        logger.info(f"  Framework : {' │ '.join(fw_parts)}")
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

    # --- Training ---
    if "trainer" in objects:
        trainer = objects["trainer"]
        train_evaluator = objects["train_evaluator"]
        val_evaluator = objects["val_evaluator"]

        _ctx = {
            "loss_sum": 0.0,
            "loss_count": 0,
            "window_t0": 0.0,
            "train_t0": 0.0,
            "eval_t0": 0.0,
            "train_times": [],
            "eval_times": [],
            "eval_speeds": [],
            "training_logged": False,
            "step_t0": 0.0,
        }

        @trainer.on(Events.STARTED)
        def _on_train_start(engine):
            _ctx["train_t0"] = time.time()

            # --- Dataset Summary ---
            if rank == 0:
                train_dataset = engine.state.dataloader.dataset
                if "val_loader" in objects:
                    val_dataset = objects["val_loader"].dataset
                else:
                    val_dataset = None

                logger.info(make_header("Dataset"))
                logger.info(f"  Name       : {train_dataset.__class__.__name__}")
                if hasattr(train_dataset, "num_classes"):
                    logger.info(f"  # Classes  : {train_dataset.num_classes}")
                logger.info(f"  # Train    : {len(train_dataset)}")
                if val_dataset is not None:
                    logger.info(f"  # Val      : {len(val_dataset)}")

                # --- Model Summary ---
                if "model" in objects:
                    model = objects["model"]
                    raw_model = getattr(model, "module", model)
                    n_params = sum(p.numel() for p in raw_model.parameters())
                    n_params_m = n_params / 1e6

                    logger.info("")
                    logger.info(make_header("Model"))
                    logger.info(f"  Name       : {raw_model.__class__.__name__}")
                    logger.info(f"  # Params   : {n_params_m:.2f}M")
                logger.info("")

        # One line per log_interval iterations, independent of the evaluation
        # interval: evaluation is expensive and therefore rare, but training
        # progress should still be visible in between.
        every = max(int(log_interval), 1)

        @trainer.on(Events.STARTED)
        def _open_training_section(engine):
            _ctx["loss_sum"] = 0.0
            _ctx["loss_count"] = 0
            _ctx["window_t0"] = time.time()
            if rank == 0 and not _ctx["training_logged"]:
                logger.info(make_header("Training"))
                _ctx["training_logged"] = True

        @trainer.on(Events.ITERATION_COMPLETED)
        def _accumulate_loss(engine):
            _ctx["loss_sum"] += engine.state.output
            _ctx["loss_count"] += 1

        @trainer.on(Events.ITERATION_COMPLETED(every=every) | Events.COMPLETED)
        def _log_train_interval(engine):
            if rank != 0 or _ctx["loss_count"] == 0:
                return
            avg_loss = _ctx["loss_sum"] / _ctx["loss_count"]
            elapsed = time.time() - _ctx["window_t0"]
            _ctx["train_times"].append(elapsed)

            lr = 0.0
            if "optimizer" in objects:
                for param_group in objects["optimizer"].param_groups:
                    lr = param_group["lr"]
                    break

            logger.info(
                f"[{_ts()}] Train  {_progress(engine.state.iteration, objects)} │ "
                f"time={_fmt_time(elapsed)} │ lr={lr:.1e} │ loss={avg_loss:.4f}"
            )
            _ctx["loss_sum"] = 0.0
            _ctx["loss_count"] = 0
            _ctx["window_t0"] = time.time()

        @train_evaluator.on(Events.STARTED)
        def _eval_start(engine):
            _ctx["eval_t0"] = time.time()

        @val_evaluator.on(Events.COMPLETED)
        def _log_eval(engine):
            if rank != 0:
                return
            elapsed = time.time() - _ctx["eval_t0"]
            _ctx["eval_times"].append(elapsed)

            # Count what was actually scored, not the dataset size: with
            # trainer.eval_train_ratio < 1 the train pass covers a subset, and
            # charging it the full dataset inflates the reported rate severalfold.
            train_n = _samples_seen(train_evaluator)
            val_n = _samples_seen(engine)
            speed = (train_n + val_n) / max(elapsed, 1e-6)
            _ctx["eval_speeds"].append(speed)

            train_m = train_evaluator.state.metrics
            val_m = engine.state.metrics
            metric_parts = " │ ".join(
                f"train_{k}={train_m[k]:.4f} │ val_{k}={val_m[k]:.4f}" for k in val_m
            )

            logger.info(
                f"[{_ts()}] Eval   {_progress(trainer.state.iteration, objects)} │ "
                f"time={_fmt_time(elapsed)} │ speed={speed:.1f} samples/s │ {metric_parts}"
            )
            logger.info("")

        @trainer.on(Events.COMPLETED)
        def _log_footer(engine):
            if rank != 0:
                return
            total = time.time() - _ctx["train_t0"]

            avg_train = sum(_ctx["train_times"]) / max(len(_ctx["train_times"]), 1)

            logger.info(make_header("Summary"))
            logger.info(f"  Total time      : {_fmt_time(total)}")
            if _ctx["eval_times"]:
                avg_eval = sum(_ctx["eval_times"]) / len(_ctx["eval_times"])
                avg_speed = sum(_ctx["eval_speeds"]) / len(_ctx["eval_speeds"])
                logger.info(
                    f"  Avg interval    : {_fmt_time(avg_train)} (train) / {_fmt_time(avg_eval)} (eval)"
                )
                logger.info(f"  Avg speed       : {avg_speed:.1f} samples/s")
            else:
                logger.info(f"  Avg interval    : {_fmt_time(avg_train)} (train)")
            logger.info("")

    # --- Evaluation only ---
    elif "evaluator" in objects:
        evaluator = objects["evaluator"]

        _eval_ctx = {
            "eval_t0": 0.0,
            "step_t0": 0.0,
        }

        @evaluator.on(Events.STARTED)
        def _on_eval_start(engine):
            _eval_ctx["eval_t0"] = time.time()

            if rank == 0:
                dataset = engine.state.dataloader.dataset
                logger.info(make_header("Dataset"))
                logger.info(f"  Name       : {dataset.__class__.__name__}")
                if hasattr(dataset, "num_classes"):
                    logger.info(f"  # Classes  : {dataset.num_classes}")
                logger.info(f"  # Samples  : {len(dataset)}")

                if "model" in objects:
                    model = objects["model"]
                    raw_model = getattr(model, "module", model)
                    n_params = sum(p.numel() for p in raw_model.parameters())
                    logger.info("")
                    logger.info(make_header("Model"))
                    logger.info(f"  Name       : {raw_model.__class__.__name__}")
                    logger.info(f"  # Params   : {n_params / 1e6:.2f}M")
                logger.info("")
                logger.info(make_header("Evaluation"))

        @evaluator.on(Events.ITERATION_STARTED)
        def _on_eval_step_start(engine):
            _eval_ctx["step_t0"] = time.time()

        # A standalone evaluation is a single pass, so there is no natural
        # interval -- only log progress if one was asked for explicitly.
        if log_interval:

            @evaluator.on(Events.ITERATION_COMPLETED(every=int(log_interval)))
            def _log_eval_step(engine):
                if rank != 0:
                    return
                step = engine.state.iteration
                total_steps = len(engine.state.dataloader)
                ws = len(str(total_steps))
                elapsed = time.time() - _eval_ctx["step_t0"]

                logger.info(
                    f"[{_ts()}] Eval   Step {step:>{ws}}/{total_steps} │ "
                    f"time={_fmt_time(elapsed)}"
                )

        @evaluator.on(Events.COMPLETED)
        def _log_eval(engine):
            if rank != 0:
                return
            total = time.time() - _eval_ctx["eval_t0"]
            metrics = engine.state.metrics
            parts = " │ ".join(f"{k}={metrics[k]:.4f}" for k in metrics)

            logger.info("")
            logger.info(make_header("Summary"))
            logger.info(f"  Total time : {_fmt_time(total)}")
            logger.info(f"  Metrics    : {parts}")
            logger.info("")
