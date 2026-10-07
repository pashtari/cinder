"""Training and evaluation engines, segmentation losses, metrics and inferers."""

from omegaconf import OmegaConf

from .engines import create_evaluator, create_trainer, fit
from .inferers import MultiScaleFlipInferer, SlidingWindowInferer
from .losses import DiceCELoss, DiceLoss, MaskedCrossEntropyLoss, SampledLoss
from .metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    IoUMetric,
    ObjectDiceMetric,
    ObjectF1Metric,
    ObjectHausdorffMetric,
)

__all__ = [
    "create_trainer",
    "create_evaluator",
    "fit",
    "SampledLoss",
    "DiceLoss",
    "MaskedCrossEntropyLoss",
    "DiceCELoss",
    "DiceMetric",
    "IoUMetric",
    "HausdorffDistanceMetric",
    "ObjectF1Metric",
    "ObjectDiceMetric",
    "ObjectHausdorffMetric",
    "SlidingWindowInferer",
    "MultiScaleFlipInferer",
]

# `${frac:value,fraction}` in the configs is `round(value * fraction)`, so lengths
# such as the warmup follow the training budget when it is overridden. It is
# registered here so both entry points, train and eval, can resolve the configs.
OmegaConf.register_new_resolver(
    "frac",
    lambda value, fraction: int(round(float(value) * float(fraction))),
    replace=True,
)
