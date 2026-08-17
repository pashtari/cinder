from omegaconf import OmegaConf

from .engine import create_trainer, create_evaluator
from .losses import make_sampled_target_fn, SampledLoss, DiceLoss, DiceCELoss
from .metrics import (
    DiceMetric,
    IoUMetric,
    HausdorffDistanceMetric,
    DetectionF1,
    ObjectDice,
    ObjectHausdorff,
)

# ``${frac:<value>,<fraction>}`` -> round(value * fraction), as an int. Lets a
# config express a schedule length as a share of the training budget rather than
# a literal that silently stops matching when the budget changes. Registered
# here because every entrypoint under cinder.engine composes a trainer config.
OmegaConf.register_new_resolver(
    "frac", lambda value, fraction: int(round(float(value) * float(fraction))), replace=True
)
