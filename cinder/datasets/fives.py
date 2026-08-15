"""FIVES — Fundus Image VEssel Segmentation dataset.

Directory structure expected::

    root/
    ├── train/
    │   ├── Original/       # RGB fundus images (*.png)
    │   └── Ground truth/   # Binary vessel masks (*.png)
    └── test/
        ├── Original/
        └── Ground truth/
"""

from pathlib import Path

from .utils import glob_datalist


def _target_fn(img_path: str) -> str:
    return img_path.replace("/Original/", "/Ground truth/")


def create_datalist(root: str | Path, split: str = "train") -> list[dict]:
    """Return a datalist for the given *split* (``"train"`` or ``"test"``)."""
    return glob_datalist(root, f"{split}/Original/*.png", _target_fn)
