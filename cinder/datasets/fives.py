"""FIVES fundus image vessel segmentation dataset.

Expected layout::

    root/
    ├── train/
    │   ├── Original/       # RGB fundus images (*.png)
    │   └── Ground truth/   # binary vessel masks (*.png), same file names
    └── test/
        ├── Original/
        └── Ground truth/
"""

import os
from pathlib import Path

from .utils import glob_datalist

__all__ = ["create_datalist"]


def create_datalist(root: str | Path, split: str = "train") -> list[dict[str, str]]:
    """List image and mask pairs for the ``"train"`` or ``"test"`` split."""

    def target_path(image_path: str) -> str:
        return os.path.join(root, split, "Ground truth", os.path.basename(image_path))

    return glob_datalist(root, f"{split}/Original/*.png", target_path)
