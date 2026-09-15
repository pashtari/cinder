"""ADE20K scene parsing dataset.

Expected layout::

    root/
    ├── images/
    │   ├── training/       # RGB scene images (*.jpg)
    │   └── validation/
    └── annotations/
        ├── training/       # label maps (*.png), 0 = unlabeled, 1-150 = classes
        └── validation/
"""

import os
from pathlib import Path

from .utils import glob_datalist

__all__ = ["create_datalist"]


def create_datalist(root: str | Path, split: str = "training") -> list[dict[str, str]]:
    """List image and label-map pairs for ``"training"`` or ``"validation"``."""

    def target_path(image_path: str) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(root, "annotations", split, f"{stem}.png")

    return glob_datalist(root, f"images/{split}/*.jpg", target_path)
