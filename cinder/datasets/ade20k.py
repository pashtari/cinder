"""ADE20K — Scene Parsing dataset (MIT).

Directory structure expected::

    root/
    ├── images/
    │   ├── training/       # RGB scene images (*.jpg)
    │   └── validation/     # RGB scene images (*.jpg)
    └── annotations/
        ├── training/       # Semantic masks (*.png, values 0–150)
        └── validation/

Annotations use class indices 0–150 where 0 is unlabelled/background
and 1–150 are the 150 semantic categories.
"""

from pathlib import Path

from .utils import glob_datalist


def _target_fn(img_path: str) -> str:
    return img_path.replace("/images/", "/annotations/").replace(".jpg", ".png")


def create_datalist(root: str | Path, split: str = "training") -> list[dict]:
    """Return a datalist for the given *split* (``"training"`` or ``"validation"``)."""
    return glob_datalist(root, f"images/{split}/*.jpg", _target_fn)
