"""GlaS gland segmentation dataset.

Expected layout, with every file in one folder::

    root/
    ├── train_1.bmp
    ├── train_1_anno.bmp
    ├── ...
    ├── testA_1.bmp
    ├── testA_1_anno.bmp
    ├── ...
    ├── testB_1.bmp
    └── testB_1_anno.bmp

The official split has 85 training, 60 Test A and 20 Test B images. The
``*_anno.bmp`` masks label each gland with its own ID, and the challenge reports
Test A and Test B separately.
"""

import os
from pathlib import Path

from .utils import glob_datalist

__all__ = ["create_datalist"]


def _target_path(image_path: str) -> str:
    stem, suffix = os.path.splitext(image_path)
    return f"{stem}_anno{suffix}"


# File-name prefixes of each split, lowercased.
_SPLIT_PREFIXES = {
    "train": ("train",),
    "testa": ("testa",),
    "testb": ("testb",),
    "test": ("testa", "testb"),
}


def create_datalist(root: str | Path, split: str = "train") -> list[dict[str, str]]:
    """List image and instance-mask pairs for an official GlaS split.

    Args:
        root: GlaS folder.
        split: ``"train"``, ``"testA"``, ``"testB"``, or ``"test"`` for both test
            sets; case-insensitive.

    Returns:
        ``{"id", "input", "target"}`` records, sorted by image path.
    """
    prefixes = _SPLIT_PREFIXES.get(split.lower())
    if prefixes is None:
        raise ValueError(
            f"Unknown split {split!r}; expected one of "
            "'train', 'testA', 'testB', 'test'."
        )

    all_samples = glob_datalist(root, "*.bmp", _target_path)
    samples = [
        sample
        for sample in all_samples
        if "_anno" not in sample["id"] and sample["id"].lower().startswith(prefixes)
    ]

    if not samples:
        raise FileNotFoundError(
            f"No GlaS samples found for split {split!r} under {root!r}."
        )
    return samples
