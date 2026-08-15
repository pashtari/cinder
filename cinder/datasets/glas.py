"""GlaS — Gland Segmentation in Colon Histology Images dataset.

Directory structure expected (flat)::

    root/
    ├── train_1.bmp
    ├── train_1_anno.bmp
    ├── ...
    ├── testA_1.bmp
    ├── testA_1_anno.bmp
    ├── ...
    ├── testB_1.bmp
    └── testB_1_anno.bmp

The dataset ships a fixed official split: 85 training images (``train_*``) and
80 test images divided into Test A (``testA_*``, 60) and Test B (``testB_*``,
20). Following the GlaS challenge convention, Test A and Test B are evaluated
and reported *separately*.
"""

import os
from pathlib import Path

from .utils import glob_datalist


def _target_fn(img_path: str) -> str:
    base, ext = os.path.splitext(img_path)
    return f"{base}_anno{ext}"


# Map user-facing section names to the filename prefix(es) that define them.
_SECTION_PREFIXES = {
    "train": ("train",),
    "training": ("train",),
    "testa": ("testa",),
    "test_a": ("testa",),
    "a": ("testa",),
    "testb": ("testb",),
    "test_b": ("testb",),
    "b": ("testb",),
    "test": ("testa", "testb"),  # both test sets combined (80 images)
}


def create_datalist(root: str | Path, section: str = "train") -> list[dict]:
    """Return the datalist for an official GlaS split.

    Args:
        root: Path to the flat GlaS directory.
        section: Which split to load — ``"train"`` (85 images), ``"testA"``
            (60), ``"testB"`` (20), or ``"test"`` (testA + testB, 80).
            Case-insensitive; ``"test_a"``/``"a"`` aliases are accepted.

    Returns:
        List of ``{"id": ..., "input": ..., "target": ...}`` dicts.
    """
    prefixes = _SECTION_PREFIXES.get(section.lower())
    if prefixes is None:
        raise ValueError(
            f"Unknown section {section!r}; expected one of "
            "'train', 'testA', 'testB', 'test'."
        )

    all_samples = glob_datalist(root, "*.bmp", _target_fn)
    # Exclude annotation files picked up by the glob, then select the split.
    samples = [
        s
        for s in all_samples
        if "_anno" not in s["id"] and s["id"].lower().startswith(prefixes)
    ]

    if not samples:
        raise FileNotFoundError(
            f"No GlaS samples found for section {section!r} under {root!r}."
        )
    return samples
