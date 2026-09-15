"""File lists of paired images and masks."""

import glob
import os
from collections.abc import Callable
from pathlib import Path


def glob_datalist(
    root: str | Path, input_glob: str, target_fn: Callable[[str], str]
) -> list[dict[str, str]]:
    """List the images matching ``input_glob`` under ``root`` with their masks.

    Each record holds the image's file stem as ``"id"``, its path as ``"input"``
    and ``target_fn(path)`` as ``"target"``. Records are sorted by image path;
    target paths are not checked for existence.
    """
    inputs = sorted(glob.glob(os.path.join(root, input_glob)))
    return [
        {
            "id": os.path.splitext(os.path.basename(path))[0],
            "input": path,
            "target": target_fn(path),
        }
        for path in inputs
    ]
