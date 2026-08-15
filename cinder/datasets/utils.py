import glob
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from PIL import Image


def glob_datalist(root: str | Path, input_glob: str, target_fn) -> list[dict]:
    """Build a datalist by globbing input paths and deriving target paths.

    Args:
        root: Root data directory.
        input_glob: Glob pattern relative to *root* for inputs.
        target_fn: Callable that maps an input path to its target path.

    Returns:
        Sorted list of ``{"id": ..., "input": ..., "target": ...}`` dicts.
    """
    pattern = os.path.join(root, input_glob)
    inputs = sorted(glob.glob(pattern))
    datalist = []
    for input_path in inputs:
        sample_id = os.path.splitext(os.path.basename(input_path))[0]
        datalist.append(
            {
                "id": sample_id,
                "input": input_path,
                "target": target_fn(input_path),
            }
        )
    return datalist


class SegmentationDataset(Dataset):
    """Dataset for paired input/target segmentation datasets.

    Uses ``torchvision.tv_tensors`` so that any ``torchvision.transforms.v2``
    pipeline automatically applies paired spatial transforms to both input and
    mask with consistent random state.

    Args:
        datalist: List of ``{"input": ..., "target": ...}`` dicts.
        transform: Optional paired spatial transform.
        num_classes: Number of segmentation classes.
            ``1`` → binary mode (target thresholded to {0, 1} float).
            ``>1`` → multi-class mode (target kept as long class indices).
    """

    _to_float = v2.ToDtype(torch.float32, scale=True)

    def __init__(self, datalist: list[dict], transform=None, num_classes: int = 1):
        self.datalist = datalist
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        input = Image.open(item["input"]).convert("RGB")
        target = Image.open(item["target"]).convert("L")

        # Wrap as tv_tensors for automatic paired dispatch in v2 transforms
        input = tv_tensors.Image(input)
        target = tv_tensors.Mask(target)

        if self.transform is not None:
            input, target = self.transform(input, target)

        # Ensure float32 and scale input to [0, 1]
        input = self._to_float(input)

        if self.num_classes == 1:
            target = (target > 0).float()
        else:
            target = target.long().squeeze(0)  # (H, W) class indices

        return input, target
