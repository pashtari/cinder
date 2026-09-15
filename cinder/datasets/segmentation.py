"""A segmentation dataset of paired image and mask files."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class SegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load RGB images and their segmentation masks.

    Images and masks are wrapped as ``torchvision.tv_tensors``, so a paired
    transform moves both identically and changes only the image's colors.

    Args:
        datalist: Records with ``"input"`` and ``"target"`` file paths.
        transform: Optional paired transform ``(image, mask) -> (image, mask)``.
        num_classes: ``1`` for binary masks; larger values keep class indices.
        instance_labels: For binary masks, keep each object's integer ID, as
            object metrics need. Leave disabled for training with binary losses.

    Returns:
        ``(image, target)`` per item: a float32 image ``(3, H, W)`` in ``[0, 1]``,
        and a target that is a float ``(1, H, W)`` mask (binary), an int64
        ``(1, H, W)`` object-ID map (``instance_labels``) or an int64 ``(H, W)``
        class map (multi-class).
    """

    _to_float = v2.ToDtype(torch.float32, scale=True)

    def __init__(
        self,
        datalist: Sequence[Mapping[str, Any]],
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        num_classes: int = 1,
        instance_labels: bool = False,
    ) -> None:
        self.datalist = datalist
        self.transform = transform
        self.num_classes = num_classes
        self.instance_labels = instance_labels

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.datalist[index]
        with Image.open(sample["input"]) as source:
            image = tv_tensors.Image(source.convert("RGB"))
        with Image.open(sample["target"]) as mask:
            # Palette and 16-bit masks hold label IDs that a grayscale conversion
            # would destroy, so only RGB binary masks are converted.
            if (
                self.num_classes == 1
                and not self.instance_labels
                and mask.mode in {"RGB", "RGBA"}
            ):
                mask = mask.convert("L")
            target = tv_tensors.Mask(mask, dtype=torch.long)
        if target.shape[0] != 1:
            raise ValueError(
                f"mask {sample['target']} must have one channel of label IDs, "
                f"got {target.shape[0]} (mode {mask.mode})"
            )

        if self.transform is not None:
            image, target = self.transform(image, target)

        image = self._to_float(image)

        if self.num_classes == 1:
            if not self.instance_labels:
                target = (target > 0).float()
        else:
            target = target.squeeze(0)

        return image, target
