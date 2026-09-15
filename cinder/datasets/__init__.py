"""Segmentation datasets and the file lists of the benchmarks."""

from . import ade20k, fives, glas
from .segmentation import SegmentationDataset
from .utils import glob_datalist

__all__ = ["SegmentationDataset", "glob_datalist", "ade20k", "fives", "glas"]
