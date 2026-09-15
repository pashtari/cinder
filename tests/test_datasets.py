"""Masks retain their label IDs through loading and paired transforms."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from cinder.datasets import SegmentationDataset, ade20k, fives, glas


@pytest.mark.parametrize("instance_labels", [False, True])
def test_palette_mask_preserves_label_indices(tmp_path, instance_labels):
    labels = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    mask = Image.fromarray(labels).convert("P")
    # A colour palette changes visual appearance, never semantic label IDs.
    mask.putpalette([255, 255, 255, 255, 0, 0, 0, 255, 0] + [0] * 759)
    mask.save(tmp_path / "mask.png")
    Image.new("RGB", (3, 2), (255, 0, 0)).save(tmp_path / "image.png")
    dataset = SegmentationDataset(
        [{"input": tmp_path / "image.png", "target": tmp_path / "mask.png"}],
        transform=v2.RandomHorizontalFlip(p=1),
        num_classes=1 if instance_labels else 3,
        instance_labels=instance_labels,
    )

    image, target = dataset[0]

    expected = torch.tensor(labels[:, ::-1].copy(), dtype=torch.long)
    if instance_labels:
        expected = expected.unsqueeze(0)
    torch.testing.assert_close(target, expected, check_device=False)
    assert image.dtype == torch.float32
    assert image.min() == 0 and image.max() == 1


def test_instance_mask_preserves_16_bit_labels(tmp_path):
    labels = np.array([[0, 1], [256, 1000]], dtype=np.uint16)
    Image.fromarray(labels).save(tmp_path / "mask.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    dataset = SegmentationDataset(
        [{"input": tmp_path / "image.png", "target": tmp_path / "mask.png"}],
        instance_labels=True,
    )

    _, target = dataset[0]

    torch.testing.assert_close(
        target, torch.tensor(labels.astype(np.int64)).unsqueeze(0)
    )


@pytest.mark.parametrize("mode", ["RGB", "I;16"])
def test_binary_masks_support_rgb_and_16_bit_images(tmp_path, mode):
    labels = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    mask = Image.fromarray(labels).convert(mode)
    mask.save(tmp_path / "mask.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    dataset = SegmentationDataset(
        [{"input": tmp_path / "image.png", "target": tmp_path / "mask.png"}],
        transform=v2.RandomHorizontalFlip(p=1),
    )

    _, target = dataset[0]

    expected = torch.tensor(labels[:, ::-1].copy() > 0).float().unsqueeze(0)
    torch.testing.assert_close(target, expected)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_benchmark_datalists_pair_images_with_their_masks(tmp_path):
    # Folder names that recur in the root must not confuse the target paths.
    root = tmp_path / "images" / "Original"
    _touch(root / "ADE20K" / "images" / "validation" / "ADE_val_1.jpg")
    _touch(root / "FIVES" / "test" / "Original" / "7_A.png")
    for name in ("testA_2.bmp", "testA_2_anno.bmp", "testB_1.bmp", "train_3.bmp"):
        _touch(root / "GlaS" / name)

    (ade,) = ade20k.create_datalist(root / "ADE20K", "validation")
    assert ade["target"] == str(
        root / "ADE20K" / "annotations" / "validation" / "ADE_val_1.png"
    )
    (fives_sample,) = fives.create_datalist(root / "FIVES", "test")
    assert fives_sample["target"] == str(
        root / "FIVES" / "test" / "Ground truth" / "7_A.png"
    )

    test = glas.create_datalist(root / "GlaS", "test")
    assert [sample["id"] for sample in test] == ["testA_2", "testB_1"]
    assert test[0]["target"] == str(root / "GlaS" / "testA_2_anno.bmp")
    assert [s["id"] for s in glas.create_datalist(root / "GlaS", "TESTA")] == [
        "testA_2"
    ]
    with pytest.raises(ValueError, match="Unknown split"):
        glas.create_datalist(root / "GlaS", "validation")
