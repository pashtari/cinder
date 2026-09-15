"""Tiled inference and test-time augmentation for dense predictions."""

import math
from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SlidingWindowInferer", "MultiScaleFlipInferer"]


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Return tile offsets along one axis; the last tile ends at the edge."""
    if length <= tile:
        return [0]
    return [*range(0, length - tile, stride), length - tile]


def _gaussian_window_2d(size: tuple[int, int], sigma_scale: float = 0.125) -> Tensor:
    """Return separable Gaussian tile weights with a peak of one.

    Weights are clamped to float32 ``eps``, so the blend's denominator stays
    positive even at tile corners.
    """
    height, width = size
    sigma_h = height * sigma_scale
    sigma_w = width * sigma_scale
    coords_h = torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
    coords_w = torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0
    log_h = -0.5 * (coords_h / sigma_h) ** 2
    log_w = -0.5 * (coords_w / sigma_w) ** 2
    window = torch.exp((log_h - log_h.max())[:, None] + (log_w - log_w.max())[None, :])
    return window.clamp_min(torch.finfo(window.dtype).eps)


class SlidingWindowInferer:
    """Predict an image tile by tile and blend the overlapping predictions.

    Tiles start every ``int(roi_size * (1 - overlap))`` pixels, plus a final
    tile flush with the bottom and right edges. An image smaller than a tile is
    zero-padded at the bottom and right, like training's padded random crops, and
    the prediction is cropped back. Overlaps are blended per pixel as
    ``sum(w * p) / sum(w)`` in float32. Argument names follow MONAI's inferer.

    Args:
        roi_size: Tile size ``(height, width)``.
        sw_batch_size: Number of tile positions per predictor call; each call
            holds one tile per input image.
        overlap: Fraction of a tile shared with its neighbor, in ``[0, 1)``.
        mode: ``"gaussian"`` weights tile centers most, which hides seams;
            ``"constant"`` weights every pixel equally.
        sigma_scale: Gaussian standard deviation as a fraction of the tile size.
        amp: Run the predictor under CUDA autocast (float16); blending stays in
            float32. Ignored on CPU.
    """

    def __init__(
        self,
        roi_size: Sequence[int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Literal["gaussian", "constant"] = "gaussian",
        sigma_scale: float = 0.125,
        amp: bool = False,
    ) -> None:
        if len(roi_size) != 2 or any(size <= 0 for size in roi_size):
            raise ValueError("roi_size must contain two positive dimensions.")
        if sw_batch_size < 1:
            raise ValueError("sw_batch_size must be positive.")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1).")
        if mode not in {"gaussian", "constant"}:
            raise ValueError("mode must be 'gaussian' or 'constant'.")
        if not math.isfinite(sigma_scale) or sigma_scale <= 0:
            raise ValueError("sigma_scale must be finite and positive.")
        self.roi_size = (roi_size[0], roi_size[1])
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.amp = amp

    @torch.no_grad()
    def __call__(self, inputs: Tensor, predictor: Callable[[Tensor], Tensor]) -> Tensor:
        """Predict ``(B, C, H, W)`` inputs as float32 ``(B, C_out, H, W)`` outputs.

        ``predictor`` maps a batch of tiles ``(N, C, h, w)`` to ``(N, C_out, h, w)``
        in the same order.
        """
        batch_size, _, height, width = inputs.shape
        tile_height, tile_width = self.roi_size

        pad_h = max(tile_height - height, 0)
        pad_w = max(tile_width - width, 0)
        if pad_h > 0 or pad_w > 0:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h))
        _, _, padded_height, padded_width = inputs.shape

        stride_h = max(1, int(tile_height * (1.0 - self.overlap)))
        stride_w = max(1, int(tile_width * (1.0 - self.overlap)))
        starts_h = _tile_starts(padded_height, tile_height, stride_h)
        starts_w = _tile_starts(padded_width, tile_width, stride_w)

        # Blend in float32 even for half-precision predictions: small Gaussian
        # weights would underflow and summed overlaps could overflow.
        if self.mode == "gaussian":
            weight = _gaussian_window_2d((tile_height, tile_width), self.sigma_scale)
            weight = weight.to(inputs.device)
        else:
            weight = torch.ones((tile_height, tile_width), device=inputs.device)

        origins = [(y, x) for y in starts_h for x in starts_w]
        use_amp = self.amp and inputs.is_cuda
        output: Tensor | None = None
        weight_sum = torch.zeros(
            (1, 1, padded_height, padded_width),
            device=inputs.device,
            dtype=torch.float32,
        )

        for start in range(0, len(origins), self.sw_batch_size):
            batch_origins = origins[start : start + self.sw_batch_size]
            tiles = torch.cat(
                [
                    inputs[:, :, y : y + tile_height, x : x + tile_width]
                    for y, x in batch_origins
                ],
                dim=0,
            )
            with torch.autocast("cuda", enabled=use_amp):
                preds = predictor(tiles).float()
            if output is None:
                output = preds.new_zeros(
                    (batch_size, preds.shape[1], padded_height, padded_width)
                )
            preds = preds.reshape(
                len(batch_origins), batch_size, preds.shape[1], tile_height, tile_width
            )
            for pred, (y, x) in zip(preds, batch_origins):
                output[:, :, y : y + tile_height, x : x + tile_width] += pred * weight
                weight_sum[:, :, y : y + tile_height, x : x + tile_width] += weight

        assert output is not None  # there is always at least one tile
        output /= weight_sum
        return output[:, :, :height, :width]


class MultiScaleFlipInferer:
    """Average predictions over rescaled and horizontally flipped views.

    Each view runs through ``inferer`` and is mapped back to the input size and
    orientation; the average is taken over logits.

    Args:
        inferer: Inferer that predicts each view, typically a
            :class:`SlidingWindowInferer`.
        scales: Positive resize factors. Include ``1.0`` for the original view.
        flip: Also predict a horizontally mirrored copy of every scale.
        align_corners: Passed to the bilinear resizing in both directions.
    """

    def __init__(
        self,
        inferer: Callable[[Tensor, Callable[[Tensor], Tensor]], Tensor],
        scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        flip: bool = True,
        align_corners: bool = False,
    ) -> None:
        self.inferer = inferer
        self.scales = tuple(scales)
        if not self.scales or any(
            not math.isfinite(scale) or scale <= 0 for scale in self.scales
        ):
            raise ValueError("scales must contain finite, positive resize factors.")
        self.flip = flip
        self.align_corners = align_corners

    def __call__(self, inputs: Tensor, predictor: Callable[[Tensor], Tensor]) -> Tensor:
        """Average the predictions of every view of a ``(B, C, H, W)`` input."""
        _, _, height, width = inputs.shape
        total: Tensor | None = None
        views = 0

        for scale in self.scales:
            if scale == 1.0:
                scaled = inputs
            else:
                size = (max(1, round(height * scale)), max(1, round(width * scale)))
                scaled = F.interpolate(
                    inputs, size=size, mode="bilinear", align_corners=self.align_corners
                )

            for mirrored in (False, True) if self.flip else (False,):
                view = torch.flip(scaled, dims=[-1]) if mirrored else scaled
                pred = self.inferer(view, predictor)
                if mirrored:
                    pred = torch.flip(pred, dims=[-1])
                if pred.shape[-2:] != (height, width):
                    pred = F.interpolate(
                        pred,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                total = pred if total is None else total + pred
                views += 1

        assert total is not None  # scales is non-empty
        return total / views
