from typing import Callable

import torch
from torch import Tensor


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Return tile start positions ensuring full coverage of *length*."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _gaussian_window_2d(size: tuple[int, int], sigma_scale: float = 0.125) -> Tensor:
    """Create a 2-D Gaussian importance map (outer product of two 1-D Gaussians)."""
    h, w = size
    sigma_h = h * sigma_scale
    sigma_w = w * sigma_scale
    coords_h = torch.arange(h, dtype=torch.float32) - (h - 1) / 2.0
    coords_w = torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0
    g_h = torch.exp(-0.5 * (coords_h / sigma_h) ** 2)
    g_w = torch.exp(-0.5 * (coords_w / sigma_w) ** 2)
    window = g_h[:, None] * g_w[None, :]  # (H, W)
    window /= window.max()
    return window


class SlidingWindowInferer:
    """Sliding-window inference with blending for 2-D inputs.

    Args:
        roi_size: Spatial size ``(rH, rW)`` of each tile.
        sw_batch_size: Maximum tiles per forward pass.
        overlap: Fraction of overlap between adjacent tiles (0–1).
        mode: ``"gaussian"`` for Gaussian blending, ``"constant"`` for uniform.
        sigma_scale: Controls the width of the Gaussian window (fraction of tile size).
    """

    def __init__(
        self,
        roi_size: tuple[int, int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: str = "gaussian",
        sigma_scale: float = 0.125,
        amp: bool = False,
    ) -> None:
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.amp = amp

    def __call__(
        self, inputs: Tensor, predictor: Callable[..., Tensor], **kwargs
    ) -> Tensor:
        """Run *predictor* on overlapping tiles and stitch results with blending.

        Args:
            inputs: Input tensor ``(B, C, H, W)``.
            predictor: Callable that maps ``(B, C, rH, rW) -> (B, C_out, rH, rW)``.
            **kwargs: Extra keyword arguments forwarded to *predictor*.

        Returns:
            Blended prediction ``(B, C_out, H, W)``.
        """
        B, _, H, W = inputs.shape
        rH, rW = self.roi_size

        # Zero-pad if the image is smaller than the tile size
        pad_h = max(rH - H, 0)
        pad_w = max(rW - W, 0)
        if pad_h > 0 or pad_w > 0:
            inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h))
        _, _, pH, pW = inputs.shape  # padded spatial dims

        # Compute tile start positions
        stride_h = max(1, int(rH * (1.0 - self.overlap)))
        stride_w = max(1, int(rW * (1.0 - self.overlap)))

        starts_h = _tile_starts(pH, rH, stride_h)
        starts_w = _tile_starts(pW, rW, stride_w)

        # Importance weight map
        if self.mode == "gaussian":
            weight = _gaussian_window_2d((rH, rW), self.sigma_scale).to(
                inputs.device, inputs.dtype
            )
        else:
            weight = torch.ones(rH, rW, device=inputs.device, dtype=inputs.dtype)

        # Collect all (y, x) tile origins
        origins = [(y, x) for y in starts_h for x in starts_w]

        # Determine autocast context
        use_amp = self.amp and inputs.is_cuda
        amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else torch.no_grad()

        # Run first tile to discover output channels
        y0, x0 = origins[0]
        tile0 = inputs[:, :, y0 : y0 + rH, x0 : x0 + rW]
        with amp_ctx:
            pred0 = predictor(tile0, **kwargs)
        pred0 = pred0.float()
        C_out = pred0.shape[1]

        # Allocate output buffers
        out = inputs.new_zeros((B, C_out, pH, pW))
        count = inputs.new_zeros((1, 1, pH, pW))

        # Write first tile
        out[:, :, y0 : y0 + rH, x0 : x0 + rW] += pred0 * weight
        count[:, :, y0 : y0 + rH, x0 : x0 + rW] += weight
        del pred0

        # Process remaining tiles in batches
        remaining = origins[1:]
        for i in range(0, len(remaining), self.sw_batch_size):
            batch_origins = remaining[i : i + self.sw_batch_size]
            tiles = torch.cat(
                [inputs[:, :, y : y + rH, x : x + rW] for y, x in batch_origins],
                dim=0,
            )  # (K*B, C, rH, rW) where K = len(batch_origins)
            with amp_ctx:
                preds = predictor(tiles, **kwargs)  # (K*B, C_out, rH, rW)
            preds = preds.float()
            del tiles

            # Scatter predictions back — reshape to (K, B, C_out, rH, rW)
            K = len(batch_origins)
            preds = preds.view(K, B, C_out, rH, rW)
            for j, (y, x) in enumerate(batch_origins):
                out[:, :, y : y + rH, x : x + rW] += preds[j] * weight
                count[:, :, y : y + rH, x : x + rW] += weight
            del preds
            if use_amp:
                torch.cuda.empty_cache()

        out /= count
        # Crop back to original spatial size
        return out[:, :, :H, :W]


class MultiScaleFlipInferer:
    """Test-time augmentation over scales and horizontal flips.

    Wraps another inferer (typically :class:`SlidingWindowInferer`) and averages
    its predictions over rescaled and mirrored copies of the input. This is the
    "MS+flip" column that ADE20K results are conventionally reported with,
    alongside the single-scale number -- the two are not interchangeable, and a
    comparison must use the same setting for every method.

    Predictions are averaged as **logits**, matching what the metrics in
    :mod:`cinder.engine.metrics` expect (they apply sigmoid or argmax
    themselves). Averaging probabilities instead is a defensible alternative
    convention; it is not what this does.

    Args:
        inferer: The base inferer to wrap.
        scales: Resize factors applied to the input. Must include 1.0 to keep
            the un-scaled view.
        flip: Also evaluate a horizontally mirrored copy of each scale.
        align_corners: Passed to the interpolation, both down and back up.
    """

    def __init__(
        self,
        inferer: Callable[..., Tensor],
        scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        flip: bool = True,
        align_corners: bool = False,
    ) -> None:
        self.inferer = inferer
        self.scales = tuple(scales)
        self.flip = flip
        self.align_corners = align_corners

    def __call__(
        self, inputs: Tensor, predictor: Callable[..., Tensor], **kwargs
    ) -> Tensor:
        _, _, H, W = inputs.shape
        total = None
        views = 0

        for scale in self.scales:
            if scale == 1.0:
                scaled = inputs
            else:
                size = (max(1, round(H * scale)), max(1, round(W * scale)))
                scaled = torch.nn.functional.interpolate(
                    inputs, size=size, mode="bilinear", align_corners=self.align_corners
                )

            for mirrored in (False, True) if self.flip else (False,):
                view = torch.flip(scaled, dims=[-1]) if mirrored else scaled
                pred = self.inferer(view, predictor, **kwargs)
                if mirrored:
                    pred = torch.flip(pred, dims=[-1])
                if pred.shape[-2:] != (H, W):
                    pred = torch.nn.functional.interpolate(
                        pred,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                total = pred if total is None else total + pred
                views += 1

        return total / views
