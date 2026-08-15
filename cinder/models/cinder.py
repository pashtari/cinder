from typing import Optional

import torch
from torch import Tensor, nn

from .utils import ModuleSpec, build_module


def create_coordinates(
    size: tuple[int, ...], coord_range: tuple[float, float] = (-1.0, 1.0), **kwargs
) -> Tensor:
    """
    Create a tensor of coordinates for a given size.

    Args:
        size (tuple[int, ...]): Shape of the grid (any number of dimensions)
        coord_range (tuple[float, float], optional): Range to normalize coordinates
            to. Defaults to (-1, 1)
        **kwargs: Additional arguments passed to torch.linspace

    Returns:
        Tensor: Coordinate tensor of shape [*, P], where * denotes any number of
            spatial dimensions and P is the number of coordinates.
    """
    min_value, max_value = coord_range
    coordinates = []
    for s in size:
        coordinates.append(torch.linspace(min_value, max_value, s, **kwargs))

    coordinates = torch.meshgrid(*coordinates, indexing="ij")
    coordinates = torch.stack(coordinates, dim=-1)  # (*, P)
    return coordinates


class CINDER(nn.Module):
    """CINDER: Conditioned Implicit Neural DecodeR for Medical Image Segmentation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: ModuleSpec,
        cinr: ModuleSpec,
        in_size: tuple[int, int] = (384, 384),
        freeze_encoder: bool = False,
        sampling_ratio: float = 1.0,
    ) -> None:
        """
        Args:
            in_channels: Number of input image channels.
            out_channels: Number of segmentation output channels (classes).
            encoder: Spec for the feature encoder (see
                :func:`~cinder.models.utils.build_module`), built as
                ``encoder(in_channels=...)`` — e.g. a Hydra ``_partial_: true``
                node, ``functools.partial(TimmEncoder, ...)``, or a
                ``(factory, params)`` pair.
            cinr: Spec for the conditional INR, built as
                ``cinr(in_features=..., out_features=..., cond_dim=...)`` —
                the spatial rank, the number of output channels, and the
                encoder's channel count, all injected here. Leave these unset
                in the spec: a partial silently loses its own value to the
                injected one, and a ``(factory, params)`` pair raises a
                duplicate-argument ``TypeError``.
            in_size: Spatial size ``(H, W)`` of the input the model is wired for.
            freeze_encoder: Freeze the encoder's weights.
            sampling_ratio: Fraction of coordinates decoded per training step.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = tuple(in_size)

        assert (
            0.0 < sampling_ratio <= 1.0
        ), f"sampling_ratio must be in (0, 1], got {sampling_ratio}"

        # Both are built here so CINDER can inject the sizes it owns; a
        # ready-made module would bypass that, so instances are rejected.
        for name, spec in (("encoder", encoder), ("cinr", cinr)):
            if isinstance(spec, nn.Module):
                raise TypeError(
                    f"`{name}` must be a factory returning a fresh nn.Module (a "
                    f"Hydra `_partial_: true` node or `functools.partial(...)`), "
                    f"not an nn.Module instance, so its sizes can be wired here."
                )

        self.encoder = build_module(encoder, in_channels=in_channels)

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)

        # The encoder produces a spatial feature map (C, H', W') that the CINR
        # conditions on directly (interpolating it at each coordinate), so its
        # channel count *is* the CINR's `cond_dim`. Probing the encoder first
        # lets that be injected rather than kept in sync by hand.
        self.encoder_out_size = self.encoder.get_output_size(self.in_size)

        self.cinr = build_module(
            cinr,
            in_features=len(self.in_size),
            out_features=out_channels,
            cond_dim=self.encoder_out_size[0],
        )

        self.sampling_ratio = sampling_ratio

        coords = create_coordinates(self.in_size, coord_range=(-1.0, 1.0))  # (H, W, 2)
        self.register_buffer("coords", coords)

        self._sample_indices: Optional[Tensor] = None

    def train(self, mode: bool = True) -> "CINDER":
        super().train(mode)
        if self.freeze_encoder:
            # Keep frozen encoder in eval mode so norm layers use pretrained
            # running statistics instead of (small) batch statistics.
            self.encoder.eval()
        return self

    def _sample_coords(self, coords: Tensor, ratio: float) -> tuple[Tensor, Tensor]:
        """Randomly sample a subset of coordinates.

        Args:
            coords: Coordinates tensor, e.g., (H, W, 2).
            ratio: Sampling ratio.

        Returns:
            sampled_coords: Sampled coordinates (N, 2).
            indices: Flat indices of sampled coordinates (N,).
        """
        coords_flat = torch.flatten(coords, 0, -2)  # (N, P)
        K = max(1, int(ratio * coords_flat.shape[0]))
        indices = torch.randperm(coords_flat.shape[0], device=coords_flat.device)[:K]
        return coords_flat[indices], indices

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C_in, H, W)
        if x.ndim != 4:
            raise ValueError(f"`x` must be 4D, got {x.ndim}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"`x` must have {self.in_channels} channels, got {x.shape[1]}"
            )

        # Extract a spatial feature map from the encoder
        feats = self.encoder(x)  # (B, C, H', W')
        if tuple(feats.shape[1:]) != tuple(self.encoder_out_size):
            raise RuntimeError(
                f"Encoder output size must be {self.encoder_out_size}, got {tuple(feats.shape[1:])}"
            )

        if self.sampling_ratio < 1.0 and self.training:
            coords, self._sample_indices = self._sample_coords(
                self.coords, self.sampling_ratio
            )  # (N, 2), (N,)
        else:
            coords = self.coords  # (H, W, 2)
            self._sample_indices = None

        out = self.cinr(coords, feats)  # (B, N, C_out) or (B, H, W, C_out)
        return out.movedim(-1, 1)  # (B, C_out, N) or (B, C_out, H, W)
