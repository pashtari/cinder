"""CINDER: Conditioned Implicit Neural DecodeR for dense prediction."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .modulators import MODULATORS
from .utils import ModuleSpec, build_module, check_sizes

__all__ = ["CINDER", "create_coordinates"]


def create_coordinates(size: tuple[int, ...]) -> Tensor:
    """Return an ``(*size, len(size))`` grid of ``ij`` coordinates in ``[-1, 1]``.

    The endpoints are pixel centers, matching
    :class:`~cinder.models.modulators.GridSampler` with ``align_corners=True``.
    """
    axes = [torch.linspace(-1.0, 1.0, length) for length in size]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


class CINDER(nn.Module):
    """Decode every pixel with an INR conditioned on the image's encoder maps.

    The encoder turns an image into feature maps ``z``, and the modulator decodes
    each pixel coordinate ``x`` as ``y = INR_theta(z)(phi(x, z))``. With
    ``out_size``, it decodes the coordinates of another grid instead, such as a
    CT volume from its X-rays.

    Components are passed as factories so that CINDER can size them: the encoder
    is built with ``in_channels``, and the modulator with the INR spec, the
    coordinate dimension, ``out_channels`` and the encoder's output shapes. Leave
    these sizes out of the specs.

    Args:
        in_channels: Number of image channels.
        out_channels: Number of predicted channels; logits unless the INR has
            an output activation.
        encoder: Encoder factory. The encoder must provide
            ``get_output_shapes(in_size)`` and return a tuple of feature maps.
        inr: INR spec, built by the modulator.
        modulator: Modulator spec from
            :data:`~cinder.models.modulators.MODULATORS`; compose several with a
            :class:`~cinder.models.modulators.ListModulators` factory.
        in_size: Spatial size ``(H, W)`` of each input image or inference window.
        out_size: Size of the decoded coordinate grid, ``in_size`` by default. A
            grid with other axes than the image, such as a volume, needs a
            modulator that maps its coordinates to the maps, as ``sample_at``
            does.
        freeze_encoder: Freeze the encoder and keep it in evaluation mode.
        sampling_ratio: Fraction of the grid's coordinates decoded per training step.
        chunk_size: Coordinates the modulator decodes at a time, all of them by
            default; chunks bound the memory of large grids, such as volumes.

    Shape:
        - Input: ``(B, in_channels, H, W)``.
        - Output: ``(B, out_channels, *out_size)``, or ``(B, out_channels, N)``
          for the ``N`` coordinates sampled in training when
          ``sampling_ratio < 1``.

    Attributes:
        sample_indices: Flat indices of the coordinates decoded by the last
            training forward, or ``None`` when every pixel was decoded. Read by
            :class:`~cinder.engine.losses.SampledLoss`.
    """

    coords: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: ModuleSpec,
        inr: ModuleSpec,
        modulator: ModuleSpec,
        in_size: tuple[int, int] = (512, 512),
        out_size: Sequence[int] | None = None,
        freeze_encoder: bool = False,
        sampling_ratio: float = 1.0,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = tuple(in_size)
        if len(self.in_size) != 2 or any(size < 1 for size in self.in_size):
            raise ValueError(
                f"in_size must contain two positive dimensions, got {in_size}"
            )
        self.out_size = self.in_size if out_size is None else tuple(out_size)
        if not self.out_size or any(size < 1 for size in self.out_size):
            raise ValueError(
                f"out_size must contain positive dimensions, got {out_size}"
            )
        if not 0.0 < sampling_ratio <= 1.0:
            raise ValueError(f"sampling_ratio must be in (0, 1], got {sampling_ratio}")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        # Check the specs before building a possibly pretrained encoder.
        for name, spec in (
            ("encoder", encoder),
            ("inr", inr),
            ("modulator", modulator),
        ):
            if isinstance(spec, nn.Module):
                raise TypeError(
                    f"`{name}` must be a factory returning a fresh nn.Module "
                    "so its sizes can be inferred"
                )

        self.encoder = build_module(encoder, in_channels=in_channels)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        cond_shape = self.encoder.get_output_shapes(self.in_size)
        sizes = dict(
            in_features=len(self.out_size),
            out_features=out_channels,
            cond_shape=tuple(tuple(shape) for shape in cond_shape),
        )
        self.modulator = build_module(modulator, MODULATORS, inr=inr, **sizes)
        check_sizes(self.modulator, "modulator", **sizes)

        self.sampling_ratio = sampling_ratio
        self.chunk_size = chunk_size
        self.register_buffer("coords", create_coordinates(self.out_size))
        self.sample_indices: Tensor | None = None

    def train(self, mode: bool = True) -> "CINDER":
        """Set the training mode, keeping a frozen encoder in evaluation mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _sample_coords(self) -> tuple[Tensor, Tensor]:
        """Sample query coordinates and their flat indices into the target."""
        flat_coords = self.coords.flatten(0, -2)
        num_coords = flat_coords.shape[0]
        num_samples = max(1, int(self.sampling_ratio * num_coords))
        indices = torch.randperm(num_coords, device=flat_coords.device)[:num_samples]
        return flat_coords[indices], indices

    def forward(self, x: Tensor) -> Tensor:
        """Predict the whole grid, or the sampled coordinates during training."""
        if x.ndim != 4:
            raise ValueError(f"`x` must be 4D, got {x.ndim}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"`x` must have {self.in_channels} channels, got {x.shape[1]}"
            )
        # The encoder may map several input sizes to the same feature shapes,
        # but the coordinate grid only fits one.
        if tuple(x.shape[2:]) != self.in_size:
            raise ValueError(
                f"`x` spatial size must be {self.in_size}, got {tuple(x.shape[2:])}"
            )

        feature_maps = self.encoder(x)
        if self.training and self.sampling_ratio < 1.0:
            coords, self.sample_indices = self._sample_coords()
        else:
            coords = self.coords
            self.sample_indices = None
        # Decode flat queries shared by the batch, (1, M, D), so full grids and
        # sampled coordinates take the same kernels.
        *query_shape, dim = coords.shape
        queries = coords.reshape(1, -1, dim)
        if self.chunk_size is None:
            out = self.modulator(queries, feature_maps)  # (B, M, C)
        else:
            chunks = queries.split(self.chunk_size, dim=1)
            out = torch.cat([self.modulator(q, feature_maps) for q in chunks], dim=1)
        return out.reshape(x.shape[0], *query_shape, self.out_channels).movedim(-1, 1)
