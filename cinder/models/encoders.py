"""Image encoders that produce spatial features for CINDER."""

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any

import timm
import torch
from timm.layers import Format
from torch import Tensor, nn

__all__ = ["BaseEncoder", "ChannelNorm", "TimmEncoder"]


class BaseEncoder(nn.Module):
    """Base class for image encoders that return spatial feature maps.

    Subclasses implement :meth:`forward`, which returns a tuple of maps shaped
    ``(B, embed_dims[i], H_i, W_i)`` even for a single stage, and may set
    :attr:`embed_dims`. CINDER sizes its modulator from :meth:`get_output_shapes`.

    Args:
        in_channels: Number of input image channels.

    Attributes:
        embed_dims: Channel count of each returned feature map.
    """

    embed_dims: tuple[int, ...]

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Map an image batch to a tuple of ``(B, C_i, H_i, W_i)`` features."""
        raise NotImplementedError

    @torch.no_grad()
    def get_output_shapes(
        self, in_size: tuple[int, int]
    ) -> tuple[tuple[int, ...], ...]:
        """Return the ``(C_i, H_i, W_i)`` map shapes for an ``(H, W)`` input.

        The shapes come from a dry run on the encoder's device and dtype. It
        runs in eval mode so running statistics are untouched, and restores
        every submodule's training mode afterward.
        """
        training_modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            reference = next(chain(self.parameters(), self.buffers()), torch.empty(0))
            dummy = torch.zeros(
                1,
                self.in_channels,
                *in_size,
                device=reference.device,
                dtype=reference.dtype,
            )
            return tuple(
                tuple(int(size) for size in features.shape[1:])
                for features in self(dummy)
            )
        finally:
            for module, training in training_modes.items():
                module.training = training


class ChannelNorm(nn.Module):
    """Learnable per-channel affine map initialized to ``(x - mean) / std``.

    Defaults to ImageNet statistics; :class:`TimmEncoder` passes the backbone's
    pretrained statistics. Statistics repeat cyclically for extra channels.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_channels: int = 3,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        mean = self.MEAN if mean is None else mean
        std = self.STD if std is None else std
        mean = [mean[i % len(mean)] for i in range(num_channels)]
        std = [std[i % len(std)] for i in range(num_channels)]
        scale = torch.tensor([1.0 / s for s in std]).view(1, num_channels, 1, 1)
        offset = torch.tensor([-m / s for m, s in zip(mean, std)])
        offset = offset.view(1, num_channels, 1, 1)
        self.alpha = nn.Parameter(scale)
        self.beta = nn.Parameter(offset)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize an image batch shaped ``(B, C, H, W)``."""
        return self.alpha * x + self.beta


class TimmEncoder(BaseEncoder):
    """Feature maps of a timm CNN or transformer as ``(B, C_i, H_i, W_i)`` tensors.

    Maps are returned in stage order at their native resolution and described by
    ``out_indices``, ``embed_dims`` and ``feature_reductions``. Channels-last
    backbones (e.g. Swin) are converted to channels-first.

    Args:
        model_name: Backbone with timm ``features_only`` support.
        pretrained: Load pretrained backbone weights.
        normalize: Apply a learnable channel LayerNorm to each output map.
            Input normalization always starts from the backbone's mean/std.
        in_channels: Number of image channels.
        **model_kwargs: Passed to timm, including ``out_indices`` (default:
            deepest level) and ``img_size`` for fixed-size transformers.
    """

    output_fmt = Format.NCHW

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        normalize: bool = True,
        in_channels: int = 3,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(in_channels)
        self.model_name = model_name
        self.normalize = normalize
        if model_kwargs.get("patch_drop_rate", 0):
            raise ValueError(
                "patch_drop_rate must be 0 to preserve dense spatial grids"
            )

        model_kwargs.setdefault("out_indices", (-1,))
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            **model_kwargs,
        )
        config = getattr(backbone, "pretrained_cfg", {})
        # Registered before the backbone, which fixes the optimizer state order.
        self.channel_norm = ChannelNorm(
            in_channels, mean=config.get("mean"), std=config.get("std")
        )
        self.backbone = backbone

        # timm emits stages in execution order but keeps its metadata in request
        # order, so resolve and sort the indices.
        info = backbone.feature_info
        num_stages = len(info)
        if not info.out_indices or any(
            not -num_stages <= index < num_stages for index in info.out_indices
        ):
            raise ValueError(
                f"out_indices must select levels in [-{num_stages}, {num_stages - 1}]"
            )
        self.out_indices = tuple(
            sorted({index % num_stages for index in info.out_indices})
        )
        self.embed_dims = tuple(info.channels(self.out_indices))
        self.feature_reductions = tuple(info.reduction(self.out_indices))

        native_format = getattr(backbone, "output_fmt", None) or Format.NCHW
        if native_format not in (Format.NCHW, Format.NHWC):
            raise ValueError(
                f"unsupported feature format {native_format!r}; expected NCHW or NHWC"
            )
        self.channels_last = native_format == Format.NHWC
        if normalize:
            self.layer_norms = nn.ModuleList(
                nn.LayerNorm(channels) for channels in self.embed_dims
            )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"input must have shape (B, {self.in_channels}, H, W)")
        features = self.backbone(self.channel_norm(x))
        if isinstance(features, Mapping):
            features = tuple(features.values())
        if not isinstance(features, (list, tuple)):
            raise TypeError("expected a sequence of spatial feature maps")
        if len(features) != len(self.embed_dims):
            raise RuntimeError(
                f"expected {len(self.embed_dims)} feature maps, got {len(features)}"
            )
        return tuple(
            self._prepare_feature(feature, i, x.shape[0])
            for i, feature in enumerate(features)
        )

    def _prepare_feature(self, feature: Tensor, index: int, batch_size: int) -> Tensor:
        """Check a map, move it to NCHW and optionally normalize its channels."""
        if not isinstance(feature, Tensor) or feature.ndim != 4:
            raise ValueError(f"feature {index} must be a 4D spatial map")
        if self.channels_last:
            feature = feature.movedim(-1, 1)
        channels = self.embed_dims[index]
        if feature.shape[:2] != (batch_size, channels):
            raise ValueError(
                f"feature {index} must have batch size {batch_size} and {channels} "
                f"channels, got shape {tuple(feature.shape)}"
            )
        if self.normalize:
            feature = self.layer_norms[index](feature.movedim(1, -1)).movedim(-1, 1)
        return feature
