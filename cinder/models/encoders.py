import timm
import torch
from torch import Tensor, nn


class BaseEncoder(nn.Module):
    """Base class for image feature encoders.

    An encoder maps an image ``(B, in_channels, H, W)`` to a spatial feature map
    ``(B, embed_dim, H', W')``. Subclasses implement :meth:`forward`; the shared
    :meth:`get_output_size` reports the feature-map size produced for a given
    input resolution.

    Args:
        in_channels: Number of input image channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in_channels, H, W)`` -> ``(B, embed_dim, H', W')``."""
        raise NotImplementedError

    @torch.no_grad()
    def get_output_size(self, in_size: tuple[int, int]) -> tuple[int, ...]:
        """Return the per-sample output size ``(embed_dim, H', W')`` for an ``in_size`` input.

        Runs a dry forward pass in eval mode -- so it neither tracks gradients
        nor perturbs normalization running statistics -- on a zero image matching
        the encoder's device and ``in_channels``.
        """
        was_training = self.training
        self.eval()
        try:
            param = next(self.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
            dummy = torch.zeros(1, self.in_channels, *in_size, device=device)
            out_size = tuple(int(s) for s in self(dummy).shape[1:])
        finally:
            self.train(was_training)
        return out_size


class ChannelNorm(nn.Module):
    """Learnable per-channel normalization layer.

    Initialized to ImageNet standard normalization:
        ``(x - mean) / std``
    with ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224, 0.225]``.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, num_channels: int = 3):
        super().__init__()
        # Cycle the ImageNet stats so any channel count is supported (identical
        # to slicing for the usual num_channels <= 3).
        mean = [self.MEAN[i % len(self.MEAN)] for i in range(num_channels)]
        std = [self.STD[i % len(self.STD)] for i in range(num_channels)]
        alpha_init = torch.tensor([1.0 / s for s in std]).reshape(1, num_channels, 1, 1)
        beta_init = torch.tensor([-m / s for m, s in zip(mean, std)]).reshape(
            1, num_channels, 1, 1
        )
        self.alpha = nn.Parameter(alpha_init)
        self.beta = nn.Parameter(beta_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class TimmEncoder(BaseEncoder):
    """Feature encoder wrapping any pretrained ``timm`` model.

    Args:
        model_name: Name of a timm model.
        pretrained: Whether to load pretrained weights.
        normalize: Whether to apply learnable LayerNorm to output features.
        out_index: If set, use ``features_only`` mode and return the feature
            map at this stage index (0 = shallowest / least downsampling).
            When ``None``, use the model's ``forward_features`` (final stage).
        **model_kwargs: Extra keyword arguments forwarded to
            ``timm.create_model`` (e.g. ``patch_size``, ``img_size``).
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        normalize: bool = True,
        out_index: int | None = None,
        in_channels: int = 3,
        **model_kwargs,
    ):
        super().__init__(in_channels)
        self.model_name = model_name
        self.normalize = normalize
        self.out_index = out_index

        # Learnable per-channel normalization
        self.channel_norm = ChannelNorm(num_channels=in_channels)

        if out_index is not None:
            # Extract intermediate features at a specific stage
            self.encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=[out_index],
                in_chans=in_channels,
                **model_kwargs,
            )
            # feature_info gives channel dims per stage
            self._embed_dim = self.encoder.feature_info.channels()[0]
        else:
            # Use full model without classification head
            self.encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_channels,
                **model_kwargs,
            )
            self._embed_dim = self.encoder.num_features

        # Learnable feature normalization
        if normalize:
            self.layer_norm = nn.LayerNorm(self._embed_dim)

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension of the model."""
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a spatial feature map from input images.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Feature map of shape ``(B, embed_dim, H', W')``.
        """
        # Apply learnable channel normalization
        x = self.channel_norm(x)

        if self.out_index is not None:
            # features_only mode returns a list of feature maps
            features = self.encoder(x)[0]  # (B, C, H', W')
        else:
            # forward_features returns a spatial map (CNN) or tokens (ViT)
            features = self.encoder.forward_features(x)

        # Canonicalize ViT token sequences (B, N, C) into a spatial grid.
        if features.ndim == 3:
            prefix = getattr(self.encoder, "num_prefix_tokens", 0)
            if prefix:  # drop CLS / distillation tokens
                features = features[:, prefix:]
            b, n, c = features.shape
            grid = getattr(getattr(self.encoder, "patch_embed", None), "grid_size", None)
            h, w = grid if grid is not None else (int(n**0.5), int(n**0.5))
            if h * w != n:
                raise RuntimeError(
                    f"cannot reshape {n} tokens into a spatial grid: the encoder "
                    f"exposes no `patch_embed.grid_size` and {n} is not square."
                )
            features = features.transpose(1, 2).reshape(b, c, h, w)

        # Optionally normalize over the channel dim at each spatial location.
        if self.normalize:
            features = self.layer_norm(features.movedim(1, -1)).movedim(-1, 1)

        return features  # (B, embed_dim, H', W')


class DoubleConv(nn.Module):
    """(Conv -- Norm -- Act) ** 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        padding = (kernel_size - 1) // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class UNetEncoderBlock(nn.Module):
    """U-Net block for one stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        depth: int = 1,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )

        for _ in range(1, depth):
            self.blocks.append(DoubleConv(out_channels, out_channels, stride=1))

    def forward(self, x):
        out = self.blocks(x)
        return out


class UNetEncoder(BaseEncoder):
    """U-Net feature encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        depth: tuple[int, ...] = (1, 1, 1, 1, 1),
        strides: tuple[int, ...] = (1, 2, 2, 2, 2),
        kernel_size: int = 3,
    ):
        super().__init__(in_channels)
        channels = [in_channels, *out_channels]
        self.blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            self.blocks.append(
                UNetEncoderBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=strides[i],
                    depth=depth[i],
                )
            )

    def forward(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)

        return out  # (B, embed_dim, H', W')
