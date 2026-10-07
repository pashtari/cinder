"""Biplanar X-ray to CT reconstruction: the parts specific to the task.

Helpers for ``biplanar_xray.ipynb``: reading LIDC-IDRI CT series and resampling
them onto a cube, simulating posterior-anterior (PA) and lateral X-rays with
DiffDRR and fitting the camera of each view, SPIDER's 2D UNet as a CINDER
encoder of both views, and the PerX2CT baseline. The rest comes from ``cinder``.

Volumes are arrays ``(z, y, x)`` on DICOM's patient axes: ``z`` runs from the feet
to the head, ``y`` from anterior to posterior and ``x`` from the patient's right
to left. Coordinates are CINDER's, ``[-1, 1]`` per axis with the ends at the
centers of the first and last voxels, in the same ``(z, y, x)`` order.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cinder import BaseEncoder, Camera, PositionalEncoding
from cinder.models.modulators import GridSampler
from cinder.models.utils import ModuleSpec, build_module

HU_RANGE = (-1000.0, 1500.0)  # air to dense bone, mapped to [0, 1]
TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


# Data ---------------------------------------------------------------------------------


def download_lidc(
    root: str | Path,
    num_patients: int = 400,
    num_slices: tuple[int, int] = (100, 400),
    workers: int = 6,
) -> list[Path]:
    """Download LIDC-IDRI CT series from TCIA's public API, one folder per patient.

    Each patient contributes their series with the most slices, if that number
    lies within ``num_slices``; the first ``num_patients`` such patients are
    kept. Folders that already hold DICOM files are skipped.
    """
    import io
    import json
    import urllib.request
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    query = f"{TCIA}/getSeries?Collection=LIDC-IDRI&Modality=CT"
    with urllib.request.urlopen(query) as response:
        series = json.load(response)
    largest = {}
    for item in series:
        best = largest.get(item["PatientID"])
        if best is None or item["ImageCount"] > best["ImageCount"]:
            largest[item["PatientID"]] = item
    chosen = [
        largest[patient]
        for patient in sorted(largest)
        if num_slices[0] <= largest[patient]["ImageCount"] <= num_slices[1]
    ][:num_patients]

    def fetch(item: dict) -> Path:
        folder = Path(root) / item["PatientID"]
        if not any(folder.glob("*.dcm")):
            url = f"{TCIA}/getImage?SeriesInstanceUID={item['SeriesInstanceUID']}"
            with urllib.request.urlopen(url) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            folder.mkdir(parents=True, exist_ok=True)
            archive.extractall(folder)
        return folder

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fetch, chosen))


def read_series(folder: str | Path) -> tuple[np.ndarray, dict]:
    """Read an axial DICOM CT series as an HU array ``(z, y, x)`` and its geometry.

    Slices are sorted from the feet up, repeated positions are dropped, and
    images stored rotated by 180 degrees are turned back. The geometry holds each
    slice's ``z`` and the position and spacing of the pixel grid, ``(y, x)``, in
    mm.
    """
    import pydicom

    slices = [pydicom.dcmread(path) for path in sorted(Path(folder).glob("*.dcm"))]
    slices = [s for s in slices if "ImagePositionPatient" in s]
    orientation = np.round(np.asarray(slices[0].ImageOrientationPatient, dtype=float))
    sign = orientation[0]  # rows and columns along +y and +x, or both reversed
    if not np.array_equal(orientation, sign * np.array([1, 0, 0, 0, 1, 0])):
        raise ValueError(f"{folder}: expected axial slices, got {orientation}")
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    keep = np.concatenate([[True], np.diff(z) > 1e-3])
    slices, z = [s for s, k in zip(slices, keep) if k], z[keep]
    hu = np.stack(
        [
            s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept)
            for s in slices
        ]
    ).astype(np.float32)
    x0, y0 = (float(v) for v in slices[0].ImagePositionPatient[:2])
    dy, dx = (float(v) for v in slices[0].PixelSpacing)  # between rows, columns
    if sign < 0:
        hu = np.ascontiguousarray(hu[:, ::-1, ::-1])
        y0, x0 = y0 - (hu.shape[1] - 1) * dy, x0 - (hu.shape[2] - 1) * dx
    return hu, dict(z=z, origin=(y0, x0), spacing=(dy, dx))


def resample_to_cube(
    hu: np.ndarray, geometry: dict, size: int, side: float
) -> np.ndarray:
    """Resample a series onto a ``size``³ cube of ``side`` mm, centered on the scan.

    Voxels outside the scan are air. Each axis is first smoothed with a Gaussian
    of ``(step / spacing - 1) / 2`` voxels against aliasing, then interpolated
    linearly; the slices may be unevenly spaced.
    """
    from scipy import ndimage

    z, (y0, x0), (dy, dx) = geometry["z"], geometry["origin"], geometry["spacing"]
    step = side / size
    sigma = [max(0.0, (step / d - 1) / 2) for d in (np.median(np.diff(z)), dy, dx)]
    smooth = ndimage.gaussian_filter(np.maximum(hu, -1000.0), sigma)
    offsets = (np.arange(size) - (size - 1) / 2) * step
    center = (
        (z[0] + z[-1]) / 2,
        y0 + (hu.shape[1] - 1) / 2 * dy,
        x0 + (hu.shape[2] - 1) / 2 * dx,
    )
    # Fractional source indices per axis; beyond the scan they fall out of range.
    index_z = np.interp(
        center[0] + offsets, z, np.arange(len(z)), left=-2, right=len(z) + 1
    )
    index_y = (center[1] + offsets - y0) / dy
    index_x = (center[2] + offsets - x0) / dx
    grid = np.meshgrid(index_z, index_y, index_x, indexing="ij")
    return ndimage.map_coordinates(smooth, grid, order=1, cval=-1000.0)


def normalize_hu(volume: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Map HU to ``[0, 1]`` over :data:`HU_RANGE`."""
    low, high = HU_RANGE
    return ((volume - low) / (high - low)).clip(0.0, 1.0)


class BiplanarDataset(torch.utils.data.Dataset):
    """Cached cases as ``(xrays, volume)``: ``(2, h, w)`` views, PA then lateral,
    and the ``(1, s, s, s)`` normalized CT."""

    def __init__(self, files: Sequence[str | Path]) -> None:
        self.files = [Path(f) for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        case = np.load(self.files[index])
        xrays = torch.from_numpy(case["xrays"]).float()
        return xrays, torch.from_numpy(case["volume"]).float()[None]


# X-rays and cameras -------------------------------------------------------------------


class Biplane:
    """A biplanar X-ray system in DiffDRR: PA and lateral views of a CT cube.

    The sources sit ``source_distance`` mm from the cube's center, behind the
    patient for the PA view and on their right for the left lateral view, and
    the detectors ``sdd`` mm from their source, with ``height``² pixels of
    ``pixel`` mm.

    Args:
        size: Voxels per axis of the cube.
        side: Side of the cube in mm.
        source_distance: Source-to-isocenter distance in mm.
        sdd: Source-to-detector distance in mm.
        height: Detector pixels per axis.
        pixel: Detector pixel size in mm.
    """

    def __init__(
        self,
        size: int,
        side: float,
        source_distance: float,
        sdd: float,
        height: int,
        pixel: float,
    ) -> None:
        from diffdrr.pose import convert

        self.size, self.step = size, side / size
        self.sdd, self.height, self.pixel = sdd, height, pixel
        # DiffDRR's poses rotate the C-arm about the head-feet axis, here from
        # its anterior source to behind the patient and to their right.
        rotations = torch.tensor([[math.pi, 0.0, 0.0], [-math.pi / 2, 0.0, 0.0]])
        translations = torch.tensor([[0.0, source_distance, 0.0]] * 2)
        self.poses = convert(
            rotations, translations, parameterization="euler_angles", convention="ZXY"
        )

    def drr(self, cube_hu: np.ndarray | Tensor):
        """DiffDRR's renderer of a ``(z, y, x)`` HU cube."""
        from diffdrr.data import read
        from diffdrr.drr import DRR
        from torchio import ScalarImage

        # torchio reads (x, y, z) arrays with an affine to RAS, which flips x and y.
        center = self.step * (self.size - 1) / 2
        affine = np.diag([-self.step, -self.step, self.step, 1.0])
        affine[:3, 3] = [center, center, -center]
        data = torch.as_tensor(cube_hu, dtype=torch.float32).permute(2, 1, 0)[None]
        subject = read(ScalarImage(tensor=data, affine=affine), orientation="AP")
        # Seen from the detector, so that the images read as radiographs: the
        # patient's left on the right of the PA view and anterior on the left of
        # the lateral one.
        return DRR(
            subject,
            sdd=self.sdd,
            height=self.height,
            delx=self.pixel,
            reverse_x_axis=False,
        )

    @torch.no_grad()
    def render(self, cube_hu: np.ndarray | Tensor, device: str = "cpu") -> Tensor:
        """The ``(2, h, h)`` X-rays of an HU cube, PA then lateral, each in [0, 1]."""
        drr = self.drr(cube_hu).to(device)
        views = drr(self.poses.to(device))[:, 0].cpu()
        low = views.amin(dim=(1, 2), keepdim=True)
        high = views.amax(dim=(1, 2), keepdim=True)
        return (views - low) / (high - low).clamp_min(1e-6)

    @torch.no_grad()
    def cameras(self, num_points: int = 4096) -> list[Camera]:
        """The pinhole camera of each view, PA first, fitted to DiffDRR's projection.

        DiffDRR projects world points to pixel corners; the cameras map CINDER's
        volume coordinates to the detector's, ``(row, column)`` in ``[-1, 1]``
        with the ends at the centers of the edge pixels.
        """
        from diffdrr.pose import RigidTransform

        drr = self.drr(np.zeros((self.size,) * 3, dtype=np.float32))
        generator = torch.Generator().manual_seed(0)
        coords = torch.rand(num_points, 3, generator=generator) * 2 - 1
        voxels = (coords + 1) / 2 * (self.size - 1)  # (z, y, x) indices
        world = drr.affine(voxels.flip(-1)[None])  # torchio indexes (x, y, z)
        cameras = []
        for view in range(2):
            pose = RigidTransform(self.poses.matrix[view : view + 1])
            uv = drr.perspective_projection(pose, world)[0]
            pixels = 2 * (uv.flip(-1) - 0.5) / (self.height - 1) - 1  # (row, column)
            cameras.append(Camera.fit(coords, pixels))
        return cameras


# Encoder ------------------------------------------------------------------------------


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


class BiplanarUNet(BaseEncoder):
    """SPIDER's view encoder: one four-level 2D UNet applied to each X-ray.

    Each input channel is a view, PA first. A view yields two maps: ``out_channels``
    features at its resolution, whose skip connections keep the detail a
    point-wise decoder samples, and the bottleneck, a coarse summary of the view.

    Shape:
        - Input: ``(B, V, h, w)`` for ``V`` views.
        - Output: ``2 V`` maps, each view's full-resolution map then bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 32,
        widths: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__(in_channels)
        self.embed_dims = (out_channels, widths[-1]) * in_channels
        self.down = nn.ModuleList()
        channels = 1
        for width in widths:
            self.down.append(_block(channels, width))
            channels = width
        self.up = nn.ModuleList()
        self.merge = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
            self.merge.append(_block(2 * width, width))
            channels = width
        self.head = nn.Conv2d(channels, out_channels, 1)

    def encode(self, view: Tensor) -> list[Tensor]:
        """The full-resolution map and the bottleneck of one ``(B, 1, h, w)`` view."""
        skips, x = [], view
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        bottleneck = x
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1])):
            x = merge(torch.cat([up(x), skip], dim=1))
        return [self.head(x), bottleneck]

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(m for view in x.split(1, dim=1) for m in self.encode(view))


# PerX2CT ------------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.body(x)


class _SelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).mT.chunk(3, dim=-1)  # (b, hw, c)
        attended = F.scaled_dot_product_attention(q, k, v).mT.reshape(b, c, h, w)
        return x + self.out(attended)


class SliceDecoder(nn.Module):
    """PerX2CT's 2D decoder, after VQGAN's (Esser et al., 2021).

    Residual blocks with self-attention at the input resolution, then two
    residual blocks per level, doubling the resolution between levels.
    """

    def __init__(
        self, in_channels: int, out_channels: int, widths: Sequence[int]
    ) -> None:
        super().__init__()
        channels = widths[0]
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            _ResBlock(channels, channels),
            _SelfAttention(channels),
            _ResBlock(channels, channels),
        ]
        for level, width in enumerate(widths):
            if level:
                layers += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, channels, 3, padding=1),
                ]
            layers += [_ResBlock(channels, width), _ResBlock(width, width)]
            channels = width
        layers += [
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PerX2CT(nn.Module):
    """PerX2CT (Kyung et al., ICASSP 2023): decode a CT volume slice by slice.

    A coarse ``g × g`` grid over a slice is projected onto both X-rays, where the
    encoder's maps are sampled; the samples, the views' global features (pooled
    maps, replicated over the grid) and a positional encoding of the grid's
    coordinates form a feature image that :class:`SliceDecoder` upsamples to the
    slice. Training decodes random slices of the three planes, shared by the
    batch, and keeps their voxels' flat indices in :attr:`sample_indices` for
    :class:`cinder.SampledLoss`; inference decodes the axial slices.

    Args:
        encoder: Encoder spec, built with ``in_channels=2``.
        sample_at: One spec per map of the module that maps the coordinates to
            the map's grid, such as a :class:`cinder.Camera`.
        size: Voxels per axis ``s`` of the volume.
        view_size: Pixels per axis of the X-rays.
        grid_size: Points per axis ``g`` of the grid sampled over a slice.
        global_maps: Indices of the maps whose pooled features are global.
        num_slices: Slices decoded per training step.
        num_frequencies: Frequencies of the positional encoding, neurofield's,
            which leaves out the π of NeRF's.
        widths: Decoder width per level; ``s = g * 2 ** (len(widths) - 1)``.
    """

    def __init__(
        self,
        encoder: ModuleSpec,
        sample_at: Sequence[ModuleSpec],
        size: int = 128,
        view_size: int = 128,
        grid_size: int = 32,
        global_maps: Sequence[int] = (1, 3),
        num_slices: int = 12,
        num_frequencies: int = 10,
        widths: Sequence[int] = (256, 128, 64),
    ) -> None:
        super().__init__()
        if grid_size * 2 ** (len(widths) - 1) != size:
            raise ValueError("the decoder must upsample the grid to the slice size")
        self.size, self.num_slices = size, num_slices
        self.global_maps = tuple(global_maps)
        self.encoder = build_module(encoder, in_channels=2)
        shapes = self.encoder.get_output_shapes((view_size, view_size))
        self.sample_at = nn.ModuleList(build_module(spec) for spec in sample_at)
        self.sampler = GridSampler()
        self.encoding = PositionalEncoding(3, num_frequencies)
        in_channels = (
            sum(shape[0] for shape in shapes)
            + sum(shapes[i][0] for i in self.global_maps)
            + self.encoding.out_features
        )
        self.decoder = SliceDecoder(in_channels, 1, widths)
        axis = torch.linspace(-1.0, 1.0, size)
        self.register_buffer("axis", axis, persistent=False)
        # The grid points are the centers of the blocks the decoder upsamples.
        grid = axis.reshape(grid_size, -1).mean(1)
        self.register_buffer("grid", grid, persistent=False)
        self.sample_indices: Tensor | None = None

    def slice_coords(self, plane: int, index: int) -> Tensor:
        """The ``(g, g, 3)`` coordinates of the grid over slice ``index`` of a plane:
        0 axial, 1 coronal, 2 sagittal."""
        rows, cols = torch.meshgrid(self.grid, self.grid, indexing="ij")
        axes = [rows, cols]
        axes.insert(plane, torch.full_like(rows, float(self.axis[index])))
        return torch.stack(axes, dim=-1)

    def slice_indices(self, plane: int, index: int) -> Tensor:
        """The ``(s²,)`` flat indices of a slice's voxels in the ``(z, y, x)`` volume."""
        steps = torch.arange(self.size, device=self.axis.device)
        rows, cols = torch.meshgrid(steps, steps, indexing="ij")
        axes = [rows, cols]
        axes.insert(plane, torch.full_like(rows, index))
        return ((axes[0] * self.size + axes[1]) * self.size + axes[2]).flatten()

    def decode(
        self, maps: Sequence[Tensor], planes: Sequence[int], indices: Sequence[int]
    ) -> Tensor:
        """Decode the given slices; returns ``(B, 1, n, s, s)``."""
        coords = torch.stack([self.slice_coords(*key) for key in zip(planes, indices)])
        batch, num, grid = maps[0].shape[0], coords.shape[0], coords.shape[1]
        local = [
            self.sampler(sample_at(coords), cond)  # (B, n, g, g, C)
            for cond, sample_at in zip(maps, self.sample_at)
        ]
        pooled = torch.cat([maps[i].mean(dim=(2, 3)) for i in self.global_maps], dim=1)
        pooled = pooled[:, None, None, None].expand(-1, num, grid, grid, -1)
        encoded = self.encoding(coords)
        encoded = encoded.expand(batch, *encoded.shape).to(pooled.dtype)
        features = torch.cat([*local, pooled, encoded], dim=-1)
        out = self.decoder(features.flatten(0, 1).permute(0, 3, 1, 2))  # (B n, 1, s, s)
        return out.unflatten(0, (batch, num)).movedim(2, 1)

    def forward(self, xrays: Tensor, chunk: int = 16) -> Tensor:
        maps = self.encoder(xrays)
        if self.training:
            planes = torch.randint(3, (self.num_slices,)).tolist()
            indices = torch.randint(self.size, (self.num_slices,)).tolist()
            self.sample_indices = torch.cat(
                [self.slice_indices(*key) for key in zip(planes, indices)]
            )
            return self.decode(maps, planes, indices).flatten(2)
        self.sample_indices = None
        slices = [
            self.decode(maps, [0] * len(part), part.tolist())
            for part in torch.arange(self.size).split(chunk)
        ]
        return torch.cat(slices, dim=2)  # the axial slices stacked along z
