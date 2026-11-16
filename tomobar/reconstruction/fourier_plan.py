"""CPU-only Fourier dimensions and workspaces shared by execution/estimation."""

from dataclasses import dataclass, fields
from types import MappingProxyType
import math

import numpy as np

from tomobar.projectors.fourier_geometry import require_fourier_geometry
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry, _validate_shape
from tomobar.options import FourierOptions

CENTER_SIZE_MIN = 192


def resolve_fourier_options(options, keywords):
    """Normalize legacy None-as-default keywords, rejecting typos/conflicts."""
    names = {field.name for field in fields(FourierOptions)}
    unknown = keywords.keys() - names
    if unknown:
        raise TypeError(f"unexpected Fourier options: {', '.join(sorted(unknown))}")
    supplied = {name: value for name, value in keywords.items() if value is not None}
    if options is not None:
        if not isinstance(options, FourierOptions):
            raise TypeError("options must be a FourierOptions instance")
        if supplied:
            raise TypeError(
                "use either FourierOptions or Fourier keyword options, not both"
            )
        return options
    return FourierOptions(**supplied)


def _chunks(length, count):
    step = (length + count - 1) // count
    return tuple((start, min(start + step, length)) for start in range(0, length, step))


@dataclass(frozen=True)
class FourierWorkspace:
    """Shape and dtype metadata for one analytical Fourier workspace."""

    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def nbytes(self):
        """Number of bytes required by this workspace."""
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True, eq=False)
class FourierPlan:
    """Immutable layout; construction performs no GPU allocations or FFTs."""

    options: FourierOptions
    input_shape: tuple[int, int, int]
    even_shape: tuple[int, int, int]
    detector_width: int
    oversampled_width: int
    center_size: int
    odd_horiz: bool
    odd_vert: bool
    output_shape: tuple[int, int, int]
    filter_slices: tuple[tuple[int, int], ...]
    filter_projections: tuple[tuple[int, int], ...]
    ifft_slices: tuple[tuple[int, int], ...]
    padding_m: int
    filter_crop: tuple[int, int]
    output_crop: tuple[int, int]
    theta: np.ndarray
    angle_range_pi_count: int
    eps: float
    mu: float
    interpolation_radius: int
    workspaces: MappingProxyType


def build_fourier_plan(
    projection_geometry,
    volume_geometry,
    input_shape,
    *,
    options=None,
    detector_padding=0,
    input_dtype=np.float32,
):
    """Validate geometry and construct an analytical Fourier inversion plan."""
    options = FourierOptions() if options is None else options
    if not isinstance(options, FourierOptions):
        raise TypeError("options must be a FourierOptions instance")
    if not isinstance(projection_geometry, Parallel3DGeometry) or not isinstance(
        volume_geometry, Volume3DGeometry
    ):
        raise ValueError("FOURIER_INV requires 3D geometry")
    require_fourier_geometry(projection_geometry, method="FOURIER_INV")
    shape = _validate_shape(input_shape, 3, "Fourier input_shape")
    if shape != projection_geometry.projection_shape:
        raise ValueError(
            f"Fourier input_shape must match acquisition shape {projection_geometry.projection_shape}"
        )
    output = volume_geometry.shape
    if output[1] != output[2] or output[0] != shape[0]:
        raise ValueError(
            "FOURIER_INV requires square slices and one volume slice per detector row"
        )
    if output[2] > shape[2]:
        raise ValueError(
            "The reconstruction size should not be larger than the size of the horizontal detector"
        )
    if isinstance(detector_padding, (bool, np.bool_)) or not isinstance(
        detector_padding, (int, np.integer)
    ):
        raise TypeError("detector_padding must be a nonnegative integer")
    if detector_padding < 0:
        raise ValueError("detector_padding must be nonnegative")
    dtype = np.dtype(input_dtype)
    if dtype.kind not in "biuf":
        raise TypeError("Fourier data_dtype must be real numeric")
    rows, views, raw_width = shape
    odd_horiz, odd_vert = bool(raw_width % 2), bool(rows % 2)
    width, height = raw_width + odd_horiz, rows + odd_vert
    n = width + 2 * (detector_padding + options.padding)
    if options.power_of_2_cropping:
        next_power = 1 << (int(n) - 1).bit_length()
        if 0.9 < n / next_power:
            n = next_power
    if options.power_of_2_oversampling:
        oversampled = 1 << (3 * width - 1).bit_length()
        if n > oversampled:
            oversampled = 1 << (int(n) - 1).bit_length()
    else:
        oversampled = max(options.oversampling_level * width, n)
    center = min(options.center_size, n * 2)
    filter_slices = _chunks(
        height,
        height if options.min_mem_usage_filter else options.filter_vol_chunk_count,
    )
    filter_projections = _chunks(views, options.filter_proj_chunk_count)
    ifft_slices = _chunks(
        height // 2, height // 2 if options.min_mem_usage_ifft2 else options.chunk_count
    )
    # Match the execution's float32 angle conversion before range arithmetic.
    with np.errstate(over="ignore"):
        theta = np.asarray(-projection_geometry.angles, dtype=np.float32)
    if not np.isfinite(theta).all():
        raise ValueError(
            "Fourier angles must be representable as finite float32 values"
        )
    theta = np.frombuffer(theta.tobytes(), dtype=np.float32)
    span = np.max(theta) - np.min(theta)
    pi_count = 1 + int(np.ceil(span / math.pi))
    eps = 1e-4
    mu = -np.log(eps) / (2 * n * n)
    radius = int(
        np.ceil(2 * n / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4))
    )
    # Quantitative mode centres the crop on the even FFT grid. Subtracting
    # odd_horiz shifts odd-width acquisitions by a full output pixel.
    crop_center = (n - odd_horiz) // 2 if options.legacy_scaling else n // 2
    lo = crop_center - output[2] // 2
    hi = crop_center + (output[2] + output[2] % 2) // 2
    specs = {}

    def workspace(name, dimensions, kind):
        specs[name] = FourierWorkspace(
            tuple(int(d) for d in dimensions), np.dtype(kind)
        )

    workspace("input", shape, dtype)
    workspace("even_input", (height, views, width), np.float32)
    workspace("filtered", (height, views, n), np.float32)
    workspace("complex_projections", (height // 2, views, n), np.complex64)
    workspace("grid", (height // 2, 2 * n, 2 * n), np.complex64)
    workspace("output", output, np.float32)
    workspace("angles", (views,), np.float32)
    workspace("angle_order", (views,), np.int64)
    workspace(
        "angle_ranges",
        (center, center, 1 + 2 * pi_count) if center >= CENTER_SIZE_MIN else (0,),
        np.uint16,
    )
    filter_rows = max(stop - start for start, stop in filter_slices)
    filter_views = max(stop - start for start, stop in filter_projections)
    workspace("filter_fft_input", (filter_rows, filter_views, oversampled), np.float32)
    workspace(
        "filter_fft_output",
        (filter_rows, filter_views, oversampled // 2 + 1),
        np.complex64,
    )
    workspace(
        "ifft_input",
        (max(stop - start for start, stop in ifft_slices), 2 * n, 2 * n),
        np.complex64,
    )
    return FourierPlan(
        options=options,
        input_shape=shape,
        even_shape=(height, views, width),
        detector_width=n,
        oversampled_width=oversampled,
        center_size=center,
        odd_horiz=odd_horiz,
        odd_vert=odd_vert,
        output_shape=output,
        filter_slices=filter_slices,
        filter_projections=filter_projections,
        ifft_slices=ifft_slices,
        padding_m=oversampled // 2 - width // 2,
        filter_crop=(oversampled // 2 - n // 2, oversampled // 2 + n // 2),
        output_crop=(lo, hi),
        theta=theta,
        angle_range_pi_count=pi_count,
        eps=eps,
        mu=mu,
        interpolation_radius=radius,
        workspaces=MappingProxyType(specs),
    )
