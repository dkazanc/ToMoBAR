"""Internal slice-wise USFFT backend reimplementation from TomoCuPy.

A matched Gaussian USFFT gather/scatter pair with unit-pixel line-integral
scaling, explicit pixel-centre phases and a circular support mask. Backprojection
is the Euclidean transpose of forward projection, up to float32 roundoff.
Inputs use (Z, Y, X) and (detY, angles, detX); sampling is in unit pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

from tomobar._projection_backend import ProjectionArray
from tomobar.cuda_kernels import get_cuda_kernel, load_cuda_module
from tomobar.geometry import interleaved_view_indices, _validate_centre_of_rotation
from tomobar.options import _positive_integer
from tomobar.supp.array_prepare import prepare_array, validate_array
from tomobar.supp.device import _on_device, _validate_device_index


@dataclass(frozen=True, slots=True)
class USFFTParameters:
    """Gaussian interpolation parameters and masked spatial deconvolution.

    stencil_radius is the half-width in Cartesian grid samples. gaussian_width
    is mu in the interpolation exponent. deconvolution includes the circular
    mask and is applied symmetrically in forward and transpose operations.
    """

    stencil_radius: int
    gaussian_width: float
    deconvolution: cp.ndarray


@dataclass(frozen=True, slots=True)
class USFFTViewPlan:
    """Device-resident angles and paired phases for one fixed view selection."""

    angles: cp.ndarray
    phase: cp.ndarray
    adjoint_phase: cp.ndarray


@dataclass(slots=True)
class USFFTWorkspace:
    """One reusable grid/polar slab for the current slice count.

    The flat polar slab supports contiguous views for unequal subset lengths.
    host_output is allocated only for NumPy execution and never escapes to GPU
    callers. Public allocating calls return independent storage.
    """

    cartesian: cp.ndarray
    polar: cp.ndarray
    host_output: cp.ndarray | None = None


class USFFTBackend:
    """Slice-wise GPU projector with a matched Euclidean transpose.

    Args:
        size: Positive square slice width N, in unit-spaced pixels.
        detector_width: Output width, including caller padding; at least N.
        angles: Nonempty one-dimensional acquisition angles in radians. Copied
            as float32 without sorting; duplicate angles are retained.
        centre_offset: Finite scalar CoR correction in detector pixels. Positive
            values move projected features towards decreasing detector columns.
        mask_radius_squared: Positive squared support radius relative to N/2,
            evaluated at physical voxel centres. The mask is applied in both
            directions; pixels on or outside the boundary are excluded.
        subset_count: Number of nonempty interleaved view partitions, default 1.
        device_index: Nonnegative CUDA device index, default 0.

    Notes:
        Inputs are real arrays with axes (Z, Y, X) or (detY, angles, detX).
        NumPy results return to the host; CuPy results remain on the selected
        GPU. Inputs are packed as contiguous float32 without modifying them.
        Backprojection reverses the forward FFTs, phases, interpolation and
        padding with their Euclidean adjoints, up to float32 roundoff. There is
        no angular or view-count normalisation. The Gaussian interpolation
        target is 1e-6; it is not a bound on total projection error.

        Odd slice widths are preserved. Additional detector bins are zero,
        with a matching central crop in backprojection. CoR translation is
        periodic over N before detector padding / after detector cropping.
        close() is idempotent and releases owned device arrays, not shared
        kernel or CuPy caches. Calls after close() fail. Instances are not
        thread-safe; atomic scatter may produce last-bit differences between
        runs. Fixed device angles and paired phases are cached lazily for full
        data and configured subsets. One grid/polar workspace is retained for
        the current slice count; a different count replaces it. NumPy calls
        also retain a GPU transfer buffer. These caches live until close().
        With GPU out, final packing writes directly into caller storage; FFT
        and shift operations still allocate some temporaries. Allocating calls
        return independent results that never alias reusable workspace storage.
        Use the geometry adapter for 2D inputs and geometry validation.
    """

    def __init__(
        self,
        *,
        size: int,
        detector_width: int,
        angles: ArrayLike,
        centre_offset: float = 0.0,
        mask_radius_squared: float = 1.0,
        subset_count: int = 1,
        device_index: int = 0,
    ) -> None:
        self.size = _positive_integer(size, "size")
        self.detector_width = _positive_integer(detector_width, "detector_width")
        if self.detector_width < self.size:
            raise ValueError("detector_width must be at least size")
        angles = np.asarray(angles)
        if angles.ndim != 1 or not angles.size or angles.dtype.kind not in "iuf":
            raise ValueError("angles must be a nonempty real one-dimensional array")
        with np.errstate(over="ignore"):
            angles = angles.astype(np.float32)
        if not np.isfinite(angles).all():
            raise ValueError("angles must be finite float32 values")
        self.centre_offset = _validate_centre_of_rotation(centre_offset)
        if not np.isfinite(mask_radius_squared) or mask_radius_squared <= 0:
            raise ValueError("mask_radius_squared must be positive and finite")
        self.device_index = _validate_device_index(device_index)
        self.subset_indices = interleaved_view_indices(len(angles), subset_count)
        self._closed = False
        self._view_plans: dict[int | None, USFFTViewPlan] = {}
        self._workspace: USFFTWorkspace | None = None
        self._parameters: USFFTParameters | None = None
        self.mask: cp.ndarray | None = None
        self._initialize(self.size, angles, mask_radius_squared)

    @_on_device(cp)
    def _initialize(
        self, n: int, angles: np.ndarray, mask_radius_squared: float
    ) -> None:
        """Allocate fixed Gaussian parameters on the selected CUDA device."""
        # Gaussian convolution in frequency is undone on the spatial lattice.
        # Use an even oversampled grid, without changing the slice width itself.
        eps = 1e-6
        mu = -math.log(eps) / (2 * n * n)
        m = math.ceil(
            2 * n / math.pi * math.sqrt(-mu * math.log(eps) + (mu * n) ** 2 / 4)
        )
        self.left_pad = (n + 1) // 2
        self.right_pad = n - self.left_pad
        lattice = cp.arange(n, dtype=cp.float32) + self.left_pad - n
        physical = cp.arange(n, dtype=cp.float32) - (n - 1) / 2
        radius_squared = (physical[:, None] ** 2 + physical[None, :] ** 2) / (
            n / 2
        ) ** 2
        self.mask = (radius_squared < mask_radius_squared).astype(cp.float32)
        phi = cp.exp(mu * (lattice[:, None] ** 2 + lattice[None, :] ** 2)) * self.mask
        self.angles = angles
        self._parameters = USFFTParameters(m, mu, phi)
        self._interpolate = get_cuda_kernel(
            load_cuda_module("usfft_interpolation"), "usfft_interpolate"
        )

    def _get_parameters(self) -> USFFTParameters:
        """Return live owned parameters; close() releases the device arrays."""
        if self._parameters is None:
            raise RuntimeError("USFFT backend is closed")
        return self._parameters

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("USFFT backend is closed")

    def _selected_angles(self, subset_index: int | None) -> np.ndarray:
        """Validate the selection and retain acquisition-order angles."""
        self._ensure_open()
        if subset_index is None:
            return self.angles
        if isinstance(subset_index, (bool, np.bool_)) or not isinstance(
            subset_index, (int, np.integer)
        ):
            raise TypeError("subset_index must be an integer")
        if not 0 <= subset_index < len(self.subset_indices):
            raise ValueError("subset_index is out of range")
        return self.angles[self.subset_indices[subset_index]]

    @_on_device(cp)
    def forward(
        self,
        volume: ProjectionArray,
        *,
        subset_index: int | None = None,
        out: ProjectionArray | None = None,
    ) -> ProjectionArray:
        """Project square slices with unit-pixel line-integral scaling.

        out must be independent contiguous float32 storage with the selected
        projection shape, on the input's library/device. It is returned by
        identity. Otherwise a fresh independent result is allocated.
        """
        angles = self._selected_angles(subset_index)
        return self._execute(
            volume, angles, forward=True, subset_index=subset_index, out=out
        )

    @_on_device(cp)
    def backproject(
        self,
        projections: ProjectionArray,
        *,
        subset_index: int | None = None,
        out: ProjectionArray | None = None,
    ) -> ProjectionArray:
        """Apply the Euclidean transpose of forward projection.

        out follows forward's storage contract with shape (Z, size, size).
        Inputs are preserved; workspace contents never escape as results.
        """
        angles = self._selected_angles(subset_index)
        return self._execute(
            projections, angles, forward=False, subset_index=subset_index, out=out
        )

    def _validate_output(
        self, out: ProjectionArray, data: ProjectionArray, shape: tuple[int, ...]
    ) -> None:
        """Reject incompatible/overlapping storage before packing or execution."""
        host = isinstance(data, np.ndarray)
        if isinstance(out, np.ndarray) != host:
            raise TypeError("out must use the same array library as the input")
        validate_array(
            out,
            shape=shape,
            device_index=self.device_index,
            array_module=cp,
            name="out",
        )
        if out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous float32 storage")
        if host and not out.flags.writeable:
            raise ValueError("out must be writable")
        overlaps = np.shares_memory if host else cp.may_share_memory
        if overlaps(data, out):
            raise ValueError("out must not overlap the input")

    def _execute(
        self,
        data: ProjectionArray,
        angles: np.ndarray,
        *,
        forward: bool,
        subset_index: int | None,
        out: ProjectionArray | None,
    ) -> ProjectionArray:
        """Pack input and execute into caller storage or a fresh result."""
        if not isinstance(data, (np.ndarray, cp.ndarray)) or data.ndim != 3:
            raise TypeError("input must be a three-dimensional NumPy or CuPy array")
        if data.shape[0] == 0:
            raise ValueError("input must contain at least one slice")
        domain = (data.shape[0], self.size, self.size)
        projections = (data.shape[0], len(angles), self.detector_width)
        input_shape, output_shape = (
            (domain, projections) if forward else (projections, domain)
        )
        validate_array(
            data, shape=input_shape, device_index=self.device_index, array_module=cp
        )
        if out is not None:
            self._validate_output(out, data, output_shape)
        host = isinstance(data, np.ndarray)
        prepared = prepare_array(
            data,
            shape=input_shape,
            device_index=self.device_index,
            array_module=cp,
            gpu=True,
            pack=True,
        )
        plan = self._get_view_plan(subset_index)
        if host:
            workspace = self._get_workspace(data.shape[0])
            required = math.prod(output_shape)
            if workspace.host_output is None or workspace.host_output.size < required:
                workspace.host_output = cp.empty(required, dtype=cp.float32)
            target = workspace.host_output[:required].reshape(output_shape)
        else:
            target = (
                out if out is not None else cp.empty(output_shape, dtype=cp.float32)
            )
        operation = self._forward_impl if forward else self._backproject_impl
        operation(prepared, angles, out=target, view_plan=plan)
        if host:
            return cp.asnumpy(target, out=out)
        return target

    def _build_view_plan(self, angles: np.ndarray) -> USFFTViewPlan:
        """Upload a fixed selection and construct both phases once."""
        device_angles = cp.asarray(angles, dtype=cp.float32)
        phase = self._phase(device_angles)
        return USFFTViewPlan(device_angles, phase, phase.conj())

    def _get_view_plan(self, subset_index: int | None) -> USFFTViewPlan:
        """Cache at most the full plan and one plan per configured subset."""
        self._ensure_open()
        if subset_index not in self._view_plans:
            if subset_index is None:
                self._view_plans[None] = self._build_view_plan(self.angles)
            else:
                full = self._get_view_plan(None)
                indices = cp.asarray(self.subset_indices[subset_index])
                self._view_plans[subset_index] = USFFTViewPlan(
                    full.angles[indices],
                    full.phase[indices],
                    full.adjoint_phase[indices],
                )
        return self._view_plans[subset_index]

    def _get_workspace(self, rows: int) -> USFFTWorkspace:
        """Keep one workspace; changing the slice count replaces it."""
        if self._workspace is None or self._workspace.cartesian.shape[0] != rows:
            self._workspace = USFFTWorkspace(
                cartesian=cp.empty(
                    (rows, 2 * self.size, 2 * self.size), dtype=cp.complex64
                ),
                polar=cp.empty(rows * len(self.angles) * self.size, dtype=cp.complex64),
            )
        return self._workspace

    def close(self) -> None:
        """Release owned device arrays; shared CUDA caches remain available."""
        self._parameters = None
        self.mask = None
        self._view_plans.clear()
        self._workspace = None
        self._closed = True

    def __enter__(self) -> USFFTBackend:
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _phase(self, angles: np.ndarray | cp.ndarray) -> cp.ndarray:
        """Map FFT lattice origins to physical voxel/detector pixel centres."""
        n = self.size
        rho = (cp.arange(n, dtype=cp.float32) - n // 2) / n
        angles = cp.asarray(angles, dtype=cp.float32)
        image_offset = n - self.left_pad - (n - 1) / 2
        detector_left = (self.detector_width - n) // 2
        detector_offset = n // 2 + detector_left - (self.detector_width - 1) / 2
        displacement = (
            detector_offset
            + self.centre_offset
            - image_offset * (cp.cos(angles) + cp.sin(angles))
        )
        return cp.exp(2j * cp.pi * displacement[:, None] * rho[None, :])

    def _grid(
        self,
        polar: cp.ndarray,
        cartesian: cp.ndarray,
        angles: cp.ndarray,
        *,
        adjoint: bool,
    ) -> None:
        """Gather or scatter in place with identical real Gaussian weights."""
        parameters = self._get_parameters()
        m, mu = parameters.stencil_radius, parameters.gaussian_width
        rows, views, n = polar.shape
        self._interpolate(
            (math.ceil(n / 16), math.ceil(views / 16), rows),
            (16, 16, 1),
            (
                polar,
                cartesian,
                angles,
                np.int32(m),
                np.float32(mu),
                np.int32(n),
                np.int32(views),
                np.int32(rows),
                np.int32(adjoint),
            ),
        )

    def _forward_impl(
        self,
        volume: cp.ndarray,
        angles: np.ndarray,
        *,
        out: cp.ndarray | None = None,
        view_plan: USFFTViewPlan | None = None,
    ) -> cp.ndarray:
        """A = pad_detector Re IFFT1 phase gather FFT2 pad_volume phi.

        Final packing writes directly into out. FFT/shift internals may still
        allocate temporaries; the oversampled grid and polar slab are reused.
        """
        n = self.size
        plan = view_plan if view_plan is not None else self._build_view_plan(angles)
        workspace = self._get_workspace(volume.shape[0])
        cartesian = workspace.cartesian
        cartesian.fill(0)
        start = self.left_pad
        cp.multiply(
            volume,
            self._get_parameters().deconvolution,
            out=cartesian[:, start : start + n, start : start + n],
        )
        axes = (-2, -1)
        cartesian[...] = cp.fft.fftshift(
            cp.fft.fft2(cp.fft.ifftshift(cartesian, axes=axes)), axes=axes
        )
        count = volume.shape[0] * len(angles) * n
        polar = workspace.polar[:count].reshape(volume.shape[0], len(angles), n)
        self._grid(polar, cartesian, plan.angles, adjoint=False)
        cp.multiply(polar, plan.phase, out=polar)
        transformed = cp.fft.fftshift(
            cp.fft.ifft(cp.fft.ifftshift(polar, axes=-1), axis=-1), axes=-1
        ).real
        if out is None:
            out = cp.empty(
                (volume.shape[0], len(angles), self.detector_width), dtype=cp.float32
            )
        left = (self.detector_width - n) // 2
        if self.detector_width != n:
            out.fill(0)
        cp.copyto(out[..., left : left + n], transformed)
        return out

    def _backproject_impl(
        self,
        projections: cp.ndarray,
        angles: np.ndarray,
        *,
        out: cp.ndarray | None = None,
        view_plan: USFFTViewPlan | None = None,
    ) -> cp.ndarray:
        """Reverse the forward operations, writing final scaling into out."""
        n = self.size
        plan = view_plan if view_plan is not None else self._build_view_plan(angles)
        workspace = self._get_workspace(projections.shape[0])
        left = (self.detector_width - n) // 2
        cropped = projections[..., left : left + n]
        count = projections.shape[0] * len(angles) * n
        polar = workspace.polar[:count].reshape(projections.shape[0], len(angles), n)
        # IFFT1* = FFT1 / n. Keep the complex Nyquist phase contribution.
        polar[...] = (
            cp.fft.fftshift(
                cp.fft.fft(cp.fft.ifftshift(cropped, axes=-1), axis=-1), axes=-1
            )
            / n
        )
        cp.multiply(polar, plan.adjoint_phase, out=polar)
        cartesian = workspace.cartesian
        # Scatter accumulates, so every call must clear the reused grid.
        cartesian.fill(0)
        self._grid(polar, cartesian, plan.angles, adjoint=True)
        axes = (-2, -1)
        padded = cp.fft.fftshift(
            cp.fft.ifft2(cp.fft.ifftshift(cartesian, axes=axes)), axes=axes
        )
        # FFT2* = (2n)^2 IFFT2; scale in place before the matching crop.
        padded *= (2 * n) ** 2
        if out is None:
            out = cp.empty((projections.shape[0], n, n), dtype=cp.float32)
        start = self.left_pad
        cp.multiply(
            padded[:, start : start + n, start : start + n].real,
            self._get_parameters().deconvolution,
            out=out,
        )
        return out
