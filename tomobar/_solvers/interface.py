"""Structural contracts for solver hosts, independent of CUDA backends.

These protocols describe only what the continuation engine consumes. Concrete
reconstructors and NumPy test hosts conform without inheriting from them.
"""

from __future__ import annotations

from typing import Protocol

from tomobar._solvers.types import Array
from tomobar.options import DataFidelity


class SolverOperator(Protocol):
    """Projection operations, including reusable output and subset storage."""

    def forward(
        self,
        volume: Array,
        *,
        subset_index: int | None = None,
        out: Array | None = None,
    ) -> Array:
        """Project into out when supplied; otherwise return independent storage."""
        ...

    def backproject(
        self,
        projections: Array,
        *,
        subset_index: int | None = None,
        out: Array | None = None,
    ) -> Array:
        """Backproject into out when supplied; otherwise allocate the result."""
        ...


class SolverReconstructor(Protocol):
    """Reconstructor services needed by solver setup, updates and continuation."""

    @property
    def operator(self) -> SolverOperator:
        """Projection operator for the current padded geometry."""
        ...

    @property
    def device_index(self) -> int:
        """CUDA device used for proximal operations."""
        ...

    @property
    def num_subsets(self) -> int:
        """Number of interleaved projection subsets."""
        ...

    @property
    def geometry_version(self) -> int:
        """Version used to invalidate geometry-dependent solver work."""
        ...

    def _ensure_open(self) -> None:
        """Reject use after resource cleanup."""
        ...

    def _prepare_inputs(
        self,
        projections: Array,
        *,
        weights: Array | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        fidelity: DataFidelity = DataFidelity(),
    ) -> tuple[Array, Array | None]:
        """Prepare projections and normalised weights on the padded grid."""
        ...

    def _initial_volume(self, initial: Array | None, *, ones: bool = False) -> Array:
        """Allocate owned working-grid storage or copy a supplied image."""
        ...

    def powermethod(self, *, full_data: bool = False) -> float:
        """Estimate the squared operator norm for full data or all subsets."""
        ...
