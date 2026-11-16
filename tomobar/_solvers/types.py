"""Typed, backend-independent storage for the continuation engine.

Arrays are NumPy or CuPy objects. These containers own references only: building
an inspection view never converts, copies or transfers their array contents.
"""

from __future__ import annotations

from collections.abc import Mapping, Iterator
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, TypeAlias, Literal

import numpy as np

from tomobar._solvers.options import SolverOptions
from tomobar._solvers.buffers import ProjectionBuffers
from tomobar._solvers.subsets import ProjectionSubsets

if TYPE_CHECKING:
    from cupy import ndarray as GPUArray

Array: TypeAlias = "np.ndarray | GPUArray"
Scalar: TypeAlias = "float | np.floating | Array"
SolverMethod: TypeAlias = Literal["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]


@dataclass(eq=False, slots=True)
class LandweberWork:
    """Full-data gradient descent needs only shared projection scratch."""

    buffers: ProjectionBuffers


@dataclass(eq=False, slots=True)
class SIRTWork:
    """Projection (R) and volume (C) diagonal normalisation factors."""

    buffers: ProjectionBuffers
    R: Array
    C: Array


@dataclass(eq=False, slots=True)
class CGLSWork:
    """Projection residual r, volume search direction d, and squared norm."""

    buffers: ProjectionBuffers
    r: Array
    d: Array
    normr2: Scalar


@dataclass(eq=False, slots=True)
class FISTAWork:
    """Subset data, extrapolated image X_t, and momentum parameter t."""

    buffers: ProjectionBuffers
    subsets: ProjectionSubsets
    lipschitz_constant: float
    t: Scalar
    X_t: Array


@dataclass(eq=False, slots=True)
class ADMMWork:
    """Data image z, scaled dual u, and retained gradient/proximal scratch."""

    buffers: ProjectionBuffers
    z: Array
    u: Array
    next_image: Array
    lipschitz_constant: float | None = None
    gradient: Array | None = None
    weight_sum: Array | None = None
    # The regulariser owns the cache schema; preserve its mutable cache API.
    tv_dual: dict[str, object] = field(default_factory=dict)


@dataclass(eq=False, slots=True)
class OSEMWork:
    """Subset data and optional cached full-data sensitivity."""

    buffers: ProjectionBuffers
    subsets: ProjectionSubsets
    sensitivity: Array | None = None


SolverWork: TypeAlias = (
    LandweberWork | SIRTWork | CGLSWork | FISTAWork | ADMMWork | OSEMWork
)


class StateView(Mapping[str, object]):
    """Live, shallow, read-only inspection using the established field names.

    A retained view follows in-place algorithm updates. As with the previous
    MappingProxyType, geometry replacement leaves old views on their old work.
    """

    __slots__ = ("_state", "_names")

    def __init__(self, state: SolverOptions | SolverWork):
        self._state = state
        self._names = tuple(
            item.name
            for item in fields(state)
            if not (
                isinstance(state, ADMMWork)
                and item.name == "lipschitz_constant"
                and state.lipschitz_constant is None
            )
        )

    def __getitem__(self, name: str) -> object:
        if name not in self._names:
            raise KeyError(name)
        return getattr(self._state, name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def copy(self) -> dict[str, object]:
        """Return a shallow dictionary, as with the previous mapping proxy."""
        return dict(self)
