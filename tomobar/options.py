"""Immutable settings for iterative data fidelity and TV regularisation."""

from dataclasses import dataclass
from numbers import Real
from typing import Literal
import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return float(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class DataFidelity:
    """Data term for FISTA/ADMM. PWLS and SWLS require ``weights``.

    Weights are raw counts in the same axes as the supplied projections. They
    are clamped and normalized internally without changing the supplied array.
    ``beta_swls`` controls stripe suppression for SWLS only.
    """

    method: Literal["LS", "PWLS", "SWLS", "KL"] = "LS"
    beta_swls: float = 0.1

    def __post_init__(self):
        if self.method not in ("LS", "PWLS", "SWLS", "KL"):
            raise ValueError("fidelity method must be LS, PWLS, SWLS or KL")
        _real(self.beta_swls, "beta_swls", allow_zero=True)


@dataclass(frozen=True)
class Regularisation:
    """TV proximal settings; pass ``None`` to a solver to disable regularisation.

    ``strength`` is the regularisation parameter. ``time_step`` applies to
    ROF_TV, while ``lipschitz_constant`` and ``tv_type`` apply to PD_TV.
    ``tv_type=0`` selects isotropic TV and 1 selects anisotropic TV.
    The regulariser always runs on the reconstructor's selected device.
    """

    method: Literal["ROF_TV", "PD_TV"] = "PD_TV"
    strength: float = 0.001
    iterations: int = 150
    time_step: float = 0.005
    lipschitz_constant: float = 12.0
    tv_type: Literal[0, 1] = 0
    half_precision: bool = False

    def __post_init__(self):
        if self.method not in ("ROF_TV", "PD_TV"):
            raise ValueError("regularisation method must be ROF_TV or PD_TV")
        _real(self.strength, "strength", allow_zero=True)
        _positive_integer(self.iterations, "regularisation iterations")
        _real(self.time_step, "time_step")
        _real(self.lipschitz_constant, "regularisation lipschitz_constant")
        if (
            isinstance(self.tv_type, (bool, np.bool_))
            or not isinstance(self.tv_type, (int, np.integer))
            or self.tv_type not in (0, 1)
        ):
            raise ValueError("tv_type must be 0 or 1")
        _boolean(self.half_precision, "half_precision")
