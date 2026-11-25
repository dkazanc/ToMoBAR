"""Shared scalar validation independent of geometry and GPU resources."""

from numbers import Real
from typing import cast

import numpy as np


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    # NumPy's scalar overloads do not model the numbers.Real ABC.
    scalar = cast(float, value)
    if not np.isfinite(scalar) or (scalar < 0 if allow_zero else scalar <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return float(value)


def _boolean(value: object, name: str) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
