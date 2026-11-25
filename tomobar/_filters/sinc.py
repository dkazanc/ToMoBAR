"""Defaults and float32 cutoff validation for the CUDA SINC filter."""

import numpy as np

from tomobar._validation import _real

SINC_DEFAULT_CUTOFF = 0.35


def validate_sinc_cutoff(cutoff: float) -> float:
    """Require a positive normal float32 shape parameter for the SINC filter.

    Canonicalise the shape parameter before passing it to the CUDA kernel.
    Reject subnormal/overflowing inputs before float32 conversion rather than
    silently rounding them to zero/inf.
    """
    cutoff = _real(cutoff, "cutoff_freq")
    limits = np.finfo(np.float32)
    if not float(limits.tiny) <= cutoff <= float(limits.max):
        raise ValueError("cutoff_freq must fit in the positive normal float32 range")
    return float(np.float32(cutoff))
