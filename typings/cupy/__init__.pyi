"""CuPy array identity and metadata for the initial CPU-only typing scope."""

import numpy as np

class ndarray:
    """GPU array metadata; numerical operations are outside this typing scope."""

    shape: tuple[int, ...]
    ndim: int
    size: int
    dtype: np.dtype[np.generic]
