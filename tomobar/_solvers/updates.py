"""Pointwise solver updates without full-volume GPU expression temporaries.

Outputs may alias inputs: each GPU thread reads and writes only its own voxel.
Disable FMA contraction to retain the rounding of the original array operations.
NumPy expressions provide the same operations for CPU reference backends.
"""

from functools import lru_cache

import numpy as np

_OPERATIONS = {
    "gradient_step": (
        "T image, T gradient, T lipschitz",
        "T updated",
        "T scaled = gradient / lipschitz; updated = image - scaled;",
    ),
    "fista_momentum": (
        "T image, T previous, T momentum",
        "T updated",
        "T difference = image - previous; T scaled = momentum * difference; "
        "updated = image + scaled;",
    ),
    "admm_data_step": (
        "T data, T previous, T dual, T gradient, T step, T rho",
        "T updated",
        "T difference = data - previous; T shifted = difference + dual; "
        "T penalty = rho * shifted; T combined = gradient + penalty; "
        "T scaled = step * combined; updated = data - scaled;",
    ),
    "admm_proximal_input": (
        "T data, T previous, T dual, T relaxation, T previous_coefficient",
        "T updated",
        "T current = relaxation * data; T old = previous_coefficient * previous; "
        "T relaxed = current + old; updated = relaxed + dual;",
    ),
    "admm_dual_step": (
        "T dual, T data, T previous, T image, T relaxation, T previous_coefficient",
        "T updated",
        "T current = relaxation * data; T old = previous_coefficient * previous; "
        "T relaxed = current + old; T shifted = dual + relaxed; "
        "updated = shifted - image;",
    ),
}


@lru_cache(maxsize=None)
def _kernel(name):
    import cupy as cp

    return cp.ElementwiseKernel(
        *_OPERATIONS[name], name=f"tomobar_{name}", options=("--fmad=false",)
    )


def gradient_step(image, gradient, lipschitz, *, out):
    """Write image - gradient/lipschitz, allowing an in-place image update."""
    if isinstance(image, np.ndarray):
        np.copyto(out, image - gradient / lipschitz)
    else:
        _kernel("gradient_step")(image, gradient, lipschitz, out)
    return out


def fista_momentum(image, previous, momentum, *, out):
    """Consume the previous image to form the next extrapolated image."""
    if isinstance(image, np.ndarray):
        np.copyto(out, image + momentum * (image - previous))
    else:
        _kernel("fista_momentum")(image, previous, momentum, out)
    return out


def admm_data_step(data, previous, dual, gradient, step, rho, *, out):
    """Write the linearised data update; the old data image may be overwritten."""
    if isinstance(data, np.ndarray):
        np.copyto(out, data - step * (gradient + rho * (data - previous + dual)))
    else:
        _kernel("admm_data_step")(data, previous, dual, gradient, step, rho, out)
    return out


def admm_proximal_input(data, previous, dual, relaxation, *, out):
    """Form the proximal input without retaining a separate relaxed image."""
    if isinstance(data, np.ndarray):
        np.copyto(out, relaxation * data + (1 - relaxation) * previous + dual)
    else:
        _kernel("admm_proximal_input")(
            data, previous, dual, relaxation, 1 - relaxation, out
        )
    return out


def admm_dual_step(dual, data, previous, image, relaxation, *, out):
    """Recompute relaxation so in-place proximal operators are also safe."""
    if isinstance(data, np.ndarray):
        relaxed = relaxation * data + (1 - relaxation) * previous
        np.copyto(out, dual + relaxed - image)
    else:
        _kernel("admm_dual_step")(
            dual, data, previous, image, relaxation, 1 - relaxation, out
        )
    return out
