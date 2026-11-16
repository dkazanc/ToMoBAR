"""Apply explicitly selected reconstruction output processing."""

from tomobar.postprocessing import apply_circular_mask


def postprocess_reconstruction(
    reconstruction, *, recon_mask_radius=None, cupyrun=False
):
    """Apply an optional circular mask in place and return the same array.

    Cropping and axis conversion remain the responsibility of the caller.
    ``cupyrun`` selects CuPy for mask construction; otherwise NumPy is used.
    """
    if recon_mask_radius is not None:
        apply_circular_mask(reconstruction, recon_mask_radius, cupyrun=cupyrun)
    return reconstruction
