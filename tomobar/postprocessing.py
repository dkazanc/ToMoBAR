"""Circular masking and central cropping of reconstructed images or volumes."""

import numpy as np

__all__ = ["apply_circular_mask", "perform_recon_crop"]


def apply_circular_mask(data, recon_mask_radius, cupyrun=False):
    """Applies a circular mask of a certain radius to zero the values outside the mask

    Args:
        data (cp or np ndarray): reconstructed volume
        recon_mask_radius (float): radius size
        cupyrun (bool): CuPy run or not

    Returns:
        cp or np ndarray: recon volume after mask applied
    """
    if cupyrun:
        import cupy as xp

    rows, columns = data.shape[-2:]
    half_y, half_x = rows // 2, columns // 2
    half_size = min(half_y, half_x)
    if cupyrun:
        Y, X = xp.ogrid[:rows, :columns]
        dist_from_center = xp.sqrt((X - half_x) ** 2 + (Y - half_y) ** 2)
    else:
        Y, X = np.ogrid[:rows, :columns]
        dist_from_center = np.sqrt((X - half_x) ** 2 + (Y - half_y) ** 2)

    if recon_mask_radius <= 1.0:
        mask = dist_from_center <= half_size - abs(
            half_size - half_size / recon_mask_radius
        )
    else:
        mask = dist_from_center <= half_size + abs(
            half_size - half_size / recon_mask_radius
        )
    data *= mask
    return data


def perform_recon_crop(data, croped_size):
    """Perform cropping of the larger 3D data array (reconstruction) to a smaller one defined by croped_size

    Args:
        data (xp.ndarray): 3D array [vertical, reconX, reconY] or 2D array [reconX, reconY] to crop
        croped_size (int): the size of the array after cropping

    Returns:
        xp.ndarray: cropped 3D array [vertical, reconX_new_size, reconY_new_size]
    """
    if len(data.shape) == 3:
        axis = 2  # or 1
    else:
        axis = 0
    original_recon_size = data.shape[axis]
    crop_limit_start = (original_recon_size - croped_size) // 2
    crop_limit_stop = croped_size + crop_limit_start

    if len(data.shape) == 3:
        return data[
            :, crop_limit_start:crop_limit_stop, crop_limit_start:crop_limit_stop
        ]
    else:
        return data[crop_limit_start:crop_limit_stop, crop_limit_start:crop_limit_stop]
