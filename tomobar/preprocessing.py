"""NumPy projection normalisation and automatic detector cropping.

Dynamic flat-field normalisation loads its additional dependencies only when
requested. Standard mean/median normalisation requires NumPy alone.
"""

import numpy as np

__all__ = ["normaliser", "autocropper"]


from tomobar._preprocessing_dynamic import (
    dynamic_flat_field_correction as _DFFC,
)


def normaliser(
    data: np.ndarray,
    flats: np.ndarray,
    darks: np.ndarray,
    log: bool = True,
    method: str = "mean",
    axis: int = 0,
    **kwargs,
) -> np.ndarray:
    """Normalise projection data using flat and dark fields.

    Args:
        data (np.ndarray): Raw 3D transmission intensities, with views on
            ``axis`` and the other dimensions describing the detector.
        flats (np.ndarray): Stack of flat-field frames in the same axis layout
            as ``data``. Its frame count can differ from the projection count.
        darks (np.ndarray or None): Stack of dark-field frames with matching
            detector dimensions, or None to use zero dark current.
        log (bool): Apply the negative logarithm to positive normalised values
            and clip negative results to zero. Default True.
        method (str): ``"mean"`` (default) or ``"median"`` averages calibration
            frames before correction; ``"dynamic"`` fits per-view flat fields.
        axis (int): View/frame axis for mean/median correction: 0 (default) for
            ``(views, detY, detX)``, or 1 for ``(detY, views, detX)``.
        **kwargs: Dynamic correction only: ``dyn_downsample`` (default 2)
            downsamples images for coefficient fitting; ``dyn_iterations``
            (default 10) controls repetitions of the parallel analysis used
            to select eigen flat fields. Ignored by mean/median correction.

    Notes:
        Dynamic correction requires ``(detY, views, detX)`` stacks for all
        inputs, regardless of ``axis``. It additionally imports SciPy,
        scikit-image and the optional ``bm3d`` package. Standard mean/median
        correction requires NumPy alone.

    Raises:
        NameError: If data is 2D or the normalisation method is unknown.
        ImportError: If a dynamic-correction dependency is unavailable.

    Returns:
        np.ndarray: Normalised data with the input shape and axis order.
    """
    if np.ndim(data) == 2:
        raise NameError("Normalisation is implemented for 3d data input")
    if darks is None:
        darks = np.zeros(np.shape(flats), dtype="float32")
    if method is None or method == "mean":
        flats = np.mean(flats, axis)  # mean across flats
        darks = np.mean(darks, axis)  # mean across darks
    elif method == "median":
        flats = np.median(flats, axis)  # median across flats
        darks = np.median(darks, axis)  # median across darks
    elif method == "dynamic":
        # dynamic flat field normalisation according to the paper of Vincent Van Nieuwenhove
        dyn_downsample_v = kwargs.get("dyn_downsample", 2)
        dyn_iterations_v = kwargs.get("dyn_iterations", 10)
        data_norm, _, _ = _DFFC(
            data,
            flats,
            darks,
            downsample=dyn_downsample_v,
            nrPArepetions=dyn_iterations_v,
        )
    else:
        raise NameError(
            "Please select an appropriate method for normalisation: mean, median or dynamic"
        )
    if method != "dynamic":
        denom = flats - darks
        denom[(np.where(denom <= 0.0))] = (
            1.0  # remove zeros/negatives in the denominator if any
        )
        if axis == 1:
            denom = denom[:, np.newaxis, :]
            darks = darks[:, np.newaxis, :]
        nomin = data - darks  # get nominator
        nomin[(np.where(nomin < 0.0))] = 1.0  # remove negatives
        data_norm = np.true_divide(nomin, denom)

    if log:
        # calculate negative log (avoiding of log(0) (= inf) and > 1.0 (negative val))
        data_norm[data_norm > 0.0] = -np.log(data_norm[data_norm > 0.0])
        data_norm[data_norm < 0.0] = 0.0  # remove negative values
    # return [data_norm, EFF, EFF_filt]
    return data_norm


def autocropper(data, addbox, backgr_pix1):
    """Crop 3D projection data to reduce the total data size.

    This method assumes that the object is positioned vertically around the central
    point of the horizontal detector. It is important since the vertical mid ROI
    of each projection is used to estimate the background noise levels.


    Args:
        data (np.ndarray): The required dimensions: [Projections, detectorsVertical, detectorsHoriz] !
        addbox: (int) to add additional pixels in addition to automatically found cropped values, i.e. increasing the cropping region (safety option)
        backgr_pix1 (int): to create rectangular ROIs to collect noise statistics on both (vertical) sides of each 2D projection
    """
    backgr_pix2 = int(2.5 * backgr_pix1)  # usually enough to collect noise statistics

    [Projections, detectorsVertical, detectorsHoriz] = np.shape(data)

    horiz_left_indices = np.zeros(Projections).astype(int)
    horiz_right_indices = np.zeros(Projections).astype(int)
    vert_up_indices = np.zeros(Projections).astype(int)
    vert_down_indices = np.zeros(Projections).astype(int)

    for i in range(0, Projections):
        proj2D = data[i, :, :]  # extract 2D projection
        detectorsVert_mid = (int)(0.5 * detectorsVertical)
        # extract two small regions which belong to the background
        RegionLEFT = proj2D[
            detectorsVert_mid - backgr_pix2 : detectorsVert_mid + backgr_pix2,
            0:backgr_pix1,
        ]
        RegionRIGHT = proj2D[
            detectorsVert_mid - backgr_pix2 : detectorsVert_mid + backgr_pix2,
            -1 - backgr_pix1 : -1,
        ]
        ValMean = np.mean(RegionLEFT) + np.mean(RegionRIGHT)
        # get 1D mean vectors
        vert_sum = np.mean(proj2D, 1)
        horiz_sum = np.mean(proj2D, 0)
        # find the maximum values across the vectors
        largest_vert_index = (vert_sum == max(vert_sum)).argmax(axis=0)
        largest_horiz_index = (horiz_sum == max(horiz_sum)).argmax(axis=0)
        # now we need to find the dips of the "gaussian" moving down from the top
        if largest_vert_index == 0:
            min_vert_index = 0
        else:
            min_vert_index = (vert_sum[largest_vert_index::-1] <= ValMean).argmax(
                axis=0
            )
        if largest_vert_index == (detectorsVertical - 1):
            max_vert_index = largest_vert_index + 1
        else:
            max_vert_index = (vert_sum[largest_vert_index:-1] <= ValMean).argmax(axis=0)
        if largest_horiz_index == 0:
            min_horiz_index = 0
        else:
            min_horiz_index = (horiz_sum[largest_horiz_index::-1] <= ValMean).argmax(
                axis=0
            )
        if largest_horiz_index == (detectorsHoriz - 1):
            max_horiz_index = largest_horiz_index + 1
        else:
            max_horiz_index = (horiz_sum[largest_horiz_index:-1] <= ValMean).argmax(
                axis=0
            )
        # checking the boudaries of the selected indices
        if min_vert_index != 0:
            min_vert_index = largest_vert_index - min_vert_index
            if (min_vert_index - addbox) >= 0:
                min_vert_index -= addbox
        if max_vert_index != (detectorsVertical):
            max_vert_index = largest_vert_index + max_vert_index
            if (max_vert_index + addbox) < detectorsVertical:
                max_vert_index += addbox
        if min_horiz_index != 0:
            min_horiz_index = largest_horiz_index - min_horiz_index
            if (min_horiz_index - addbox) >= 0:
                min_horiz_index -= addbox
        if max_horiz_index != (detectorsHoriz):
            max_horiz_index = largest_horiz_index + max_horiz_index
            if (max_horiz_index + addbox) < detectorsHoriz:
                max_horiz_index += addbox
        horiz_left_indices[i] = min_horiz_index
        horiz_right_indices[i] = max_horiz_index
        vert_up_indices[i] = min_vert_index
        vert_down_indices[i] = max_vert_index

    crop_left_horiz = np.min(horiz_left_indices)
    crop_right_horiz = np.max(horiz_right_indices)
    crop_up_vert = np.min(vert_up_indices)
    crop_down_vert = np.max(vert_down_indices)

    # Finally time to crop the data
    cropped_data = data[
        :, crop_up_vert:crop_down_vert, crop_left_horiz:crop_right_horiz
    ]
    return cropped_data
