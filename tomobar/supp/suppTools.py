"""Supplementary pre/post processing data tools:

List of functions:

* normaliser - Projection data normalisation module.
* autocropper - automatically crops 3D projection data to reduce its size.

"""

import numpy as np

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        print("CuPy is installed but the GPU device is inaccessible")
except ImportError:
    import numpy as xp


try:
    from skimage.transform import downscale_local_mean
except ImportError:
    print("____! Skimage module is required for Dynamic Flat fields estimation !____")
try:
    from skimage.restoration import estimate_sigma
except ImportError:
    print("____! Skimage module is required for Dynamic Flat fields estimation !____")
try:
    import scipy
except ImportError:
    print("____! scipy module is required for Dynamic Flat fields estimation !____")
try:
    import bm3d
except ImportError:
    # "____! BM3D module is required to use for dynamic flat fields calculation !____"
    pass


def _DFFC(data, flats, darks, downsample, nrPArepetions):
    # Load frames
    meanDarkfield = np.mean(darks, axis=1, dtype=np.float64)
    whiteVect = np.zeros(
        (flats.shape[1], flats.shape[0] * flats.shape[2]), dtype=np.float64
    )
    for i in range(flats.shape[1]):
        whiteVect[i] = flats[:, i, :].flatten() - meanDarkfield.flatten()
    mn = np.mean(whiteVect, axis=0)

    # Substract mean flat field
    M, N = whiteVect.shape
    Data = whiteVect - mn

    # =============================================================================
    # Parallel Analysis (EEFs selection):
    #      Selection of the number of components for PCA using parallel Analysis.
    #      Each flat field is a single row of the matrix flatFields, different
    #      rows are different observations.
    # =============================================================================

    def cov(X):
        one_vector = np.ones((1, X.shape[0]))
        mu = np.dot(one_vector, X) / X.shape[0]
        X_mean_subtract = X - mu
        covA = np.dot(X_mean_subtract.T, X_mean_subtract) / (X.shape[0] - 1)
        return covA

    def parallelAnalysis(flatFields, repetitions):
        stdEFF = np.std(flatFields, axis=0, ddof=1, dtype=np.float64)
        H, W = flatFields.shape
        keepTrack = np.zeros((H, repetitions), dtype=np.float64)
        stdMatrix = np.tile(stdEFF, (H, 1))
        for i in range(repetitions):
            print(f"Parallel Analysis - repetition {i}")
            sample = stdMatrix * np.random.randn(H, W)
            D1, _ = np.linalg.eig(np.cov(sample))
            keepTrack[:, i] = D1.copy()
        mean_flat_fields_EFF = np.mean(flatFields, axis=0)
        F = flatFields - mean_flat_fields_EFF
        D1, V1 = np.linalg.eig(np.cov(F))
        selection = np.zeros((1, H))
        # mean + 2 * std
        selection[
            :, D1 > (np.mean(keepTrack, axis=1) + 2 * np.std(keepTrack, axis=1, ddof=1))
        ] = 1
        numberPC = np.sum(selection)
        return V1, D1, int(numberPC)

    # Parallel Analysis
    nrEigenflatfields = 0
    print("Parallel Analysis:")
    while nrEigenflatfields <= 0:
        V1, D1, nrEigenflatfields = parallelAnalysis(Data, nrPArepetions)
    print(f"{nrEigenflatfields} eigen flat fields selected!")
    idx = D1.argsort()[::-1]
    D1 = D1[idx]
    V1 = V1[:, idx]

    # Calculation eigen flat fields
    H, C, W = data.shape
    eig0 = mn.reshape((H, W))
    EFF = np.zeros((nrEigenflatfields + 1, H, W))  # n_EFF + 1 eig0
    EFF_denoised = np.zeros((nrEigenflatfields + 1, H, W))  # n_EFF + 1 eig0
    print("Calculating EFFs:")
    EFF[0] = eig0
    for i in range(nrEigenflatfields):
        EFF[i + 1] = (np.matmul(Data.T, V1[i]).T).reshape((H, W))

    EFF_denoised = EFF.copy()
    # Denoise eigen flat fields
    print("Denoising EFFs using BM3D method:")
    for i in range(1, len(EFF)):
        print(f"Denoising EFF {i}")
        EFF_max, EFF_min = EFF_denoised[i, :, :].max(), EFF_denoised[i, :, :].min()
        EFF_denoised[i, :, :] = (EFF_denoised[i, :, :] - EFF_min) / (EFF_max - EFF_min)
        sigma_bm3d = estimate_sigma(EFF_denoised[i, :, :]) * 10
        # print(f"Estimated sigma: {sigma_bm3d}")
        EFF_denoised[i, :, :] = bm3d.bm3d(EFF_denoised[i, :, :], sigma_bm3d)
        EFF_denoised[i, :, :] = (EFF_denoised[i, :, :] * (EFF_max - EFF_min)) + EFF_min

    print("Denoising completed.")
    # =============================================================================
    # cost_func: cost funcion used to estimate the weights using TV
    # =============================================================================

    def cost_func(x, *args):
        (projections, meanFF, FF, DF) = args
        FF_eff = np.zeros((FF.shape[1], FF.shape[2]))
        for i in range(len(FF)):
            FF_eff = FF_eff + x[i] * FF[i]
        logCorProj = (
            (projections - DF)
            / (meanFF + FF_eff)
            * np.mean(meanFF.flatten() + FF_eff.flatten())
        )
        Gx, Gy = np.gradient(logCorProj)
        mag = (Gx**2 + Gy**2) ** (1 / 2)
        cost = np.sum(mag.flatten())
        return cost

    # =============================================================================
    # CondTVmean function: finds the optimal estimates  of the coefficients of the
    # eigen flat fields.
    # =============================================================================

    def condTVmean(projection, meanFF, FF, DF, x, DS):
        # Downsample image
        projection = downscale_local_mean(projection, (DS, DS))
        meanFF = downscale_local_mean(meanFF, (DS, DS))
        FF2 = np.zeros((FF.shape[0], meanFF.shape[0], meanFF.shape[1]))
        for i in range(len(FF)):
            FF2[i] = downscale_local_mean(FF[i], (DS, DS))
        FF = FF2
        DF = downscale_local_mean(DF, (DS, DS))
        # Optimize weights (x)
        x = scipy.optimize.minimize(
            cost_func, x, args=(projection, meanFF, FF, DF), method="BFGS", tol=1e-8
        )
        return x.x

    H, C, W = data.shape
    print("TV optimisation for DFF coefficients:")
    clean_DFFC = np.zeros((H, C, W), dtype=np.float64)
    for i in range(C):
        if i % 5 == 0:
            print("Normalising projection", i)
        projection = data[:, i, :]
        # Estimate weights for a single projection
        meanFF = EFF_denoised[0]
        FF = EFF_denoised[1:]
        weights = np.zeros(nrEigenflatfields)
        x = condTVmean(projection, meanFF, FF, meanDarkfield, weights, downsample)
        # Dynamic FFC
        FFeff = np.zeros(meanDarkfield.shape)
        for j in range(nrEigenflatfields):
            FFeff = FFeff + x[j] * EFF_denoised[j + 1]
        tmp = np.divide((projection - meanDarkfield), (EFF_denoised[0] + FFeff))
        clean_DFFC[:, i, :] = tmp

    return [clean_DFFC, EFF, EFF_denoised]


def normaliser(
    data: np.array,
    flats: np.array,
    darks: np.array,
    log: bool = True,
    method: str = "mean",
    axis: int = 0,
    **kwargs,
) -> np.ndarray:
    """Data normalisation module

    Args:
        data (np.array): 3d numpy array of raw data.
        flats (np.array): 2d numpy array for flat field.
        darks (np.array): 2d numpy array for dark field.
        log (bool, optional): Take negative log. Defaults to True.
        method (str, optional): Normalisation method, choose "mean", "median" or "dynamic". Defaults to "mean".
        axis (int, optional): Define the ANGLES axis.
        dyn_downsample (int, optional): Parameter for "dynamic" method. Defaults to 2.
        dyn_iterations (int, optional): Parameter for "dynamic" method. Defaults to 10.


    Raises:
        NameError: method error

    Returns:
        np.ndarray: 3d numpy array of normalised data
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
        for key, value in kwargs.items():
            if key == "dyn_downsample":
                dyn_downsample_v = value
            else:
                dyn_downsample_v = 2
            if key == "dyn_iterations":
                dyn_iterations_v = value
            else:
                dyn_iterations_v = 10
        [data_norm, EFF, EFF_filt] = _DFFC(
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
        denom[
            (np.where(denom <= 0.0))
        ] = 1.0  # remove zeros/negatives in the denominator if any
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
    """The method crops 3D projection data in order to reduce the total data size.
    Method assumes that the object is positioned vertically around the central
    point of the horizontal detector. It is important since the vertical mid ROI
    of each projection is used to estimate the background noise levels.


    Args:
        data (np.ndarray) The required dimensions: [Projections, detectorsVertical, detectorsHoriz] !
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


def _apply_circular_mask(data, recon_mask_radius, axis=2):
    """Applies a circular mask of a certain radius to zero the values outside tha mask

    Args:
        data (cp or np ndarray): reconstructed volume
        recon_mask_radius (float): radius size

    Returns:
        cp or np ndarray: recon volume after mask applied
    """
    recon_size = data.shape[axis]
    Y, X = xp.ogrid[:recon_size, :recon_size]
    half_size = recon_size // 2
    dist_from_center = xp.sqrt((X - half_size) ** 2 + (Y - half_size) ** 2)
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


def _check_kwargs(reconstruction, **kwargs):
    # Iterating over optional parameters:
    for key, value in kwargs.items():
        if key == "recon_mask_radius" and value is not None:
            _apply_circular_mask(reconstruction, value)
    return reconstruction


def circ_mask(X, diameter):
    # applying a circular mask to the reconstructed image/volume
    # Make the 'diameter' smaller than 1.0 in order to shrink it
    obj_shape = np.shape(X)
    X_masked = np.float32(np.zeros(obj_shape))
    if np.ndim(X) == 2:
        objsize = obj_shape[0]
    elif np.ndim(X) == 3:
        objsize = obj_shape[1]
    else:
        print("Object input size is wrong for the mask to apply to")
    c = np.linspace(
        -(objsize * (1.0 / diameter)) / 2.0, (objsize * (1.0 / diameter)) / 2.0, objsize
    )
    x, y = np.meshgrid(c, c)
    mask = np.float32(np.array((x**2 + y**2 < (objsize / 2.0) ** 2)))
    if np.ndim(X) == 3:
        for z in range(0, obj_shape[0]):
            X_masked[z, :, :] = np.multiply(X[z, :, :], mask)
    else:
        X_masked = np.multiply(X, mask)
    return X_masked
