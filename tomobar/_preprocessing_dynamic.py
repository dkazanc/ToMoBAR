"""Optional dynamic flat-field correction implementation."""

import numpy as np


def dynamic_flat_field_correction(data, flats, darks, downsample, nrPArepetions):
    """Apply dynamic flat-field correction using PCA and TV coefficient fitting."""
    try:
        from skimage.transform import downscale_local_mean
        from skimage.restoration import estimate_sigma
        from scipy.optimize import minimize
        import bm3d
    except ImportError as exc:
        raise ImportError(
            "Dynamic flat-field normalisation requires scikit-image, scipy and bm3d"
        ) from exc

    meanDarkfield = np.mean(darks, axis=1, dtype=np.float64)
    whiteVect = np.zeros(
        (flats.shape[1], flats.shape[0] * flats.shape[2]), dtype=np.float64
    )
    for i in range(flats.shape[1]):
        whiteVect[i] = flats[:, i, :].flatten() - meanDarkfield.flatten()
    mn = np.mean(whiteVect, axis=0)
    Data = whiteVect - mn

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
        selection[
            :, D1 > (np.mean(keepTrack, axis=1) + 2 * np.std(keepTrack, axis=1, ddof=1))
        ] = 1
        return V1, D1, int(np.sum(selection))

    nrEigenflatfields = 0
    print("Parallel Analysis:")
    while nrEigenflatfields <= 0:
        V1, D1, nrEigenflatfields = parallelAnalysis(Data, nrPArepetions)
    print(f"{nrEigenflatfields} eigen flat fields selected!")
    idx = D1.argsort()[::-1]
    D1, V1 = D1[idx], V1[:, idx]

    H, C, W = data.shape
    eig0 = mn.reshape((H, W))
    EFF = np.zeros((nrEigenflatfields + 1, H, W))
    EFF[0] = eig0
    for i in range(nrEigenflatfields):
        EFF[i + 1] = (np.matmul(Data.T, V1[i]).T).reshape((H, W))
    EFF_denoised = EFF.copy()
    print("Denoising EFFs using BM3D method:")
    for i in range(1, len(EFF)):
        print(f"Denoising EFF {i}")
        EFF_max, EFF_min = EFF_denoised[i].max(), EFF_denoised[i].min()
        EFF_denoised[i] = (EFF_denoised[i] - EFF_min) / (EFF_max - EFF_min)
        sigma_bm3d = estimate_sigma(EFF_denoised[i]) * 10
        EFF_denoised[i] = bm3d.bm3d(EFF_denoised[i], sigma_bm3d)
        EFF_denoised[i] = EFF_denoised[i] * (EFF_max - EFF_min) + EFF_min
    print("Denoising completed.")

    def cost_func(x, *args):
        projections, meanFF, FF, DF = args
        FF_eff = np.zeros(FF.shape[1:])
        for i in range(len(FF)):
            FF_eff += x[i] * FF[i]
        logCorProj = (projections - DF) / (meanFF + FF_eff) * np.mean(meanFF + FF_eff)
        Gx, Gy = np.gradient(logCorProj)
        return np.sum(np.sqrt(Gx**2 + Gy**2))

    def condTVmean(projection, meanFF, FF, DF, x, DS):
        projection = downscale_local_mean(projection, (DS, DS))
        meanFF = downscale_local_mean(meanFF, (DS, DS))
        FF = np.asarray([downscale_local_mean(field, (DS, DS)) for field in FF])
        DF = downscale_local_mean(DF, (DS, DS))
        return minimize(
            cost_func, x, args=(projection, meanFF, FF, DF), method="BFGS", tol=1e-8
        ).x

    print("TV optimisation for DFF coefficients:")
    clean_DFFC = np.zeros((H, C, W), dtype=np.float64)
    for i in range(C):
        if i % 5 == 0:
            print("Normalising projection", i)
        projection = data[:, i, :]
        meanFF, FF = EFF_denoised[0], EFF_denoised[1:]
        x = condTVmean(
            projection,
            meanFF,
            FF,
            meanDarkfield,
            np.zeros(nrEigenflatfields),
            downsample,
        )
        FFeff = np.zeros(meanDarkfield.shape)
        for j in range(nrEigenflatfields):
            FFeff += x[j] * EFF_denoised[j + 1]
        clean_DFFC[:, i, :] = np.divide(
            projection - meanDarkfield, EFF_denoised[0] + FFeff
        )
    return [clean_DFFC, EFF, EFF_denoised]
