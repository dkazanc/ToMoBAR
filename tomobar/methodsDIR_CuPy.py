"""Reconstruction class for 3D direct methods using CuPy-library.

* :func:`RecToolsDIRCuPy.FORWPROJ` and :func:`RecToolsDIRCuPy.BACKPROJ` - Forward/Backward projection modules (ASTRA with DirectLink and CuPy).
* :func:`RecToolsDIRCuPy.FBP` - Filtered Back Projection (ASTRA, the filter is implemented with CuPy).
* :func:`RecToolsDIRCuPy.FOURIER_INV` - Fourier direct reconstruction on unequally spaced grids (interpolation in image space), aka log-polar method [NIKITIN2017]_.
"""

import numpy as np
import math

try:
    import cupy as xp

    from cupyx.scipy.fft import fft, ifft2, rfftfreq, rfft, irfft
except ImportError:
    import numpy as xp

    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )

from tomobar.supp.suppTools import check_kwargs
from tomobar.supp.funcs import _data_dims_swapper
from tomobar.fourier import _filtersinc3D_cupy, calc_filter
from tomobar.cuda_kernels import load_cuda_module

from tomobar.methodsDIR import RecToolsDIR

_CENTER_SIZE_MIN = 192  # must be divisible by 8


class RecToolsDIRCuPy(RecToolsDIR):
    """Reconstruction class using direct methods with CuPy API.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float, ndarray): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        device_projector (int): An index (integer) of a specific GPU device.
    """

    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimV,  # Vertical detector dimension (3D case)
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # Reconstructed object dimensions (scalar)
        device_projector=0,  # set an index (integer) of a specific GPU device
    ):
        super().__init__(
            DetectorsDimH,
            DetectorsDimV,
            CenterRotOffset,
            AnglesVec,
            ObjSize,
            device_projector,
        )
        # if DetectorsDimV == 0 or DetectorsDimV is None:
        #     raise ValueError("2D CuPy reconstruction is not yet supported, only 3D is")

    def FORWPROJ(self, data: xp.ndarray, **kwargs) -> xp.ndarray:
        """Module to perform forward projection of 2d/3d data as a cupy array

        Args:
            data (xp.ndarray): 2D or 3D object as a cupy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the OUTPUT data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            xp.ndarray: Forward projected cupy array (projection data)
        """
        projected = self.Atools._forwprojCuPy(data)
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                projected = _data_dims_swapper(
                    projected, value, ["detY", "angles", "detX"]
                )

        return projected

    def BACKPROJ(self, projdata: xp.ndarray, **kwargs) -> xp.ndarray:
        """Module to perform back-projection of 2d/3d data as a cupy array

        Args:
            projdata (xp.ndarray): 2D/3D projection data as a cupy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            xp.ndarray: Backprojected 2D/3D object
        """
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                projdata = _data_dims_swapper(
                    projdata, value, ["detY", "angles", "detX"]
                )

        return self.Atools._backprojCuPy(projdata)

    def FBP(self, data: xp.ndarray, **kwargs) -> xp.ndarray:
        """Filtered backprojection reconstruction on a CuPy array using a custom built SINC filter.

        Args:
            data (xp.ndarray): projection data as a CuPy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["angles", "detY", "detX"] for 3D.
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].
            cutoff_freq (float): Cutoff frequency parameter for the sinc filter.

        Returns:
            xp.ndarray: The FBP reconstructed volume as a CuPy array.
        """
        kwargs.update({"cupyrun": True})  # needed for agnostic array cropping
        cutoff_freq = 0.35  # default value
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                data = _data_dims_swapper(data, value, ["angles", "detY", "detX"])
            if key == "cutoff_freq" and value is not None:
                cutoff_freq = value

        # filter the data on the GPU and keep the result there
        data = _filtersinc3D_cupy(data, cutoff=cutoff_freq)
        data = xp.ascontiguousarray(xp.swapaxes(data, 0, 1))
        cache = xp.fft.config.get_plan_cache()
        cache.clear()  # flush FFT cache here before backprojection
        xp._default_memory_pool.free_all_blocks()  # free everything related to the filtering before starting Astra
        reconstruction = self.Atools._backprojCuPy(data)  # 3d backprojecting
        xp._default_memory_pool.free_all_blocks()
        return check_kwargs(reconstruction, **kwargs)

    def FOURIER_INV(self, data: xp.ndarray, **kwargs) -> xp.ndarray:
        """Fourier direct inversion in 3D on unequally spaced (also called as NonUniform FFT/NUFFT) grids using CuPy array as an input.
        This implementation follows V. Nikitin's CUDA-C implementation:
        https://github.com/nikitinvv/radonusfft and TomoCuPy package.


        Args:
            data (xp.ndarray): projection data as a CuPy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                        When "None" we assume  ["angles", "detX"] for 2D and ["angles", "detY", "detX"] for 3D.
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].
            filter_type (str): Filter type, the accepted values are: none, ramp, shepp, cosine, cosine2, hamming, hann, parzen.
            cutoff_freq (float): Cutoff frequency parameter for different filter. The higher values increase resolution and noise.

        Returns:
            xp.ndarray: The NUFFT reconstructed volume as a CuPy array.
        """
        kwargs.update({"cupyrun": True})  # needed for agnostic array cropping
        cutoff_freq = 1.0  # default value
        filter_type = "shepp"  # default filter

        center_size = 6144
        block_dim = [16, 16]
        block_dim_center = [32, 4]

        chunk_count = 4

        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                data = _data_dims_swapper(data, value, ["detY", "angles", "detX"])
            elif key == "center_size" and value is not None:
                center_size = value
            elif key == "block_dim" and value is not None:
                block_dim = value
            elif key == "block_dim_center" and value is not None:
                block_dim_center = value
            if key == "cutoff_freq" and value is not None:
                cutoff_freq = value
            if key == "filter_type" and value is not None:
                if value not in [
                    "none",
                    "ramp",
                    "shepp",
                    "cosine",
                    "cosine2",
                    "hamming",
                    "hann",
                    "parzen",
                ]:
                    print(
                        "Unknown filter name, please use: none, ramp, shepp, cosine, cosine2, hamming, hann or parzen. Set to shepp filter"
                    )
                else:
                    filter_type = value
            if key == "chunk_count" and value is not None:
                if not isinstance(value, int) or value <= 0:
                    print(f"Invalid chunk count: {value}. Set to 1")
                else:
                    chunk_count = value

        # extract kernels from CUDA modules
        module = load_cuda_module("fft_us_kernels")
        gather_kernel = module.get_function("gather_kernel")
        gather_kernel_partial = module.get_function("gather_kernel_partial")
        gather_kernel_center_angle_based_prune = module.get_function(
            "gather_kernel_center_angle_based_prune"
        )
        gather_kernel_center = module.get_function("gather_kernel_center")
        wrap_kernel = module.get_function("wrap_kernel")
        r2c_c1dfftshift = module.get_function("r2c_c1dfftshift")
        c1dfftshift = module.get_function("c1dfftshift")
        c2dfftshift = module.get_function("c2dfftshift")
        unpadding_mul_phi = module.get_function("unpadding_mul_phi")

        # initialisation
        [nz, nproj, n] = data.shape
        recon_size = self.Atools.recon_size
        if recon_size > n:
            raise ValueError(
                "The reconstruction size {} should not be larger than the size of the horizontal detector {}".format(
                    recon_size, n
                )
            )

        odd_horiz = bool(n % 2)
        odd_vert = bool(nz % 2)

        n += odd_horiz
        nz += odd_vert

        if odd_horiz or odd_vert:
            data_p = xp.empty((nz, nproj, n), dtype=xp.float32)
            data_p[: nz - odd_vert, :, : n - odd_horiz] = data
            data_p[: nz - odd_vert, :, -odd_horiz] = data[..., -odd_horiz]
            data = data_p
            del data_p

        rotation_axis = self.Atools.centre_of_rotation + 0.5
        theta = xp.array(-self.Atools.angles_vec, dtype=xp.float32)
        if center_size >= _CENTER_SIZE_MIN:
            sorted_theta_indices = xp.argsort(theta)
            sorted_theta = theta[sorted_theta_indices]
            sorted_theta_cpu = sorted_theta.get()

            theta_full_range = abs(sorted_theta_cpu[nproj - 1] - sorted_theta_cpu[0])
            angle_range_pi_count = 1 + int(np.ceil(theta_full_range / math.pi))
            angle_range = xp.zeros(
                [center_size, center_size, 1 + angle_range_pi_count * 2], dtype=xp.uint16
            )


        # usfft parameters
        eps = 1e-4  # accuracy of usfft
        mu = -np.log(eps) / (2 * n * n)
        m = int(
            np.ceil(
                2 * n * 1 / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n) * (mu * n) / 4)
            )
        )
        oversampling_level = 2  # at least 2 or larger required

        # Limit the center size parameter
        center_size = min(center_size, n * 2 + m * 2)

        # init filter
        ne = oversampling_level * n
        padding_m = ne // 2 - n // 2
        padding_p = ne // 2 + n // 2
        wfilter = calc_filter(ne, filter_type, cutoff_freq)

        # STEP0: FBP filtering
        t = rfftfreq(ne).astype(xp.float32)
        w = wfilter * xp.exp(-2 * xp.pi * 1j * t * (rotation_axis))

        # FBP filtering output
        tmp_p = xp.empty(data.shape, dtype=xp.float32)

        slice_count_per_chunk = np.ceil(nz / chunk_count)
        # Loop over the chunks
        for chunk_index in range(0, chunk_count):
            start_index = min(chunk_index * slice_count_per_chunk, nz)
            end_index = min((chunk_index + 1) * slice_count_per_chunk, nz)
            if start_index >= end_index:
                break

            tmp = xp.pad(
                data[start_index:end_index, :, :],
                ((0, 0), (0, 0), (padding_m, padding_m)),
                mode="edge",
            )

            tmp = w * rfft(tmp, axis=2)
            tmp = irfft(tmp, axis=2)
            tmp_p[start_index:end_index, :, :] = tmp[:, :, padding_m:padding_p]

            del tmp

        # Memory clean up of filter and input data
        del data, t, wfilter, w

        # BACKPROJECTION
        # input data
        datac = xp.empty((nz // 2, nproj, n), dtype=xp.complex64)

        # padded fft, reusable by chunks
        fde = xp.empty([nz // 2, 2 * m + 2 * n, 2 * m + 2 * n], dtype=xp.complex64)

        # STEP1: fft 1d
        r2c_c1dfftshift(
            (
                int(np.ceil(n / 32)),
                int(np.ceil(nproj / 32)),
                np.int32(nz // 2),
            ),
            (32, 32, 1),
            (tmp_p, datac, n, nproj, nz // 2),
        )

        # Memory clean up of interpolation extra arrays
        del tmp_p

        datac = fft(datac)

        c1dfftshift(
            (
                int(np.ceil(n / 32)),
                int(np.ceil(nproj / 32)),
                np.int32(nz // 2),
            ),
            (32, 32, 1),
            (datac, np.float32(4 / n), n, nproj, nz // 2),
        )

        # STEP2: interpolation (gathering) in the frequency domain
        # Use original one kernel at low dimension.
        if center_size >= _CENTER_SIZE_MIN:
            if center_size != (n * 2 + m * 2):
                gather_kernel_partial(
                    (
                        int(np.ceil(n / block_dim[0])),
                        int(np.ceil(nproj / block_dim[1])),
                        nz // 2,
                    ),
                    (block_dim[0], block_dim[1], 1),
                    (
                        datac,
                        fde,
                        theta,
                        np.int32(m),
                        np.float32(mu),
                        np.int32(center_size),
                        np.int32(n),
                        np.int32(nproj),
                        np.int32(nz // 2),
                    ),
                )

            gather_kernel_center_angle_based_prune(
                (int(np.ceil(center_size / 256)), center_size, 1),
                (256, 1, 1),
                (
                    angle_range,
                    angle_range_pi_count * 2 + 1,
                    sorted_theta,
                    np.int32(m),
                    np.int32(center_size),
                    np.int32(n),
                    np.int32(nproj),
                ),
            )

            gather_kernel_center(
                (
                    int(np.ceil(center_size / block_dim_center[0])),
                    int(np.ceil(center_size / block_dim_center[1])),
                    nz // 2,
                ),
                (block_dim_center[0], block_dim_center[1], 1),
                (
                    datac,
                    fde,
                    angle_range,
                    angle_range_pi_count * 2 + 1,
                    theta,
                    sorted_theta_indices,
                    np.int32(m),
                    np.float32(mu),
                    np.int32(center_size),
                    np.int32(n),
                    np.int32(nproj),
                    np.int32(nz // 2),
                ),
            )
        else:
            gather_kernel(
                (
                    int(np.ceil(n / block_dim[0])),
                    int(np.ceil(nproj / block_dim[1])),
                    nz // 2,
                ),
                (block_dim[0], block_dim[1], 1),
                (
                    datac,
                    fde,
                    theta,
                    np.int32(m),
                    np.float32(mu),
                    np.int32(n),
                    np.int32(nproj),
                    np.int32(nz // 2),
                ),
            )

        wrap_kernel(
            (
                int(np.ceil((2 * n + 2 * m) / 32)),
                int(np.ceil((2 * n + 2 * m) / 32)),
                np.int32(nz // 2),
            ),
            (32, 32, 1),
            (fde, n, nz // 2, m),
        )

        del datac

        # STEP3: ifft 2d
        c2dfftshift(
            (
                int(np.ceil((2 * n + 2 * m) / 32)),
                int(np.ceil((2 * n + 2 * m) / 8)),
                np.int32(nz // 2),
            ),
            (32, 8, 1),
            (fde, n, nz // 2, m),
        )

        slice_count_per_chunk = np.ceil(nz // 2 / chunk_count)
        # Loop over the chunks
        for chunk_index in range(0, chunk_count):
            start_index = min(chunk_index * slice_count_per_chunk, nz // 2)
            end_index = min((chunk_index + 1) * slice_count_per_chunk, nz // 2)
            if start_index >= end_index:
                break

            tmp = fde[start_index:end_index, :, :]
            tmp = ifft2(tmp, axes=(-2, -1), overwrite_x=True)
            fde[start_index:end_index, :, :] = tmp

            del tmp
        
        c2dfftshift(
            (
                int(np.ceil((2 * n + 2 * m) / 32)),
                int(np.ceil((2 * n + 2 * m) / 8)),
                np.int32(nz // 2),
            ),
            (32, 8, 1),
            (fde, n, nz // 2, m),
        )

        # Unpadded recon output size
        odd_recon_size = bool(recon_size % 2)
        unpad_z = nz - odd_vert
        unpad_recon_m = (n - odd_horiz) // 2 - recon_size // 2
        unpad_recon_p = (n - odd_horiz) // 2 + (recon_size + odd_recon_size) // 2
        unpad_recon_size = unpad_recon_p - unpad_recon_m

        # memory for recon
        recon_up = xp.empty(
            [unpad_z, unpad_recon_size, unpad_recon_size], dtype=xp.float32
        )

        # STEP4: unpadding, multiplication by phi and restructure memory
        unpadding_mul_phi(
            (
                int(np.ceil(unpad_recon_size / 32)),
                int(np.ceil(unpad_recon_size / 32)),
                np.int32(nz // 2),
            ),
            (32, 32, 1),
            (
                recon_up,
                fde,
                np.float32(mu),
                nproj,
                unpad_recon_p,
                unpad_z,
                unpad_recon_m,
                n,
                nz // 2,
                m,
            ),
        )

        del fde

        return check_kwargs(
            recon_up,
            **kwargs,
        )
