"""Reconstruction class for 3D direct methods using CuPy-library.

* :func:`RecToolsDIRCuPy.FORWPROJ` and :func:`RecToolsDIRCuPy.BACKPROJ` - Forward/Backward projection modules (ASTRA with DirectLink and CuPy).
* :func:`RecToolsDIRCuPy.FBP` - Filtered Back Projection (ASTRA, the filter is implemented with CuPy).
* :func:`RecToolsDIRCuPy.FOURIER_INV` - Fourier direct reconstruction on unequally spaced grids (interpolation in image space), aka log-polar method [NIKITIN2017]_.
"""

import numpy as np
import math
import cupy as cp
from cupyx.scipy.fft import fft, ifft2, rfftfreq, rfft, irfft
from cupyx.scipy.fftpack import get_fft_plan

from tomobar.supp.memory_estimator_helpers import DeviceMemStack
from tomobar.supp.suppTools import check_kwargs, _apply_horiz_detector_padding
from tomobar.supp.funcs import _data_dims_swapper
from tomobar.fourier import _filtersinc3D_cupy, calc_filter
from tomobar.cuda_kernels import load_cuda_module

from tomobar.methodsDIR import RecToolsDIR

_CENTER_SIZE_MIN = 192  # must be divisible by 8


class RecToolsDIRCuPy(RecToolsDIR):
    """Reconstruction class using direct methods with CuPy API.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimH_pad (int): The amount of padding for the horizontal detector.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float, ndarray): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        device_projector (int): An index (integer) of a specific GPU device.
    """

    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimH_pad,  # Padding size of horizontal detector
        DetectorsDimV,  # Vertical detector dimension (3D case)
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # Reconstructed object dimensions (scalar)
        device_projector=0,  # set an index (integer) of a specific GPU device
    ):
        super().__init__(
            DetectorsDimH,
            DetectorsDimH_pad,
            DetectorsDimV,
            CenterRotOffset,
            AnglesVec,
            ObjSize,
            device_projector,
        )
        # if DetectorsDimV == 0 or DetectorsDimV is None:
        #     raise ValueError("2D CuPy reconstruction is not yet supported, only 3D is")

    def FORWPROJ(self, data: cp.ndarray, **kwargs) -> cp.ndarray:
        """Module to perform forward projection of 2d/3d data as a cupy array

        Args:
            data (cp.ndarray): 2D or 3D object as a cupy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the OUTPUT data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            cp.ndarray: Forward projected cupy array (projection data)
        """
        projected = self.Atools._forwprojCuPy(data)
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                projected = _data_dims_swapper(
                    projected, value, ["detY", "angles", "detX"]
                )

        return projected

    def BACKPROJ(self, data: cp.ndarray, **kwargs) -> cp.ndarray:
        """Module to perform back-projection of 2d/3d data as a cupy array

        Args:
            data (cp.ndarray): 2D/3D projection data as a cupy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            cp.ndarray: Backprojected 2D/3D object
        """
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                data = _data_dims_swapper(data, value, ["detY", "angles", "detX"])

        data = _apply_horiz_detector_padding(
            data, self.Atools.detectors_x_pad, cupyrun=True
        )
        return self.Atools._backprojCuPy(data)

    def FBP(self, data: cp.ndarray, **kwargs) -> cp.ndarray:
        """Filtered backprojection reconstruction on a CuPy array using a custom built SINC filter.
           See more about the method `here <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction/FBP3d_tomobar.html>`__.

        Args:
            data (cp.ndarray): projection data as a CuPy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["angles", "detY", "detX"] for 3D.
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].
            cutoff_freq (float): Cutoff frequency parameter for the sinc filter. Defaults to 0.35.

        Returns:
            cp.ndarray: The FBP reconstructed volume as a CuPy array.
        """

        cupyrun = True
        kwargs.update({"cupyrun": cupyrun})  # needed for agnostic array cropping
        cutoff_freq = 0.35  # default value

        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                data = _data_dims_swapper(data, value, ["angles", "detY", "detX"])
            if key == "cutoff_freq" and value is not None:
                cutoff_freq = value

        # Edge-Pad the horizontal detector (detX) on both sides symmetrically using the provided amount of pixels.
        data = _apply_horiz_detector_padding(data, self.Atools.detectors_x_pad, cupyrun)
        # filter the data on the GPU and keep the result there
        data = _filtersinc3D_cupy(data, cutoff=cutoff_freq)
        data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))
        cache = cp.fft.config.get_plan_cache()
        cache.clear()  # flush FFT cache here before backprojection
        cp._default_memory_pool.free_all_blocks()  # free everything related to the filtering before starting Astra
        reconstruction = self.Atools._backprojCuPy(data)  # 3d backprojecting
        return check_kwargs(reconstruction, **kwargs)

    def FOURIER_INV(self, data: cp.ndarray, **kwargs) -> cp.ndarray:
        """Fourier direct inversion in 3D on unequally spaced (also called as NonUniform FFT/NUFFT) grids using CuPy array as an input, see more in
        [NIKITIN2017]_. This implementation is originated from V. Nikitin's CUDA-C implementation:
        https://github.com/nikitinvv/radonusfft and TomoCuPy package.
        See more about the method `here <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction/LPRec3d_tomobar.html>`__.


        Args:
            data (cp.ndarray): projection data as a CuPy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                        When "None" we assume  ["angles", "detX"] for 2D and ["angles", "detY", "detX"] for 3D.
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].
            filter_type (str): Filter type, the accepted values are: none, ramp, shepp, cosine, cosine2, hamming, hann, parzen.
            cutoff_freq (float): Cutoff frequency parameter for different filter. The higher values increase resolution and noise.

        Returns:
            cp.ndarray: The NUFFT reconstructed volume as a CuPy array.
        """
        cupyrun = True
        kwargs.update({"cupyrun": cupyrun})  # needed for agnostic array cropping
        cutoff_freq = 1.0  # default value
        filter_type = "shepp"  # default filter

        center_size = 32768
        block_dim = [16, 16]
        block_dim_center = [32, 4]

        chunk_count = 4
        filter_vol_chunk_count = 4
        filter_proj_chunk_count = 4
        oversampling_level = 4  # at least 3 or larger required
        power_of_2_oversampling = True
        power_of_2_cropping = False
        min_mem_usage_filter = True
        min_mem_usage_ifft2 = True
        padding = 0

        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                data = _data_dims_swapper(data, value, ["detY", "angles", "detX"])
            elif key == "center_size" and value is not None:
                center_size = value
            elif key == "block_dim" and value is not None:
                block_dim = value
            elif key == "block_dim_center" and value is not None:
                block_dim_center = value
            elif key == "cutoff_freq" and value is not None:
                cutoff_freq = value
            elif key == "filter_type" and value is not None:
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
            elif key == "power_of_2_oversampling" and value is not None:
                power_of_2_oversampling = value
            elif key == "power_of_2_cropping" and value is not None:
                power_of_2_cropping = value
            elif key == "min_mem_usage_filter" and value is not None:
                min_mem_usage_filter = value
            elif key == "min_mem_usage_ifft2" and value is not None:
                min_mem_usage_ifft2 = value
            elif key == "padding" and value is not None:
                if not isinstance(value, int) or value < 0:
                    print(f"Invalid padding: {value}. Set to 0")
                else:
                    padding = value
            elif key == "chunk_count" and value is not None:
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
        r2c_c1dfftshift = module.get_function("r2c_c1dfftshift")
        c1dfftshift = module.get_function("c1dfftshift")
        c2dfftshift = module.get_function("c2dfftshift")
        unpadding_mul_phi = module.get_function("unpadding_mul_phi")

        # initialisation
        [nz, nproj, data_n] = data.shape
        recon_size = self.Atools.recon_size
        if recon_size > data_n:
            raise ValueError(
                "The reconstruction size {} should not be larger than the size of the horizontal detector {}".format(
                    recon_size, data_n
                )
            )

        mem_stack = DeviceMemStack().instance()
        if mem_stack:
            mem_stack.malloc(np.prod(data.shape) * data.dtype.itemsize)

        odd_horiz = bool(data_n % 2)
        odd_vert = bool(nz % 2)

        data_n += odd_horiz
        nz += odd_vert

        if odd_horiz or odd_vert:
            if mem_stack:
                mem_stack.malloc(np.prod((nz, nproj, data_n)) * cp.float32().itemsize)
            else:
                data_p = cp.zeros((nz, nproj, data_n), dtype=cp.float32)
                data_p[: nz - odd_vert, :, : data_n - odd_horiz] = data
                data_p[: nz - odd_vert, :, -odd_horiz] = data[..., -odd_horiz]
                data = data_p
                del data_p

        n = data_n + self.Atools.detectors_x_pad * 2 + padding * 2
        if power_of_2_cropping:
            n_pow2 = 2 ** math.ceil(math.log2(n))
            if 0.9 < n / n_pow2:
                n = n_pow2

        # Limit the center size parameter
        center_size = min(center_size, n * 2)

        if mem_stack:
            mem_stack.malloc(
                np.prod(self.Atools.angles_vec.shape) * cp.float32().itemsize
            )

        theta = cp.array(-self.Atools.angles_vec, dtype=cp.float32)

        if center_size >= _CENTER_SIZE_MIN:
            if mem_stack:
                mem_stack.malloc(
                    np.prod(self.Atools.angles_vec.shape) * np.int64().itemsize
                )
                mem_stack.malloc(
                    np.prod(self.Atools.angles_vec.shape) * np.float32().itemsize
                )

                sorted_theta_cpu = cp.sort(theta).get()
            else:
                sorted_theta_indices = cp.argsort(theta)
                sorted_theta = theta[sorted_theta_indices]
                sorted_theta_cpu = sorted_theta.get()

            theta_full_range = abs(sorted_theta_cpu[nproj - 1] - sorted_theta_cpu[0])
            angle_range_pi_count = 1 + int(np.ceil(theta_full_range / math.pi))

            if mem_stack:
                mem_stack.malloc(
                    np.prod((center_size, center_size, (1 + angle_range_pi_count * 2)))
                    * cp.uint16().itemsize
                )
            else:
                angle_range = cp.zeros(
                    [center_size, center_size, 1 + angle_range_pi_count * 2],
                    dtype=cp.uint16,
                )

        # usfft parameters
        eps = 1e-4  # accuracy of usfft
        mu = -np.log(eps) / (2 * n * n)

        # STEP0: FBP filtering
        if mem_stack:
            self._fbp_filtering_estimator(
                data_n,
                n,
                nproj,
                nz,
                power_of_2_oversampling,
                oversampling_level,
                filter_vol_chunk_count,
                filter_proj_chunk_count,
                min_mem_usage_filter,
            )

            mem_stack.free(np.prod((nz, nproj, data_n)) * cp.float32().itemsize)
        else:
            tmp_p = self._fbp_filtering(
                data,
                data_n,
                n,
                nproj,
                nz,
                power_of_2_oversampling,
                oversampling_level,
                filter_type,
                cutoff_freq,
                filter_vol_chunk_count,
                filter_proj_chunk_count,
                min_mem_usage_filter,
            )

            del data

        # Memory clean up of interpolation extra arrays
        if mem_stack:
            self._setup_backprojection_input_estimator(n, nproj, nz)
            mem_stack.free(np.prod((nz, nproj, n)) * cp.float32().itemsize)
        else:
            (datac, fde) = self._setup_backprojection_input(
                tmp_p, n, nproj, nz, center_size, r2c_c1dfftshift
            )

            del tmp_p

        # BACKPROJECTION
        if mem_stack:
            self._fft_and_interpolation_estimator(n, nproj, nz)
            mem_stack.free(np.prod((nz // 2, nproj, n)) * cp.complex64().itemsize)
        else:
            self._fft_and_interpolation(
                datac,
                fde,
                n,
                nproj,
                nz,
                center_size,
                block_dim,
                block_dim_center,
                theta,
                sorted_theta,
                sorted_theta_indices,
                angle_range_pi_count,
                angle_range,
                eps,
                mu,
                c1dfftshift,
                gather_kernel_partial,
                gather_kernel_center_angle_based_prune,
                gather_kernel_center,
                gather_kernel,
            )

            del datac

        # STEP3: ifft 2d
        if mem_stack:
            self.ifft_gathered_projections_estimator(
                n, nz, chunk_count, min_mem_usage_ifft2
            )
        else:
            self.ifft_gathered_projections(
                fde, n, nz, chunk_count, min_mem_usage_ifft2, c2dfftshift
            )

        # Unpadded recon output size
        if mem_stack:
            self.unpad_reconstructed_data_estimator(
                n, nz, odd_horiz, odd_vert, recon_size
            )

            mem_stack.free(np.prod((nz // 2, 2 * n, 2 * n)) * cp.complex64().itemsize)
            return mem_stack.highwater * 1.35

        recon_up = self.unpad_reconstructed_data(
            fde,
            n,
            nproj,
            nz,
            odd_horiz,
            odd_vert,
            recon_size,
            mu,
            unpadding_mul_phi,
        )

        del fde

        return check_kwargs(
            recon_up,
            **kwargs,
        )

    def _fbp_filtering(
        self,
        data: cp.ndarray,
        raw_detector_width: int,
        detector_width: int,
        projection_count: int,
        detector_height: int,
        power_of_2_oversampling: bool,
        oversampling_level: int,
        filter_type: str,
        cutoff_freq,
        filter_vol_chunk_count: int,
        filter_proj_chunk_count: int,
        min_mem_usage_filter: bool,
    ) -> cp.ndarray:
        # init filter
        if power_of_2_oversampling:
            oversampled_detector_width = 2 ** math.ceil(
                math.log2(raw_detector_width * 3)
            )
            if detector_width > oversampled_detector_width:
                oversampled_detector_width = 2 ** math.ceil(math.log2(detector_width))
        else:
            oversampled_detector_width = int(oversampling_level * raw_detector_width)
            oversampled_detector_width = max(oversampled_detector_width, detector_width)

        padding_m = oversampled_detector_width // 2 - raw_detector_width // 2
        unpad_m = oversampled_detector_width // 2 - detector_width // 2
        unpad_p = oversampled_detector_width // 2 + detector_width // 2

        rotation_axis = self.Atools.centre_of_rotation + 0.5

        wfilter = calc_filter(oversampled_detector_width, filter_type, cutoff_freq)
        t = rfftfreq(oversampled_detector_width).astype(cp.float32)
        w = wfilter * cp.exp(-2 * cp.pi * 1j * t * (rotation_axis))

        # FBP filtering output
        tmp_p = cp.empty(
            (detector_height, projection_count, detector_width), dtype=cp.float32
        )

        if min_mem_usage_filter:
            filter_vol_chunk_count = detector_height

        slice_count_per_chunk = np.ceil(detector_height / filter_vol_chunk_count)
        # Loop over the chunks
        for chunk_index in range(0, filter_vol_chunk_count):
            slice_start_index = min(
                chunk_index * slice_count_per_chunk, detector_height
            )
            slice_end_index = min(
                (chunk_index + 1) * slice_count_per_chunk, detector_height
            )
            if slice_start_index >= slice_end_index:
                break

            # processing by chunks over the second dimension
            # to avoid increased data sizes due to oversampling
            projection_count_per_projection_chunk = np.ceil(
                projection_count / filter_proj_chunk_count
            )
            for projection_chunk_index in range(filter_proj_chunk_count):
                projection_start_index = min(
                    projection_chunk_index * projection_count_per_projection_chunk,
                    projection_count,
                )
                projection_end_index = min(
                    (projection_chunk_index + 1)
                    * projection_count_per_projection_chunk,
                    projection_count,
                )
                if projection_start_index >= projection_end_index:
                    break

                tmp = cp.pad(
                    data[
                        slice_start_index:slice_end_index,
                        projection_start_index:projection_end_index,
                        :,
                    ],
                    ((0, 0), (0, 0), (padding_m, padding_m)),
                    mode="edge",
                )

                tmp = w * rfft(tmp, axis=2)
                tmp = irfft(tmp, axis=2)
                tmp_p[
                    slice_start_index:slice_end_index,
                    projection_start_index:projection_end_index,
                    :,
                ] = tmp[:, :, unpad_m:unpad_p]

                del tmp

        # Memory clean up of filter data
        del t, wfilter, w
        return tmp_p

    def _fbp_filtering_estimator(
        self,
        raw_detector_width: int,
        detector_width: int,
        projection_count: int,
        detector_height: int,
        power_of_2_oversampling: bool,
        oversampling_level: int,
        filter_vol_chunk_count: int,
        filter_proj_chunk_count: int,
        min_mem_usage_filter: bool,
    ) -> cp.ndarray:
        # init filter
        if power_of_2_oversampling:
            oversampled_detector_width = 2 ** math.ceil(
                math.log2(raw_detector_width * 3)
            )
            if detector_width > oversampled_detector_width:
                oversampled_detector_width = 2 ** math.ceil(math.log2(detector_width))
        else:
            oversampled_detector_width = int(oversampling_level * raw_detector_width)
            oversampled_detector_width = max(oversampled_detector_width, detector_width)

        padding_m = oversampled_detector_width // 2 - raw_detector_width // 2

        mem_stack = DeviceMemStack.instance()
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)

        # FBP filtering output
        mem_stack.malloc(
            np.prod((detector_height, projection_count, detector_width))
            * cp.float32().itemsize
        )

        if min_mem_usage_filter:
            filter_vol_chunk_count = detector_height

        slice_count_per_chunk = np.ceil(detector_height / filter_vol_chunk_count)
        # Loop over the chunks
        for chunk_index in range(0, filter_vol_chunk_count):
            slice_start_index = min(
                chunk_index * slice_count_per_chunk, detector_height
            )
            slice_end_index = min(
                (chunk_index + 1) * slice_count_per_chunk, detector_height
            )
            if slice_start_index >= slice_end_index:
                break

            # processing by chunks over the second dimension
            # to avoid increased data sizes due to oversampling
            projection_count_per_projection_chunk = np.ceil(
                projection_count / filter_proj_chunk_count
            )
            for projection_chunk_index in range(filter_proj_chunk_count):
                projection_start_index = min(
                    projection_chunk_index * projection_count_per_projection_chunk,
                    projection_count,
                )
                projection_end_index = min(
                    (projection_chunk_index + 1)
                    * projection_count_per_projection_chunk,
                    projection_count,
                )
                if projection_start_index >= projection_end_index:
                    break

                rfft_input = cp.empty(
                    (
                        int(slice_end_index - slice_start_index),
                        int(projection_end_index - projection_start_index),
                        int(raw_detector_width + padding_m * 2),
                    ),
                    cp.float32,
                )
                mem_stack.malloc(rfft_input.nbytes)

                rfft_plan = get_fft_plan(rfft_input, axes=(2), value_type="R2C")
                mem_stack.malloc(rfft_plan.work_area.mem.size)
                mem_stack.free(rfft_plan.work_area.mem.size)

                rfft_output_shape = (
                    int(slice_end_index - slice_start_index),
                    int(projection_end_index - projection_start_index),
                    int(raw_detector_width + padding_m * 2) // 2 + 1,
                )
                rfft_output = cp.empty(
                    rfft_output_shape,
                    cp.complex64,
                )

                mem_stack.malloc(rfft_output.nbytes)
                mem_stack.free(rfft_input.nbytes)

                irfft_plan = get_fft_plan(rfft_output, axes=(2), value_type="C2R")
                mem_stack.malloc(irfft_plan.work_area.mem.size)
                mem_stack.free(irfft_plan.work_area.mem.size)

                irfft_output_size = (
                    np.prod(
                        (
                            slice_end_index - slice_start_index,
                            projection_end_index - projection_start_index,
                            (raw_detector_width + padding_m * 2),
                        )
                    )
                    * cp.float32().itemsize
                )

                mem_stack.malloc(irfft_output_size)
                mem_stack.free(rfft_output.nbytes)
                mem_stack.free(irfft_output_size)

        # Memory clean up of filter data
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)

    def _setup_backprojection_input(
        self,
        tmp_p: cp.ndarray,
        detector_width: int,
        projection_count: int,
        detector_height: int,
        center_size: int,
        r2c_c1dfftshift: cp.RawKernel,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        # input data
        datac = cp.empty(
            (detector_height // 2, projection_count, detector_width),
            dtype=cp.complex64,
        )

        # fft, reusable by chunks
        if center_size >= _CENTER_SIZE_MIN:
            fde = cp.empty(
                [detector_height // 2, 2 * detector_width, 2 * detector_width],
                dtype=cp.complex64,
            )
        else:
            fde = cp.zeros(
                [detector_height // 2, 2 * detector_width, 2 * detector_width],
                dtype=cp.complex64,
            )

        # STEP1: fft 1d
        r2c_c1dfftshift(
            (
                int(np.ceil(detector_width / 32)),
                int(np.ceil(projection_count / 32)),
                np.int32(detector_height // 2),
            ),
            (32, 32, 1),
            (tmp_p, datac, detector_width, projection_count, detector_height // 2),
        )

        return (datac, fde)

    def _setup_backprojection_input_estimator(
        self,
        detector_width: int,
        projection_count: int,
        detector_height: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        mem_stack = DeviceMemStack.instance()
        mem_stack.malloc(
            np.prod((detector_height // 2, projection_count, detector_width))
            * cp.complex64().itemsize
        )
        mem_stack.malloc(
            np.prod((detector_height // 2, 2 * detector_width, 2 * detector_width))
            * cp.complex64().itemsize
        )

    def _fft_and_interpolation(
        self,
        datac: cp.ndarray,
        fde: cp.ndarray,
        detector_width: int,
        projection_count: int,
        detector_height: int,
        center_size: int,
        block_dim,
        block_dim_center,
        theta: cp.ndarray,
        sorted_theta: cp.ndarray,
        sorted_theta_indices: cp.ndarray,
        angle_range_pi_count: int,
        angle_range: cp.ndarray,
        eps: float,
        mu: float,
        c1dfftshift: cp.RawKernel,
        gather_kernel_partial: cp.RawKernel,
        gather_kernel_center_angle_based_prune: cp.RawKernel,
        gather_kernel_center: cp.RawKernel,
        gather_kernel: cp.RawKernel,
    ):
        # STEP1: fft 1d
        datac = fft(datac)

        m = int(
            np.ceil(
                2
                * detector_width
                * 1
                / np.pi
                * np.sqrt(
                    -mu * np.log(eps)
                    + (mu * detector_width) * (mu * detector_width) / 4
                )
            )
        )

        c1dfftshift(
            (
                int(np.ceil(detector_width / 32)),
                int(np.ceil(projection_count / 32)),
                np.int32(detector_height // 2),
            ),
            (32, 32, 1),
            (
                datac,
                np.float32(4 / detector_width),
                detector_width,
                projection_count,
                detector_height // 2,
            ),
        )

        # STEP2: interpolation (gathering) in the frequency domain
        # Use original kernel at low dimension.

        if center_size >= _CENTER_SIZE_MIN:
            if center_size != (detector_width * 2):
                gather_kernel_partial(
                    (
                        int(np.ceil(detector_width / block_dim[0])),
                        int(np.ceil(projection_count / block_dim[1])),
                        detector_height // 2,
                    ),
                    (block_dim[0], block_dim[1], 1),
                    (
                        datac,
                        fde,
                        theta,
                        np.int32(m),
                        np.float32(mu),
                        np.int32(center_size),
                        np.int32(detector_width),
                        np.int32(projection_count),
                        np.int32(detector_height // 2),
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
                    np.int32(detector_width),
                    np.int32(projection_count),
                ),
            )

            gather_kernel_center(
                (
                    int(np.ceil(center_size / block_dim_center[0])),
                    int(np.ceil(center_size / block_dim_center[1])),
                    detector_height // 2,
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
                    np.int32(detector_width),
                    np.int32(projection_count),
                    np.int32(detector_height // 2),
                ),
            )
        else:
            gather_kernel(
                (
                    int(np.ceil(detector_width / block_dim[0])),
                    int(np.ceil(projection_count / block_dim[1])),
                    detector_height // 2,
                ),
                (block_dim[0], block_dim[1], 1),
                (
                    datac,
                    fde,
                    theta,
                    np.int32(m),
                    np.float32(mu),
                    np.int32(detector_width),
                    np.int32(projection_count),
                    np.int32(detector_height // 2),
                ),
            )

    def _fft_and_interpolation_estimator(
        self,
        detector_width: int,
        projection_count: int,
        detector_height: int,
    ):
        mem_stack = DeviceMemStack.instance()
        fft_input = cp.empty(
            (detector_height // 2, projection_count, detector_width), cp.complex64
        )
        fft_plan = get_fft_plan(fft_input, axes=(-1))
        mem_stack.malloc(fft_plan.work_area.mem.size)
        mem_stack.free(fft_plan.work_area.mem.size)

    def ifft_gathered_projections(
        self,
        fde: cp.ndarray,
        detector_width: int,
        detector_height: int,
        chunk_count: int,
        min_mem_usage_ifft2: bool,
        c2dfftshift: cp.RawKernel,
    ):
        c2dfftshift(
            (
                int(np.ceil((2 * detector_width) / 32)),
                int(np.ceil((2 * detector_width) / 8)),
                np.int32(detector_height // 2),
            ),
            (32, 8, 1),
            (fde, detector_width, detector_height // 2),
        )

        if min_mem_usage_ifft2:
            chunk_count = detector_height // 2

        slice_count_per_chunk = np.ceil(detector_height // 2 / chunk_count)
        # Loop over the chunks
        for chunk_index in range(0, chunk_count):
            start_index = min(chunk_index * slice_count_per_chunk, detector_height // 2)
            end_index = min(
                (chunk_index + 1) * slice_count_per_chunk, detector_height // 2
            )
            if start_index >= end_index:
                break

            tmp = fde[start_index:end_index, :, :]
            tmp = ifft2(tmp, axes=(-2, -1), overwrite_x=True)
            fde[start_index:end_index, :, :] = tmp
            del tmp

        c2dfftshift(
            (
                int(np.ceil((2 * detector_width) / 32)),
                int(np.ceil((2 * detector_width) / 8)),
                np.int32(detector_height // 2),
            ),
            (32, 8, 1),
            (fde, detector_width, detector_height // 2),
        )

    def ifft_gathered_projections_estimator(
        self,
        detector_width: int,
        detector_height: int,
        chunk_count: int,
        min_mem_usage_ifft2: bool,
    ):
        mem_stack = DeviceMemStack.instance()

        if min_mem_usage_ifft2:
            chunk_count = detector_height // 2

        slice_count_per_chunk = np.ceil(detector_height // 2 / chunk_count)
        # Loop over the chunks
        for chunk_index in range(0, chunk_count):
            start_index = min(chunk_index * slice_count_per_chunk, detector_height // 2)
            end_index = min(
                (chunk_index + 1) * slice_count_per_chunk, detector_height // 2
            )
            if start_index >= end_index:
                break

            ifft2_input = cp.empty(
                (
                    int(end_index - start_index),
                    int(2 * detector_width),
                    int(2 * detector_width),
                ),
                cp.complex64,
            )
            mem_stack.malloc(ifft2_input.nbytes)

            ifft2_plan = get_fft_plan(ifft2_input, axes=(-2, -1))
            mem_stack.malloc(ifft2_plan.work_area.mem.size)
            mem_stack.free(ifft2_plan.work_area.mem.size)
            mem_stack.free(ifft2_input.nbytes)

    def unpad_reconstructed_data(
        self,
        fde: cp.ndarray,
        detector_width: int,
        projection_count: int,
        detector_height: int,
        odd_horiz: bool,
        odd_vert: bool,
        recon_size: int,
        mu,
        unpadding_mul_phi: cp.RawKernel,
    ) -> cp.ndarray:
        odd_recon_size = bool(recon_size % 2)
        unpad_z = detector_height - odd_vert
        unpad_recon_m = (detector_width - odd_horiz) // 2 - recon_size // 2
        unpad_recon_p = (detector_width - odd_horiz) // 2 + (
            recon_size + odd_recon_size
        ) // 2
        unpad_recon_size = unpad_recon_p - unpad_recon_m

        # memory for recon
        recon_up = cp.empty(
            [unpad_z, unpad_recon_size, unpad_recon_size], dtype=cp.float32
        )

        # STEP4: unpadding, multiplication by phi and restructure memory
        unpadding_mul_phi(
            (
                int(np.ceil(unpad_recon_size / 32)),
                int(np.ceil(unpad_recon_size / 32)),
                np.int32(detector_height // 2),
            ),
            (32, 32, 1),
            (
                recon_up,
                fde,
                np.float32(mu),
                projection_count,
                unpad_recon_p,
                unpad_z,
                unpad_recon_m,
                detector_width,
                detector_height // 2,
            ),
        )

        return recon_up

    def unpad_reconstructed_data_estimator(
        self,
        detector_width: int,
        detector_height: int,
        odd_horiz: bool,
        odd_vert: bool,
        recon_size: int,
    ) -> cp.ndarray:
        odd_recon_size = bool(recon_size % 2)
        unpad_z = detector_height - odd_vert
        unpad_recon_m = (detector_width - odd_horiz) // 2 - recon_size // 2
        unpad_recon_p = (detector_width - odd_horiz) // 2 + (
            recon_size + odd_recon_size
        ) // 2
        unpad_recon_size = unpad_recon_p - unpad_recon_m

        # memory for recon
        mem_stack = DeviceMemStack.instance()
        mem_stack.malloc(
            np.prod((unpad_z, unpad_recon_size, unpad_recon_size))
            * cp.float32().itemsize
        )
