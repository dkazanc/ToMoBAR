"""Reconstruction class for direct reconstruction methods (2D/3D).

* :func:`RecToolsDIR.forward` and :func:`RecToolsDIR.backproject` Forward/Backward 2D/3D projection (ASTRA-Toolbox)
* :func:`RecToolsDIR.FBP` Filtered Back Projection 2D/3D (ASTRA with the custom built filter).
"""

import math
from typing import Tuple

import cupy as cp
import numpy as np
from cupyx.scipy.fft import fft, ifft2, rfftfreq, rfft, irfft
from cupyx.scipy.fftpack import get_fft_plan

from tomobar.cuda_kernels import load_cuda_module
from tomobar.fourier import _filtersinc3D_cupy, calc_filter
from tomobar.supp.axis_order import _transpose_data_axes, _resolve_data_axes
from tomobar.supp.device import _on_device
from tomobar.supp.memory_estimator_helpers import DeviceMemStack
from tomobar.supp.suppTools import check_kwargs
from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)
from tomobar.reconstruction_geometry import GeometryReconstruction

_CENTER_SIZE_MIN = 192  # must be divisible by 8


class RecToolsDIR(GeometryReconstruction):
    """Direct reconstruction from explicit acquisition and volume geometry.

    Use Parallel3DGeometry with Volume3DGeometry, or Parallel2DGeometry with
    Volume2DGeometry. detector_padding adds edge pixels per detector side and
    derives a padded scan without modifying either supplied geometry object.
    The requested volume shape is always preserved. Reconstruction uses CUDA
    projectors selected by device_index; NumPy inputs and outputs stay on the
    host while projection and filtering execute on the GPU. CuPy arrays stay
    on the selected GPU. FOURIER_INV accepts CuPy arrays or estimator shapes.
    """

    def __init__(
        self,
        *,
        projection_geometry: Parallel2DGeometry | Parallel3DGeometry,
        volume_geometry: Volume2DGeometry | Volume3DGeometry,
        detector_padding: int = 0,
        device_index: int = 0,
    ):
        self._init_device(device_index)
        self._init_geometry(projection_geometry, volume_geometry, detector_padding)
        wrapper = AstraTools3D if self.geom == "3D" else AstraTools2D
        self.Atools = wrapper(
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=volume_geometry,
            device_index=self.device_index,
        )

    @_on_device(cp)
    def forward(
        self, data: np.ndarray | cp.ndarray, *, output_axes=None, **kwargs
    ) -> np.ndarray | cp.ndarray:
        """Module to perform forward projection of 2d/3d data NumPy or CuPy array

        Args:
            data (np.ndarray or cp.ndarray): 2D or 3D object

        Keyword Args:
            output_axes (list or tuple, optional): Explicit projection axis order.
                Uses the default below when omitted. Conflicting aliases raise ValueError.
            data_axes_labels_order (list or tuple, optional): Legacy alias for output_axes.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray or cp.ndarray: Projection data, matching the input library
        """
        native_axes = (
            ("angles", "detX") if self.geom == "2D" else ("detY", "angles", "detX")
        )
        axes = _resolve_data_axes(
            output_axes, kwargs.get("data_axes_labels_order"), native_axes
        )
        self._ensure_open()
        projected = self.Atools.forward(data)
        projected = self._crop_projection(projected)
        return _transpose_data_axes(
            projected, source_axes=native_axes, target_axes=axes
        )

    @_on_device(cp)
    def backproject(
        self, data: np.ndarray | cp.ndarray, *, input_axes=None, **kwargs
    ) -> np.ndarray | cp.ndarray:
        """Module to perform back-projection of 2d/3d data NumPy or CuPy array

        Args:
            data (np.ndarray or cp.ndarray): 2D/3D projection data

        Keyword Args:
            input_axes (list or tuple, optional): Explicit projection axis order.
                Uses the default below when omitted. Conflicting aliases raise ValueError.
            data_axes_labels_order (list or tuple, optional): Legacy alias for input_axes.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray or cp.ndarray: Backprojected object, matching the input library
        """
        data = self._prepare_projection(
            data,
            array_module=cp,
            input_axes=input_axes,
            legacy_axes=kwargs.get("data_axes_labels_order"),
        )
        return self.Atools.backproject(data)

    @_on_device(cp)
    def FBP(
        self, data, *, input_axes=None, filter_type=None, cutoff_freq=0.35, **kwargs
    ):
        """Filtered backprojection, returning the input array library.

        Projection axes default to (detY, angles, detX) in 3D and
        (angles, detX) in 2D, independently of the filter and array library.
        Use input_axes to supply any other order.

        In 3D, filter_type="SINC" (the default) uses the established SINC
        coefficients. "SINC_CuPy" selects the custom CUDA real-FFT filter;
        cutoff_freq controls this filter only and defaults to 0.35.
        In 2D, filter_type defaults to "ram-lak" and accepts ASTRA FBP_CUDA
        filter names, with filter_parameter and filter_d passed to ASTRA.
        """
        if (
            isinstance(self.projection_geometry, Parallel3DGeometry)
            and not self.projection_geometry.is_standard_tomography
        ):
            raise ValueError(
                "This analytical method requires standard Z-axis tomography; use iterative reconstruction for general vectors or laminography"
            )
        self._ensure_open()
        cupyrun = not isinstance(data, np.ndarray) and isinstance(data, cp.ndarray)
        kwargs["cupyrun"] = cupyrun
        selected_filter = (
            filter_type
            if filter_type is not None
            else ("ram-lak" if self.geom == "2D" else "SINC")
        )
        if self.geom == "3D" and selected_filter not in ("SINC", "SINC_CuPy"):
            raise ValueError("3D FBP filter_type must be 'SINC' or 'SINC_CuPy'")
        if self.geom == "2D" and selected_filter in ("SINC", "SINC_CuPy"):
            raise ValueError("SINC and SINC_CuPy filters require 3D geometry")
        data = self._prepare_projection(
            data,
            array_module=cp,
            input_axes=input_axes,
            legacy_axes=kwargs.get("data_axes_labels_order"),
            gpu=self.geom == "3D",
            pack=True,
            target_axes=(
                ("angles", "detY", "detX") if selected_filter == "SINC_CuPy" else None
            ),
        )
        if self.geom == "2D":
            # ASTRA's algorithm/data API takes host storage, unlike direct_BP.
            self.Atools.fbp_filter_type = selected_filter
            self.Atools.fbp_filter_parameter = kwargs.get("filter_parameter")
            self.Atools.fbp_filter_d = kwargs.get("filter_d")
            host_data = cp.asnumpy(data) if cupyrun else data
            reconstruction = self.Atools._fbp(
                np.ascontiguousarray(host_data, dtype=np.float32)
            )
            if cupyrun:
                reconstruction = cp.asarray(reconstruction)
        else:
            gpu_data = data
            if selected_filter == "SINC_CuPy":
                # This filter divides by shape[0] (the number of views).
                angles_first = gpu_data
                filtered = _filtersinc3D_cupy(angles_first, cutoff=cutoff_freq)
                filtered = cp.ascontiguousarray(filtered.swapaxes(0, 1))
                del angles_first, gpu_data, data
                # Release temporary filter buffers before allocating ASTRA's output.
                cp.fft.config.get_plan_cache().clear()
                cp._default_memory_pool.free_all_blocks()
            else:
                filtered = _filtersinc3D(gpu_data, device_index=self.device_index)
            reconstruction = self.Atools.backproject(filtered)
            if not cupyrun:
                reconstruction = cp.asnumpy(reconstruction)
        return check_kwargs(reconstruction, **kwargs)

    @_on_device(cp)
    def FOURIER_INV(
        self, data: cp.ndarray | Tuple[int, int, int], *, input_axes=None, **kwargs
    ) -> cp.ndarray:
        """Fourier direct inversion in 3D on unequally spaced (also called as NonUniform FFT/NUFFT) grids using CuPy array as an input, see more in
        [NIKITIN2017]_. This implementation is originated from V. Nikitin's CUDA-C implementation:
        https://github.com/nikitinvv/radonusfft and TomoCuPy package.
        See more about the method `here <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction/LPRec3d_tomobar.html>`__.


        Args:
            data (cp.ndarray): projection data as a CuPy array

        Keyword Args:
            input_axes (list or tuple, optional): Explicit projection axis order.
                Uses the default below when omitted. Conflicting aliases raise ValueError.
            data_axes_labels_order (list or tuple, optional): Legacy alias for input_axes.
                 Defaults to ["detY", "angles", "detX"] for this 3D method,
                 including shape tuples supplied to the memory estimator.
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].
            filter_type (str): Filter type, the accepted values are: none, ramp, shepp, cosine, cosine2, hamming, hann, parzen.
            cutoff_freq (float): Cutoff frequency parameter for different filter. The higher values increase resolution and noise.

        Returns:
            cp.ndarray: The NUFFT reconstructed volume as a CuPy array.
        """
        if (
            isinstance(self.projection_geometry, Parallel3DGeometry)
            and not self.projection_geometry.is_standard_tomography
        ):
            raise ValueError(
                "This analytical method requires standard Z-axis tomography; use iterative reconstruction for general vectors or laminography"
            )
        if isinstance(data, tuple):
            # Estimator shapes carry no storage to validate or pack.
            native_axes = ("detY", "angles", "detX")
            axes = _resolve_data_axes(
                input_axes, kwargs.get("data_axes_labels_order"), native_axes
            )
            data = _transpose_data_axes(data, source_axes=axes, target_axes=native_axes)
        else:
            # Fourier kernels apply detector padding during their own FFT setup.
            data = self._prepare_projection(
                data,
                array_module=cp,
                input_axes=input_axes,
                legacy_axes=kwargs.get("data_axes_labels_order"),
                gpu=True,
                pack=True,
                pad_detector=False,
            )
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
            if key == "center_size" and value is not None:
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

        self._ensure_open()
        shape = self.volume_geometry.shape
        if (
            self.geom != "3D"
            or shape[1] != shape[2]
            or shape[0] != self.projection_geometry.detector_shape[0]
        ):
            raise ValueError(
                "FOURIER_INV requires square slices and one volume slice per detector row"
            )
        if np.any(self.projection_geometry.detector_shifts != 0):
            raise ValueError("FOURIER_INV does not support detector_shifts")

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
        mem_stack = DeviceMemStack().instance()
        if mem_stack:
            mem_stack.malloc(np.prod(data) * kwargs["data_dtype"].itemsize)
            [nz, nproj, data_n] = data
        else:
            [nz, nproj, data_n] = data.shape

        recon_size = self.volume_geometry.shape[-1]
        if recon_size > data_n:
            raise ValueError(
                "The reconstruction size {} should not be larger than the size of the horizontal detector {}".format(
                    recon_size, data_n
                )
            )

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

        n = data_n + self.detector_padding * 2 + padding * 2
        if power_of_2_cropping:
            n_pow2 = 2 ** math.ceil(math.log2(n))
            if 0.9 < n / n_pow2:
                n = n_pow2

        # Limit the center size parameter
        center_size = min(center_size, n * 2)

        if mem_stack:
            mem_stack.malloc(
                np.prod(self.projection_geometry.angles.shape) * cp.float32().itemsize
            )

        theta = cp.array(-self.projection_geometry.angles, dtype=cp.float32)

        if center_size >= _CENTER_SIZE_MIN:
            if mem_stack:
                mem_stack.malloc(
                    np.prod(self.projection_geometry.angles.shape) * np.int64().itemsize
                )
                mem_stack.malloc(
                    np.prod(self.projection_geometry.angles.shape)
                    * np.float32().itemsize
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
            tmp_p = self._fbp_filtering_estimator(
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

            if odd_horiz or odd_vert:
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
            datac, fde = self._setup_backprojection_input_estimator(n, nproj, nz)
            tmp_p_shape, tmp_p_dtype = tmp_p
            mem_stack.free(np.prod(tmp_p_shape) * tmp_p_dtype.itemsize)
        else:
            datac, fde = self._setup_backprojection_input(
                tmp_p, n, nproj, nz, center_size, r2c_c1dfftshift
            )

            del tmp_p

        # BACKPROJECTION
        if mem_stack:
            self._fft_and_interpolation_estimator(datac)
            datac_shape, datac_dtype = datac
            mem_stack.free(np.prod(datac_shape) * datac_dtype.itemsize)
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
                sorted_theta if center_size >= _CENTER_SIZE_MIN else None,
                sorted_theta_indices if center_size >= _CENTER_SIZE_MIN else None,
                angle_range_pi_count if center_size >= _CENTER_SIZE_MIN else None,
                angle_range if center_size >= _CENTER_SIZE_MIN else None,
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
                fde, nz, chunk_count, min_mem_usage_ifft2
            )
        else:
            self.ifft_gathered_projections(
                fde, n, nz, chunk_count, min_mem_usage_ifft2, c2dfftshift
            )

        # Unpadded recon output size
        if mem_stack:
            recon_up_shape = self.unpad_reconstructed_data_estimator(
                n, nz, odd_horiz, odd_vert, recon_size
            )

            fde_shape, fde_dtype = fde
            mem_stack.free(np.prod(fde_shape) * fde_dtype.itemsize)
        else:
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

        if mem_stack:
            mem_stack.malloc(np.prod(recon_up_shape[1:]) * cp.float32().itemsize)
            mem_stack.malloc(np.prod(recon_up_shape) * cp.float32().itemsize)
            mem_stack.free(np.prod(recon_up_shape) * cp.float32().itemsize)
            mem_stack.free(np.prod(recon_up_shape[1:]) * cp.float32().itemsize)
            return recon_up_shape

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

        rotation_axis = self.projection_geometry.centre_of_rotation + 0.5

        wfilter = calc_filter(oversampled_detector_width, filter_type, cutoff_freq)
        t = rfftfreq(oversampled_detector_width).astype(cp.float32)
        w = wfilter * cp.exp(-2 * cp.pi * 1j * t * (rotation_axis))

        # FBP filtering output
        tmp_p = cp.empty(
            (detector_height, projection_count, detector_width), dtype=cp.float32
        )

        if min_mem_usage_filter:
            filter_vol_chunk_count = detector_height

        slice_count_per_chunk = int(np.ceil(detector_height / filter_vol_chunk_count))
        projection_count_per_projection_chunk = int(
            np.ceil(projection_count / filter_proj_chunk_count)
        )
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
    ) -> tuple:
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

        mem_stack = DeviceMemStack.instance()
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.malloc(
            (oversampled_detector_width // 2 + 1) * np.complex64().itemsize
        )

        # FBP filtering output
        tmp_p_shape = (detector_height, projection_count, detector_width)
        tmp_p_dtype = cp.float32()
        mem_stack.malloc(np.prod(tmp_p_shape) * tmp_p_dtype.itemsize)

        if min_mem_usage_filter:
            filter_vol_chunk_count = detector_height

        slice_count_per_chunk = int(np.ceil(detector_height / filter_vol_chunk_count))
        projection_count_per_projection_chunk = int(
            np.ceil(projection_count / filter_proj_chunk_count)
        )
        # Loop over the chunks
        rfft_input = cp.empty(
            (
                slice_count_per_chunk,
                projection_count_per_projection_chunk,
                oversampled_detector_width,
            ),
            cp.float32,
        )
        rfft_input_nbytes = rfft_input.nbytes
        mem_stack.malloc(rfft_input_nbytes)

        rfft_plan = get_fft_plan(rfft_input, axes=(2), value_type="R2C")
        rfft_plan_work_area_mem_size = rfft_plan.work_area.mem.size
        del rfft_input, rfft_plan
        mem_stack.malloc(rfft_plan_work_area_mem_size)

        rfft_output = cp.empty(
            (
                slice_count_per_chunk,
                projection_count_per_projection_chunk,
                oversampled_detector_width // 2 + 1,
            ),
            cp.complex64,
        )
        rfft_output_nbytes = rfft_output.nbytes
        mem_stack.malloc(rfft_output_nbytes)
        mem_stack.free(rfft_output_nbytes)

        irfft_plan = get_fft_plan(rfft_output, axes=(2), value_type="C2R")
        irfft_plan_work_area_mem_size = irfft_plan.work_area.mem.size
        del rfft_output, irfft_plan
        mem_stack.malloc(irfft_plan_work_area_mem_size)

        irfft_output_size = (
            np.prod(
                (
                    slice_count_per_chunk,
                    projection_count_per_projection_chunk,
                    oversampled_detector_width,
                )
            )
            * cp.float32().itemsize
        )

        mem_stack.malloc(irfft_output_size)
        mem_stack.free(irfft_output_size)
        mem_stack.free(rfft_input_nbytes)

        # Memory clean up of filter data
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.complex64().itemsize)

        return (tmp_p_shape, tmp_p_dtype)

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
    ):
        mem_stack = DeviceMemStack.instance()
        datac_shape = (detector_height // 2, projection_count, detector_width)
        datac_dtype = cp.complex64()
        fde_shape = (detector_height // 2, 2 * detector_width, 2 * detector_width)
        fde_dtype = cp.complex64()
        mem_stack.malloc(np.prod(datac_shape) * datac_dtype.itemsize)
        mem_stack.malloc(np.prod(fde_shape) * fde_dtype.itemsize)

        return ((datac_shape, datac_dtype), (fde_shape, fde_dtype))

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
        sorted_theta: cp.ndarray | None,
        sorted_theta_indices: cp.ndarray | None,
        angle_range_pi_count: int | None,
        angle_range: cp.ndarray | None,
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
        datac: tuple,
    ):
        mem_stack = DeviceMemStack.instance()
        datac_shape, datac_dtype = datac
        fft_input = cp.empty(datac_shape, datac_dtype)
        mem_stack.malloc(fft_input.nbytes)

        fft_plan = get_fft_plan(fft_input, axes=(-1))
        mem_stack.malloc(fft_plan.work_area.mem.size)
        mem_stack.free(fft_plan.work_area.mem.size)
        del fft_input, fft_plan

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

        slice_count_per_chunk = int(np.ceil(detector_height // 2 / chunk_count))
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
        fde: tuple,
        detector_height: int,
        chunk_count: int,
        min_mem_usage_ifft2: bool,
    ):
        mem_stack = DeviceMemStack.instance()

        if min_mem_usage_ifft2:
            chunk_count = detector_height // 2

        slice_count_per_chunk = int(np.ceil(detector_height // 2 / chunk_count))
        fde_shape, fde_dtype = fde
        ifft2_input = cp.empty((slice_count_per_chunk, *fde_shape[1:]), fde_dtype)

        ifft2_plan = get_fft_plan(ifft2_input, axes=(-2, -1))
        mem_stack.malloc(ifft2_plan.work_area.mem.size)
        mem_stack.malloc(ifft2_input.nbytes)
        mem_stack.free(ifft2_input.nbytes)
        del ifft2_input, ifft2_plan

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
    ):
        odd_recon_size = bool(recon_size % 2)
        unpad_z = detector_height - odd_vert
        unpad_recon_m = (detector_width - odd_horiz) // 2 - recon_size // 2
        unpad_recon_p = (detector_width - odd_horiz) // 2 + (
            recon_size + odd_recon_size
        ) // 2
        unpad_recon_size = unpad_recon_p - unpad_recon_m
        recon_up_shape = (unpad_z, unpad_recon_size, unpad_recon_size)

        # memory for recon
        mem_stack = DeviceMemStack.instance()
        mem_stack.malloc(np.prod(recon_up_shape) * cp.float32().itemsize)

        return recon_up_shape


def _filtersinc3D(projection3D: np.ndarray, *, device_index: int = 0):
    """Applies a 3D filter to 3D projection data for FBP

    Args:
        projection3D (np.ndarray or cp.ndarray): (detY, angles, detX) projection data

    Returns:
        cp.ndarray: Filtered data on the selected GPU
    """
    a = 1.1
    [DetectorsLengthV, projectionsNum, DetectorsLengthH] = np.shape(projection3D)
    w = np.linspace(
        -np.pi,
        np.pi - (2 * np.pi) / DetectorsLengthH,
        DetectorsLengthH,
        dtype="float32",
    )

    rn1 = np.abs(2.0 / a * np.sin(a * w / 2.0))
    rn2 = np.sin(a * w / 2.0)
    rd = (a * w) / 2.0
    rd_c = np.zeros([1, DetectorsLengthH])
    rd_c[0, :] = rd
    r = rn1 * (np.dot(rn2, np.linalg.pinv(rd_c))) ** 2
    multiplier = 1.0 / projectionsNum
    # Retain the established filter coefficients; run the batched FFT on CUDA.

    with cp.cuda.Device(device_index):
        data = cp.asarray(projection3D, dtype=cp.float32)
        spectrum = cp.fft.fft2(data, axes=(0, 2))
        spectrum *= cp.asarray(np.fft.fftshift(r), dtype=cp.float32)[None, None, :]
        filtered = cp.fft.ifft2(spectrum, axes=(0, 2)).real
        return multiplier * filtered
