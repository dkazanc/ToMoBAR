"""CUDA Fourier execution and workspace tracing consume one CPU-built plan."""

from types import SimpleNamespace
import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft, ifft2, rfftfreq, rfft, irfft
from cupyx.scipy.fftpack import get_fft_plan
from tomobar.cuda_kernels import get_cuda_kernel, load_cuda_module
from tomobar.fourier import calc_filter
from tomobar.fourier_plan import CENTER_SIZE_MIN as _CENTER_SIZE_MIN
from tomobar.supp.suppTools import check_kwargs


class FourierExecutor:
    def __init__(self, projection_geometry, plan):
        self.projection_geometry = projection_geometry
        self.plan = plan
        self.mem_stack = None

    def _kernels(self):
        module = load_cuda_module("fft_us_kernels")
        return SimpleNamespace(
            **{
                name: get_cuda_kernel(module, name)
                for name in (
                    "gather_kernel",
                    "gather_kernel_partial",
                    "gather_kernel_center_angle_based_prune",
                    "gather_kernel_center",
                    "r2c_c1dfftshift",
                    "c1dfftshift",
                    "c2dfftshift",
                    "unpadding_mul_phi",
                )
            }
        )

    def execute(self, data):
        p = self.plan
        o = p.options
        nz, _, data_n = p.even_shape
        center_size = p.center_size
        kernels = self._kernels()
        if p.odd_horiz or p.odd_vert:
            padded = cp.zeros(p.even_shape, dtype=cp.float32)
            padded[: nz - p.odd_vert, :, : data_n - p.odd_horiz] = data
            padded[: nz - p.odd_vert, :, -p.odd_horiz] = data[..., -p.odd_horiz]
            data = padded
            del padded
        theta = cp.array(p.theta, dtype=cp.float32)
        sorted_theta = sorted_theta_indices = angle_range = None
        if center_size >= _CENTER_SIZE_MIN:
            sorted_theta_indices = cp.argsort(theta)
            sorted_theta = theta[sorted_theta_indices]
            angle_range = cp.zeros(p.workspaces["angle_ranges"].shape, dtype=cp.uint16)
        tmp_p = self._fbp_filtering(data)
        del data
        datac, fde = self._setup_backprojection_input(tmp_p, kernels.r2c_c1dfftshift)
        del tmp_p
        self._fft_and_interpolation(
            datac, fde, theta, sorted_theta, sorted_theta_indices, angle_range, kernels
        )
        del datac
        self.ifft_gathered_projections(fde, kernels.c2dfftshift)
        result = self.unpad_reconstructed_data(fde, kernels.unpadding_mul_phi)
        del fde
        return check_kwargs(result, recon_mask_radius=o.recon_mask_radius, cupyrun=True)

    def estimate(self, mem_stack):
        """Trace existing memory-accounting semantics; query only FFT work areas.

        FFT plan work-area sizes require CUDA. Dimension/shape planning does not.
        No interpolation kernels are compiled or launched by this path.
        """
        self.mem_stack = mem_stack
        p = self.plan
        mem_stack.malloc(p.workspaces["input"].nbytes)
        if p.odd_horiz or p.odd_vert:
            mem_stack.malloc(p.workspaces["even_input"].nbytes)
        mem_stack.malloc(p.workspaces["angles"].nbytes)
        if p.center_size >= _CENTER_SIZE_MIN:
            mem_stack.malloc(p.workspaces["angle_order"].nbytes)
            mem_stack.malloc(p.workspaces["angles"].nbytes)
            mem_stack.malloc(p.workspaces["angle_ranges"].nbytes)
        self._fbp_filtering_estimator()
        if p.odd_horiz or p.odd_vert:
            mem_stack.free(p.workspaces["even_input"].nbytes)
        datac, fde = self._setup_backprojection_input_estimator()
        mem_stack.free(p.workspaces["filtered"].nbytes)
        self._fft_and_interpolation_estimator(datac)
        mem_stack.free(p.workspaces["complex_projections"].nbytes)
        self.ifft_gathered_projections_estimator(fde)
        shape = self.unpad_reconstructed_data_estimator()
        mem_stack.free(p.workspaces["grid"].nbytes)
        # Preserve the legacy conservative output/mask workspace accounting.
        mask_bytes = np.prod(shape[1:]) * np.dtype(np.float32).itemsize
        output_bytes = p.workspaces["output"].nbytes
        mem_stack.malloc(mask_bytes)
        mem_stack.malloc(output_bytes)
        mem_stack.free(output_bytes)
        mem_stack.free(mask_bytes)
        return shape

    def _fbp_filtering(self, data: cp.ndarray):
        filter_type = self.plan.options.filter_type
        cutoff_freq = self.plan.options.cutoff_freq
        oversampled_detector_width = self.plan.oversampled_width
        padding_m = self.plan.padding_m
        unpad_m, unpad_p = self.plan.filter_crop

        rotation_axis = self.projection_geometry.centre_of_rotation + 0.5

        wfilter = calc_filter(oversampled_detector_width, filter_type, cutoff_freq)
        t = rfftfreq(oversampled_detector_width).astype(cp.float32)
        w = wfilter * cp.exp(-2 * cp.pi * 1j * t * (rotation_axis))

        # FBP filtering output
        tmp_p = cp.empty(self.plan.workspaces["filtered"].shape, dtype=cp.float32)

        for slice_start_index, slice_end_index in self.plan.filter_slices:
            for (
                projection_start_index,
                projection_end_index,
            ) in self.plan.filter_projections:
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

    def _fbp_filtering_estimator(self):
        oversampled_detector_width = self.plan.oversampled_width
        mem_stack = self.mem_stack
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.malloc((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.malloc(
            (oversampled_detector_width // 2 + 1) * np.complex64().itemsize
        )

        # FBP filtering output
        tmp_p_shape = self.plan.workspaces["filtered"].shape
        tmp_p_dtype = cp.float32()
        mem_stack.malloc(np.prod(tmp_p_shape) * tmp_p_dtype.itemsize)

        rfft_input = cp.empty(
            self.plan.workspaces["filter_fft_input"].shape, cp.float32
        )
        rfft_input_nbytes = rfft_input.nbytes
        mem_stack.malloc(rfft_input_nbytes)

        rfft_plan = get_fft_plan(rfft_input, axes=(2), value_type="R2C")
        rfft_plan_work_area_mem_size = rfft_plan.work_area.mem.size
        del rfft_input, rfft_plan
        mem_stack.malloc(rfft_plan_work_area_mem_size)

        rfft_output = cp.empty(
            self.plan.workspaces["filter_fft_output"].shape, cp.complex64
        )
        rfft_output_nbytes = rfft_output.nbytes
        mem_stack.malloc(rfft_output_nbytes)
        mem_stack.free(rfft_output_nbytes)

        irfft_plan = get_fft_plan(rfft_output, axes=(2), value_type="C2R")
        irfft_plan_work_area_mem_size = irfft_plan.work_area.mem.size
        del rfft_output, irfft_plan
        mem_stack.malloc(irfft_plan_work_area_mem_size)

        irfft_output_size = self.plan.workspaces["filter_fft_input"].nbytes

        mem_stack.malloc(irfft_output_size)
        mem_stack.free(irfft_output_size)
        mem_stack.free(rfft_input_nbytes)

        # Memory clean up of filter data
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.float32().itemsize)
        mem_stack.free((oversampled_detector_width // 2 + 1) * np.complex64().itemsize)

    def _setup_backprojection_input(
        self, tmp_p: cp.ndarray, r2c_c1dfftshift: cp.RawKernel
    ):
        detector_height, projection_count, _ = self.plan.even_shape
        detector_width = self.plan.detector_width
        center_size = self.plan.center_size
        datac = cp.empty(
            self.plan.workspaces["complex_projections"].shape,
            dtype=cp.complex64,
        )

        # Only the full-centre kernel overwrites every grid element. Both the
        # original and partial gather kernels use atomicAdd and must start at
        # zero, including points outside their interpolation support.
        if center_size >= _CENTER_SIZE_MIN and center_size == 2 * detector_width:
            fde = cp.empty(
                self.plan.workspaces["grid"].shape,
                dtype=cp.complex64,
            )
        else:
            fde = cp.zeros(
                self.plan.workspaces["grid"].shape,
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

    def _setup_backprojection_input_estimator(self):
        mem_stack = self.mem_stack
        datac_shape = self.plan.workspaces["complex_projections"].shape
        datac_dtype = cp.complex64()
        fde_shape = self.plan.workspaces["grid"].shape
        fde_dtype = cp.complex64()
        mem_stack.malloc(np.prod(datac_shape) * datac_dtype.itemsize)
        mem_stack.malloc(np.prod(fde_shape) * fde_dtype.itemsize)

        return ((datac_shape, datac_dtype), (fde_shape, fde_dtype))

    def _fft_and_interpolation(
        self,
        datac: cp.ndarray,
        fde: cp.ndarray,
        theta,
        sorted_theta,
        sorted_theta_indices,
        angle_range,
        kernels,
    ):
        detector_height, projection_count, _ = self.plan.even_shape
        detector_width = self.plan.detector_width
        center_size = self.plan.center_size
        block_dim = self.plan.options.block_dim
        block_dim_center = self.plan.options.block_dim_center
        angle_range_pi_count = self.plan.angle_range_pi_count
        mu = self.plan.mu
        datac = fft(datac)

        m = self.plan.interpolation_radius

        kernels.c1dfftshift(
            (
                int(np.ceil(detector_width / 32)),
                int(np.ceil(projection_count / 32)),
                np.int32(detector_height // 2),
            ),
            (32, 32, 1),
            (
                datac,
                # The angular quadrature for a half-turn is pi/2 for the
                # existing doubled ramp filter. The former factor 4 inflated
                # physical amplitudes by 8/pi.
                np.float32(
                    (4 if self.plan.options.legacy_scaling else np.pi / 2)
                    / detector_width
                ),
                detector_width,
                projection_count,
                detector_height // 2,
            ),
        )

        # STEP2: interpolation (gathering) in the frequency domain
        # Use original kernel at low dimension.

        if center_size >= _CENTER_SIZE_MIN:
            if center_size != (detector_width * 2):
                kernels.gather_kernel_partial(
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

            kernels.gather_kernel_center_angle_based_prune(
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

            kernels.gather_kernel_center(
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
            kernels.gather_kernel(
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
        mem_stack = self.mem_stack
        datac_shape, datac_dtype = datac
        fft_input = cp.empty(datac_shape, datac_dtype)
        mem_stack.malloc(fft_input.nbytes)

        fft_plan = get_fft_plan(fft_input, axes=(-1))
        mem_stack.malloc(fft_plan.work_area.mem.size)
        mem_stack.free(fft_plan.work_area.mem.size)
        del fft_input, fft_plan

    def ifft_gathered_projections(self, fde: cp.ndarray, c2dfftshift: cp.RawKernel):
        detector_height = self.plan.even_shape[0]
        detector_width = self.plan.detector_width
        c2dfftshift(
            (
                int(np.ceil((2 * detector_width) / 32)),
                int(np.ceil((2 * detector_width) / 8)),
                np.int32(detector_height // 2),
            ),
            (32, 8, 1),
            (fde, detector_width, detector_height // 2),
        )

        for start_index, end_index in self.plan.ifft_slices:
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

    def ifft_gathered_projections_estimator(self, fde: tuple):
        mem_stack = self.mem_stack

        fde_shape, fde_dtype = fde
        ifft2_input = cp.empty(self.plan.workspaces["ifft_input"].shape, fde_dtype)

        ifft2_plan = get_fft_plan(ifft2_input, axes=(-2, -1))
        mem_stack.malloc(ifft2_plan.work_area.mem.size)
        mem_stack.malloc(ifft2_input.nbytes)
        mem_stack.free(ifft2_input.nbytes)
        del ifft2_input, ifft2_plan

    def unpad_reconstructed_data(
        self, fde: cp.ndarray, unpadding_mul_phi: cp.RawKernel
    ):
        detector_height, projection_count, _ = self.plan.even_shape
        detector_width = self.plan.detector_width
        mu = self.plan.mu
        unpad_z = self.plan.output_shape[0]
        unpad_recon_m, unpad_recon_p = self.plan.output_crop
        unpad_recon_size = self.plan.output_shape[2]

        # memory for recon
        recon_up = cp.empty(self.plan.output_shape, dtype=cp.float32)

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

    def unpad_reconstructed_data_estimator(self):
        workspace = self.plan.workspaces["output"]
        self.mem_stack.malloc(workspace.nbytes)
        return workspace.shape
