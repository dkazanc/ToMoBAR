import cupy as cp
import numpy as np
import math
from tomobar.cuda_kernels import load_cuda_module


def unpad_reconstructed_data(
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
    recon_up = cp.empty([unpad_z, unpad_recon_size, unpad_recon_size], dtype=cp.float32)

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


class FFTProjectorCuPy:
    """USFFT-based forward and backward projection wrapper."""

    def __init__(
        self,
        nx: int,
        nz: int,
        AnglesVec: np.ndarray,
        CenterRotOffset: float,
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.AnglesVec = AnglesVec
        self.CenterRotOffset = CenterRotOffset

        # Initialize kernels

        usfft_kernel_module = load_cuda_module("fft_us_kernels")
        self.gather_kernel = usfft_kernel_module.get_function("gather_kernel")
        self.scatter_kernel = usfft_kernel_module.get_function("scatter_kernel")
        self.gather_kernel_partial = usfft_kernel_module.get_function(
            "gather_kernel_partial"
        )
        self.gather_kernel_center = usfft_kernel_module.get_function(
            "gather_kernel_center"
        )
        self.gather_kernel_center_angle_based_prune = usfft_kernel_module.get_function(
            "gather_kernel_center_angle_based_prune"
        )
        self.unpadding_mul_phi = usfft_kernel_module.get_function("unpadding_mul_phi")

        # Additional parameters

        eps = 1e-3  # accuracy of usfft
        self.mu = -math.log(eps) / (2 * nx * nx)
        self.m = math.ceil(
            2
            * nx
            * 1
            / math.pi
            * math.sqrt(-self.mu * math.log(eps) + (self.mu * nx) * (self.mu * nx) / 4)
        )
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, nx, endpoint=False, dtype="float32")
        [dx, dy] = cp.meshgrid(t, t)
        self.phi = cp.exp(
            (self.mu * (nx * nx) * (dx * dx + dy * dy)).astype("float32")
        ) * (1 - nx % 4)

        # (+1,-1) arrays for fftshift
        self.c1dfftshift = (1 - 2 * ((cp.arange(1, nx + 1) % 2))).astype("int8")
        c2dtmp = 1 - 2 * ((cp.arange(1, 2 * nx + 1) % 2)).astype("int8")
        self.c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        # create mask
        x = cp.linspace(-1, 1, nx)
        [x, y] = cp.meshgrid(x, x)
        mask_r = 1
        mask = (x**2 + y**2 < mask_r).astype("float32")
        # normalization, incorporate mask in phi for optimization
        self.phi *= mask / (cp.float32(4 * nx) * cp.sqrt(nx * AnglesVec.size))
        self.center_size = max(nx - 256, 0)
        if self.center_size > 0:
            theta_full_range = abs(self.AnglesVec[-1] - self.AnglesVec[0])
            self.angle_range_pi_count = 1 + int(np.ceil(theta_full_range / math.pi))
            self.angle_range = cp.zeros(
                [self.center_size, self.center_size, 1 + self.angle_range_pi_count * 2],
                dtype=cp.uint16,
            )
            self.gather_kernel_center_angle_based_prune(
                (int(np.ceil(self.center_size / 256)), self.center_size, 1),
                (256, 1, 1),
                (
                    self.angle_range,
                    np.int32(self.angle_range_pi_count * 2 + 1),
                    cp.asarray(self.AnglesVec),
                    np.int32(self.m),
                    np.int32(self.center_size),
                    np.int32(nx),
                    np.int32(self.AnglesVec.size),
                ),
            )

    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray:
        slices, y, x = object3D.shape
        assert slices == self.nz
        assert y == self.nx
        assert x == self.nx
        sino = cp.zeros([slices, self.AnglesVec.size, x], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = object3D * self.phi
        fde = cp.pad(fde, ((0, 0), (x // 2, x // 2), (x // 2, x // 2)))
        # STEP1: fft 2d
        fde = cp.fft.fft2(fde * self.c2dfftshift) * self.c2dfftshift

        self.scatter_kernel(
            (math.ceil(x / 32), math.ceil(self.AnglesVec.size / 32), slices),
            (32, 32, 1),
            (
                sino,
                fde,
                cp.asarray(self.AnglesVec),
                np.int32(self.m),
                np.float32(self.mu),
                np.int32(x),
                np.int32(self.AnglesVec.size),
                np.int32(slices),
            ),
        )
        # STEP3: ifft 1d
        sino = cp.fft.ifft(self.c1dfftshift * sino) * self.c1dfftshift

        # STEP4: Shift based on the rotation axis, not needed if rotation_axis==n/2
        t = cp.fft.fftfreq(x).astype("float32")
        w = cp.exp(2 * cp.pi * 1j * t * self.CenterRotOffset)
        sino = cp.fft.ifft(w * cp.fft.fft(sino))
        return cp.flip(sino.real, axis=1)

    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray:
        vert, projs, hori = proj_data.shape
        assert vert == self.nz
        assert projs == self.AnglesVec.size
        assert proj_data.dtype == "float32"
        unpad = (proj_data.shape[2] - self.nx) // 2
        if unpad > 0:
            proj_data = proj_data[:, :, unpad:-unpad]
        assert proj_data.shape[2] == self.nx

        t = cp.fft.fftfreq(self.nx).astype("float32")
        w = cp.exp(-2 * cp.pi * 1j * t * self.CenterRotOffset)
        proj_data = cp.fft.ifft(w * cp.fft.fft(proj_data))
        sino = cp.fft.fft(self.c1dfftshift * proj_data) * self.c1dfftshift
        fde = cp.zeros([vert, 2 * self.nx, 2 * self.nx], dtype="complex64")
        if self.center_size > 0:
            theta = cp.asarray(self.AnglesVec)
            self.gather_kernel_partial(
                (
                    int(math.ceil(self.nx / 32)),
                    int(math.ceil(projs / 32)),
                    vert,
                ),
                (32, 32, 1),
                (
                    sino,
                    fde,
                    theta,
                    np.int32(self.m),
                    np.float32(self.mu),
                    np.int32(self.center_size),
                    np.int32(self.nx),
                    np.int32(projs),
                    np.int32(vert),
                ),
            )
            sorted_theta_indices = cp.argsort(theta)
            self.gather_kernel_center(
                (math.ceil(self.nx / 32), math.ceil(projs / 32), vert),
                (32, 32, 1),
                (
                    sino,
                    fde,
                    self.angle_range,
                    np.int32(self.angle_range_pi_count * 2 + 1),
                    theta,
                    sorted_theta_indices,
                    np.int32(self.m),
                    np.float32(self.mu),
                    np.int32(self.center_size),
                    np.int32(self.nx),
                    np.int32(projs),
                    np.int32(vert),
                ),
            )
        else:
            self.gather_kernel(
                (
                    int(math.ceil(self.nx / 32)),
                    int(math.ceil(projs / 32)),
                    vert,
                ),
                (32, 32, 1),
                (
                    sino,
                    fde,
                    cp.asarray(self.AnglesVec),
                    np.int32(self.m),
                    np.float32(self.mu),
                    np.int32(self.nx),
                    np.int32(projs),
                    np.int32(vert),
                ),
            )
        fde = cp.fft.ifft2(fde * self.c2dfftshift) * self.c2dfftshift
        fde = fde[
            :, self.nx // 2 : 3 * self.nx // 2, self.nx // 2 : 3 * self.nx // 2
        ] * (self.phi * 4)
        return cp.flip(fde.real, axis=2)
        recon_up = unpad_reconstructed_data(
            fde,
            hori,
            self.AnglesVec.size,
            self.nz,
            False,
            False,
            self.nx,
            self.mu,
            self.unpadding_mul_phi,
        )
        return recon_up
