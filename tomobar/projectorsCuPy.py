from abc import ABC, abstractmethod
import cupy as cp
import numpy as np
import math
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.cuda_kernels import load_cuda_module


class ProjectorBase(ABC):
    @abstractmethod
    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray: ...  # pragma: nocover

    @abstractmethod
    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray: ...  # pragma: nocover


class AstraProjectorCuPy(ProjectorBase):
    def __init__(self, atools: AstraTools3D):
        self.atools = atools

    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray:
        return self.atools._forwprojCuPy(object3D)

    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray:
        return self.atools._backprojCuPy(proj_data)


class FFTProjectorCuPy(ProjectorBase):
    """USFFT-based forward and backward projection wrapper."""

    def __init__(
        self,
        n: int,
        theta: cp.ndarray,
        mask_r: float,
        detector_x: int,
        CenterRotOffset: int,
    ):
        """Usfft parameters
        mask_r - circle radius"""

        eps = 1e-3  # accuracy of usfft
        mu = -math.log(eps) / (2 * n * n)
        m = math.ceil(
            2
            * n
            * 1
            / math.pi
            * math.sqrt(-mu * math.log(eps) + (mu * n) * (mu * n) / 4)
        )
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        [dx, dy] = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (
            1 - n % 4
        )

        # (+1,-1) arrays for fftshift
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp = 1 - 2 * ((cp.arange(1, 2 * n + 1) % 2)).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        # create mask
        x = cp.linspace(-1, 1, n)
        [x, y] = cp.meshgrid(x, x)
        mask = (x**2 + y**2 < mask_r).astype("float32")
        # normalization, incorporate mask in phi for optimization
        phi *= mask / (cp.float32(4 * n) * cp.sqrt(n * len(theta)))

        self.n = n
        self.ntheta = len(theta)
        self.theta = cp.sort(-1 * cp.asarray(theta, dtype="float32"))
        self.mask = mask
        self.raxis = n // 2 - CenterRotOffset
        self.pars = m, mu, phi, c1dfftshift, c2dfftshift
        self.detector_x = detector_x
        self.left_pad = (self.detector_x - self.n) // 2
        self.right_pad = self.detector_x - self.n - self.left_pad
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
        self.center_size = max(n - 256, 0)
        if self.center_size > 0:
            theta_full_range = abs(self.theta[-1] - self.theta[0])
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
                    cp.asarray(self.theta),
                    np.int32(m),
                    np.int32(self.center_size),
                    np.int32(n),
                    np.int32(self.theta.size),
                ),
            )

    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray:
        """Radon transform"""
        [nz, n, n] = object3D.shape
        assert self.n == n

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars
        sino = cp.zeros([nz, self.ntheta, n], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = object3D * phi
        fde = cp.pad(fde, ((0, 0), (n // 2, n // 2), (n // 2, n // 2)))
        # STEP1: fft 2d
        fde = cp.fft.fft2(fde * c2dfftshift) * c2dfftshift

        self.scatter_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, cp.float32(mu), n, self.ntheta, nz),
        )
        # STEP3: ifft 1d
        sino = cp.fft.ifft(c1dfftshift * sino) * c1dfftshift

        # STEP4: Shift based on the rotation axis, not needed if rotation_axis==n/2
        if self.raxis is not None:
            t = cp.fft.fftfreq(n).astype("float32")
            w = cp.exp(2 * cp.pi * 1j * t * (-self.raxis + n / 2))
            sino = cp.fft.ifft(w * cp.fft.fft(sino))
        # return cp.flip(cp.flip(sino.real, axis=2), axis=1)
        return cp.pad(
            sino.real,
            pad_width=[(0, 0), (0, 0), (self.left_pad, self.right_pad)],
            mode="constant",
            constant_values=0,
        )

    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray:
        """Adjoint Radon transform"""

        assert proj_data.shape[1] == self.theta.size
        if proj_data.shape[2] != self.n:
            proj_data = proj_data[..., self.left_pad : -self.right_pad]
        [nz, ntheta, n] = proj_data.shape
        assert self.n == n

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: Shift based on the rotation axis, not  needed if rotation_axis==n/2
        if self.raxis is not None:
            t = cp.fft.fftfreq(n).astype("float32")
            w = cp.exp(-2 * cp.pi * 1j * t * (-self.raxis + n / 2))
            proj_data = cp.fft.ifft(w * cp.fft.fft(proj_data))

        # STEP1: fft 1d
        sino = cp.fft.fft(c1dfftshift * proj_data) * c1dfftshift

        # STEP2: interpolation (gathering) in the frequency domain
        fde = cp.zeros([nz, 2 * n, 2 * n], dtype="complex64")
        if self.center_size > 0:
            self.gather_kernel_partial(
                (
                    int(math.ceil(self.n / 32)),
                    int(math.ceil(ntheta / 32)),
                    nz,
                ),
                (32, 32, 1),
                (
                    sino,
                    fde,
                    self.theta,
                    np.int32(m),
                    np.float32(mu),
                    np.int32(self.center_size),
                    np.int32(self.n),
                    np.int32(ntheta),
                    np.int32(nz),
                ),
            )
            sorted_theta_indices = cp.argsort(self.theta)
            self.gather_kernel_center(
                (
                    math.ceil(self.n / 32),
                    math.ceil(ntheta / 32),
                    nz,
                ),
                (32, 32, 1),
                (
                    sino,
                    fde,
                    self.angle_range,
                    np.int32(self.angle_range_pi_count * 2 + 1),
                    self.theta,
                    sorted_theta_indices,
                    np.int32(m),
                    np.float32(mu),
                    np.int32(self.center_size),
                    np.int32(self.n),
                    np.int32(ntheta),
                    np.int32(nz),
                ),
            )
        else:
            self.gather_kernel(
                (
                    int(math.ceil(self.n / 32)),
                    int(math.ceil(ntheta / 32)),
                    nz,
                ),
                (32, 32, 1),
                (
                    sino,
                    fde,
                    self.theta,
                    np.int32(m),
                    np.float32(mu),
                    np.int32(self.n),
                    np.int32(ntheta),
                    np.int32(nz),
                ),
            )  # STEP3: ifft 2d
        fde = cp.fft.ifft2(fde * c2dfftshift) * c2dfftshift * self.n

        # STEP4: unpadding, multiplication by phi
        fde = fde[:, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * (phi * 4)

        # return cp.flip(fde.real, axis=1)
        return fde.real
