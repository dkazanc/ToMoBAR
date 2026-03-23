import cupy as cp
import math
from tomobar.cuda_kernels import load_cuda_module


class ProjectorsCuPy:
    def __init__(self, n: int, theta: cp.ndarray, mask_r: float, raxis=None):
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
        self.theta = theta
        self.mask = mask
        self.raxis = raxis
        self.pars = m, mu, phi, c1dfftshift, c2dfftshift
        self.gather_kernel = load_cuda_module("fft_us_kernels").get_function("gather")

    def fwd_tomo(self, obj):
        """Radon transform"""
        [nz, n, n] = obj.shape

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars
        sino = cp.zeros([nz, self.ntheta, n], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = obj * phi
        fde = cp.pad(fde, ((0, 0), (n // 2, n // 2), (n // 2, n // 2)))
        # STEP1: fft 2d
        fde = cp.fft.fft2(fde * c2dfftshift) * c2dfftshift

        mua = cp.array([mu], dtype="float32")

        self.gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, mua, n, self.ntheta, nz, 0),
        )
        # STEP3: ifft 1d
        sino = cp.fft.ifft(c1dfftshift * sino) * c1dfftshift

        # STEP4: Shift based on the rotation axis, not needed if rotation_axis==n/2
        if self.raxis is not None:
            t = cp.fft.fftfreq(n).astype("float32")
            w = cp.exp(2 * cp.pi * 1j * t * (-self.raxis + n / 2))
            sino = cp.fft.ifft(w * cp.fft.fft(sino))
        # return cp.flip(cp.flip(sino.real, axis=2), axis=1)
        return sino.real

    def adj_tomo(self, data):
        """Adjoint Radon transform"""

        [nz, ntheta, n] = data.shape

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: Shift based on the rotation axis, not  needed if rotation_axis==n/2
        if self.raxis is not None:
            t = cp.fft.fftfreq(n).astype("float32")
            w = cp.exp(-2 * cp.pi * 1j * t * (-self.raxis + n / 2))
            data = cp.fft.ifft(w * cp.fft.fft(data))

        # STEP1: fft 1d
        sino = cp.fft.fft(c1dfftshift * data) * c1dfftshift

        # STEP2: interpolation (gathering) in the frequency domain
        mua = cp.array([mu], dtype="float32")
        fde = cp.zeros([nz, 2 * n, 2 * n], dtype="complex64")
        self.gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, mua, n, self.ntheta, nz, 1),
        )
        # STEP3: ifft 2d
        fde = cp.fft.ifft2(fde * c2dfftshift) * c2dfftshift * self.n

        # STEP4: unpadding, multiplication by phi
        fde = fde[:, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * (phi * 4)

        # return cp.flip(fde.real, axis=1)
        return fde.real
