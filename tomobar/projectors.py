from abc import ABC, abstractmethod
from typing import Optional
import cupy as cp
from cupyx.scipy.fft import fft, ifft2
import numpy as np
import math
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.cuda_kernels import load_cuda_module
from tomobar.supp.suppTools import _apply_horiz_detector_padding


class ProjectorBase(ABC):
    @abstractmethod
    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray: ...  # pragma: nocover

    @abstractmethod
    def forwprojOS(
        self, object3D: cp.ndarray, sub_ind: int
    ) -> cp.ndarray: ...  # pragma: nocover

    @abstractmethod
    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray: ...  # pragma: nocover

    @abstractmethod
    def backprojOS(
        self, proj_data: cp.ndarray, sub_ind: int
    ) -> cp.ndarray: ...  # pragma: nocover

    @abstractmethod
    def update_projection_width(
        self, proj_data: cp.ndarray
    ) -> cp.ndarray: ...  # pragma: nocover


class AstraProjector(ProjectorBase):
    def __init__(self, atools: AstraTools3D):
        self.atools = atools

    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray:
        return self.atools._forwprojCuPy(object3D)

    def forwprojOS(self, object3D: cp.ndarray, sub_ind: int) -> cp.ndarray:
        return self.atools._forwprojOSCuPy(object3D, sub_ind)

    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray:
        return self.atools._backprojCuPy(proj_data)

    def backprojOS(self, proj_data: cp.ndarray, sub_ind: int) -> cp.ndarray:
        return self.atools._backprojOSCuPy(proj_data, sub_ind)

    def update_projection_width(self, proj_data: cp.ndarray) -> cp.ndarray:
        return _apply_horiz_detector_padding(
            proj_data, self.atools.detectors_x_pad, cupyrun=True
        )


gather_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void gather(float2* g, float2* f, float* theta, int m, float* mu,
                                  int n, int ntheta, int nz, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= ntheta || tz >= nz) return;

    float M_PI = 3.141592653589793238f;
    float2 g0, g0t;
    float w, coeff0;
    float w0, w1, x0, y0, coeff1;
    int ell0, ell1, g_ind, f_ind, f_indx, f_indy;

    g_ind = tx + ty * n + tz * n * ntheta;  

    if (dir == 0) g0 = {};
    else g0 = g[g_ind];

    coeff0 = M_PI / mu[0];
    coeff1 = -M_PI * M_PI / mu[0];

    x0 = (tx - n / 2) / (float)n * __cosf(theta[ty]);
    y0 = -(tx - n / 2) / (float)n * __sinf(theta[ty]);

    for (int i1 = 0; i1 < 2 * m + 1; i1++)
    {
        ell1 = floorf(2 * n * y0) - m + i1;
        for (int i0 = 0; i0 < 2 * m + 1; i0++)
        {
            ell0 = floorf(2 * n * x0) - m + i0;

            w0 = ell0 / (float)(2 * n) - x0;
            w1 = ell1 / (float)(2 * n) - y0;

            w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));
            
            f_indx = (n + ell0 + 2 * n) % (2 * n);
            f_indy = (n + ell1 + 2 * n) % (2 * n);

            f_ind = f_indx + (2 * n) * f_indy + tz * (2 * n) * (2 * n);

            if (dir == 0)
            {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            }
            else
            {
                atomicAdd(&(f[f_ind].x), w * g0.x);
                atomicAdd(&(f[f_ind].y), w * g0.y);
            }
        }
    }

    if (dir == 0)
    {
        g[g_ind].x = g0.x / n;
        g[g_ind].y = g0.y / n;
    }
}
""",
    "gather",
)


class FFTProjector(ProjectorBase):
    """USFFT-based forward and backward projection wrapper."""

    def __init__(
        self,
        n: int,
        theta: np.ndarray,
        mask_r: float = 1.0,
        CenterRotOffset: Optional[float] = None,
        DetectorDimH_pad: int = 0,
        indVec: Optional[np.ndarray] = None,
        numProjBins: Optional[int] = None,
    ):
        """Usfft parameters
        mask_r - circle radius"""
        self.n = n
        self.DetectorDimH_pad = DetectorDimH_pad
        self.indVec = indVec
        self.numProjBins = numProjBins
        assert (self.indVec is None) == (self.numProjBins is None)

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
        phi = (
            cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32"))
            # * (1 - n % 4)
            * n
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

        self.ntheta = len(theta)
        self.theta = theta
        self.mask = mask
        self.raxis = (
            self.n // 2 - CenterRotOffset if CenterRotOffset is not None else None
        )
        self.pars = m, mu, phi, c1dfftshift, c2dfftshift
        self.left_pad = n // 2
        self.right_pad = n - self.left_pad

    def _get_OS_theta(self, sub_ind: int) -> np.ndarray:
        assert self.indVec is not None
        assert self.numProjBins is not None
        indVec = self.indVec[sub_ind, :]
        if indVec[self.numProjBins - 1] == 0:
            indVec = indVec[:-1]  # shrink vector size
        return self.theta[indVec]

    def _forwproj_impl(self, object3D: cp.ndarray, theta: np.ndarray) -> cp.ndarray:
        """Radon transform"""
        [nz, ny, n] = object3D.shape
        assert ny == self.n
        assert n == self.n

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars
        sino = cp.zeros([nz, theta.size, n], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = object3D * phi
        fde = cp.pad(
            fde,
            (
                (0, 0),
                (self.left_pad, self.right_pad),
                (self.left_pad, self.right_pad),
            ),
        )
        assert fde.shape[1] == 2 * n
        assert fde.shape[2] == 2 * n
        # STEP1: fft 2d
        fde = cp.fft.fft2(fde[:, ::-1, :] * c2dfftshift) * c2dfftshift

        mua = cp.array([mu], dtype="float32")

        gather_kernel(
            (math.ceil(n / 32), math.ceil(theta.size / 32), nz),
            (32, 32, 1),
            (sino, fde, cp.asarray(theta), m, mua, n, theta.size, nz, 0),
        )
        # STEP3: ifft 1d
        sino = cp.fft.ifft(c1dfftshift * sino) * c1dfftshift

        # STEP4: Shift based on the rotation axis, not needed if rotation_axis==n/2
        if self.raxis is not None:
            t = cp.fft.fftfreq(n).astype("float32")
            w = cp.exp(2 * cp.pi * 1j * t * (-self.raxis + n / 2))
            sino = cp.fft.ifft(w * cp.fft.fft(sino))
        return cp.pad(
            sino.real, ((0, 0), (0, 0), (self.DetectorDimH_pad, self.DetectorDimH_pad))
        )

    def forwproj(self, object3D: cp.ndarray) -> cp.ndarray:
        return self._forwproj_impl(object3D, self.theta)

    def forwprojOS(self, object3D: cp.ndarray, sub_ind: int) -> cp.ndarray:
        return self._forwproj_impl(object3D, self._get_OS_theta(sub_ind))

    def backproj(self, proj_data: cp.ndarray) -> cp.ndarray:
        """Adjoint Radon transform"""

        [nz, ntheta, n] = proj_data.shape
        assert n == self.n + 2 * self.DetectorDimH_pad
        assert ntheta == self.ntheta

        if self.DetectorDimH_pad > 0:
            proj_data = proj_data[..., self.DetectorDimH_pad : -self.DetectorDimH_pad]
            n = proj_data.shape[2]
            assert n == self.n

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: Shift based on the rotation axis, not  needed if rotation_axis==n/2
        if self.raxis is not None:
            t = cp.fft.fftfreq(n).astype("float32")
            w = cp.exp(-2 * cp.pi * 1j * t * (-self.raxis + n / 2))
            proj_data = cp.fft.ifft(w * cp.fft.fft(proj_data))

        # STEP1: fft 1d
        sino = cp.fft.fft(c1dfftshift * proj_data) * c1dfftshift

        # STEP2: interpolation (gathering) in the frequency domain
        mua = cp.array([mu], dtype="float32")
        fde = cp.zeros([nz, 2 * n, 2 * n], dtype="complex64")
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, cp.asarray(self.theta), m, mua, n, self.ntheta, nz, 1),
        )
        # STEP3: ifft 2d
        fde = cp.fft.ifft2(fde[:, ::-1, :] * c2dfftshift) * c2dfftshift

        # STEP4: unpadding, multiplication by phi
        fde = fde[
            :, self.left_pad : -self.right_pad, self.left_pad : -self.right_pad
        ] * (phi * 4)

        return fde.real

    def backprojOS(self, proj_data: cp.ndarray, sub_ind: int) -> cp.ndarray:
        assert False

    def update_projection_width(self, proj_data: cp.ndarray) -> cp.ndarray:
        if proj_data.shape[2] == self.n:
            return proj_data
        assert proj_data.shape[2] > self.n
        left_unpad = (proj_data.shape[2] - self.n) // 2
        right_unpad = proj_data.shape[2] - self.n - left_unpad
        return proj_data[..., left_unpad:-right_unpad]
