"""Reconstruction class for direct reconstructon methods.

* Fourier Slice Theorem reconstruction (adopted from Tim Day's code)
* Forward/Backward projection (ASTRA)
* Filtered Back Projection (ASTRA)
"""

import numpy as np
import scipy.fftpack

from typing import Union
from tomobar.astra_wrappers import AstraTools2D, AstraTools3D
from tomobar.supp.funcs import _data_dims_swapper, _parse_device_argument


def _filtersinc3D(projection3D: np.ndarray):
    """Applies a 3D filter to 3D projection data for FBP

    Args:
        projection3D (np.ndarray): projection data

    Returns:
        np.ndarray: Filtered data
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
    f = scipy.fftpack.fftshift(r)
    # making a 2d filter for projection
    f_2d = np.zeros((DetectorsLengthV, DetectorsLengthH), dtype="float32")
    f_2d[0::, :] = f
    filtered = np.zeros(np.shape(projection3D), dtype="float32")

    for i in range(0, projectionsNum):
        IMG = scipy.fftpack.fft2(projection3D[:, i, :])
        fimg = IMG * f_2d
        filtered[:, i, :] = np.real(scipy.fftpack.ifft2(fimg))
    return multiplier * filtered


def _filtersinc2D(sinogram):
    # applies filters to __2D projection data__ in order to achieve FBP
    a = 1.1
    [projectionsNum, DetectorsLengthH] = np.shape(sinogram)
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
    f = scipy.fftpack.fftshift(r)
    filtered = np.zeros(np.shape(sinogram))

    for i in range(0, projectionsNum):
        IMG = scipy.fftpack.fft(sinogram[i, :])
        fimg = IMG * f
        filtered[i, :] = multiplier * np.real(scipy.fftpack.ifft(fimg))
    return np.float32(filtered)


class RecToolsDIR:
    """Reconstruction class using DIRect methods (FBP and Fourier).

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float, ndarray): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        device_projector (str, int, optional): 'cpu' or 'gpu'  device OR provide a GPU index (integer) of a specific GPU device.
        cupyrun (bool, optional): instantiate CuPy class if True.
    """

    def __init__(
        self,
        DetectorsDimH,  # DetectorsDimH # detector dimension (horizontal)
        DetectorsDimV,  # DetectorsDimV # detector dimension (vertical) for 3D case only
        CenterRotOffset,  # Centre of Rotation (CoR) scalar or a vector
        AnglesVec,  # Array of angles in radians
        ObjSize,  # A scalar to define reconstructed object dimensions
        device_projector="gpu",  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
    ):
        device_projector, GPUdevice_index = _parse_device_argument(device_projector)

        if DetectorsDimV == 0 or DetectorsDimV is None:
            self.geom = "2D"
            self.Atools = AstraTools2D(
                DetectorsDimH,
                AnglesVec,
                CenterRotOffset,
                ObjSize,
                device_projector,
                GPUdevice_index,
            )
        else:
            self.geom = "3D"
            self.Atools = AstraTools3D(
                DetectorsDimH,
                DetectorsDimV,
                AnglesVec,
                CenterRotOffset,
                ObjSize,
                device_projector,
                GPUdevice_index,
            )

    def FORWPROJ(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Module to perform forward projection of 2d/3d data numpy array

        Args:
            data (np.ndarray): 2D or 3D object

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the OUTPUT data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray: Forward projected numpy array (projection data)
        """
        projected = self.Atools._forwproj(data)
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                if self.geom == "2D":
                    projected = _data_dims_swapper(projected, value, ["angles", "detX"])
                else:
                    projected = _data_dims_swapper(
                        projected, value, ["detY", "angles", "detX"]
                    )

        return projected

    def BACKPROJ(self, projdata: np.ndarray, **kwargs) -> np.ndarray:
        """Module to perform back-projection of 2d/3d data numpy array

        Args:
            projdata (np.ndarray): 2D/3D projection data

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray: Backprojected 2D/3D object
        """
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                if self.geom == "2D":
                    projdata = _data_dims_swapper(projdata, value, ["angles", "detX"])
                else:
                    projdata = _data_dims_swapper(
                        projdata, value, ["detY", "angles", "detX"]
                    )

        return self.Atools._backproj(projdata)

    def FBP(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Filtered backprojection reconstruction module for 2D or 3D data.

        Args:
            data (np.ndarray): 2D or 3D projection data.

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray: FBP reconstructed 2D or 3D object.
        """
        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                if self.geom == "2D":
                    data = _data_dims_swapper(data, value, ["angles", "detX"])
                else:
                    data = _data_dims_swapper(data, value, ["detY", "angles", "detX"])

        if self.geom == "2D":
            "dealing with FBP 2D not working for parallel_vec geometry and CPU"
            if self.Atools.processing_arch == "gpu":
                return self.Atools._fbp(data)
            else:
                return self.Atools._backproj(_filtersinc2D(data))
        else:
            return self.Atools._backproj(_filtersinc3D(data))

    def FOURIER(self, sinogram: np.ndarray, method: str = "linear") -> np.ndarray:
        """2D Reconstruction using Fourier slice theorem (scipy required)
        for griddata interpolation module choose nearest, linear or cubic

        Args:
            sinogram (np.ndarray): 2D sinogram data
            method (str, optional): Interpolation type (nearest, linear, or cubic). Defaults to "linear".

        Returns:
            np.ndarray: Reconstructed object
        """
        if sinogram.ndim == 3:
            raise ValueError(
                "Fourier method is currently for 2D data only, use FBP if 3D needed "
            )
        else:
            pass
        if (method == "linear") or (method == "nearest") or (method == "cubic"):
            pass
        else:
            raise ValueError(
                "For griddata interpolation module choose nearest, linear or cubic"
            )
        import scipy.interpolate
        from scipy.fft import fftshift, ifftshift, fft, ifft2
        import scipy.misc
        import scipy.ndimage.interpolation

        DetectorsDimH = self.Atools.detectors_x
        ObjSize = self.Atools.recon_size
        # Fourier transform the rows of the sinogram, move the DC component to the row's centre
        sinogram_fft_rows = fftshift(fft(ifftshift(sinogram, axes=1)), axes=1)
        # Coordinates of sinogram FFT-ed rows' samples in 2D FFT space
        a = -self.Atools.angles_vec
        r = np.arange(DetectorsDimH) - DetectorsDimH / 2
        r, a = np.meshgrid(r, a)
        r = r.flatten()
        a = a.flatten()
        srcx = (DetectorsDimH / 2) + r * np.cos(a)
        srcy = (DetectorsDimH / 2) + r * np.sin(a)

        # Coordinates of regular grid in 2D FFT space
        dstx, dsty = np.meshgrid(np.arange(DetectorsDimH), np.arange(DetectorsDimH))
        dstx = dstx.flatten()
        dsty = dsty.flatten()
        # Interpolate the 2D Fourier space grid from the transformed sinogram rows
        fft2 = scipy.interpolate.griddata(
            (srcy, srcx),
            sinogram_fft_rows.flatten(),
            (dsty, dstx),
            method,
            fill_value=0.0,
        ).reshape((DetectorsDimH, DetectorsDimH))
        # Transform from 2D Fourier space back to a reconstruction of the target
        recon = np.real(fftshift(ifft2(ifftshift(fft2))))

        # Cropping the reconstruction to size of the original image
        image = recon[
            int(((DetectorsDimH - ObjSize) / 2) + 1) : DetectorsDimH
            - int(((DetectorsDimH - ObjSize) / 2) - 1),
            int(((DetectorsDimH - ObjSize) / 2)) : DetectorsDimH
            - int(((DetectorsDimH - ObjSize) / 2)),
        ]
        return image

    def FOURIER3D(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Fourier direct reconstruction.
        Args:
            data (np.ndarray): projection data or a sinogram

        Keyword Args:
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].

        Returns:
            np.ndarray: The Fourier reconstructed volume or an image.
        """
        import scipy.interpolate
        from scipy.fft import fftshift, ifftshift

        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                if self.geom == "2D":
                    data = _data_dims_swapper(data, value, ["angles", "detX"])
                else:
                    data = _data_dims_swapper(data, value, ["detY", "angles", "detX"])

        # Fourier transform the rows of the sinogram, move the DC component to the row's centre
        sinogram_fft_rows = scipy.fftpack.fftshift(
            scipy.fftpack.fft(scipy.fftpack.ifftshift(data, axes=1)), axes=1
        )

        proj_f = scipy.fft.rfft(data, axis=-1, norm="backward", overwrite_x=True)

        return sinogram_fft_rows
