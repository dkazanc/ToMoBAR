"""Reconstruction class for direct reconstruction methods (2D/3D).

* :func:`RecToolsDIR.FORWPROJ` and :func:`RecToolsDIR.BACKPROJ` Forward/Backward 2D/3D projection (ASTRA-Toolbox)
* :func:`RecToolsDIR.FOURIER` Fourier Slice Theorem-based reconstruction in 2D only (adopted from the Tim Day's code)
* :func:`RecToolsDIR.FBP` Filtered Back Projection 2D/3D (ASTRA with the custom built filter).
"""

import numpy as np
import scipy.fftpack

from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
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

    def FOURIER(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """2D Reconstruction using Fourier slice theorem (scipy required)
        for griddata interpolation module choose nearest, linear or cubic

        Args:
            data (np.ndarray): 2D sinogram data

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.
            method (str, optional): Interpolation type (nearest, linear, or cubic). Defaults to "linear".

        Returns:
            np.ndarray: Reconstructed object
        """
        if data.ndim == 3:
            raise ValueError(
                "Fourier method is currently for 2D data only, use FBP if 3D reconstruction needed"
            )

        for key, value in kwargs.items():
            if key == "data_axes_labels_order" and value is not None:
                data = _data_dims_swapper(data, value, ["angles", "detX"])
            if key == "method":
                if value not in ["linear", "nearest", "cubic"]:
                    raise ValueError(
                        "For griddata interpolation module choose nearest, linear or cubic"
                    )
                else:
                    method = value

        from scipy.fft import fftshift, ifftshift, fft, ifft2
        from scipy.interpolate import griddata

        ObjSize = self.Atools.recon_size
        # pad sinogram and move it to compensate for CoR
        oversampling = 2  # 2 or larger
        angles_tot, DetectorsDimH = data.shape
        if (DetectorsDimH % 2) != 0:
            raise ValueError(
                "The horizontal detector size of the projection data (sinogram) must be even"
            )
        det_x_up = oversampling * DetectorsDimH
        sino_up = np.zeros([angles_tot, det_x_up], dtype=np.float32)
        pad_from = DetectorsDimH // 2 + int(self.Atools.centre_of_rotation)
        pad_to = det_x_up - DetectorsDimH // 2 + int(self.Atools.centre_of_rotation)
        sino_up[:, pad_from:pad_to] = data

        # Fourier transform the rows of the sinogram, move the DC component to the row's centre
        sinogram_fft_rows = fftshift(fft(ifftshift(sino_up, axes=1)), axes=1)
        # Coordinates of sinogram FFT-ed rows' samples in 2D FFT space
        a = -self.Atools.angles_vec
        r = np.arange(det_x_up) - det_x_up / 2
        r, a = np.meshgrid(r, a)
        r = r.flatten()
        a = a.flatten()
        srcx = (det_x_up / 2) + r * np.cos(a)
        srcy = (det_x_up / 2) + r * np.sin(a)

        # Coordinates of regular grid in 2D FFT space
        dstx, dsty = np.meshgrid(np.arange(det_x_up), np.arange(det_x_up))
        dstx = dstx.flatten()
        dsty = dsty.flatten()
        # Interpolate the 2D Fourier space grid from the transformed sinogram rows
        fft2 = griddata(
            (srcy, srcx),
            sinogram_fft_rows.flatten(),
            (dsty, dstx),
            method,
            fill_value=0.0,
        ).reshape((det_x_up, det_x_up))
        # Transform from 2D Fourier space back to a reconstruction of the target
        recon = np.real(fftshift(ifft2(ifftshift(fft2))))

        # Cropping the reconstruction to size of the original image
        unpad_from = det_x_up // 2 - ObjSize // 2
        unpad_to = det_x_up // 2 + ObjSize // 2
        return recon[unpad_from:unpad_to, unpad_from:unpad_to]
