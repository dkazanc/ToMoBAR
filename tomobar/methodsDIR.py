#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A reconstruction class for direct reconstructon methods.

-- Fourier Slice Theorem reconstruction (adopted from Tim Day's code)
-- Forward/Backward projection (ASTRA and ASTRA with CuPy)
-- Filtered Back Projection (ASTRA and ASTRA with CuPy)

@author: Daniil Kazantsev
"""

import numpy as np
from tomobar.recon_base import RecTools
from tomobar.supp.suppTools import _data_swap

import scipy.fftpack


def _filtersinc3D(projection3D):
    # applies filters to __3D projection data__ in order to achieve FBP
    # Data format [DetectorVert, Projections, DetectorHoriz]
    # adopted from Matlabs code by  Waqas Akram
    # "a":	This parameter varies the filter magnitude response.
    # When "a" is very small (a<<1), the response approximates |w|
    # As "a" is increased, the filter response starts to
    # roll off at high frequencies.
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


class RecToolsDIR(RecTools):
    """----------------------------------------------------------------------------------------------------------
    A class for reconstruction using DIRect methods (FBP and Fourier)
    ----------------------------------------------------------------------------------------------------------
    Parameters of the class function main specifying the projection geometry:
      *DetectorsDimH,     # Horizontal detector dimension
      *DetectorsDimV,     # Vertical detector dimension for 3D case, 0 or None for 2D case
      *CenterRotOffset,   # The Centre of Rotation (CoR) scalar or a vector
      *AnglesVec,         # A vector of projection angles in radians
      *ObjSize,           # Reconstructed object dimensions (a scalar)
      *device_projector   # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
      *data_axis_labels   # set the order of axis labels of the input data, e.g. ['detY', 'angles', 'detX']
    ----------------------------------------------------------------------------------------------------------
    """

    def __init__(
        self,
        DetectorsDimH,  # DetectorsDimH # detector dimension (horizontal)
        DetectorsDimV,  # DetectorsDimV # detector dimension (vertical) for 3D case only
        CenterRotOffset,  # Centre of Rotation (CoR) scalar or a vector
        AnglesVec,  # Array of angles in radians
        ObjSize,  # A scalar to define reconstructed object dimensions
        device_projector="gpu",  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
        data_axis_labels=None,  # the input data axis labels
    ):
        super().__init__(
            DetectorsDimH,
            DetectorsDimV,
            CenterRotOffset,
            AnglesVec,
            ObjSize,
            device_projector=device_projector,
            data_axis_labels=data_axis_labels,  # inherit from the base class
        )

    def FORWPROJ(self, data):
        """Module to perform forward projection of 2d/3d data array

        Args:
            data (ndarray): 2d or 3d array to project, either numpy or cupy
        Returns:
            ndarray: Forward projected array either numpy or cupy
        """
        # perform check here if the given array is numpy or not
        # if not we assume that it is a CuPy array (loose assumption of course)
        if isinstance(data, np.ndarray):
            projdata = self.Atools.forwproj(data)
        else:
            projdata = self.Atools.forwprojCuPy(data)
        return projdata

    def BACKPROJ(self, projdata):
        """Module to perform back-projection of 2d/3d data array

        Args:
            projdata (ndarray): 2d or 3d array to backproject, either numpy or cupy

        Returns:
            ndarray: backprojected array either numpy or cupy
        """
        # perform check here if the given array is numpy or not
        # if not we assume that it is a CuPy array (loose assumption of course)
        if isinstance(projdata, np.ndarray):
            backproj = self.Atools.backproj(projdata)
        else:
            backproj = self.Atools.backprojCuPy(projdata)
        return backproj

    def FOURIER(self, sinogram, method="linear"):
        """
        2D Reconstruction using Fourier slice theorem (scipy required)
        for griddata interpolation module choose nearest, linear or cubic
        """
        if sinogram.ndim == 3:
            raise (
                "Fourier method is currently for 2D data only, use FBP if 3D needed "
            )
        else:
            pass
        if (method == "linear") or (method == "nearest") or (method == "cubic"):
            pass
        else:
            raise ("For griddata interpolation module choose nearest, linear or cubic ")
        import scipy.interpolate
        import scipy.fftpack
        import scipy.misc
        import scipy.ndimage.interpolation

        # Fourier transform the rows of the sinogram, move the DC component to the row's centre
        sinogram_fft_rows = scipy.fftpack.fftshift(
            scipy.fftpack.fft(scipy.fftpack.ifftshift(sinogram, axes=1)), axes=1
        )
        # Coordinates of sinogram FFT-ed rows' samples in 2D FFT space
        a = -self.AnglesVec
        r = np.arange(self.DetectorsDimH) - self.DetectorsDimH / 2
        r, a = np.meshgrid(r, a)
        r = r.flatten()
        a = a.flatten()
        srcx = (self.DetectorsDimH / 2) + r * np.cos(a)
        srcy = (self.DetectorsDimH / 2) + r * np.sin(a)

        # Coordinates of regular grid in 2D FFT space
        dstx, dsty = np.meshgrid(
            np.arange(self.DetectorsDimH), np.arange(self.DetectorsDimH)
        )
        dstx = dstx.flatten()
        dsty = dsty.flatten()
        # Interpolate the 2D Fourier space grid from the transformed sinogram rows
        fft2 = scipy.interpolate.griddata(
            (srcy, srcx),
            sinogram_fft_rows.flatten(),
            (dsty, dstx),
            method,
            fill_value=0.0,
        ).reshape((self.DetectorsDimH, self.DetectorsDimH))
        # Transform from 2D Fourier space back to a reconstruction of the target
        recon = np.real(
            scipy.fftpack.fftshift(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(fft2)))
        )

        # Cropping reconstruction to size of the original image
        image = recon[
            int(((self.DetectorsDimH - self.ObjSize) / 2) + 1) : self.DetectorsDimH
            - int(((self.DetectorsDimH - self.ObjSize) / 2) - 1),
            int(((self.DetectorsDimH - self.ObjSize) / 2)) : self.DetectorsDimH
            - int(((self.DetectorsDimH - self.ObjSize) / 2)),
        ]
        return image

    def FBP(self, data):
        # get the input data into the right format dimension-wise
        data = _data_swap(data, self.data_swap_list)
        if self.geom == "2D":
            "dealing with FBP 2D not working for parallel_vec geometry and CPU"
            if self.device_projector == "gpu":
                FBP_rec = self.Atools.fbp2D(data)  # GPU reconstruction
            else:
                filtered_sino = _filtersinc2D(data)  # filtering sinogram
                FBP_rec = self.Atools.backproj(filtered_sino)  # backproject
        if self.geom == "3D":
            filtered_sino = _filtersinc3D(data)  # filtering projection data
            FBP_rec = self.Atools.backproj(filtered_sino)  # 3d backproject
        return FBP_rec
