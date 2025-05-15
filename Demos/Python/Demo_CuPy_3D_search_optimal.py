#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script that demonstrates the reconstruction of CuPy arrays while keeping
the data on the GPU (device-to-device)

Dependencies:
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom
    * CuPy package

@author: Daniil Kazantsev
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

print("center_size, phantom size, angle number, slice number, time")

for N_size in [512, 1024, 1536, 2048]:
    for angles_num in [128, 256, 512, 1024, 1536, 2048]:

        # print("Building 3D phantom using TomoPhantom software")
        tic = timeit.default_timer()
        model = 13  # select a model number from the library
        # N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
        path = os.path.dirname(tomophantom.__file__)
        path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

        phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

        # Projection geometry related parameters:
        Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
        Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
        # angles_num = int(0.3 * np.pi * N_size)  # angles number
        angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
        angles_rad = angles * (np.pi / 180.0)

        # print("Generate 3D analytical projection data with TomoPhantom")
        projData3D_analyt = TomoP3D.ModelSino(
            model, N_size, Horiz_det, Vert_det, angles, path_library3D
        )
        input_data_labels = ["detY", "angles", "detX"]

        # print(np.shape(projData3D_analyt))

        # transfering numpy array to CuPy array
        projData3D_analyt_cupy = cp.asarray(projData3D_analyt, order="C")

        for slice_number in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]:

            RecToolsCP = RecToolsDIRCuPy(
                DetectorsDimH=Horiz_det,  # Horizontal detector dimension
                DetectorsDimV=slice_number,  # Vertical detector dimension (3D case)
                CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
                AnglesVec=angles_rad,  # A vector of projection angles in radians
                ObjSize=N_size,  # Reconstructed object dimensions (scalar)
                device_projector="gpu",
            )

            # tic = timeit.default_timer()
            # for x in range(80):
            #     Fourier_cupy = RecToolsCP.FOURIER_INV(
            #         projData3D_analyt_cupy[1:slice_number, :, :],
            #         recon_mask_radius=0.95,
            #         data_axes_labels_order=input_data_labels,
            #         center_size=2048,
            #         block_dim=[16, 16],
            #         block_dim_center=[32, 4],
            #     )
            # toc = timeit.default_timer()

            # Run_time = (toc - tic)/80
            # print("Phantom size: {}, angle number: {}, and slice number: {}, in time: {} seconds".format(N_size, angles_num, slice_number, Run_time))

            for center_size in [0, 128, 256, 384, 448, 512, 640, 672, 704, 768, 800, 864, 928, 1024, 1280, 1536, 1792, 2048, 2560, 3072]:
                tic = timeit.default_timer()
                for x in range(80):
                    Fourier_cupy = RecToolsCP.FOURIER_INV(
                        projData3D_analyt_cupy[1:slice_number, :, :],
                        recon_mask_radius=0.95,
                        center_size=center_size,
                        data_axes_labels_order=input_data_labels,
                    )
                toc = timeit.default_timer()
                Run_time = (toc - tic)/80
                print("{}, {}, {}, {}, {}".format(center_size, N_size, angles_num, slice_number, Run_time))
