"""The series of classes for wrapping of ASTRA toolbox to perform projection/backprojection
and reconstruction of of 2D/3D parallel beam data.

GPLv3 license (defined by ASTRA toolbox)
"""

from contextlib import ExitStack
import weakref

import numpy as np
from typing import Union
from tomobar.supp.funcs import (
    _normalise_center_rot_offset,
    _validate_ordsub_number,
    _vec_geom_init2D,
)

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        print("CuPy is installed but the GPU device is inaccessible")
except ImportError:
    import numpy as xp

    print("CuPy is not installed")

try:
    import astra
except ImportError:
    print("____! Astra-toolbox package is missing, please install !____")


###########Base class############
class AstraBase:
    """The base class for projection/backprojection operations and various reconstruction algorithms using ASTRA toolbox wrappers.

    Args:
        detectors_x (int): Horizontal detector dimension in pixel units.
        detectors_x_pad (int): Padded size of horizontal detector with edge values
        angles_vec (np.ndarray): A vector of projection angles in radians.
        centre_of_rotation (float, np.ndarray): 2D horizontal detector offsets
            in pixels, as a scalar or (n_angles,) array. None means zero.
        recon_size (int): The size of the reconstructed object (a slice) defined as [recon_size, recon_size].
        processing_arch (str): 'cpu' or 'gpu' - based processing.
        device_index (int, optional): An integer for the GPU device, -1 for CPU computing and >0 for the GPU device.
        ordsub_number (int): Number of ordered subsets, from 1 to the number
            of views. None means one subset.
        detectors_y (int): Vertical detector dimension in pixel units (3D case).

    Use ``close()`` or a ``with`` block to release persistent projectors promptly.
    Otherwise they are released when the wrapper is garbage collected.
    ``subset_indices`` holds one read-only index array per interleaved subset;
    these arrays have their actual lengths and contain no padding.
    """

    _is_3d = False

    def __init__(
        self,
        detectors_x,
        detectors_x_pad,
        angles_vec,
        centre_of_rotation,
        recon_size,
        processing_arch,
        device_index,
        ordsub_number,
        detectors_y,
    ):
        self.detectors_x = detectors_x
        self.detectors_x_pad = detectors_x_pad
        self.angles_vec = angles_vec
        self.detectors_y = detectors_y
        self.centre_of_rotation = centre_of_rotation
        self.recon_size = recon_size
        self.processing_arch = processing_arch
        self.device_index = device_index
        self.ordsub_number = ordsub_number
        self._setOS_indices(len(self.angles_vec))
        self.fbp_filter_type = "ram-lak"  # ASTRA 2D FBP specific parameter
        self.fbp_filter_parameter = None  # ASTRA 2D FBP specific parameter
        self.fbp_filter_d = None  # ASTRA 2D FBP specific parameter

        self._init_projector_resources()

    def _init_projector_resources(self):
        """Initialise shared ownership without constructing geometry."""
        self._closed = False
        self._projector_cleanup = ExitStack()
        # The callback retains the cleanup stack, not this wrapper.
        self._projector_finalizer = weakref.finalize(
            self, self._projector_cleanup.close
        )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("The ASTRA wrapper has been closed")

    def _register_projector(self, projector_id):
        manager = astra.projector3d if self._is_3d else astra.projector
        self._projector_cleanup.callback(manager.delete, projector_id)

    def close(self):
        """Release wrapper-owned projectors. Repeated calls are harmless."""
        self._closed = True
        self._projector_finalizer()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def detectors_x(self) -> int:
        return self._detectors_x

    @detectors_x.setter
    def detectors_x(self, detectors_x_val):
        if detectors_x_val <= 0:
            raise ValueError(
                "The size of the horizontal detector cannot be negative or zero"
            )
        self._detectors_x = detectors_x_val

    @property
    def detectors_x_pad(self) -> int:
        return self._detectors_x_pad

    @detectors_x_pad.setter
    def detectors_x_pad(self, detectors_x_pad_val):
        if detectors_x_pad_val < 0:
            raise ValueError(
                "The padding size of the horizontal detector cannot be negative"
            )
        self._detectors_x_pad = detectors_x_pad_val

    @property
    def angles_vec(self) -> np.ndarray:
        return self._angles_vec

    @angles_vec.setter
    def angles_vec(self, angles_vec_val):
        if len(angles_vec_val) == 0:
            raise ValueError("The length of angles array cannot be zero")
        if np.ndim(angles_vec_val) >= 2:
            raise ValueError("The array of angles must be 1D")
        self._angles_vec = angles_vec_val

    @property
    def centre_of_rotation(self) -> Union[float, np.ndarray]:
        return self._centre_of_rotation

    @centre_of_rotation.setter
    def centre_of_rotation(self, centre_of_rotation_val):
        self._centre_of_rotation = _normalise_center_rot_offset(
            centre_of_rotation_val, len(self.angles_vec), is_3d=False
        )

    @property
    def recon_size(self) -> int:
        return self._recon_size

    @recon_size.setter
    def recon_size(self, recon_size_val):
        if isinstance(recon_size_val, tuple):
            raise ValueError(
                "Reconstruction is currently available for squared or cubic objects only, please provide a scalar"
            )
        if recon_size_val <= 0:
            raise ValueError("The size of the reconstruction object cannot be zero")
        self._recon_size = recon_size_val

    @property
    def processing_arch(self) -> str:
        return self._processing_arch

    @processing_arch.setter
    def processing_arch(self, processing_arch_val):
        if processing_arch_val != "cpu" and processing_arch_val != "gpu":
            raise ValueError(
                "Please choose the processing architecture to be either 'cpu' or 'gpu'"
            )
        if processing_arch_val == "cpu" and np.any(self.centre_of_rotation != 0):
            raise ValueError(
                "Correction for the CoR with the vector geometry is implemented on the GPU only"
            )
        self._processing_arch = processing_arch_val

    @property
    def device_index(self) -> int:
        return self._device_index

    @device_index.setter
    def device_index(self, device_index_val):
        if device_index_val <= -2:
            raise ValueError("The GPU device index can be only -1, 0 and larger than 0")
        if self.processing_arch == "cpu":
            self._device_index = -1
        else:
            self._device_index = device_index_val
            astra.set_gpu_index(device_index_val)

    @property
    def ordsub_number(self) -> int:
        return self._ordsub_number

    @ordsub_number.setter
    def ordsub_number(self, ordsub_number_val):
        self._ordsub_number = _validate_ordsub_number(
            ordsub_number_val, len(self.angles_vec)
        )

    @property
    def detectors_y(self) -> int:
        return self._detectors_y

    @detectors_y.setter
    def detectors_y(self, detectors_y_val):
        if detectors_y_val is not None:
            if detectors_y_val <= 0:
                raise ValueError(
                    "The size of the vertical detector cannot be negative or zero"
                )
        self._detectors_y = detectors_y_val

    def _setOS_indices(self, view_count):
        """Create interleaved view indices without padding or sentinel values."""
        self.subset_indices = tuple(
            np.arange(subset, view_count, self.ordsub_number, dtype=np.intp)
            for subset in range(self.ordsub_number)
        )
        for indices in self.subset_indices:
            indices.setflags(write=False)
        # Legacy metadata: newInd_Vec is now a tuple, not a padded matrix.
        self.newInd_Vec = self.subset_indices
        self.NumbProjBins = max(map(len, self.subset_indices))

    def _set_vol2d_geometry(self):
        """set the reconstruction (vol_geom)"""
        self.vol_geom = astra.create_vol_geom(self.recon_size, self.recon_size)

    def _set_cpu_projection2d_parallel_geometry(self):
        """the classical 2D projection geometry (cpu)"""
        self.proj_geom = astra.create_proj_geom(
            "parallel",
            1.0,
            self.detectors_x + 2 * self.detectors_x_pad,
            self.angles_vec,
        )
        self.proj_id = astra.create_projector("line", self.proj_geom, self.vol_geom)
        self._register_projector(self.proj_id)

    def _set_gpu_projection2d_parallel_geometry(self):
        """the classical projection geometry (gpu)"""
        vectors = _vec_geom_init2D(self.angles_vec, self.centre_of_rotation)
        self.proj_geom = astra.create_proj_geom(
            "parallel_vec", self.detectors_x + 2 * self.detectors_x_pad, vectors
        )
        self.proj_id = astra.create_projector(
            "cuda", self.proj_geom, self.vol_geom
        )  # for GPU
        self._register_projector(self.proj_id)

    def _set_projection2d_OS_parallel_geometry(self):
        """organising 2d OS projection geometry CPU/GPU"""
        self.proj_geom_OS = {}
        self.proj_id_OS = {}
        for sub_ind in range(self.ordsub_number):
            self.indVec = self.subset_indices[sub_ind]  # OS-specific indices
            anglesOS = self.angles_vec[self.indVec]  # OS-specific angles

            if np.ndim(self.centre_of_rotation) == 0:
                # centre_of_rotation is a _scalar_
                vectorsOS = _vec_geom_init2D(anglesOS, self.centre_of_rotation)
            else:
                # centre_of_rotation is a _vector_
                vectorsOS = _vec_geom_init2D(
                    anglesOS, self.centre_of_rotation[self.indVec]
                )
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom(
                "parallel_vec", self.detectors_x + 2 * self.detectors_x_pad, vectorsOS
            )
            if self.processing_arch == "cpu":
                self.proj_id_OS[sub_ind] = astra.create_projector(
                    "line", self.proj_geom_OS[sub_ind], self.vol_geom
                )  # for CPU
                self._register_projector(self.proj_id_OS[sub_ind])
            else:
                self.proj_id_OS[sub_ind] = astra.create_projector(
                    "cuda", self.proj_geom_OS[sub_ind], self.vol_geom
                )  # for GPU
                self._register_projector(self.proj_id_OS[sub_ind])

    def _runAstraBackproj2D(
        self,
        sinogram: np.ndarray,
        method: str,
        iterations: int,
        os_index: Union[int, None],
    ) -> np.ndarray:
        """2D ASTRA-based back-projector

        Args:
            sinogram (np.ndarray): a sinogram to backproject
            method (str): A method to choose from the ASTRA's provided ones.
            iterations (int): The number of iterations for iterative methods.
            os_index (Union[int, None]): The number of ordered subsets.

        Returns:
            np.ndarray: The reconstructed 2D image.
        """

        self._ensure_open()
        with ExitStack() as resources:
            if self.ordsub_number != 1 and os_index is not None:
                # ordered-subsets
                sinogram_id = astra.data2d.create(
                    "-sino", self.proj_geom_OS[os_index], sinogram
                )
                resources.callback(astra.data2d.delete, sinogram_id)
            else:
                # traditional geometry
                sinogram_id = astra.data2d.create("-sino", self.proj_geom, sinogram)
                resources.callback(astra.data2d.delete, sinogram_id)

            # Create a data object for the reconstruction
            rec_id = astra.data2d.create("-vol", self.vol_geom)
            resources.callback(astra.data2d.delete, rec_id)

            # Create algorithm object
            cfg = astra.astra_dict(method)
            if self.processing_arch == "cpu":
                if self.ordsub_number != 1:
                    cfg["ProjectorId"] = self.proj_id_OS[os_index]
                else:
                    cfg["ProjectorId"] = self.proj_id
            else:
                cfg["option"] = {"GPUindex": self.device_index}
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectionDataId"] = sinogram_id
            if method == "FBP" or method == "FBP_CUDA":
                cfg["FilterType"] = self.fbp_filter_type
                if self.fbp_filter_parameter is not None:
                    cfg["FilterParameter"] = self.fbp_filter_parameter
                if self.fbp_filter_d is not None:
                    cfg["FilterD"] = self.fbp_filter_d

            # Create the algorithm object from the configuration structure
            alg_id = astra.algorithm.create(cfg)
            resources.callback(astra.algorithm.delete, alg_id)
            astra.algorithm.run(alg_id, iterations)

            # Get the result
            recon_slice = astra.data2d.get(rec_id)

            return recon_slice

    def _runAstraProj2D(
        self, image: np.ndarray, os_index: Union[int, None], method: str
    ) -> np.ndarray:
        """2D Forward projector for ASTRA parallel-beam

        Args:
            image (np.ndarray): Object to perform forward projection on.
            os_index (Union[int, None]): The number of ordered subsets.
            method (str): A method to choose from ASTRA's provided ones.

        Returns:
            np.ndarray: projected 2d data (sinogram)
        """

        self._ensure_open()
        with ExitStack() as resources:
            if isinstance(image, np.ndarray):
                rec_id = astra.data2d.link("-vol", self.vol_geom, image)
                resources.callback(astra.data2d.delete, rec_id)
            else:
                rec_id = image
            if self.ordsub_number != 1 and os_index is not None:
                # ordered-subsets
                sinogram_id = astra.data2d.create(
                    "-sino", self.proj_geom_OS[os_index], 0
                )
                resources.callback(astra.data2d.delete, sinogram_id)
            else:
                # traditional full data parallel beam projection geometry
                sinogram_id = astra.data2d.create("-sino", self.proj_geom, 0)
                resources.callback(astra.data2d.delete, sinogram_id)

            # Create algorithm object
            cfg = astra.astra_dict(method)
            if self.processing_arch == "cpu":
                if self.ordsub_number != 1:
                    cfg["ProjectorId"] = self.proj_id_OS[os_index]
                else:
                    cfg["ProjectorId"] = self.proj_id
            else:
                cfg["option"] = {"GPUindex": self.device_index}
            cfg["VolumeDataId"] = rec_id
            cfg["ProjectionDataId"] = sinogram_id

            # Create the algorithm object from the configuration structure
            alg_id = astra.algorithm.create(cfg)
            resources.callback(astra.algorithm.delete, alg_id)
            astra.algorithm.run(alg_id)

            # Get the result
            sinogram = astra.data2d.get(sinogram_id)

            return sinogram

    def _prepare_cupy_input(self, data, geometry, name):
        """Validate GPU storage and pack real input as contiguous float32.

        Copies are made only when dtype or layout requires one. Arrays from
        another device are rejected rather than silently transferred.
        """
        if xp is np:
            raise RuntimeError("CuPy with an accessible GPU is required")
        if not isinstance(data, xp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        expected_shape = tuple(astra.geom_size(geometry))
        if data.ndim != 3 or data.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {data.shape}"
            )
        if data.dtype.kind not in "biuf":
            raise TypeError(f"{name} must have a real numeric dtype, got {data.dtype}")
        if data.device.id != self.device_index:
            raise ValueError(
                f"{name} is on GPU {data.device.id}, but the projector uses "
                f"GPU {self.device_index}"
            )
        with xp.cuda.Device(self.device_index):
            return xp.ascontiguousarray(data, dtype=xp.float32)
