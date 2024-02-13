from tomobar.astra_wrappers.astra_base import AstraBase

import numpy as np

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp
except ImportError:
    import numpy as xp


###########Child class############
class AstraTools3D(AstraBase):
    """3D parallel beam projection/backprojection child class.

    Args:
        AstraBase (class): the inhereted base class.
    """

    def __init__(
        self,
        detectors_x,
        detectors_y,
        angles_vec,
        centre_of_rotation,
        recon_size,
        processing_arch,
        device_index,
        ordsub_number=1,
        verbosity=False,
    ):
        super().__init__(
            detectors_x,
            angles_vec,
            centre_of_rotation,
            recon_size,
            processing_arch,
            device_index,
            ordsub_number,
            detectors_y,
        )
        if verbosity:
            print("The shape of the input data has been established...")
        super()._set_vol3d_geometry()
        if verbosity:
            print("3D volume geometry initialised...")
        if ordsub_number == 1 or ordsub_number is None:
            if processing_arch == "cpu":
                raise ValueError(
                    "3D CPU reconstruction is not supported, please use GPU"
                )
            else:
                super()._set_gpu_projection3d_parallel_geometry()
            if verbosity:
                print(
                    "3D <{}> classical parallel-beam projection geometry initialised...".format(
                        processing_arch
                    )
                )
        else:
            super()._setOS_indices()
            super()._set_projection3d_OS_parallel_geometry()
            if verbosity:
                print(
                    "3D <{}> ordered-subsets parallel-beam projection geometry initialised...".format(
                        processing_arch
                    )
                )

    def _forwproj(self, object3D: np.ndarray) -> np.ndarray:
        return super().runAstraProj3D(object3D, None)

    def _forwprojOS(self, object3D: np.ndarray, os_index: int) -> np.ndarray:
        return super().runAstraProj3D(object3D, os_index)

    def _forwprojCuPy(self, object3D: xp.ndarray) -> xp.ndarray:
        return super().runAstraProj3DCuPy(
            object3D, None
        )  # 3D forward projection using CuPy array

    def _forwprojOSCuPy(self, object3D: xp.ndarray, os_index: int) -> xp.ndarray:
        return super().runAstraProj3DCuPy(
            object3D, os_index
        )  # 3D forward projection using CuPy array

    def _backproj(self, proj_data: np.ndarray) -> np.ndarray:
        return super().runAstraBackproj3D(
            proj_data, "BP3D_CUDA", 1, None
        )  # 3D backprojection

    def _backprojOS(self, proj_data: np.ndarray, os_index: int) -> np.ndarray:
        return super().runAstraBackproj3D(
            proj_data, "BP3D_CUDA", 1, os_index
        )  # 3D OS backprojection

    def _backprojCuPy(self, proj_data: xp.ndarray) -> xp.ndarray:
        return super().runAstraBackproj3DCuPy(
            proj_data, "BP3D_CUDA", None
        )  # 3D backprojection using CuPy array

    def _backprojOSCuPy(self, proj_data: xp.ndarray, os_index: int) -> xp.ndarray:
        return super().runAstraBackproj3DCuPy(
            proj_data, "BP3D_CUDA", os_index
        )  # 3d back-projection using CuPy array for a specific subset

    def _sirt(self, proj_data: np.ndarray, iterations: int) -> np.ndarray:
        return super().runAstraBackproj3D(
            proj_data, "SIRT3D_CUDA", iterations, None
        )  # 3D SIRT reconstruction

    def _cgls(self, proj_data: np.ndarray, iterations: int) -> np.ndarray:
        return super().runAstraBackproj3D(
            proj_data, "CGLS3D_CUDA", iterations, None
        )  # 3D CGLS reconstruction
