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

from typing import Union, Optional


###########Child class############
class AstraTools3D(AstraBase):
    """3D parallel beam projection/backprojection child class.

    Args:
        AstraBase (class): the inherited base class.
    """

    def __init__(
        self,
        detectors_x: int,
        detectors_x_pad: int,
        detectors_y: int,
        angles_vec: np.ndarray,
        centre_of_rotation: Union[float, np.ndarray],
        recon_size: int,
        processing_arch: str,
        device_index: int,
        ordsub_number: Optional[int] = None,
        verbosity: bool = False,
    ):
        super().__init__(
            detectors_x,
            detectors_x_pad,
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

    def _fbp(self, proj_data: np.ndarray) -> np.ndarray:
        """
        as 3D FBP using ASTRA is not implemented by the third-party, we return just a backprojection here, the
        3D FBP implementation using SINC filter is in methodsDIR.
        """
        return super().runAstraBackproj3D(
            proj_data, "BP3D_CUDA", 1, None
        )  # NOTE: 3D FBP using ASTRA is not implemented by the third-part, we use the bespoke implementation here

    def _backprojCuPy(self, proj_data: xp.ndarray) -> xp.ndarray:
        return super().runAstraBackproj3DCuPy(
            proj_data, "BP3D_CUDA", None
        )  # 3D backprojection using CuPy array

    def _backprojOSCuPy(self, proj_data: xp.ndarray, os_index: int) -> xp.ndarray:
        return super().runAstraBackproj3DCuPy(
            proj_data, "BP3D_CUDA", os_index
        )  # 3d back-projection using CuPy array for a specific subset
