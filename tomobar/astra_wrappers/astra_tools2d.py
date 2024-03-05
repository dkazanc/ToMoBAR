from tomobar.astra_wrappers.astra_base import AstraBase

import numpy as np


###########Child class############
class AstraTools2D(AstraBase):
    """2D parallel beam projection/backprojection child class.

    Args:
        AstraBase (class): the base class.
    """

    def __init__(
        self,
        detectors_x,
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
            detectors_y=None,
        )

        if verbosity:
            print("The shape of the input data has been established...")
        super()._set_vol2d_geometry()
        if verbosity:
            print("2D volume geometry initialised...")
        if ordsub_number == 1 or ordsub_number is None:
            if processing_arch == "cpu":
                super()._set_cpu_projection2d_parallel_geometry()
            else:
                super()._set_gpu_projection2d_parallel_geometry()
            if verbosity:
                print(
                    "2D <{}> classical parallel-beam projection geometry initialised...".format(
                        processing_arch
                    )
                )
        else:
            super()._setOS_indices()
            super()._set_projection2d_OS_parallel_geometry()
            if verbosity:
                print(
                    "2D <{}> ordered-subsets parallel-beam projection geometry initialised...".format(
                        processing_arch
                    )
                )

    def _forwproj(self, image: np.ndarray) -> np.ndarray:
        astra_method = "FP_CUDA"  # 2d forward projection
        if self.processing_arch == "cpu":
            astra_method = "FP"
        return super()._runAstraProj2D(image, None, astra_method)

    def _forwprojOS(self, image: np.ndarray, os_index: int) -> np.ndarray:
        astra_method = "FP_CUDA"  # 2d forward projection
        if self.processing_arch == "cpu":
            astra_method = "FP"
        return super()._runAstraProj2D(image, os_index, astra_method)

    def _backproj(self, sinogram: np.ndarray) -> np.ndarray:
        astra_method = "BP_CUDA"  # 2D back projection
        if self.processing_arch == "cpu":
            astra_method = "BP"
        return super()._runAstraBackproj2D(sinogram, astra_method, 1, None)

    def _backprojOS(self, sinogram: np.ndarray, os_index: int) -> np.ndarray:
        astra_method = "BP_CUDA"  # 2D back projection
        if self.processing_arch == "cpu":
            astra_method = "BP"
        return super()._runAstraBackproj2D(sinogram, astra_method, 1, os_index)

    def _fbp(self, sinogram: np.ndarray) -> np.ndarray:
        astra_method = "FBP_CUDA"  # 2D FBP reconstruction
        if self.processing_arch == "cpu":
            astra_method = "FBP"
        return super()._runAstraBackproj2D(sinogram, astra_method, 1, None)

    def _sirt(self, sinogram: np.ndarray, iterations: int) -> np.ndarray:
        astra_method = "SIRT_CUDA"  # perform 2D SIRT reconstruction
        if self.processing_arch == "cpu":
            astra_method = "SIRT"
        return super()._runAstraBackproj2D(sinogram, astra_method, iterations, None)

    def _cgls(self, sinogram: np.ndarray, iterations: int) -> np.ndarray:
        astra_method = "CGLS_CUDA"  # perform 2D CGLS reconstruction
        if self.processing_arch == "cpu":
            astra_method = "CGLS"
        return super()._runAstraBackproj2D(sinogram, astra_method, iterations, None)
