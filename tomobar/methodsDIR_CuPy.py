"""Reconstruction class for direct methods using CuPy-library.

* Forward/Backward projection (ASTRA with DirectLink and CuPy)
* Filtered Back Projection (ASTRA, Filter implemented in CuPy)
"""

#import numpy as np
nocupy = False
try:
    import cupy as xp
    from cupyx import scipy
except ImportError:
    nocupy = True
    print("Cupy library is a required dependency for this part of the code, please install")

if nocupy:
    import numpy as xp
    import scipy

from tomobar.cuda_kernels import load_cuda_module
from tomobar.supp.suppTools import _check_kwargs
from tomobar.supp.funcs import _data_dims_swapper
from tomobar.methodsDIR import RecToolsDIR

def _filtersinc3D_cupy(projection3D: xp.ndarray) -> xp.ndarray:
    """Applies a SINC filter to 3D projection data

    Args:
        data : xp.ndarray
            Projection data as a CuPy array.

    Returns:
        xp.ndarray
            The filtered projectiond data as a CuPy array.
    """
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = xp.shape(projection3D)

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # Use real FFT to save space and time
    proj_f = scipy.fft.rfft(
        projection3D, axis=-1, norm="backward", overwrite_x=True
    )

    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    f = xp.empty((1, 1, DetectorsLengthH // 2 + 1), dtype=xp.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0 / projectionsNum / DetectorsLengthH
    filter_prep(
        grid=(1, 1, 1),
        block=(bx, 1, 1),
        args=(xp.float32(a), f, xp.int32(DetectorsLengthH), xp.float32(multiplier)),
        shared_mem=bx * 4,
    )

    # actual filtering
    proj_f *= f

    return scipy.fft.irfft(
        proj_f, projection3D.shape[2], axis=-1, norm="forward", overwrite_x=True
    )


class RecToolsDIRCuPy(RecToolsDIR):
    """Reconstruction class using DIRect methods with CuPy API.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float, ndarray): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        device_projector (int): An index (integer) of a specific GPU device.        
    """

    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimV,  # Vertical detector dimension (3D case)
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # Reconstructed object dimensions (scalar)
        device_projector=0,  # set an index (integer) of a specific GPU device
    ):
        super().__init__(
            DetectorsDimH,
            DetectorsDimV,
            CenterRotOffset,
            AnglesVec,
            ObjSize,
            device_projector,
        )
        if DetectorsDimV == 0 or DetectorsDimV is None:
            raise ValueError("2D CuPy reconstruction is not yet supported, only 3D is")

    def FORWPROJ(self, data: xp.ndarray, **kwargs) -> xp.ndarray:
        """Module to perform forward projection of 2d/3d data as a cupy array

        Args:
            data (xp.ndarray): 2D or 3D object as a cupy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the OUTPUT data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.
            
        Returns:
            xp.ndarray: Forward projected cupy array (projection data)
        """
        projected = self.Atools._forwprojCuPy(data)
        for key, value in kwargs.items():
                if key == "data_axes_labels_order" and value is not None:
                    if self.geom == "2D":
                        projected = _data_dims_swapper(projected, value, ["angles", "detX"])
                    else:
                        projected = _data_dims_swapper(projected, value, ["detY", "angles", "detX"])

        return projected

    def BACKPROJ(self, projdata: xp.ndarray, **kwargs) -> xp.ndarray:
        """Module to perform back-projection of 2d/3d data as a cupy array

        Args:
            projdata (xp.ndarray): 2D/3D projection data as a cupy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.
                        
        Returns:
            xp.ndarray: Backprojected 2D/3D object
        """
        for key, value in kwargs.items():
                if key == "data_axes_labels_order" and value is not None:
                    if self.geom == "2D":
                        projdata = _data_dims_swapper(projdata, value, ["angles", "detX"])
                    else:
                        projdata = _data_dims_swapper(projdata, value, ["detY", "angles", "detX"])

        return self.Atools._backprojCuPy(projdata)

    def FBP(self, data: xp.ndarray, **kwargs) -> xp.ndarray:
        """Filtered backprojection reconstruction on a CuPy array using a custom built SINC filter.

        Args:
            data (xp.ndarray): projection data as a CuPy array

        Keyword Args:
            data_axes_labels_order (Union[list, None], optional): The order of the axes labels for the input data.
                 When "None" we assume  ["angles", "detX"] for 2D and ["angles", "detY", "detX"] for 3D.
            recon_mask_radius (float): zero values outside the circular mask of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].

        Returns:
            xp.ndarray: The FBP reconstructed volume as a CuPy array.
        """
        for key, value in kwargs.items():
                if key == "data_axes_labels_order" and value is not None:
                    if self.geom == "2D":
                        data = _data_dims_swapper(data, value, ["angles", "detX"])
                    else:
                        data = _data_dims_swapper(data, value, ["angles", "detY", "detX"])

        # filter the data on the GPU and keep the result there
        data = _filtersinc3D_cupy(data)
        data = xp.ascontiguousarray(xp.swapaxes(data, 0, 1))
        reconstruction = self.Atools._backprojCuPy(data)  # 3d backprojecting
        xp._default_memory_pool.free_all_blocks()
        return _check_kwargs(reconstruction, **kwargs)    
