"""Reconstruction class for direct reconstruction methods (2D/3D).

* :func:`RecToolsDIR.forward` and :func:`RecToolsDIR.backproject` Forward/Backward 2D/3D projection (ASTRA-Toolbox)
* :func:`RecToolsDIR.FBP` Filtered Back Projection 2D/3D (ASTRA with the custom built filter).
"""

import numpy as np

from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)
from tomobar.reconstruction_geometry import GeometryReconstruction
from tomobar.supp.axis_order import (
    _transpose_data_axes,
    _resolve_data_axes,
)
from tomobar.supp.suppTools import check_kwargs, _apply_horiz_detector_padding


class RecToolsDIR(GeometryReconstruction):
    """Direct reconstruction from explicit acquisition and volume geometry.

    Use Parallel3DGeometry with VolumeGeometry, or Parallel2DGeometry with
    Volume2DGeometry. detector_padding adds edge pixels per detector side and
    derives a padded scan without modifying either supplied geometry object.
    The requested volume shape is always preserved. Reconstruction uses CUDA
    projectors selected by device_index; NumPy inputs and outputs stay on the
    host while projection and filtering execute on the GPU.
    """

    def __init__(
        self,
        *,
        projection_geometry: Parallel2DGeometry | Parallel3DGeometry,
        volume_geometry: Volume2DGeometry | VolumeGeometry,
        detector_padding: int = 0,
        device_index: int = 0,
    ):
        self._init_device(device_index)
        self._init_geometry(projection_geometry, volume_geometry, detector_padding)
        wrapper = AstraTools3D if self.geom == "3D" else AstraTools2D
        self.Atools = wrapper(
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=volume_geometry,
            device_index=self.device_index,
        )

    def forward(self, data: np.ndarray, *, output_axes=None, **kwargs) -> np.ndarray:
        """Module to perform forward projection of 2d/3d data numpy array

        Args:
            data (np.ndarray): 2D or 3D object

        Keyword Args:
            output_axes (list or tuple, optional): Explicit projection axis order.
                Uses the default below when omitted. Conflicting aliases raise ValueError.
            data_axes_labels_order (list or tuple, optional): Legacy alias for output_axes.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray: Forward projected numpy array (projection data)
        """
        native_axes = (
            ("angles", "detX") if self.geom == "2D" else ("detY", "angles", "detX")
        )
        axes = _resolve_data_axes(
            output_axes, kwargs.get("data_axes_labels_order"), native_axes
        )
        self._ensure_open()
        projected = self.Atools.forward(data)
        projected = self._crop_projection(projected)
        return _transpose_data_axes(
            projected, source_axes=native_axes, target_axes=axes
        )

    def backproject(self, data: np.ndarray, *, input_axes=None, **kwargs) -> np.ndarray:
        """Module to perform back-projection of 2d/3d data numpy array

        Args:
            data (np.ndarray): 2D/3D projection data

        Keyword Args:
            input_axes (list or tuple, optional): Explicit projection axis order.
                Uses the default below when omitted. Conflicting aliases raise ValueError.
            data_axes_labels_order (list or tuple, optional): Legacy alias for input_axes.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.

        Returns:
            np.ndarray: Backprojected 2D/3D object
        """
        native_axes = (
            ("angles", "detX") if self.geom == "2D" else ("detY", "angles", "detX")
        )
        axes = _resolve_data_axes(
            input_axes, kwargs.get("data_axes_labels_order"), native_axes
        )
        data = _transpose_data_axes(data, source_axes=axes, target_axes=native_axes)

        data = _apply_horiz_detector_padding(data, self.detector_padding, cupyrun=False)
        self._ensure_open()
        return self.Atools.backproject(data)

    def FBP(self, data: np.ndarray, *, input_axes=None, **kwargs) -> np.ndarray:
        """Filtered backprojection reconstruction module for 2D or 3D data.

        Args:
            data (np.ndarray): 2D or 3D projection data.

        Keyword Args:
            input_axes (list or tuple, optional): Explicit projection axis order.
                Uses the default below when omitted. Conflicting aliases raise ValueError.
            data_axes_labels_order (list or tuple, optional): Legacy alias for input_axes.
                 When "None" we assume  ["angles", "detX"] for 2D and ["detY", "angles", "detX"] for 3D.
            filter_type (str) : type of projection filter, see https://astra-toolbox.com/docs/algs/FBP_CUDA.html for all available options
            filter_parameter (float) : Parameter value for the 'tukey', 'gaussian', 'blackman' and 'kaiser' filter types (see ASTRA's API).
            filter_d (float) : "D" parameter value for 'shepp-logan', 'cosine', 'hamming' and 'hann' filter types (see ASTRA's API).

        Returns:
            np.ndarray: FBP reconstructed 2D or 3D object.
        """
        native_axes = (
            ("angles", "detX") if self.geom == "2D" else ("detY", "angles", "detX")
        )
        axes = _resolve_data_axes(
            input_axes, kwargs.get("data_axes_labels_order"), native_axes
        )
        data = _transpose_data_axes(data, source_axes=axes, target_axes=native_axes)
        for key, value in kwargs.items():
            if key == "filter_type":
                self.Atools.fbp_filter_type = value
            if key == "filter_parameter":
                self.Atools.fbp_filter_parameter = value
            if key == "filter_d":
                self.Atools.fbp_filter_d = value

        cupyrun = False
        kwargs.update({"cupyrun": cupyrun})  # needed for agnostic array cropping

        data = _apply_horiz_detector_padding(data, self.detector_padding, cupyrun)
        if self.geom == "2D":
            return check_kwargs(self.Atools._fbp(data), **kwargs)
        return check_kwargs(
            self.Atools.backproject(
                _filtersinc3D(data, device_index=self.device_index)
            ),
            **kwargs,
        )


def _filtersinc3D(projection3D: np.ndarray, *, device_index: int = 0):
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
    # Retain the established filter coefficients; run the batched FFT on CUDA.
    import cupy as cp

    with cp.cuda.Device(device_index):
        data = cp.asarray(projection3D, dtype=cp.float32)
        spectrum = cp.fft.fft2(data, axes=(0, 2))
        spectrum *= cp.asarray(np.fft.fftshift(r), dtype=cp.float32)[None, None, :]
        filtered = cp.fft.ifft2(spectrum, axes=(0, 2)).real
        return cp.asnumpy(multiplier * filtered)
