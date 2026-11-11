"""The series of classes for wrapping of ASTRA toolbox to perform projection/backprojection
and reconstruction of of 2D/3D parallel beam data.

GPLv3 license (defined by ASTRA toolbox)
"""

from contextlib import ExitStack
import weakref

import numpy as np
from tomobar.geometry import interleaved_view_indices

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
    """Shared ASTRA execution and resource ownership for geometry-based wrappers.

    Subclasses provide converted dictionaries, device selection and subset
    indices. Use close() or a context manager for prompt projector cleanup.
    """

    _is_3d = False

    def _init_projector_resources(self):
        """Initialise shared ownership without constructing geometry."""
        self._closed = False
        self._projector_handles = {}
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

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _setOS_indices(self, view_count):
        """Create interleaved view indices without padding or sentinel values."""
        self.subset_indices = interleaved_view_indices(view_count, self.ordsub_number)
        # Legacy metadata: newInd_Vec is now a tuple, not a padded matrix.
        self.newInd_Vec = self.subset_indices
        self.NumbProjBins = max(map(len, self.subset_indices))

    def _selected_geometry(self, subset_index):
        if subset_index is None:
            return self.proj_geom
        if isinstance(subset_index, (bool, np.bool_)) or not isinstance(
            subset_index, (int, np.integer)
        ):
            raise TypeError("subset_index must be an integer or None")
        if not 0 <= subset_index < self.ordsub_number:
            raise ValueError("subset_index is out of range")
        return self.proj_geom_OS[int(subset_index)]

    def _projector_handle(self, subset_index):
        """Cache one handle for all views and one for each used subset."""
        self._ensure_open()
        geometry = self._selected_geometry(subset_index)
        if subset_index not in self._projector_handles:
            handle = astra.create_projector(
                "cuda3d" if self._is_3d else "cuda",
                geometry,
                self.vol_geom,
                options={"GPUindex": self.device_index},
            )
            self._register_projector(handle)
            self._projector_handles[subset_index] = handle
        return self._projector_handles[subset_index]

    def _project(self, data, subset_index, *, forward):
        self._ensure_open()
        projection = self._selected_geometry(subset_index)
        manager = astra.projector3d if self._is_3d else astra.projector
        input_geometry = self.vol_geom if forward else projection
        output_geometry = projection if forward else self.vol_geom
        if isinstance(data, np.ndarray):
            if data.shape != tuple(astra.geom_size(input_geometry)):
                raise ValueError(
                    f"Input must have shape {astra.geom_size(input_geometry)}, got {data.shape}"
                )
            if data.dtype.kind not in "biuf":
                raise TypeError("Input must have a real numeric dtype")
            prepared = np.ascontiguousarray(data, dtype=np.float32)
            out = np.empty(astra.geom_size(output_geometry), dtype=np.float32)
            handle = self._projector_handle(subset_index)
            operation = manager.direct_FP if forward else manager.direct_BP
            operation(handle, prepared, out=out)
        else:
            prepared = self._prepare_cupy_input(data, input_geometry, "data")
            with xp.cuda.Device(self.device_index):
                out = xp.empty(astra.geom_size(output_geometry), dtype=xp.float32)
                handle = self._projector_handle(subset_index)
                operation = manager.direct_FP if forward else manager.direct_BP
                operation(handle, prepared, out=out)
        return out

    def forward(self, volume, *, subset_index=None):
        """Project a NumPy/CuPy array using ASTRA 2.5's direct DLPack API.

        None selects all views, even when subsets are configured. A subset
        index selects its interleaved views. Output uses the input array library.
        Noncontiguous or non-float32 real inputs are packed as float32.
        """
        return self._project(volume, subset_index, forward=True)

    def backproject(self, projections, *, subset_index=None):
        """Backproject NumPy/CuPy projections using the same cached handle."""
        return self._project(projections, subset_index, forward=False)

    def close(self):
        """Release cached handles. Repeated calls are harmless."""
        self._closed = True
        self._projector_finalizer()
        self._projector_handles.clear()

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
        if data.shape != expected_shape:
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
