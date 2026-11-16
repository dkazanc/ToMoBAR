"""The series of classes for wrapping of ASTRA toolbox to perform projection/backprojection
and reconstruction of of 2D/3D parallel beam data.

GPLv3 license (defined by ASTRA toolbox)
"""

from contextlib import ExitStack
import weakref

import numpy as np
from tomobar.geometry import interleaved_view_indices
from tomobar.supp.array_prepare import prepare_array, validate_array

import cupy as xp
import astra


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

    def _validate_output(self, out, data, geometry):
        """Require writable, independent storage; never pack an output buffer."""
        host = isinstance(data, np.ndarray)
        if isinstance(out, np.ndarray) != host:
            raise TypeError("out must use the same array library as the input")
        validate_array(
            out,
            shape=astra.geom_size(geometry),
            device_index=self.device_index,
            array_module=xp,
            name="out",
        )
        if out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous float32 storage")
        if host and not out.flags.writeable:
            raise ValueError("out must be writable")
        overlaps = np.shares_memory if host else xp.may_share_memory
        if overlaps(data, out):
            raise ValueError("out must not overlap the input")

    def _project(self, data, subset_index, *, forward, out=None):
        self._ensure_open()
        projection = self._selected_geometry(subset_index)
        manager = astra.projector3d if self._is_3d else astra.projector
        input_geometry = self.vol_geom if forward else projection
        output_geometry = projection if forward else self.vol_geom
        if out is not None:
            self._validate_output(out, data, output_geometry)
        if isinstance(data, np.ndarray):
            prepared = prepare_array(
                data,
                shape=astra.geom_size(input_geometry),
                device_index=self.device_index,
                array_module=xp,
                pack=True,
            )
            if out is None:
                out = np.empty(astra.geom_size(output_geometry), dtype=np.float32)
            handle = self._projector_handle(subset_index)
            operation = manager.direct_FP if forward else manager.direct_BP
            operation(handle, prepared, out=out)
        else:
            prepared = self._prepare_cupy_input(data, input_geometry, "data")
            with xp.cuda.Device(self.device_index):
                if out is None:
                    out = xp.empty(astra.geom_size(output_geometry), dtype=xp.float32)
                handle = self._projector_handle(subset_index)
                operation = manager.direct_FP if forward else manager.direct_BP
                operation(handle, prepared, out=out)
        return out

    def forward(self, volume, *, subset_index=None, out=None):
        """Project a NumPy/CuPy array using ASTRA 2.5's direct DLPack API.

        None selects all views, even when subsets are configured. A subset
        index selects its interleaved views. Output uses the input array library.
        Noncontiguous or non-float32 real inputs are packed as float32.
        Optional out must be independent C-contiguous float32 storage on the
        same device/library with the exact output shape. It is overwritten
        and returned by identity. Without out, the result is a fresh array.
        """
        return self._project(volume, subset_index, forward=True, out=out)

    def backproject(self, projections, *, subset_index=None, out=None):
        """Backproject using a cached handle and the same out contract as forward."""
        return self._project(projections, subset_index, forward=False, out=out)

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
        return prepare_array(
            data,
            shape=astra.geom_size(geometry),
            device_index=self.device_index,
            array_module=xp,
            name=name,
            pack=True,
        )
