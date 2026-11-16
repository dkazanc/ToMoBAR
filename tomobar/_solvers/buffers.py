"""Private projection scratch owned by one solver state and geometry segment."""

import math


class ProjectionBuffers:
    """Reuse one projection slab and one volume, including uneven subsets.

    Returned arrays are scratch: the next operation overwrites them. Persistent
    solver quantities (CGLS residual/direction, OSEM sensitivity) must not use
    these buffers as their storage.
    """

    def __init__(self, projections, image, subset_count, array_module):
        self._shape = projections.shape
        self._projection_dtype = projections.dtype
        self._volume_shape = image.shape
        self._volume_dtype = image.dtype
        self._subsets = subset_count
        self._xp = array_module
        self._projection = None
        self._volume = None
        self._views = {}

    def projection(self, subset_index=0):
        rows, angles, columns = self._shape
        count = len(range(subset_index, angles, self._subsets))
        if self._projection is None:
            maximum = math.ceil(angles / self._subsets)
            self._projection = self._xp.empty(
                rows * maximum * columns, dtype=self._projection_dtype
            )
        if count not in self._views:
            self._views[count] = self._projection[: rows * count * columns].reshape(
                rows, count, columns
            )
        return self._views[count]

    def volume(self):
        if self._volume is None:
            self._volume = self._xp.empty(self._volume_shape, dtype=self._volume_dtype)
        return self._volume
