"""Geometry-owned bin ordering and bounded shared-memory USFFT scatter."""

from dataclasses import dataclass
import math

import cupy as cp
import numpy as np

from tomobar.cuda_kernels.loader import get_cuda_kernel, load_cuda_module


@dataclass(frozen=True, slots=True)
class ScatterPlan:
    """Reusable sample positions, bin ordering and bounded work chunks.

    Planning synchronises once to create task metadata from per-bin counts.
    Arrays are slice-independent and owned by the corresponding view plan.
    """

    positions: cp.ndarray
    order: cp.ndarray
    offsets: cp.ndarray
    task_bins: cp.ndarray
    task_starts: cp.ndarray
    tile: int
    chunk: int
    bins_per_axis: int

    @classmethod
    def build(cls, angles, n, *, tile=32, chunk=512):
        """Bin samples using precisely the CUDA interpolation coordinates."""
        if tile not in (16, 32) or chunk < 1:
            raise ValueError("tile must be 16 or 32 and chunk must be positive")
        count = n * len(angles)
        if count > np.iinfo(np.int32).max:
            raise ValueError("scatter sample count exceeds int32 planning capacity")
        bins_per_axis = math.ceil(2 * n / tile)
        positions = cp.empty((count, 2), dtype=cp.float32)
        bins = cp.empty(count, dtype=cp.int32)
        kernel = get_cuda_kernel(
            load_cuda_module("usfft_scatter"), "usfft_scatter_bins"
        )
        kernel(
            (math.ceil(count / 256),),
            (256,),
            (
                angles,
                positions,
                bins,
                *(np.int32(v) for v in (n, len(angles), tile, bins_per_axis)),
            ),
        )
        order = cp.argsort(bins).astype(cp.int32)
        counts = cp.asnumpy(cp.bincount(bins, minlength=bins_per_axis**2)).astype(
            np.int64
        )
        offsets = np.concatenate(([0], np.cumsum(counts)))
        bins_host, starts = [], []
        for index, size in enumerate(counts):
            for start in range(int(offsets[index]), int(offsets[index] + size), chunk):
                bins_host.append(index)
                starts.append(start)
        return cls(
            positions,
            order,
            cp.asarray(offsets, dtype=cp.int32),
            cp.asarray(bins_host, dtype=cp.int32),
            cp.asarray(starts, dtype=cp.int32),
            tile,
            chunk,
            bins_per_axis,
        )

    @property
    def nbytes(self):
        """Persistent geometry metadata bytes, excluding shared kernel caches."""
        return sum(
            a.nbytes
            for a in (
                self.positions,
                self.order,
                self.offsets,
                self.task_bins,
                self.task_starts,
            )
        )

    def scatter(self, polar, cartesian, mu, *, radius=7):
        """Accumulate into a pre-cleared Cartesian grid without allocating arrays."""
        rows, views, n = polar.shape
        if radius not in (5, 7):
            raise ValueError("tiled scatter supports radius 5 or 7")
        kernel = get_cuda_kernel(
            load_cuda_module("usfft_scatter"),
            "usfft_scatter_tiled" if radius == 7 else "usfft_scatter_tiled_5",
        )
        kernel(
            (len(self.task_bins), rows),
            (256,),
            (
                polar,
                cartesian,
                self.positions,
                self.order,
                self.task_bins,
                self.task_starts,
                self.offsets,
                np.int32(n),
                np.int32(views),
                np.float32(mu),
                np.int32(self.tile),
                np.int32(self.bins_per_axis),
                np.int32(self.chunk),
            ),
            shared_mem=(self.tile + 2 * radius) ** 2 * 8,
        )
