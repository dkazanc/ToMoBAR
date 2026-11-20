"""ASTRA ownership regression tests; no ASTRA, CuPy or CUDA installation needed.

Run with pytest or ``python -m unittest tests.astra.test_resource_lifecycle``. Real wrapper modules are loaded with
isolated dependency mocks, without replacing modules used by other tests.
"""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tests.helpers.reconstruction import MockGPUArray, ResourceHarness
from tomobar.projectors._astra.geometry import (
    parallel2d_vectors,
    parallel3d_vectors,
)
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)


class TestAstraResourceLifecycle(unittest.TestCase, ResourceHarness):
    def setUp(self):
        ResourceHarness.__init__(self)
        self.addCleanup(self.close)

    def assert_no_temporaries(self):
        for manager in ("data2d", "data3d", "algorithm"):
            self.assertFalse(self.backend.live[manager], manager)

    def test_2d_handles_are_lazy_shared_and_closed_once(self):
        for subsets in (1, 2):
            wrapper = self.make_wrapper(subsets=subsets)
            self.assertFalse(self.backend.live["projector"])
            self.astra.geom_size.side_effect = lambda geom: (
                (geom["GridRowCount"], geom["GridColCount"])
                if "GridRowCount" in geom
                else (len(geom["Vectors"]), geom["DetectorCount"])
            )
            handles = {}
            for index in (None, *range(subsets), None):
                projected = wrapper.forward(
                    np.ones((2, 2), dtype=np.float64), subset_index=index
                )
                handle = self.astra.projector.direct_FP.call_args.args[0]
                result = wrapper.backproject(projected, subset_index=index)
                self.assertEqual(result.shape, (2, 2))
                self.assertEqual(
                    self.astra.projector.direct_BP.call_args.args[0], handle
                )
                if index in handles:
                    self.assertEqual(handles[index], handle)
                handles[index] = handle
                self.assertEqual(len(self.backend.live["projector"]), len(handles))
                self.assert_no_temporaries()
            self.astra.data2d.create.assert_not_called()
            self.astra.data2d.link.assert_not_called()
            self.astra.algorithm.create.assert_not_called()
            wrapper.close()
            wrapper.close()
            self.assertFalse(self.backend.live["projector"])
            for handle in handles.values():
                self.assertEqual(self.backend.deleted.count(("projector", handle)), 1)
            self.astra.geom_size.side_effect = None

    def test_3d_handles_are_lazy_reused_and_released_once(self):
        wrapper = self.make_wrapper(3, subsets=3)
        self.assertFalse(self.backend.live["projector3d"])
        self.astra.create_projector.assert_not_called()
        handles = {}
        for index in (None, 2, 0, None, 2):
            wrapper.forward(self.array, subset_index=index)
            fp_handle = self.astra.projector3d.direct_FP.call_args.args[0]
            wrapper.backproject(self.array, subset_index=index)
            bp_handle = self.astra.projector3d.direct_BP.call_args.args[0]
            self.assertEqual(fp_handle, bp_handle)
            if index in handles:
                self.assertEqual(handles[index], fp_handle)
            handles[index] = fp_handle
            self.assertEqual(len(self.backend.live["projector3d"]), len(handles))
            self.assert_no_temporaries()
        self.assertEqual(len(handles), 3)
        self.assertNotIn(1, wrapper._projector_handles)
        self.assertEqual(self.astra.create_projector.call_count, 3)
        self.astra.data3d.link.assert_not_called()
        self.astra.data3d.create.assert_not_called()
        self.astra.algorithm.create.assert_not_called()
        wrapper.close()
        wrapper.close()
        self.assertFalse(self.backend.live["projector3d"])
        for handle in handles.values():
            self.assertEqual(self.backend.deleted.count(("projector3d", handle)), 1)
        for operation in (wrapper.forward, wrapper.backproject):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation(self.array)

    def test_supplied_gpu_output_is_reused_without_allocation(self):
        wrapper = self.make_wrapper(3)
        self.cupy.may_share_memory = lambda a, b: np.shares_memory(a.storage, b.storage)
        for operation in (wrapper.forward, wrapper.backproject):
            self.output = MockGPUArray(np.empty((2, 3, 4), dtype=np.float32))
            self.assertIs(operation(self.array, out=self.output), self.output)
        self.cupy.empty.assert_not_called()

    def test_wrong_device_output_is_rejected_before_allocation(self):
        wrapper = self.make_wrapper(3)
        out = MockGPUArray(np.empty((2, 3, 4), dtype=np.float32), device=1)
        with self.assertRaisesRegex(ValueError, "out is on GPU"):
            wrapper.forward(self.array, out=out)
        self.cupy.empty.assert_not_called()
        self.astra.create_projector.assert_not_called()

    def test_readonly_host_output_is_rejected(self):
        wrapper = self.make_wrapper(3)
        data = np.ones((2, 3, 4), dtype=np.float32)
        out = np.empty_like(data)
        out.flags.writeable = False
        with self.assertRaisesRegex(ValueError, "writable"):
            wrapper.forward(data, out=out)
        self.astra.create_projector.assert_not_called()

    def test_fbp2d_scopes_temporary_resources_on_success_and_failure(self):
        wrapper = self.make_wrapper(2)
        for failure in (
            None,
            ("data2d.create", 1),
            ("data2d.create", 2),
            ("algorithm.create", 1),
            ("algorithm.run", 1),
            ("data2d.get", 1),
        ):
            self.backend.counts.clear()
            self.backend.fail_at = failure
            if failure:
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    wrapper._fbp(np.ones((4, 4), dtype=np.float32))
            else:
                self.assertIs(
                    wrapper._fbp(np.ones((4, 4), dtype=np.float32)), self.backend.result
                )
            self.assert_no_temporaries()
        self.backend.fail_at = None
        wrapper.close()

    def test_2d_rejects_astra_ids_and_non_array_inputs(self):
        wrapper = self.make_wrapper(2)
        for data in (123, object()):
            with self.assertRaisesRegex(TypeError, "CuPy"):
                wrapper.forward(data)
        self.astra.create_projector.assert_not_called()

    def test_failed_direct_operations_retain_cached_handles_until_close(self):
        for forward in (False, True):
            for failure in (("empty", 1), ("projector3d.create", 1), ("direct", 1)):
                wrapper = self.make_wrapper(3, subsets=2)
                self.backend.counts.clear()
                self.backend.fail_at = failure
                operation = wrapper.forward if forward else wrapper.backproject
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    operation(self.array, subset_index=1)
                expected = 1 if failure[0] == "direct" else 0
                self.assertEqual(len(wrapper._projector_handles), expected)
                self.assert_no_temporaries()
                self.backend.fail_at = None
                operation(self.array, subset_index=1)
                self.assertEqual(len(wrapper._projector_handles), 1)
                wrapper.close()
                self.assertFalse(self.backend.live["projector3d"])

    def test_context_manager_closes_after_exception_and_rejects_reuse(self):
        wrapper = self.make_wrapper()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with wrapper as entered:
                self.assertIs(entered, wrapper)
                raise RuntimeError("body failed")
        self.assertFalse(self.backend.live["projector"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            wrapper.forward(np.ones((2, 2)))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            wrapper.__enter__()

    def test_finalizer_releases_projectors_without_retaining_wrapper(self):
        for dim in (2, 3):
            wrapper = self.make_wrapper(dim, register_cleanup=False)
            wrapper.forward(
                self.array if dim == 3 else np.ones((2, 3, 4), dtype=np.float32)
            )
            ref = weakref.ref(wrapper)
            del wrapper
            gc.collect()
            self.assertIsNone(ref())
            self.assertFalse(
                self.backend.live["projector" if dim == 2 else "projector3d"]
            )

    def test_failed_2d_subset_creation_keeps_existing_handle_until_close(self):
        wrapper = self.make_wrapper(2, subsets=2)
        array = np.ones((2, 3, 4), dtype=np.float32)
        wrapper.forward(array, subset_index=0)
        first_handle = wrapper._projector_handles[0]
        self.backend.fail_at = ("projector.create", 2)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            wrapper.forward(array, subset_index=1)
        self.assertEqual(wrapper._projector_handles, {0: first_handle})
        self.backend.fail_at = None
        wrapper.forward(array, subset_index=1)
        wrapper.close()
        self.assertFalse(self.backend.live["projector"])

    def test_real_inputs_are_packed_without_modifying_the_original(self):
        for forward in (False, True):
            wrapper = self.make_wrapper(3)
            for dtype in (np.float32, np.float64, np.float16, np.int16, np.bool_):
                base = np.arange(24).reshape(2, 3, 4).astype(dtype)
                layouts = {
                    "contiguous": base,
                    "fortran": np.asfortranarray(base),
                    "transposed": np.arange(24)
                    .reshape(3, 2, 4)
                    .astype(dtype)
                    .swapaxes(0, 1),
                    "strided": np.arange(48).reshape(2, 3, 8).astype(dtype)[..., ::2],
                    "reversed": base[..., ::-1],
                }
                for layout, storage in layouts.items():
                    with self.subTest(forward=forward, dtype=dtype, layout=layout):
                        data = MockGPUArray(storage)
                        before = storage.copy()
                        if forward:
                            result = wrapper.forward(data, subset_index=None)
                        else:
                            result = wrapper.backproject(data, subset_index=None)
                        np.testing.assert_array_equal(
                            self.projected_input, before.astype(np.float32)
                        )
                        np.testing.assert_array_equal(storage, before)
                        self.assertEqual(result.dtype, np.float32)
                        self.assertTrue(result.flags.c_contiguous)
                        direct = (
                            self.astra.projector3d.direct_FP
                            if forward
                            else self.astra.projector3d.direct_BP
                        )
                        passed_input = direct.call_args.args[1]
                        self.assertIs(direct.call_args.kwargs["out"], result)
                        if dtype == np.float32 and storage.flags.c_contiguous:
                            self.assertIs(passed_input, data)
                        else:
                            self.assertIsNot(passed_input, data)
            wrapper.close()

    def test_invalid_gpu_inputs_fail_before_linking_or_allocating(self):
        cases = (
            (object(), TypeError, "CuPy array"),
            (MockGPUArray(np.ones((2, 4))), ValueError, "shape"),
            (MockGPUArray(np.ones((3, 2, 4))), ValueError, "shape"),
            (MockGPUArray(np.ones((2, 3, 8))), ValueError, "shape"),
            (
                MockGPUArray(np.ones((2, 3, 4), dtype=np.complex64)),
                TypeError,
                "real numeric",
            ),
            (MockGPUArray(np.ones((2, 3, 4)), device=1), ValueError, "GPU 1"),
        )
        wrapper = self.make_wrapper(3)
        for forward in (False, True):
            for data, error, message in cases:
                with self.subTest(forward=forward, message=message):
                    self.backend.counts.clear()
                    self.astra.create_projector.reset_mock()
                    with self.assertRaisesRegex(error, message):
                        if forward:
                            wrapper.forward(data, subset_index=None)
                        else:
                            wrapper.backproject(data, subset_index=None)
                    self.assertFalse(self.backend.counts)
                    self.astra.create_projector.assert_not_called()

    def test_shapes_follow_volume_and_selected_projection_geometry(self):
        for subsets in (1, 2):
            wrapper = self.make_wrapper(3, subsets)
            volume_shape = (2, 2, 2)
            projection_shape = (2, 4 // subsets, 4)

            def geometry_shape(
                geom,
                wrapper=wrapper,
                volume_shape=volume_shape,
                projection_shape=projection_shape,
            ):
                return volume_shape if geom == wrapper.vol_geom else projection_shape

            self.astra.geom_size.side_effect = geometry_shape
            index = None if subsets == 1 else 1
            volume = MockGPUArray(np.ones(volume_shape, dtype=np.float32))
            projections = MockGPUArray(np.ones(projection_shape, dtype=np.float32))
            with self.subTest(subsets=subsets):
                result = wrapper.forward(volume, subset_index=index)
                self.assertEqual(result.shape, projection_shape)
                expected_geom = (
                    wrapper.proj_geom if subsets == 1 else wrapper.proj_geom_OS[index]
                )
                self.assertIs(
                    self.astra.create_projector.call_args.args[1], expected_geom
                )
                result = wrapper.backproject(projections, subset_index=index)
                self.assertEqual(result.shape, volume_shape)
                if subsets != 1:
                    with self.assertRaisesRegex(ValueError, "shape"):
                        wrapper.backproject(
                            MockGPUArray(np.ones((2, 4, 4))), subset_index=index
                        )
            wrapper.close()
            self.astra.geom_size.side_effect = None

    def test_noncurrent_device_is_selected_and_restored_on_success_and_failure(self):
        wrapper = self.tools3d.AstraTools3D(
            projection_geometry=Parallel3DGeometry(
                angles=np.arange(4.0), detector_shape=(2, 4)
            ),
            volume_geometry=Volume3DGeometry(shape=(2, 2, 2)),
            device_index=1,
        )
        self.addCleanup(wrapper.close)
        data = MockGPUArray(np.ones((2, 3, 4), dtype=np.float64), device=1)
        for forward in (False, True):
            for failure in (None, ("ascontiguousarray", 1), ("direct", 1)):
                with self.subTest(forward=forward, failure=failure):
                    self.current_device = 2
                    self.backend.counts.clear()
                    self.backend.fail_at = failure

                    def invoke(forward=forward):
                        if forward:
                            return wrapper.forward(data, subset_index=None)
                        return wrapper.backproject(data, subset_index=None)

                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "injected failure"):
                            invoke()
                    else:
                        self.assertEqual(invoke().device.id, 1)
                    self.assertEqual(self.current_device, 2)
                    self.assertEqual(len(self.backend.live["projector3d"]), 1)
                    self.assert_no_temporaries()
        self.backend.fail_at = None

    def test_cupy_paths_reject_numpy_fallback_explicitly(self):
        wrapper = self.make_wrapper(3)
        with patch.object(self.base, "xp", np):
            with self.assertRaisesRegex(RuntimeError, "CuPy"):
                wrapper.forward(self.array, subset_index=None)
            with self.assertRaisesRegex(RuntimeError, "CuPy"):
                wrapper.backproject(self.array, subset_index=None)

    def test_3d_cor_and_detector_shifts_have_independent_meanings(self):
        angles = np.array([0.0, np.pi / 2])
        cases = (
            (0.0, None, [[0, 0, 0], [0, 0, 0]]),
            (2.5, None, [[2.5, 0, 0], [0, 2.5, 0]]),
            (0.0, [[2, 4], [-3, -5]], [[2, 0, 4], [0, -3, -5]]),
            (1.5, [[2, 4], [-3, -5]], [[3.5, 0, 4], [0, -1.5, -5]]),
        )
        for cor, shifts, expected in cases:
            with self.subTest(cor=cor, shifts=shifts):
                with self.make_wrapper(
                    3, offset=cor, detector_shifts=shifts, angles=angles
                ) as wrapper:
                    vectors = wrapper.proj_geom["Vectors"]
                    np.testing.assert_allclose(vectors[:, 3:6], expected, atol=1e-14)
                    self.assertEqual(
                        wrapper.projection_geometry.centre_of_rotation, cor
                    )
                    np.testing.assert_array_equal(
                        wrapper.projection_geometry.detector_shifts,
                        np.zeros((2, 2)) if shifts is None else shifts,
                    )

    def test_offset_inputs_are_copied_and_2d_vectors_remain_compatible(self):
        offsets = np.array([1.0, -2.0, 0.5, 3.0])
        original = offsets.copy()
        wrapper = self.make_wrapper(
            3, detector_shifts=np.column_stack((offsets, offsets * 0))
        )
        offsets[:] = 99.0
        np.testing.assert_array_equal(
            wrapper.projection_geometry.detector_shifts[:, 0], original
        )
        angles = np.arange(4.0)
        v2 = parallel2d_vectors(
            Parallel2DGeometry(
                angles=angles, detector_shape=(4,), detector_shifts=original
            )
        )
        v3 = parallel3d_vectors(wrapper.projection_geometry)
        np.testing.assert_allclose(v2, v3[:, [0, 1, 3, 4, 6, 7]])
        np.testing.assert_array_equal(v3[:, 5], np.zeros(4))

    def test_full_and_subset_offset_geometry_agree(self):
        angles = np.linspace(0.0, np.pi, 5, endpoint=False)
        offsets = (
            None,
            [[0.75, 0.0]] * 5,
            [[-1.0, 2.0], [0.0, 3.0], [1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )
        for offset in offsets:
            with self.subTest(offset=offset):
                full = self.make_wrapper(
                    3, offset=1.25, detector_shifts=offset, angles=angles
                )
                full_vectors = full.proj_geom["Vectors"]
                self.astra.create_proj_geom.reset_mock()
                subsets = self.make_wrapper(
                    3, subsets=2, offset=1.25, detector_shifts=offset, angles=angles
                )
                self.assertEqual(len(subsets.proj_geom_OS), 2)
                for subset, geometry in subsets.proj_geom_OS.items():
                    np.testing.assert_array_equal(
                        geometry["Vectors"], full_vectors[subset::2]
                    )
                full.close()
                subsets.close()

    def test_3d_converts_once_before_binding_and_slices_full_vectors(self):
        angles = np.array([1.2, 0.0, -0.3, 2.0, 0.7])
        shifts = np.column_stack((np.arange(5.0), -np.arange(5.0)))
        for count in (1, 2, 5):
            self.astra.create_projector.reset_mock()
            with patch.object(
                self.tools3d,
                "parallel3d_vectors",
                wraps=self.tools3d.parallel3d_vectors,
            ) as convert:
                wrapper = self.tools3d.AstraTools3D(
                    projection_geometry=Parallel3DGeometry.tomography(
                        angles=angles, detector_shape=(3, 8), detector_shifts=shifts
                    ),
                    volume_geometry=Volume3DGeometry(shape=(3, 2, 2)),
                    device_index=0,
                    ordsub_number=count,
                )
            self.addCleanup(wrapper.close)
            convert.assert_called_once_with(wrapper.projection_geometry)
            self.assertEqual(wrapper.projection_geometry.detector_shape, (3, 8))
            self.assertEqual(wrapper.volume_geometry.shape, (3, 2, 2))
            np.testing.assert_array_equal(wrapper.projection_geometry.angles, angles)
            np.testing.assert_array_equal(
                wrapper.projection_geometry.detector_shifts, shifts
            )
            if count == 1:
                self.astra.create_projector.assert_not_called()
                self.assertIs(wrapper.proj_geom["Vectors"], wrapper.vectors)
            else:
                self.astra.create_projector.assert_not_called()
                for index, indices in enumerate(wrapper.subset_indices):
                    subset = wrapper.proj_geom_OS[index]
                    self.assertEqual(subset["DetectorColCount"], 8)
                    np.testing.assert_array_equal(
                        subset["Vectors"], wrapper.vectors[indices]
                    )
            wrapper.close()

    def test_2d_converts_vectors_once_and_slices_subset_views(self):
        scan = Parallel2DGeometry(
            angles=[1.2, 0.0, -0.3, 2.0, 0.7],
            detector_shape=(8,),
            centre_of_rotation=0.25,
            detector_shifts=[1, -2, 3, -4, 5],
        )
        grid = Volume2DGeometry(shape=(3, 5))
        for count in (1, 2, 5):
            with patch.object(
                self.tools2d,
                "parallel2d_vectors",
                wraps=self.tools2d.parallel2d_vectors,
            ) as convert:
                with self.tools2d.AstraTools2D(
                    projection_geometry=scan,
                    volume_geometry=grid,
                    ordsub_number=count,
                ) as wrapper:
                    convert.assert_called_once_with(scan)
                    self.assertIs(wrapper.projection_geometry, scan)
                    self.assertIs(wrapper.volume_geometry, grid)
                    if count == 1:
                        self.assertIs(wrapper.proj_geom["Vectors"], wrapper.vectors)
                    else:
                        for index, views in enumerate(wrapper.subset_indices):
                            np.testing.assert_array_equal(
                                wrapper.proj_geom_OS[index]["Vectors"],
                                wrapper.vectors[views],
                            )

    def test_3d_uses_supplied_geometry_with_independent_non_cubic_volume(self):
        scan = Parallel3DGeometry(
            angles=[0.0, 0.5, 1.0],
            detector_shape=(7, 11),
            centre_of_rotation=0.75,
            detector_shifts=[[1, 2], [3, 4], [5, 6]],
        )
        volume = Volume3DGeometry(shape=(3, 5, 9))
        with self.tools3d.AstraTools3D(
            projection_geometry=scan, volume_geometry=volume
        ) as wrapper:
            self.assertIs(wrapper.projection_geometry, scan)
            self.assertIs(wrapper.volume_geometry, volume)
            self.assertEqual(wrapper.vol_geom["GridSliceCount"], 3)
            self.assertEqual(wrapper.vol_geom["GridRowCount"], 5)
            self.assertEqual(wrapper.vol_geom["GridColCount"], 9)
            self.assertEqual(wrapper.proj_geom["DetectorRowCount"], 7)
            self.assertEqual(wrapper.proj_geom["DetectorColCount"], 11)
            for field in (
                "_angles_vec",
                "_centre_of_rotation",
                "_recon_size",
                "_detectors_x",
                "_detectors_y",
                "_detectors_x_pad",
            ):
                self.assertNotIn(field, vars(wrapper))
            with self.assertRaises(AttributeError):
                wrapper.projection_geometry = scan
            with self.assertRaises(AttributeError):
                wrapper.volume_geometry = volume

    def test_3d_requires_geometry_objects_and_rejects_old_constructor(self):
        scan = Parallel3DGeometry(angles=[0.0], detector_shape=(2, 4))
        volume = Volume3DGeometry(shape=(2, 2, 2))
        self.astra.create_projector.reset_mock()
        self.astra.set_gpu_index.reset_mock()
        for projection, reconstruction, message in (
            (None, volume, "projection_geometry"),
            (scan, (2, 2, 2), "volume_geometry"),
        ):
            with self.assertRaisesRegex(TypeError, message):
                self.tools3d.AstraTools3D(
                    projection_geometry=projection, volume_geometry=reconstruction
                )
        with self.assertRaisesRegex(TypeError, "detectors_x"):
            self.tools3d.AstraTools3D(
                projection_geometry=scan, volume_geometry=volume, detectors_x=4
            )
        self.astra.create_projector.assert_not_called()
        self.astra.set_gpu_index.assert_not_called()

    def test_invalid_offsets_are_rejected_before_astra_allocation(self):
        invalid = (
            ([1.0, 2.0, 3.0], ValueError),
            (np.zeros((3, 2)), ValueError),
            (np.zeros((4, 1)), ValueError),
            (np.zeros((4, 3)), ValueError),
            (np.zeros((4, 2, 1)), ValueError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            ([0.0, 1.0, np.nan, 3.0], ValueError),
            ([[0.0, 0.0], [0.0, np.inf], [0.0, 0.0], [0.0, 0.0]], ValueError),
            (1 + 2j, TypeError),
            (["0", "1", "2", "3"], TypeError),
        )
        for offset, error in invalid:
            with self.subTest(offset=offset):
                self.astra.create_projector.reset_mock()
                self.astra.create_proj_geom.reset_mock()
                with self.assertRaises(error):
                    self.make_wrapper(3, offset=offset)
                self.astra.create_projector.assert_not_called()
                self.astra.create_proj_geom.assert_not_called()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.make_wrapper(2, detector_shifts=np.zeros((4, 2)))

    def test_2d_gpu_accepts_detector_shifts_without_architecture_selection(self):
        with self.make_wrapper(2, detector_shifts=[0.0, 0.0, 1.0, 0.0]) as wrapper:
            self.assertFalse(hasattr(wrapper, "processing_arch"))
            np.testing.assert_array_equal(
                wrapper.projection_geometry.detector_shifts, [0, 0, 1, 0]
            )

    def test_gpu_is_selected_per_cached_handle_and_device_is_immutable(self):
        wrapper = self.tools3d.AstraTools3D(
            projection_geometry=Parallel3DGeometry(angles=[0.0], detector_shape=(2, 4)),
            volume_geometry=Volume3DGeometry(shape=(2, 2, 2)),
            device_index=2,
        )
        self.addCleanup(wrapper.close)
        self.astra.set_gpu_index.assert_not_called()
        with self.assertRaises(AttributeError):
            wrapper.device_index = 1
        data = MockGPUArray(np.ones((2, 3, 4), dtype=np.float32), device=2)
        wrapper.forward(data)
        self.assertEqual(
            self.astra.create_projector.call_args.kwargs["options"], {"GPUindex": 2}
        )
        self.assertEqual(self.current_device, 0)

    def test_subset_indices_cover_every_view_once_without_padding(self):
        for n_views in range(1, 10):
            for count in range(1, n_views + 1):
                with self.subTest(n_views=n_views, count=count):
                    wrapper = self.make_wrapper(
                        3, subsets=count, angles=np.arange(float(n_views))
                    )
                    indices = wrapper.subset_indices
                    self.assertIsInstance(indices, tuple)
                    self.assertEqual(len(indices), count)
                    self.assertTrue(all(len(part) > 0 for part in indices))
                    self.assertTrue(all(not part.flags.writeable for part in indices))
                    all_views = np.concatenate(indices)
                    np.testing.assert_array_equal(
                        np.sort(all_views), np.arange(n_views)
                    )
                    self.assertEqual(np.count_nonzero(all_views == 0), 1)
                    self.assertLessEqual(
                        max(map(len, indices)) - min(map(len, indices)), 1
                    )
                    self.assertIs(wrapper.newInd_Vec, indices)
                    self.assertEqual(wrapper.NumbProjBins, max(map(len, indices)))
                    wrapper.close()

    def test_one_view_per_subset_preserves_zero_and_offset_geometry(self):
        angles = np.arange(4.0)
        for dim in (2, 3):
            offset = (
                np.arange(4.0)
                if dim == 2
                else np.column_stack((np.arange(4.0), np.arange(4.0) + 10))
            )
            self.astra.create_proj_geom.reset_mock()
            wrapper = self.make_wrapper(
                dim, subsets=4, detector_shifts=offset, angles=angles
            )
            expected = (
                parallel2d_vectors(wrapper.projection_geometry)
                if dim == 2
                else parallel3d_vectors(wrapper.projection_geometry)
            )
            actual = [geometry["Vectors"] for geometry in wrapper.proj_geom_OS.values()]
            self.assertEqual(len(actual), 4)
            for index, vectors in enumerate(actual):
                np.testing.assert_allclose(vectors, expected[index : index + 1])
            wrapper.close()

    def test_subset_count_validation(self):
        for count, error in (
            (0, ValueError),
            (-1, ValueError),
            (5, ValueError),
            (2.0, TypeError),
            (True, TypeError),
            (np.bool_(False), TypeError),
            ("2", TypeError),
        ):
            with self.subTest(count=count):
                self.astra.create_projector.reset_mock()
                with self.assertRaises(error):
                    self.make_wrapper(3, subsets=count)
                self.astra.create_projector.assert_not_called()
        for count, expected in ((None, 1), (1, 1), (np.int64(2), 2)):
            wrapper = self.make_wrapper(3, subsets=count)
            self.assertEqual(wrapper.ordsub_number, expected)
            self.assertEqual(len(wrapper.subset_indices), expected)
            wrapper.close()

    def test_fista_admm_subset_data_and_weights_stay_aligned(self):
        module = self.load_iterative_module()
        for method in ("FISTA", "ADMM"):
            for n_views, count in ((1, 1), (4, 4), (5, 2), (7, 3)):
                for fidelity in ("LS", "PWLS", "SWLS"):
                    with self.subTest(
                        method=method, n_views=n_views, count=count, fidelity=fidelity
                    ):
                        rec = self.make_iterative(module, n_views, count)
                        data = np.arange(1, 1 + n_views * 2, dtype=np.float32).reshape(
                            1, n_views, 2
                        )
                        weights = None if fidelity == "LS" else data + 100
                        self.gradient_mock.reset_mock()
                        getattr(rec, method)(
                            data,
                            weights=weights,
                            fidelity=module.DataFidelity(fidelity),
                            iterations=1,
                            lipschitz_constant=1.0,
                        )
                        if weights is not None:
                            weights = weights / weights.max()
                        self.assertEqual(
                            self.gradient_mock.call_count,
                            2 if method == "ADMM" else count,
                        )
                        for subset, call in enumerate(
                            self.gradient_mock.call_args_list
                        ):
                            selected = (
                                np.arange(n_views)
                                if method == "ADMM"
                                else np.arange(subset, n_views, count)
                            )
                            np.testing.assert_array_equal(
                                call.args[2], data[:, selected, :]
                            )
                            self.assertEqual(
                                call.args[4], 0 if method == "ADMM" else subset
                            )
                            self.assertEqual(
                                call.args[3], method != "ADMM" and count > 1
                            )
                            if weights is None:
                                self.assertIsNone(call.args[5])
                            else:
                                np.testing.assert_array_equal(
                                    call.args[5], weights[:, selected, :]
                                )
                            if fidelity == "SWLS":
                                np.testing.assert_allclose(
                                    call.args[6],
                                    weights[:, selected, :].sum(axis=1) + 0.1,
                                )
                        rec.close()

    def test_osem_uses_every_subset_including_view_zero(self):
        module = self.load_iterative_module()
        for n_views, count in ((1, 1), (4, 4), (5, 2)):
            rec = self.make_iterative(module, n_views, count)
            data = np.arange(1, 1 + n_views * 2, dtype=np.float32).reshape(
                1, n_views, 2
            )
            rec._operator = SimpleNamespace(backend="astra")
            rec.operator.forward = Mock(
                side_effect=lambda x, *, subset_index=None, rec=rec, **kwargs: np.ones(
                    (
                        1,
                        len(
                            rec.Atools.subset_indices[
                                0 if subset_index is None else subset_index
                            ]
                        ),
                        2,
                    )
                )
            )
            rec.operator.backproject = Mock(
                side_effect=lambda *args, **kwargs: np.ones((1, 2, 2))
            )
            rec.OSEM(data, iterations=1)
            self.assertEqual(rec.operator.forward.call_count, count)
            self.assertEqual(rec.operator.backproject.call_count, 2 * count)
            for subset, call in enumerate(rec.operator.backproject.call_args_list[::2]):
                expected = np.ones_like(data[:, subset::count, :])
                np.testing.assert_array_equal(call.args[0], expected)
                self.assertEqual(
                    call.kwargs["subset_index"], subset if count > 1 else None
                )
            for subset, call in enumerate(
                rec.operator.backproject.call_args_list[1::2]
            ):
                np.testing.assert_array_equal(call.args[0], data[:, subset::count, :])
                self.assertEqual(
                    call.kwargs["subset_index"], subset if count > 1 else None
                )
            rec.close()

    def test_power_method_does_not_index_subset_weights_with_global_indices(self):
        module = self.load_iterative_module()
        for n_views, count in ((5, 2), (4, 4)):
            rec = self.make_iterative(module, n_views, count)
            rec.Atools.forward = Mock(
                side_effect=lambda x, subset_index, rec=rec: np.ones(
                    (
                        1,
                        len(
                            rec.Atools.subset_indices[
                                0 if subset_index is None else subset_index
                            ]
                        ),
                        2,
                    )
                )
            )
            rec.Atools.backproject = Mock(return_value=np.ones((1, 2, 2)))
            result = rec.powermethod()
            self.assertTrue(np.isfinite(result))
            self.assertEqual(rec.Atools.forward.call_count, 16 * count)
            self.assertEqual(rec.Atools.backproject.call_count, 15 * count)
            for call in rec.Atools.backproject.call_args_list:
                subset = call.kwargs["subset_index"]
                expected = (1, len(rec.Atools.subset_indices[subset]), 2)
                self.assertEqual(call.args[0].shape, expected)
            self.assertEqual(
                {
                    call.kwargs["subset_index"]
                    for call in rec.Atools.backproject.call_args_list
                },
                set(range(count)),
            )
            rec.close()

    def test_iterative_constructor_validates_subset_count(self):
        module = self.load_iterative_module()
        for count, error in (
            (0, ValueError),
            (5, ValueError),
            (True, TypeError),
            (2.0, TypeError),
        ):
            with self.assertRaises(error):
                self.make_iterative(module, 4, count)

    def test_iterative_constructor_keeps_detector_shifts_and_cor_separate(self):
        module = self.load_iterative_module()
        shifts = np.array([[1.0, 2.0], [-3.0, 4.0]])
        rec = module.IterativeReconstructor(
            projection_geometry=Parallel3DGeometry(
                angles=[0.0, 0.5],
                detector_shape=(3, 5),
                centre_of_rotation=0.75,
                detector_shifts=shifts,
            ),
            volume_geometry=Volume3DGeometry(shape=(3, 5, 5)),
        )
        self.addCleanup(rec.close)
        scan = rec.Atools.projection_geometry
        self.assertEqual(scan.centre_of_rotation, 0.75)
        np.testing.assert_array_equal(scan.detector_shifts, shifts)

    def test_iterative_constructor_builds_padded_geometry(self):
        module = self.load_iterative_module()
        rec = module.IterativeReconstructor(
            projection_geometry=Parallel3DGeometry(
                angles=[0.0, 0.5], detector_shape=(3, 5), centre_of_rotation=0.75
            ),
            volume_geometry=Volume3DGeometry(shape=(2, 4, 6)),
            detector_padding=2,
        )
        self.addCleanup(rec.close)
        self.assertEqual(rec.detector_padding, 2)
        self.assertEqual(rec.projection_geometry.detector_shape, (3, 5))
        self.assertEqual(rec.Atools.projection_geometry.detector_shape, (3, 9))
        self.assertEqual(rec.Atools.volume_geometry.shape, (2, 4, 6))

    def test_3d_rejects_array_cor_and_invalid_detector_shifts(self):
        for cor in (None, [0.0] * 4, np.zeros((4, 2))):
            with self.subTest(cor=cor):
                self.astra.create_projector.reset_mock()
                with self.assertRaisesRegex(
                    (ValueError, TypeError), "centre_of_rotation"
                ):
                    self.make_wrapper(3, offset=cor)
                self.astra.create_projector.assert_not_called()
        for shifts in (1.0, np.zeros(4), np.zeros((3, 2)), np.full((4, 2), np.nan)):
            with self.subTest(shifts=shifts):
                self.astra.create_projector.reset_mock()
                with self.assertRaisesRegex(ValueError, "detector_shifts"):
                    self.make_wrapper(3, detector_shifts=shifts)
                self.astra.create_projector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
