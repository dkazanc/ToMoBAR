"""ASTRA ownership regression tests; no ASTRA, CuPy or CUDA installation needed.

Run with pytest or ``python -m unittest discover -s tests
-p test_astra_resource_lifecycle.py``. Real wrapper modules are loaded with
isolated dependency mocks, without replacing modules used by other tests.
"""

import gc
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)
from tomobar.astra_wrappers.geometry import parallel2d_vectors, parallel3d_vectors

ROOT = Path(__file__).resolve().parents[1] / "tomobar"


class MockGPUArray:
    """NumPy-backed storage for testing GPU metadata and packing without CUDA."""

    def __init__(self, storage, device=0):
        self.storage = storage
        self.device = SimpleNamespace(id=device)
        self.shape = storage.shape
        self.ndim = storage.ndim
        self.dtype = storage.dtype
        self.itemsize = storage.itemsize
        self.flags = storage.flags
        self.data = SimpleNamespace(ptr=storage.ctypes.data)


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockAstra:
    """Strict ID registries catch wrong-manager and double deletion."""

    def __init__(self):
        self.live = {
            name: set()
            for name in ("data2d", "data3d", "projector", "projector3d", "algorithm")
        }
        self.next_id = 1
        self.counts = {}
        self.fail_at = None
        self.configs = {}
        self.deleted = []
        self.result = np.ones((2, 2), dtype=np.float32)
        self.module = ModuleType("astra")
        for name in self.live:
            manager = SimpleNamespace(
                create=Mock(
                    side_effect=lambda *a, _n=name: self.allocate(_n, "create", *a)
                ),
                link=Mock(
                    side_effect=lambda *a, _n=name: self.allocate(_n, "link", *a)
                ),
                delete=Mock(side_effect=lambda ident, _n=name: self.delete(_n, ident)),
                get=Mock(side_effect=lambda ident, _n=name: self.get(_n, ident)),
            )
            setattr(self.module, name, manager)
        self.module.algorithm.run = Mock(side_effect=self.run)
        self.module.set_gpu_index = Mock()
        self.module.astra_dict = Mock(side_effect=lambda method: {"type": method})
        self.module.create_vol_geom = Mock(return_value="volume")
        self.module.create_proj_geom = Mock(side_effect=lambda *a: object())
        self.module.geom_size = Mock(return_value=(2, 3, 4))
        self.module.create_projector = Mock(side_effect=self.create_projector)

    def hit(self, event):
        self.counts[event] = self.counts.get(event, 0) + 1
        if self.fail_at == (event, self.counts[event]):
            raise RuntimeError("injected failure")

    def allocate(self, manager, operation, *args):
        self.hit(f"{manager}.{operation}")
        ident = self.next_id
        self.next_id += 1
        self.live[manager].add(ident)
        if manager == "algorithm":
            cfg = args[0]
            if "ProjectorId" in cfg:
                assert cfg["ProjectorId"] in self.live["projector"]
            self.configs[ident] = cfg
        return ident

    def create_projector(self, kind, *args, **kwargs):
        return self.allocate(
            "projector3d" if kind == "cuda3d" else "projector", "create"
        )

    def delete(self, manager, ident):
        assert ident in self.live[manager], (
            "wrong manager or double deletion",
            manager,
            ident,
        )
        if manager != "algorithm":
            for alg in self.live["algorithm"]:
                cfg = self.configs[alg]
                for key in (
                    "ProjectorId",
                    "VolumeDataId",
                    "ProjectionDataId",
                    "ReconstructionDataId",
                ):
                    assert (
                        cfg.get(key) != ident
                    ), "delete algorithms before their dependencies"
        self.live[manager].remove(ident)
        self.deleted.append((manager, ident))

    def get(self, manager, ident):
        self.hit(f"{manager}.get")
        assert ident in self.live[manager]
        return self.result

    def run(self, ident, *args):
        self.hit("algorithm.run")
        assert ident in self.live["algorithm"]


class TestAstraResourceLifecycle(unittest.TestCase):
    def setUp(self):
        self.backend = MockAstra()
        self.astra = self.backend.module
        self.array = MockGPUArray(np.ones((2, 3, 4), dtype=np.float32))
        self.output = None
        self.current_device = 0
        cupy = ModuleType("cupy")
        cupy.ndarray = MockGPUArray
        cupy.float32 = np.float32
        owner = self

        class Device:
            compute_capability = "80"

            def __init__(self, index):
                self.index = index

            def __enter__(self):
                self.previous = owner.current_device
                owner.current_device = self.index
                return self

            def __exit__(self, *exc):
                owner.current_device = self.previous

        cupy.cuda = SimpleNamespace(Device=Mock(side_effect=Device))
        cupy.ascontiguousarray = Mock(side_effect=self.ascontiguousarray)
        cupy.empty = Mock(side_effect=self.empty)
        self.cupy = cupy
        self.astra.projector.direct_FP = Mock(side_effect=self.direct)
        self.astra.projector.direct_BP = Mock(side_effect=self.direct)
        self.astra.projector3d.direct_FP = Mock(side_effect=self.direct)
        self.astra.projector3d.direct_BP = Mock(side_effect=self.direct)
        # Restore sys.modules immediately after importing isolated module objects.
        with patch.dict(
            "sys.modules",
            {
                "astra": self.astra,
                "cupy": cupy,
            },
        ):
            axes = load_module("_resource_test_axes", "supp/axis_order.py")
            self.axes = axes
            with patch.dict("sys.modules", {"tomobar.supp.axis_order": axes}):
                self.base = load_module(
                    "_resource_test_base", "astra_wrappers/astra_base.py"
                )
                with patch.dict(
                    "sys.modules", {"tomobar.astra_wrappers.astra_base": self.base}
                ):
                    self.tools2d = load_module(
                        "_resource_test_2d", "astra_wrappers/astra_tools2d.py"
                    )
                    self.tools3d = load_module(
                        "_resource_test_3d", "astra_wrappers/astra_tools3d.py"
                    )

    def ascontiguousarray(self, data, dtype):
        self.backend.hit("ascontiguousarray")
        self.assertEqual(self.current_device, data.device.id)
        packed = np.ascontiguousarray(data.storage, dtype=dtype)
        result = (
            data if packed is data.storage else MockGPUArray(packed, data.device.id)
        )
        self.prepared = weakref.ref(result)
        return result

    def empty(self, shape, dtype):
        self.backend.hit("empty")
        self.output = MockGPUArray(np.empty(shape, dtype=dtype), self.current_device)
        return self.output

    def direct(self, ident, data, *, out):
        self.backend.hit("direct")
        self.assertIn(
            ident, self.backend.live["projector3d"] | self.backend.live["projector"]
        )
        if isinstance(data, np.ndarray):
            self.assertTrue(data.flags.c_contiguous)
            self.assertEqual(data.dtype, np.float32)
            out[:] = 1.0
            return
        prepared = self.prepared()
        self.assertIsNotNone(
            prepared, "normalized storage must survive the direct call"
        )
        self.assertTrue(prepared.flags.c_contiguous)
        self.assertEqual(prepared.dtype, np.float32)
        self.projected_input = prepared.storage.copy()
        self.assertEqual(prepared.device.id, self.current_device)
        self.assertEqual(self.output.device.id, self.current_device)
        self.assertIs(data, prepared)
        self.assertIs(out, self.output)
        self.assertEqual(
            self.astra.create_projector.call_args.kwargs["options"],
            {"GPUindex": self.current_device},
        )

    def make_wrapper(
        self,
        dim=2,
        subsets=1,
        register_cleanup=True,
        offset=0.0,
        angles=None,
        detector_shifts=None,
    ):
        common = dict(
            detectors_x=4,
            detectors_x_pad=0,
            angles_vec=np.arange(4.0) if angles is None else angles,
            centre_of_rotation=offset,
            recon_size=2,
            device_index=0,
            ordsub_number=subsets,
        )
        if dim == 2:
            wrapper = self.tools2d.AstraTools2D(
                projection_geometry=Parallel2DGeometry(
                    angles=common["angles_vec"],
                    detector_shape=(4,),
                    centre_of_rotation=offset,
                    detector_shifts=detector_shifts,
                ),
                volume_geometry=Volume2DGeometry(shape=(2, 2)),
                device_index=0,
                ordsub_number=subsets,
            )
        else:
            wrapper = self.tools3d.AstraTools3D(
                projection_geometry=Parallel3DGeometry.tomography(
                    angles=common["angles_vec"],
                    detector_shape=(2, 4),
                    centre_of_rotation=offset,
                    detector_shifts=detector_shifts,
                ),
                volume_geometry=Volume3DGeometry(shape=(2, 2, 2)),
                device_index=0,
                ordsub_number=subsets,
            )
        if register_cleanup:
            self.addCleanup(wrapper.close)
        return wrapper

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
            self.astra.geom_size.side_effect = lambda geom: (
                volume_shape if geom == wrapper.vol_geom else projection_shape
            )
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
                    persistent = set(self.backend.live["projector3d"])

                    def invoke():
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

    def load_iterative_module(self):
        """Run actual iterative control flow with NumPy and mocked projectors."""
        cupy = ModuleType("cupy")
        cupy.__getattr__ = lambda name: getattr(np, name)
        cupy.random = SimpleNamespace(
            randn=lambda *shape, dtype: np.ones(shape, dtype=dtype)
        )
        from contextlib import nullcontext

        cupy.cuda = SimpleNamespace(
            Device=Mock(side_effect=lambda index: nullcontext())
        )
        cupy._default_memory_pool = SimpleNamespace(free_all_blocks=Mock())
        supp = ModuleType("suppTools")
        supp.check_kwargs = lambda data, **kw: data
        supp.perform_recon_crop = lambda data, *args: data
        regularisers = ModuleType("regularisers")
        regularisers.prox_regul = Mock()
        fidelities = ModuleType("fidelities")
        fidelities.grad_data_term = Mock(
            side_effect=lambda rec, x, *args, **kwargs: np.zeros_like(x)
        )
        with patch.dict(
            "sys.modules",
            {
                "cupy": cupy,
                "astra": self.astra,
                "tomobar.supp.axis_order": self.axes,
                "tomobar.supp.suppTools": supp,
                "tomobar.regularisers": regularisers,
                "tomobar.astra_wrappers.astra_tools3d": self.tools3d,
                "tomobar.data_fidelities": fidelities,
            },
        ):
            return load_module("_subset_test_ir", "methodsIR.py")

    def make_iterative(self, module, n_views, count):
        rec = module.RecToolsIR(
            projection_geometry=Parallel3DGeometry(
                angles=np.arange(float(n_views)), detector_shape=(1, 2)
            ),
            volume_geometry=Volume3DGeometry(shape=(1, 2, 2)),
            OS_number=count,
        )
        self.addCleanup(rec.close)
        return rec

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
                        module.grad_data_term.reset_mock()
                        getattr(rec, method)(
                            data,
                            weights=weights,
                            fidelity=module.DataFidelity(fidelity),
                            iterations=1,
                            lipschitz_constant=1.0,
                        )
                        if weights is not None:
                            weights = weights / weights.max()
                        self.assertEqual(module.grad_data_term.call_count, count)
                        for subset, call in enumerate(
                            module.grad_data_term.call_args_list
                        ):
                            selected = np.arange(subset, n_views, count)
                            np.testing.assert_array_equal(
                                call.args[2], data[:, selected, :]
                            )
                            self.assertEqual(call.args[4], subset)
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
            rec._Ax = Mock(
                side_effect=lambda x, subset, use_os: np.ones(
                    (1, len(rec.Atools.subset_indices[subset]), 2)
                )
            )
            rec._Atb = Mock(side_effect=lambda *args: np.ones((1, 2, 2)))
            rec.OSEM(data, iterations=1)
            self.assertEqual(rec._Ax.call_count, count)
            self.assertEqual(rec._Atb.call_count, count + 1)
            for subset, call in enumerate(rec._Atb.call_args_list[1:]):
                np.testing.assert_array_equal(call.args[0], data[:, subset::count, :])
                self.assertEqual(call.args[1], subset)
            rec.close()

    def test_power_method_does_not_index_subset_weights_with_global_indices(self):
        module = self.load_iterative_module()
        for n_views, count in ((5, 2), (4, 4)):
            rec = self.make_iterative(module, n_views, count)
            subset_shape = (1, len(rec.Atools.subset_indices[0]), 2)
            rec.Atools.forward = Mock(return_value=np.ones(subset_shape))
            rec.Atools.backproject = Mock(return_value=np.ones((1, 2, 2)))
            result = rec.powermethod()
            self.assertTrue(np.isfinite(result))
            self.assertEqual(rec.Atools.forward.call_count, 16)
            self.assertEqual(rec.Atools.backproject.call_count, 15)
            for call in rec.Atools.backproject.call_args_list:
                self.assertEqual(call.args[0].shape, subset_shape)
                self.assertEqual(call.kwargs["subset_index"], 0)
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
        rec = module.RecToolsIR(
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
        rec = module.RecToolsIR(
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
