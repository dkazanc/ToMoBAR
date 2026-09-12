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
            funcs = load_module("_resource_test_funcs", "supp/funcs.py")
            self.funcs = funcs
            with patch.dict("sys.modules", {"tomobar.supp.funcs": funcs}):
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
        self.assertIn(ident, self.backend.live["projector3d"])
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
        arch="cpu",
        register_cleanup=True,
        offset=0.0,
        angles=None,
    ):
        common = dict(
            detectors_x=4,
            detectors_x_pad=0,
            angles_vec=np.arange(4.0) if angles is None else angles,
            centre_of_rotation=offset,
            recon_size=2,
            processing_arch=arch,
            device_index=0,
            ordsub_number=subsets,
        )
        if dim == 2:
            wrapper = self.tools2d.AstraTools2D(**common)
        else:
            wrapper = self.tools3d.AstraTools3D(detectors_y=2, **common)
        if register_cleanup:
            self.addCleanup(wrapper.close)
        return wrapper

    def assert_no_temporaries(self):
        for manager in ("data2d", "data3d", "algorithm"):
            self.assertFalse(self.backend.live[manager], manager)

    def test_2d_projectors_survive_repeated_calls_and_close_once(self):
        for arch in ("cpu", "gpu"):
            for subsets in (1, 2):
                with self.subTest(arch=arch, subsets=subsets):
                    wrapper = self.make_wrapper(subsets=subsets, arch=arch)
                    owned = set(self.backend.live["projector"])
                    indices = [None] if subsets == 1 else range(subsets)
                    for _ in range(2):
                        for index in indices:
                            self.assertIs(
                                wrapper._runAstraProj2D(np.ones((2, 2)), index, "FP"),
                                self.backend.result,
                            )
                            self.assertIs(
                                wrapper._runAstraBackproj2D(
                                    np.ones((4, 4)), "BP", 1, index
                                ),
                                self.backend.result,
                            )
                            self.assertEqual(self.backend.live["projector"], owned)
                            self.assert_no_temporaries()
                    wrapper.close()
                    wrapper.close()
                    self.assertFalse(self.backend.live["projector"])
                    for ident in owned:
                        self.assertEqual(
                            self.backend.deleted.count(("projector", ident)), 1
                        )

    def test_3d_direct_full_and_subsets_release_only_temporary_projectors(self):
        for subsets in (1, 2):
            with self.subTest(subsets=subsets):
                wrapper = self.make_wrapper(3, subsets, "gpu")
                persistent = set(self.backend.live["projector3d"])
                indices = [None] if subsets == 1 else range(subsets)
                for _ in range(2):
                    for index in indices:
                        self.assertIs(
                            wrapper.runAstraProj3DCuPy(self.array, index), self.output
                        )
                        self.assertIs(
                            wrapper.runAstraBackproj3DCuPy(
                                self.array, "BP3D_CUDA", index
                            ),
                            self.output,
                        )
                        self.assertEqual(self.backend.live["projector3d"], persistent)
                        self.assert_no_temporaries()
                self.astra.data3d.link.assert_not_called()
                wrapper.close()
                self.assertFalse(self.backend.live["projector3d"])

    def test_numpy_paths_cleanup_on_success_and_failures(self):
        for dim in (2, 3):
            for forward in (False, True):
                for subsets in (1, 2):
                    wrapper = self.make_wrapper(
                        dim, subsets, "cpu" if dim == 2 else "gpu"
                    )
                    manager = f"data{dim}d"
                    first = f"{manager}.link" if forward else f"{manager}.create"
                    second_count = 1 if forward else 2
                    for failure in (
                        None,
                        (first, 1),
                        (f"{manager}.create", second_count),
                        ("algorithm.create", 1),
                        ("algorithm.run", 1),
                        (f"{manager}.get", 1),
                    ):
                        with self.subTest(
                            dim=dim, forward=forward, subsets=subsets, failure=failure
                        ):
                            self.backend.counts.clear()
                            self.backend.fail_at = failure
                            index = None if subsets == 1 else 1
                            data = np.ones((2,) * dim)

                            def invoke():
                                if dim == 2:
                                    return (
                                        wrapper._runAstraProj2D(data, index, "FP")
                                        if forward
                                        else wrapper._runAstraBackproj2D(
                                            data, "BP", 1, index
                                        )
                                    )
                                return (
                                    wrapper.runAstraProj3D(data, index)
                                    if forward
                                    else wrapper.runAstraBackproj3D(
                                        data, "BP3D_CUDA", 1, index
                                    )
                                )

                            if failure:
                                with self.assertRaisesRegex(
                                    RuntimeError, "injected failure"
                                ):
                                    invoke()
                            else:
                                self.assertIs(invoke(), self.backend.result)
                            self.assert_no_temporaries()
                    self.backend.fail_at = None
                    wrapper.close()

    def test_caller_owned_data_ids_are_preserved(self):
        for dim in (2, 3):
            wrapper = self.make_wrapper(dim, arch="cpu" if dim == 2 else "gpu")
            manager = f"data{dim}d"
            borrowed = self.backend.allocate(manager, "create")
            for failure in (None, ("algorithm.run", 1)):
                with self.subTest(dim=dim, failure=failure):
                    self.backend.counts.clear()
                    self.backend.fail_at = failure

                    def invoke():
                        return (
                            wrapper._runAstraProj2D(borrowed, None, "FP")
                            if dim == 2
                            else wrapper.runAstraProj3D(borrowed, None)
                        )

                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "injected failure"):
                            invoke()
                    else:
                        invoke()
                    self.assertEqual(self.backend.live[manager], {borrowed})
                    self.assertFalse(self.backend.live["algorithm"])
            self.backend.fail_at = None
            self.backend.delete(manager, borrowed)
            wrapper.close()

    def test_direct_paths_cleanup_at_every_allocation_and_execution_stage(self):
        for forward in (False, True):
            for subsets in (1, 2):
                wrapper = self.make_wrapper(3, subsets, "gpu")
                persistent = set(self.backend.live["projector3d"])
                for failure in (
                    ("empty", 1),
                    ("projector3d.create", 1),
                    ("direct", 1),
                ):
                    with self.subTest(
                        forward=forward, subsets=subsets, failure=failure
                    ):
                        self.backend.counts.clear()
                        self.backend.fail_at = failure
                        index = None if subsets == 1 else 1
                        with self.assertRaisesRegex(RuntimeError, "injected failure"):
                            if forward:
                                wrapper.runAstraProj3DCuPy(self.array, index)
                            else:
                                wrapper.runAstraBackproj3DCuPy(
                                    self.array, "BP3D_CUDA", index
                                )
                        self.assertEqual(self.backend.live["projector3d"], persistent)
                        self.assert_no_temporaries()
                self.backend.fail_at = None
                wrapper.close()

    def test_context_manager_closes_after_exception_and_rejects_reuse(self):
        wrapper = self.make_wrapper()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with wrapper as entered:
                self.assertIs(entered, wrapper)
                raise RuntimeError("body failed")
        self.assertFalse(self.backend.live["projector"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            wrapper._forwproj(np.ones((2, 2)))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            wrapper.__enter__()

    def test_finalizer_releases_projectors_without_retaining_wrapper(self):
        for dim in (2, 3):
            wrapper = self.make_wrapper(
                dim, arch="cpu" if dim == 2 else "gpu", register_cleanup=False
            )
            ref = weakref.ref(wrapper)
            del wrapper
            gc.collect()
            self.assertIsNone(ref())
            self.assertFalse(
                self.backend.live["projector" if dim == 2 else "projector3d"]
            )

    def test_partial_subset_constructor_failure_releases_created_projectors(self):
        self.backend.fail_at = ("projector.create", 2)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.make_wrapper(subsets=2)
        # Mock call history retains exception tracebacks and thus the constructor.
        self.astra.create_projector.reset_mock()
        gc.collect()
        self.assertFalse(self.backend.live["projector"])

    def test_real_inputs_are_packed_without_modifying_the_original(self):
        for forward in (False, True):
            wrapper = self.make_wrapper(3, arch="gpu")
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
                            result = wrapper.runAstraProj3DCuPy(data, None)
                        else:
                            result = wrapper.runAstraBackproj3DCuPy(
                                data, "BP3D_CUDA", None
                            )
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
            (np.ones((2, 3, 4), dtype=np.float32), TypeError, "CuPy array"),
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
        wrapper = self.make_wrapper(3, arch="gpu")
        for forward in (False, True):
            for data, error, message in cases:
                with self.subTest(forward=forward, message=message):
                    self.backend.counts.clear()
                    self.astra.create_projector.reset_mock()
                    with self.assertRaisesRegex(error, message):
                        if forward:
                            wrapper.runAstraProj3DCuPy(data, None)
                        else:
                            wrapper.runAstraBackproj3DCuPy(data, "BP3D_CUDA", None)
                    self.assertFalse(self.backend.counts)
                    self.astra.create_projector.assert_not_called()

    def test_shapes_follow_volume_and_selected_projection_geometry(self):
        for subsets in (1, 2):
            wrapper = self.make_wrapper(3, subsets, "gpu")
            volume_shape = (2, 2, 2)
            projection_shape = (2, 4 // subsets, 4)
            self.astra.geom_size.side_effect = lambda geom: (
                volume_shape if geom == wrapper.vol_geom else projection_shape
            )
            index = None if subsets == 1 else 1
            volume = MockGPUArray(np.ones(volume_shape, dtype=np.float32))
            projections = MockGPUArray(np.ones(projection_shape, dtype=np.float32))
            with self.subTest(subsets=subsets):
                result = wrapper.runAstraProj3DCuPy(volume, index)
                self.assertEqual(result.shape, projection_shape)
                expected_geom = (
                    wrapper.proj_geom if subsets == 1 else wrapper.proj_geom_OS[index]
                )
                self.assertIs(
                    self.astra.create_projector.call_args.args[1], expected_geom
                )
                result = wrapper.runAstraBackproj3DCuPy(projections, "BP3D_CUDA", index)
                self.assertEqual(result.shape, volume_shape)
                if subsets != 1:
                    with self.assertRaisesRegex(ValueError, "shape"):
                        wrapper.runAstraBackproj3DCuPy(
                            MockGPUArray(np.ones((2, 4, 4))), "BP3D_CUDA", index
                        )
            wrapper.close()
            self.astra.geom_size.side_effect = None

    def test_noncurrent_device_is_selected_and_restored_on_success_and_failure(self):
        wrapper = self.make_wrapper(3, arch="gpu")
        wrapper.device_index = 1
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
                            return wrapper.runAstraProj3DCuPy(data, None)
                        return wrapper.runAstraBackproj3DCuPy(data, "BP3D_CUDA", None)

                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "injected failure"):
                            invoke()
                    else:
                        self.assertEqual(invoke().device.id, 1)
                    self.assertEqual(self.current_device, 2)
                    self.assertEqual(self.backend.live["projector3d"], persistent)
                    self.assert_no_temporaries()
        self.backend.fail_at = None

    def test_cupy_paths_reject_numpy_fallback_explicitly(self):
        wrapper = self.make_wrapper(3, arch="gpu")
        with patch.object(self.base, "xp", np):
            with self.assertRaisesRegex(RuntimeError, "CuPy"):
                wrapper.runAstraProj3DCuPy(self.array, None)
            with self.assertRaisesRegex(RuntimeError, "CuPy"):
                wrapper.runAstraBackproj3DCuPy(self.array, "BP3D_CUDA", None)

    def test_3d_offset_representations_and_signs(self):
        angles = np.array([0.0, np.pi / 2])
        cases = (
            (None, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            (2.5, [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]),
            (np.array(2.5), [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]),
            ([2.0, -3.0], [[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]]),
            ([[2.0, 4.0], [-3.0, -5.0]], [[2.0, 0.0, 4.0], [0.0, -3.0, -5.0]]),
        )
        for offset, expected in cases:
            with self.subTest(offset=offset):
                wrapper = self.make_wrapper(3, arch="gpu", offset=offset, angles=angles)
                vectors = self.astra.create_proj_geom.call_args.args[-1]
                np.testing.assert_allclose(vectors[:, 3:6], expected, atol=1e-14)
                np.testing.assert_allclose(
                    vectors[:, :3], [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-14
                )
                np.testing.assert_allclose(
                    vectors[:, 6:9], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-14
                )
                np.testing.assert_array_equal(vectors[:, 9:], [[0.0, 0.0, 1.0]] * 2)
                wrapper.close()

    def test_offset_inputs_are_copied_and_2d_vectors_remain_compatible(self):
        offsets = np.array([1.0, -2.0, 0.5, 3.0])
        original = offsets.copy()
        wrapper = self.make_wrapper(3, arch="gpu", offset=offsets)
        offsets[:] = 99.0
        np.testing.assert_array_equal(wrapper.centre_of_rotation, original)
        angles = np.arange(4.0)
        v2 = self.funcs._vec_geom_init2D(angles, original.tolist())
        v3 = self.funcs._vec_geom_init3D(angles, 1.0, 1.0, original)
        np.testing.assert_allclose(v2, v3[:, [0, 1, 3, 4, 6, 7]])
        np.testing.assert_array_equal(v3[:, 5], np.zeros(4))

    def test_full_and_subset_offset_geometry_agree(self):
        angles = np.linspace(0.0, np.pi, 5, endpoint=False)
        offsets = (
            0.75,
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [[-1.0, 2.0], [0.0, 3.0], [1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )
        for offset in offsets:
            with self.subTest(offset=offset):
                full = self.make_wrapper(3, arch="gpu", offset=offset, angles=angles)
                full_vectors = self.astra.create_proj_geom.call_args.args[-1]
                self.astra.create_proj_geom.reset_mock()
                subsets = self.make_wrapper(
                    3, subsets=2, arch="gpu", offset=offset, angles=angles
                )
                self.assertEqual(self.astra.create_proj_geom.call_count, 2)
                for subset, call in enumerate(
                    self.astra.create_proj_geom.call_args_list
                ):
                    np.testing.assert_allclose(call.args[-1], full_vectors[subset::2])
                full.close()
                subsets.close()

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
                    self.make_wrapper(3, arch="gpu", offset=offset)
                with self.assertRaises(error):
                    self.funcs._vec_geom_init3D(np.arange(4.0), 1.0, 1.0, offset)
                self.astra.create_projector.assert_not_called()
                self.astra.create_proj_geom.assert_not_called()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.make_wrapper(2, arch="gpu", offset=np.zeros((4, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.funcs._vec_geom_init2D(np.arange(4.0), np.zeros((4, 2)))

    def test_cpu_zero_offset_arrays_are_accepted_and_nonzero_rejected(self):
        with self.make_wrapper(2, arch="cpu", offset=[0.0, 0.0, 0.0, 0.0]):
            pass
        with self.assertRaisesRegex(ValueError, "GPU only"):
            self.make_wrapper(2, arch="cpu", offset=[0.0, 0.0, 1.0, 0.0])

    def test_single_view_pairs_and_nonunit_spacing_preserve_legacy_units(self):
        vectors = self.funcs._vec_geom_init3D(np.array([0.0]), 2.0, 3.0, [[4.0, -5.0]])
        np.testing.assert_array_equal(
            vectors, [[0.0, -1.0, 0.0, 4.0, 0.0, -5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0]]
        )

    def test_gpu_selection_preserves_callable_and_cpu_does_not_select_gpu(self):
        select_gpu = self.astra.set_gpu_index
        wrapper = self.make_wrapper(3, arch="gpu")
        for index in (0, 1, 2):
            select_gpu.reset_mock()
            wrapper.device_index = index
            select_gpu.assert_called_once_with(index)
            self.assertIs(self.astra.set_gpu_index, select_gpu)
        select_gpu.reset_mock()
        self.make_wrapper(2, arch="cpu")
        select_gpu.assert_not_called()

    def test_subset_indices_cover_every_view_once_without_padding(self):
        for n_views in range(1, 10):
            for count in range(1, n_views + 1):
                with self.subTest(n_views=n_views, count=count):
                    wrapper = self.make_wrapper(
                        3, subsets=count, arch="gpu", angles=np.arange(float(n_views))
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
                dim, subsets=4, arch="gpu", offset=offset, angles=angles
            )
            expected = (
                self.funcs._vec_geom_init2D(angles, offset)
                if dim == 2
                else self.funcs._vec_geom_init3D(angles, 1.0, 1.0, offset)
            )
            self.assertEqual(self.astra.create_proj_geom.call_count, 4)
            for index, call in enumerate(self.astra.create_proj_geom.call_args_list):
                np.testing.assert_array_equal(
                    call.args[-1], expected[index : index + 1]
                )
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
                    self.make_wrapper(3, subsets=count, arch="gpu")
                self.astra.create_projector.assert_not_called()
        for count, expected in ((None, 1), (1, 1), (np.int64(2), 2)):
            wrapper = self.make_wrapper(3, subsets=count, arch="gpu")
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
        cupy._default_memory_pool = SimpleNamespace(free_all_blocks=Mock())
        supp = ModuleType("suppTools")
        supp.check_kwargs = lambda data, **kw: data
        supp.perform_recon_crop = lambda data, *args: data
        supp._apply_horiz_detector_padding = lambda data, *args, **kw: data
        dictionaries = ModuleType("dicts")
        dictionaries.dicts_check = Mock()
        regularisers = ModuleType("regularisers")
        regularisers.prox_regul = Mock()
        fidelities = ModuleType("fidelities")
        fidelities.grad_data_term = Mock(
            side_effect=lambda rec, x, *args: np.zeros_like(x)
        )
        with patch.dict(
            "sys.modules",
            {
                "cupy": cupy,
                "astra": self.astra,
                "tomobar.supp.funcs": self.funcs,
                "tomobar.supp.suppTools": supp,
                "tomobar.supp.dicts": dictionaries,
                "tomobar.regularisersCuPy": regularisers,
                "tomobar.astra_wrappers.astra_tools3d": self.tools3d,
                "tomobar.data_fidelities": fidelities,
            },
        ):
            return load_module("_subset_test_ir", "methodsIR_CuPy.py")

    def make_iterative(self, module, n_views, count):
        rec = module.RecToolsIRCuPy(
            2, 0, 1, 0.0, np.arange(float(n_views)), 2, OS_number=count
        )
        self.addCleanup(rec.Atools.close)
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
                        data_dict = {
                            "projection_data": data,
                            "data_fidelity": fidelity,
                            "beta_SWLS": 0.1,
                        }
                        algorithm = {
                            "iterations": 1,
                            "lipschitz_const": 1.0,
                            "nonnegativity": False,
                            "recon_mask_radius": 1.0,
                            "ADMM_rho_const": 1.0,
                            "ADMM_relax_par": 1.0,
                            "verbose": False,
                        }
                        regularisation = {"method": None, "regul_param": 1.0}
                        rec._RecToolsIRCuPy__common_initialisation = Mock(
                            return_value=(
                                data_dict,
                                algorithm,
                                regularisation,
                                np.ones((1, 2, 2), dtype=np.float32),
                                weights,
                                count > 1,
                            )
                        )
                        module.grad_data_term.reset_mock()
                        getattr(rec, method)(data_dict, algorithm, regularisation)
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
                        rec.Atools.close()

    def test_osem_uses_every_subset_including_view_zero(self):
        module = self.load_iterative_module()
        for n_views, count in ((1, 1), (4, 4), (5, 2)):
            rec = self.make_iterative(module, n_views, count)
            data = np.arange(1, 1 + n_views * 2, dtype=np.float32).reshape(
                1, n_views, 2
            )
            rec._RecToolsIRCuPy__common_initialisation = Mock(
                return_value=(
                    {"projection_data": data},
                    {"iterations": 1, "recon_mask_radius": 1.0},
                    {"method": None},
                    np.ones((1, 2, 2), dtype=np.float32),
                    None,
                    count > 1,
                )
            )
            rec._Ax = Mock(
                side_effect=lambda x, subset, use_os: np.ones(
                    (1, len(rec.Atools.subset_indices[subset]), 2)
                )
            )
            rec._Atb = Mock(side_effect=lambda *args: np.ones((1, 2, 2)))
            rec.OSEM({"projection_data": data})
            self.assertEqual(rec._Ax.call_count, count)
            self.assertEqual(rec._Atb.call_count, count + 1)
            for subset, call in enumerate(rec._Atb.call_args_list[1:]):
                np.testing.assert_array_equal(call.args[0], data[:, subset::count, :])
                self.assertEqual(call.args[1], subset)
            rec.Atools.close()

    def test_power_method_does_not_index_subset_weights_with_global_indices(self):
        module = self.load_iterative_module()
        for n_views, count in ((5, 2), (4, 4)):
            rec = self.make_iterative(module, n_views, count)
            subset_shape = (1, len(rec.Atools.subset_indices[0]), 2)
            rec.Atools._forwprojOSCuPy = Mock(return_value=np.ones(subset_shape))
            rec.Atools._backprojOSCuPy = Mock(return_value=np.ones((1, 2, 2)))
            result = rec.powermethod({"data_fidelity": "PWLS"})
            self.assertTrue(np.isfinite(result))
            self.assertEqual(rec.Atools._forwprojOSCuPy.call_count, 16)
            self.assertEqual(rec.Atools._backprojOSCuPy.call_count, 15)
            for call in rec.Atools._backprojOSCuPy.call_args_list:
                self.assertEqual(call.args[0].shape, subset_shape)
                self.assertEqual(call.args[1], 0)
            rec.Atools.close()

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


if __name__ == "__main__":
    unittest.main()
