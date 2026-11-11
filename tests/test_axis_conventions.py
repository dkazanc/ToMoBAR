"""Axis-order regression tests using real methods and mocked GPU dependencies."""

import importlib.util
from itertools import permutations
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)

ROOT = Path(__file__).resolve().parents[1] / "tomobar"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class TestAxisConventions(unittest.TestCase):
    def setUp(self):
        identity = lambda data, *args, **kwargs: data

        class Device:
            compute_capability = "80"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        cupy = stub_module(
            "cupy",
            ndarray=np.ndarray,
            RawKernel=Mock,
            float32=np.float32,
            ascontiguousarray=np.ascontiguousarray,
            swapaxes=np.swapaxes,
            cuda=SimpleNamespace(Device=Mock(return_value=Device())),
            fft=SimpleNamespace(
                config=SimpleNamespace(
                    get_plan_cache=Mock(return_value=SimpleNamespace(clear=Mock()))
                )
            ),
            _default_memory_pool=SimpleNamespace(free_all_blocks=Mock()),
        )
        self.filter = Mock(side_effect=identity)
        self.kernels = Mock(side_effect=RuntimeError("stop before CUDA kernels"))
        stubs = {
            "cupy": cupy,
            "scipy": stub_module("scipy"),
            "scipy.fftpack": stub_module("scipy.fftpack"),
            "cupyx.scipy.fft": stub_module(
                "cupyx.scipy.fft",
                **{
                    name: Mock()
                    for name in ("fft", "ifft2", "rfftfreq", "rfft", "irfft")
                },
            ),
            "cupyx.scipy.fftpack": stub_module(
                "cupyx.scipy.fftpack", get_fft_plan=Mock()
            ),
            "tomobar.astra_wrappers.astra_tools2d": stub_module(
                "astra_tools2d", AstraTools2D=Mock()
            ),
            "tomobar.astra_wrappers.astra_tools3d": stub_module(
                "astra_tools3d", AstraTools3D=Mock()
            ),
            "tomobar.supp.suppTools": stub_module(
                "suppTools",
                check_kwargs=identity,
                _apply_horiz_detector_padding=identity,
            ),
            "tomobar.supp.memory_estimator_helpers": stub_module(
                "memory", DeviceMemStack=Mock()
            ),
            "tomobar.fourier": stub_module(
                "fourier", _filtersinc3D_cupy=self.filter, calc_filter=Mock()
            ),
            "tomobar.cuda_kernels": stub_module(
                "cuda_kernels", load_cuda_module=self.kernels
            ),
        }
        with patch.dict("sys.modules", stubs):
            self.axes = load_module("_axis_test_axes", "supp/axis_order.py")
            with patch.dict("sys.modules", {"tomobar.supp.axis_order": self.axes}):
                self.numpy_module = load_module("_axis_test_dir", "methodsDIR.py")
                with patch.dict(
                    "sys.modules", {"tomobar.methodsDIR": self.numpy_module}
                ):
                    self.cupy_module = load_module(
                        "_axis_test_dir_cupy", "methodsDIR_CuPy.py"
                    )
        self.numpy_module._filtersinc3D = self.filter
        self.numpy_module._filtersinc2D = self.filter
        self.classes = (self.numpy_module.RecToolsDIR, self.cupy_module.RecToolsDIRCuPy)

    def reconstructor(self, cls, native, dim=3):
        rec = cls.__new__(cls)
        if dim == 3:
            scan = Parallel3DGeometry(
                angles=np.arange(native.shape[1], dtype=float),
                detector_shape=(native.shape[0], native.shape[-1]),
            )
            volume = VolumeGeometry(
                shape=(native.shape[0], native.shape[-1], native.shape[-1])
            )
        else:
            scan = Parallel2DGeometry(
                angles=np.arange(native.shape[0], dtype=float),
                detector_shape=(native.shape[-1],),
            )
            volume = Volume2DGeometry(shape=(native.shape[-1], native.shape[-1]))
        rec._init_device(0)
        rec._init_geometry(scan, volume, 0)
        rec.Atools = SimpleNamespace(
            close=Mock(),
            forward=Mock(return_value=native),
            backproject=Mock(side_effect=lambda data: data),
            _fbp=Mock(side_effect=lambda data: data),
            detectors_x_pad=0,
        )
        self.addCleanup(rec.close)
        return rec

    def test_direct_constructors_forward_detector_shifts_separately_from_cor(self):
        shifts = np.array([[1.0, 2.0], [-3.0, 4.0]])
        wrapper = self.numpy_module.AstraTools3D
        for cls in self.classes:
            wrapper.reset_mock()
            original = Parallel3DGeometry(
                angles=[0.0, 0.5],
                detector_shape=(3, 5),
                centre_of_rotation=0.75,
                detector_shifts=shifts,
            )
            volume = VolumeGeometry(shape=(2, 4, 6))
            rec = cls(
                projection_geometry=original, volume_geometry=volume, detector_padding=2
            )
            self.addCleanup(rec.close)
            call = wrapper.call_args
            self.assertEqual(call.args, ())
            scan = call.kwargs["projection_geometry"]
            self.assertEqual(scan.centre_of_rotation, 0.75)
            np.testing.assert_array_equal(scan.detector_shifts, shifts)
            self.assertEqual(rec.detector_padding, 2)
            self.assertEqual(scan.detector_shape, (3, 9))
            self.assertIs(call.kwargs["volume_geometry"], volume)
            self.assertIs(rec.projection_geometry, original)
            self.assertIs(rec.volume_geometry, volume)

    def test_all_source_target_permutations_and_shape_tuples(self):
        for axes, shape in (
            (("angles", "detX"), (3, 5)),
            (("detY", "angles", "detX"), (2, 3, 5)),
        ):
            original = np.arange(np.prod(shape)).reshape(shape)
            for source in permutations(axes):
                supplied = original.transpose(tuple(axes.index(a) for a in source))
                for target in permutations(axes):
                    with self.subTest(source=source, target=target):
                        expected = original.transpose(
                            tuple(axes.index(a) for a in target)
                        )
                        result = self.axes._transpose_data_axes(
                            supplied, source_axes=source, target_axes=target
                        )
                        np.testing.assert_array_equal(result, expected)
                        self.assertTrue(np.shares_memory(result, original))
                        self.assertEqual(
                            self.axes._transpose_data_axes(
                                supplied.shape, source_axes=source, target_axes=target
                            ),
                            expected.shape,
                        )
                        np.testing.assert_array_equal(
                            self.axes._transpose_data_axes(
                                supplied,
                                source_axes=list(source),
                                target_axes=list(target),
                            ),
                            expected,
                        )

    def test_forward_output_and_backprojection_roundtrip_for_all_layouts(self):
        for cls in self.classes:
            for axes, shape in (
                (("angles", "detX"), (3, 5)),
                (("detY", "angles", "detX"), (2, 3, 5)),
            ):
                native = np.arange(np.prod(shape)).reshape(shape)
                rec = self.reconstructor(cls, native, len(shape))
                for target in permutations(axes):
                    for legacy in (False, True):
                        with self.subTest(
                            cls=cls.__name__, target=target, legacy=legacy
                        ):
                            out_kw = {
                                (
                                    "data_axes_labels_order"
                                    if legacy
                                    else "output_axes"
                                ): target
                            }
                            in_kw = {
                                (
                                    "data_axes_labels_order" if legacy else "input_axes"
                                ): target
                            }
                            result = rec.forward(np.zeros((2,) * len(shape)), **out_kw)
                            expected = native.transpose(
                                tuple(axes.index(a) for a in target)
                            )
                            np.testing.assert_array_equal(result, expected)
                            np.testing.assert_array_equal(
                                rec.backproject(result, **in_kw), native
                            )

    def test_default_orders_and_matching_aliases_preserve_behavior(self):
        axes = ("detY", "angles", "detX")
        native = np.arange(30).reshape(2, 3, 5)
        for cls in self.classes:
            rec = self.reconstructor(cls, native)
            self.assertIs(rec.forward(native), native)
            self.assertIs(rec.backproject(native), native)
            self.assertIs(
                rec.forward(
                    native, output_axes=axes, data_axes_labels_order=list(axes)
                ),
                native,
            )
            self.assertIs(
                rec.backproject(
                    native, input_axes=axes, data_axes_labels_order=list(axes)
                ),
                native,
            )

    def test_fbp_input_orders_and_distinct_legacy_defaults(self):
        astra_axes = ("detY", "angles", "detX")
        native = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
        for cls in self.classes:
            rec = self.reconstructor(cls, native)
            is_cupy = cls is self.cupy_module.RecToolsDIRCuPy
            filter_input = native.swapaxes(0, 1) if is_cupy else native
            np.testing.assert_array_equal(rec.FBP(filter_input), native)
            np.testing.assert_array_equal(self.filter.call_args.args[0], filter_input)
            for order in permutations(astra_axes):
                supplied = native.transpose(tuple(astra_axes.index(a) for a in order))
                for key in ("input_axes", "data_axes_labels_order"):
                    with self.subTest(cls=cls.__name__, order=order, key=key):
                        np.testing.assert_array_equal(
                            rec.FBP(supplied, **{key: order}), native
                        )
                        np.testing.assert_array_equal(
                            self.filter.call_args.args[0], filter_input
                        )

    def test_fourier_inv_normalizes_arrays_and_estimator_shapes_before_kernels(self):
        axes = ("detY", "angles", "detX")
        native = np.arange(24).reshape(2, 3, 4)
        rec = self.reconstructor(self.cupy_module.RecToolsDIRCuPy, native)

        def capture(data, **kwargs):
            self.normalized = self.axes._transpose_data_axes(data, **kwargs)
            return self.normalized

        with patch.object(
            self.cupy_module, "_transpose_data_axes", side_effect=capture
        ):
            for shape_only in (False, True):
                for order in permutations(axes):
                    supplied = native.transpose(tuple(axes.index(a) for a in order))
                    for key in ("input_axes", "data_axes_labels_order"):
                        with self.subTest(shape_only=shape_only, order=order, key=key):
                            with self.assertRaisesRegex(
                                RuntimeError, "stop before CUDA"
                            ):
                                rec.FOURIER_INV(
                                    supplied.shape if shape_only else supplied,
                                    **{key: order},
                                )
                            if shape_only:
                                self.assertEqual(self.normalized, native.shape)
                            else:
                                np.testing.assert_array_equal(self.normalized, native)
                with self.assertRaisesRegex(RuntimeError, "stop before CUDA"):
                    rec.FOURIER_INV(native.shape if shape_only else native)
                if shape_only:
                    self.assertEqual(self.normalized, native.shape)
                else:
                    self.assertIs(self.normalized, native)

    def test_bad_labels_rank_and_conflicting_aliases_are_rejected(self):
        valid = ("detY", "angles", "detX")
        invalid = (
            ("angles", "angles", "detX"),
            ("detY", "angles", "unknown"),
            ("angles", "detY"),
            "angles",
            (0, "angles", "detX"),
        )
        data = np.zeros((2, 3, 5))
        for labels in invalid:
            with self.subTest(labels=labels):
                for source, target in ((labels, valid), (valid, labels)):
                    with self.assertRaises(ValueError):
                        self.axes._transpose_data_axes(
                            data, source_axes=source, target_axes=target
                        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.axes._transpose_data_axes(
                np.zeros((3, 5)), source_axes=valid, target_axes=valid
            )
        for cls in self.classes:
            rec = self.reconstructor(cls, data)
            for method, alias in (
                ("forward", "output_axes"),
                ("backproject", "input_axes"),
                ("FBP", "input_axes"),
            ):
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    getattr(rec, method)(
                        data,
                        **{
                            alias: valid,
                            "data_axes_labels_order": ("angles", "detX", "detY"),
                        },
                    )
                for labels in invalid:
                    with self.assertRaises(ValueError):
                        getattr(rec, method)(data, **{alias: labels})

            rec.Atools.forward.assert_not_called()
            rec.Atools.backproject.assert_not_called()

    def test_legacy_swap_sequences_are_unchanged_and_accept_tuples(self):
        cases = (
            (("detY", "angles", "detX"), ("detY", "angles", "detX"), [None, None]),
            (("angles", "detY", "detX"), ("detY", "angles", "detX"), [(0, 1), None]),
            (("angles", "detX", "detY"), ("detY", "angles", "detX"), [(0, 2), (1, 2)]),
            (("detX", "angles"), ("angles", "detX"), [(0, 1), None]),
        )
        for source, target, expected in cases:
            labels = list(source)
            self.assertEqual(
                self.axes._swap_data_axes_to_accepted(labels, target), expected
            )
            self.assertEqual(labels, list(source))
            self.assertEqual(
                self.axes._swap_data_axes_to_accepted(source, target), expected
            )


if __name__ == "__main__":
    unittest.main()
