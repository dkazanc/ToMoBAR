"""Axis-order regression tests using real methods and mocked GPU dependencies."""

import importlib.util
from itertools import permutations
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

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
        cupy = stub_module(
            "cupy",
            ndarray=np.ndarray,
            RawKernel=Mock,
            float32=np.float32,
            ascontiguousarray=np.ascontiguousarray,
            swapaxes=np.swapaxes,
            cuda=SimpleNamespace(
                Device=Mock(return_value=SimpleNamespace(compute_capability="80"))
            ),
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
            self.funcs = load_module("_axis_test_funcs", "supp/funcs.py")
            with patch.dict("sys.modules", {"tomobar.supp.funcs": self.funcs}):
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
        rec.geom = f"{dim}D"
        rec.Atools = SimpleNamespace(
            _forwproj=Mock(return_value=native),
            _forwprojCuPy=Mock(return_value=native),
            _backproj=Mock(side_effect=lambda data: data),
            _backprojCuPy=Mock(side_effect=lambda data: data),
            _fbp=Mock(side_effect=lambda data: data),
            detectors_x_pad=0,
            processing_arch="gpu",
        )
        return rec

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
                        result = self.funcs._transpose_data_axes(
                            supplied, source_axes=source, target_axes=target
                        )
                        np.testing.assert_array_equal(result, expected)
                        self.assertTrue(np.shares_memory(result, original))
                        self.assertEqual(
                            self.funcs._transpose_data_axes(
                                supplied.shape, source_axes=source, target_axes=target
                            ),
                            expected.shape,
                        )
                        np.testing.assert_array_equal(
                            self.funcs._data_dims_swapper(
                                supplied, list(source), list(target)
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
                            result = rec.FORWPROJ(np.zeros((2,) * len(shape)), **out_kw)
                            expected = native.transpose(
                                tuple(axes.index(a) for a in target)
                            )
                            np.testing.assert_array_equal(result, expected)
                            np.testing.assert_array_equal(
                                rec.BACKPROJ(result, **in_kw), native
                            )

    def test_default_orders_and_matching_aliases_preserve_behavior(self):
        axes = ("detY", "angles", "detX")
        native = np.arange(30).reshape(2, 3, 5)
        for cls in self.classes:
            rec = self.reconstructor(cls, native)
            self.assertIs(rec.FORWPROJ(native), native)
            self.assertIs(rec.BACKPROJ(native), native)
            self.assertIs(
                rec.FORWPROJ(
                    native, output_axes=axes, data_axes_labels_order=list(axes)
                ),
                native,
            )
            self.assertIs(
                rec.BACKPROJ(
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
            self.normalized = self.funcs._transpose_data_axes(data, **kwargs)
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

    def test_numpy_fourier_normalizes_input_before_fft(self):
        native = np.arange(12, dtype=np.float32).reshape(3, 4)
        rec = self.reconstructor(self.numpy_module.RecToolsDIR, native, dim=2)
        rec.Atools.recon_size = 2
        rec.Atools.centre_of_rotation = 0.0
        rec.Atools.angles_vec = np.arange(3.0)

        def stop_at_fft(data, **kwargs):
            np.testing.assert_array_equal(data[:, 2:6], native)
            raise RuntimeError("stop before FFT")

        fft = stub_module(
            "scipy.fft",
            fftshift=Mock(),
            ifftshift=stop_at_fft,
            fft=Mock(),
            ifft2=Mock(),
        )
        interp = stub_module("scipy.interpolate", griddata=Mock())
        with patch.dict("sys.modules", {"scipy.fft": fft, "scipy.interpolate": interp}):
            for order in (("angles", "detX"), ("detX", "angles")):
                supplied = native if order[0] == "angles" else native.T
                for key in ("input_axes", "data_axes_labels_order"):
                    with self.assertRaisesRegex(RuntimeError, "stop before FFT"):
                        rec.FOURIER(supplied, method="linear", **{key: order})

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
                        self.funcs._transpose_data_axes(
                            data, source_axes=source, target_axes=target
                        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.funcs._transpose_data_axes(
                np.zeros((3, 5)), source_axes=valid, target_axes=valid
            )
        for cls in self.classes:
            rec = self.reconstructor(cls, data)
            for method, alias in (
                ("FORWPROJ", "output_axes"),
                ("BACKPROJ", "input_axes"),
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
            rec.Atools._forwproj.assert_not_called()
            rec.Atools._forwprojCuPy.assert_not_called()
            rec.Atools._backproj.assert_not_called()
            rec.Atools._backprojCuPy.assert_not_called()

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
                self.funcs._swap_data_axes_to_accepted(labels, target), expected
            )
            self.assertEqual(labels, list(source))
            self.assertEqual(
                self.funcs._swap_data_axes_to_accepted(source, target), expected
            )


if __name__ == "__main__":
    unittest.main()
