"""Axis-order regression tests using real methods and mocked GPU dependencies."""

import unittest
from itertools import permutations
from unittest.mock import patch

import numpy as np

from tests.helpers.reconstruction import DirectHarness
from tomobar.geometry import (
    Parallel3DGeometry,
    Volume3DGeometry,
)


class TestAxisConventions(unittest.TestCase, DirectHarness):
    def setUp(self):
        DirectHarness.__init__(self)
        self.addCleanup(self.close)

    def test_direct_constructors_forward_detector_shifts_separately_from_cor(self):
        shifts = np.array([[1.0, 2.0], [-3.0, 4.0]])
        wrapper = self.tools3d.AstraTools3D
        for cls in self.classes:
            wrapper.reset_mock()
            original = Parallel3DGeometry(
                angles=[0.0, 0.5],
                detector_shape=(3, 5),
                centre_of_rotation=0.75,
                detector_shifts=shifts,
            )
            volume = Volume3DGeometry(shape=(2, 4, 6))
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
                    with self.subTest(cls=cls.__name__, target=target):
                        result = rec.forward(
                            np.zeros((2,) * len(shape)), output_axes=target
                        )
                        expected = native.transpose(
                            tuple(axes.index(a) for a in target)
                        )
                        np.testing.assert_array_equal(result, expected)
                        np.testing.assert_array_equal(
                            rec.backproject(result, input_axes=target), native
                        )

    def test_default_and_explicit_orders_preserve_behavior(self):
        axes = ("detY", "angles", "detX")
        native = np.arange(30).reshape(2, 3, 5)
        for cls in self.classes:
            rec = self.reconstructor(cls, native)
            self.assertIs(rec.forward(native), native)
            self.assertIs(rec.backproject(native), native)
            self.assertIs(
                rec.forward(native, output_axes=axes),
                native,
            )
            self.assertIs(
                rec.backproject(native, input_axes=axes),
                native,
            )

    def test_fbp_input_orders_for_default_and_explicit_filter(self):
        astra_axes = ("detY", "angles", "detX")
        native = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
        for filter_type in (None, "SINC", "RAM-LAK", "ram-lak", "ramp"):
            rec = self.reconstructor(self.direct_module.DirectReconstructor, native)
            filter_input = native.swapaxes(0, 1)
            for order in permutations(astra_axes):
                supplied = native.transpose(tuple(astra_axes.index(a) for a in order))
                with self.subTest(filter_type=filter_type, order=order):
                    np.testing.assert_array_equal(
                        rec.FBP(supplied, filter_type=filter_type, input_axes=order),
                        native,
                    )
                    np.testing.assert_array_equal(
                        self.filter.call_args.args[0], filter_input
                    )

    def test_fourier_inv_normalizes_arrays_and_estimator_shapes_before_kernels(self):
        axes = ("detY", "angles", "detX")
        native = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        rec = self.reconstructor(self.direct_module.DirectReconstructor, native)

        def capture(data, **kwargs):
            self.normalized = self.axes._transpose_data_axes(data, **kwargs)
            return self.normalized

        prepare = rec._prepare_projection

        def capture_prepared(data, **kwargs):
            self.normalized = prepare(data, **kwargs)
            return self.normalized

        with (
            patch.object(
                self.direct_module, "_transpose_data_axes", side_effect=capture
            ),
            patch.object(rec, "_prepare_projection", side_effect=capture_prepared),
        ):
            for shape_only in (False, True):
                for order in permutations(axes):
                    supplied = native.transpose(tuple(axes.index(a) for a in order))
                    with self.subTest(shape_only=shape_only, order=order):
                        with self.assertRaisesRegex(RuntimeError, "stop before CUDA"):
                            rec.FOURIER_INV(
                                supplied.shape if shape_only else supplied,
                                input_axes=order,
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

    def test_bad_labels_and_rank_are_rejected(self):
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
                for labels in invalid:
                    with self.assertRaises(ValueError):
                        getattr(rec, method)(data, **{alias: labels})

            rec.Atools.forward.assert_not_called()
            rec.Atools.backproject.assert_not_called()


if __name__ == "__main__":
    unittest.main()
