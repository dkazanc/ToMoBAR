"""Geometry updates preserve configuration and invalidate projector resources."""

from dataclasses import replace
import gc
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)


def make_rec(mock, kind, padding=2):
    if kind == "2d":

        def size(geometry):
            if "GridRowCount" in geometry:
                return geometry["GridRowCount"], geometry["GridColCount"]
            return len(geometry["Vectors"]), geometry["DetectorCount"]

        mock.backend.astra.geom_size.side_effect = size

        def execute(handle, data, *, out):
            assert handle in mock.backend.backend.live["projector"]
            out[:] = 1

        mock.backend.astra.projector.direct_FP.side_effect = execute
        mock.backend.astra.projector.direct_BP.side_effect = execute
        scan = Parallel2DGeometry(angles=[0.0, 0.4, 1.0, 2.0, 3.0], detector_shape=(7,))
        grid = Volume2DGeometry((4, 6))
    else:
        scan = Parallel3DGeometry(
            angles=[0.0, 0.4, 1.0, 2.0, 3.0], detector_shape=(3, 7)
        )
        grid = Volume3DGeometry((2, 4, 6))
    cls = (
        mock.iterative.IterativeReconstructor
        if kind == "iterative"
        else mock.direct.DirectReconstructor
    )
    options = dict(num_subsets=2, output_shape=(1, 2, 4)) if kind == "iterative" else {}
    return cls(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=padding,
        **options,
    )


@pytest.mark.parametrize("kind", ["2d", "direct", "iterative"])
@pytest.mark.parametrize("padding", [0, 2])
def test_replacement_releases_all_handles_and_preserves_configuration(
    mock_reconstruction, kind, padding
):
    rec = make_rec(mock_reconstruction, kind, padding)
    backend = mock_reconstruction.backend
    image = np.ones(rec.volume_geometry.shape, dtype=np.float32)
    data = np.ones(rec.projection_geometry.projection_shape, dtype=np.float32)
    grid = rec.volume_geometry
    with rec:
        rec.forward(image)
        if kind == "iterative":
            for i in range(2):
                rec.forward(image, subset_index=i)
        old = rec.Atools
        handles = set(old._projector_handles.values())
        calls = backend.astra.create_projector.call_count
        changed = replace(rec.projection_geometry, centre_of_rotation=1.5)
        rec.replace_geometry(changed)
        assert rec.geometry_version == 1
        assert rec.projection_geometry is changed
        assert rec.volume_geometry is grid
        assert rec.detector_padding == padding
        assert rec.Atools.device_index == old.device_index
        assert rec.Atools.ordsub_number == old.ordsub_number
        assert old._closed and not old._projector_handles
        assert not rec.Atools._projector_handles
        assert backend.astra.create_projector.call_count == calls
        manager = backend.backend.live["projector" if kind == "2d" else "projector3d"]
        assert not handles.intersection(manager)
        assert rec.forward(image).shape == data.shape
        np.testing.assert_array_equal(image, 1)
        np.testing.assert_array_equal(data, 1)
        assert rec.backproject(data).shape == grid.shape
        if kind == "iterative":
            assert rec.output_shape == (1, 2, 4)
            np.testing.assert_array_equal(rec.subset_indices[1], [1, 3])
            expected = changed.to_vectors()[1::2]
            np.testing.assert_array_equal(
                rec.Atools.proj_geom_OS[1]["Vectors"], expected
            )
        assert backend.astra.create_projector.call_count == calls + 1
    assert not backend.backend.live["projector" if kind == "2d" else "projector3d"]


@pytest.mark.parametrize("kind", ["2d", "direct", "iterative"])
def test_rejected_updates_and_noop_preserve_handles(mock_reconstruction, kind):
    with make_rec(mock_reconstruction, kind) as rec:
        rec.forward(np.ones(rec.volume_geometry.shape))
        old = rec.Atools
        current = rec.projection_geometry
        rec.replace_geometry(current)
        assert rec.geometry_version == 0
        for invalid in (None, Volume3DGeometry((2, 4, 6))):
            with pytest.raises(TypeError):
                rec.replace_geometry(invalid)
        changed_shape = (*current.detector_shape[:-1], 9)
        with pytest.raises(ValueError, match="preserve"):
            rec.replace_geometry(replace(current, detector_shape=changed_shape))
        with pytest.raises(ValueError, match="preserve"):
            rec.replace_geometry(replace(current, angles=[0.0], detector_shifts=None))
        with patch.object(
            type(old), "__init__", side_effect=RuntimeError("construction failed")
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                rec.replace_geometry(replace(current, centre_of_rotation=2.0))
        assert rec.Atools is old and not old._closed
        assert rec.projection_geometry is current
        assert rec.geometry_version == 0
        assert old._projector_handles
    with pytest.raises(RuntimeError, match="closed"):
        rec.replace_geometry(current)


def test_repeat_tomography_laminography_and_vector_updates(mock_reconstruction):
    with make_rec(mock_reconstruction, "iterative") as rec:
        for version, angle in enumerate([0.7, 0.9, np.pi / 2], start=1):
            scan = Parallel3DGeometry.laminography(
                angles=[0.0, 0.4, 1.0, 2.0, 3.0],
                detector_shape=(3, 7),
                tilt_angle=angle,
            )
            rec.replace_geometry(scan)
            assert rec.geometry_version == version
            np.testing.assert_array_equal(rec.Atools.vectors, scan.to_vectors())
            rec.forward(np.ones(rec.volume_geometry.shape), subset_index=1)
        vectors = scan.transformed(rotation=np.eye(3), translation=[1, 2, 3])
        rec.replace_geometry(vectors)
        np.testing.assert_array_equal(rec.Atools.vectors, vectors.to_vectors())
        assert rec.geometry_version == 4


def test_2d_filter_settings_survive(mock_reconstruction):
    with make_rec(mock_reconstruction, "2d") as rec:
        rec.Atools.fbp_filter_type = "hann"
        rec.Atools.fbp_filter_parameter = 0.7
        rec.Atools.fbp_filter_d = 0.8
        rec.replace_geometry(replace(rec.projection_geometry, centre_of_rotation=1.0))
        assert rec.Atools.fbp_filter_type == "hann"
        assert rec.Atools.fbp_filter_parameter == 0.7
        assert rec.Atools.fbp_filter_d == 0.8


def test_cuda_replacement_matches_fresh_reconstructor(gpu):
    from tomobar.reconstruction.iterative import IterativeReconstructor

    grid = Volume3DGeometry((4, 6, 8))
    original = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 5, endpoint=False), detector_shape=(6, 12)
    )
    updated = Parallel3DGeometry.laminography(
        angles=original.angles,
        detector_shape=original.detector_shape,
        tilt_angle=0.8,
        detector_shifts=np.tile([0.5, -0.25], (5, 1)),
    )
    rng = np.random.default_rng(42)
    x = rng.uniform(size=grid.shape).astype(np.float32)
    b = rng.uniform(size=original.projection_shape).astype(np.float32)
    with IterativeReconstructor(
        projection_geometry=original,
        volume_geometry=grid,
        num_subsets=2,
        detector_padding=2,
    ) as rec:
        for subset in (None, 0, 1):
            rec.forward(x, subset_index=subset)
        old = rec.Atools
        rec.replace_geometry(updated)
        assert not old._projector_handles
        with IterativeReconstructor(
            projection_geometry=updated,
            volume_geometry=grid,
            num_subsets=2,
            detector_padding=2,
        ) as fresh:
            for subset in (None, 0, 1):
                data = b if subset is None else b[:, subset::2, :]
                np.testing.assert_allclose(
                    rec.forward(x, subset_index=subset),
                    fresh.forward(x, subset_index=subset),
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    rec.backproject(data, subset_index=subset),
                    fresh.backproject(data, subset_index=subset),
                    rtol=1e-6,
                    atol=1e-6,
                )


@pytest.mark.parametrize("kind", ["2d", "direct", "iterative"])
@pytest.mark.parametrize("padding", [0, 2])
def test_reconstructor_owns_one_padded_operator_per_geometry(
    mock_reconstruction, kind, padding
):
    rec = make_rec(mock_reconstruction, kind, padding)
    image = np.ones(rec.volume_geometry.shape, np.float32)
    operator = rec.operator
    assert rec.operator is operator
    assert operator._backend is rec.Atools
    assert operator.projection_geometry is rec.padded_projection_geometry
    assert (
        operator.range_shape[-1]
        == rec.projection_geometry.detector_shape[-1] + 2 * padding
    )
    assert operator.subset_count == (2 if kind == "iterative" else 1)
    with pytest.raises(AttributeError):
        rec.operator = None
    operator.forward(image)
    calls = mock_reconstruction.backend.astra.create_projector.call_count
    changed = replace(rec.projection_geometry, centre_of_rotation=0.75)
    rec.replace_geometry(changed)
    assert rec.operator is not operator
    assert rec.operator._backend is rec.Atools
    assert rec.operator.projection_geometry is rec.padded_projection_geometry
    assert rec.operator.subset_count == operator.subset_count
    assert rec.operator.domain_shape == operator.domain_shape
    assert operator.projection_geometry is not rec.padded_projection_geometry
    assert not rec.operator._backend._projector_handles
    assert mock_reconstruction.backend.astra.create_projector.call_count == calls
    with pytest.raises(RuntimeError, match="closed"):
        operator.forward(image)
    current = rec.operator
    if kind == "iterative":
        # Continuation's legacy helpers resolve the current operator each time.
        out = np.empty(current.subset_shape(1), np.float32)
        assert rec.operator.forward(image, subset_index=1, out=out) is out
        back = np.empty(current.domain_shape, np.float32)
        assert rec.operator.backproject(out, subset_index=1, out=back) is back
    else:
        current.forward(image)
    rec.close()
    rec.close()
    with pytest.raises(RuntimeError, match="closed"):
        current.forward(image)
    with pytest.raises(RuntimeError, match="closed"):
        _ = rec.operator


@pytest.mark.parametrize("kind", ["2d", "direct", "iterative"])
def test_failed_operator_install_leaves_current_geometry_usable(
    mock_reconstruction, kind
):
    with make_rec(mock_reconstruction, kind) as rec:
        operator, geometry = rec.operator, rec.projection_geometry
        operator.forward(np.ones(rec.volume_geometry.shape, np.float32))
        replacements = []

        def fail(backend, **kwargs):
            replacements.append(backend)
            raise RuntimeError("operator creation failed")

        with patch.object(type(operator), "_from_backend", side_effect=fail):
            with pytest.raises(RuntimeError, match="operator creation failed"):
                rec.replace_geometry(replace(geometry, centre_of_rotation=1.0))
        assert rec.operator is operator and rec.Atools is operator._backend
        assert rec.projection_geometry is geometry and rec.geometry_version == 0
        assert replacements[0]._closed
        operator.forward(np.ones(rec.volume_geometry.shape, np.float32))


@pytest.mark.parametrize("method", ["FISTA", "OSEM"])
def test_cuda_subset_continuation_uses_replacement_operator(gpu, method):
    from tomobar.reconstruction.iterative import IterativeReconstructor

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 9, endpoint=False), detector_shape=(3, 12)
    )
    grid = Volume3DGeometry((3, 6, 8))
    data = gpu.asarray(
        np.random.default_rng(7).uniform(0.1, 1, scan.projection_shape),
        dtype=gpu.float32,
    )
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        num_subsets=3,
        detector_padding=2,
    ) as rec:
        options = {"lipschitz_constant": rec.powermethod()} if method == "FISTA" else {}
        state = rec.initialise_solver(method, data, **options)
        rec.run(state, iterations=2)
        image = state.image.copy()
        old_operator = rec.operator
        changed = replace(scan, centre_of_rotation=0.4)
        rec.replace_geometry(changed)
        current = rec.operator
        assert current is not old_operator and not current._backend._projector_handles
        with pytest.raises(RuntimeError, match="closed"):
            old_operator.forward(image)
        rec.run(state, iterations=2)
        with IterativeReconstructor(
            projection_geometry=changed,
            volume_geometry=grid,
            num_subsets=3,
            detector_padding=2,
        ) as fresh:
            reference = fresh.initialise_solver(method, data, initial=image, **options)
            fresh.run(reference, iterations=2)
            gpu.testing.assert_allclose(
                state.image, reference.image, rtol=1e-5, atol=1e-6
            )
        assert rec.operator is current and current._backend._projector_handles
        assert (
            state.restart_count == 1 and state.geometry_version == rec.geometry_version
        )


@pytest.mark.parametrize("kind", ["2d", "direct", "iterative"])
def test_owner_collection_closes_retained_borrowed_operator(mock_reconstruction, kind):
    rec = make_rec(mock_reconstruction, kind)
    operator = rec.operator
    image = np.ones(operator.domain_shape, np.float32)
    operator.forward(image)
    owner = weakref.ref(rec)
    del rec
    gc.collect()
    assert owner() is None
    assert not operator._backend._projector_handles
    with pytest.raises(RuntimeError, match="closed"):
        operator.forward(image)
