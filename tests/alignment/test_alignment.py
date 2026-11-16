"""Translation convention, alignment orchestration and synthetic-data recovery."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import shift as image_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import iradon, radon

from tomobar._solvers import state as solver_state
from tomobar.alignment import _constrain_updates, _estimate_updates, align
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry


def estimate(predicted, measured, upsample_factor=10):
    return _estimate_updates(
        predicted,
        measured,
        register=phase_cross_correlation,
        xp=np,
        upsample_factor=upsample_factor,
        normalization=None,
        window=None,
    )


def test_archive_registration_sign_order_and_truth(data_shifted, synth_data_shifts):
    # Undo known motion to obtain registration references. Cubic interpolation
    # also handles archives containing fractional-pixel synthetic shifts.
    measured = data_shifted.copy()
    predicted = np.empty_like(measured)
    for view, (dx, dy) in enumerate(synth_data_shifts):
        predicted[:, view, :] = image_shift(
            measured[:, view, :],
            (-float(dy), -float(dx)),
            order=3,
            mode="constant",
            prefilter=True,
        )
    updates, valid = estimate(predicted, measured)
    assert valid.all()
    np.testing.assert_allclose(updates, -synth_data_shifts.astype(float), atol=0.2)
    np.testing.assert_array_equal(data_shifted, measured)


@pytest.mark.parametrize("normalization", [None, "phase"])
def test_fractional_translation(normalization):
    rng = np.random.default_rng(2)
    reference = np.zeros((48, 1, 64), dtype=np.float32)
    reference[10:35, 0, 15:45] = rng.random((25, 30))
    moving = image_shift(reference, (1.5, 0, -2.25), order=3, mode="constant")
    updates, valid = _estimate_updates(
        reference,
        moving,
        register=phase_cross_correlation,
        xp=np,
        upsample_factor=20,
        normalization=normalization,
        window=None,
    )
    assert valid.all()
    np.testing.assert_allclose(updates[0], [2.25, -1.5], atol=0.15)


class AlignmentRec:
    """A known reprojection target tests orchestration without a CUDA device.

    Real reconstruction and reprojection are exercised separately by the GPU
    archive test below; this double never claims to estimate a volume.
    """

    def __init__(self):
        self.projection_geometry = Parallel3DGeometry(
            angles=[0, 0.5, 1], detector_shape=(32, 40)
        )
        self.geometry_version = 0
        self.calls = []
        self.closed = False
        rng = np.random.default_rng(13)
        self.target = np.zeros((32, 3, 40), dtype=np.float32)
        self.target[8:24, :, 10:30] = rng.random((16, 3, 20))

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("closed")

    def initialise_solver(self, method, data, **options):
        self.options = options
        self.state = SimpleNamespace(
            projections=data.copy(),
            image=np.zeros((2, 4, 4)),
            geometry_version=self.geometry_version,
            iteration=0,
        )
        return self.state

    def _crop_projection(self, data):
        return data

    def run(self, state, iterations):
        state.geometry_version = self.geometry_version
        state.iteration += iterations
        self.calls.append(iterations)

    def forward(self, image):
        out = np.empty_like(self.target)
        for view, (dx, dy) in enumerate(self.projection_geometry.detector_shifts):
            out[:, view, :] = image_shift(
                self.target[:, view, :], (-dy, -dx), mode="constant", order=1
            )
        return out

    def replace_geometry(self, geometry):
        self.projection_geometry = geometry
        self.geometry_version += 1

    def solver_image(self, state, recon_mask_radius=None):
        return state.image.copy()


def test_alignment_geometry_updates_and_final_state():
    rec = AlignmentRec()
    motion = np.array([[2, -1], [-1, 2], [-1, -1]])
    measured = np.empty_like(rec.target)
    for view, (dx, dy) in enumerate(motion):
        measured[:, view, :] = image_shift(rec.target[:, view, :], (dy, dx), order=0)
    before = measured.copy()
    result = align(rec, measured, xp=np, iterations=5, taper=False, tolerance=0.01)
    np.testing.assert_allclose(result.shifts, -motion, atol=0.1)
    assert result.converged
    assert result.residuals[-1] < result.residuals[0] * 0.01
    assert result.state.geometry_version == rec.geometry_version
    assert len(rec.calls) == len(result.convergence) + 1
    assert len(result.shift_history) == len(result.convergence) + 1
    assert len(result.residuals) == len(result.convergence) + 1
    assert result.geometry is rec.projection_geometry
    np.testing.assert_array_equal(measured, before)
    assert not np.shares_memory(result.image, result.state.image)


def test_reference_constrain_updatest_and_update_bound():
    updates = np.array([[2.0, 3.0], [-4.0, 1.0], [6.0, -2.0]])
    valid = np.array([True, True, False])
    updates[~valid] = 0
    mean = _constrain_updates(
        updates, valid, reference_view=None, relaxation=0.5, max_shift=1.0
    )
    np.testing.assert_allclose(mean[valid].mean(axis=0), 0, atol=1e-12)
    assert np.max(np.abs(mean)) <= 1.0
    anchored = _constrain_updates(
        updates, valid, reference_view=1, relaxation=1.0, max_shift=2.0
    )
    np.testing.assert_array_equal(anchored[1], 0)
    np.testing.assert_array_equal(anchored[2], 0)
    with pytest.raises(ValueError, match="reference_view"):
        _constrain_updates(
            updates, valid, reference_view=2, relaxation=1.0, max_shift=None
        )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"iterations": 0}, ValueError),
        ({"upsample_factor": 0}, ValueError),
        ({"reference_view": 3}, ValueError),
        ({"reference_view": True}, TypeError),
        ({"relaxation": 2.0}, ValueError),
        ({"normalization": "bad"}, ValueError),
        ({"registration_backend": "bad"}, ValueError),
        ({"solver_options": {"initial": None}}, ValueError),
    ],
)
def test_validation_before_geometry_changes(kwargs, error):
    rec = AlignmentRec()
    with pytest.raises(error):
        align(rec, rec.target, xp=np, **kwargs)
    assert rec.geometry_version == 0


def test_failed_registration_restores_start_geometry(monkeypatch):
    import tomobar.alignment as module

    rec = AlignmentRec()
    original = rec.projection_geometry
    calls = []

    def estimate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("registration failed")
        return np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]), np.ones(3, dtype=bool)

    monkeypatch.setattr(module, "_estimate_updates", estimate)
    with pytest.raises(RuntimeError, match="registration failed"):
        align(rec, rec.target, xp=np, iterations=2)
    assert rec.projection_geometry is original


def test_blank_views_and_nan_data():
    blank = np.zeros((8, 3, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="usable registration signal"):
        estimate(blank, blank)
    rec = AlignmentRec()
    data = rec.target.copy()
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        align(rec, data, xp=np)
    assert rec.geometry_version == 0


@pytest.mark.parametrize("inner", [1, 3])
def test_sequential_and_joint_iteration_counts(inner):
    rec = AlignmentRec()
    result = align(
        rec,
        rec.target,
        xp=np,
        reconstruction_iterations=inner,
        final_reconstruction_iterations=2,
        taper=False,
    )
    assert rec.calls == [inner, 2]
    assert result.state.iteration == inner + 2


@pytest.mark.parametrize("use_cupy", [False, True], ids=["numpy-input", "cupy-input"])
def test_cuda_archive_joint_alignment(
    gpu, data_shifted, data_shifted_cupy, synth_data_shifts, use_cupy
):
    cp = gpu
    from tomobar.reconstruction.iterative import RecToolsIR

    supplied = data_shifted_cupy if use_cupy else data_shifted
    original = data_shifted.copy()
    views = supplied.shape[1]
    scan = Parallel3DGeometry(
        angles=np.deg2rad(np.linspace(0.0, 179.9, views, dtype=np.float32)),
        detector_shape=(128, 181),
    )
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((128, 128, 128))
    ) as rec:
        result = rec.align(
            supplied,
            iterations=50,
            reconstruction_method="SIRT",
            upsample_factor=10,
            taper=False,
            tolerance=0.05,
            max_shift=5.0,
            relaxation=0.8,
        )
        assert result.residuals[-1] < result.residuals[0]
        assert result.image.shape == (128, 128, 128)
        assert cp.isfinite(result.image).all()
        # Detector corrections have the opposite sign to the simulated motion.
        expected_shifts = -synth_data_shifts
        shift_error = result.shifts - expected_shifts
        horizontal_error = shift_error[:, 0]
        vertical_error = shift_error[:, 1]

        # A globally translated volume can still be correctly aligned. Its
        # horizontal shift difference follows a sine/cosine pattern over the
        # scan angles. Fit that pattern, allowing a common reference offset.
        translation_patterns = np.column_stack(
            (np.sin(scan.angles), np.cos(scan.angles), np.ones(views))
        )
        translation_coefficients, *_ = np.linalg.lstsq(
            translation_patterns, horizontal_error, rcond=None
        )
        global_horizontal_difference = translation_patterns @ translation_coefficients
        horizontal_error = horizontal_error - global_horizontal_difference

        # Vertical volume translation appears as the same offset in every view.
        vertical_error = vertical_error - vertical_error.mean()

        horizontal_rmse = np.sqrt(np.mean(horizontal_error**2))
        vertical_rmse = np.sqrt(np.mean(vertical_error**2))
        assert horizontal_rmse < 2.0
        assert vertical_rmse < 1.0
        np.testing.assert_array_equal(cp.asnumpy(result.state.projections), original)
        np.testing.assert_array_equal(data_shifted, original)
        np.testing.assert_array_equal(cp.asnumpy(data_shifted_cupy), original)
        if use_cupy:
            assert not cp.shares_memory(result.state.projections, data_shifted_cupy)


# Independent CPU tomography backend for an end-to-end alignment check. This
# exercises the production SIRT state engine without claiming ASTRA equivalence.


class CPUReferenceReconstructor:
    OS_number = 1
    geometry_version = 0

    def __init__(self):
        self.theta = np.linspace(0.0, 179.9, 100)
        self.projection_geometry = Parallel3DGeometry(
            angles=np.deg2rad(self.theta), detector_shape=(32, 46)
        )
        self.volume_geometry = Volume3DGeometry((32, 32, 32))
        self.operator = SimpleNamespace(
            forward=self._forward, backproject=self._backproject
        )

    def _ensure_open(self):
        pass

    def _run_options(self, iterations, **kwargs):
        return iterations

    def _regularisation_options(self, r):
        pass

    def _prepare_inputs(self, p, **kw):
        return p, None

    def _initial_volume(self, initial, ones=False):
        if initial is not None:
            return initial.copy()
        return np.full((32, 32, 32), int(ones), dtype=np.float32)

    def _crop_projection(self, p):
        return p

    def _forward(self, x, *, subset_index=None, out=None):
        b = np.stack(
            [radon(v, theta=self.theta, circle=False, preserve_range=True).T for v in x]
        )
        for j, (dx, dy) in enumerate(self.projection_geometry.detector_shifts):
            b[:, j, :] = image_shift(
                b[:, j, :], (-dy, -dx), order=1, mode="constant", prefilter=False
            )
        if out is not None:
            out[...] = b
            return out
        return b

    def _backproject(self, b, *, subset_index=None, out=None):
        b = b.copy()
        for j, (dx, dy) in enumerate(self.projection_geometry.detector_shifts):
            b[:, j, :] = image_shift(
                b[:, j, :], (dy, dx), order=1, mode="constant", prefilter=False
            )
        result = np.stack(
            [
                iradon(
                    v.T,
                    theta=self.theta,
                    circle=False,
                    output_size=32,
                    filter_name=None,
                    preserve_range=True,
                )
                for v in b
            ]
        )
        if out is not None:
            out[...] = result
            return out
        return result

    def forward(self, x):
        return self.operator.forward(x)

    def replace_geometry(self, g):
        self.projection_geometry = g
        self.geometry_version += 1

    def initialise_solver(self, method, p, **kw):
        return solver_state.initialise(self, method, p, xp=np, **kw)

    def run(self, state, iterations):
        return solver_state.run(
            self, state, iterations, xp=np, gradient=None, proximal=None
        )

    def solver_image(self, state, **kw):
        return state.image.copy()


def test_cpu_archive_reconstruction_reprojection_recovery(
    data_shifted, synth_data_shifts
):
    # Fourfold detector decimation keeps this numerical regression inexpensive.
    measured = data_shifted[::4, :, ::4].copy()
    before = measured.copy()
    rec = CPUReferenceReconstructor()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = align(
            rec,
            measured,
            xp=np,
            iterations=15,
            taper=False,
            tolerance=0.005,
            relaxation=0.8,
            max_shift=2.0,
        )
    assert result.residuals[-1] < 0.6 * result.residuals[0]
    error = result.shifts + synth_data_shifts.astype(float) / 4
    theta = rec.projection_geometry.angles
    basis = np.column_stack((np.sin(theta), np.cos(theta), np.ones(len(theta))))
    error[:, 0] -= basis @ np.linalg.lstsq(basis, error[:, 0], rcond=None)[0]
    error[:, 1] -= error[:, 1].mean()
    # Express accuracy back in original detector pixels, allowing interpolation
    # and the independent CPU projector's discretisation error.
    assert np.sqrt(np.mean(error[:, 0] ** 2)) * 4 < 1.0
    assert np.sqrt(np.mean(error[:, 1] ** 2)) * 4 < 1.0
    assert result.state.geometry_version == rec.geometry_version
    np.testing.assert_array_equal(measured, before)


def test_partial_blank_views_do_not_claim_convergence():
    rec = AlignmentRec()
    rec.target[:, 0, :] = 0
    result = align(rec, rec.target, xp=np, iterations=2, taper=False)
    assert not result.converged
    np.testing.assert_array_equal(result.valid_views[:, 0], False)
    np.testing.assert_array_equal(result.shift_history[:, 0], 0)


def test_nonfinite_prediction_is_rejected():
    data = np.ones((8, 2, 10))
    predicted = data.copy()
    predicted[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite registration image"):
        estimate(predicted, data)


def test_cuda_httomolibgpu_registration(gpu):
    from importlib.util import find_spec

    if find_spec("httomolibgpu") is None:
        pytest.skip("httomolibgpu is not installed")
    from tomobar.alignment import _registration_function

    cp = gpu
    reference = cp.asarray(np.random.default_rng(8).random((32, 40)).astype(np.float32))
    moving = cp.roll(reference, (3, -2), axis=(0, 1))
    shift = _registration_function("httomolibgpu", cp)(
        reference, moving, upsample_factor=10, normalization=None
    )[0]
    np.testing.assert_allclose(shift, [-3, 2], atol=0.1)


@pytest.mark.parametrize("value", [0.0, 0.1, 17.3])
def test_constant_float32_views_have_no_registration_signal(value):
    data = np.full((8, 3, 10), value, dtype=np.float32)
    with pytest.raises(ValueError, match="usable registration signal"):
        estimate(data, data)


@pytest.mark.parametrize("shift", [[np.nan, 0.0], [np.inf, 1.0], [1.0], [[1.0, 2.0]]])
def test_invalid_backend_results_are_rejected(shift):
    rec = AlignmentRec()

    def invalid_registration(*args, **kwargs):
        return shift, 0.0, 0.0

    with pytest.raises(ValueError, match="invalid registration shift for view 0"):
        _estimate_updates(
            rec.target,
            rec.target,
            register=invalid_registration,
            xp=np,
            upsample_factor=10,
            normalization=None,
            window=None,
        )


@pytest.mark.parametrize("shape", [(32, 2, 40), (32, 40)])
def test_inconsistent_projection_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="same 3D shape"):
        estimate(np.ones(shape), np.ones((32, 3, 40)))


def test_starting_shifts_and_cor_are_preserved_at_convergence():
    rec = AlignmentRec()
    starting_shifts = np.array([[2.0, -1.0], [-1.0, 2.0], [-1.0, -1.0]])
    original = replace(
        rec.projection_geometry,
        detector_shifts=starting_shifts,
        centre_of_rotation=0.75,
    )
    rec.projection_geometry = original
    measured = rec.forward(np.zeros((2, 4, 4)))
    result = align(rec, measured, xp=np, taper=False)
    assert result.converged
    np.testing.assert_array_equal(result.shifts, starting_shifts)
    assert rec.projection_geometry is original
    assert rec.projection_geometry.centre_of_rotation == 0.75
    assert rec.geometry_version == 0
    assert result.state.geometry_version == 0


def test_reference_view_keeps_its_nonzero_starting_offset(monkeypatch):
    import tomobar.alignment as module

    rec = AlignmentRec()
    starting = np.array([[1.0, 3.0], [2.0, -1.0], [3.0, -2.0]])
    rec.projection_geometry = replace(rec.projection_geometry, detector_shifts=starting)

    def residual_updates(*args, **kwargs):
        return np.array([[3.0, -1.0], [1.0, 2.0], [-2.0, -1.0]]), np.ones(3, dtype=bool)

    monkeypatch.setattr(module, "_estimate_updates", residual_updates)
    result = align(
        rec,
        rec.target,
        xp=np,
        iterations=3,
        reference_view=1,
        relaxation=0.5,
        max_shift=0.5,
    )
    np.testing.assert_array_equal(
        result.shift_history[:, 1], np.tile(starting[1], (4, 1))
    )
    assert not result.converged
    assert len(result.convergence) == 3


@pytest.mark.parametrize("failure_point", ["final_solve", "output_snapshot"])
def test_failure_after_geometry_update_rolls_back(monkeypatch, failure_point):
    import tomobar.alignment as module

    rec = AlignmentRec()
    original = replace(rec.projection_geometry, centre_of_rotation=0.25)
    rec.projection_geometry = original

    def residual_updates(*args, **kwargs):
        return np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]), np.ones(3, dtype=bool)

    monkeypatch.setattr(module, "_estimate_updates", residual_updates)
    if failure_point == "final_solve":
        run = rec.run

        def fail_on_final_solve(state, iterations):
            if rec.calls:
                raise RuntimeError("final solve failed")
            run(state, iterations)

        rec.run = fail_on_final_solve
    else:

        def fail_on_snapshot(*args, **kwargs):
            raise RuntimeError("snapshot failed")

        rec.solver_image = fail_on_snapshot
    with pytest.raises(RuntimeError, match="failed"):
        align(rec, rec.target, xp=np, iterations=1)
    assert rec.projection_geometry is original
    np.testing.assert_array_equal(
        rec.projection_geometry.detector_shifts, original.detector_shifts
    )


def test_solver_options_and_warm_start_are_forwarded_without_mutation():
    rec = AlignmentRec()
    initial = np.ones((2, 4, 4))
    weights = np.ones_like(rec.target)
    options = dict(nonnegativity=False, weights=weights)
    axes = ["detY", "angles", "detX"]
    result = align(
        rec,
        rec.target,
        xp=np,
        initial=initial,
        input_axes=axes,
        solver_options=options,
        taper=False,
    )
    assert result.converged
    assert rec.options["initial"] is initial
    assert rec.options["input_axes"] is axes
    assert rec.options["weights"] is weights
    assert not rec.options["nonnegativity"]
    assert options == dict(nonnegativity=False, weights=weights)
    defaults = {}
    align(rec, rec.target, xp=np, solver_options=defaults, taper=False)
    assert defaults == {} and rec.options["nonnegativity"]


def test_closed_reconstructor_rejected_before_preparation():
    rec = AlignmentRec()
    rec.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        align(rec, rec.target, xp=np)
    assert not hasattr(rec, "state")


def test_public_alignment_with_axes_weights_padding_and_output_crop(
    mock_reconstruction,
):
    """Exercise the public API and real state engine with isolated projectors."""
    from tomobar.options import DataFidelity

    module = mock_reconstruction.iterative
    # This fixture runs CuPy operations through NumPy; only host transfer needs
    # an explicit equivalent for the registration adapter.
    module.cp.asnumpy = np.asarray
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 5, endpoint=False),
        detector_shape=(3, 7),
        centre_of_rotation=0.4,
    )
    data = np.random.default_rng(3).random(scan.projection_shape).astype(np.float32)
    weights = np.linspace(0.1, 1.0, data.size, dtype=np.float32).reshape(data.shape)
    initial = np.ones((2, 4, 6), dtype=np.float32)
    axes = ["angles", "detY", "detX"]
    supplied = data.swapaxes(0, 1).copy()
    supplied_weights = weights.swapaxes(0, 1).copy()
    options = dict(
        fidelity=DataFidelity("PWLS"),
        weights=supplied_weights,
        lipschitz_constant=3.0,
        nonnegativity=False,
    )
    with module.RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(initial.shape),
        output_shape=(1, 2, 4),
        detector_padding=2,
    ) as rec:
        # A known, nonconstant reprojection target avoids testing ASTRA here.
        rec.Atools.forward = lambda image, **kw: np.pad(
            data, ((0, 0), (0, 0), (2, 2)), mode="edge"
        )
        result = rec.align(
            supplied,
            initial=initial,
            input_axes=axes,
            reconstruction_method="FISTA",
            reconstruction_iterations=2,
            final_reconstruction_iterations=1,
            solver_options=options,
            taper=False,
            recon_mask_radius=None,
        )
        assert result.converged and rec.geometry_version == 0
        assert result.image.shape == (1, 2, 4)
        assert result.state.image.shape == initial.shape
        assert result.state.iteration == 3
        np.testing.assert_array_equal(result.state.image, initial)
        np.testing.assert_array_equal(
            result.state.projections,
            np.pad(data, ((0, 0), (0, 0), (2, 2)), mode="edge"),
        )
        np.testing.assert_allclose(
            result.state.weights, np.pad(weights, ((0, 0), (0, 0), (2, 2)), mode="edge")
        )
        assert result.geometry.centre_of_rotation == 0.4
        result.image[:] = 99
        np.testing.assert_array_equal(result.state.image, initial)
    np.testing.assert_array_equal(supplied, data.swapaxes(0, 1))
    np.testing.assert_array_equal(supplied_weights, weights.swapaxes(0, 1))
    assert options["weights"] is supplied_weights
    assert "initial" not in options


def test_broken_optional_backend_reports_the_original_import_error(monkeypatch):
    import builtins

    from tomobar.alignment import _registration_function

    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "httomolibgpu.recon._phase_cross_correlation":
            raise ModuleNotFoundError(
                "No module named 'tomobar.methodsDIR_CuPy'",
                name="tomobar.methodsDIR_CuPy",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ImportError, match="tomobar.methodsDIR_CuPy") as error:
        _registration_function("httomolibgpu", np)
    assert error.value.__cause__.name == "tomobar.methodsDIR_CuPy"


@pytest.mark.parametrize("stage", ["reconstruction", "reprojection"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_solver_output_reports_stage_and_restores_geometry(stage, value):
    rec = AlignmentRec()
    original = rec.projection_geometry
    if stage == "reconstruction":

        def bad_run(state, iterations):
            state.image[:] = value

        rec.run = bad_run
    else:
        rec.forward = lambda image: np.full_like(rec.target, value)
    with pytest.raises(ValueError, match=f"OSEM produced a nonfinite {stage}.*cycle 1"):
        align(rec, rec.target, xp=np, reconstruction_method="OSEM")
    assert rec.projection_geometry is original


def test_direct_regularisation_is_forwarded_without_mutating_options():
    from tomobar.options import Regularisation

    rec = AlignmentRec()
    regularisation = Regularisation(method="PD_TV", strength=0.01)
    options = {"nonnegativity": True}
    align(
        rec,
        rec.target,
        xp=np,
        reconstruction_method="OSEM",
        regularisation=regularisation,
        solver_options=options,
    )
    assert rec.options["regularisation"] is regularisation
    assert options == {"nonnegativity": True}
    align(rec, rec.target, xp=np, solver_options={"regularisation": regularisation})
    assert rec.options["regularisation"] is regularisation
    with pytest.raises(ValueError, match="not both"):
        align(
            rec,
            rec.target,
            xp=np,
            regularisation=regularisation,
            solver_options={"regularisation": regularisation},
        )
