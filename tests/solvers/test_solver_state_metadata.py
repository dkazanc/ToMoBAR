"""Public state inspection must not corrupt solver continuation metadata."""

import weakref

import numpy as np
import pytest

from tomobar.options import DataFidelity


@pytest.mark.parametrize(
    "name",
    [
        "method",
        "iteration",
        "segment_iteration",
        "restart_count",
        "geometry_version",
        "converged",
        "image",
        "projections",
        "weights",
        "energy_iterations",
        "energy_versions",
        "energy",
        "energy_times",
        "residuals",
        "options",
        "algorithm_state",
    ],
)
def test_public_state_attributes_cannot_be_rebound_or_deleted(cpu_rec, name):
    state = cpu_rec.initialise_solver(
        "FISTA", np.ones(cpu_rec.shape), return_energy=True
    )
    with pytest.raises(AttributeError):
        setattr(state, name, object())
    with pytest.raises(AttributeError):
        delattr(state, name)
    # Both failed changes leave the solver usable.
    cpu_rec.run(state)
    assert state.method == "FISTA" and state.iteration == 1


def test_slots_reject_misspelled_metadata_and_preserve_weakrefs(cpu_rec):
    state = cpu_rec.initialise_solver("FISTA", np.ones(cpu_rec.shape))
    assert not hasattr(state, "__dict__")
    with pytest.raises(AttributeError):
        state.iteraton = 200
    assert weakref.ref(state)() is state
    assert "method='FISTA'" in repr(state)
    assert "iteration=0" in repr(state)


def test_inspecting_array_bindings_does_not_copy_storage(cpu_rec):
    state = cpu_rec.initialise_solver(
        "FISTA",
        np.ones(cpu_rec.shape),
        fidelity=DataFidelity("PWLS"),
        weights=np.ones(cpu_rec.shape),
    )
    assert state.image is state._image
    assert state.projections is state._projections
    assert state.weights is state._weights
    snapshot = cpu_rec.solver_image(state, recon_mask_radius=None)
    snapshot.fill(99)
    assert not np.shares_memory(snapshot, state.image)
    np.testing.assert_array_equal(state.image, 0)


def test_energy_sample_indices_are_independent_list_snapshots(cpu_rec):
    state = cpu_rec.initialise_solver(
        "FISTA", np.ones(cpu_rec.shape), return_energy=True
    )
    iterations, versions = state.energy_iterations, state.energy_versions
    iterations.append(99)
    versions.clear()
    assert state.energy_iterations == [0]
    assert state.energy_versions == [0]
    with pytest.raises(AttributeError):
        state.energy_iterations += [100]
    assert state.energy_iterations == [0]
    cpu_rec.run(state, iterations=2)
    assert state.energy_iterations == [0, 1, 2]
    assert state.energy_versions == [0, 0, 0]
    assert iterations == [0, 99] and versions == []
    cpu_rec.geometry_version += 1
    cpu_rec.run(state)
    assert state.energy_iterations == [0, 1, 2, 2, 3]
    assert state.energy_versions == [0, 0, 0, 1, 1]


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_rejected_metadata_changes_preserve_continuation_and_restart(cpu_rec, method):
    data = np.arange(2, 8, dtype=np.float32).reshape(cpu_rec.shape)
    state = cpu_rec.initialise_solver(method, data, return_energy=True)
    reference = cpu_rec.initialise_solver(method, data, return_energy=True)
    cpu_rec.run(state)
    cpu_rec.run(reference)
    for name, value in (
        ("method", "OSEM" if method != "OSEM" else "FISTA"),
        ("iteration", 100),
        ("segment_iteration", 100),
        ("restart_count", 100),
        ("geometry_version", 100),
        ("converged", True),
    ):
        with pytest.raises(AttributeError):
            setattr(state, name, value)
    # Editing returned histories cannot change continuation's sampling metadata.
    state.energy_iterations.clear()
    state.energy_versions.append(99)
    state.energy.fill(99)
    state.energy_times.fill(99)
    if method == "ADMM":
        for values in state.residuals.values():
            values.fill(99)
    cpu_rec.geometry_version += 1
    cpu_rec.scale = 1.3
    cpu_rec.run(state, iterations=2)
    cpu_rec.run(reference, iterations=2)
    np.testing.assert_allclose(state.image, reference.image, rtol=1e-6)
    np.testing.assert_allclose(state.energy, reference.energy, rtol=1e-6)
    assert state.iteration == reference.iteration
    assert state.segment_iteration == reference.segment_iteration
    assert state.restart_count == reference.restart_count == 1
    assert state.geometry_version == reference.geometry_version == 1
    assert state.converged == reference.converged
    assert state.energy_iterations == reference.energy_iterations
    assert state.energy_versions == reference.energy_versions
    assert state.energy_times[0] == 0
    if method == "ADMM":
        for name in state.residuals:
            np.testing.assert_allclose(state.residuals[name], reference.residuals[name])
