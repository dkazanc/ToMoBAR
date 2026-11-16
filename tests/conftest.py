"""Shared fixtures and CPU/GPU selection for all test categories."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def ensure_clean_memory(gpu):
    gpu.get_default_memory_pool().free_all_blocks()
    gpu.get_default_pinned_memory_pool().free_all_blocks()
    try:
        yield None
    finally:
        gpu.get_default_memory_pool().free_all_blocks()
        gpu.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")


def pytest_addoption(parser):
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="run CPU tests; skip CUDA cases and unavailable optional GPU modules",
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only",
    )
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run more memory/computation intensive tests",
    )


def pytest_collection_modifyitems(config, items):
    """Mark GPU dependencies and select expensive suites consistently."""
    cpu_only = config.getoption("--cpu-only")
    performance = config.getoption("--performance")
    full = config.getoption("--full")
    for item in items:
        if "gpu" in item.fixturenames:
            item.add_marker(pytest.mark.gpu)
        if cpu_only and "gpu" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="CUDA test excluded by --cpu-only"))

        expensive = "perf" in item.keywords or "full" in item.keywords
        selected = (performance and "perf" in item.keywords) or (
            full and "full" in item.keywords
        )
        if performance or full:
            if not selected:
                item.add_marker(
                    pytest.mark.skip(reason="outside the requested expensive suite")
                )
        elif expensive:
            item.add_marker(
                pytest.mark.skip(
                    reason="use --performance or --full to run expensive tests"
                )
            )


# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = os.path.join(test_data_path, "normalised_data.npz")
    with np.load(in_file) as archive:
        yield archive


@pytest.fixture(scope="session")
def data_raw_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    with np.load(in_file) as archive:
        yield archive


@pytest.fixture(scope="session")
def data_shifted_file(test_data_path):
    """Analytical misaligned projections and per-view shift vectors."""
    in_file = os.path.join(test_data_path, "projData3D_analyt_shifted.npz")
    with np.load(in_file, allow_pickle=False) as archive:
        yield archive


@pytest.fixture
def data_shifted(data_shifted_file):
    """Independent float32 projections, stored in archive order (128, 100, 181)."""
    return np.copy(data_shifted_file["array"])


@pytest.fixture
def synth_data_shifts(data_shifted_file):
    """Independent per-view (horizontal, vertical) shifts in the archive dtype."""
    return np.copy(data_shifted_file["vectors"])


@pytest.fixture
def data_shifted_cupy(gpu, data_shifted):
    """Misaligned projections on the GPU; skip when CUDA is unavailable."""
    return gpu.asarray(data_shifted)


@pytest.fixture
def raw_data(data_raw_file):
    return np.float32(np.copy(data_raw_file["data"]))


@pytest.fixture
def raw_data_cupy(gpu, raw_data):
    return gpu.asarray(raw_data)


@pytest.fixture
def flats(data_raw_file):
    return np.float32(np.copy(data_raw_file["flats"]))


@pytest.fixture
def darks(data_raw_file):
    return np.float32(np.copy(data_raw_file["darks"]))


@pytest.fixture
def data(data_file):
    return np.copy(data_file["data_norm"])


@pytest.fixture
def data_cupy(gpu, data):
    return gpu.asarray(data)


@pytest.fixture
def angles(data_file):
    return np.copy(data_file["angles"])


@pytest.fixture
def angles_cupy(gpu, angles):
    return gpu.asarray(angles)


@pytest.fixture
def gpu(request):
    if request.config.getoption("--cpu-only"):
        pytest.skip("CUDA test excluded by --cpu-only")
    astra = pytest.importorskip("astra")
    if not astra.use_cuda():
        pytest.skip("ASTRA CUDA is unavailable")
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy CUDA is unavailable")
    return cp


@pytest.fixture
def mock_reconstruction():
    """Isolated ASTRA/CuPy doubles exposing reconstructors and their operator."""
    from tests.helpers.reconstruction import DirectHarness, ResourceHarness

    with ResourceHarness() as wrapper, DirectHarness() as direct:
        direct.backend_factory._astra_backend_types = lambda: (
            wrapper.tools2d.AstraTools2D,
            wrapper.tools3d.AstraTools3D,
        )

        def size(geometry):
            if "GridSliceCount" in geometry:
                return (
                    geometry["GridSliceCount"],
                    geometry["GridRowCount"],
                    geometry["GridColCount"],
                )
            return (
                geometry["DetectorRowCount"],
                len(geometry["Vectors"]),
                geometry["DetectorColCount"],
            )

        wrapper.astra.geom_size.side_effect = size
        wrapper.direct_arrays = []

        def execute(handle, data, *, out):
            assert handle in wrapper.backend.live["projector3d"]
            assert data.flags.c_contiguous
            assert data.dtype == np.float32
            wrapper.direct_arrays.append(data.copy())
            out[:] = 1

        wrapper.astra.projector3d.direct_FP.side_effect = execute
        wrapper.astra.projector3d.direct_BP.side_effect = execute

        iterative = wrapper.load_iterative_module()
        yield SimpleNamespace(
            backend=wrapper,
            direct=direct.direct_module,
            iterative=iterative,
            ProjectionOperator=iterative.ProjectionOperator,
        )


@pytest.fixture
def cpu_rec():
    """Fresh explicit NumPy harness for the production solver state engine."""
    from tests.helpers.solver import CPURec

    return CPURec()


@pytest.fixture
def two_pixel_rec(cpu_rec, monkeypatch):
    from tests.helpers.solver import two_pixel_tv_prox

    rec = cpu_rec
    rec.shape = (1, 1, 2)
    rec.operator.forward = lambda image, *args, **kwargs: image.copy()
    rec.operator.backproject = lambda data, *args, **kwargs: data.copy()
    rec._finish_reconstruction = lambda image, **kwargs: image
    monkeypatch.setattr(rec, "proximal", two_pixel_tv_prox)
    return rec
