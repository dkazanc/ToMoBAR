# Defines common fixtures and makes them available to all tests

import os
from contextlib import ExitStack
from types import SimpleNamespace
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    print("Cupy library is a required to run some tests")

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: mark test as performance test")
    config.addinivalue_line(
        "markers", "full: Run more memory/computation intensive tests"
    )


def pytest_addoption(parser):
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
    if config.getoption("--performance"):
        skip_other = pytest.mark.skip(reason="not a performance test")
        for item in items:
            if "perf" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="performance test - use '--performance' to run"
        )
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)
    if config.getoption("--full"):
        skip_other = pytest.mark.skip(reason="not memory/computation intensive tests")
        for item in items:
            if "full" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="for memory/computation intensive tests - use '--full' to run"
        )
        for item in items:
            if "full" in item.keywords:
                item.add_marker(skip_perf)


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
def data_XRFfile(test_data_path):
    in_file = os.path.join(test_data_path, "Ga-Ka_aligned.npz")
    with np.load(in_file) as archive:
        yield archive


@pytest.fixture
def raw_data(data_raw_file):
    return np.float32(np.copy(data_raw_file["data"]))


@pytest.fixture
def raw_data_Xrf(data_XRFfile):
    return np.float32(np.copy(data_XRFfile["arr_0"]))


@pytest.fixture
def angles_data_Xrf(data_XRFfile):
    return np.float32(np.copy(data_XRFfile["arr_1"]))


@pytest.fixture
def raw_data_cupy(raw_data):
    return cp.asarray(raw_data)


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
def data_cupy(data):
    return cp.asarray(data)


@pytest.fixture
def angles(data_file):
    return np.copy(data_file["angles"])


@pytest.fixture
def angles_cupy(angles):
    return cp.asarray(angles)


@pytest.fixture
def gpu():
    astra = pytest.importorskip("astra")
    if not astra.use_cuda():
        pytest.skip("ASTRA CUDA is unavailable")
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy CUDA is unavailable")
    return cp


@pytest.fixture
def mock_reconstruction():
    """Isolated ASTRA/CuPy doubles exposing the backend and public modules."""
    from tests import test_astra_resource_lifecycle as resources
    from tests import test_axis_conventions as axes

    with ExitStack() as cleanup:
        wrapper = resources.TestAstraResourceLifecycle()
        cleanup.callback(wrapper.doCleanups)
        wrapper.setUp()
        direct = axes.TestAxisConventions()
        cleanup.callback(direct.doCleanups)
        direct.setUp()
        direct.direct_module.AstraTools3D = wrapper.tools3d.AstraTools3D

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
            backend=wrapper, direct=direct.direct_module, iterative=iterative
        )
