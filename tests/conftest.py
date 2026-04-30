# Defines common fixtures and makes them available to all tests

import os
import numpy as np
import pytest
import tomophantom
from tomophantom import TomoP3D

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
    return np.load(in_file)


@pytest.fixture(scope="session")
def data_raw_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    return np.load(in_file)


@pytest.fixture
def raw_data(data_raw_file):
    return np.float32(np.copy(data_raw_file["data"]))


@pytest.fixture
def flats(data_raw_file):
    return np.float32(np.copy(data_raw_file["flats"]))


@pytest.fixture
def darks(
    data_raw_file,
):
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
def phantom_model():
    return 13


@pytest.fixture
def phantom_path_library():
    path = os.path.dirname(tomophantom.__file__)
    path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
    return path_library3D


@pytest.fixture
def phantom_N_size():
    return 128


@pytest.fixture
def phantom_3D_volume(phantom_model, phantom_N_size, phantom_path_library):
    # This will generate a N_size x N_size x N_size phantom (3D)
    return TomoP3D.Model(phantom_model, phantom_N_size, phantom_path_library)


@pytest.fixture
def phantom_3D_projection_angles_rad(phantom_N_size):
    angles_num = int(0.25 * np.pi * phantom_N_size)
    return (
        np.linspace(0.0, 179.9, angles_num, dtype="float32") * np.pi / 180
    )  # in degrees


@pytest.fixture
def phantom_3D_projections(
    phantom_model,
    phantom_N_size,
    phantom_3D_projection_angles_rad,
    phantom_path_library,
):
    Horiz_det = int(np.sqrt(2) * phantom_N_size)  # detector column count (horizontal)
    Vert_det = (
        phantom_N_size  # detector row count (vertical) (no reason for it to be > N)
    )
    return TomoP3D.ModelSino(
        phantom_model,
        phantom_N_size,
        Horiz_det,
        Vert_det,
        phantom_3D_projection_angles_rad,
        phantom_path_library,
    )
