# Defines common fixtures and makes them available to all tests

import os
import numpy as np
import pytest
import cupy as cp

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: mark test as performance test")


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
