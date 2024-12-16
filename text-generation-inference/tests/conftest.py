import os

import pytest

from optimum.tpu import jetstream_pt_available


# See https://stackoverflow.com/a/61193490/217945 for run_slow
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="function")
def quantization_jetstream_int8():
    # Setup
    old_environ = dict(os.environ)
    os.environ["QUANTIZATION"] = "jetstream_int8"
    yield
    # Clean up
    os.environ.clear()
    os.environ.update(old_environ)


def pytest_runtest_setup(item):
    marker_names = [marker.name for marker in item.iter_markers()]
    jetstream_pt_enabled = jetstream_pt_available()
    # Skip tests that require torch xla but not jetstream
    if "torch_xla" in marker_names and "jetstream" not in marker_names:
        if jetstream_pt_enabled:
            pytest.skip("Jetstream is enabled: xla test will be skipped")
    elif "jetstream" in marker_names and not jetstream_pt_enabled:
        pytest.skip("Test requires Jetstream PyTorch to be enabled")
