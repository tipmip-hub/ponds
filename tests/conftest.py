import numpy as np
import pytest


class MockBlob:
    """
    Always returns a mask with a single True value at (0, 0).
    This is a mock implementation for testing purposes.
    It simulates the behavior of a blob generator that creates a mask
    for a pond.
    """

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        self.param_dict = {"type": "MOCKBLOB"}

    def make_blob(self, lats, lons):
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        mask[0, 0] = True
        return mask


class MockShift:
    """
    Always returns a shift dictionary with a fixed shift_id of -1.
    This is a mock implementation for testing purposes.
    It simulates the behavior of a shift generator that creates a shift
    for a pond."""

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        self.param_dict = {"type": "MOCKSHIFT"}

    def make_shift(
        self,
        time,
        # shift_time,
    ):
        t_shift = len(time) // 2
        shift_effect = np.zeros_like(time, dtype=float)

        return shift_effect, t_shift


@pytest.fixture
def mock_blob():
    return MockBlob


@pytest.fixture
def mock_shift():
    return MockShift
