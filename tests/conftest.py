import numpy as np
import pytest


class MockBlob:
    """
    Always returns a mask with a single True value at (0, 0).
    This is a mock implementation for testing purposes.
    It simulates the behavior of a blob generator that creates a mask
    for a pond.
    """

    def __init__(self, ponds, random_seed):
        self.ponds = ponds
        self.random_seed = random_seed

    def make_blob(self):
        mask = np.zeros((self.ponds.lat_size, self.ponds.lon_size), dtype=bool)
        mask[0, 0] = True
        return mask


class MockShift:
    """
    Always returns a shift dictionary with a fixed shift_id of -1.
    This is a mock implementation for testing purposes.
    It simulates the behavior of a shift generator that creates a shift
    for a pond."""

    def __init__(self, ponds, mask, shift_time, random_seed):
        self.ponds = ponds
        self.mask = mask
        self.shift_time = shift_time
        self.random_seed = random_seed

    def make_shift(
        self,
    ):
        shift_effect = np.zeros(self.ponds.time_size)
        shift_id = -1  # Fixed shift_id for testing purposes
        return shift_effect, shift_id


@pytest.fixture
def mock_blob():
    return MockBlob


@pytest.fixture
def mock_shift():
    return MockShift
