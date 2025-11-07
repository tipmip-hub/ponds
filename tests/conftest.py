import numpy as np
import pytest


class MockShape:
    """
    Always returns a mask with a single True value at (0, 0).
    This is a mock implementation for testing purposes.
    It simulates the behavior of a shape generator that creates a mask
    for a pond.
    """

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        self.param_dict = {"shape_type": MockShape}

    def make_shape(self, lats, lons):
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        mask[0, 0] = True
        return mask


class MockBackground:
    """
    This is a mock implementation for testing purposes.
    It simulates the behavior of a background generator that creates
    a background for a pond.
    """

    def __init__(self):
        self.param_dict = {"background_type": MockBackground}

    def make_background(self, time_size):
        return np.zeros(time_size)  # Fixed background value


class MockShift:
    """
    This is a mock implementation for testing purposes.
    It simulates the behavior of a shift generator that creates a shift
    for a pond."""

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        self.param_dict = {"shift_type": MockShift}

    def make_shift(
        self,
        time_arr,
        time_offset: int = 0,
    ):
        t_shift = len(time_arr) // 2 + time_offset
        shift_effect = np.zeros_like(time_arr, dtype=float)

        return shift_effect, t_shift


class MockCorrelation:
    """
    This is a mock implementation for testing purposes.
    """

    def __init__(
        self,
    ):
        self.param_dict = {
            "correlation_type": MockCorrelation,
        }

    def make_time_offsets(
        self,
        shape_mask,
    ) -> np.ndarray:
        offsets = np.zeros(shape_mask.shape, dtype=int)

        return offsets


class MockDynamicalSystem:
    """
    Always returns an array that is the input parameter array multiplied by 2.
    This is a mock implementation for testing purposes.
    It simulates the behavior of a nonlinear system that can be integrated based 
    on an input parameter array.
    """
    def __init__(self, forced_parameter='param'):
        self.forced_parameter = forced_parameter
        self.state_dict = {
            'param': {
                'state1': 42,
            }
        }

    def integrate(self, parameter_array, dt, sampling_factor):
        return 2*parameter_array
    
    def get_parameter_value(self, state='state1'):
        return self.state_dict[state]


@pytest.fixture
def mock_shape():
    return MockShape


@pytest.fixture
def mock_background():
    return MockBackground


@pytest.fixture
def mock_shift():
    return MockShift


@pytest.fixture
def mock_corr():
    return MockCorrelation


@pytest.fixture
def mock_dynamical_system():
    return MockDynamicalSystem
