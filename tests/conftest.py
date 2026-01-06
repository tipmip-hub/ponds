import numpy as np
import pytest
from typing import Optional, Type
from ponds.shapegen import ShapeMethod
from ponds.shiftgen import ShiftMethod
from ponds.backgroundgen import BackgroundMethod
from ponds.correlationgen import CorrelationMethod



class MockShape(ShapeMethod):
    """
    Always returns a mask with a single True value at (0, 0).
    This is a mock implementation for testing purposes.
    It simulates the behavior of a shape generator that creates a mask
    for a pond.
    """

    def __init__(
            self,
            random_seed: Optional[int] = None,
        ) -> None:
        self.random_seed = random_seed

        self.param_dict = {"shape_type": MockShape}

    def make_shape(
            self,
            lats: Optional[np.ndarray] = np.array([0,1]),
            lons: Optional[np.ndarray] = np.array([0,1]),
        ) -> np.ndarray:
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        mask[0, 0] = True
        return mask


class MockBackground(BackgroundMethod):
    """
    This is a mock implementation for testing purposes.
    It simulates the behavior of a background generator that creates
    a background for a pond.
    """

    def __init__(
            self,
        ) -> None:
        self.param_dict = {"background_type": MockBackground}

    def make_background(
            self,
            time_size: int,
        ) -> np.ndarray:
        return np.zeros(time_size)  # Fixed background value


class MockShift(ShiftMethod):
    """
    This is a mock implementation for testing purposes.
    It simulates the behavior of a shift generator that creates a shift
    for a pond."""

    def __init__(
        self,
        central_time: int = 0,
        random_seed: Optional[int] = None,
    ) -> None:
        self.random_seed = random_seed
        self.central_time = central_time

        self.param_dict = {"shift_type": MockShift}

    def prepare_shift(
        self,
        time_arr: np.ndarray,
        feature_name: str = "central_time",
        #time_offset: int = 0,
    ) -> object:
        
        def make_shift(
                feature_offset: int = 0,
                #time_offset: int = 0,
        ) -> tuple[np.ndarray, int]:
            t_shift = len(time_arr) // 2 + feature_offset
            shift_effect = np.zeros_like(time_arr, dtype=float)

            return shift_effect, t_shift
        
        return make_shift

class MockCorrelation(CorrelationMethod):
    """
    This is a mock implementation for testing purposes.
    """

    def __init__(
        self,
        feature_name: str = "central_time",
    ) -> None:
        self.feature_name = feature_name
        self.param_dict = {
            "correlation_type": MockCorrelation,
        }

    def make_feature_offsets(
        self,
        shape_mask: np.ndarray,
    ) -> np.ndarray:
        offsets = np.zeros(np.shape(shape_mask), dtype=int)
        offsets[0, 0] = 2  # Fixed offset for testing purposes

        return offsets


@pytest.fixture
def mock_shape() -> Type[MockShape]:
    return MockShape


@pytest.fixture
def mock_background() -> Type[MockBackground]:
    return MockBackground


@pytest.fixture
def mock_shift() -> Type[MockShift]:
    return MockShift


@pytest.fixture
def mock_corr() -> Type[MockCorrelation]:
    return MockCorrelation
