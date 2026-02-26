from abc import ABC, abstractmethod
import numpy as np

from typing import Any
from numpy.typing import NDArray

from numba import njit


# Abstract class
class ShiftMethod(ABC):
    # explicit annotation so subclasses' self.param_dict has a known type
    param_dict: dict[str, Any]

    @abstractmethod
    def __init__(
        self,
        central_time: int,
    ) -> None:
        self.central_time = central_time

    @abstractmethod
    def prepare_shift(
        self,
        time_arr: NDArray[np.float64],
        feature_name: str,
    ) -> object:
        @abstractmethod
        def make_shift(
            feature_offset: float = 0.0,
        ) -> tuple[NDArray[np.float64], int]:
            pass

        return make_shift


###################################################################################################
############################################ SIGMOID ##############################################
###################################################################################################


class SIGMOID(ShiftMethod):
    def __init__(
        self,
        central_time: int | None = None,
        shift_steepness: float | None = None,
        shift_magnitude: float | None = None,
        shift_seed: int | None = None,
    ) -> None:
        if shift_seed is not None:
            rng = np.random.default_rng(shift_seed)
        else:
            rng = np.random.default_rng()

        if shift_steepness is None:
            shift_steepness = rng.uniform(1.5, 3.0)
        if shift_magnitude is None:
            shift_magnitude = rng.uniform(0.95, 1.05)
        if central_time is None:
            central_time_rel = rng.uniform(0, 1)
            self.central_time_rel = central_time_rel

        self.shift_steepness = shift_steepness
        self.shift_magnitude = shift_magnitude
        self.central_time = central_time

        self.param_dict = {
            "shift_type": "SIGMOID",
            "central_time": central_time,
            "shift_steepness": shift_steepness,
            "shift_magnitude": shift_magnitude,
        }

    def prepare_shift(
        self,
        time_arr: NDArray[np.float64],
        feature_name: str = "central_time",
    ) -> object:
        """
        Generate a shift in the time series data using a sigmoid function.

        Args:
            time_arr (np.ndarray): Time array.
            time_offset (int): Time offset for the shift.
        Returns:
            tuple: A tuple containing the shift effect (np.ndarray) and the actual
            shift time (int; central_time +/- offset).
        """
        if self.central_time is None:
            # Calculate the actual shift time based on the relative shift time
            self.central_time = int(
                np.round(self.central_time_rel * (len(time_arr) - 1))
            )
            self.param_dict["central_time"] = self.central_time

        if not hasattr(self, feature_name):
            raise AttributeError(
                f"Feature '{feature_name}' does not exist in the shiftgen method."
            )

        original_val = getattr(self, feature_name)

        # bind the feature_name for fast access when called repeatedly
        def make_shift(
            feature_offset: int | float = 0,
        ) -> tuple[NDArray[np.float64], int]:
            shifted_value = original_val + feature_offset
            # temporary change the feature value by the offset
            if feature_name == "central_time":
                t_shift = int(shifted_value)
                shift_steepness = self.shift_steepness
                shift_magnitude = self.shift_magnitude
            elif feature_name == "shift_steepness":
                shift_steepness = shifted_value
                shift_magnitude = self.shift_magnitude
                t_shift = self.central_time
            elif feature_name == "shift_magnitude":
                shift_magnitude = shifted_value
                shift_steepness = self.shift_steepness
                t_shift = self.central_time
            else:
                raise ValueError(
                    f"Feature '{feature_name}' cannot be shifted in SIGMOID method."
                )
            # make computations with altered feature value
            # - ensure t_shift is within bounds
            t_shift = max(0, min(t_shift, len(time_arr) - 1))

            # -> Start of core function for SIGMOID shift generation
            shift_effect = _sigmoid_shift_core(
                time_arr,
                t_shift,
                shift_steepness,
                shift_magnitude,
            )
            # -> End of core function for SIGMOID shift generation

            # change the feature value back to original
            setattr(self, feature_name, original_val)  # reset to original value

            return shift_effect, t_shift

        return make_shift


@njit
def _sigmoid_shift_core(
    time_arr: NDArray[np.float64],
    t_shift: int,
    steepness: float,
    magnitude: float,
) -> NDArray[np.float64]:
    """
    Sigmoid transition function

    Args:
        time_arr (np.ndarray): Time array.
        t_shift (float): Time at which the sigmoid is centered.
        steepness (float, optional): Steepness of the sigmoid. Defaults to 1.0.
        magnitude (float): Magnitude of the sigmoid.
    Returns:
        np.ndarray: Sigmoid values for the input time array.
    """

    out = np.empty_like(time_arr)
    t0 = time_arr[t_shift]

    for i in range(time_arr.size):
        out[i] = magnitude / (1.0 + np.exp(-steepness * (time_arr[i] - t0)))

    return out


# @njit
# def _sigmoid(
#     t: NDArray[np.float64],
#     t0: float,
#     k: float = 1.0,
# ) -> NDArray[np.float64]:
#     """
#     Sigmoid transition function

#     Args:
#         t (np.ndarray): Time array.
#         t0 (float): Time at which the sigmoid is centered.
#         k (float, optional): Steepness of the sigmoid. Defaults to 1.0.
#     Returns:
#         np.ndarray: Sigmoid values for the input time array.
#     """
#     return 1 / (1 + np.exp(-k * (t - t0)))


###################################################################################################
############################################ BIMODAL ##############################################
###################################################################################################


class BIMODAL(ShiftMethod):
    def __init__(
        self,
        central_time: int | None = None,
        interval_length: int | None = None,
        duration: int | None = None,
        shift_magnitude: float | None = None,
        shift_seed: int | None = None,
    ) -> None:
        if shift_seed is not None:
            rng = np.random.default_rng(shift_seed)
        else:
            rng = np.random.default_rng()

        if central_time is None:
            central_time_rel = rng.uniform(0, 1)
            self.central_time_rel = central_time_rel
        if interval_length is None:
            interval_length = 10
        if shift_magnitude is None:
            shift_magnitude = rng.uniform(0.95, 1.05)

        self.central_time = central_time
        self.interval_length = interval_length
        self.duration = duration
        self.shift_magnitude = shift_magnitude
        # self.shift_seed = shift_seed

        self.param_dict = {
            "shift_type": "BIMODAL",
            "central_time": central_time,
            "interval_length": interval_length,
            "duration": duration,
            "shift_magnitude": shift_magnitude,
            # "shift_seed": shift_seed,
        }

    def prepare_shift(
        self,
        time_arr: NDArray[np.float64],
        feature_name: str = "central_time",
    ) -> tuple[NDArray[np.float64], int]:
        """
        Generate a bimodal shift in the time series data.

        Args:
            time_arr (np.ndarray): Time array.
            time_offset (int): Time offset for the shift.
        Returns:
            tuple: A tuple containing the shift effect (np.ndarray) and the actual
            shift time (int; central_time +/- offset).
        """

        if self.interval_length == 0:
            raise ValueError("interval_length cannot be zero for BIMODAL shift.")

        if self.central_time is None:
            # Calculate the actual shift time based on the relative shift time
            self.central_time = int(
                np.round(self.central_time_rel * (len(time_arr) - 1))
            )
            self.param_dict["central_time"] = self.central_time

        if not hasattr(self, feature_name):
            raise AttributeError(
                f"Feature '{feature_name}' does not exist in the shiftgen method."
            )

        # bind the feature_name for fast access when called repeatedly
        def make_shift(
            feature_offset: int | float = 0,
        ) -> tuple[NDArray[np.float64], int]:
            original_val = getattr(self, feature_name)
            # temporary change the feature value by the offset
            setattr(self, feature_name, original_val + feature_offset)
            # make computations with altered feature value
            # - ensure t_shift is within bounds
            t_shift = self.central_time
            t_shift = max(0, min(t_shift, len(time_arr) - 1))

            # -> Start of core function for BIMODAL shift generation
            shift_effect = np.zeros_like(time_arr, dtype=np.float64)
            final_time = len(time_arr)
            start_time = 0

            # bootstrap loop
            start = int(self.central_time)
            j = 0

            if self.duration is not None:
                if self.interval_length > 0:
                    final_time = self.central_time + self.duration - 1
                else:
                    start_time = self.central_time - self.duration

            while not (start > final_time) and not (start <= start_time):
                if self.interval_length > 0:  # Forward
                    shift_effect[start + 1 :] = self.shift_magnitude * abs(j % 2 - 1)
                else:  # Backward
                    shift_effect[:start] = self.shift_magnitude * abs(j % 2 - 1)

                j += 1
                start = int(self.central_time + j * self.interval_length)
            # -> End of core function for BIMODAL shift generation

            # change the feature value back to original
            setattr(self, feature_name, original_val)  # reset to original value

            return shift_effect, t_shift

        return make_shift
