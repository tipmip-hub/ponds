from abc import ABC, abstractmethod
import numpy as np


# Abstract class
class ShiftMethod(ABC):
    @abstractmethod
    def __init__(
        self,
        central_time: int,
    ) -> None:
        self.central_time = central_time

    @abstractmethod
    def make_shift(
        self,
        time_arr,
        time_offset,
    ) -> tuple[np.ndarray, dict]:
        pass


###################################################################################################
###################################################################################################
###################################################################################################


class SIGMOID(ShiftMethod):
    def __init__(
        self,
        central_time: int = None,
        shift_steepness: float = None,
        shift_magnitude: float = None,
        shift_seed: int = None,
    ) -> None:
        if shift_seed is not None:
            rng = np.random.RandomState(shift_seed)
        else:
            rng = np.random

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
        self.shift_seed = shift_seed

        self.param_dict = {
            "shift_type": SIGMOID,
            "central_time": central_time,
            "shift_steepness": shift_steepness,
            "shift_magnitude": shift_magnitude,
            "shift_seed": shift_seed,
        }

    def make_shift(
        self,
        time_arr,
        time_offset: int = 0,
    ) -> tuple[np.ndarray, int]:
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

        # get actual shift time for this cell -> shift_time +/- noise
        t_shift = self.central_time + time_offset

        # Ensure t_shift is within bounds
        t_shift = max(0, min(t_shift, len(time_arr) - 1))
        shift_effect = self.shift_magnitude * sigmoid(
            time_arr, time_arr[t_shift], self.shift_steepness
        )

        return shift_effect, t_shift


###################################################################################################
################################### Helper Functions ##############################################
###################################################################################################


def sigmoid(
        t,
        t0,
        k: float = 1.0
) -> np.ndarray:
    """
    Sigmoid transition function

    Args:
        t (np.ndarray): Time array.
        t0 (float): Time at which the sigmoid is centered.
        k (float, optional): Steepness of the sigmoid. Defaults to 1.0.
    Returns:
        np.ndarray: Sigmoid values for the input time array.
    """
    return 1 / (1 + np.exp(-k * (t - t0)))
