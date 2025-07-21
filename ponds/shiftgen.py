from abc import ABC, abstractmethod
import numpy as np


# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def make_shift(
        self,
        ponds,
        mask,
        shift_time,
    ) -> tuple[np.ndarray, dict]:
        pass


###################################################################################################
###################################################################################################
###################################################################################################


class SIGMOID(ShiftsMethod):
    def __init__(
        self,
        random_seed: int = None,
        sigma_time: float = None,
        shift_steepness: float = None,
        shift_magnitude: float = None,
    ):
        if random_seed is not None:
            np.random.seed(random_seed)

        if shift_steepness is None:
            shift_steepness = np.random.uniform(1.5, 3.0)
        if shift_magnitude is None:
            shift_magnitude = np.random.uniform(0.95, 1.05)
        if sigma_time is None:
            sigma_time = 1.0  # Sharp shift

        self.sigma_time = sigma_time
        self.shift_steepness = shift_steepness
        self.shift_magnitude = shift_magnitude

        self.param_dict = {
            "type": "SIGMOID",
            "shift_steepness": shift_steepness,
            "shift_magnitude": shift_magnitude,
            "sigma_time": sigma_time,
        }

    def make_shift(
        self,
        time,
        shift_time,
    ):
        """
        Generate a shift in the time series data using a sigmoid function.

        Args:
            sigma_time (float, optional): Standard deviation for the time shift. Defaults to 1.0 for a sharp shift.
            shift_steepness (float, optional): Steepness of the sigmoid function. Defaults to a random value between 1.5 and 3.0.
            shift_magnitude (float, optional): Magnitude of the shift. Defaults to a random value between 0.95 and 1.05.
        Returns:
            tuple: A tuple containing the modified data array and a dictionary with shift parameters.
        The dictionary includes:
            - "shift_id": Identifier for the shift (always 0 for sigmoid).
            - "shift_steepness": Steepness of the sigmoid function.
            - "shift_magnitude": Magnitude of the shift.
            - "sigma_time": Standard deviation for the time shift.
        """

        t_shift = int(
            np.round(
                shift_time
                + np.random.uniform(-self.sigma_time, self.sigma_time, size=1)[0]
            )
        )
        # Ensure t_shift is within bounds
        t_shift = max(0, min(t_shift, len(time) - 1))
        shift_effect = self.shift_magnitude * sigmoid(
            time, time[t_shift], self.shift_steepness
        )

        return shift_effect, t_shift


###################################################################################################
################################### Helper Functions ##############################################
###################################################################################################


def sigmoid(t, t0, k: float = 1.0) -> np.ndarray:
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
