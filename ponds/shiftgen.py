from abc import ABC, abstractmethod
import numpy as np


# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def make_shift(
        self,
        # ponds,
        # center_lat,
        # center_lon,
        # sigma_lat,
        # sigma_lon,
    ) -> tuple[np.ndarray, dict]:
        pass


###################################################################################################
###################################################################################################
###################################################################################################


class SIGMOID(ShiftsMethod):
    def __init__(self, ponds):
        """
        ...
        """
        self.ponds = ponds

    def make_shift(
        self,
        mask,
        shift_time,
        shift_steepness: float = None,
        shift_magnitude: float = None,
        sigma_time: float = None,
    ):
        shift_id = 0  # id for sigmoid shift

        if not (shift_steepness):
            shift_steepness = np.random.uniform(1.5, 3.0, size=1)
        if not (shift_magnitude):
            shift_magnitude = np.random.uniform(0.95, 1.05, size=1)
        if not (sigma_time):
            sigma_time = np.ones(1)  # Sharp shift

        for i in range(len(self.ponds.lats)):
            for j in range(len(self.ponds.lons)):
                if mask[i, j]:
                    t_shift = int(shift_time + np.random.normal(0, sigma_time))
                    t_shift = max(0, min(t_shift, len(self.ponds.time) - 1))

                    shift_effect = shift_magnitude * self.sigmoid(
                        self.ponds.time, self.ponds.time[t_shift], shift_steepness
                    )
                    # add sigmoid shift effect to the data in single grid cell, all time steps at once
                    self.ponds.data.values[:, i, j] += shift_effect
        dict_return = {
            "shift_id": shift_id,
            "shift_steepness": shift_steepness,
            "shift_magnitude": shift_magnitude,
            "sigma_time": sigma_time,
        }

        return self.ponds.data, dict_return

    def sigmoid(self, t, t0, k: float = 1.0) -> np.ndarray:
        """
        Sigmoid transition function
        """
        return 1 / (1 + np.exp(-k * (t - t0)))
