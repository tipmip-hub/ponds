from abc import ABC, abstractmethod
import numpy as np


# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def make_shift(self,
                   ponds,
                   mask,
                   shift_time,
    ) -> tuple[np.ndarray, dict]:
        
        pass
    

###################################################################################################
###################################################################################################
###################################################################################################

class SIGMOID(ShiftsMethod):
    def __init__(self,
                 ponds,
                 mask: np.ndarray,
                 shift_time: int,
                 random_seed: int = None,
                 ):
        
        self.ponds = ponds
        self.mask = mask
        self.shift_time = shift_time
        self.random_seed = random_seed

    def make_shift(self,
                   sigma_time: float = None,
                   shift_steepness: float = None,
                   shift_magnitude: float = None,
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
        
        shift_id = 0  # id for sigmoid shift

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if shift_steepness is None: shift_steepness = np.random.uniform(1.5, 3.0, size=1)
        if shift_magnitude is None: shift_magnitude = np.random.uniform(0.95, 1.05, size=1)
        if sigma_time is None: sigma_time = np.ones(1)  # Sharp shift

        for i in range(len(self.ponds.lats)):   
            for j in range(len(self.ponds.lons)):
                if self.mask[i,j]:
                    t_shift = int(np.round(self.shift_time +
                                       np.random.uniform(-sigma_time,
                                                         sigma_time, size=1)[0]))
                    t_shift = max(0, min(t_shift, len(self.ponds.time) - 1))

                    shift_effect = (
                        shift_magnitude
                        * sigmoid(self.ponds.time, self.ponds.time[t_shift], shift_steepness)
                    )
                    # add sigmoid shift effect to the data in single grid cell, all time steps at once
                    self.ponds.data.values[:, i, j] += shift_effect
        dict_return = {"shift_id" : shift_id,
                       "shift_steepness" : shift_steepness,
                       "shift_magnitude" : shift_magnitude,
                       "sigma_time" : sigma_time}

        return dict_return
    
def sigmoid(t,
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