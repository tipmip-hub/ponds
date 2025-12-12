from abc import ABC, abstractmethod
import numpy as np

from typing import Optional, Any, Dict
from numpy.typing import NDArray


# Abstract class
class BackgroundMethod(ABC):
    # explicit annotation so subclasses' self.param_dict has a known type
    param_dict: dict[str, Any]

    @abstractmethod
    def make_background(
        self,
        time_size: int,
    ) -> NDArray[np.float64]:
        """
        Generates the background effect for the time series.

        Parameters:
            time_size (int): The size of the time dimension for which the background is generated.

        Returns:
            NDArray[np.float64]: The generated background effect that shall be added to the PONDS ts.
        """
        pass


###################################################################################################
############################################# White Noise #########################################
###################################################################################################

class WhiteNoise(BackgroundMethod):
    def __init__(
        self,
        magnitude: float = 0.05,  # Magnitude of the white noise
        background_seed: int = 1,  # Random seed for reproducibility
        background_time: Optional[NDArray[np.int_]] = None,
    ) -> None:
        """
        Initializes the WhiteNoise background method.

        Parameters:
            background_seed (int): Random seed for reproducibility.
            background_time (np.ndarray[int]): Indices in the time series where the background is applied.
        """
        self.magnitude = magnitude
        self.background_seed = background_seed
        self.background_time = background_time

        self.param_dict: Dict[str, Any] = {
            "background_type": WhiteNoise,
            "magnitude": magnitude,
            "background_seed": background_seed,
            "background_time": background_time,
        }

    def make_background(
        self,
        time_size: int,
    ) -> NDArray[np.float64]:

        if self.background_time is None:
            self.background_time = np.arange(time_size)

        noise = np.zeros(time_size)
        noise[self.background_time] = self.magnitude * self.rng.uniform(low=-1, high=1,#normal(
            size=len(self.background_time)
        )
        return noise

###################################################################################################
############################################## Red Noise ##########################################
###################################################################################################

class RedNoise(BackgroundMethod):
    def __init__(
        self,
        magnitude: float = 0.05,    # Magnitude of the white noise
        ac_time: float = 1.0,       # Autocorrelation time
        dt: float = 1.0,            # Time step size
        background_seed: int = 1,   # Random seed for reproducibility
        background_time: Optional[NDArray[np.int_]] = None,
    ) -> None:
        """
        Initializes the WhiteNoise background method.

        Parameters:
            magnitude (float): Magnitude of the red noise.
            AC_time (float): Autocorrelation time as the e-folding time scale.
            background_seed (int): Random seed for reproducibility.
            background_time (np.ndarray[int]): Indices in the time series where the background is applied.
        """
        self.magnitude = magnitude
        self.ac_time = ac_time
        self.dt = dt
        self.background_seed = background_seed
        self.background_time = background_time

        self.param_dict: Dict[str, Any] = {
            "background_type": RedNoise,
            "magnitude": magnitude,
            "ac_time": ac_time,
            "dt": dt,
            "background_seed": background_seed,
            "background_time": background_time,
        }

    def make_background(
        self,
        time_size: int,
    ) -> NDArray[np.float64]:

        if self.background_time is None:
            self.background_time = np.arange(time_size)

        a = _lag1(self.ac_time, dt=self.dt)     # lag-1 autocorrelation coefficient
        b = (1 - a**2)**0.5     # coefficient for white noise term

        signal = np.random.uniform(low=-1, high=1, size=time_size)
        for i in range(1, time_size):
            signal[i] += a * signal[i - 1] + b * signal[i]

        return signal * self.magnitude
    
def _lag1(
        Te: float,
        dt: float = 1.0,
    ) -> float:
    """
    Lag-1 autocorrelation coefficient for an exponential decay process.

    Indicates after how many dt time steps the correlation has decayed to 1/e.

    Parameters
    ----------
    Te : float
        E-folding time (time for the correlation to decay to 1/e).
    dt : float, optional
        Time step size. Default is 1.
    Returns
    -------
    lag1 : float
        Lag-1 autocorrelation coefficient.
    """

    if Te < 0:
        raise ValueError("Te must be positive or zero.")
    if Te == 0:
        return 0.0
    
    lag1 = np.exp(-dt / Te)
    return lag1


####################################################################################################
############################################# TREND ################################################
####################################################################################################

class Trend(BackgroundMethod):
    def __init__(
        self,
        slope: float = 0.01,  # Slope of the trend
        intercept: float = 0.0,  # Intercept of the trend at start (t=0)
        background_time: Optional[NDArray[np.int_]] = None,
    ) -> None:
        """
        Initializes the Trend background method.

        Parameters:
            slope (float): Slope of the linear trend.
            intercept (float): Intercept of the trend at the start of the time series.
            background_time (np.ndarray[int]): Indices in the time series where the background is applied.
        """

        self.slope = slope
        self.intercept = intercept
        if background_time is not None:
            self.background_time = np.array(background_time)
        else:
            self.background_time = None

        self.param_dict: Dict[str, Any] = {
            "background_type": Trend,
            "slope": slope,
            "intercept": intercept,
            "background_time": background_time,
        }

    def make_background(
        self,
        time_size: int,
    ) -> NDArray[np.float64]:
        if self.background_time is None:
            self.background_time = np.arange(time_size)

        trend = np.zeros(time_size, dtype=np.float64)
        trend[self.background_time] = self.background_time * self.slope + self.intercept
        return trend
