from abc import ABC, abstractmethod
import numpy as np


# Abstract class
class BackgroundMethod(ABC):
    @abstractmethod
    def make_background(
        self,
        time_size: int,
    ) -> np.ndarray:
        """
        Generates the background effect for the time series.

        Parameters:
            time_size (int): The size of the time dimension for which the background is generated.

        Returns:
            np.ndarray: The generated background effect that shall be added to the PONDS ts.
        """
        pass


###################################################################################################
############################################# White Noise #########################################
###################################################################################################


class WhiteNoise(BackgroundMethod):
    def __init__(
        self,
        magnitude: float = 0.05,  # Magnitude of the white noise
        random_seed: int = 1,
        background_time: np.ndarray[int] = None,
    ):
        """
        Initializes the WhiteNoise background method.

        Parameters:
            random_seed (int): Random seed for reproducibility.
            background_time (np.ndarray[int]): Indices in the time series where the background is applied.
        """
        self.magnitude = magnitude
        self.random_seed = random_seed
        self.background_time = background_time

        self.param_dict = {
            "background_type": WhiteNoise,
            "magnitude": magnitude,
            "random_seed": random_seed,
            "background_time": background_time,
        }

    def make_background(
        self,
        time_size: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if self.background_time is None:
            self.background_time = np.arange(time_size)

        noise = np.zeros(time_size)
        noise[self.background_time] = self.magnitude * rng.normal(
            size=len(self.background_time)
        )
        return noise


####################################################################################################
############################################# TREND ################################################
####################################################################################################


class Trend(BackgroundMethod):
    def __init__(
        self,
        slope: float = 0.01,  # Slope of the trend
        intercept: float = 0.0,  # Intercept of the trend at start (t=0)
        background_time: np.ndarray[int] = None,
    ):
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

        self.param_dict = {
            "background_type": Trend,
            "slope": slope,
            "background_time": background_time,
        }

    def make_background(
        self,
        time_size: int,
    ) -> np.ndarray:
        if self.background_time is None:
            self.background_time = np.arange(time_size)

        trend = np.zeros(time_size)
        trend[self.background_time] = (
            self.background_time * self.slope + self.intercept
        )
        return trend
