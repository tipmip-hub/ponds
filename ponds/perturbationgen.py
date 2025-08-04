from abc import ABC, abstractmethod
import numpy as np
import warnings


# Abstract class
class PerturbationMethod(ABC):
    @abstractmethod
    def make_perturbation(
        self,
        time_size: int,
    ) -> np.ndarray:
        """
        Generates the perturbation effect for the time series.

        Parameters:
            time_size (int): The size of the time dimension for which the perturbation is generated.

        Returns:
            np.ndarray: The generated perturbation effect that shall be added to the PONDS ts.
        """
        pass


###################################################################################################
############################################# White Noise #########################################
###################################################################################################


class WhiteNoise(PerturbationMethod):
    def __init__(
        self,
        magnitude: float = 0.05,  # Magnitude of the white noise
        random_seed: int = 1,
        perturbation_time: np.ndarray[int] = None,
    ):
        """
        Initializes the WhiteNoise perturbation method.

        Parameters:
            random_seed (int): Random seed for reproducibility.
            perturbation_time (np.ndarray[int]): Indices in the time series where the perturbation is applied.
        """

        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            warnings.warn(
                "No random seed set for PONDS. Results will not be reproducible."
            )

        self.magnitude = magnitude
        self.random_seed = random_seed
        self.perturbation_time = perturbation_time

        self.param_dict = {
            "perturbation_type": WhiteNoise,
            "magnitude": magnitude,
            "random_seed": random_seed,
            "perturbation_time": perturbation_time,
        }

    def make_perturbation(
        self,
        time_size: int,
    ) -> np.ndarray:
        if self.perturbation_time is None:
            self.perturbation_time = np.arange(time_size)

        noise = np.zeros(time_size)
        noise[self.perturbation_time] = self.magnitude * np.random.normal(
            size=len(self.perturbation_time)
        )
        return noise


####################################################################################################
############################################# TREND ################################################
####################################################################################################


class Trend(PerturbationMethod):
    def __init__(
        self,
        slope: float = 0.01,  # Slope of the trend
        intercept: float = 0.0,  # Intercept of the trend at start (t=0)
        perturbation_time: np.ndarray[int] = None,
    ):
        """
        Initializes the Trend perturbation method.

        Parameters:
            slope (float): Slope of the linear trend.
            intercept (float): Intercept of the trend at the start of the time series.
            perturbation_time (np.ndarray[int]): Indices in the time series where the perturbation is applied.
        """

        self.slope = slope
        self.intercept = intercept
        if perturbation_time is not None:
            self.perturbation_time = np.array(perturbation_time)
        else:
            self.perturbation_time = None

        self.param_dict = {
            "perturbation_type": Trend,
            "slope": slope,
            "perturbation_time": perturbation_time,
        }

    def make_perturbation(
        self,
        time_size: int,
    ) -> np.ndarray:
        if self.perturbation_time is None:
            self.perturbation_time = np.arange(time_size)

        trend = np.zeros(time_size)
        trend[self.perturbation_time] = (
            self.perturbation_time * self.slope + self.intercept
        )
        return trend
