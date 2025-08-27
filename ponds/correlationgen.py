from abc import ABC, abstractmethod
from turtle import pd
import numpy as np
import warnings

# for Gauss
from scipy.stats import multivariate_normal


# Abstract class
class CorrelationMethod(ABC):
    @abstractmethod
    def make_time_offsets(
        self,
        time,
    ) -> np.ndarray:
        """
        Generate time offsets for the time of the shift. An offset value for each grid cell of the
        spatial grid is created at once.
        """
        pass



###################################################################################################
############################################# Noise ###############################################
###################################################################################################

class Noise(CorrelationMethod):

    def __init__(
        self,
        max_offset: int = 5,
        random_seed: int = None,
    ):
        self.max_offset = max_offset
        self.random_seed = random_seed

        self.param_dict = {
            "correlation_type": Noise,
            "max_offset": max_offset,
            "random_seed": random_seed,
        }

    def make_time_offsets(
        self,
        shape_mask,
    ) -> np.ndarray:
        
        if self.random_seed is not None:
            rng = np.random.RandomState(self.random_seed)
        else:
            rng = np.random
            warnings.warn(
                "No random seed set for PONDS Correlation Method. Results will not be reproducible."
            )

        offsets = rng.uniform(
            -self.max_offset,
            self.max_offset,
            size=shape_mask.shape
        ).astype(int)

        return offsets


###################################################################################################
############################################# Gauss ###############################################
###################################################################################################

class Gauss(CorrelationMethod):
    """
    Creates a Gauss-like correlation pattern. For a positive value `max_offset`, the central shift
    time occurs at the peak of the 2D gauss curve. All values around it are pushed towards the past.
    """
    def __init__(
            self,
            center_lat=None,
            center_lon=None,
            sigma_lat=10,
            sigma_lon=10,
            max_offset=10,
            #random_seed=None,
    ):

        #if random_seed is not None:
        #    rng = np.random.RandomState(random_seed)
        #else:
        #    rng = np.random

        #if center_lat is None:
        #    center_lat = rng.uniform(-80, 80)
        #if center_lon is None:
        #    center_lon = rng.uniform(-180, 180)

        self.center_lat = center_lat
        self.center_lon = center_lon
        self.sigma_lat = sigma_lat
        self.sigma_lon = sigma_lon
        self.max_offset = max_offset

        self.param_dict = {
            "correlation_type": Gauss,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "sigma_lat": sigma_lat,
            "sigma_lon": sigma_lon,
            "max_offset": max_offset,
        }

    def make_time_offsets(
            self,
            shape_mask,
    ) -> np.ndarray:

        # Create lat/lon grid
        lats = np.linspace(-90, 90, shape_mask.shape[0])
        lons = np.linspace(-180, 180, shape_mask.shape[1])

        extended_lons = np.concatenate([lons - 360, lons, lons + 360])

        # Create grids
        lat_grid, extended_lon_grid = np.meshgrid(lats, extended_lons, indexing="ij")
        pos_extended = np.dstack((lat_grid, extended_lon_grid))

        # Calculate Gaussian over the whole extended grid
        cov = np.array([[self.sigma_lat**2, 0], [0, self.sigma_lon**2]])
        rv = multivariate_normal(mean=[self.center_lat, self.center_lon], cov=cov)
        extended_gauss = rv.pdf(pos_extended)
        extended_gauss_scaled = extended_gauss / np.max(extended_gauss) * self.max_offset 

        # Apply shift to data
        offsets = np.zeros(shape_mask.shape, dtype=int)
        for i in range(len(lats)):
            for j in range(len(lons)):
                # Sum contributions from periodic longitudes
                gauss_contribution = np.sum(
                    [extended_gauss_scaled[i, j + k * len(lons)] for k in range(3)],
                )
                offsets[i, j] = gauss_contribution - self.max_offset

        return offsets
