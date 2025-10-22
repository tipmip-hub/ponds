from abc import ABC, abstractmethod
import numpy as np
import warnings

# for Gauss
from scipy.stats import multivariate_normal


# Abstract class
class CorrelationMethod(ABC):
    @abstractmethod
    def make_time_offsets(
        self,
        shape_mask,
    ) -> np.ndarray:
        """
        Generate time offsets for the time of the shift. An offset value for each grid cell of the
        spatial grid is created at once.
        """
        pass


###################################################################################################
############################################# Uniform #############################################
###################################################################################################


class NoCorr(CorrelationMethod):
    """
    Default correlation method that does not apply any correlation. All offsets are zero.
    """

    def __init__(
        self,
    ):
        self.param_dict = {
            "correlation_type": NoCorr,
        }

    def make_time_offsets(
        self,
        shape_mask,
    ) -> np.ndarray:
        offsets = np.zeros(shape_mask.shape, dtype=int)

        return offsets


###################################################################################################
############################################# Noise ###############################################
###################################################################################################


class Noise(CorrelationMethod):
    def __init__(
        self,
        max_offset: int = 5,
        corr_seed: int = None,
    ) -> None:
        self.max_offset = max_offset
        self.corr_seed = corr_seed

        self.param_dict = {
            "correlation_type": Noise,
            "max_offset": max_offset,
            "corr_seed": corr_seed,
        }

    def make_time_offsets(
        self,
        shape_mask,
    ) -> np.ndarray:
        if self.corr_seed is not None:
            rng = np.random.RandomState(self.corr_seed)
        else:
            rng = np.random
            warnings.warn(
                "No random seed set for PONDS Correlation Method. Results will not be reproducible."
            )

        offsets = rng.uniform(
            -self.max_offset, self.max_offset, size=shape_mask.shape
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
        corr_center_lat=None,
        corr_center_lon=None,
        sigma_lat=10,
        sigma_lon=10,
        max_offset=10,
    ) -> None:
        self.corr_center_lat = corr_center_lat
        self.corr_center_lon = corr_center_lon
        self.sigma_lat = sigma_lat
        self.sigma_lon = sigma_lon
        self.max_offset = max_offset

        self.param_dict = {
            "correlation_type": Gauss,
            "corr_center_lat": corr_center_lat,
            "corr_center_lon": corr_center_lon,
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
        rv = multivariate_normal(
            mean=[self.corr_center_lat, self.corr_center_lon], cov=cov
        )
        extended_gauss = rv.pdf(pos_extended)
        extended_gauss_scaled = (
            extended_gauss / np.max(extended_gauss) * self.max_offset
        )

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


###################################################################################################
############################################# Custom ##############################################
###################################################################################################

class Custom(CorrelationMethod):
    """
    Custom correlation method that allows for user-defined shift patterns.
    """

    def __init__(
        self,
        offset_mask: np.ndarray,
    ) -> None:
        self.offset_mask = offset_mask
        self.param_dict = {
            "correlation_type": Custom,
        }

    def make_time_offsets(
        self,
        shape_mask,
    ) -> np.ndarray:
        # check if the offset_mask is of the correct shape
        if shape_mask.shape != self.offset_mask.shape:
            raise ValueError(
                f"Offset mask shape {self.offset_mask.shape} does not match grid shape {shape_mask.shape}."
            )
        # check if the offset_mask only contains integers
        if not np.issubdtype(self.offset_mask.dtype, np.integer):
            raise ValueError("Offset mask must be all integer values.")

        return self.offset_mask


###################################################################################################
############################################ Cascading ############################################
###################################################################################################

class Cascading(CorrelationMethod):
    """
    
    """

    ...