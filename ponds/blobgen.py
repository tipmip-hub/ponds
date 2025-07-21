from abc import ABC, abstractmethod
import numpy as np

# for gauss
from scipy.stats import multivariate_normal


# Abstract class for shifts detection methods
class BlobsMethod(ABC):
    @abstractmethod
    def make_blob(
        self,
        **kwargs,
    ) -> np.ndarray:
        pass


###################################################################################################
###################################################################################################
###################################################################################################


class GAUSS(BlobsMethod):
    def __init__(
        self,
        random_seed=None,  # random seed for reproducibility
        center_lat=None,  # center latitude of the blob
        center_lon=None,  # center longitude of the blob
        sigma_lat=None,  # standard deviation in latitude
        sigma_lon=None,  # standard deviation in longitude
    ):
        if random_seed is not None:
            np.random.seed(random_seed)

        if center_lat is None:
            center_lat = np.random.uniform(-80, 80)
        if center_lon is None:
            center_lon = np.random.uniform(-180, 180)
        if sigma_lat is None:
            sigma_lat = np.random.uniform(10, 20)
        if sigma_lon is None:
            sigma_lon = np.random.uniform(15, 40)
        
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.sigma_lat = sigma_lat
        self.sigma_lon = sigma_lon

        self.param_dict = {
            "type": "GAUSS",
            "center_lat": center_lat,
            "center_lon": center_lon,
            "sigma_lat": sigma_lat,
            "sigma_lon": sigma_lon,
        }

    def make_blob(
        self,
        lats,
        lons,
    ) -> np.ndarray:
        """
        Generate a spatial blob of parameters on the grid in shape of a 2D Gauss, given positional and
        spatial shape parameters. Returns a mask of the same shape as the grid, where values above 0.5
        indicate presence of the blob.

        Returns:
            blob_mask (np.ndarray): A 2D array of shape (lat_size, lon_size).
        """
        #if self.random_seed is not None:
        #    np.random.seed(self.random_seed)  # set random seed for reproducibility
        #center_lat = np.random.uniform(-80, 80)
        #center_lon = np.random.uniform(-180, 180)
        #sigma_lat = np.random.uniform(10, 20)
        #sigma_lon = np.random.uniform(15, 40)

        # Handle longitude periodicity to ensure continuity at the dateline (global dataset!)
        extended_lons = np.concatenate([lons - 360, lons, lons + 360])

        # Create grids
        lat_grid, extended_lon_grid = np.meshgrid(lats, extended_lons, indexing="ij")
        pos_extended = np.dstack((lat_grid, extended_lon_grid))

        # Calculate Gaussian blob
        cov = np.array([[self.sigma_lat**2, 0], [0, self.sigma_lon**2]])
        # - create multivariate normal distribution
        rv = multivariate_normal(mean=[self.center_lat, self.center_lon], cov=cov)
        # - compute probability density function (PDF) over the extended grid
        extended_blob = rv.pdf(pos_extended)
        blob_normalized = extended_blob / np.max(extended_blob)

        # add up contributions from periodic longitudes, blob_contribution is of shape (lat_size, lon_size)
        blob_contribution = np.sum(
            blob_normalized.reshape(len(lats), 3, len(lons)).transpose(1, 0, 2),
            axis=0,
        ).reshape(len(lats), len(lons))
        # create mask where blob_contribution is above 0.5
        blob_mask = np.where(blob_contribution > 0.5, True, False)

        return blob_mask


###################################################################################################
