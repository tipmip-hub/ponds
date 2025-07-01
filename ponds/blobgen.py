from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
import warnings

# for gauss
from scipy.stats import multivariate_normal


# Abstract class for shifts detection methods
class BlobsMethod(ABC):
    @abstractmethod
    def make_blob(self,
                  **kwargs,
    ) -> np.ndarray:
        pass
    

###################################################################################################
###################################################################################################
###################################################################################################

class GAUSS(BlobsMethod):

    def __init__(self,                       # kind of parameter:
                 ponds,
        ):
        self.ponds = ponds
    
    def make_blob(self,
                  ) -> np.ndarray:
        
        """
        Generate a spatial blob of parameters on the grid in shape of a 2D Gauss, given positional and
        spatial shape parameters. Returns a mask of the same shape as the grid, where values above 0.5
        indicate presence of the blob.

        Returns:
            blob_mask (np.ndarray): A 2D array of shape (lat_size, lon_size).
        """

        center_lat = np.random.uniform(-80, 80)
        center_lon = np.random.uniform(-180, 180)
        sigma_lat = np.random.uniform(10, 20)
        sigma_lon = np.random.uniform(15, 40)

        # Handle longitude periodicity to ensure continuity at the dateline (global dataset!)
        extended_lons = np.concatenate([self.ponds.lons - 360, self.ponds.lons, self.ponds.lons + 360])

        # Create grids
        lat_grid, extended_lon_grid = np.meshgrid(self.ponds.lats, extended_lons, indexing="ij")
        pos_extended = np.dstack((lat_grid, extended_lon_grid))

        # Calculate Gaussian blob
        cov = np.array([[sigma_lat**2, 0], [0, sigma_lon**2]])
        # - create multivariate normal distribution
        rv = multivariate_normal(mean=[center_lat, center_lon], cov=cov)
        # - compute probability density function (PDF) over the extended grid
        extended_blob = rv.pdf(pos_extended)
        blob_normalized = extended_blob / np.max(extended_blob)

        # add up contributions from periodic longitudes, blob_contribution is of shape (lat_size, lon_size)
        blob_contribution = np.sum(
            blob_normalized.reshape(
                            len(self.ponds.lats),
                            3,
                            len(self.ponds.lons)).transpose(1, 0, 2),
                            axis=0).reshape(len(self.ponds.lats),
                                            len(self.ponds.lons))
        # create mask where blob_contribution is above 0.5
        blob_mask = np.where(blob_contribution > 0.5, 1, 0)

        return blob_mask

###################################################################################################
    
class SINGLE(BlobsMethod):

    def __init__(self,
                 ponds,
    ):
        self.ponds = ponds

    def make_blob(self,
    ):
        """
        Reshapes the ponds object to a single time series, erasing all previous time series.
        Returns a mask of shape (1, 1) with True value indicating the single time series.

        Returns:
            blob_mask (np.ndarray): A 2D array of shape (1, 1) with a single True value.
        """
        # 1) check current size of ponds object
        if self.ponds.lat_size != 1 or self.ponds.lon_size != 1:
            warnings.warn("Shape method SINGLE used on a non-single ponds object." \
            "This step erases all time series and starts the ponds object over.")
            self.ponds.lat_size = 1
            self.ponds.lon_size = 1
            self.ponds.lats = np.array([0])#np.linspace(-90, 90, self.lat_size)
            self.ponds.lons = np.array([0])#np.linspace(-180, 180, self.lon_size)
            self.ponds.ts_param = xr.Dataset(
                coords={
                    "lat": self.ponds.lats,
                    "lon": self.ponds.lons,
                }
            )
            self.ponds.ts_param["shift_id"] = (("lat", "lon"), np.empty((self.ponds.lat_size, self.ponds.lon_size),dtype=object))

            # 2) Generate background data with trend and noise
            data = np.zeros((self.ponds.time_size, self.ponds.lat_size, self.ponds.lon_size))
            for i in range(self.ponds.lat_size):
                for j in range(self.ponds.lon_size):
                    data[:, i, j] = (
                        self.ponds.background_trend * self.ponds.time + 
                        self.ponds.background_noise * np.random.randn(self.ponds.time_size)
                    )
            self.ponds.data = xr.DataArray(
                data,
                coords={"time": self.ponds.time,
                        "lat": self.ponds.lats,
                        "lon": self.ponds.lons,
                        },
                dims=["time", "lat", "lon"],
            )
        blob_mask = np.array([[True]])

        return blob_mask
        


