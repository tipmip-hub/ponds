import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal

class PONDS:
    """
    Main class for PONDS (Pond of Nonlinear Dynamics in Space) dataset generation.

    Represents the dataset containing of spatial grid cells. Each grid cell contains a time series
    of base values (e.g. background noise). In certain areas additional features (e.g. abrupt shifts)
    are clustered.

    Args:
        lat_size (int): Number of latitude points
        lon_size (int): Number of longitude points
        time_size (int): Number of time points, governs temporal resolution
        background_trend (float): Linear trend coefficient
        background_noise (float): Amplitude of random noise
        n_shifts (int): Number of abrupt shift events to add
        random_seed (int): Seed for reproducible results
    """
    def __init__(self,
                 lat_size: int = 90,
                 lon_size: int = 180,
                 time_size: int = 80,
                 background_trend: float = 0.0,
                 background_noise: float = 0.005,
                 n_shifts: int = 3,
                 random_seed: int = 42):
        """
        Generate spatial grid and seed with time series generator parameters.
        """
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.time_size = time_size
        self.background_trend = background_trend
        self.background_noise = background_noise
        self.n_shifts = n_shifts
        self.random_seed = random_seed

        # 1) Set up coordinates
        #time = np.linspace(0, 100, self.time_size)
        self.lats = np.linspace(-90, 90, self.lat_size)
        self.lons = np.linspace(-180, 180, self.lon_size)

        # 2) Define time series seeds on spatial grid
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)

        # init grid of time series parameters
        self.ts_param = xr.Dataset(
            coords={
                "lat": self.lats,
                "lon": self.lons,
            }
        )
        #self.ts_param = [
        #    [Cell(lat=lat_grid[i, j], lon=lon_grid[i, j], time=time_size) for j in range(lat_grid.shape[1])]
        #    for i in range(lat_grid.shape[0])
        #]

        # 3) Generate random blobs
        center_lat = np.random.uniform(-80, 80, size=n_shifts)
        center_lon = np.random.uniform(-180, 180, size=n_shifts)
        sigma_lat = np.random.uniform(10, 20, size=n_shifts)
        sigma_lon = np.random.uniform(15, 40, size=n_shifts)

        for i in range(n_shifts):
            self.spatial_blob(center_lat=center_lat[i],
                              center_lon=center_lon[i],
                              sigma_lat=sigma_lat[i],
                              sigma_lon=sigma_lon[i])


    def spatial_blob(
            self,                       # kind of parameter:
            center_lat: float = 0,      # - position
            center_lon: float = 0,      # - position
            sigma_lat: float = 10,      # - shape
            sigma_lon: float = 20,      # - shape
        ):
        """
        Generate a spatial blob of parameters on the grid, given positional and spatial shape parameters.
        
        SIMPLE VERSION: random Gaussian blob
        """

        # Handle longitude periodicity to ensure continuity at the dateline (global dataset!)
        extended_lons = np.concatenate([self.lons - 360, self.lons, self.lons + 360])

        # Create grids
        lat_grid, extended_lon_grid = np.meshgrid(self.lats, extended_lons, indexing="ij")
        pos_extended = np.dstack((lat_grid, extended_lon_grid))

        # Calculate Gaussian blob
        cov = np.array([[sigma_lat**2, 0], [0, sigma_lon**2]])
        # - create multivariate normal distribution
        rv = multivariate_normal(mean=[center_lat, center_lon], cov=cov)
        # - compute probability density function (PDF) over the extended grid
        extended_blob = rv.pdf(pos_extended)
        blob_normalized = extended_blob / np.max(extended_blob)

        #print("Blob shape:", blob_normalized.shape)
        #print(blob_normalized)

        # add up contributions from periodic longitudes, blob_contribution is of shape (lat_size, lon_size)
        blob_contribution = np.sum(blob_normalized.reshape(len(self.lats),3,len(self.lons)).transpose(1, 0, 2), axis=0).reshape(len(self.lats), len(self.lons))




    def get_cell_tsparam(
            self,
            lat_idx: int,
            lon_idx: int
        ):
        """
        Retrieve a specific cell from the grid based on latitude and longitude indices.
        """
        return self.ts_param[lat_idx][lon_idx]



###################################################################################################
################################# CLASS: Cell #####################################################
###################################################################################################

class Cell:
    """
    Class representing a single cell in the PONDS dataset.
    """

    def __init__(self, 
                 lat,
                 lon,
                 time
    ):
        """
        Initialize a cell with its coordinates and time series data.
        """
        self.lat = lat
        self.lon = lon
        self.time = time
