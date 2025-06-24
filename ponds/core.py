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
        time = np.linspace(0, 100, self.time_size)
        lats = np.linspace(-90, 90, self.lat_size)
        lons = np.linspace(-180, 180, self.lon_size)

        # 2) Define time series seeds on spatial grid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        self.ts_param = [
            [Cell(lat=lat_grid[i, j], lon=lon_grid[i, j]) for j in range(lat_grid.shape[1])]
            for i in range(lat_grid.shape[0])
        ]

    def get_cell_param(
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
                 lon
    ):
        """
        Initialize a cell with its coordinates and time series data.
        """
        self.lat = lat
        self.lon = lon

        