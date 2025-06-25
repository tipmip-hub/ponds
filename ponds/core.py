import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal

class PONDS:
    """
    Main class for PONDS (Pond of Nonlinear Dynamics in Space) dataset generation.

    Represents the dataset containing of spatial grid cells. Each grid cell contains a time series
    of base values (e.g. background noise). In certain areas additional features (e.g. abrupt shifts)
    are clustered.

    SIMPLE VERSION: random Gaussian blobs, sigmoid function for abrupt shifts.

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
        self.lats = np.linspace(-90, 90, self.lat_size)
        self.lons = np.linspace(-180, 180, self.lon_size)
        self.time = np.linspace(0, self.time_size, self.time_size)

        # 2) Generate background data with trend and noise
        data = np.zeros((time_size, lat_size, lon_size))
        for i in range(lat_size):
            for j in range(lon_size):
                data[:, i, j] = (
                    background_trend * self.time + background_noise * np.random.randn(self.time_size)
                )

        # - Create xarray DataArray
        self.ts_data = xr.DataArray(
            data,
            coords={"time": self.time, "lat": self.lats, "lon": self.lons},
            dims=["time", "lat", "lon"],
        )

        # 3) Define time series seeds on spatial grid
        # init grid of time series parameters
        self.ts_param = xr.Dataset(
            coords={
                "lat": self.lats,
                "lon": self.lons,
            }
        )
        self.ts_param["sig_num"] = (("lat", "lon"), np.zeros((self.lat_size, self.lon_size),dtype=int))
        self.ts_param["sig_tim"] = (("lat", "lon"), np.empty((self.lat_size, self.lon_size),dtype=object))

        # 3) Generate random blobs
        center_lat = np.random.uniform(-80, 80, size=n_shifts)
        center_lon = np.random.uniform(-180, 180, size=n_shifts)
        sigma_lat = np.random.uniform(10, 20, size=n_shifts)
        sigma_lon = np.random.uniform(15, 40, size=n_shifts)
        sigma_time = np.ones(n_shifts)  # Sharp shifts
        steepness = np.random.uniform(1.5, 3.0, size=n_shifts)
        shift_magnitude = np.random.uniform(0.95, 1.05, size=n_shifts)
        time_buffer = self.time_size // 10
        shift_times = np.linspace(time_buffer, self.time_size - time_buffer, n_shifts).astype(int)

        for i in range(n_shifts):
            blob_mask = self.gauss_blob(center_lat=center_lat[i],
                              center_lon=center_lon[i],
                              sigma_lat=sigma_lat[i],
                              sigma_lon=sigma_lon[i])
            self.ts_param["sig_num"] += blob_mask
            # - assign shift times to the cells
            self.param_append(
                mask=blob_mask,
                value=shift_times[i],
                var_str="sig_tim"       # sig_tim is the central shift time
            )
            self.param_append(
                mask=blob_mask,
                value=sigma_time[i],
                var_str="sig_tsg"       # sig_tsg is the time sigma of the central shift time
            )
            self.param_append(
                mask=blob_mask,
                value=steepness[i],
                var_str="sig_ste"       # steepness is the steepness of the sigmoid function
            )
            self.param_append(
                mask=blob_mask,
                value=shift_magnitude[i],
                var_str="sig_mag"       # shift_magnitude is the magnitude of the shift
            )


    def generate_shifts(
            self,
    ):
        """
        Generates the time series data based on the parameters defined in the ts_param grid.
        """

        for i in range(len(self.lats)):
            for j in range(len(self.lons)):

                for k in range(self.ts_param.sig_num.values[i, j]):
                    center_time = self.ts_param.sig_tim.values[i, j][k]
                    steepness = self.ts_param.sig_ste.values[i, j][k]
                    magnitude = self.ts_param.sig_mag.values[i, j][k]
                    sigma_time = self.ts_param.sig_tsg.values[i, j][k]
                    t_shift = int(center_time + np.random.normal(0, sigma_time))
                    t_shift = max(0, min(t_shift, len(self.time) - 1))

                    shift_effect = (
                        magnitude
                        * sigmoid(self.time, self.time[t_shift], steepness)
                    )
                    # add sigmoid shift effect to the data in single grid cell, all time steps at once
                    self.ts_data.values[:, i, j] += shift_effect



    def gauss_blob(
            self,                       # kind of parameter:
            center_lat: float = 0,      # - position
            center_lon: float = 0,      # - position
            sigma_lat: float = 10,      # - shape
            sigma_lon: float = 20,      # - shape
        ):
        """
        Generate a spatial blob of parameters on the grid, given positional and spatial shape parameters.
        Returns a mask of the same shape as the grid, where values above 0.5 indicate presence of the blob.
        
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

        # add up contributions from periodic longitudes, blob_contribution is of shape (lat_size, lon_size)
        blob_contribution = np.sum(blob_normalized.reshape(len(self.lats),3,len(self.lons)).transpose(1, 0, 2), axis=0).reshape(len(self.lats), len(self.lons))
        blob_mask = np.where(blob_contribution > 0.5, 1, 0)

        return blob_mask


    def param_append(
            self,
            mask,
            value,
            var_str
        ):
        """
        Append a value to the time series parameter grid based on a mask.
        """
        # CASE: variable doe not exist yet
        if var_str not in self.ts_param:
            self.ts_param[var_str] = (("lat", "lon"), np.empty((self.lat_size, self.lon_size), dtype=object))
        # extract current variable data
        data = self.ts_param[var_str].values

        # init list or append value to existing list for each cell
        for i in range(len(self.lats)):
            for j in range(len(self.lons)):
                if mask[i, j]:
                    if data[i, j] is None:
                        data[i, j] = [value]
                    else:
                        data[i, j].append(value)

        self.ts_param[var_str].values = data

    def get_cell_tsparam(
            self,
            lat_idx: int,
            lon_idx: int
        ):
        """
        Retrieve a specific cell from the grid based on latitude and longitude indices.
        """
        return self.ts_param[lat_idx][lon_idx]


##################################################################################################
##################################################################################################
##################################################################################################

def sigmoid(t, t0, k: float = 1.0) -> np.ndarray:
    """Sigmoid transition function"""
    return 1 / (1 + np.exp(-k * (t - t0)))