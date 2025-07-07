import numpy as np
import xarray as xr
from ponds.blobgen import BlobsMethod
from ponds.shiftgen import ShiftsMethod


class PONDS:
    """
    Main class for PONDS (Pond of Nonlinear Dynamics in Space) dataset generation.

    Represents the dataset containing of spatial grid cells. Each grid cell contains a time series
    of base values (e.g. background noise). In certain areas additional features (e.g. abrupt shifts)
    are clustered.

    Central for the shift data generation is the xarray Dataset `ts_param`, which contains meta-parameters
    about the time series in each grid cell. This dataset minimally contains the variable `shift_id`, which
    stores a list of ids for abrupt shift events in each grid cell. The ids are used to reference which kind of
    shift is present in the time series data:
        - 0: sigmoid shift
        - 1: ...

    The time series data itself is stored in the `data` DataArray.

    Args:
        lat_size (int): Number of latitude points
        lon_size (int): Number of longitude points
        time_size (int): Number of time points, governs temporal resolution
        background_trend (float): Linear trend coefficient
        background_noise (float): Amplitude of random noise
        n_shifts (int): Number of abrupt shift events to add
        random_seed (int): Seed for reproducible results
    """

    def __init__(
        self,
        lat_size: int = 90,
        lon_size: int = 180,
        time_size: int = 80,
        background_trend: float = 0.0,
        background_noise: float = 0.005,
        random_seed: int = 42,
    ):
        """
        Generate spatial grid and seed with time series generator parameters.
        """
        np.random.seed(random_seed)

        self.lat_size = lat_size
        self.lon_size = lon_size
        self.time_size = time_size
        self.background_trend = background_trend
        self.background_noise = background_noise
        self.random_seed = random_seed
        self.n_shifts = 0

        # 1) Set up coordinates
        self.lats = np.linspace(-90, 90, self.lat_size)
        self.lons = np.linspace(-180, 180, self.lon_size)
        self.time = np.linspace(0, self.time_size, self.time_size)

        # 2) Generate background data with trend and noise
        data = np.zeros((time_size, lat_size, lon_size))
        for i in range(lat_size):
            for j in range(lon_size):
                data[:, i, j] = (
                    background_trend * self.time
                    + background_noise * np.random.randn(self.time_size)
                )

        # - Create xarray DataArray
        self.data = xr.DataArray(
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
        self.ts_param["shift_id"] = (
            ("lat", "lon"),
            np.empty((self.lat_size, self.lon_size), dtype=object),
        )

    def add_shift(
        self,
        shape_method: BlobsMethod,  # currently only gauss is implemented
        shift_method: ShiftsMethod,  # currently only sigmoid is implemented
        random_seed: int = None,
        kwargs: dict = dict(),
    ):
        """
        Adds a shift to the time series data.

        Shape Methods:
            - "GAUSS": Adds a shift in the shape of a Gaussian blob.
            - "SINGLE": Return a single time series.
            - ...
        Shift Methods:
            - "SIGMOID": Adds a shift using a sigmoid function.
            - ...

        Args:
            shape_method (BlobsMethod): Method to generate the shape of the shift.
            shift_method (ShiftsMethod): Method to generate the shift in the time series.
            random_seed (int, optional): Seed for reproducibility. Defaults to None.
            kwargs (dict, optional): Additional keyword arguments for the shift method.
        """
        dict_param = kwargs.copy()

        if random_seed is not None:
            np.random.seed(random_seed)
        self.n_shifts += 1

        # 1) generate blob mask
        blob_generator = shape_method(self, random_seed=random_seed)
        blob_mask = blob_generator.make_blob()

        # 2) generate shift data
        # - choose random time step
        time_buffer = self.time_size // 10
        shift_time = np.random.uniform(
            time_buffer, self.time_size - time_buffer, 1
        ).astype(int)[0]

        # - generate shift data and add to time serie
        for i in range(self.lat_size):
            for j in range(self.lon_size):
                if blob_mask[i, j]:
                    shift_generator = shift_method(
                        self,
                        mask=blob_mask,
                        shift_time=shift_time,
                        random_seed=random_seed,
                    )
                    shift_effect, dict_param = shift_generator.make_shift(**kwargs)
                    # add shift effect to the data in single grid cell, all time steps at once
                    self.data.values[:, i, j] += shift_effect

        # - store shift parameter
        self.param_append(
            blob_mask,
            val_list=list(dict_param.values()),
            var_str_list=list(dict_param.keys()),
        )

    def param_append(
        self,
        mask: np.ndarray,
        val_list: list,
        var_str_list: list,
    ):
        """
        Append values to the corresponding list in the time series parameter grid based on a boolean mask.

        Args:
            mask (np.ndarray): Boolean mask indicating where to append values.
            val_list (list): List of values to append.
            var_str_list (list): List of variable names corresponding to the values.
        """
        for k in range(len(var_str_list)):
            var_str = var_str_list[k]
            val = val_list[k]

            # CASE: variable does not exist yet
            if var_str not in self.ts_param:
                self.ts_param[var_str] = (
                    ("lat", "lon"),
                    np.empty((self.lat_size, self.lon_size), dtype=object),
                )
            # extract current variable data
            data = self.ts_param[var_str].values

            # init list or append value to existing list for each cell
            for i in range(len(self.lats)):
                for j in range(len(self.lons)):
                    if mask[i, j]:
                        if data[i, j] is None:
                            data[i, j] = [val]
                        else:
                            data[i, j].append(val)

            self.ts_param[var_str].values = data

    def get_ts(
        self,
        single: bool = False,
    ):
        """
        Retrieve all time series data as a numpy array.

        Args:
            single (bool): If True, return a single flattened array of time series data.
                          If False, return a 3D array with dimensions (time, lat, lon).
        Returns:
            np.ndarray: Time series data in the specified format.
        """

        if single:
            return self.data.values.flatten()
        else:
            return self.data.values

    def get_cell_tsparam(self, lat_idx: int, lon_idx: int):
        """
        Retrieve a specific cell from the grid based on latitude and longitude indices.

        Args:
            lat_idx (int): Index of the latitude.
            lon_idx (int): Index of the longitude.
        """
        return self.ts_param[lat_idx][lon_idx]
