import numpy as np
import xarray as xr
import warnings
from ponds.blobgen import BlobsMethod
from ponds.shiftgen import ShiftsMethod


class PONDS:
    """
    Main class for PONDS dataset generation.

    Represents the dataset containing of spatial grid cells. Each grid cell contains a time series
    of base values (e.g. background noise). In certain areas additional features (e.g. abrupt shifts)
    are clustered.

    A shift can be added by calling the class function add_shift with custom shape- and shift-generation
    methods. Each shift is stored as a boolean mask in the underlying xarray.Dataset and labeled using
    the cluster_key argument.

    The time series data itself is stored in the `ts` DataArray.

    Args:
        lat_size (int): Number of latitude points
        lon_size (int): Number of longitude points
        time_size (int): Number of time points, governs temporal resolution
        background_trend (float): Linear trend coefficient
        background_noise (float): Amplitude of random noise
        n_shifts (int): Number of abrupt shift events to add
        random_seed (int): Seed for reproducible results
        cluster_key (str): Key to use for the variable name in the time series parameter grid
                           (e.g. "c" for "c1", "c2", ...).
    """


    def __init__(
        self,
        lat_size: int = 1,
        lon_size: int = 1,
        time_size: int = 100,
        background_trend: float = 0.0,
        background_noise: float = 0.005,
        random_seed: int = 42,
        cluster_key: str = "c",
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
        self.cluster_key = cluster_key

        # 1) Set up coordinates
        self.lats = np.linspace(-90, 90, self.lat_size)
        self.lons = np.linspace(-180, 180, self.lon_size)
        self.time = np.linspace(0, self.time_size, self.time_size)

        # 2) Generate background data with trend and noise
        data_ts = np.zeros((time_size, lat_size, lon_size))
        for i in range(lat_size):
            for j in range(lon_size):
                data_ts[:, i, j] = (
                    background_trend * self.time
                    + background_noise * np.random.randn(self.time_size)
                )

        # - Create xarray Dataset, init with background time series data
        self.data = xr.Dataset(
            {
                "ts": (["time", "lat", "lon"], data_ts),
            },
            coords={"time": self.time, "lat": self.lats, "lon": self.lons},
        )

    def add_shift(
        self,
        shift_method: ShiftsMethod,         # currently only sigmoid is implemented
        shape_method: BlobsMethod = None,
        random_seed: int = None,
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

        if random_seed is not None:
            np.random.seed(random_seed)
        self.n_shifts += 1

        # 1) generate blob mask
        if shape_method == None:
            if self.lat_size != 1 or self.lon_size != 1:
                warnings.warn(
                    "No shape method provided for a non-single ponds grid."
                    "Shift method is applied to all cells."
                )
            blob_mask = np.ones((self.lat_size,self.lon_size),dtype=bool)
        else:
            if self.lat_size == 1 and self.lon_size == 1:
                warnings.warn(
                    "Shape method specified for a (1x1) grid. PONDS is in single-mode."
                    "Chosen shape method is likely to fail."
                )
            blob_mask = shape_method.make_blob(lats=self.lats,
                                               lons=self.lons,)

        # 2) generate shift data
        # - choose random time step
        time_buffer = self.time_size // 10
        shift_time = np.random.uniform(
            time_buffer, self.time_size - time_buffer, 1
        ).astype(int)[0]

        # - generate shift data and add to time serie
        mask3d = np.zeros((self.time_size, *blob_mask.shape), dtype=bool)
        for i in range(self.lat_size):
            for j in range(self.lon_size):
                if blob_mask[i, j]:
                    shift_effect, t_shift = shift_method.make_shift(self.time,
                                                                    shift_time)
                    
                    # add shift effect to the data in single grid cell, all time steps at once
                    self.data.ts.values[:, i, j] += shift_effect
                    # update mask3d to indicate where the shift was applied
                    mask3d[t_shift, i, j] = True

        # - store shift id
        self._add_shift_to_ds(
            mask=mask3d,
            param_dict=shift_method.param_dict,
        )

    def _add_shift_to_ds(
        self,
        mask: np.ndarray,
        param_dict: dict,
    ):
        """
        Append a xr.DataArray to the ponds Dataset in form of a boolean mask. It identifies
        the spatial and temporal location of the shift event generated by a user specified shift method.
        The parameters used to generate the shift are stored as attributes of the DataArray.

        This method assumes that each added cluster is stored in a separate variable in the
        time series parameter grid, where the cluster_key is used as the variable name followed
        by a number (e.g. "c1", "c2", ...).

        Args:
            mask (np.ndarray): Boolean mask indicating where to append the shift ID.
            param_dict (dict): Dictionary containing parameters used to generate the shift.
            cluster_key (str): Key to use for the variable name in the time series parameter grid
                               (e.g. "c" for "c1", "c2", ...).
        """

        # get variable name for new cluster
        variable_names = list(self.data.data_vars.keys())
        # print(f"Current variable names: {variable_names}")
        variable_names = [
            name for name in variable_names if name.startswith(self.cluster_key)
        ]
        # print(f"Variable names: {variable_names}")
        last_variable_index = int(len(variable_names))
        new_variable_name = f"{self.cluster_key}{last_variable_index + 1}"

        # Create a new DataArray with the same shape as the mask
        new_da = xr.DataArray(mask, dims=self.data.dims, coords=self.data.coords)
        new_da.attrs["type"] = param_dict.get(
            "type", "UNKNOWN"
        )  # set type to "UNKNOWN" if not specified
        new_da.attrs["generation_params"] = {
            k: v
            for k, v in param_dict.items()  # exclude "type" from generation_params
            if k != "type"
        }

        # Append the new DataArray to the Dataset
        self.data[new_variable_name] = new_da

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
            return self.data.ts.values.flatten()
        else:
            return self.data.ts.values

    def get_cluster_masks(
        self,
        cluster_key: str = "c",
    ):
        """
        Returns all the variables that start with 'cluster_key'. These arrays are
        the masks that identify if a shift happened in a given space-time-coordinate.

        Args:
            cluster_key (str):
        """

        masks = []
        shift_ids = []
        for name, da in self.data.data_vars.items():
            if name.startswith(cluster_key):
                # get shift_id
                ids = np.unique(da.values)
                if len(ids[ids >= 0]) > 1:
                    warnings.warn(
                        "More than one shift-id found in cluster mask."
                        "Smallest shift-id chosen to proceed..."
                    )
                shift_ids.append(ids[ids >= 0][0])

                # convert shift_id-mask to boolean mask
                da_mask = np.zeros(da.shape, dtype=bool)
                mask_ix = np.argwhere(da.values >= 0)
                da_mask[mask_ix[:, 0], mask_ix[:, 1], mask_ix[:, 2]] = True
                masks.append(da_mask)

        return np.array(masks), np.array(shift_ids)
