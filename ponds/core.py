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
        background_noise: float = 0.0,
        cluster_key: str = "c",
        random_seed: int = 1,
    ):
        """
        Generate spatial grid and seed with time series generator parameters.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 1) Set up coordinates
        self.lats = np.linspace(-90, 90, lat_size)
        self.lons = np.linspace(-180, 180, lon_size)
        self.time = np.linspace(0, time_size - 1, time_size)

        # 2) Generate background data with trend and noise
        data_ts = np.zeros((time_size, lat_size, lon_size))
        for i in range(lat_size):
            for j in range(lon_size):
                data_ts[:, i, j] = (
                    background_trend * self.time
                    + background_noise * np.random.randn(time_size)
                )

        # - Create xarray DataArray, init with background time series data
        dims = ("time", "lat", "lon")
        da_ts = xr.DataArray(
            data_ts,
            dims=dims,
            coords={
                "time": self.time,
                "lat": self.lats,
                "lon": self.lons,
            },
            attrs={
                "random_seed": random_seed,
                "background_trend": background_trend,
                "background_noise": background_noise,
            },
        )
        # - Create xarray Dataset to hold the time series data
        self.data = xr.Dataset(
            {
                "ts": da_ts,
            },
            coords={"time": self.time, "lat": self.lats, "lon": self.lons},
            attrs={
                "cluster_key": cluster_key,
            },
        )

    def add_shift(
        self,
        shift_method: ShiftsMethod,
        shape_method: BlobsMethod = None,
    ):
        """
        Adds a shift to the time series data.

        Shape Methods:
            - "GAUSS": Adds a shift in the shape of a Gaussian blob.
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

        if self.data.ts.attrs["random_seed"] is not None:
            np.random.seed(self.data.ts.attrs["random_seed"])
        else:
            warnings.warn(
                "No random seed set for PONDS. Results will not be reproducible."
            )
        # self.n_shifts += 1

        # 1) generate blob mask
        if shape_method is None:
            if self.data.sizes["lat"] != 1 or self.data.sizes["lon"] != 1:
                warnings.warn(
                    "No shape method provided for a non-single ponds grid."
                    "Shift method is applied to all cells."
                )
            blob_mask = np.ones(
                (self.data.sizes["lat"], self.data.sizes["lon"]), dtype=bool
            )
            blob_dict = {"shape_type": "SINGLE"}
        else:
            if self.data.sizes["lat"] == 1 and self.data.sizes["lon"] == 1:
                warnings.warn(
                    "Shape method specified for a (1x1) grid. PONDS is in single-mode."
                    "Chosen shape method is likely to fail."
                )
            blob_mask = shape_method.make_blob(
                lats=self.lats,
                lons=self.lons,
            )
            blob_dict = shape_method.param_dict

        # 2) generate shift data
        # - generate shift data and add to time serie
        mask3d = np.zeros((self.data.sizes["time"], *blob_mask.shape), dtype=bool)
        for i in range(self.data.sizes["lat"]):
            for j in range(self.data.sizes["lon"]):
                if blob_mask[i, j]:
                    shift_effect, t_shift = shift_method.make_shift(
                        self.time,
                    )

                    # add shift effect to the data in single grid cell, all time steps at once
                    self.data.ts.values[:, i, j] += shift_effect
                    # update mask3d to indicate where the shift was applied
                    mask3d[t_shift, i, j] = True

        # check if shift_method param_dict contains 'shift_type' key
        if "shift_type" not in shift_method.param_dict:
            warnings.warn(
                "Shift method param_dict does not contain 'shift_type' key. "
                "Setting shift_type to 'UNKNOWN'."
            )
            shift_method.param_dict["shift_type"] = "UNKNOWN"
        # check if shape_method param_dict contains 'shape_type' key
        if "shape_type" not in blob_dict:
            warnings.warn(
                "Shape method param_dict does not contain 'shape_type' key. "
                "Setting shape_type to 'UNKNOWN'."
            )
            blob_dict["shape_type"] = "UNKNOWN"

        # - store shift as a DataArray in the Dataset
        self._add_shift_to_ds(
            mask=mask3d,
            shift_dict=shift_method.param_dict,
            blob_dict=blob_dict,
        )

    def _get_new_cluster_key(self):
        """
        Generate a new cluster key based on the current number of shifts.
        The cluster key is used to name the variables in the Dataset.
        """
        variable_names = list(self.data.data_vars.keys())
        variable_names = [
            name
            for name in variable_names
            if name.startswith(self.data.attrs["cluster_key"])
        ]
        last_variable_index = int(len(variable_names))
        return f"{self.data.attrs['cluster_key']}{last_variable_index + 1}"

    def _add_shift_to_ds(
        self,
        mask: np.ndarray,
        shift_dict: dict,
        blob_dict: dict,
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
        new_cluster_name = self._get_new_cluster_key()

        # Create a new DataArray with the same shape as the mask
        new_da = xr.DataArray(mask, dims=self.data.dims, coords=self.data.coords)
        # Add attributes to the DataArray
        # - add shift attributes
        for k, v in shift_dict.items():
            new_da.attrs[k] = v
        # - add blob attributes
        for k, v in blob_dict.items():
            new_da.attrs[k] = v

        # Append the new DataArray to the Dataset
        self.data[new_cluster_name] = new_da

    ###################################################################################################
    ######################################### Get Functions ###########################################
    ###################################################################################################

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

    def get_ponds_dict(self):
        """
        Extracts the metadata from the PONDS instance neccessary to initialize
        a PONDS replica and returns it as a dictionary.

        Returns:
            dict: Dictionary containing the metadata of the PONDS instance.
        """
        params = {
            "lat_size": self.data.sizes["lat"],
            "lon_size": self.data.sizes["lon"],
            "time_size": self.data.sizes["time"],
            "background_trend": self.data.ts.attrs["background_trend"],
            "background_noise": self.data.ts.attrs["background_noise"],
            "cluster_key": self.data.attrs["cluster_key"],
            "random_seed": self.data.ts.attrs["random_seed"],
        }
        return params

    def get_cluster_dict(self, cluster: str):
        """
        Extracts the shift and shape metadata from a given cluster mask.

        Args:
            ponds (PONDS): The PONDS instance to extract metadata from.
            cluster (str): The cluster to filter the metadata attributes.

        Returns:
            dict: A dictionary containing the shift metadata.
        """

        # get list of keys in order of appearance
        keys = list(self.data[cluster].attrs.keys())
        # find index of "shape_type"
        shape_type_index = keys.index("shape_type")

        # split keys into shift and shape dicts
        # - shift dict contains all keys before "shape_type"
        shift_dict = {k: self.data[cluster].attrs[k] for k in keys[1:shape_type_index]}
        # - shape dict contains all keys after "shape_type"
        shape_dict = {
            k: self.data[cluster].attrs[k] for k in keys[shape_type_index + 1 :]
        }

        return shift_dict, shape_dict
    
###################################################################################################
###################################### Read and Write #############################################
###################################################################################################

    def save(
            self,
            filename="ponds_data.nc",
    ):
        """
        Save the ponds data to a NetCDF file.
        
        Parameters:
            filename (str): The name of the file to save the data to.
        """
        self.data.to_netcdf(filename, engine="netcdf4")

    @staticmethod
    def load(filename="ponds_data.nc"):
        """
        Read the ponds data from a NetCDF file.

        Returns:
            pd (PONDS): An instance of the PONDS class with the data loaded.
        """
        with xr.open_dataset(filename, engine="netcdf4") as ds:
            data = ds.load()  # Detach from file completely

        pd = PONDS.__new__(PONDS)
        pd.data = data
        pd.lats = data.coords["lat"].values
        pd.lons = data.coords["lon"].values
        pd.time = data.coords["time"].values
        return pd

