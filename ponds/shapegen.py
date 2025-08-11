from abc import ABC, abstractmethod
import numpy as np

# for BLOB
#from scipy.stats import multivariate_normal


# Abstract class
class ShapesMethod(ABC):
    @abstractmethod
    def make_shape(
        self,
        lats,
        lons,
    ) -> np.ndarray:
        pass


###################################################################################################
############################################# SINGLE ###############################################
###################################################################################################


class SINGLE(ShapesMethod):
    def __init__(self):
        """
            A placeholder for a single shape method. This is used when no specific shape method is defined.
        """
            
        self.param_dict = {
            "shape_type": SINGLE,
            }

    def make_shape(
        self,
        lats,
        lons,
    ) -> np.ndarray:
        return np.ones(
            (len(lats), len(lons)), dtype=bool
        )  # Single shape covering the entire grid


###################################################################################################
############################################ BLOB #################################################
###################################################################################################


class BLOB(ShapesMethod):
    def __init__(
        self,
        random_seed=1,
        center_lat=None,    # center latitude of the blob
        center_lon=None,    # center longitude of the blob
        ecc_lat=None,     # standard deviation in latitude
        ecc_lon=None,     # standard deviation in longitude
    ):
        if random_seed is not None:
            np.random.seed(random_seed)

        if center_lat is None:
            center_lat = np.random.uniform(-80, 80)
        if center_lon is None:
            center_lon = np.random.uniform(-180, 180)
        if ecc_lat is None:
            ecc_lat = np.random.uniform(10, 20)
        if ecc_lon is None:
            ecc_lon = np.random.uniform(15, 40)

        self.center_lat = center_lat
        self.center_lon = center_lon
        self.ecc_lat = ecc_lat
        self.ecc_lon = ecc_lon

        self.param_dict = {
            "shape_type": BLOB,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "ecc_lat": ecc_lat,
            "ecc_lon": ecc_lon,
        }

    def make_shape(
        self,
        lats,
        lons,
    ) -> np.ndarray:
        """
        Generates a 2D spatial blob of eliptic shape. Given are the central latitude and longitude, as well as the
        extend of the blob in latitude and longitude. The function returns a mask of the same shape as the grid,
        where are values that are part of the blob are True, and all other values are False.

        Elliptic Formula:
        ((lat - center_lat) / sigma_lat)**2 + ((lon - center_lon) / sigma_lon)**2 <= 1
        """
        # Create coordinate combinations of all latitudes and longitudes
        # - handle longitude periodicity to ensure continuity at the dateline (global dataset!)
        extended_lons = np.concatenate([lons - 360, lons, lons + 360])
        # - create meshgrid for latitudes and extended longitudes
        lat_grid, lon_grid = np.meshgrid(lats, extended_lons, indexing='ij')

        # Calculate the mask for the elliptic shape on the longitude-extended grid
        shape_mask = ((lat_grid - self.center_lat) / self.ecc_lat)**2 + ((lon_grid - self.center_lon) / self.ecc_lon)**2 <= 1

        # add up contributions from periodic longitudes, shape_contribution is of shape (lat_size, lon_size)
        # - split shape_mask into three parts, one for each longitude section
        shape_mask = shape_mask.reshape(len(lats), 3, len(lons))
        # - transpose to have latitudes as first dimension
        shape_mask = shape_mask.transpose(1, 0, 2)
        # - if any of the three sections is True, then the point is part of the shape
        shape_mask = np.any(shape_mask, axis=0)

        return shape_mask


###################################################################################################
########################################### CUSTOM ################################################
###################################################################################################


class CUSTOM(ShapesMethod):
    def __init__(self, shape_mask: np.ndarray):
        """
        Custom shape method that uses a pre-defined mask.

        Args:
            shape_mask (np.ndarray): A 2D boolean array of shape (lat_size, lon_size) representing
            the shape mask.
        """
        self.shape_mask = np.array(shape_mask)
        self.param_dict = {
            "shape_type": CUSTOM,
        }

    def make_shape(
        self,
        lats,
        lons,
    ) -> np.ndarray:
        """
        Returns the pre-defined shape mask. Checks if the mask is of the correct shape and type.

        Returns:
            shape_mask (np.ndarray): The pre-defined shape mask.
        """
        # check if the shape_mask is of the correct shape
        if self.shape_mask.shape != (len(lats), len(lons)):
            raise ValueError(
                f"Shape mask shape {self.shape_mask.shape} does not match grid shape {(len(lats), len(lons))}."
            )
        # check if the shape_mask is a boolean array
        if not np.issubdtype(self.shape_mask.dtype, np.bool_):
            raise ValueError("Shape mask must be a boolean array.")

        return self.shape_mask
