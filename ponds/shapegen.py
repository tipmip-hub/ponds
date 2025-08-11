from abc import ABC, abstractmethod
import numpy as np

# for gauss
from scipy.stats import multivariate_normal


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
        sigma_lat=None,     # standard deviation in latitude
        sigma_lon=None,     # standard deviation in longitude
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
            "shape_type": BLOB,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "sigma_lat": sigma_lat,
            "sigma_lon": sigma_lon,
        }

    def make_shape(
        self,
        lats,
        lons,
    ) -> np.ndarray:
        """
        Generates a spatial blob of parameters on the grid in shape of a 2D Gauss, given positional and
        spatial shape parameters. Returns a mask of the same shape as the grid, where values above 0.5
        indicate presence of the blob.

        Returns:
            shape_mask (np.ndarray): A 2D array of shape (lat_size, lon_size).
        """

        # Handle longitude periodicity to ensure continuity at the dateline (global dataset!)
        extended_lons = np.concatenate([lons - 360, lons, lons + 360])

        # Create grids
        lat_grid, extended_lon_grid = np.meshgrid(lats, extended_lons, indexing="ij")
        pos_extended = np.dstack((lat_grid, extended_lon_grid))

        # Calculate Gaussian Shape
        cov = np.array([[self.sigma_lat**2, 0], [0, self.sigma_lon**2]])
        # - create multivariate normal distribution
        rv = multivariate_normal(mean=[self.center_lat, self.center_lon], cov=cov)
        # - compute probability density function (PDF) over the extended grid
        extended_shape = rv.pdf(pos_extended)
        shape_normalized = extended_shape / np.max(extended_shape)

        # add up contributions from periodic longitudes, shape_contribution is of shape (lat_size, lon_size)
        shape_contribution = np.sum(
            shape_normalized.reshape(len(lats), 3, len(lons)).transpose(1, 0, 2),
            axis=0,
        ).reshape(len(lats), len(lons))
        # create mask where shape_contribution is above 0.5
        shape_mask = np.where(shape_contribution > 0.5, True, False)

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
