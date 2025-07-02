import pytest
import numpy as np
from ponds import PONDS, GAUSS

"""

Things to Test:
    - Random Gauss blob
        x must return boolean 2D array
        x blob mask must have the same shape as the ponds grid
        x does the same seed produce the same blob mask
        x check compatibility with the PONDS.add_shift() function
        x check if in blob mask at least one cell is True
        - do the blobs scale with the grid size
    - Custom Gauss blob
        - is the location correct
        - is the shape correct

Legend:
    - : to do
    x : done
"""

###################################################################################################
####################################### Random Gauss ##############################################
###################################################################################################


def test_gauss_blob_output():
    """
    Checking two cases: small and large grid size.
        - check if the output is a boolean 2D array
        - check if the shape of the output matches the ponds grid size
    """
    # >> small ponds
    # 1) setup variables
    lat_size = 10
    lon_size = 20
    ponds_small = PONDS(lat_size=lat_size,
                        lon_size=lon_size)
    # 2) call function
    gauss_small = GAUSS(ponds_small)
    blob_small = gauss_small.make_blob()
    # 3) check results
    assert isinstance(blob_small, np.ndarray), "Blob must be a numpy array."
    assert blob_small.dtype == bool, "Blob must be a boolean array."
    assert blob_small.shape == (lat_size,
                                lon_size)

    # >> large ponds
    # 1) setup variables
    lat_size = 200
    lon_size = 300
    ponds_large = PONDS(lat_size=lat_size,
                        lon_size=lon_size)
    # 2) call function
    gauss_large = GAUSS(ponds_large)
    blob_large = gauss_large.make_blob()
    # 3) check results
    assert isinstance(blob_large, np.ndarray), "Blob must be a numpy array."
    assert blob_large.dtype == bool, "Blob must be a boolean array."
    assert blob_large.shape == (lat_size,
                                lon_size)


def test_gauss_same_seed_produces_same_blob_mask(R=5):
    """
    - check if the same random seed produces the same blob mask
    """

    # 1.1) setup variables
    lat_size = 100
    lon_size = 200

    for _ in range(R):

        # 1.2) setup variables
        seed_pond = np.random.randint(0, 1000)
        seed_blob = np.random.randint(0, 1000)
        ponds1 = PONDS(lat_size=lat_size,
                       lon_size=lon_size,
                       random_seed=seed_pond)        
        ponds2 = PONDS(lat_size=lat_size,
                       lon_size=lon_size,
                       random_seed=seed_pond)
        
        # 2) call function
        gauss1 = GAUSS(ponds1,
                       random_seed=seed_blob)
        blob1 = gauss1.make_blob()
        gauss2 = GAUSS(ponds2,
                       random_seed=seed_blob)
        blob2 = gauss2.make_blob()

        # 3) check results
        assert np.array_equal(blob1, blob2), "Blobs with same seed should be identical."

def test_gauss_blob_add_shift_integration(mock_shift):
    """
    - test compatibility with PONDS.add_shift()
    - check if blob_mask is not all False
    """

    # 1) setup variables
    ponds = PONDS(lat_size=10,
                  lon_size=20,
                  time_size=30)

    # 2) call function
    ponds.add_shift(shape_method=GAUSS,
                    shift_method=mock_shift)
    
    # 3) check results
    # - Check ts_param is updated at least somewhere
    shift_grid = ponds.ts_param["shift_id"].values
    assert any(cell is not None for row in shift_grid for cell in row), "Shift should be applied somewhere on grid."


###################################################################################################
####################################### Custom Gauss ##############################################
###################################################################################################
