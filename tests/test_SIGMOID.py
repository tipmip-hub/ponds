import pytest
import numpy as np
from ponds import PONDS, SIGMOID

"""
Things to Test:
    - General Tests
        x is the right blob mask used
        - is the right shift time used
    - Random sigmoid shift
        - do same seed produce same shift
    - Custom sigmoid shift
        - check if custom parameters are used
            - sigma_time
            - shift_steepness
            - shift_magnitude

Legend:
    - : to do
    x : done
"""

###################################################################################################
########################################## General ################################################
###################################################################################################

def test_sigmoid_blob_mask_applied_correctly():
    """
    - make boolean mask and check if only the masked cells are shifted
    """

    # 1) setup variables
    lat_size = 3
    lon_size = 3
    time_size = 10
    ponds = PONDS(lat_size=lat_size,
                  lon_size=lon_size,
                  time_size=time_size,
                  background_trend=0,
                  background_noise=0)
    # - create a mask with a single True value at (0, 1)
    mask = np.zeros((lat_size,lon_size), dtype=bool)
    mask[0,1] = True
    # - shift at midpoint of time series
    shift_time = int(time_size // 2)

    # 2) call function
    sigmoid_shift = SIGMOID(ponds, mask, shift_time)
    sigmoid_shift.make_shift()

    # Only [0,1] cell should be non-zero
    for i in range(3):
        for j in range(3):
            ts = ponds.data[:,i,j].values
            if mask[i,j]:
                assert not np.allclose(ts, 0), f"Shift not applied to masked cell {(i,j)}"
            else:
                assert np.allclose(ts, 0), f"Shift incorrectly applied to unmasked cell {(i,j)}"

def test_sigmoid_shift_time_used():

    # 1) setup variables
    time_size = 100
    ponds = PONDS(lat_size=1,
                  lon_size=1,
                  time_size=time_size,
                  background_trend=0,
                  background_noise=0)
    mask = np.ones((1,1), dtype=bool)
    np.random.seed(None)
    shift_time = np.random.randint(0, time_size)

    # 2) call function
    shift_dict = {"sigma_time": 0}
    sigmoid_shift = SIGMOID(ponds,
                            mask,
                            shift_time)
    params = sigmoid_shift.make_shift(**shift_dict)

    # 3) check results
    # - Calculate where sigmoid crosses ~0.5
    ts = ponds.data[:,0,0].values
    t0_index = np.argmax(np.diff(ts)) +1   # +1 because 0-indexing
    # -
    assert abs(t0_index - shift_time) <= 2, f"Shift applied at wrong time index {t0_index}, expected around {shift_time}"





###################################################################################################
####################################### Random Shift ##############################################
###################################################################################################



###################################################################################################
####################################### Custom Shift ##############################################
###################################################################################################
