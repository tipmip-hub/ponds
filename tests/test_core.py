import pytest
import numpy as np
from ponds.core import PONDS

"""
Testing the PONDS core class and its methods.

Things to Test:
    x Initialization of PONDS object
        x does the grid have the right shape (space & time)
        x is a background trend added correctly (check slope)
        x is a background noise added correctly (check variance)
        x does the same seed produce the same data
    x Adding shifts -> use mock methods for shape and shift
        x is the shape of the grid unaltered
        x are the time series parameters stored correctly (check multiple)

Legend:
    - : to do
    x : done
"""

###################################################################################################
#################################### Initialization ###############################################
###################################################################################################

def test_ponds_initialization_shape():

    # 1) setup variables
    # - world grid case
    lat_size1 = 10
    lon_size1 = 20
    time_size1 = 30
    # - single case
    lat_size2 = 1
    lon_size2 = 1
    time_size2 = 1

    # 2) call function
    ponds1 = PONDS(lat_size=lat_size1,
                   lon_size=lon_size1,
                   time_size=time_size1)
    ponds2 = PONDS(lat_size=lat_size2,
                   lon_size=lon_size2,
                   time_size=time_size2)

    # 3) check results
    assert ponds1.data.shape == (time_size1, lat_size1, lon_size1)
    assert ponds2.data.shape == (time_size2, lat_size2, lon_size2)

def test_ponds_background_trend(err_tol=1e-4):

    # 1) setup variables
    trend = 0.1

    # 2) call function
    # - (2x2) grid, check trend in all 4 time series
    ponds = PONDS(lat_size=2,
                  lon_size=2,
                  time_size=10,
                  background_trend=trend,
                  background_noise=0.0)

    # 3) check results
    ts1 = ponds.data[:,0,0].values
    ts2 = ponds.data[:,0,1].values
    ts3 = ponds.data[:,1,0].values
    ts4 = ponds.data[:,1,1].values
    # - Fit a linear regression to confirm the slope
    slope1 = (ts1[-1] - ts1[0]) / (ponds.time[-1] - ponds.time[0])
    slope2 = (ts2[-1] - ts2[0]) / (ponds.time[-1] - ponds.time[0])
    slope3 = (ts3[-1] - ts3[0]) / (ponds.time[-1] - ponds.time[0])
    slope4 = (ts4[-1] - ts4[0]) / (ponds.time[-1] - ponds.time[0])
    # - 
    assert np.isclose(slope1, trend, atol=err_tol)
    assert np.isclose(slope2, trend, atol=err_tol)
    assert np.isclose(slope3, trend, atol=err_tol)
    assert np.isclose(slope4, trend, atol=err_tol)


def test_ponds_background_noise_variance(err_tol=1e-4):

    # 1) setup variables
    noise = 0.01

    # 2) call function
    # - (2x2) grid, check variance in all 4 time series
    ponds = PONDS(lat_size=2,
                  lon_size=2,
                  time_size=int(1/err_tol),
                  background_trend=0.0,
                  background_noise=noise)
    
    # 3) check results
    ts1 = ponds.data[:,0,0].values
    ts2 = ponds.data[:,0,1].values
    ts3 = ponds.data[:,1,0].values
    ts4 = ponds.data[:,1,1].values
    # -
    assert np.isclose(np.std(ts1), noise, atol=err_tol)
    assert np.isclose(np.std(ts2), noise, atol=err_tol)
    assert np.isclose(np.std(ts3), noise, atol=err_tol)
    assert np.isclose(np.std(ts4), noise, atol=err_tol)

def test_ponds_same_seed_reproducibility(R=5):
    """
    Check if same random seed produces the same data. Also check that different
    seeds produce different data.
    R: int
        Number of random seeds to test.
    """

    for _ in range(R):
        # 1) setup variables
        seed1 = np.random.randint(0, R)
        seed2 = np.random.randint(0, R)
        if seed1 == seed2:  # ensure different seeds
            seed2 = (seed1 + 1) % R
        trend = np.random.uniform(0, 0.1)
        noise = np.random.uniform(0, 0.05)

        # 2) call function
        ponds1 = PONDS(random_seed=seed1,           # <- benchmark
                       background_trend=trend,
                       background_noise=noise)
        ponds2 = PONDS(random_seed=seed1,           # <- same seed as benchmark
                       background_trend=trend,
                       background_noise=noise)
        ponds3 = PONDS(random_seed=seed2,           # <- different seed
                       background_trend=trend,
                       background_noise=noise)

        # 3) check results
        assert np.array_equal(ponds1.data.values, ponds2.data.values)
        assert not np.array_equal(ponds1.data.values, ponds3.data.values)

###################################################################################################
######################################## Add Shifts ###############################################
###################################################################################################

def test_add_shift_shape_unaltered(mock_blob, mock_shift):

    # 1) setup variables
    lat_size = 10
    lon_size = 20
    time_size = 30
    original_shape = (time_size, lat_size, lon_size)
    ponds = PONDS(lat_size=lat_size,
                  lon_size=lon_size,
                  time_size=time_size)

    # 2) call function
    ponds.add_shift(mock_blob, mock_shift)

    # 3) check results
    assert ponds.data.shape == original_shape

def test_add_shift_parameter_storage(mock_blob, mock_shift):

    # 1) setup variables
    ponds = PONDS(lat_size=2,
                  lon_size=2,
                  time_size=10)

    # 2) call function
    ponds.add_shift(mock_blob, mock_shift)
    ponds.add_shift(mock_blob, mock_shift)

    # 3) check results
    stored = ponds.ts_param["shift_id"].values
    assert stored[0,0] == [-1, -1]
    assert stored[0,1] == None
    assert stored[1,0] == None
    assert stored[1,1] == None


