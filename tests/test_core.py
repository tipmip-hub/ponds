import numpy as np
from ponds import PONDS, GAUSS, SIGMOID

"""
Testing the PONDS core class and its methods.

Things to Test:
    x Initialization of PONDS object
        x does the ts data grid exist and have the right shape (space & time)
        x is a background trend added correctly (check slope)
        x is a background noise added correctly (check variance)
        x do the same custom parameters produce the same data
    x Adding shifts -> use mock methods for shape and shift
        x is the shape of the grid unaltered
        x is the shift-type stored as an attribute on Dataset Level
        x is the shift- and shape-type stored as an attribute on DataArray Level
    x Reproducibility using stored attributes -> use SIGMOID and GAUSS to benchmark
        x can I reproduce a shift using the stored attributes?


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
    ponds1 = PONDS(lat_size=lat_size1, lon_size=lon_size1, time_size=time_size1)
    ponds2 = PONDS(lat_size=lat_size2, lon_size=lon_size2, time_size=time_size2)

    # 3) check results
    assert ponds1.data.ts.shape == (time_size1, lat_size1, lon_size1)
    assert ponds2.data.ts.shape == (time_size2, lat_size2, lon_size2)


def test_ponds_background_trend(err_tol=1e-4):
    # 1) setup variables
    trend = 0.1

    # 2) call function
    # - (2x2) grid, check trend in all 4 time series
    ponds = PONDS(
        lat_size=2,
        lon_size=2,
        time_size=10,
        background_trend=trend,
        background_noise=0.0,
    )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values
    ts2 = ponds.data.ts[:, 0, 1].values
    ts3 = ponds.data.ts[:, 1, 0].values
    ts4 = ponds.data.ts[:, 1, 1].values
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
    ponds = PONDS(
        lat_size=2,
        lon_size=2,
        time_size=int(1 / err_tol),
        background_trend=0.0,
        background_noise=noise,
    )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values
    ts2 = ponds.data.ts[:, 0, 1].values
    ts3 = ponds.data.ts[:, 1, 0].values
    ts4 = ponds.data.ts[:, 1, 1].values
    # -
    assert np.isclose(np.std(ts1), noise, atol=err_tol)
    assert np.isclose(np.std(ts2), noise, atol=err_tol)
    assert np.isclose(np.std(ts3), noise, atol=err_tol)
    assert np.isclose(np.std(ts4), noise, atol=err_tol)


def test_ponds_reproducibility(R=5):
    """
    - check if same input parameters produces the same data
    - check that different seeds produce different data, even though rest of the parameters are the same
    R: int
        Number of random seeds to test.
    """

    for _ in range(R):
        # 1) setup variables
        seed1 = np.random.randint(0, 100)
        seed2 = np.random.randint(100, 200)
        trend = np.random.uniform(0, 0.1)
        noise = np.random.uniform(0, 0.05)

        # 2) call function
        ponds1 = PONDS(
            random_seed=seed1,  # <- benchmark
            background_trend=trend,
            background_noise=noise,
        )
        ponds2 = PONDS(
            random_seed=seed1,  # <- same seed as benchmark
            background_trend=trend,
            background_noise=noise,
        )
        ponds3 = PONDS(
            random_seed=seed2,  # <- different seed
            background_trend=trend,
            background_noise=noise,
        )

        # 3) check results
        assert np.array_equal(ponds1.data.ts.values, ponds2.data.ts.values)
        assert not np.array_equal(ponds1.data.ts.values, ponds3.data.ts.values)


###################################################################################################
######################################## Add Shifts ###############################################
###################################################################################################


def test_add_shift_shape_unaltered(mock_blob, mock_shift):
    # 1) setup variables
    lat_size = 10
    lon_size = 20
    time_size = 30
    original_shape = (time_size, lat_size, lon_size)
    ponds = PONDS(lat_size=lat_size, lon_size=lon_size, time_size=time_size)

    # 2) call function
    ponds.add_shift(
        shift_method=mock_shift(),
        shape_method=mock_blob(),
    )

    # 3) check results
    assert ponds.data.ts.shape == original_shape


def test_add_shift_parameter_storage(mock_blob, mock_shift):
    """
    - check if the shift- and shape type is stored as an attribute in the xarray DataArray
    - check if PONDS takes a custom cluster_key for the variable name
    """

    # 1) setup variables
    lat_size = 2
    lon_size = 2
    time_size = 10
    background_noise = 0.01
    background_trend = 0.005
    num_shifts = 2
    cluster_key = "custom_key"
    random_seed = 2
    ponds = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        random_seed=random_seed,
        background_noise=background_noise,
        background_trend=background_trend,
        cluster_key=cluster_key,
    )

    # 2) call function
    for _ in range(num_shifts):
        ponds.add_shift(
            shift_method=mock_shift(),
            shape_method=mock_blob(),
        )

    # 3) check results
    # - check Dataset attributes
    ponds_dict = ponds.get_ponds_dict()#data.attrs["ponds_params"]
    assert ponds_dict["lat_size"] == lat_size
    assert ponds_dict["lon_size"] == lon_size
    assert ponds_dict["time_size"] == time_size
    assert ponds_dict["background_trend"] == background_trend
    assert ponds_dict["background_noise"] == background_noise
    assert ponds_dict["cluster_key"] == cluster_key
    assert ponds_dict["random_seed"] == random_seed

    #assert ponds.data.attrs["num_clusters"] == num_shifts
    assert ponds.data.custom_key1.attrs["shift_type"] == "MOCKSHIFT"
    assert ponds.data.custom_key2.attrs["shift_type"] == "MOCKSHIFT"

    # - check DataArray attributes
    # -- check if noise and trend are stored
    # -- check if shift_type is stored
    assert ponds.data.custom_key1.attrs["shift_type"] == "MOCKSHIFT"
    assert ponds.data.custom_key2.attrs["shift_type"] == "MOCKSHIFT"
    # -- check if shape_type is stored
    assert ponds.data.custom_key1.attrs["shape_type"] == "MOCKBLOB"
    assert ponds.data.custom_key2.attrs["shape_type"] == "MOCKBLOB"


###################################################################################################
####################################### Reproducibility ###########################################
###################################################################################################

# use SIGMOID and GAUSS to benchmark reproducibility


def test_reproducibility(R=10):
    """
    - check if stored attributes can be used to reproduce a shift
        -> do this using SIGMOID and GAUSS

        Idea: - create a PONDS object using a certain random seed
                - add a shift using SIGMOID and GAUSS, each with a random seed
                - extract all parameters from the Dataset
              - reproduce the ponds object using the same random seed
                - add a shift using SIGMOID and GAUSS, each with the stored parameters (not the seed)
              - check if the time series and masks are equal
    """

    # 1) setup variables
    # - basic parameters
    lat_size = 100
    lon_size = 100
    time_size = 100
    noise = 0.01
    trend = 0.005

    for _ in range(R):
        rs1 = np.random.randint(0, 100)
        rs2 = np.random.randint(100, 200)
        rs3 = np.random.randint(200, 300)

        # - create PONDS object
        ponds = PONDS(
            lat_size=lat_size,
            lon_size=lon_size,
            time_size=time_size,
            background_noise=noise,
            background_trend=trend,
            random_seed=rs1,
        )
        # - add a shift using SIGMOID and GAUSS
        ponds.add_shift(
            shape_method=GAUSS(random_seed=rs2), shift_method=SIGMOID(random_seed=rs3)
        )
        # - extract all parameters from the Dataset
        ponds_dict_ex = ponds.get_ponds_dict()
        shift_dict_ex, shape_dict_ex = ponds.get_cluster_dict("c1")
        # - reproduce the shift using the stored parameters

        # 2) call function
        # - reproduce the ponds object with extracted parameters
        ponds_rep = PONDS(**ponds_dict_ex)
        ponds_rep.add_shift(
            shape_method=GAUSS(**shape_dict_ex), shift_method=SIGMOID(**shift_dict_ex)
        )

        # 3) check results
        # - check if attributes are equal
        assert (
            ponds.get_ponds_dict() == ponds_rep.get_ponds_dict()
            #ponds.data.attrs["ponds_params"] == ponds_rep.data.attrs["ponds_params"]
        ), "PONDS parameters do not match."

        # - check if time series are equal
        assert np.array_equal(ponds.data.ts.values, ponds_rep.data.ts.values), (
            "Reproduced time series do not match the original time series."
        )
        # - check if the masks are equal
        assert np.array_equal(ponds.data.c1.values, ponds_rep.data.c1.values), (
            "Reproduced mask for c1 does not match the original mask."
        )
