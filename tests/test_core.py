import numpy as np
from ponds import PONDS, ELLIPSIS, SIGMOID, WhiteNoise

"""
Testing the PONDS core class and its methods.

Things to Test:
    x Initialization of PONDS object
        x does the ts data grid exist and have the right shape (space & time)
        x do the same custom parameters produce the same data
    - Adding perturbations -> use mock methods for perturbation
        x is the shape of the time series unaltered
        x is the perturbation metadata added to the ts attributes
        x does ponds take a custom perturbation_key for the variable name
    x Adding shifts -> use mock methods for shape and shift
        x is the shape of the grid unaltered
        x is the shift-type stored as an attribute on Dataset Level
        x is the shift- and shape-type stored as an attribute on DataArray Level
    x Reproducibility using stored attributes -> use SIGMOID and ELLIPSIS to benchmark
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


def test_ponds_reproducibility(R=5):
    """
    - check if same input parameters produces the same data
    R: int
        Number of random seeds to test.
    """

    for _ in range(R):
        # 1) setup variables
        lat_size1 = np.random.randint(1, 100)
        lon_size1 = np.random.randint(1, 100)
        time_size1 = np.random.randint(1, 100)
        lat_size2 = np.random.randint(1, 100)
        if lat_size2 == lat_size1:
            lat_size2 += 1
        lon_size2 = np.random.randint(1, 100)
        if lon_size2 == lon_size1:
            lon_size2 += 1
        time_size2 = np.random.randint(1, 100)
        if time_size2 == time_size1:
            time_size2 += 1

        # 2) call function
        ponds1 = PONDS(
            lat_size=lat_size1,
            lon_size=lon_size1,
            time_size=time_size1,
        )
        ponds2 = PONDS(
            lat_size=lat_size1,
            lon_size=lon_size1,
            time_size=time_size1,
        )
        ponds3 = PONDS(
            lat_size=lat_size2,
            lon_size=lon_size2,
            time_size=time_size2,
        )

        # 3) check results
        assert np.array_equal(ponds1.data.ts.values, ponds2.data.ts.values)
        assert not np.array_equal(ponds1.data.ts.values, ponds3.data.ts.values)


###################################################################################################
#################################### Add Perturbations ############################################
###################################################################################################


def test_add_perturbation_shape(mock_shape, mock_perturbation):
    """
    - check if the shape of the time series is unaltered after adding a perturbation
    - check if the perturbation metadata is added to the ts attributes
    """

    # 1) setup variables
    lat_size = 10
    lon_size = 20
    time_size = 30
    original_shape = (time_size, lat_size, lon_size)
    perturbation_key = "custom_key"
    ponds = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        perturbation_key=perturbation_key,
    )

    # 2) call function
    ponds.add_perturbation(
        perturbation_method=mock_perturbation(),
        shape_method=mock_shape(),
    )

    # 3) check results
    # - check if the shape of the time series is unaltered
    assert ponds.data.ts.shape == original_shape
    # - check if the perturbation metadata is added to the ts attributes
    assert (
        ponds.data.ts.attrs[perturbation_key + "1_" + "perturbation_type"]
        == mock_perturbation
    )
    # assert perturbation_key+"shape_type" in ponds.data.ts.attrs
    assert ponds.data.ts.attrs[perturbation_key + "1_" + "shape_type"] == mock_shape


###################################################################################################
######################################## Add Shifts ###############################################
###################################################################################################


def test_add_shift_shape_unaltered(mock_shape, mock_shift):
    # 1) setup variables
    lat_size = 10
    lon_size = 20
    time_size = 30
    original_shape = (time_size, lat_size, lon_size)
    ponds = PONDS(lat_size=lat_size, lon_size=lon_size, time_size=time_size)

    # 2) call function
    ponds.add_shift(
        shift_method=mock_shift(),
        shape_method=mock_shape(),
    )

    # 3) check results
    assert ponds.data.ts.shape == original_shape


def test_add_shift_parameter_storage(mock_shape, mock_shift):
    """
    - check if the shift- and shape type is stored as an attribute in the xarray DataArray
    - check if PONDS takes a custom cluster_key for the variable name
    """

    # 1) setup variables
    lat_size = 2
    lon_size = 2
    time_size = 10
    num_shifts = 2
    cluster_key = "custom_key"
    ponds = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        cluster_key=cluster_key,
    )

    # 2) call function
    for _ in range(num_shifts):
        ponds.add_shift(
            shift_method=mock_shift(),
            shape_method=mock_shape(),
        )

    # 3) check results
    # - check DataArray attributes
    # -- check if shift_type is stored
    assert ponds.data.custom_key1.attrs["shift_type"] == mock_shift
    assert ponds.data.custom_key2.attrs["shift_type"] == mock_shift
    # -- check if shape_type is stored
    assert ponds.data.custom_key1.attrs["shape_type"] == mock_shape
    assert ponds.data.custom_key2.attrs["shape_type"] == mock_shape


###################################################################################################
####################################### Reproducibility ###########################################
###################################################################################################

# use SIGMOID and ELLIPSIS to benchmark reproducibility


def test_reproducibility(R=5):
    """
    - check if stored attributes can be used to reproduce a shift
        -> do this using SIGMOID, ELLIPSIS and WhiteNoise

        Idea: - create a PONDS object using a certain random seed
                - add a shift using SIGMOID, ELLIPSIS and WhiteNoise, each with a random seed
                - extract all parameters from the Dataset
              - reproduce the ponds object using the same random seed
              - check if the time series and masks are equal
    """

    # 1) setup variables
    # - basic parameters
    lat_size = 100
    lon_size = 100
    time_size = 100
    noise = 0.05

    for _ in range(R):
        rs1 = np.random.randint(0, 100)
        rs2 = np.random.randint(100, 200)
        rs3 = np.random.randint(200, 300)

        # - create PONDS object
        ponds = PONDS(
            lat_size=lat_size,
            lon_size=lon_size,
            time_size=time_size,
        )
        # - add two perturbations using WhiteNoise
        ponds.add_perturbation(
            perturbation_method=WhiteNoise(magnitude=noise, random_seed=rs1),
        )
        # - add a shift using SIGMOID and ELLIPSIS
        ponds.add_shift(
            shape_method=ELLIPSIS(random_seed=rs2), shift_method=SIGMOID(random_seed=rs3)
        )
        # - extract all parameters from the Dataset
        ponds_dict_ex = ponds.get_ponds_dict()
        perturbation_instance_ex, shape_instance_p_ex = ponds.get_perturbation_instance(
            "p1"
        )
        shift_instance_ex, shape_instance_s_ex = ponds.get_cluster_instance("c1")
        # - reproduce the shift using the stored parameters

        # 2) call function
        # - reproduce the ponds object with extracted parameters
        ponds_rep = PONDS(**ponds_dict_ex)
        ponds_rep.add_perturbation(
            perturbation_method=perturbation_instance_ex,
            shape_method=shape_instance_p_ex,
        )
        # - add a shift using the stored parameters
        ponds_rep.add_shift(
            shift_method=shift_instance_ex,
            shape_method=shape_instance_s_ex,
        )

        # 3) check results
        # - check if attributes are equal
        assert (
            ponds.get_ponds_dict() == ponds_rep.get_ponds_dict()
            # ponds.data.attrs["ponds_params"] == ponds_rep.data.attrs["ponds_params"]
        ), "PONDS parameters do not match."

        # - check if time series are equal
        assert np.array_equal(ponds.data.ts.values, ponds_rep.data.ts.values), (
            "Reproduced time series do not match the original time series."
        )
        # - check if the masks are equal
        assert np.array_equal(ponds.data.c1.values, ponds_rep.data.c1.values), (
            "Reproduced mask for c1 does not match the original mask."
        )
