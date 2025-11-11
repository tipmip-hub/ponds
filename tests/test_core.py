import numpy as np
from ponds import PONDS, ELLIPSIS, SIGMOID, WhiteNoise

"""
Testing the PONDS core class and its methods.

Things to Test:
    x Initialization of PONDS object
        x does the ts data grid exist and have the right shape (space & time)
        x do the same custom parameters produce the same data
    - Adding backgrounds -> use mock methods for background
        x is the shape of the time series unaltered
        x is the background metadata added to the ts attributes
        x does ponds take a custom background_key for the variable name
    x Adding shifts -> use mock methods for shape and shift
        x is the shape of the grid unaltered
        x is the shift-type stored as an attribute on Dataset Level
        x is the shift- and shape-type stored as an attribute on DataArray Level
    x Reproducibility using stored attributes -> use SIGMOID and ELLIPSIS to benchmark
        x can I reproduce a shift using the stored attributes?
    - Delete functions for backgrounds and shifts
        - can I delete a background and shift and are the attributes removed correctly?


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
#################################### Add Backgrounds ############################################
###################################################################################################


def test_add_background_shape(mock_shape, mock_background):
    """
    - check if the shape of the time series is unaltered after adding a background
    - check if the background metadata is added to the ts attributes
    """

    # 1) setup variables
    lat_size = 10
    lon_size = 20
    time_size = 30
    original_shape = (time_size, lat_size, lon_size)
    background_key = "custom_key"
    ponds = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        background_key=background_key,
    )

    # 2) call function
    ponds.add_background(
        background_method=mock_background(),
        shape_method=mock_shape(),
    )

    # 3) check results
    # - check if the shape of the time series is unaltered
    assert ponds.data.ts.shape == original_shape
    # - check if the background metadata is added to the ts attributes
    assert (
        ponds.data.ts.attrs[background_key + "1_" + "background_type"]
        == mock_background
    )
    # assert background_key+"shape_type" in ponds.data.ts.attrs
    assert ponds.data.ts.attrs[background_key + "1_" + "shape_type"] == mock_shape


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
        # - add two backgrounds using WhiteNoise
        ponds.add_background(
            background_method=WhiteNoise(
                magnitude=noise,
                background_seed=rs1,
            ),
        )
        # - add a shift using SIGMOID and ELLIPSIS
        ponds.add_shift(
            shape_method=ELLIPSIS(shape_seed=rs2),
            shift_method=SIGMOID(shift_seed=rs3),
        )
        # - extract all parameters from the Dataset
        ponds_dict_ex = ponds.get_ponds_dict()
        background_instance_ex, shape_instance_b_ex = ponds.get_background_instance(
            "b1"
        )
        shift_instance_ex, shape_instance_s_ex, correlation_instance_ex = (
            ponds.get_cluster_instance("c1")
        )
        # - reproduce the shift using the stored parameters

        # 2) call function
        # - reproduce the ponds object with extracted parameters
        ponds_rep = PONDS(**ponds_dict_ex)
        ponds_rep.add_background(
            background_method=background_instance_ex,
            shape_method=shape_instance_b_ex,
        )
        # - add a shift using the stored parameters
        ponds_rep.add_shift(
            shift_method=shift_instance_ex,
            shape_method=shape_instance_s_ex,
            correlation_method=correlation_instance_ex,
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


###################################################################################################
############################################ Deletion #############################################
###################################################################################################


def test_delete_background():
    """
    - check if a background can be deleted and if the attributes are removed correctly
        -> use WhiteNoise as background method
        -> doing this by creating two ponds objects
            > - first ponds object: add two backgrounds and delete the first one
            > - second ponds object: add only the second background
            > - compare the time series of both ponds objects -> should be identical
            > - check if the attributes related to the deleted background are removed correctly
    """

    # 1) setup variables
    lat_size = 30
    lon_size = 60
    time_size = 100

    # create first ponds object with two backgrounds
    ponds1 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )

    # add two backgrounds to first ponds object
    ponds1.add_background(
        background_method=WhiteNoise(
            background_seed=42,
        ),
    )
    ponds1.add_background(
        background_method=WhiteNoise(
            background_seed=43,
        ),
    )

    # delete the first background
    ponds1.delete_background("b1")

    # create second ponds object with only the second background
    ponds2 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )
    ponds2.add_background(
        background_method=WhiteNoise(
            background_seed=43,  # <- needs to be the same as in ponds1, b2
        ),
    )

    # 2) check results
    # - check if the time series of both ponds objects are identical
    assert np.all(
        np.isclose(ponds1.data.ts.values, ponds2.data.ts.values, rtol=1e-05, atol=1e-08)
    ), "Time series after deleting background do not match the expected time series."
    # - check if the attributes related to the deleted background are removed
    assert "b1_background_type" not in ponds1.data.ts.attrs, (
        "Attributes related to deleted background 'b1' were not removed."
    )
    assert "b1_shape_type" not in ponds1.data.ts.attrs, (
        "Attributes related to deleted background 'b1' were not removed."
    )


def test_delete_shift():
    """
    - check if a shift can be deleted and if the attributes are removed correctly
        -> use SIGMOID as shift method
        -> doing this by creating two ponds objects
            > - first ponds object: add two shifts and delete the first one
            > - second ponds object: add only the second shift
            > - compare the time series of both ponds objects -> should be identical
            > - check if the attributes related to the deleted shift are removed correctly
    """

    # 1) setup variables
    lat_size = 10
    lon_size = 20
    time_size = 30

    # create first ponds object with two shifts
    ponds1 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )

    # add two shifts to first ponds object
    ponds1.add_shift(
        shift_method=SIGMOID(
            shift_seed=42,
        ),
    )
    ponds1.add_shift(
        shift_method=SIGMOID(
            shift_seed=43,
        ),
    )

    # delete the first shift
    ponds1.delete_shift("c1")

    # create second ponds object with only the second shift
    ponds2 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )
    ponds2.add_shift(
        shift_method=SIGMOID(
            shift_seed=43,
        ),
    )

    # 2) check results
    # - check if the time series of both ponds objects are identical
    assert np.all(
        np.isclose(ponds1.data.ts.values, ponds2.data.ts.values, rtol=1e-05, atol=1e-08)
    ), "Time series after deleting shift do not match the expected time series."
    # - check if the attributes related to the deleted shift are removed
    assert "c1" not in ponds1.data.data_vars, (
        "Attributes related to deleted shift 'c1' were not removed."
    )
    assert "c2" in ponds1.data.data_vars, (
        "Attributes related to existing shift 'c2' were incorrectly removed."
    )
