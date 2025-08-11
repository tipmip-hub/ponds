import numpy as np
import pytest
from ponds import PONDS, CUSTOM, SIGMOID

"""

Things to Test:
    x check if the output mask is exactly the same as the input mask
    x check if wrong shape or non-boolean output raises an error
    x use in PONDS.add_shift() function and make sure it works correctly (use SIGMOID)

Legend:
    - : to do
    x : done
"""

###################################################################################################


def test_custom_input_output_equal():
    """
    - check if the output mask is exactly the same as the input mask
    """

    # 1) setup variables
    lat_size = np.random.randint(10, 100)
    lon_size = np.random.randint(10, 100)
    shape_mask = np.random.choice([True, False], size=(lat_size, lon_size))
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)

    # 2) call function
    custom = CUSTOM(shape_mask=shape_mask)
    shape_output = custom.make_shape(lats=lats, lons=lons)

    # 3) check results
    assert shape_output.shape == (lat_size, lon_size), (
        "Output shape does not match input shape."
    )
    assert np.array_equal(shape_output, shape_mask), (
        "Output mask does not match input mask."
    )


def test_custom_wrong_shape_type_raises_error():
    """
    - check if wrong shape or non-boolean output raises an error
    """
    lat_size = 14
    lon_size = 26
    shape_mask_wrong_shape1 = np.random.choice(
        [True, False], size=(lat_size + 1, lon_size)
    )
    shape_mask_wrong_shape2 = np.random.choice(
        [True, False], size=(lat_size, lon_size - 1)
    )
    shape_mask_wrong_type1 = np.random.rand(lat_size, lon_size)  # float array
    shape_mask_wrong_type2 = np.random.randint(
        0, 2, size=(lat_size, lon_size)
    )  # int array
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)

    # 2) call function + 3) check results (expecting ValueError)
    with pytest.raises(ValueError):
        custom = CUSTOM(shape_mask=shape_mask_wrong_shape1)
        _ = custom.make_shape(lats=lats, lons=lons)

    with pytest.raises(ValueError):
        custom = CUSTOM(shape_mask=shape_mask_wrong_shape2)
        _ = custom.make_shape(lats=lats, lons=lons)

    with pytest.raises(ValueError):
        custom = CUSTOM(shape_mask=shape_mask_wrong_type1)
        _ = custom.make_shape(lats=lats, lons=lons)

    with pytest.raises(ValueError):
        custom = CUSTOM(shape_mask=shape_mask_wrong_type2)
        _ = custom.make_shape(lats=lats, lons=lons)


def test_custom_add_shift():
    """
    - use in PONDS.add_shift() function and make sure it works correctly
    """

    # 1) setup variables
    lat_size = 2
    lon_size = 2
    shape_mask = [[True, False], [False, True]]
    pd = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=3,
    )

    # 2) call function
    pd.add_shift(
        shape_method=CUSTOM(shape_mask=shape_mask),
        shift_method=SIGMOID(
            central_time=1,
            shift_magnitude=1.0,
        ),
    )

    # 3) check results
    ts = pd.get_ts()
    # - check FALSE values
    assert ts[0, 0, 1] == 0.0
    assert ts[-1, 0, 1] == 0.0
    assert ts[0, 1, 0] == 0.0
    assert ts[-1, 1, 0] == 0.0
    # - check TRUE values
    assert ts[0, 0, 0] < 0.5
    assert ts[-1, 0, 0] > 0.5
    assert ts[0, 1, 1] < 0.5
    assert ts[-1, 1, 1] > 0.5
