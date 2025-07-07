import warnings
import numpy as np
from ponds import PONDS, SINGLE

"""

Things to Test:
    x check if the blob mask is a boolean 2D array of shape (1, 1)
    x check if non-single ponds object is collapsed to single and all former data is lost
    x check if user is warned when using SINGLE on a non-single ponds object
    x check if all former data is kept for single ponds object input
        x ts_param
        x ts_data

Legend:
    - : to do
    x : done
"""


def test_single_blob_mask_shape_and_type():
    """
    - check if the blob mask is a boolean 2D array of shape (1, 1)
    - check if the blob mask has a single True value
    """

    # 1) setup variables
    ponds = PONDS(lat_size=1, lon_size=1)

    # 2) call function
    single_blob = SINGLE(ponds)
    blob_mask = single_blob.make_blob()

    # 3) check results
    assert blob_mask.shape == (1, 1)
    assert blob_mask.dtype == np.bool_, "Blob mask must be boolean."
    assert blob_mask[0, 0], "Blob mask must have a single True value."


def test_single_blob_collapses_non_single_ponds():
    """
    - check if non-single ponds object is collapsed to single and all former data is lost
    - check if user is warned when using SINGLE on a non-single ponds object
    """

    # 1) setup variables
    lat_size = 5
    lon_size = 5
    time_size = 10
    ponds = PONDS(lat_size=lat_size, lon_size=lon_size, time_size=time_size)
    original_data = ponds.data.copy(deep=True)

    # 2) call function
    single_blob = SINGLE(ponds)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = single_blob.make_blob()

        # - check if warning is raised when SINGLE is used on a non-single ponds object
        assert any(
            "Shape method SINGLE used on a non-single ponds object"
            in str(warning.message)
            for warning in w
        ), "Warning not raised when SINGLE is used on non-single ponds object."

    # 3) check results
    # - Check ponds is now single
    assert ponds.lat_size == 1
    assert ponds.lon_size == 1
    assert ponds.data.shape == (time_size, 1, 1)

    # - Confirm that non of the original data is kept and used in the single
    for i in range(lat_size):
        for j in range(lon_size):
            ts_original = original_data.isel(lat=i, lon=j).values
            assert not np.array_equal(
                ts_original, ponds.data.isel(lat=0, lon=0).values
            ), (
                f"Original time series at lat={i}, lon={j} is still present in single ponds object."
            )


def test_single_blob_keeps_existing_single_ponds_data():
    """
    - check if calling SINGLE on a single ponds object keeps the existing data
    - check if ponds.ts_param is unchanged for existing variables
    """

    # 1) setup variables
    ponds = PONDS(lat_size=1, lon_size=1, time_size=50)
    original_data = ponds.data.copy()
    # - Add a variable to ts_param
    ponds.ts_param["test_param"] = (
        ("lat", "lon"),
        np.empty((ponds.lat_size, ponds.lon_size), dtype=object),
    )
    ponds.ts_param["test_param"].values[0, 0] = ["value"]

    # 2) call function
    single_blob = SINGLE(ponds)
    _ = single_blob.make_blob()

    # 3) check results
    # - Check data is unchanged
    assert ponds.data.equals(original_data)

    # - Check ts_param is unchanged for existing variable
    assert "test_param" in ponds.ts_param
    assert ponds.ts_param["test_param"].values[0, 0] == ["value"]
