import numpy as np
from ponds import PONDS, BLOB

"""

Things to Test:
    - Random Blob
        x must return boolean 2D array
        x blob mask must have the same shape as the ponds grid
        x does the same seed produce the same blob mask
        x check compatibility with the PONDS.add_shift() function
        x check if in blob mask at least one cell is True
        - do the blobs scale with the grid size
    - Custom Blob
        x are the correct shape-params stored as attributes on the DataArray Level
        x is the location correct
        - is the shape correct

Legend:
    - : to do
    x : done
"""

###################################################################################################
######################################## Random Blob ##############################################
###################################################################################################


def test_blob_output():
    """
    Checking two cases: small and large grid size.
        - check if the output is a boolean 2D array
        - check if the shape of the output matches the ponds grid size
    """
    # >> small ponds
    # 1) setup variables
    lat_size = 10
    lon_size = 20
    ponds_small = PONDS(lat_size=lat_size, lon_size=lon_size)
    # 2) call function
    blob_small = BLOB()
    blob_small = blob_small.make_shape(lats=ponds_small.lats, lons=ponds_small.lons)
    # 3) check results
    assert isinstance(blob_small, np.ndarray), "Blob must be a numpy array."
    assert blob_small.dtype == bool, "Blob must be a boolean array."
    assert blob_small.shape == (lat_size, lon_size)

    # >> large ponds
    # 1) setup variables
    lat_size = 200
    lon_size = 300
    ponds_large = PONDS(lat_size=lat_size, lon_size=lon_size)
    # 2) call function
    blob_large = BLOB()
    blob_large = blob_large.make_shape(lats=ponds_large.lats, lons=ponds_large.lons)
    # 3) check results
    assert isinstance(blob_large, np.ndarray), "Blob must be a numpy array."
    assert blob_large.dtype == bool, "Blob must be a boolean array."
    assert blob_large.shape == (lat_size, lon_size)


def test_blob_same_seed_produces_same_blob_mask(R=5):
    """
    - check if the same random seed produces the same blob mask
    """

    # 1.1) setup variables
    lat_size = 100
    lon_size = 200

    for _ in range(R):
        # 1.2) setup variables
        seed_blob = np.random.randint(0, 1000)
        ponds1 = PONDS(
            lat_size=lat_size,
            lon_size=lon_size,
        )
        ponds2 = PONDS(
            lat_size=lat_size,
            lon_size=lon_size,
        )

        # 2) call function
        blob1 = BLOB(random_seed=seed_blob)
        blob1 = blob1.make_shape(lats=ponds1.lats, lons=ponds1.lons)
        blob2 = BLOB(random_seed=seed_blob)
        blob2 = blob2.make_shape(lats=ponds2.lats, lons=ponds2.lons)

        # 3) check results
        assert np.array_equal(blob1, blob2), "Blobs with same seed should be identical."


def test_blob_add_shift_integration(mock_shift):
    """
    - test compatibility with PONDS.add_shift()
    - check if blob_mask is not all False
    """

    # 1) setup variables
    ponds = PONDS(lat_size=10, lon_size=20, time_size=30)

    # 2) call function
    ponds.add_shift(shape_method=BLOB(), shift_method=mock_shift())

    # 3) check results
    # - Check if cluster mask is True at least somewhere
    shift_grid = ponds.data.c1.values
    assert any(cell is not False for row in shift_grid for cell in row), (
        "Shift should be applied somewhere on grid."
    )


###################################################################################################
######################################## Custom Blob ##############################################
###################################################################################################


def test_blob_custom_parameter_storage():
    """
    - check if the custom parameters are stored correctly in the GAUSS class
    - expected behavior:
        - center_lat, center_lon, ecc_lat, ecc_lon should be stored as attributes
    """

    # 1) setup variables
    custom_center_lat = 42.0
    custom_center_lon = 63.0
    custom_ecc_lat = 5
    custom_ecc_lon = 10

    # 2) call function
    shape_method = BLOB(
        center_lat=custom_center_lat,
        center_lon=custom_center_lon,
        ecc_lat=custom_ecc_lat,
        ecc_lon=custom_ecc_lon,
    )

    # 3) check results
    shape_dict = shape_method.param_dict
    assert shape_dict["shape_type"] == BLOB, "Type should be 'BLOB'."
    assert shape_dict["center_lat"] == custom_center_lat, (
        f"Expected center_lat: {custom_center_lat}, got: {shape_dict['center_lat']}"
    )
    assert shape_dict["center_lon"] == custom_center_lon, (
        f"Expected center_lon: {custom_center_lon}, got: {shape_dict['center_lon']}"
    )
    assert shape_dict["ecc_lat"] == custom_ecc_lat, (
        f"Expected ecc_lat: {custom_ecc_lat}, got: {shape_dict['ecc_lat']}"
    )
    assert shape_dict["ecc_lon"] == custom_ecc_lon, (
        f"Expected ecc_lon: {custom_ecc_lon}, got: {shape_dict['ecc_lon']}"
    )


def test_blob_custom_parameters_center(err=1):
    """
    - Test if custom parameters for the central lat and lon for the Blob are used correctly.
    - expected behavior:
        - center_lat: mean of the latitudes in the blob should be equal to the center_lat parameter
        - center_lon: mean of the longitudes in the blob should be equal to the center_lon parameter
    Arguments:
        - err: allowed offset for the index of the center lat and lon
    """

    # 1) setup variables
    lat_size = 180
    lon_size = 360

    custom_center_lat = [-90, -88, 0, 42, 90]  # needs to be in range [-90, 90]
    custom_center_lon = [-179, -110, 0, 63, 179]  # needs to be in range [-180, 180]
    custom_ecc_lat = 0.75  # small eccentricity for a sharp blob (only one cell)
    custom_ecc_lon = 0.75  # small eccentricity for a sharp blob (only one cell)

    for center_lat in custom_center_lat:
        for center_lon in custom_center_lon:
            # 2) call function
            shape_method = BLOB(
                center_lat=center_lat,
                center_lon=center_lon,
                ecc_lat=custom_ecc_lat,
                ecc_lon=custom_ecc_lon,
            )
            mask_custom = shape_method.make_shape(
                lats=np.linspace(-90, 90, lat_size),
                lons=np.linspace(-180, 180, lon_size),
            )

            # 3) check results
            # - center_lat
            # -> ix of center_lat is 90 + custom_center_lat
            center_lat_exp = 90 + center_lat
            ix_lat_mean = np.mean(np.where(mask_custom.any(axis=1))[0])
            assert np.isclose(ix_lat_mean, center_lat_exp, atol=err), (
                f"Expected center_lat: {center_lat_exp}, got: {ix_lat_mean}"
            )
            # - center_lon
            # -> ix of center_lon is 180 + custom_center_lon
            center_lon_exp = 180 + center_lon
            ix_lon_mean = np.mean(np.where(mask_custom.any(axis=0))[0])
            assert np.isclose(ix_lon_mean, center_lon_exp, atol=err), (
                f"Expected center_lon: {center_lon_exp}, got: {ix_lon_mean}"
            )
