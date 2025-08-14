import numpy as np
from ponds import PONDS, BLOB

"""

Things to Test:
    x Random Blob
        x is the shape reproducible using stored metadata?
        x does the same seed produce the same blob mask?
    x Custom Blob
        x do the center coordinates match the expected location?
        x bigger `size` value leads to larger blob?
        x does the blob wrap around the lon=0° line?

Legend:
    - : to do
    x : done
"""


###################################################################################################
##################################### Random Blob ################################################
###################################################################################################

def test_blob_same_seed_produces_same_mask(R=5):
    """
    Check if the same random seed produces the same BLOB mask.
    """
    lat_size = 80
    lon_size = 160

    for _ in range(R):
        seed_blob = np.random.randint(0, 1000)
        ponds1 = PONDS(lat_size=lat_size, lon_size=lon_size)
        ponds2 = PONDS(lat_size=lat_size, lon_size=lon_size)

        blob1 = BLOB(random_seed=seed_blob)
        mask1 = blob1.make_shape(lats=ponds1.lats, lons=ponds1.lons)

        blob2 = BLOB(random_seed=seed_blob)
        mask2 = blob2.make_shape(lats=ponds2.lats, lons=ponds2.lons)

        assert np.array_equal(mask1, mask2), "BLOBs with same seed should be identical."


def test_blob_reproducibility_from_params():
    """
    Check if a BLOB can be exactly reproduced using stored parameters in param_dict.
    """
    lat_size = 60
    lon_size = 120
    ponds = PONDS(lat_size=lat_size, lon_size=lon_size)

    blob = BLOB(random_seed=42)
    mask1 = blob.make_shape(lats=ponds.lats, lons=ponds.lons)

    # Recreate from stored params
    params = blob.param_dict
    blob_copy = BLOB(
        center_lat=params["center_lat"],
        center_lon=params["center_lon"],
        size=params["size"],
        max_bellys=params["max_bellys"],
        goo=params["goo"],
        random_seed=params["random_seed"],
    )
    mask2 = blob_copy.make_shape(lats=ponds.lats, lons=ponds.lons)

    assert np.array_equal(mask1, mask2), "BLOB recreated from params should match original."


###################################################################################################
##################################### Custom Blob ################################################
###################################################################################################

def test_blob_center_coordinates(err=10):
    """
    Test if custom center coordinates for BLOB result in a mask roughly centered there.
    err: allowed offset in index space.
    """
    lat_size = 180
    lon_size = 360
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)

    centers_lat = [-60, 0, 45]
    centers_lon = [-120, 0, 150]

    for clat in centers_lat:
        for clon in centers_lon:
            blob = BLOB(
                center_lat=clat,
                center_lon=clon,
                size=8,
                max_bellys=1,
            )
            mask = blob.make_shape(lats=lats, lons=lons)

            # Find center of mass in index space
            lat_indices = np.where(mask.any(axis=1))[0]
            lon_indices = np.where(mask.any(axis=0))[0]

            lat_center_idx = np.mean(lat_indices)
            lon_center_idx = np.mean(lon_indices)

            # Expected index positions
            lat_expected_idx = np.argmin(np.abs(lats - clat))
            lon_expected_idx = np.argmin(np.abs(lons - clon))

            assert np.isclose(lat_center_idx, lat_expected_idx, atol=err), (
                f"Lat center mismatch: expected {lat_expected_idx}, got {lat_center_idx}"
            )
            assert np.isclose(lon_center_idx, lon_expected_idx, atol=err), (
                f"Lon center mismatch: expected {lon_expected_idx}, got {lon_center_idx}"
            )


def test_blob_size_scaling():
    """
    Larger 'size' values should produce larger blobs.
    """
    lat_size = 100
    lon_size = 200
    ponds = PONDS(lat_size=lat_size, lon_size=lon_size)

    blob_small = BLOB(size=5, max_bellys=1, goo=2.0, random_seed=1)
    mask_small = blob_small.make_shape(lats=ponds.lats, lons=ponds.lons)

    blob_large = BLOB(size=15, max_bellys=1, goo=2.0, random_seed=1)
    mask_large = blob_large.make_shape(lats=ponds.lats, lons=ponds.lons)

    assert mask_large.sum() > mask_small.sum(), (
        "Blob with larger 'size' should cover more grid cells."
    )


def test_blob_wraps_around_dateline():
    """
    Ensure blobs that cross the longitude ±180 boundary still connect properly.
    """
    lat_size = 60
    lon_size = 120
    ponds = PONDS(lat_size=lat_size, lon_size=lon_size)

    blob = BLOB(
        center_lat=0,
        center_lon=179,
        size=10,
        max_bellys=1,
    )
    mask = blob.make_shape(lats=ponds.lats, lons=ponds.lons)

    # Check that some True cells appear on both ends of longitude axis
    left_edge = mask[:, 0].any()
    right_edge = mask[:, -1].any()

    assert left_edge and right_edge, (
        "Blob near dateline should wrap around to both longitude edges."
    )
