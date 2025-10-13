import numpy as np
from ponds import PONDS, Gauss, ELLIPSIS
"""

Things to Test:
    x are the specified center coordinates used correctly
    - are the specified sigma values used correctly
    x is the specified max offset used correctly
    x when using ELLIPSIS and not specifying any center coordinates, are both centers aligned?

Legend:
    - : to do
    x : done
"""

def test_gauss_center_coordinates(err=2):
    """
    - Check if the specified correlation center coordinates (lat/lon)
      correctly affect where the maximum offset occurs.
    - expected behavior:
        - the maximum offset value should occur near the specified (lat, lon) center
    """
    lat_size = 180
    lon_size = 360

    test_centers = [
        (-60, -120),
        (-30, 45),
        (0, 0),
        (30, 90),
        (60, 150),
    ]

    for center_lat, center_lon in test_centers:
        gauss = Gauss(
            corr_center_lat=center_lat,
            corr_center_lon=center_lon,
            sigma_lat=5,                    # <- smaller sigma for sharper peak
            sigma_lon=5,
            max_offset=20,                  # <- larger max offset for better contrast
        )

        mask = np.ones((lat_size, lon_size), dtype=bool)
        offsets = gauss.make_time_offsets(mask)

        # Find indices of max offset (should correspond to center)
        max_pos = np.unravel_index(np.argmax(offsets), offsets.shape)
        lat_ix_exp = int((center_lat + 90) / 180 * (lat_size - 1))
        lon_ix_exp = int((center_lon + 180) / 360 * (lon_size - 1))

        assert abs(max_pos[0] - lat_ix_exp) <= err, (
            f"Lat center mismatch: expected {lat_ix_exp}, got {max_pos[0]}"
        )
        assert abs(max_pos[1] - lon_ix_exp) <= err, (
            f"Lon center mismatch: expected {lon_ix_exp}, got {max_pos[1]}"
        )

'''
def test_gauss_sigma_values_affect_spread():
    """
    - Check that sigma_lat and sigma_lon control the spread of offsets correctly.
    - expected behavior:
        - larger sigma -> broader distribution (lower spatial gradients)
    """
    lat_size, lon_size = 20, 20
    mask = np.ones((lat_size, lon_size), dtype=bool)

    gauss_small_sigma = Gauss(
        corr_center_lat=0,
        corr_center_lon=0,
        sigma_lat=10,
        sigma_lon=10,
        max_offset=20,
    )
    gauss_large_sigma = Gauss(
        corr_center_lat=0,
        corr_center_lon=0,
        sigma_lat=20,
        sigma_lon=20,
        max_offset=20,
    )

    off_small = gauss_small_sigma.make_time_offsets(mask)
    off_large = gauss_large_sigma.make_time_offsets(mask)

    # Measure how fast offsets decay away from center (via variance)
    var_small = np.var(np.abs(off_small))
    var_large = np.var(np.abs(off_large))

    assert var_large < var_small, (
        f"Larger sigma should produce smoother offsets (lower variance). "
        f"Got small_sigma_var={var_small}, large_sigma_var={var_large}"
    )
'''

def test_gauss_max_offset_scaling():
    """
    - Check that the specified max_offset value scales the amplitude correctly.
    - expected behavior:
        - doubling max_offset should approximately double the absolute offset values
    """
    lat_size, lon_size = 50, 80
    mask = np.ones((lat_size, lon_size), dtype=bool)

    g_low = Gauss(
        corr_center_lat=0,
        corr_center_lon=0,
        sigma_lat=10,
        sigma_lon=10,
        max_offset=5,
    )
    g_high = Gauss(
        corr_center_lat=0,
        corr_center_lon=0,
        sigma_lat=10,
        sigma_lon=10,
        max_offset=10,
    )

    off_low = g_low.make_time_offsets(mask)
    off_high = g_high.make_time_offsets(mask)

    # Compare means or max values (excluding zeros)
    assert np.abs(off_high).max() > np.abs(off_low).max(), (
        "Higher max_offset should produce larger absolute offsets."
    )

def test_gauss_and_ellipsis_alignment(mock_shift):
    """
    - When using ELLIPSIS without specifying a center, check that both
      GAUSS and ELLIPSIS are centered at the same default position.
    - expected behavior:
        - The peak (GAUSS) and the geometric center (ELLIPSIS) align
    """
    # Random Center Coordinates
    pd = PONDS(
        lat_size=50,
        lon_size=100,
    )
    pd.add_shift(
        shift_method=mock_shift(),
        shape_method=ELLIPSIS(),
        correlation_method=Gauss(),
    )

    # Get centers
    gauss_center_lat_ix = pd.data.c1.attrs["corr_center_lat"]
    gauss_center_lon_ix = pd.data.c1.attrs["corr_center_lon"]
    ellipsis_lat_ix = pd.data.c1.attrs["center_lat"]
    ellipsis_lon_ix = pd.data.c1.attrs["center_lon"]

    assert gauss_center_lat_ix == ellipsis_lat_ix, (
        f"Lat centers should align: Gauss {gauss_center_lat_ix}, ELLIPSIS {ellipsis_lat_ix}"
    )
    assert gauss_center_lon_ix == ellipsis_lon_ix, (
        f"Lon centers should align: Gauss {gauss_center_lon_ix}, ELLIPSIS {ellipsis_lon_ix}"
    )
