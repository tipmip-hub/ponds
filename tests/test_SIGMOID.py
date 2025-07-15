import numpy as np
from ponds import PONDS, SIGMOID


"""
Things to Test:
    x General Tests
        x is the right blob mask used
        x is the right shift time used
    x Random sigmoid shift
        x do same seed produce same shift
        - check if random parameters are stored correctly in ts_param
    - Custom sigmoid shift
        - check if custom parameters are used
            x sigma_time
            - shift_steepness
            - shift_magnitude
        - check if custom parameters are stored correctly in ts_param

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
    ponds = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        background_trend=0,
        background_noise=0,
    )
    # - create a mask with a single True value at (0, 1)
    mask = np.zeros((lat_size, lon_size), dtype=bool)
    mask[0, 1] = True
    # - shift at midpoint of time series
    shift_time = int(time_size // 2)

    # 2) call function
    sigmoid_shift = SIGMOID(ponds, mask, shift_time)
    sigmoid_shift.make_shift()

    # Only [0,1] cell should be non-zero
    for i in range(3):
        for j in range(3):
            ts = ponds.data.ts[:, i, j].values
            if mask[i, j]:
                assert not np.allclose(ts, 0), (
                    f"Shift not applied to masked cell {(i, j)}"
                )
            else:
                assert np.allclose(ts, 0), (
                    f"Shift incorrectly applied to unmasked cell {(i, j)}"
                )


def test_sigmoid_shift_time_used():
    # 1) setup variables
    time_size = 100
    shift_time = 20
    ponds = PONDS(
        lat_size=1,
        lon_size=1,
        time_size=time_size,
        background_trend=0,
        background_noise=0,
    )
    mask = np.ones((1, 1), dtype=bool)

    # 2) call function
    # - fix sigmoid parameters, otherwise they would be random
    shift_dict = {"sigma_time": 0, "shift_steepness": 1, "shift_magnitude": 1}
    sigmoid_shift = SIGMOID(ponds, mask, shift_time)
    _ = sigmoid_shift.make_shift(**shift_dict)

    # 3) check results
    # - Calculate where sigmoid crosses ~0.5
    ts = ponds.data.ts[:, 0, 0].values
    t0_index = np.argmax(np.diff(ts)) + 1  # +1 because np.diff reduces length by 1
    # -
    assert abs(t0_index - shift_time) == 0, (
        f"Shift applied at wrong time index {t0_index}, expected around {shift_time}"
    )


###################################################################################################
####################################### Random Shift ##############################################
###################################################################################################


def test_sigmoid_same_seed_produces_same_shift(R=5):
    """
    - Test if two SIGMOID instances with the same random seed produce the same shift.
    - This is important for reproducibility.
    - R: number of repetitions to test
    """

    # 1) setup variables
    lat_size = 1
    lon_size = 1
    time_size = 10
    background_trend = 0
    background_noise = 0
    shift_time = 5
    # - create two PONDS instances with same parameters
    ponds1 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        background_trend=background_trend,
        background_noise=background_noise,
    )
    ponds2 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        background_trend=background_trend,
        background_noise=background_noise,
    )
    mask = np.ones((1, 1), dtype=bool)

    for _ in range(R):
        random_seed = np.random.randint(0, 1000)

        # 2) call function
        sigmoid1 = SIGMOID(
            ponds1, mask, shift_time, random_seed=random_seed
        )  # <- testing this
        _ = (
            sigmoid1.make_shift()
        )  # <- testing this: no shift parameters provided, so random values are used

        sigmoid2 = SIGMOID(
            ponds2, mask, shift_time, random_seed=random_seed
        )  # <- testing this
        _ = (
            sigmoid2.make_shift()
        )  # <- testing this: no shift parameters provided, so random values are used

        # 3) check results
        assert np.array_equal(ponds1.data.ts.values, ponds2.data.ts.values), (
            "Shifts with same seed should produce identical results."
        )


###################################################################################################
####################################### Custom Shift ##############################################
###################################################################################################


def test_sigmoid_custom_parameters_used(err_tol=1e-1):
    """
    - Test if custom parameters for the sigmoid shift are used correctly.
    - expected bevahior:
        - sigma_time: choosen from a uniform distribution of the interval [-sigma_time, sigma_time]
    """

    # 1) setup variables
    lat_size = 100
    lon_size = 100
    time_size = 10
    ponds = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
        background_trend=0,
        background_noise=0,
    )
    mask = np.ones(
        (lat_size, lon_size), dtype=bool
    )  # all cells are masked: producing (lat_size x lon_size) shifts
    shift_time = 5

    custom_sigma_time = 3.0
    custom_steepness = 5.0
    custom_magnitude = 2.0

    # 2) call function
    sigmoid_shift = SIGMOID(ponds, mask, shift_time)
    _ = sigmoid_shift.make_shift(
        sigma_time=custom_sigma_time,
        shift_steepness=custom_steepness,
        shift_magnitude=custom_magnitude,
    )

    # 3) check results
    # - extract the shift parameters from the generated data
    # -- sigma_time
    t0_std = np.std(
        [
            np.argmax(np.diff(ponds.data.ts[:, i, j].values)) + 1
            for i in range(lat_size)
            for j in range(lon_size)
        ]
    )
    t0_std_expected = (
        custom_sigma_time * 2 / np.sqrt(12)
    )  # for a uniform distribution, the standard deviation is (b-a)/sqrt(12), where a and b are the bounds of the distribution
    assert np.isclose(t0_std, t0_std_expected, atol=err_tol), (
        f"Expected standard deviation {t0_std_expected} not used, got {t0_std}."
    )
    assert np.abs(t0_std - custom_sigma_time) > 0, (
        f"Custom sigma_time {custom_sigma_time} not used, got {t0_std}."
    )

    # -- shift_steepness

    # assert np.allclose(params["sigma_time"], custom_sigma_time), "Custom sigma_time not used"
    # assert np.allclose(params["shift_steepness"], custom_steepness), "Custom shift_steepness not used"
    # assert np.allclose(params["shift_magnitude"], custom_magnitude), "Custom shift_magnitude not used"
