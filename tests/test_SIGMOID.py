import numpy as np
from ponds import PONDS, SIGMOID

"""
Things to Test:
    - Random sigmoid shift
        x do same seed produce same shift
        - check if random parameters are stored correctly in ts_param
    - Custom sigmoid shift
        - check if custom parameters are used
            x central_time
            x sigma_time
            - shift_steepness
            - shift_magnitude
        - check if the correct shift parameters are stored as attributes on the DataArray Level

Legend:
    - : to do
    x : done
"""

###################################################################################################
########################################## General ################################################
###################################################################################################




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
    time = np.arange(0, 20, 1)  # Example time array

    for _ in range(R):
        random_seed = np.random.randint(0, 1000)

        # 2) call function
        sigmoid1 = SIGMOID(random_seed=random_seed)  # <- testing this: no shift parameters provided, so random values are used
        shift_effect1, _ = sigmoid1.make_shift(
            time,
        )

        sigmoid2 = SIGMOID(random_seed=random_seed)  # <- testing this: no shift parameters provided, so random values are used
        shift_effect2, _ = sigmoid2.make_shift(
            time,
        )

        # 3) check results
        assert np.array_equal(shift_effect1, shift_effect2), (
            "Shifts with same seed are not producing identical results."
        )


###################################################################################################
####################################### Custom Shift ##############################################
###################################################################################################


def test_sigmoid_custom_parameters_used(err_tol=1e-1):
    """
    - Test if custom parameters for the sigmoid shift are used correctly.
    - expected behavior:
        - central_time: mean of the shift time should be equal to central_time
        - sigma_time: chosen from a uniform distribution of the interval [-sigma_time, sigma_time]
    """

    # 1) setup variables
    lat_size = 100
    lon_size = 100
    time = np.arange(0, 20, 1)  # Example time array
    
    central_time = 5
    custom_sigma_time = 3.0
    custom_steepness = 5.0
    custom_magnitude = 2.0

    # 2) call function
    t_shifts = np.zeros((lat_size, lon_size))
    sigmoid_shift = SIGMOID(
        central_time=central_time,
        sigma_time=custom_sigma_time,
        shift_steepness=custom_steepness,
        shift_magnitude=custom_magnitude,
    )
    for i in range(lat_size):
        for j in range(lon_size):
            _, t_shifts[i, j] = sigmoid_shift.make_shift(time)

    # 3) check results
    # - extract the shift parameters from the generated data
    # -- central_time
    t0 = np.round(np.mean(t_shifts.flatten()))
    assert int(t0) == central_time, (
        f"Expected central_time {central_time} not used, got {t0}."
    )

    # -- sigma_time
    t0_std = np.std(t_shifts.flatten())
    t0_std_expected = (
        custom_sigma_time * 2 / np.sqrt(12)
    )  # for a uniform distribution, the standard deviation is (b-a)/sqrt(12), where a and b are the bounds of the distribution
    assert np.isclose(t0_std, t0_std_expected, atol=err_tol), (
        f"Expected standard deviation {t0_std_expected} not used, got {t0_std}."
    )

    # -- shift_steepness

    # assert np.allclose(params["sigma_time"], custom_sigma_time), "Custom sigma_time not used"
    # assert np.allclose(params["shift_steepness"], custom_steepness), "Custom shift_steepness not used"
    # assert np.allclose(params["shift_magnitude"], custom_magnitude), "Custom shift_magnitude not used"
