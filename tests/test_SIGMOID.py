import numpy as np
from ponds import PONDS, SIGMOID


"""
Things to Test:
    x General Tests
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

def test_sigmoid_shift_time_used():
    # 1) setup variables
    time_size = 100
    shift_time_in = 20
    ponds = PONDS(
        lat_size=1,
        lon_size=1,
        time_size=time_size,
        background_trend=0,
        background_noise=0,
    )

    # 2) call function
    # - fix sigmoid parameters, otherwise they would be random
    shift_dict = {"sigma_time": 0,
                  "shift_steepness": 1,
                  "shift_magnitude": 1}
    sigmoid_shift = SIGMOID(**shift_dict)
    _, shift_time_out = sigmoid_shift.make_shift(ponds.time,
                                                 shift_time_in)

    # 3) check results
    assert abs(shift_time_out - shift_time_in) == 0, (
        f"Shift applied at wrong time index {shift_time_out}, expected around {shift_time_in}"
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
    time = np.arange(0, 20, 1)  # Example time array
    shift_time = 5

    for _ in range(R):
        random_seed = np.random.randint(0, 1000)

        # 2) call function
        sigmoid1 = SIGMOID(random_seed=random_seed)  # <- testing this
        shift_effect1, _ = sigmoid1.make_shift(time, shift_time)  # <- testing this: no shift parameters provided, so random values are used

        sigmoid2 = SIGMOID(random_seed=random_seed)  # <- testing this
        shift_effect2, _ = sigmoid2.make_shift(time, shift_time)   # <- testing this: no shift parameters provided, so random values are used

        # 3) check results
        assert np.array_equal(shift_effect1, shift_effect2), (
            "Shifts with same seed should produce identical results."
        )


###################################################################################################
####################################### Custom Shift ##############################################
###################################################################################################


def test_sigmoid_custom_parameters_used(err_tol=1e-1):
    """
    - Test if custom parameters for the sigmoid shift are used correctly.
    - expected behavior:
        - sigma_time: choosen from a uniform distribution of the interval [-sigma_time, sigma_time]
    """

    # 1) setup variables
    lat_size = 100
    lon_size = 100
    time = np.arange(0, 20, 1)  # Example time array
    #time_size = 10
    #ponds = PONDS(
    #    lat_size=lat_size,
    #    lon_size=lon_size,
    #    time_size=time_size,
    #    background_trend=0,
    #    background_noise=0,
    #)
    shift_time = 5

    custom_sigma_time = 3.0
    custom_steepness = 5.0
    custom_magnitude = 2.0

    # 2) call function
    t_shifts = np.zeros((lat_size,lon_size))
    sigmoid_shift = SIGMOID(sigma_time=custom_sigma_time,
                            shift_steepness=custom_steepness,
                            shift_magnitude=custom_magnitude,
                            )
    for i in range(lat_size):
        for j in range(lon_size):
            _, t_shifts[i, j] = sigmoid_shift.make_shift(time,
                                                         shift_time)

    # 3) check results
    # - extract the shift parameters from the generated data
    # -- sigma_time
    t0_std = np.std(t_shifts.flatten())
    t0_std_expected = (
        custom_sigma_time * 2 / np.sqrt(12)
    )  # for a uniform distribution, the standard deviation is (b-a)/sqrt(12), where a and b are the bounds of the distribution
    assert np.isclose(t0_std, t0_std_expected, atol=err_tol), (
        f"Expected standard deviation {t0_std_expected} not used, got {t0_std}."
    )
    #assert np.abs(t0_std - custom_sigma_time) > 0, (
    #    f"Custom sigma_time {custom_sigma_time} not used, got {t0_std}."
    #)

    # -- shift_steepness

    # assert np.allclose(params["sigma_time"], custom_sigma_time), "Custom sigma_time not used"
    # assert np.allclose(params["shift_steepness"], custom_steepness), "Custom shift_steepness not used"
    # assert np.allclose(params["shift_magnitude"], custom_magnitude), "Custom shift_magnitude not used"
