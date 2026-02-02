import numpy as np
from ponds import SIGMOID, PONDS

"""
Things to Test:
    - Random sigmoid shift
        x do same seed produce same shift
    - Custom sigmoid shift
        - check if custom parameters are used
            x central_time
            - shift_steepness
            - shift_magnitude
        - check if the correct shift parameters are stored as attributes on the DataArray Level
    - test if sigmoid features can be correlated using correlationgen-methods

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
        shift_seed = np.random.randint(0, 1000)

        # 2) call function
        sigmoid1 = SIGMOID(
            shift_seed=shift_seed
        )  # <- testing this: no shift parameters provided, so random values are used
        make_shift1 = sigmoid1.prepare_shift(
            time,
        )
        shift_effect1, _ = make_shift1()

        sigmoid2 = SIGMOID(
            shift_seed=shift_seed
        )  # <- testing this: no shift parameters provided, so random values are used
        make_shift2 = sigmoid2.prepare_shift(
            time,
        )
        shift_effect2, _ = make_shift2()

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
    # custom_sigma_time = 3.0
    custom_steepness = 5.0
    custom_magnitude = 2.0

    # 2) call function
    t_shifts = np.zeros((lat_size, lon_size))
    sigmoid_shift = SIGMOID(
        central_time=central_time,
        # sigma_time=custom_sigma_time,
        shift_steepness=custom_steepness,
        shift_magnitude=custom_magnitude,
    )
    make_shift = sigmoid_shift.prepare_shift(
        time,
    )
    for i in range(lat_size):
        for j in range(lon_size):
            _, t_shifts[i, j] = make_shift()

    # 3) check results
    # - extract the shift parameters from the generated data
    # -- central_time
    t0 = np.round(np.mean(t_shifts.flatten()))
    assert int(t0) == central_time, (
        f"Expected central_time {central_time} not used, got {t0}."
    )

    # -- sigma_time
    # t0_std = np.std(t_shifts.flatten())
    # t0_std_expected = (
    #    custom_sigma_time * 2 / np.sqrt(12)
    # )  # for a uniform distribution, the standard deviation is (b-a)/sqrt(12), where a and b are the bounds of the distribution
    # assert np.isclose(t0_std, t0_std_expected, atol=err_tol), (
    #    f"Expected standard deviation {t0_std_expected} not used, got {t0_std}."
    # )

    # -- shift_steepness

    # assert np.allclose(params["sigma_time"], custom_sigma_time), "Custom sigma_time not used"
    # assert np.allclose(params["shift_steepness"], custom_steepness), "Custom shift_steepness not used"
    # assert np.allclose(params["shift_magnitude"], custom_magnitude), "Custom shift_magnitude not used"


###################################################################################################
######################################## Correlation ##############################################
###################################################################################################


def test_sigmoid_correlation(mock_corr, mock_shape):
    """
    - apply mock correlation method to sigmoid shift features, on at a time
        -> see if offset is only applied to the intended feature
    """

    # 1) setup variables
    # - ponds parameters
    lat_size = 2
    lon_size = 2
    time_size = 100
    # - shift parameters
    central_time = 50
    shift_magnitude = 1.0
    shift_steepness = 1.0

    pd0 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )

    pd1 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )

    pd2 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )

    pd3 = PONDS(
        lat_size=lat_size,
        lon_size=lon_size,
        time_size=time_size,
    )

    # 2) call function
    # - first add a baseline shift without correlation
    pd0.add_shift(
        shift_method=SIGMOID(
            central_time=central_time,
            shift_steepness=shift_steepness,
            shift_magnitude=shift_magnitude,
        ),
        shape_method=mock_shape(),
    )

    # - apply for central_time feature
    pd1.add_shift(
        shift_method=SIGMOID(
            central_time=central_time,
            shift_steepness=shift_steepness,
            shift_magnitude=shift_magnitude,
        ),
        shape_method=mock_shape(),
        correlation_method=mock_corr(
            feature_name="central_time",
        ),
    )

    # - apply for shift_steepness feature
    pd2.add_shift(
        shift_method=SIGMOID(
            central_time=central_time,
            shift_steepness=shift_steepness,
            shift_magnitude=shift_magnitude,
        ),
        shape_method=mock_shape(),
        correlation_method=mock_corr(
            feature_name="shift_steepness",
        ),
    )

    # - apply for shift_magnitude feature
    pd3.add_shift(
        shift_method=SIGMOID(
            central_time=central_time,
            shift_steepness=shift_steepness,
            shift_magnitude=shift_magnitude,
        ),
        shape_method=mock_shape(),
        correlation_method=mock_corr(
            feature_name="shift_magnitude",
        ),
    )

    # 3) check results
    # - compare the shifted data at position (0,0) where the offset is applied
    # - baseline data
    data0 = pd0.data.ts.values
    assert np.all(data0[:50, 0, 0] <= 0.3), (
        "Baseline data incorrect at central_time feature."
    )
    assert np.all(data0[50:, 0, 0] >= 0.3), (
        "Baseline data incorrect at central_time feature."
    )
    assert np.isclose(np.max(data0), 1.0), (
        "Baseline data exceeds expected maximum value."
    )

    # - central time should be shifted by 2 at position (0,0)
    data1 = pd1.data.ts.values
    assert np.all(data1[:52, 0, 0] <= 0.3), (
        "Correlation offset not applied correctly to central_time feature."
    )
    assert np.all(data1[52:, 0, 0] >= 0.3), (
        "Correlation offset not applied correctly to central_time feature."
    )
    assert np.isclose(np.max(data1), 1.0), (
        "Baseline data exceeds expected maximum value."
    )

    # - shift_steepness should be shifted by 2 at position (0,0)
    ...

    # - shift_magnitude should be shifted by 2 at position (0,0) -> so 1.0 + 2.0 = 3.0
    data3 = pd3.data.ts.values
    assert np.all(data3[:50, 0, 0] <= 0.3 * 3), (
        "Correlation offset not applied correctly to shift_magnitude feature."
    )
    assert np.all(data3[50:, 0, 0] >= 0.3 * 3), (
        "Correlation offset not applied correctly to shift_magnitude feature."
    )
    assert np.isclose(np.max(data3), 3.0), (
        "Data exceeds expected maximum value after applying correlation to shift_magnitude feature."
    )
