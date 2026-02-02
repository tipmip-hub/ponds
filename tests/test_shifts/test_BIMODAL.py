import numpy as np
from ponds import BIMODAL

"""
Things to Test:
    x check reproducibility of seeds
    x check reproducibility with custom params
    x check if central time is used as starting time
    x check if interval length is used correctly
    x check if duration is used correctly
    x check if shift magnitude is used correctly

Legend:
    - : to do
    x : done
"""


def test_bimodal_reproducibility_seed(R=5):
    """
    Check if BIMODAL shift is reproducible with same seed.
    """
    time_arr = np.arange(0, 100)
    for _ in range(R):
        seed = np.random.randint(0, 1000)
        bimodal_shift1 = BIMODAL(shift_seed=seed)
        bimodal_shift2 = BIMODAL(shift_seed=seed)
        time_arr = np.arange(0, 100)
        make_shift1 = bimodal_shift1.prepare_shift(time_arr)
        make_shift2 = bimodal_shift2.prepare_shift(time_arr)
        shift_effect1, _ = make_shift1()
        shift_effect2, _ = make_shift2()

        assert np.array_equal(shift_effect1, shift_effect2)


def test_bimodal_reproducibility_custom_params(R=5):
    """
    Check if BIMODAL shift is reproducible with same custom parameters.
    """
    time_arr = np.arange(0, 100)
    for _ in range(R):
        central_time = np.random.randint(10, 90)
        interval_length = np.random.randint(5, 20)
        duration = np.random.randint(5, 15)
        shift_magnitude = np.random.uniform(0.9, 1.1)

        bimodal_shift1 = BIMODAL(
            central_time=central_time,
            interval_length=interval_length,
            duration=duration,
            shift_magnitude=shift_magnitude,
        )
        bimodal_shift2 = BIMODAL(
            central_time=central_time,
            interval_length=interval_length,
            duration=duration,
            shift_magnitude=shift_magnitude,
        )

        make_shift1 = bimodal_shift1.prepare_shift(time_arr)
        make_shift2 = bimodal_shift2.prepare_shift(time_arr)
        shift_effect1, _ = make_shift1()
        shift_effect2, _ = make_shift2()

        assert np.array_equal(shift_effect1, shift_effect2)


def test_bimodal_central_time():
    """
    Check if BIMODAL shift uses the specified central time.
    """
    time_arr = np.arange(0, 100)
    central_time = 50
    bimodal_shift = BIMODAL(
        central_time=central_time,
        interval_length=10,
        shift_magnitude=1.0,
    )
    make_shift = bimodal_shift.prepare_shift(time_arr)
    shift_effect, actual_central_time = make_shift()

    assert actual_central_time == central_time
    assert np.all(shift_effect[: central_time + 1] == 0)
    assert shift_effect[central_time + 1] != 0  #


def test_bimodal_interval_length():
    """
    Check if BIMODAL shift uses the specified interval length.
    """
    time_arr = np.arange(0, 100)
    central_time = 50
    interval_length = 23
    bimodal_shift = BIMODAL(
        central_time=central_time,
        interval_length=interval_length,
        shift_magnitude=1.0,
    )
    make_shift = bimodal_shift.prepare_shift(time_arr)
    shift_effect, _ = make_shift()

    assert np.all(
        shift_effect[central_time + 1 : central_time + interval_length + 1] != 0
    )
    assert np.all(
        shift_effect[
            central_time + interval_length + 1 : central_time + 2 * interval_length + 1
        ]
        == 0
    )


def test_bimodal_shift_duration():
    """
    Check if BIMODAL shift uses the specified duration.
    - check for duration = None -> switching until end of time series
    - check for duration = specific value, interval_length positive
    - check for duration = specific value, interval_length negative
    """
    # check duration = None
    time_arr = np.arange(0, 11)
    central_time = 5
    interval_length = 1
    bimodal_shift = BIMODAL(
        central_time=central_time,
        interval_length=interval_length,
        shift_magnitude=1.0,
    )
    make_shift = bimodal_shift.prepare_shift(time_arr)
    shift_effect, _ = make_shift()

    assert shift_effect[-1] != 0

    # check duration = specific value, interval_length positive
    time_arr = np.arange(0, 11)
    central_time = 5
    interval_length = 1
    duration = 2
    bimodal_shift = BIMODAL(
        central_time=central_time,
        interval_length=interval_length,
        duration=duration,
        shift_magnitude=1.0,
    )
    make_shift = bimodal_shift.prepare_shift(time_arr)
    shift_effect, _ = make_shift()

    assert np.all(shift_effect[central_time + 1 + duration :] == 0)

    # check duration = specific value, interval_length negative
    time_arr = np.arange(0, 11)
    central_time = 5
    interval_length = -1
    duration = 2
    bimodal_shift = BIMODAL(
        central_time=central_time,
        interval_length=interval_length,
        duration=duration,
        shift_magnitude=1.0,
    )
    make_shift = bimodal_shift.prepare_shift(time_arr)
    shift_effect, _ = make_shift()

    assert np.all(shift_effect[: central_time - duration + 1] == 0)


def test_bimodal_shift_magnitude():
    """
    Check if BIMODAL shift uses the specified shift magnitude.
    """
    time_arr = np.arange(0, 100)
    central_time = 50
    interval_length = 10
    shift_magnitude = 1.2
    bimodal_shift = BIMODAL(
        central_time=central_time,
        interval_length=interval_length,
        shift_magnitude=shift_magnitude,
    )
    make_shift = bimodal_shift.prepare_shift(time_arr)
    shift_effect, _ = make_shift()

    assert np.all(
        shift_effect[central_time + 1 : central_time + interval_length + 1]
        == shift_magnitude
    )
    assert np.all(
        shift_effect[
            central_time + interval_length + 1 : central_time + 2 * interval_length + 1
        ]
        == 0
    )
