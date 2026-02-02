import numpy as np
import warnings
from ponds import PONDS, WhiteNoise

"""
Testing the PONDS core class and its methods.

Things to Test:
    x does the magnitude work
    x does the random seed work
    x check if noise is reproducible with same seed
    x check that noise is not exact same in different cells
    x does the background time work


Legend:
    - : to do
    x : done
"""


def test_magnitude(err_tol=5e-2):
    # 1) setup variables
    magnitude = 1

    ponds = PONDS(
        lat_size=1,
        lon_size=1,
        time_size=10000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ponds.add_background(
            background_method=WhiteNoise(magnitude=magnitude),
        )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values

    # -
    assert np.max(ts1) <= magnitude + err_tol
    assert np.min(ts1) >= -magnitude - err_tol


def test_random_seed(R=5):
    """
    - check if same input parameters produces the same data
    - check if different random seeds produce different data
    """

    for _ in range(R):
        # 1) setup variables
        seed1 = np.random.randint(0, 1000)
        seed2 = np.random.randint(0, 1000)
        if seed1 == seed2:
            seed2 += 1

        # 2) call function
        ponds1 = PONDS(
            lat_size=2,
            lon_size=2,
            time_size=100,
        )
        ponds1.add_background(
            background_method=WhiteNoise(
                background_seed=seed1,  # <- benchmark
            ),
        )

        ponds2 = PONDS(
            lat_size=2,
            lon_size=2,
            time_size=100,
        )
        ponds2.add_background(
            background_method=WhiteNoise(
                background_seed=seed1,  # <- same seed as benchmark
            ),
        )

        ponds3 = PONDS(
            lat_size=2,
            lon_size=2,
            time_size=100,
        )
        ponds3.add_background(
            background_method=WhiteNoise(
                background_seed=seed2,  # <- different seed
            ),
        )

        # 3) check results
        assert np.array_equal(ponds1.data.ts, ponds2.data.ts), (
            "PONDS data is not reproducible with the same random seed."
        )
        assert not np.array_equal(ponds1.data.ts, ponds3.data.ts), (
            "PONDS data is reproducible with different random seeds."
        )


def test_reproducibility(R=5):
    """
    - check if same input parameters produces the same data
    """

    for _ in range(R):
        # 1) setup variables
        seed = np.random.randint(0, 1000)

        # 2) call function
        ponds1 = PONDS(
            lat_size=10,
            lon_size=10,
            time_size=100,
        )
        ponds1.add_background(
            background_method=WhiteNoise(
                background_seed=seed,  # <- benchmark
            ),
        )

        ponds2 = PONDS(
            lat_size=10,
            lon_size=10,
            time_size=100,
        )
        ponds2.add_background(
            background_method=WhiteNoise(
                background_seed=seed,  # <- same seed as benchmark
            ),
        )

        # 3) check results
        assert np.array_equal(ponds1.data.ts, ponds2.data.ts), (
            "PONDS data is not reproducible with the same random seed."
        )


def test_different_cells():
    """
    - check that noise is not exact same in different cells
    """
    seed = 42

    # 1) setup variables
    seed = np.random.randint(0, 1000)

    # 2) call function
    ponds = PONDS(
        lat_size=5,
        lon_size=5,
        time_size=100,
    )
    ponds.add_background(
        background_method=WhiteNoise(
            background_seed=seed,
        ),
    )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values
    ts2 = ponds.data.ts[:, 0, 1].values
    ts3 = ponds.data.ts[:, 1, 0].values
    ts4 = ponds.data.ts[:, 1, 1].values

    assert not np.array_equal(ts1, ts2), (
        "Background noise is identical in different cells."
    )
    assert not np.array_equal(ts1, ts3), (
        "Background noise is identical in different cells."
    )
    assert not np.array_equal(ts1, ts4), (
        "Background noise is identical in different cells."
    )
    assert not np.array_equal(ts2, ts3), (
        "Background noise is identical in different cells."
    )
    assert not np.array_equal(ts2, ts4), (
        "Background noise is identical in different cells."
    )
    assert not np.array_equal(ts3, ts4), (
        "Background noise is identical in different cells."
    )


def test_background_time():
    """
    - check if background is applied only at the specified times
        - expected values at specified times: non-zero values
        - expected values at non-specified times: zero values
    """

    # 1) setup variables
    time_size = 10
    background_time = np.array([3, 4, 5, 6, 7])
    magnitude = 1

    # 2) call function
    ponds = PONDS(
        lat_size=2,
        lon_size=2,
        time_size=time_size,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ponds.add_background(
            background_method=WhiteNoise(
                magnitude=magnitude,
                background_time=background_time,
            ),
        )

    # 3) check results
    # - Check that the background is applied only at the specified times
    assert not (
        np.array_equal(
            ponds.data.ts[background_time, 0, 0].values, np.zeros(len(background_time))
        )
    ), "Background values do not match expected values at specified times."
    ix = np.setdiff1d(np.arange(time_size), background_time)
    assert np.array_equal(
        ponds.data.ts[ix, 0, 0].values, np.zeros(time_size - len(background_time))
    ), "Unexpected background values at non-specified times."
