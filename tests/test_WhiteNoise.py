import numpy as np
from ponds import PONDS, WhiteNoise

"""
Testing the PONDS core class and its methods.

Things to Test:
    x does the magnitude work
    x does the random seed work
    x does the perturbation time work


Legend:
    - : to do
    x : done
"""

def test_magnitude(err_tol=1e-2):

    # 1) setup variables
    magnitude = 1

    # 2) call function
    ponds = PONDS(
        lat_size=2,
        lon_size=2,
        time_size=10000,
    )
    ponds.add_perturbation(
        perturbation_method=WhiteNoise(magnitude=magnitude),
    )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values
    ts2 = ponds.data.ts[:, 0, 1].values
    ts3 = ponds.data.ts[:, 1, 0].values
    ts4 = ponds.data.ts[:, 1, 1].values

    # -
    assert np.isclose(np.std(ts1), magnitude, atol=err_tol)
    assert np.isclose(np.std(ts2), magnitude, atol=err_tol)
    assert np.isclose(np.std(ts3), magnitude, atol=err_tol)
    assert np.isclose(np.std(ts4), magnitude, atol=err_tol)

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
            lat_size=1,
            lon_size=1,
            time_size=100,
        )
        ponds1.add_perturbation(
            perturbation_method=WhiteNoise(
                random_seed=seed1,    # <- benchmark
                ),
        )

        ponds2 = PONDS(
            lat_size=1,
            lon_size=1,
            time_size=100,
        )
        ponds2.add_perturbation(
            perturbation_method=WhiteNoise(
                random_seed=seed1,    # <- same seed as benchmark
                ),
        )

        ponds3 = PONDS(
            lat_size=1,
            lon_size=1,
            time_size=100,
        )
        ponds3.add_perturbation(
            perturbation_method=WhiteNoise(
                random_seed=seed2,  # <- different seed
                ),
        )

        # 3) check results
        assert np.array_equal(ponds1.data.ts, ponds2.data.ts), (
            "PONDS data is not reproducible with the same random seed."
        )
        assert not np.array_equal(ponds1.data.ts, ponds3.data.ts), (
            "PONDS data is reproducible with different random seeds."
        )

def test_perturbation_time():
    """
    - check if perturbation is applied only at the specified times
        - expected values at specified times: non-zero values
        - expected values at non-specified times: zero values
    """

    # 1) setup variables
    time_size = 10
    perturbation_time = np.array([3, 4, 5, 6, 7])
    magnitude = 1

    # 2) call function
    ponds = PONDS(
        lat_size=1,
        lon_size=1,
        time_size=time_size,
    )
    ponds.add_perturbation(
        perturbation_method=WhiteNoise(
            magnitude=magnitude,
            perturbation_time=perturbation_time,
        ),
    )

    # 3) check results
    # - Check that the perturbation is applied only at the specified times
    assert ~np.array_equal(ponds.data.ts[perturbation_time, 0, 0].values,
                          np.zeros(len(perturbation_time))), (
                            "Perturbation values do not match expected values at specified times."
    )
    ix = np.setdiff1d(np.arange(time_size), perturbation_time)
    assert np.array_equal(ponds.data.ts[ix, 0, 0].values,
                          np.zeros(time_size - len(perturbation_time))), (
                            "Unexpected perturbation values at non-specified times."
    )