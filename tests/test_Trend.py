import numpy as np
from ponds import PONDS, Trend

"""
Testing the PONDS core class and its methods.

Things to Test:
    x does the slope work
    x does the intercept work
    x does the perturbation time work


Legend:
    - : to do
    x : done
"""


def test_slope(err_tol=1e-4):
    # 1) setup variables
    trend = 0.1

    # 2) call function
    # - (2x2) grid, check trend in all 4 time series
    ponds = PONDS(
        lat_size=2,
        lon_size=2,
        time_size=10,
    )
    ponds.add_perturbation(
        perturbation_method=Trend(slope=trend, intercept=0.0),
    )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values
    ts2 = ponds.data.ts[:, 0, 1].values
    ts3 = ponds.data.ts[:, 1, 0].values
    ts4 = ponds.data.ts[:, 1, 1].values
    # - Fit a linear regression to confirm the slope
    slope1 = (ts1[-1] - ts1[0]) / (ponds.time[-1] - ponds.time[0])
    slope2 = (ts2[-1] - ts2[0]) / (ponds.time[-1] - ponds.time[0])
    slope3 = (ts3[-1] - ts3[0]) / (ponds.time[-1] - ponds.time[0])
    slope4 = (ts4[-1] - ts4[0]) / (ponds.time[-1] - ponds.time[0])
    # -
    assert np.isclose(slope1, trend, atol=err_tol)
    assert np.isclose(slope2, trend, atol=err_tol)
    assert np.isclose(slope3, trend, atol=err_tol)
    assert np.isclose(slope4, trend, atol=err_tol)


def test_intercept(err_tol=1e-4):
    # 1) setup variables
    intercept = -5

    # 2) call function
    # - (2x2) grid, check trend in all 4 time series
    ponds = PONDS(
        lat_size=2,
        lon_size=2,
        time_size=10,
    )
    ponds.add_perturbation(
        perturbation_method=Trend(slope=0.0, intercept=intercept),
    )

    # 3) check results
    ts1 = ponds.data.ts[:, 0, 0].values
    ts2 = ponds.data.ts[:, 0, 1].values
    ts3 = ponds.data.ts[:, 1, 0].values
    ts4 = ponds.data.ts[:, 1, 1].values
    # - Fit a linear regression to confirm the slope
    intercept1 = ts1[0]
    intercept2 = ts2[0]
    intercept3 = ts3[0]
    intercept4 = ts4[0]
    # -
    assert np.isclose(intercept1, intercept, atol=err_tol)
    assert np.isclose(intercept2, intercept, atol=err_tol)
    assert np.isclose(intercept3, intercept, atol=err_tol)
    assert np.isclose(intercept4, intercept, atol=err_tol)


def test_perturbation_time():
    # 1) setup variables
    time_size = 10
    perturbation_time = np.array([3, 4, 5, 6, 7])
    slope = 1
    intercept = 3

    # 2) call function
    # - (2x2) grid, check trend in all 4 time series
    ponds = PONDS(
        lat_size=1,
        lon_size=1,
        time_size=time_size,
    )
    ponds.add_perturbation(
        perturbation_method=Trend(
            slope=slope, intercept=intercept, perturbation_time=perturbation_time
        ),
    )

    # 3) check results
    # - Check that the perturbation is applied only at the specified times
    assert np.array_equal(
        ponds.data.ts[perturbation_time, 0, 0].values, perturbation_time + intercept
    ), "Perturbation values do not match expected values at specified times."
    ix = np.setdiff1d(np.arange(time_size), perturbation_time)
    assert np.array_equal(
        ponds.data.ts[ix, 0, 0].values, np.zeros(time_size - len(perturbation_time))
    ), "Unexpected perturbation values at non-specified times."
