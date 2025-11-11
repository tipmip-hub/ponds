import numpy as np
from ponds import Propagating

"""

Things to Test:
    x check if max offset is respected
    x check if number of cells with offsets is correct
    x test reproducibility

Legend:
    - : to do
    x : done
"""


def test_max_offset():
    """
    - Check that all offsets are within the specified max_offset range
    """

    lat_size, lon_size = 30, 60
    shape_mask = np.ones((lat_size, lon_size), dtype=bool)
    max_offset = 10
    corr_instance = Propagating(
        n_cells=5,
        max_offset=max_offset,
        corr_seed=123,
    )

    offsets = corr_instance.make_time_offsets(shape_mask)

    assert np.all(offsets <= max_offset), (
        "Some offsets exceed the maximum allowed value."
    )
    assert np.all(offsets >= 0), "Some offsets are negative, which is not allowed."


def test_number_of_cells():
    """
    Check that the number of cells with non-zero offsets matches n_cells
    """
    lat_size, lon_size = 30, 60
    shape_mask = np.ones((lat_size, lon_size), dtype=bool)
    max_offset = 100
    corr_instance = Propagating(
        n_cells=5,
        max_offset=max_offset,
        corr_seed=14,
    )

    offsets = corr_instance.make_time_offsets(shape_mask)

    non_zero_offsets = np.count_nonzero(np.unique(offsets))
    assert non_zero_offsets == (
        corr_instance.n_cells - 1
    ), (  # -1 as starting offset is zero
        f"Expected {corr_instance.n_cells - 1} non-zero offsets, but got {non_zero_offsets}."
    )


def test_reproducibility(R=5):
    """
    Check that using the same seed produces identical offset arrays.
    """

    lat_size, lon_size = 30, 60
    shape_mask = np.ones((lat_size, lon_size), dtype=bool)
    for _ in range(R):
        seed = np.random.randint(0, 10000)
        corr_instance1 = Propagating(
            n_cells=7,
            max_offset=50,
            corr_seed=seed,
        )
        corr_instance2 = Propagating(
            n_cells=7,
            max_offset=50,
            corr_seed=seed,
        )

        offsets1 = corr_instance1.make_time_offsets(shape_mask)
        offsets2 = corr_instance2.make_time_offsets(shape_mask)

        assert np.array_equal(offsets1, offsets2), (
            f"Offsets differ for the same seed {seed}."
        )
