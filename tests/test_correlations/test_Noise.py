import numpy as np
from ponds import Noise

"""

Things to Test:
    x test simple workflow with Noise correlation
    x test if the same seed produces the same offsets
    x test if offsets are within the specified range
    x check dict entries
    x check reproducibility

Legend:
    - : to do
    x : done
"""


def test_noise_output_shape_and_type():
    """
    - Check that the Noise.make_time_offsets() returns a numpy array
    - Check that the output has the same shape as the input grid (mask)
    - Check dtype (should be integer)
    """
    # 1) setup
    lat_size, lon_size = 10, 20
    shape_mask = np.ones((lat_size, lon_size), dtype=bool)
    corr_instance = Noise(max_offset=5, corr_seed=42)

    # 2) call
    offsets = corr_instance.make_time_offsets(shape_mask)

    # 3) check
    assert isinstance(offsets, np.ndarray), "Offsets must be a numpy array."
    assert offsets.shape == shape_mask.shape, "Offsets must match input shape."
    assert np.issubdtype(offsets.dtype, np.integer), "Offsets must be integer values."


def test_noise_same_seed_produces_same_offsets(R=5):
    """
    - test if the same random seed produces identical offset arrays
    - test if different seeds produce different offset arrays
    - run multiple times with different random seeds
    """
    lat_size, lon_size = 50, 100
    shape_mask = np.ones((lat_size, lon_size), dtype=bool)

    for _ in range(R):
        seed1 = np.random.randint(0, 10000)
        seed2 = np.random.randint(0, 10000)
        while seed2 == seed1:  # re-roll to ensure different seeds
            seed2 = np.random.randint(0, 10000)
        n1 = Noise(corr_seed=seed1)
        n2 = Noise(corr_seed=seed1)
        n3 = Noise(corr_seed=seed2)

        off1 = n1.make_time_offsets(shape_mask)
        off2 = n2.make_time_offsets(shape_mask)
        off3 = n3.make_time_offsets(shape_mask)

        assert np.array_equal(off1, off2), (
            "Offsets must be identical for the same seed."
        )
        assert not np.array_equal(off1, off3), (
            "Offsets must differ for different seeds."
        )


def test_noise_offsets_within_range(R=5):
    """
    - Check if offsets are within [-max_offset, max_offset]
    - run multiple times with different ranges
    """
    lat_size, lon_size = 30, 40
    for _ in range(R):
        max_offset = np.random.randint(1, 20)
        shape_mask = np.ones((lat_size, lon_size), dtype=bool)
        seed = np.random.randint(0, 10000)
        noise = Noise(max_offset=max_offset, corr_seed=seed)

        offsets = noise.make_time_offsets(shape_mask)

        assert offsets.min() >= -max_offset, f"Offsets below -{max_offset} detected."
        assert offsets.max() < max_offset, (
            f"Offsets equal or above {max_offset} detected."
        )


def test_noise_param_dict():
    """
    - Check if the Noise parameters are stored correctly
    """
    max_offset = 3
    corr_seed = 101
    noise = Noise(max_offset=max_offset, corr_seed=corr_seed)

    pdict = noise.param_dict
    assert pdict["correlation_type"] == Noise, "Wrong correlation_type stored."
    assert pdict["max_offset"] == max_offset, (
        f"Expected max_offset={max_offset}, got {pdict['max_offset']}"
    )
    assert pdict["corr_seed"] == corr_seed, (
        f"Expected corr_seed={corr_seed}, got {pdict['corr_seed']}"
    )

def test_reproducibility():
    """
    Check if same seed produces same offsets.
    """

    lat_size, lon_size = 25, 35
    shape_mask = np.ones((lat_size, lon_size), dtype=bool)
    seed = 1234
    noise1 = Noise(max_offset=7, corr_seed=seed)
    noise2 = Noise(max_offset=7, corr_seed=seed)

    offsets1 = noise1.make_time_offsets(shape_mask)
    offsets2 = noise2.make_time_offsets(shape_mask)

    assert np.array_equal(offsets1, offsets2), "Offsets should be identical for the same seed."