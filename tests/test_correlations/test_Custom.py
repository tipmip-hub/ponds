import numpy as np
from ponds import Custom

"""

Things to Test:
    x check if custom mask is used

Legend:
    - : to do
    x : done
"""


def test_custom_mask_usage():
    """
    - Check that the Custom correlation uses the provided custom mask correctly
    """
    custom_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, -2, 2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -2, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, -2, 2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -2, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, -2, 2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    shape_mask = custom_mask.astype(bool)

    corr_instance = Custom(
        offset_mask=custom_mask,
    )

    offsets = corr_instance.make_time_offsets(shape_mask)

    # Check that offsets are only applied where custom_mask is True
    assert np.all(offsets[~shape_mask] == 0), (
        "Offsets should be zero where custom_mask is 0."
    )
    assert np.all(offsets[shape_mask] != 0), (
        "Offsets should be non-zero where custom_mask has a non-zero value."
    )
