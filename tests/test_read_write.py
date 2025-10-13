import os
import tempfile
import numpy as np
from ponds import PONDS

"""
Testing the PONDS read and write functions.

Things to Test:
    x take dummy data and write it to a netcdf file
        x does the file exist under the given name? 
    x read the file back in
        x does the read data match the original data?
            x ts
            x mask
            x attributes

Legend:
    - : to do
    x : done
"""

###################################################################################################
###################################################################################################
###################################################################################################


def test_save_and_load(mock_shape, mock_shift, mock_perturbation, mock_corr):
    # Create dummy PONDS instance
    ponds = PONDS(
        lat_size=4,
        lon_size=3,
        time_size=10,
        # background_trend=1.0,
        # background_noise=0.5,
        cluster_key="custom_key",
        perturbation_key="custom_p_key",
        # random_seed=5,
    )

    # Add perturbation
    ponds.add_perturbation(
        perturbation_method=mock_perturbation(),
        shape_method=mock_shape(),
    )

    # Add fake shift for extra test coverage
    ponds.add_shift(
        shift_method=mock_shift(),
        shape_method=mock_shape(),
        correlation_method=mock_corr(),
    )

    # Create temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_ponds.nc")

        # 1. Save to file
        ponds.save(filename=filepath)

        # 1a. File exists?
        assert os.path.exists(filepath), "Saved file does not exist."

        # 2. Load from file
        loaded = PONDS.load(
            filepath,
            type_dict={
                "MockShape": mock_shape,
                "MockPerturbation": mock_perturbation,
                "MockShift": mock_shift,
                "MockCorrelation": mock_corr,
            },
        )

        # 2a. Time series data matches
        np.testing.assert_array_equal(
            loaded.data.ts.values,
            ponds.data.ts.values,
            err_msg="Time series data does not match after load.",
        )

        # 2b. All cluster masks match
        for var in ponds.data.data_vars:
            if var != "ts":
                np.testing.assert_array_equal(
                    loaded.data[var].values,
                    ponds.data[var].values,
                    err_msg=f"Mask data for cluster {var} does not match.",
                )

        # 2c. Attributes match (dataset-level)
        assert loaded.data.attrs == ponds.data.attrs, "Dataset attributes do not match."

        # 2d. Attributes match (ts-level)
        assert loaded.data.ts.attrs == ponds.data.ts.attrs, (
            "Time series attributes do not match."
        )

        # 2e. Attributes match (mask-level)
        for var in ponds.data.data_vars:
            if var != "ts":
                assert loaded.data[var].attrs == ponds.data[var].attrs, (
                    f"Attributes for {var} do not match."
                )
