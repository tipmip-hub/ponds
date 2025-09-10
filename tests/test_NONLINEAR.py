import numpy as np
import pytest

from ponds.nonlinear import Lorenz

# tests for the nonlinear module

def test_Lorenz_get_parameter_value():
    system = Lorenz()

    # check for correct type of dict entries
    for state in system.state_dict["beta"].keys():
        value = system.get_parameter_value(state=state)
        assert isinstance(value, float)

    # check for error if state is not supported
    made_up_state = "b3"
    with pytest.raises(ValueError, match=f"Unsupported state: {made_up_state}"):
        system.get_parameter_value(state=made_up_state)

def test_Lorenz_integrate():
    system = Lorenz()

    N = 100
    param_value = Lorenz().get_parameter_value(state="c1")
    param_array = np.full(N, param_value)

    timeseries = system.integrate(parameter_array=param_array)

    assert isinstance(timeseries, np.ndarray)
    assert timeseries.shape == (N,)
    assert not np.isnan(timeseries).any()
