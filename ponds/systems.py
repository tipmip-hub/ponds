from abc import ABC, abstractmethod
import numpy as np

from tqdm.auto import tqdm


######################## Abstract base class for dynamical systems ###############################

class DynamicalSystem(ABC):
    @abstractmethod
    def integrate(self, parameter_array: np.ndarray) -> np.ndarray:
        """
        Integrates the dynamical system over time.

        Parameters:
            parameter_array (np.ndarray): Array of parameter values for the integration.

        Returns:
            np.ndarray: The integrated states over time.
        """

    @abstractmethod
    def get_parameter_value(self, state: str) -> float:
        """
        Retrieves the parameter value of a specified state of the dynamical system.

        Parameters:
            forced_parameter (str): The name of the parameter to retrieve.
            state (str): The state for which to retrieve the parameter value.

        Returns:
            float: The value of the specified parameter.
        """

###################################### Dynamical Systems #########################################

class Lorenz(DynamicalSystem):
    def __init__(
            self,
            forced_parameter: str = "beta",
            sigma: float = 10.0,
            rho: float = 28.0,
            beta: float = 8/3
    ):
        # store parameters
        self.parameters = {
            "sigma": sigma,
            "rho": rho,
            "beta": beta
        }
        self.forced_parameter = forced_parameter
        if self.forced_parameter not in self.parameters:
            raise ValueError(f"Unsupported parameter: {self.forced_parameter}")
        
        # create dictionary of supported dynamical states (to be extended)
        self.state_dict = {
            "beta": {
                "chaotic_1": 0.6,       # chaotic state
                "chaotic_2": 0.8,       # chaotic state
                "periodic_1": 0.7       # periodic state
            }
        }

    def integrate(
            self,
            parameter_array,
            dt: float = 1e-4,
            sampling_factor: int = 500
        ) -> np.ndarray:
        """
        Generate a Lorenz time series.

        Uses the Lorenz equations to generate a timeseries on the Lorenz attractor. That is, it 
        takes an array of the chosen bifurcation parameter as input, takes intermediate integration
        steps to reduce the resolution of the timeseries, and returns a 1D timeseries (i.e. only
        the first coordinate of the 3D Lorenz timeseries).

        Arguments
        ---------
        parameter_array : numpy NDArray
            array of shifting bifurcation parameter
        dt : float
            integration timestep
        sampling_factor : int
            number of timesteps to take in between sampled timesteps. Total amount of
            integrated timesteps will be N * sampling_factor.

        Returns
        -------
        numpy NDArray
            1D Lorenz timeseries
        """
        # set initial values
        x0, y0, z0 = [.1, .1, .1]

        # spinup to get rid of transient
        spinup = int(10/dt)  # number of steps to spin up
        self.parameters[self.forced_parameter] = parameter_array[0]  # use first value for spinup
        for _ in range(spinup):
            x0, y0, z0 = self.integrate_step(
                x0, y0, z0, dt=dt, **self.parameters)

        # initialize timeseries with spun-up initial values
        timeseries = [[x0, y0, z0]]

        # calculate timeseries, but only store every sampling_factor-th value
        x, y, z = (x0, y0, z0)
        for n in tqdm(range(1, (len(parameter_array))),
                      desc="Integrating Lorenz system",
                      unit=" timesteps"
                      ):
            # get current values
            for m in range(sampling_factor):
                # linearly interpolate beta value for the next step
                self.parameters[self.forced_parameter] = \
                    parameter_array[n-1] + (parameter_array[n] - parameter_array[n-1]) \
                        * (m / sampling_factor)
                # calculate next timestep values
                x, y, z = self.integrate_step(x, y, z, dt, **self.parameters)
            # store the last value of the sampling_factor steps
            timeseries.append([x, y, z])

        # return only the first dimension of the timeseries
        return np.array(timeseries)[:, 0]

    def integrate_step(self, x, y, z, dt, sigma, rho, beta):
        """
        Returns an integration timestep X_(n+1) of the Lorenz system,
        given X_n and the bifurcation parameters.
        """
        x_step = x + (sigma * (y - x)) * dt
        y_step = y + (x * (rho - z) - y) * dt
        z_step = z + (x * y - beta * z) * dt

        return x_step, y_step, z_step

    def get_parameter_value(self, state: str = "c1") -> float:
        """
        Get the parameter value for the desired type of state.
        """
        if state not in self.state_dict[self.forced_parameter]:
            raise ValueError(f"Unsupported state: {state}")

        return self.state_dict[self.forced_parameter][state]
