import abc

import jax.numpy as jnp


class Dynamics(metaclass=abc.ABCMeta):
    """Abstract base class for representing continuous-time dynamics in the context of Hamilton-Jacobi reachability.

    TODO: Consider allowing for state/time-dependent control/disturbance spaces.

    Attributes:
        control_mode: Whether the controller is trying to "max"imize or "min"imize the value.
        disturbance_mode: Whether the disturbance is trying to "max"imize or "min"imize the value.
        control_space: A `BoundedSet` defining the (time-invariant) set of possible controls.
        disturbance_space: A `BoundedSet` defining the (time-invariant) set of possible disturbances.
    """

    def __init__(self, control_mode, disturbance_mode, control_space, disturbance_space):
        self.control_mode = control_mode
        self.disturbance_mode = disturbance_mode
        self.control_space = control_space
        self.disturbance_space = disturbance_space

    @abc.abstractmethod
    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""

    @abc.abstractmethod
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""

    def optimal_control(self, state, time, grad_value):
        """Computes the optimal control realized by the HJ PDE Hamiltonian."""
        return self.optimal_control_and_disturbance(state, time, grad_value)[0]

    def optimal_disturbance(self, state, time, grad_value):
        """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
        return self.optimal_control_and_disturbance(state, time, grad_value)[1]

    def hamiltonian(self, state, time, value, grad_value):
        """Evaluates the HJ PDE Hamiltonian."""
        del value  # unused
        control, disturbance = self.optimal_control_and_disturbance(state, time, grad_value)
        return grad_value @ self(state, control, disturbance, time)

    @abc.abstractmethod
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""


class ControlAndDisturbanceAffineDynamics(Dynamics):
    """Abstract base class for representing control- and disturbance-affine dynamics."""

    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_jacobian(state, time) @ disturbance)

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time):
        """Implements the open loop dynamics `f(x, t)`."""

    @abc.abstractmethod
    def control_jacobian(self, state, time):
        """Implements the control Jacobian `G_u(x, t)`."""

    @abc.abstractmethod
    def disturbance_jacobian(self, state, time):
        """Implements the disturbance Jacobian `G_d(x, t)`."""

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return (self.control_space.extreme_point(control_direction),
                self.disturbance_space.extreme_point(disturbance_direction))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_jacobian(state, time)) @ self.disturbance_space.max_magnitudes)
