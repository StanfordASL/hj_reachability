import functools

import jax
import jax.numpy as jnp
import numpy as np

from hj_reachability import utils


def lax_friedrichs_numerical_hamiltonian(hamiltonian, state, time, value, left_grad_value, right_grad_value,
                                         dissipation_coefficients):
    hamiltonian_value = hamiltonian(state, time, value, (left_grad_value + right_grad_value) / 2)
    dissipation_value = dissipation_coefficients @ (right_grad_value - left_grad_value) / 2
    return hamiltonian_value - dissipation_value


@functools.partial(jax.jit, static_argnames="dynamics")
def euler_step(solver_settings, dynamics, grid, time, values, time_step=None, max_time_step=None):
    time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
    signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics.hamiltonian(*args, **kwargs)
    left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, values)
    dissipation_coefficients = solver_settings.artificial_dissipation_scheme(dynamics.partial_max_magnitudes,
                                                                             grid.states, time, values,
                                                                             left_grad_values, right_grad_values)
    dvalues_dt = -solver_settings.hamiltonian_postprocessor(time_direction * utils.multivmap(
        lambda state, value, left_grad_value, right_grad_value, dissipation_coefficients:
        (lax_friedrichs_numerical_hamiltonian(signed_hamiltonian, state, time, value,
                                              left_grad_value, right_grad_value, dissipation_coefficients)),
        np.arange(grid.ndim))(grid.states, values, left_grad_values, right_grad_values, dissipation_coefficients))
    if time_step is None:
        time_step_bound = 1 / jnp.max(jnp.sum(dissipation_coefficients / jnp.array(grid.spacings), -1))
        time_step = time_direction * jnp.minimum(solver_settings.CFL_number * time_step_bound, jnp.abs(max_time_step))
    # TODO: Think carefully about whether `solver_settings.value_postprocessor` should be applied here instead.
    return time + time_step, values + time_step * dvalues_dt


def first_order_total_variation_diminishing_runge_kutta(solver_settings, dynamics, grid, time, values, target_time):
    time_1, values_1 = euler_step(solver_settings, dynamics, grid, time, values, max_time_step=target_time - time)
    return time_1, solver_settings.value_postprocessor(time_1, values_1)


def second_order_total_variation_diminishing_runge_kutta(solver_settings, dynamics, grid, time, values, target_time):
    time_1, values_1 = euler_step(solver_settings, dynamics, grid, time, values, max_time_step=target_time - time)
    time_step = time_1 - time
    _, values_2 = euler_step(solver_settings, dynamics, grid, time_1, values_1, time_step)
    return time_1, solver_settings.value_postprocessor(time_1, (values + values_2) / 2)


def third_order_total_variation_diminishing_runge_kutta(solver_settings, dynamics, grid, time, values, target_time):
    time_1, values_1 = euler_step(solver_settings, dynamics, grid, time, values, max_time_step=target_time - time)
    time_step = time_1 - time
    _, values_2 = euler_step(solver_settings, dynamics, grid, time_1, values_1, time_step)
    time_0_5, values_0_5 = time + time_step / 2, (3 / 4) * values + (1 / 4) * values_2
    _, values_1_5 = euler_step(solver_settings, dynamics, grid, time_0_5, values_0_5, time_step)
    return time_1, solver_settings.value_postprocessor(time_1, (1 / 3) * values + (2 / 3) * values_1_5)
