import contextlib
import dataclasses
import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from hj_reachability import artificial_dissipation
from hj_reachability import grid as _grid
from hj_reachability import time_integration
from hj_reachability.finite_differences import upwind_first

from typing import Callable, Text

# Hamiltonian postprocessors.
identity = lambda x: x
backwards_reachable_tube = lambda x: jnp.minimum(x, 0)


@dataclasses.dataclass(frozen=True)
class SolverSettings:
    upwind_scheme: Callable = upwind_first.WENO5
    artificial_dissipation_scheme: Callable = artificial_dissipation.global_lax_friedrichs
    hamiltonian_postprocessor: Callable = identity
    time_integrator: Callable = time_integration.third_order_total_variation_diminishing_runge_kutta
    CFL_number: float = 0.75

    @classmethod
    def with_accuracy(cls, accuracy: Text, **kwargs) -> "SolverSettings":
        if accuracy == "low":
            upwind_scheme = upwind_first.first_order
            time_integrator = time_integration.first_order_total_variation_diminishing_runge_kutta
        elif accuracy == "medium":
            upwind_scheme = upwind_first.WENO3
            time_integrator = time_integration.second_order_total_variation_diminishing_runge_kutta
        elif accuracy == "high":
            upwind_scheme = upwind_first.WENO3
            time_integrator = time_integration.third_order_total_variation_diminishing_runge_kutta
        elif accuracy == "very_high":
            upwind_scheme = upwind_first.WENO5
            time_integrator = time_integration.third_order_total_variation_diminishing_runge_kutta
        return cls(upwind_scheme=upwind_scheme, time_integrator=time_integrator, **kwargs)


def step(solver_settings, dynamics, grid, time, values, target_time, progress_bar=False, compile_loop=True):
    if compile_loop and not progress_bar:
        return _step(solver_settings, dynamics, grid.boundary_conditions, grid.arrays, time, values, target_time)
    if compile_loop and progress_bar:
        # TODO: Look into `jax.experimental.host_callback` for progress monitoring under `jax.jit`.
        warnings.warn("The option `progress_bar=True` is incompatible with (and overrides) `compile_loop=True`.")
    with (_try_get_progress_bar(np.abs(target_time - time)) if progress_bar else contextlib.nullcontext()) as bar:
        initial_time = time
        while np.abs(target_time - time) > 0:
            time, values = solver_settings.time_integrator(solver_settings, dynamics, grid, time, values, target_time)
            if bar is not None:
                bar.update(np.abs(time - initial_time) - bar.n)
    return values


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _step(solver_settings, dynamics, boundary_conditions, grid_arrays, time, values, target_time):
    grid = _grid.Grid(**grid_arrays, boundary_conditions=boundary_conditions)
    return jax.lax.while_loop(
        lambda time_values: jnp.abs(target_time - time_values[0]) > 0,
        lambda time_values: solver_settings.time_integrator(solver_settings, dynamics, grid, *time_values, target_time),
        (time, values))[1]


def solve(solver_settings, dynamics, grid, times, initial_values, progress_bar=False, compile_loop=True):
    if compile_loop and not progress_bar:
        return _solve(solver_settings, dynamics, grid.boundary_conditions, grid.arrays, times, initial_values)
    if compile_loop and progress_bar:
        # TODO: Look into `jax.experimental.host_callback` for progress monitoring under `jax.jit`.
        warnings.warn("The option `progress_bar=True` is incompatible with (and overrides) `compile_loop=True`.")
    with (_try_get_progress_bar(np.abs(times[-1] - times[0])) if progress_bar else contextlib.nullcontext()) as bar:
        all_values = [initial_values]
        time, values = times[0], initial_values
        for target_time in times[1:]:
            while np.abs(target_time - time) > 0:
                time, values = solver_settings.time_integrator(solver_settings, dynamics, grid, time, values,
                                                               target_time)
                if bar is not None:
                    bar.update(np.abs(time - times[0]) - bar.n)
            all_values.append(values)
    return jnp.stack(all_values)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _solve(solver_settings, dynamics, boundary_conditions, grid_arrays, times, initial_values):
    grid = _grid.Grid(**grid_arrays, boundary_conditions=boundary_conditions)
    helper = lambda t, v: ((t, v), v)
    return jnp.concatenate([
        initial_values[np.newaxis],
        jax.lax.scan(
            lambda time_values, target_time: helper(target_time,
                                                    step(solver_settings, dynamics, grid, *time_values, target_time)),
            (times[0], initial_values), times[1:])[1]
    ])


def _try_get_progress_bar(total):
    try:
        import tqdm
    except ImportError:
        raise ImportError("The option `progress_bar=True` requires the 'tqdm' package to be installed.")
    return tqdm.tqdm(total=total,
                     unit="sim_s",
                     bar_format="{l_bar}{bar}| {n:7.4f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                     ascii=True)
