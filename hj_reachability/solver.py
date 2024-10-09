import contextlib
import functools

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from hj_reachability import artificial_dissipation
from hj_reachability import time_integration
from hj_reachability.finite_differences import upwind_first

from typing import Callable, Text

# Hamiltonian postprocessors.
identity = lambda *x: x[-1]  # Returns the last argument so that this may also be used as a value postprocessor.
backwards_reachable_tube = lambda x: jnp.minimum(x, 0)

# Value postprocessors.
static_obstacle = lambda obstacle: (lambda t, v: jnp.maximum(v, obstacle))


@struct.dataclass
class SolverSettings:
    upwind_scheme: Callable = struct.field(
        default=upwind_first.WENO5,
        pytree_node=False,
    )
    artificial_dissipation_scheme: Callable = struct.field(
        default=artificial_dissipation.global_lax_friedrichs,
        pytree_node=False,
    )
    hamiltonian_postprocessor: Callable = struct.field(
        default=identity,
        pytree_node=False,
    )
    time_integrator: Callable = struct.field(
        default=time_integration.third_order_total_variation_diminishing_runge_kutta,
        pytree_node=False,
    )
    value_postprocessor: Callable = struct.field(
        default=identity,
        pytree_node=False,
    )
    CFL_number: float = 0.75

    @classmethod
    def with_accuracy(cls, accuracy: Text, **kwargs) -> "SolverSettings":
        if accuracy == "low":
            upwind_scheme = upwind_first.first_order
            time_integrator = time_integration.first_order_total_variation_diminishing_runge_kutta
        elif accuracy == "medium":
            upwind_scheme = upwind_first.ENO2
            time_integrator = time_integration.second_order_total_variation_diminishing_runge_kutta
        elif accuracy == "high":
            upwind_scheme = upwind_first.WENO3
            time_integrator = time_integration.third_order_total_variation_diminishing_runge_kutta
        elif accuracy == "very_high":
            upwind_scheme = upwind_first.WENO5
            time_integrator = time_integration.third_order_total_variation_diminishing_runge_kutta
        return cls(upwind_scheme=upwind_scheme, time_integrator=time_integrator, **kwargs)


@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def step(solver_settings, dynamics, grid, time, values, target_time, progress_bar=True):
    with (_try_get_progress_bar(time, target_time)
          if progress_bar is True else contextlib.nullcontext(progress_bar)) as bar:

        def sub_step(time_values):
            t, v = solver_settings.time_integrator(solver_settings, dynamics, grid, *time_values, target_time)
            if bar is not False:
                bar.update_to(jnp.abs(t - bar.reference_time))
            return t, v

        return jax.lax.while_loop(lambda time_values: jnp.abs(target_time - time_values[0]) > 0, sub_step,
                                  (time, values))[1]


@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def solve(solver_settings, dynamics, grid, times, initial_values, progress_bar=True):
    with (_try_get_progress_bar(times[0], times[-1])
          if progress_bar is True else contextlib.nullcontext(progress_bar)) as bar:
        make_carry_and_output_slice = lambda t, v: ((t, v), v)
        return jnp.concatenate([
            initial_values[np.newaxis],
            jax.lax.scan(
                lambda time_values, target_time: make_carry_and_output_slice(
                    target_time, step(solver_settings, dynamics, grid, *time_values, target_time, bar)),
                (times[0], initial_values), times[1:])[1]
        ])


def _try_get_progress_bar(reference_time, target_time):
    try:
        import tqdm
    except ImportError:
        raise ImportError("The option `progress_bar=True` requires the 'tqdm' package to be installed.")
    return TqdmWrapper(tqdm,
                       reference_time,
                       total=jnp.abs(target_time - reference_time),
                       unit="sim_s",
                       bar_format="{l_bar}{bar}| {n:7.4f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                       ascii=True)


class TqdmWrapper:

    def __init__(self, tqdm, reference_time, total, *args, **kwargs):
        self.reference_time = reference_time
        jax.experimental.io_callback(
            lambda total: self._create_tqdm(tqdm, float(total), *args, **kwargs),
            None,
            total,
            ordered=True,
        )

    def _create_tqdm(self, tqdm, total, *args, **kwargs):
        self._tqdm = tqdm.tqdm(total=total, *args, **kwargs)

    def update_to(self, n):
        jax.experimental.io_callback(
            lambda n: self._tqdm.update(float(n) - self._tqdm.n) and None,
            None,
            n,
            ordered=True,
        )

    def close(self):
        jax.experimental.io_callback(
            lambda: self._tqdm.close(),
            None,
            ordered=True,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
