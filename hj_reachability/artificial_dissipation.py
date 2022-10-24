import jax
import jax.numpy as jnp
import numpy as np

from hj_reachability import sets
from hj_reachability import utils


def global_lax_friedrichs(partial_max_magnitudes, states, time, values, left_grad_values, right_grad_values):
    """Implements the Global Lax-Friedrichs (GLF) scheme for computing dissipation coefficients."""
    grid_axes = np.arange(values.ndim)
    grad_value_box = sets.Box(jnp.minimum(jnp.min(left_grad_values, grid_axes), jnp.min(right_grad_values, grid_axes)),
                              jnp.maximum(jnp.max(left_grad_values, grid_axes), jnp.max(right_grad_values, grid_axes)))
    return utils.multivmap(lambda state, value: partial_max_magnitudes(state, time, value, grad_value_box),
                           grid_axes)(states, values)


def local_lax_friedrichs(partial_max_magnitudes, states, time, values, left_grad_values, right_grad_values):
    """Implements the Local Lax-Friedrichs (LLF) scheme for computing dissipation coefficients."""
    grid_axes = np.arange(values.ndim)
    global_grad_value_box = sets.Box(
        jnp.minimum(jnp.min(left_grad_values, grid_axes), jnp.min(right_grad_values, grid_axes)),
        jnp.maximum(jnp.max(left_grad_values, grid_axes), jnp.max(right_grad_values, grid_axes)))
    local_local_grad_value_boxes = sets.Box(jnp.minimum(left_grad_values, right_grad_values),
                                            jnp.maximum(left_grad_values, right_grad_values))
    local_grad_value_boxes = jax.tree_map(
        lambda global_grad_value, local_local_grad_values:
        (jnp.broadcast_to(global_grad_value, values.shape +
                          (values.ndim,) * 2).at[..., grid_axes, grid_axes].set(local_local_grad_values)),
        global_grad_value_box, local_local_grad_value_boxes)
    return utils.multivmap(
        lambda state, value, grad_value_box: partial_max_magnitudes(state, time, value, grad_value_box),
        grid_axes)(states, values, local_grad_value_boxes)


def local_local_lax_friedrichs(partial_max_magnitudes, states, time, values, left_grad_values, right_grad_values):
    """Implements the Local Local Lax-Friedrichs (LLLF) scheme for computing dissipation coefficients."""
    grid_axes = np.arange(values.ndim)
    local_local_grad_value_boxes = sets.Box(jnp.minimum(left_grad_values, right_grad_values),
                                            jnp.maximum(left_grad_values, right_grad_values))
    return utils.multivmap(
        lambda state, value, grad_value_box: partial_max_magnitudes(state, time, value, grad_value_box),
        grid_axes)(states, values, local_local_grad_value_boxes)
