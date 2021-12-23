import jax.numpy as jnp

from typing import Any, Callable

Array = Any

BoundaryCondition = Callable[[Array, int], Array]


def periodic(x: Array, pad_width: int) -> Array:
    """Pads a 1D array `x` by wrapping values, using the start values to pad the end and vice versa."""
    return jnp.pad(x, ((pad_width, pad_width)), "wrap")


def extrapolate(x: Array, pad_width: int) -> Array:
    """Pads a 1D array `x` by extrapolating using the slope at each end."""
    return jnp.concatenate(
        [x[0] + (x[1] - x[0]) * jnp.arange(-pad_width, 0), x, x[-1] + (x[-1] - x[-2]) * jnp.arange(1, pad_width + 1)])


def extrapolate_away_from_zero(x: Array, pad_width: int) -> Array:
    """Pads a 1D array `x` by extrapolating away from zero using the (possibly negated) slope at each end."""
    return jnp.concatenate([
        x[0] - jnp.sign(x[0]) * jnp.abs(x[1] - x[0]) * jnp.arange(-pad_width, 0), x,
        x[-1] + jnp.sign(x[-1]) * jnp.abs(x[-1] - x[-2]) * jnp.arange(1, pad_width + 1)
    ])
