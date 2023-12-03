import functools

import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, Callable, Iterable, List, Mapping, Optional, TypeVar, Union

T = TypeVar("T")
Tree = Union[T, Iterable["Tree[T]"], Mapping[Any, "Tree[T]"]]


def multivmap(fun: Callable,
              in_axes: Tree[Optional[np.ndarray]],
              out_axes: Tree[Optional[np.ndarray]] = None) -> Callable:
    """Applies `jax.vmap` over multiple axes (equivalent to multiple nested `jax.vmap`s).

    Args:
        fun: Function to be mapped over additional axes (see `jax.vmap` for more details).
        in_axes: Similar to the specification of `in_axes` for `jax.vmap`, with the main difference being that instead
            of `Optional[int]` for axis specification, it's `Optional[np.ndarray]`. For each corresponding input of
            `fun`, the `np.ndarray` specifies a sequence of axes to `jax.vmap` over; note that these axes are not
            specified directly as a `list` so as not to conflict with the possible structure of `in_axes`. All
            non-`None` leaves of `in_axes` (there must be at least one) must have the same length. This length is the
            number of times `jax.vmap` will be applied to `fun`.
        out_axes: Similar to the specification of `out_axes` for `jax.vmap`, with the main difference being that instead
            of `Optional[int]` for axis specification, it's `Optional[np.ndarray]`. For each corresponding output of
            `fun`, the `np.ndarray` specifies a sequence of additional mapped axes to appear in the output. The length
            of non-`None` leaves of `out_axes` must be the same as the length of non-`None` leaves of `in_axes`; the
            order of both axes specifications corresponds to successive nested `jax.vmap` applications. If not provided,
            `out_axes` defaults to `in_axes`.

    Returns:
        A batched/vectorized version of `fun` with arguments that correspond to those of `fun`, but with (possibly
        multiple per input) extra array axes at positions indicated by `in_axes`, and a return value that corresponds
        to that of `fun`, but with (possibly multiple per output) extra array axes at positions indicated by `out_axes`.

    Raises:
        ValueError: if any specified axes are negative or repeated.
    """

    def get_axis_sequence(axis_array: np.ndarray) -> List:
        axis_list = axis_array.tolist()
        if any(axis < 0 for axis in axis_list):
            raise ValueError(f"All `multivmap` axes must be nonnegative; got {axis_list}.")
        if len(axis_list) != len(set(axis_list)):
            raise ValueError(f"All `multivmap` axes must be distinct; got {axis_list}.")
        for i in range(len(axis_list)):
            for j in range(i + 1, len(axis_list)):
                if axis_list[i] > axis_list[j]:
                    axis_list[i] -= 1
        return axis_list

    multivmap_kwargs = {"in_axes": in_axes, "out_axes": in_axes if out_axes is None else out_axes}
    axis_sequence_structure = jax.tree_util.tree_structure(
        next(a for a in jax.tree_util.tree_leaves(in_axes) if a is not None).tolist())
    vmap_kwargs = jax.tree_util.tree_transpose(jax.tree_util.tree_structure(multivmap_kwargs), axis_sequence_structure,
                                               jax.tree_map(get_axis_sequence, multivmap_kwargs))
    return functools.reduce(lambda f, kwargs: jax.vmap(f, **kwargs), vmap_kwargs, fun)


def unit_vector(x):
    """Normalizes a vector `x`, returning a unit vector in the same direction, or a zero vector if `x` is zero."""
    norm2 = jnp.sum(jnp.square(x))
    iszero = norm2 < jnp.finfo(jnp.zeros(()).dtype).eps**2
    return jnp.where(iszero, jnp.zeros_like(x), x / jnp.sqrt(jnp.where(iszero, 1, norm2)))
