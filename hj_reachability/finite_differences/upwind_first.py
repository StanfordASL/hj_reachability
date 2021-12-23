import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.polynomial.polynomial as poly

from types import ModuleType
from typing import Any, Callable, Optional, Tuple

Array = Any

WENO_EPS = 1e-6


def weighted_essentially_non_oscillatory(eno_order: int, values: Array, spacing: float,
                                         boundary_condition: Callable[[Array, int], Array]) -> Tuple[Array, Array]:
    """Implements an upwind weighted essentially non-oscillatory (WENO) scheme for first derivative approximation.

    Args:
        eno_order: The order of the underlying essentially non-oscillatory (ENO) scheme; the resulting WENO scheme is
            `(2 * eno_order - 1)`th-order accurate.
        values: 1-dimensional array of function values assumed to be evaluated at a uniform grid in the domain.
        spacing: Grid spacing of the `values`.
        boundary_condition: A function used to pad `values` to implement a boundary condition (e.g., periodic).

    Returns:
        A tuple of arrays `(left_derivatives, right_derivatives)` each the same shape as `values` which contain,
        respectively, left and right approximations of the first derivative at the grid points of `values`.
    """
    if eno_order < 1:
        raise ValueError(f"`eno_order` must be at least 1; got {eno_order}.")

    values = boundary_condition(values, eno_order)
    diffs = (values[1:] - values[:-1]) / spacing

    if eno_order == 1:
        return (diffs[:-1], diffs[1:])

    substencil_approximations = tuple(
        _unrolled_correlate(diffs[i:len(diffs) - eno_order + i], c)
        for (i, c) in enumerate(_diff_coefficients(eno_order)))
    diffs2 = diffs[1:] - diffs[:-1]
    smoothness_indicators = [
        sum(
            _unrolled_correlate(diffs2[i + j:len(diffs2) - eno_order + i + 1], L[j:, j])**2
            for j in range(eno_order - 1))
        for (i, L) in enumerate(np.linalg.cholesky(_smoothness_indicator_quad_form(eno_order)))
    ]
    left_and_right_unnormalized_weights = [[
        c / (s[i:len(s) + i - 1] + WENO_EPS)**2 for (c, s) in zip(coefficients, smoothness_indicators)
    ] for (i, coefficients) in enumerate(_substencil_coefficients(eno_order))]
    return tuple(
        sum(w * a for (w, a) in zip(unnormalized_weights, substencil_approximations[i:eno_order + i])) /
        sum(unnormalized_weights) for (i, unnormalized_weights) in enumerate(left_and_right_unnormalized_weights))


def essentially_non_oscillatory(order: int, values: Array, spacing: float,
                                boundary_condition: Callable[[Array, int], Array]) -> Tuple[Array, Array]:
    """Implements an upwind essentially non-oscillatory (ENO) scheme for first derivative approximation.

    Args:
        order: The desired order of accuracy for the ENO scheme.
        values: 1-dimensional array of function values assumed to be evaluated at a uniform grid in the domain.
        spacing: Grid spacing of the `values`.
        boundary_condition: A function used to pad `values` to implement a boundary condition (e.g., periodic).

    Returns:
        A tuple of arrays `(left_derivatives, right_derivatives)` each the same shape as `values` which contain,
        respectively, left and right approximations of the first derivative at the grid points of `values`.
    """
    if order < 1:
        raise ValueError(f"`order` must be at least 1; got {order}.")

    values = boundary_condition(values, order)
    diffs = (values[1:] - values[:-1]) / spacing

    if order == 1:
        return (diffs[:-1], diffs[1:])

    substencil_approximations = tuple(
        _unrolled_correlate(diffs[i:len(diffs) - order + i], c) for (i, c) in enumerate(_diff_coefficients(order)))

    undivided_differences = []
    for i in range(2, order):
        diffs = diffs[1:] - diffs[:-1]
        undivided_differences.append(diffs[order - i:i - order])

    abs_diffs = jnp.abs(diffs[1:] - diffs[:-1])
    stencil_indices = abs_diffs[1:] < abs_diffs[:-1]
    for diffs in reversed(undivided_differences):
        abs_diffs = jnp.abs(diffs)
        stencil_indices = jnp.where(abs_diffs[1:] < abs_diffs[:-1], stencil_indices[1:] + 1, stencil_indices[:-1])

    return (jnp.select([stencil_indices[:-1] == i for i in range(order - 1)], substencil_approximations[:-2],
                       substencil_approximations[-2]),
            jnp.select([stencil_indices[1:] == i for i in range(order - 1)], substencil_approximations[1:-1],
                       substencil_approximations[-1]))


first_order = WENO1 = functools.partial(weighted_essentially_non_oscillatory, 1)
WENO3 = functools.partial(weighted_essentially_non_oscillatory, 2)
WENO5 = functools.partial(weighted_essentially_non_oscillatory, 3)
ENO1 = functools.partial(essentially_non_oscillatory, 1)
ENO2 = functools.partial(essentially_non_oscillatory, 2)
ENO3 = functools.partial(essentially_non_oscillatory, 3)


def _weighted_essentially_non_oscillatory_vectorized(
        eno_order: int, values: Array, spacing: float, boundary_condition: Callable[[Array, int],
                                                                                    Array]) -> Tuple[Array, Array]:
    """Implements a more "vectorized" but ultimately slower version of `weighted_essentially_non_oscillatory`."""
    if eno_order < 1:
        raise ValueError(f"`eno_order` must be at least 1; got {eno_order}.")

    values = boundary_condition(values, eno_order)
    diffs = (values[1:] - values[:-1]) / spacing

    if eno_order == 1:
        return (diffs[:-1], diffs[1:])

    substencil_approximations = _align_substencil_values(
        jax.vmap(jnp.correlate, (None, 0), 0)(diffs, _diff_coefficients(eno_order)), jnp)
    diffs2 = diffs[1:] - diffs[:-1]
    chol_T = jnp.asarray(np.linalg.cholesky(_smoothness_indicator_quad_form(eno_order)).swapaxes(-1, -2))
    smoothness_indicators = _align_substencil_values(
        jnp.sum(jnp.square(jax.vmap(jax.vmap(jnp.correlate, (None, 0), 1), (None, 0), 0)(diffs2, chol_T)), -1), jnp)
    unscaled_weights = 1 / jnp.square(smoothness_indicators + WENO_EPS)
    unnormalized_weights = (jnp.asarray(_substencil_coefficients(eno_order)[..., np.newaxis]) *
                            jnp.stack([unscaled_weights[:, :-1], unscaled_weights[:, 1:]]))
    weights = unnormalized_weights / jnp.sum(unnormalized_weights, 1, keepdims=True)
    return tuple(jnp.sum(jnp.stack([substencil_approximations[:-1], substencil_approximations[1:]]) * weights, 1))


def _unrolled_correlate(a: Array, v: Array) -> Array:
    """An unrolled equivalent of `np.correlate`."""
    return sum(a[i:len(a) - len(v) + i + 1] * x for (i, x) in enumerate(v))


def _substencils(k: int) -> Array:
    """Returns the `k + 1` subranges of length `k + 1` from the full stencil range `[-k, k + 1)`."""
    return np.arange(k + 1) + np.arange(k + 1)[:, np.newaxis] - k


def _spread_substencil_values(x: Array, np: ModuleType = np) -> Array:
    """Offsets each successive row of a matrix `x` by one additional column."""
    return np.reshape(np.reshape(np.pad(x, ((0, 0), (0, x.shape[0]))), -1)[:-x.shape[0]], (x.shape[0], -1))


def _align_substencil_values(x: Array, np: ModuleType = np) -> Array:
    """Slices and stacks windows, each offset by one column from the previous, from rows of a matrix `x`."""
    return np.reshape(np.pad(np.reshape(x, -1), (0, x.shape[0])), (x.shape[0], -1))[:, :-x.shape[0]]


def _diff_coefficients(k: Optional[int] = None, stencil: Optional[Array] = None) -> Array:
    """Returns first derivative approximation finite difference coefficients for function value first differences."""
    if k is None:
        if stencil is None:
            raise ValueError("One of `k` or `stencil` must be provided.")
        k = stencil.shape[-1] - 1
    else:
        if stencil is None:
            stencil = _substencils(k)
        elif k != stencil.shape[-1] - 1:
            raise ValueError("`k` must match `stencil.shape[-1] - 1` if both arguments are provided; got "
                             f"{(k, stencil.shape[-1] - 1)}.")
    return np.linalg.solve(
        np.diff(poly.polyvander(stencil, k), axis=-2)[..., 1:].swapaxes(-1, -2),
        np.eye(k)[(np.newaxis,) * (stencil.ndim - 1) + (0,)])


def _substencil_coefficients(k: int) -> Array:
    """Returns coefficients for combining substencil approximations to yield higher order left/right approximations."""
    left_coefficients = np.linalg.solve(
        _spread_substencil_values(_diff_coefficients(k))[:-1, :k].T,
        _diff_coefficients(stencil=np.arange(-k, k))[:k])
    return np.array([left_coefficients, left_coefficients[::-1]])


def _polyder_operator(k: int, d: int) -> Array:
    """Returns a matrix `D` such that `D @ p == poly.polyder(p, d)` for polynomials `p` of degree `k`."""
    return np.concatenate([np.zeros((k + 1 - d, d)), np.diag(poly.polyder(np.ones(k + 1), d))], 1)


def _smoothness_indicator_quad_form(k: int) -> Array:
    """Returns quadratic forms for computing substencil smoothness indicators as functions of second differences."""
    interp_poly_second_der = (poly.polyder(np.ones(k + 1), 2)[:, np.newaxis] *
                              np.linalg.inv(np.diff(poly.polyvander(_substencils(k)[1:], k), 2, axis=-2)[..., 2:]))

    quad_form = np.zeros((k, k - 1, k - 1))
    for m in range(k - 1):
        integrator_matrix = 1 / (np.arange(k - 1 - m) + np.arange(k - 1 - m)[:, np.newaxis] + 1)
        interp_poly_m_plus_2_der = _polyder_operator(k - 2, m) @ interp_poly_second_der
        quad_form += interp_poly_m_plus_2_der.swapaxes(-1, -2) @ integrator_matrix @ interp_poly_m_plus_2_der
    return quad_form
