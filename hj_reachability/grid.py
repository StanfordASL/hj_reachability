import functools

from flax import struct
import jax.numpy as jnp
import numpy as np

from hj_reachability import boundary_conditions as _boundary_conditions
from hj_reachability.finite_differences import upwind_first
from hj_reachability import sets
from hj_reachability import utils

from typing import Any, Callable, Optional, Tuple, Union
from hj_reachability.boundary_conditions import BoundaryCondition

Array = Any


@struct.dataclass
class Grid:
    """Class for representing Cartesian state grids with uniform spacing in each dimension.

    Attributes:
        states: An `(N + 1)` dimensional array containing the state values at each grid location. The first `N`
            dimensions correspond to the location in the grid, while the last dimension (itself of size `N`) contains
            the state vector.
        domain: A `Box` representing the domain of grid.
        coordinate_vectors: A tuple of `N` arrays containing the discrete state values in each dimension. The `states`
            attribute is produced by `stack`ing a `meshgrid` of these coordinate vectors.
        spacings: A tuple of `N` scalars containing the grid spacing (the difference between successive elements of the
            corresponding coordinate vector) in each dimension.
        boundary_conditions: A tuple of `N` boundary conditions for each dimension. These boundary conditions are
            functions used to pad values (notably not stored in this `Grid` data structure) to implement a boundary
            condition (e.g., periodic).
    """
    states: Array
    domain: sets.Box
    coordinate_vectors: Tuple[Array, ...]
    spacings: Tuple[Array, ...]
    boundary_conditions: Tuple[BoundaryCondition, ...] = struct.field(pytree_node=False)

    @classmethod
    def from_lattice_parameters_and_boundary_conditions(
            cls,
            domain: sets.Box,
            shape: Tuple[int, ...],
            boundary_conditions: Optional[Tuple[BoundaryCondition, ...]] = None,
            periodic_dims: Optional[Union[int, Tuple[int, ...]]] = None) -> "Grid":
        """Constructs a `Grid` from a domain, shape, and boundary conditions.

        Args:
            domain: A `Box` representing the domain of grid.
            shape: A tuple of `N` integers denoting the number of discretization nodes in each dimension.
            boundary_conditions: A tuple of `N` boundary conditions for each dimension. If not provided, defaults to
                `extrapolate_away_from_zero` in each dimension, with the exception of those dimensions that appear in
                `periodic_dims` where the `periodic` boundary condition is used instead.
            periodic_dims: A single integer or tuple of integers denoting which dimensions are periodic in the case that
                the `boundary_conditions` are not explicitly provided as input to this factory method.

        Returns:
            A `Grid` constructed according to the provided specifications.
        """
        ndim = len(shape)
        if boundary_conditions is None:
            if not isinstance(periodic_dims, tuple):
                periodic_dims = (periodic_dims,)
            boundary_conditions = tuple(
                _boundary_conditions.periodic if i in periodic_dims else _boundary_conditions.extrapolate_away_from_zero
                for i in range(ndim))

        coordinate_vectors, spacings = zip(
            *(jnp.linspace(l, h, n, endpoint=bc is not _boundary_conditions.periodic, retstep=True)
              for l, h, n, bc in zip(domain.lo, domain.hi, shape, boundary_conditions)))
        states = jnp.stack(jnp.meshgrid(*coordinate_vectors, indexing="ij"), -1)

        return cls(states, domain, coordinate_vectors, spacings, boundary_conditions)

    @property
    def ndim(self) -> int:
        """Returns the dimension `N` of the grid."""
        return self.states.ndim - 1

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the grid, a tuple of `N` integers."""
        return self.states.shape[:-1]

    def upwind_grad_values(self, upwind_scheme: Callable, values: Array) -> Tuple[Array, Array]:
        """Returns `(left_grad_values, right_grad_values)`."""
        left_derivatives, right_derivatives = zip(*[
            utils.multivmap(lambda values: upwind_scheme(values, spacing, boundary_condition),
                            np.array([j
                                      for j in range(self.ndim)
                                      if j != i]))(values)
            for i, (spacing, boundary_condition) in enumerate(zip(self.spacings, self.boundary_conditions))
        ])
        return (jnp.stack(left_derivatives, -1), jnp.stack(right_derivatives, -1))

    def grad_values(self, values: Array, upwind_scheme: Optional[Callable] = None) -> Array:
        """Returns a central difference-based approximation of `grad_values`."""
        # TODO: Implement central difference schemes in `hj_reachability.finite_differences`.
        if upwind_scheme is None:
            upwind_scheme = upwind_first.first_order
        return sum(self.upwind_grad_values(upwind_scheme, values)) / 2

    def position(self, state: Array) -> Array:
        """Returns an array of `float`s corresponding to the position of `state` in the grid."""
        position = (state - self.domain.lo) / jnp.array(self.spacings)
        return jnp.where(self._is_periodic_dim, position % np.array(self.shape), position)

    def nearest_index(self, state: Array) -> Array:
        """Returns the result of rounding `self.position(state)` to the nearest grid index."""
        return jnp.round(self.position(state)).astype(jnp.int32)

    def interpolate(self, values, state):
        """Interpolates `values` (possibly multidimensional per node) defined over the grid at the given `state`."""
        position = (state - self.domain.lo) / jnp.array(self.spacings)
        index_lo = jnp.floor(position).astype(jnp.int32)
        index_hi = index_lo + 1
        weight_hi = position - index_lo
        weight_lo = 1 - weight_hi
        index_lo, index_hi = tuple(
            jnp.where(self._is_periodic_dim, index % np.array(self.shape), jnp.clip(index, 0,
                                                                                    np.array(self.shape) - 1))
            for index in (index_lo, index_hi))
        weight = functools.reduce(lambda x, y: x * y, jnp.ix_(*jnp.stack([weight_lo, weight_hi], -1)))
        # TODO: Double-check numerical stability here and/or switch to `tuple`s and `itertools.product` for clarity.
        result = jnp.sum(
            weight[(...,) + (np.newaxis,) * (values.ndim - self.ndim)] *
            values[jnp.ix_(*jnp.stack([index_lo, index_hi], -1))], list(range(self.ndim)))
        return jnp.where(jnp.any(~self._is_periodic_dim & ((state < self.domain.lo) | (state > self.domain.hi))),
                         jnp.nan, result)

    @property
    def _is_periodic_dim(self) -> Array:
        """Returns a boolean vector indicating which dimensions (if any) are periodic."""
        return np.array([bc is _boundary_conditions.periodic for bc in self.boundary_conditions])
