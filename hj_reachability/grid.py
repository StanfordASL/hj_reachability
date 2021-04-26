import dataclasses

import chex
import jax.numpy as jnp
import numpy as np

from hj_reachability import boundary_conditions as _boundary_conditions
from hj_reachability import sets
from hj_reachability import utils

from chex import Array
from typing import Callable, Optional, Tuple, Union
from hj_reachability.boundary_conditions import BoundaryCondition


@dataclasses.dataclass(frozen=True)
class Grid:
    """Class for representing Cartesian state grids with uniform spacing in each dimension.

    Attributes:
        states: An `(N + 1)` dimensional array containing the state values at each grid location. The first `N`
            dimensions correspond to the location in the grid, while the last dimension (itself of size `N`) contains
            the state vector.
        state_domain: A `Box` representing the domain of grid.
        coordinate_vectors: A tuple of `N` arrays containing the discrete state values in each dimension. The `states`
            attribute is produced by `stack`ing a `meshgrid` of these coordinate vectors.
        spacings: A tuple of `N` scalars containing the grid spacing (the difference between successive elements of the
            corresponding coordinate vector) in each dimension.
        boundary_conditions: A tuple of `N` boundary conditions for each dimension. These boundary conditions are
            functions used to pad values (notably not stored in this `Grid` data structure) to implement a boundary
            condition (e.g., periodic).
    """
    states: Array
    state_domain: sets.Box
    coordinate_vectors: Tuple[Array, ...]
    spacings: Tuple[Array, ...]
    boundary_conditions: Tuple[BoundaryCondition, ...]

    @classmethod
    def from_grid_definition_and_initial_values(cls,
                                                state_domain: sets.Box,
                                                shape: Tuple[int, ...],
                                                boundary_conditions: Optional[Tuple[BoundaryCondition, ...]] = None,
                                                periodic_dims: Optional[Union[int, Tuple[int, ...]]] = None) -> "Grid":
        """Constructs a `Grid` from a domain, shape, and boundary conditions.

        Args:
            state_domain: A `Box` representing the domain of grid.
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
              for l, h, n, bc in zip(state_domain.lo, state_domain.hi, shape, boundary_conditions)))
        states = jnp.stack(jnp.meshgrid(*coordinate_vectors, indexing="ij"), -1)

        return cls(states, state_domain, coordinate_vectors, spacings, boundary_conditions)

    @property
    def ndim(self) -> int:
        """Returns the dimension `N` of the grid."""
        return self.states.ndim - 1

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the grid, a tuple of `N` integers."""
        return self.states.shape[:-1]

    @property
    def arrays(self) -> "GridArrays":
        """Returns the arrays that define the nodes of the grid."""
        return GridArrays(states=self.states,
                          state_domain=self.state_domain,
                          coordinate_vectors=self.coordinate_vectors,
                          spacings=self.spacings)

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


@chex.dataclass(frozen=True)
class GridArrays:
    """Arrays that define the nodes of a `Grid`; see the `Grid` docstring for more details."""
    states: Array
    state_domain: sets.Box
    coordinate_vectors: Tuple[Array, ...]
    spacings: Tuple[float, ...]
