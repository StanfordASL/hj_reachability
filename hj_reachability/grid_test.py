from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from hj_reachability import grid as _grid
from hj_reachability import sets


class BoundaryConditionsTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_grid_interpolate(self):
        grid_domain = sets.Box(np.zeros(2), np.ones(2))
        grid_shape = (3, 2)
        grid = _grid.Grid.from_lattice_parameters_and_boundary_conditions(grid_domain, grid_shape, periodic_dims=1)
        values = np.random.random((3, 2))
        np.testing.assert_allclose(grid.interpolate(values, np.array([0.25, 2.75])), np.mean(values[0:2, 0:2]))
        np.testing.assert_allclose(grid.interpolate(values, np.zeros(2)), values[0, 0])
        np.testing.assert_allclose(grid.interpolate(values, np.ones(2)), values[-1, 0])
        values = np.random.random((3, 2, 3, 4))
        np.testing.assert_allclose(grid.interpolate(values, np.array([0.75, 2.75])), np.mean(values[1:3, 0:2], (0, 1)))
        np.testing.assert_allclose(grid.interpolate(values, np.zeros(2)), values[0, 0])
        np.testing.assert_allclose(grid.interpolate(values, np.ones(2)), values[-1, 0])

    def test_grid_interpolate_on_grid(self):
        grid_domain = sets.Box(jnp.zeros(2), jnp.ones(2))
        grid_shape = (3, 4)
        for value_shape in ((), (5,)):
            values = jnp.array(np.random.random(grid_shape + value_shape))
            grid = _grid.Grid.from_lattice_parameters_and_boundary_conditions(grid_domain, grid_shape)
            np.testing.assert_allclose(jax.vmap(jax.vmap(lambda x: grid.interpolate(values, x)))(grid.states),
                                       values,
                                       atol=1e-6)

            grid = _grid.Grid.from_lattice_parameters_and_boundary_conditions(grid_domain, grid_shape, periodic_dims=0)
            states = grid.states + (grid._is_periodic_dim * np.arange(-3, 4)[:, None, None, None] *
                                    (grid.domain.hi - grid.domain.lo))
            np.testing.assert_allclose(jax.vmap(jax.vmap(jax.vmap(lambda x: grid.interpolate(values, x))))(states),
                                       np.broadcast_to(values, states.shape[:1] + values.shape),
                                       atol=1e-6)

    def test_grid_interpolate_off_grid(self):
        grid_domain = sets.Box(jnp.zeros(2), jnp.ones(2))
        grid_shape = (3, 4)
        for value_shape in ((), (5,)):
            a = np.random.random((2,) + value_shape)
            grid = _grid.Grid.from_lattice_parameters_and_boundary_conditions(grid_domain, grid_shape)
            values = grid.states @ a
            states = grid.domain.lo + np.random.random((100, 2)) * (grid.domain.hi - grid.domain.lo)
            np.testing.assert_allclose(jax.vmap(lambda x: grid.interpolate(values, x))(states), states @ a, atol=1e-6)

            grid = _grid.Grid.from_lattice_parameters_and_boundary_conditions(grid_domain, grid_shape, periodic_dims=0)
            values = jnp.array(np.random.random(grid_shape + value_shape))
            grid_unwrapped = _grid.Grid.from_lattice_parameters_and_boundary_conditions(
                grid.domain, tuple(d + 1 if p else d for d, p in zip(grid.shape, grid._is_periodic_dim)))
            values_unwrapped = jnp.concatenate([values, values[:1]])
            states = states + (grid._is_periodic_dim * np.arange(-3, 4)[:, None, None] *
                               (grid.domain.hi - grid.domain.lo))
            np.testing.assert_allclose(jax.vmap(jax.vmap(lambda x: grid.interpolate(values, x)))(states),
                                       jax.vmap(jax.vmap(lambda x: grid_unwrapped.interpolate(values_unwrapped, x)))
                                       ((states - grid.domain.lo) % (grid.domain.hi - grid.domain.lo) + grid.domain.lo),
                                       atol=1e-6)

    def test_grid_interpolate_extrapolate_nan(self):
        grid_domain = sets.Box(jnp.zeros(2), jnp.ones(2))
        grid_shape = (3, 4)
        for value_shape in ((), (5,)):
            values = jnp.array(np.random.random(grid_shape + value_shape))
            grid = _grid.Grid.from_lattice_parameters_and_boundary_conditions(grid_domain, grid_shape)
            states = grid.domain.lo + (grid.domain.hi - grid.domain.lo) * np.array(
                [[0.5 + dx, 0.5 + dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx or dy])
            result = jax.vmap(lambda x: grid.interpolate(values, x))(states)
            self.assertEqual(result.shape, (8,) + value_shape)
            self.assertTrue(np.all(np.isnan(result)))


if __name__ == "__main__":
    absltest.main()
