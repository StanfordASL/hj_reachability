from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from hj_reachability import utils


class UtilsTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_multivmap(self):
        a = np.random.random((3, 4, 5, 6))
        np.testing.assert_allclose(utils.multivmap(jnp.max, np.array([0, 1]))(a), np.max(a, (2, 3)))
        np.testing.assert_allclose(utils.multivmap(jnp.max, np.array([0, 1, 2]))(a), np.max(a, -1))
        np.testing.assert_allclose(utils.multivmap(jnp.max, np.array([0, 1, 3]), np.array([0, 1, 2]))(a), np.max(a, 2))
        np.testing.assert_allclose(
            utils.multivmap(jnp.max, np.array([1, 0, 2]), np.array([0, 1, 2]))(a),
            np.max(a, 3).swapaxes(0, 1))
        np.testing.assert_allclose(
            utils.multivmap(jnp.max, np.array([3, 2]), np.array([0, 1]))(a),
            np.max(a, (0, 1)).swapaxes(0, 1))

    def test_unit_vector(self):
        unsafe_unit_vector = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        for d in range(1, 4):
            np.testing.assert_array_equal(utils.unit_vector(np.zeros(d)), np.zeros(d))
            self.assertTrue(np.all(np.isfinite(jax.jacobian(utils.unit_vector)(np.zeros(d)))))
            self.assertTrue(np.all(np.isnan(jax.jacobian(unsafe_unit_vector)(np.zeros(d)))))
            a = np.random.random((100, d))
            np.testing.assert_allclose(jax.vmap(utils.unit_vector)(a), unsafe_unit_vector(a), atol=1e-6)
            np.testing.assert_allclose(jax.vmap(jax.jacobian(utils.unit_vector))(a),
                                       jax.vmap(jax.jacobian(unsafe_unit_vector))(a),
                                       atol=1e-6)


if __name__ == "__main__":
    absltest.main()
