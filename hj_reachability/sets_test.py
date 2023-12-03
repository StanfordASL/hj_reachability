from absl.testing import absltest
import jax
import numpy as np

from hj_reachability import sets


class SetsTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_box(self):
        box = sets.Box(np.ones(3), 2 * np.ones(3))
        np.testing.assert_allclose(box.extreme_point(np.array([1, -1, 1])), np.array([2, 1, 2]))
        self.assertTrue(np.all(np.isfinite(box.extreme_point(np.zeros(3)))))
        self.assertEqual(box.bounding_box, box)
        np.testing.assert_allclose(box.max_magnitudes, 2 * np.ones(3))
        self.assertEqual(box.ndim, 3)

    def test_ball(self):
        ball = sets.Ball(np.ones(3), np.sqrt(3))
        np.testing.assert_allclose(ball.extreme_point(np.array([1, -1, 1])), np.array([2, 0, 2]), atol=1e-6)
        self.assertTrue(np.all(np.isfinite(ball.extreme_point(np.zeros(3)))))
        jax.tree_map(np.testing.assert_allclose, ball.bounding_box,
                     sets.Box((1 - np.sqrt(3)) * np.ones(3), (1 + np.sqrt(3)) * np.ones(3)))
        np.testing.assert_allclose(ball.max_magnitudes, (1 + np.sqrt(3)) * np.ones(3))
        self.assertEqual(ball.ndim, 3)


if __name__ == "__main__":
    absltest.main()
