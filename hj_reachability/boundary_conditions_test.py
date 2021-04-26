from absl.testing import absltest
import numpy as np

from hj_reachability import boundary_conditions


class BoundaryConditionsTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_periodic(self):
        x = np.arange(5)
        np.testing.assert_array_equal(boundary_conditions.periodic(x, 0), x)
        np.testing.assert_array_equal(boundary_conditions.periodic(x, 1), [4, 0, 1, 2, 3, 4, 0])
        np.testing.assert_array_equal(boundary_conditions.periodic(x, 2), [3, 4, 0, 1, 2, 3, 4, 0, 1])

    def test_extrapolate(self):
        x = np.arange(5)
        np.testing.assert_array_equal(boundary_conditions.extrapolate(x, 0), x)
        np.testing.assert_array_equal(boundary_conditions.extrapolate(x, 1), np.arange(-1, 6))
        np.testing.assert_array_equal(boundary_conditions.extrapolate(x, 2), np.arange(-2, 7))

    def test_extrapolate_away_from_zero(self):
        x = np.arange(1, 5)
        np.testing.assert_array_equal(boundary_conditions.extrapolate_away_from_zero(x, 0), x)
        np.testing.assert_array_equal(boundary_conditions.extrapolate_away_from_zero(x, 1), [2, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(boundary_conditions.extrapolate_away_from_zero(x, 2), [3, 2, 1, 2, 3, 4, 5, 6])

        x = x[::-1]
        np.testing.assert_array_equal(boundary_conditions.extrapolate_away_from_zero(x, 0), x)
        np.testing.assert_array_equal(boundary_conditions.extrapolate_away_from_zero(x, 1), [5, 4, 3, 2, 1, 2])
        np.testing.assert_array_equal(boundary_conditions.extrapolate_away_from_zero(x, 2), [6, 5, 4, 3, 2, 1, 2, 3])


if __name__ == "__main__":
    absltest.main()
