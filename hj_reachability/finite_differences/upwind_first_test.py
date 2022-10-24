from absl.testing import absltest
import jax
import numpy as np

from hj_reachability import boundary_conditions
from hj_reachability.finite_differences import upwind_first


class UpwindFirstTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_weighted_essentially_non_oscillatory(self):

        def _WENO5(values, spacing, boundary_condition):
            values = boundary_condition(values, 3)
            diffs = (values[1:] - values[:-1]) / spacing

            def compute_weno(v):
                phi = [
                    v[0] / 3 - 7 * v[1] / 6 + 11 * v[2] / 6,
                    -v[1] / 6 + 5 * v[2] / 6 + v[3] / 3,
                    v[2] / 3 + 5 * v[3] / 6 - v[4] / 6,
                ]
                s = [(13 / 12) * (v[0] - 2 * v[1] + v[2])**2 + (1 / 4) * (v[0] - 4 * v[1] + 3 * v[2])**2,
                     (13 / 12) * (v[1] - 2 * v[2] + v[3])**2 + (1 / 4) * (v[1] - v[3])**2,
                     (13 / 12) * (v[2] - 2 * v[3] + v[4])**2 + (1 / 4) * (3 * v[2] - 4 * v[3] + v[4])**2]
                a = [w / (x + upwind_first.WENO_EPS)**2 for (w, x) in zip([0.1, 0.6, 0.3], s)]
                w = [x / sum(a) for x in a]
                return sum(p * w for (p, w) in zip(phi, w))

            return (compute_weno([diffs[i:-5 + i] for i in range(5)]),
                    compute_weno([diffs[5 - i:None if i == 0 else -i] for i in range(5)]))

        values = np.random.rand(1000)
        spacing = 0.1
        jax.tree_map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-5),
                     upwind_first.WENO5(values, spacing, boundary_conditions.periodic),
                     _WENO5(values, spacing, boundary_conditions.periodic))

    def test_essentially_non_oscillatory(self):

        def _brute_force_essentially_non_oscillatory(order, values, spacing, boundary_condition):

            def _divided_difference(x, i, spacing=1):
                if isinstance(i, int):
                    return x[i]
                order = len(i) - 1
                return np.diff(x[i], order)[0] / (np.math.factorial(order) * spacing**order)

            v = np.array(boundary_condition(values, order))
            x = np.arange(len(v)) * spacing

            p = [np.poly1d(v[i]) for i in range(order - 1, len(v) - order)]
            ks = []
            for i in range(len(p)):
                j = i + order - 1
                p[i] += _divided_difference(v, [j, j + 1], spacing) * np.poly1d([x[j]], True)
                k = j
                for d in range(2, order + 1):
                    a = _divided_difference(v, np.arange(k, k + d + 1), spacing)
                    b = _divided_difference(v, np.arange(k - 1, k + d), spacing)
                    if np.abs(a) >= np.abs(b):
                        c = b
                        k_next = k - 1
                    else:
                        c = a
                        k_next = k
                    p[i] += c * np.poly1d(x[k:k + d], True)
                    k = k_next
                ks.append(k - j)
            p_x = [np.polyder(f) for f in p]
            return (np.array([np.polyval(f, x) for (f, x) in zip(p_x[:-1], x[order:-order])]),
                    np.array([np.polyval(f, x) for (f, x) in zip(p_x[1:], x[order:-order])]))

        values = np.random.rand(1000)
        spacing = 0.1
        for order in range(1, 5):
            jax.tree_map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-5),
                         upwind_first.essentially_non_oscillatory(order, values, spacing, boundary_conditions.periodic),
                         _brute_force_essentially_non_oscillatory(order, values, spacing, boundary_conditions.periodic))

    def test_weighted_essentially_non_oscillatory_vectorized(self):
        values = np.random.rand(1000)
        spacing = 0.1
        for eno_order in range(1, 5):
            jax.tree_map(
                lambda x, y: np.testing.assert_allclose(x, y, atol=1e-5),
                upwind_first.weighted_essentially_non_oscillatory(eno_order, values, spacing,
                                                                  boundary_conditions.periodic),
                upwind_first._weighted_essentially_non_oscillatory_vectorized(eno_order, values, spacing,
                                                                              boundary_conditions.periodic))

    def test_diff_coefficients(self):
        # k = 1
        np.testing.assert_allclose(upwind_first._diff_coefficients(1), np.ones((2, 1)))

        # k = 2
        np.testing.assert_allclose(upwind_first._diff_coefficients(2), np.array([[-1, 3], [1, 1], [3, -1]]) / 2)

        # k = 3
        np.testing.assert_allclose(upwind_first._diff_coefficients(3),
                                   np.array([[2, -7, 11], [-1, 5, 2], [2, 5, -1], [11, -7, 2]]) / 6)

    def test_substencil_coefficients(self):
        # k = 1
        np.testing.assert_allclose(upwind_first._substencil_coefficients(1), np.ones((2, 1)))

        # k = 2
        np.testing.assert_allclose(upwind_first._substencil_coefficients(2), np.array([[1, 2], [2, 1]]) / 3)

        # k = 3
        np.testing.assert_allclose(upwind_first._substencil_coefficients(3), np.array([[1, 6, 3], [3, 6, 1]]) / 10)

    def test_smoothness_indicator_quad_form(self):
        diff_operator = lambda k: np.eye(k - 1, k, 1) - np.eye(k - 1, k, 0)
        square_outer = lambda v: v[..., np.newaxis] * v[..., np.newaxis, :]

        # k = 1
        np.testing.assert_allclose(
            diff_operator(1).T @ upwind_first._smoothness_indicator_quad_form(1) @ diff_operator(1), [[[0]]])

        # k = 2
        np.testing.assert_allclose(
            diff_operator(2).T @ upwind_first._smoothness_indicator_quad_form(2) @ diff_operator(2),
            square_outer(np.array([[1, -1], [1, -1]])))

        # k = 3
        np.testing.assert_allclose(
            diff_operator(3).T @ upwind_first._smoothness_indicator_quad_form(3) @ diff_operator(3),
            (13 / 12) * square_outer(np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])) +
            (1 / 4) * square_outer(np.array([[1, -4, 3], [1, 0, -1], [3, -4, 1]])))


if __name__ == "__main__":
    absltest.main()
