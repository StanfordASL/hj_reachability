from absl.testing import absltest
import numpy as np

import hj_reachability as hj


class SolverTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)
        solver_settings = hj.SolverSettings.with_accuracy("low")
        dynamics = hj.systems.Air3d()
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-6., -10., 0.]),
                                                                                   np.array([20., 10., 2 * np.pi])),
                                                                       (11, 10, 10),
                                                                       periodic_dims=2)
        self.problem_definition = {
            "solver_settings": solver_settings,
            "dynamics": dynamics,
            "grid": grid,
        }

    def test_step(self):
        values = np.linalg.norm(self.problem_definition["grid"].states[..., :2], axis=-1) - 5
        target_values = hj.step(**self.problem_definition, time=0., values=values, target_time=-0.1, progress_bar=False)
        self.assertEqual(target_values.shape, values.shape)
        np.testing.assert_allclose(
            target_values,
            hj.step(**self.problem_definition, time=0., values=values, target_time=-0.1, progress_bar=True))

    def test_solve(self):
        times = np.linspace(0, -0.1, 3)
        initial_values = np.linalg.norm(self.problem_definition["grid"].states[..., :2], axis=-1) - 5
        all_values = hj.solve(**self.problem_definition, times=times, initial_values=initial_values, progress_bar=False)
        self.assertEqual(all_values.shape, (len(times),) + initial_values.shape)
        np.testing.assert_allclose(all_values[0], initial_values)
        np.testing.assert_allclose(all_values[-1],
                                   hj.step(**self.problem_definition,
                                           time=0.,
                                           values=initial_values,
                                           target_time=-0.1,
                                           progress_bar=False),
                                   atol=1e-2)
        np.testing.assert_allclose(
            all_values,
            hj.solve(**self.problem_definition, times=times, initial_values=initial_values, progress_bar=True))
