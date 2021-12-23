import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class Air3d(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 evader_speed=5.,
                 pursuer_speed=5.,
                 evader_max_turn_rate=1.,
                 pursuer_max_turn_rate=1.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.evader_speed = evader_speed
        self.pursuer_speed = pursuer_speed
        if control_space is None:
            control_space = sets.Box(jnp.array([-evader_max_turn_rate]), jnp.array([evader_max_turn_rate]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([-pursuer_max_turn_rate]), jnp.array([pursuer_max_turn_rate]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v_a, v_b = self.evader_speed, self.pursuer_speed
        return jnp.array([-v_a + v_b * jnp.cos(psi), v_b * jnp.sin(psi), 0.])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array([
            [y],
            [-x],
            [-1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [1.],
        ])


DubinsCarCAvoid = Air3d
