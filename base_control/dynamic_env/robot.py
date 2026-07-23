"""Dynamic-environment robot wrapper for the retained 2D robot models."""

from base_control.robots.robot import BaseRobot


class BaseRobotDyn(BaseRobot):
    """Compatibility wrapper for the project-specific dynamic environment."""

    def __init__(self, X0, robot_spec, dt, ax):
        super().__init__(X0, robot_spec, dt, ax)
        self.collision_parabola_patches = []
        self.collision_cone_patches = []
        self.rel_vel_patches = []

    def draw_collision_cone(self, X, obs_list, ax):
        """No-op: collision-cone bicycle models are not part of this runtime."""

    def draw_collision_parabola(self, X, obs_list, ax):
        """No-op: collision-parabola bicycle models are not part of this runtime."""
