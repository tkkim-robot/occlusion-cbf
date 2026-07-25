"""Regression tests for shared obstacle handling."""

from __future__ import annotations

import unittest

import numpy as np

from base_control.robots.unicycle2D import Unicycle2D
from base_control.tracking import LocalTrackingController


class _RobotStub:
    robot_radius = 0.5

    def __init__(self, position=(0.0, 0.0), yaw=0.0):
        self.X = np.array([[position[0]], [position[1]], [yaw]], dtype=float)

    def get_position(self):
        return self.X[:2, 0]

    def get_orientation(self):
        return float(self.X[2, 0])


class BaseControlHotfixTests(unittest.TestCase):
    def test_static_unicycle_barrier_accepts_row_and_column_obstacles(self):
        model = Unicycle2D(0.05, {})
        state = np.array([[1.0], [2.0], [0.3]])
        row = np.array([4.0, 6.0, 0.5, 0.0, 0.0, -2.0, 2.0, 0.0])

        h_row, grad_row = model.agent_barrier(state, row, robot_radius=0.25)
        h_col, grad_col = model.agent_barrier(
            state,
            row.reshape(-1, 1),
            robot_radius=0.25,
        )

        self.assertEqual(np.asarray(grad_row).shape, (1, 3))
        np.testing.assert_allclose(h_row, h_col)
        np.testing.assert_allclose(grad_row, grad_col)

    def test_eight_column_circle_collision_is_detected(self):
        controller = LocalTrackingController.__new__(LocalTrackingController)
        controller.robot = _RobotStub()
        controller.unknown_obs = np.empty((0, 8))
        controller.obs = np.array(
            [[0.2, 0.0, 0.5, 0.0, 0.0, -2.0, 2.0, 1.0]],
            dtype=float,
        )
        controller.robot_spec = {"model": "Unicycle2D"}

        self.assertTrue(controller.is_collide_unknown())

    def test_obstacle_fallback_respects_requested_count(self):
        controller = LocalTrackingController.__new__(LocalTrackingController)
        controller.robot = _RobotStub()
        controller.robot_spec = {"model": "Unicycle2D"}
        controller.obs = np.array(
            [
                [-float(index + 1), 0.0, 0.2, 0.0, 0.0, -2.0, 2.0, 1.0]
                for index in range(10)
            ],
            dtype=float,
        )

        selected = controller.get_nearest_unpassed_obs(
            detected_obs=np.empty((0, 8)),
            obs_num=7,
        )

        self.assertEqual(selected.shape[0], 7)

    def test_unicycle_obstacle_selection_defaults_to_full_circle(self):
        obstacles = np.array(
            [
                [1.0, 0.0, 0.2, 0.0, 0.0, -2.0, 2.0, 1.0],
                [-1.0, 0.0, 0.2, 0.0, 0.0, -2.0, 2.0, 1.0],
            ],
            dtype=float,
        )

        for model in ("Unicycle2D", "DynamicUnicycle2D"):
            with self.subTest(model=model):
                controller = LocalTrackingController.__new__(LocalTrackingController)
                controller.robot = _RobotStub(yaw=0.0)
                controller.robot_spec = {"model": model}
                controller.obs = obstacles.copy()

                selected = controller.get_nearest_unpassed_obs(
                    detected_obs=np.empty((0, 8)),
                    obs_num=2,
                )

                self.assertEqual(selected.shape[0], 2)
                np.testing.assert_allclose(
                    np.sort(selected[:, 0]),
                    np.array([-1.0, 1.0]),
                )

                forward_only = controller.get_nearest_unpassed_obs(
                    detected_obs=np.empty((0, 8)),
                    angle_unpassed=np.pi,
                    obs_num=2,
                )

                self.assertEqual(forward_only.shape[0], 1)
                self.assertEqual(forward_only[0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
