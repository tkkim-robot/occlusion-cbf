"""Finite-difference checks for moving-circle explicit-time CBF terms."""

from __future__ import annotations

import unittest

import numpy as np

from base_control.robots.double_integrator2D import DoubleIntegrator2D
from base_control.robots.dynamic_unicycle2D import DynamicUnicycle2D
from base_control.robots.unicycle2D import Unicycle2D


def _obstacle_at_time(obstacle, time_s):
    shifted = np.asarray(obstacle, dtype=float).copy()
    shifted[:2] += shifted[3:5] * float(time_s)
    return shifted


class MovingCircleBarrierTests(unittest.TestCase):
    EPS = 1.0e-6
    ROBOT_RADIUS = 0.28
    OBSTACLE = np.array([0.4, -0.7, 0.35, 0.31, -0.22, -2.0, 2.0, 1.0])

    def _central_time_difference(self, model, state, output_index):
        plus = model.dynamic_agent_barrier(
            state,
            _obstacle_at_time(self.OBSTACLE, self.EPS),
            self.ROBOT_RADIUS,
        )
        minus = model.dynamic_agent_barrier(
            state,
            _obstacle_at_time(self.OBSTACLE, -self.EPS),
            self.ROBOT_RADIUS,
        )
        return (float(plus[output_index]) - float(minus[output_index])) / (
            2.0 * self.EPS
        )

    def test_double_integrator_h_dot_explicit_time_derivative(self):
        model = DoubleIntegrator2D(0.05, {})
        state = np.array([[1.3], [0.2], [0.45], [-0.18]])

        _, _, grad_h_dot, h_dot_t = model.dynamic_agent_barrier(
            state, self.OBSTACLE, self.ROBOT_RADIUS
        )
        numerical = self._central_time_difference(model, state, output_index=1)

        self.assertEqual(grad_h_dot.shape, (1, 4))
        self.assertAlmostEqual(float(h_dot_t), numerical, places=7)

    def test_dynamic_unicycle_h_dot_explicit_time_derivative(self):
        model = DynamicUnicycle2D(0.05, {})
        state = np.array([[1.3], [0.2], [0.63], [0.72]])

        _, _, grad_h_dot, h_dot_t = model.dynamic_agent_barrier(
            state, self.OBSTACLE, self.ROBOT_RADIUS
        )
        numerical = self._central_time_difference(model, state, output_index=1)

        self.assertEqual(grad_h_dot.shape, (1, 4))
        self.assertAlmostEqual(float(h_dot_t), numerical, places=7)

    def test_unicycle_barrier_explicit_time_derivative(self):
        model = Unicycle2D(0.05, {})
        state = np.array([[1.3], [0.2], [0.63]])

        _, grad_h, h_t = model.dynamic_agent_barrier(
            state, self.OBSTACLE, self.ROBOT_RADIUS
        )
        numerical = self._central_time_difference(model, state, output_index=0)

        self.assertEqual(grad_h.shape, (1, 3))
        self.assertAlmostEqual(float(h_t), numerical, places=7)


if __name__ == "__main__":
    unittest.main()
