"""Regression tests for CBF-QP actuator bounds without obstacle rows."""

from __future__ import annotations

import unittest

import numpy as np

from base_control.position_control.cbf_qp import CBFQP
from dynamic_env.main import LocalCBFQP


class CBFInputBoundsTests(unittest.TestCase):
    ROBOT_SPEC = {
        "model": "Unicycle2D",
        "v_max": 1.0,
        "w_max": 0.8,
    }
    STATE = np.zeros((3, 1))
    CONTROL_REF = {"u_ref": np.array([[25.95], [2.0]])}

    def _assert_projected(self, controller):
        control = controller.solve_control_problem(
            self.STATE,
            self.CONTROL_REF,
            obs_list=None,
        )

        self.assertIn(controller.status, {"optimal", "optimal_inaccurate"})
        np.testing.assert_allclose(
            np.asarray(control).reshape(-1),
            np.array([1.0, 0.8]),
            atol=1.0e-7,
        )

    def test_base_controller_enforces_bounds_without_obstacles(self):
        controller = CBFQP(object(), dict(self.ROBOT_SPEC), num_obs=1)
        self._assert_projected(controller)

    def test_runtime_controller_enforces_bounds_without_obstacles(self):
        controller = LocalCBFQP(object(), dict(self.ROBOT_SPEC), num_obs=1)
        self._assert_projected(controller)


if __name__ == "__main__":
    unittest.main()
