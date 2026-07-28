"""Regression tests for CBF-QP actuator bounds without obstacle rows."""

from __future__ import annotations

import unittest

import numpy as np

from base_control.position_control.cbf_qp import CBFQP
from dynamic_env.main import LocalCBFQP
from position_control.control_tree_mpc import ControlTreeMPC
from position_control.oa_mpc import OAMPC
from position_control.oacp_mpc import OACPMPC
from position_control.single_risk_mpc import SingleRiskMPC


class CBFInputBoundsTests(unittest.TestCase):
    ROBOT_SPEC = {
        "model": "Unicycle2D",
        "v_max": 1.0,
        "w_max": 0.8,
    }
    STATE = np.zeros((3, 1))
    CONTROL_REF = {"u_ref": np.array([[25.95], [2.0]])}
    REVERSE_CONTROL_REF = {"u_ref": np.array([[-25.95], [-2.0]])}

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

    def _assert_forward_only(self, controller):
        control = controller.solve_control_problem(
            self.STATE,
            self.REVERSE_CONTROL_REF,
            obs_list=None,
        )

        self.assertIn(controller.status, {"optimal", "optimal_inaccurate"})
        np.testing.assert_allclose(
            np.asarray(control).reshape(-1),
            np.array([0.0, -0.8]),
            atol=1.0e-7,
        )

    def _assert_reverse_capable(self, controller):
        control = controller.solve_control_problem(
            self.STATE,
            self.REVERSE_CONTROL_REF,
            obs_list=None,
        )

        self.assertIn(controller.status, {"optimal", "optimal_inaccurate"})
        np.testing.assert_allclose(
            np.asarray(control).reshape(-1),
            np.array([-1.0, -0.8]),
            atol=1.0e-7,
        )

    def test_base_controller_defaults_to_forward_only(self):
        controller = CBFQP(object(), dict(self.ROBOT_SPEC), num_obs=1)
        self._assert_forward_only(controller)

    def test_runtime_controller_defaults_to_forward_only(self):
        controller = LocalCBFQP(object(), dict(self.ROBOT_SPEC), num_obs=1)
        self._assert_forward_only(controller)

    def test_corridor_runtime_controller_defaults_to_forward_only(self):
        robot_spec = dict(
            self.ROBOT_SPEC,
            position_corridor={
                "enabled": True,
                "x_min": -10.0,
                "x_max": 10.0,
                "buffer": 0.0,
            },
        )
        controller = LocalCBFQP(object(), robot_spec, num_obs=1)
        self._assert_forward_only(controller)

    def test_base_controller_honors_unicycle_v_min(self):
        robot_spec = dict(self.ROBOT_SPEC, v_min=0.0)
        controller = CBFQP(object(), robot_spec, num_obs=1)
        self._assert_forward_only(controller)

    def test_runtime_controller_honors_unicycle_v_min(self):
        robot_spec = dict(self.ROBOT_SPEC, v_min=0.0)
        controller = LocalCBFQP(object(), robot_spec, num_obs=1)
        self._assert_forward_only(controller)

    def test_base_controller_honors_explicit_reverse_bound(self):
        robot_spec = dict(self.ROBOT_SPEC, v_min=-1.0)
        controller = CBFQP(object(), robot_spec, num_obs=1)
        self._assert_reverse_capable(controller)

    def test_runtime_controller_honors_explicit_reverse_bound(self):
        robot_spec = dict(self.ROBOT_SPEC, v_min=-1.0)
        controller = LocalCBFQP(object(), robot_spec, num_obs=1)
        self._assert_reverse_capable(controller)

    def test_control_tree_defaults_to_forward_only_input_bound(self):
        controller = object.__new__(ControlTreeMPC)
        controller.model = "Unicycle2D"
        controller.robot_spec = dict(self.ROBOT_SPEC)
        controller.forward_only = False
        controller.v_plan_min = 0.0

        lower, upper = controller._input_bounds()

        np.testing.assert_allclose(lower, np.array([0.0, -0.8]))
        np.testing.assert_allclose(upper, np.array([1.0, 0.8]))

    def test_control_tree_honors_explicit_reverse_bound(self):
        controller = object.__new__(ControlTreeMPC)
        controller.model = "Unicycle2D"
        controller.robot_spec = dict(self.ROBOT_SPEC, v_min=-1.0)
        controller.forward_only = False
        controller.v_plan_min = -1.0

        lower, upper = controller._input_bounds()

        np.testing.assert_allclose(lower, np.array([-1.0, -0.8]))
        np.testing.assert_allclose(upper, np.array([1.0, 0.8]))

    def test_mpc_baselines_default_to_forward_only_input_bound(self):
        for controller_cls in (OAMPC, SingleRiskMPC, OACPMPC):
            with self.subTest(controller=controller_cls.__name__):
                controller = object.__new__(controller_cls)
                controller.model = "Unicycle2D"
                controller.robot_spec = dict(self.ROBOT_SPEC)
                controller.forward_only = False

                lower, upper = controller._input_bounds()

                np.testing.assert_allclose(lower, np.array([0.0, -0.8]))
                np.testing.assert_allclose(upper, np.array([1.0, 0.8]))

    def test_mpc_baselines_honor_explicit_reverse_bound(self):
        for controller_cls in (OAMPC, SingleRiskMPC, OACPMPC):
            with self.subTest(controller=controller_cls.__name__):
                controller = object.__new__(controller_cls)
                controller.model = "Unicycle2D"
                controller.robot_spec = dict(self.ROBOT_SPEC, v_min=-1.0)
                controller.forward_only = False

                lower, upper = controller._input_bounds()

                np.testing.assert_allclose(lower, np.array([-1.0, -0.8]))
                np.testing.assert_allclose(upper, np.array([1.0, 0.8]))


if __name__ == "__main__":
    unittest.main()
