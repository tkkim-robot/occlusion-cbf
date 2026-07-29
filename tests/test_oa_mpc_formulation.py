"""Regression tests for the paper-aligned OA-MPC formulation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from examples.test_crowd_narrow import _prepare_crowd_runtime
from position_control import oa_mpc
from position_control.oa_mpc import OAMPC


class OAMPCFormulationTests(unittest.TestCase):
    def _projection_controller(self):
        controller = object.__new__(OAMPC)
        controller.N = 2
        controller.dt = 0.1
        controller.model = "DoubleIntegrator2D"
        controller.robot_spec = {
            "dynamic_obs_types": [1],
            "v_max": 1.0,
        }
        controller.dsafe = 0.5
        controller.robot_radius = 0.25
        controller.visible_reach_mode = "worst_case"
        controller.v_visible_max_default = 1.0
        controller.hidden_agent_radius = 0.0
        controller.v_hidden_max_default = 1.0
        controller.prune_far_constraints = False
        controller.prune_far_margin = 0.0
        controller.max_constraints_per_step = 32
        controller.use_complementarity = True
        controller.complementarity_for_circles = False
        controller.complementarity_for_visible_dynamic = True
        return controller

    def test_visible_dynamic_and_static_circle_rows_are_distinguished(self):
        controller = self._projection_controller()
        x_bar = np.zeros((4, controller.N + 1), dtype=float)
        visible_dynamic = [
            np.array([2.0, 0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])
        ]
        static_pointcloud = [(np.array([3.0, 0.0]), 0.2)]

        targets = controller._build_projection_targets(
            x_bar=x_bar,
            visible_obs=visible_dynamic,
            occ_scenarios=[],
            static_pc_circles=static_pointcloud,
        )

        rows = targets[1]
        by_source = {row["source"]: row for row in rows}
        self.assertFalse(by_source["static_pointcloud"]["use_comp"])
        self.assertTrue(by_source["visible_dynamic"]["use_comp"])

    @unittest.skipUnless(oa_mpc._CASADI_AVAILABLE, "CasADi is required")
    def test_complementarity_motion_uses_full_state_change(self):
        # Position does not change, but velocity does. A position-only test
        # would incorrectly classify this Double Integrator transition as stop.
        di_states = oa_mpc.ca.DM(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        di_motion = OAMPC._complementarity_motion_sq(di_states, 1)
        self.assertAlmostEqual(float(di_motion), 1.0)

        # Rotation in place is likewise a state change under the paper's
        # z_k == z_{k-1} stopping definition.
        uni_states = oa_mpc.ca.DM(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.5],
            ]
        )
        uni_motion = OAMPC._complementarity_motion_sq(uni_states, 1)
        self.assertAlmostEqual(float(uni_motion), 0.25)

    def test_paper_mode_enables_dynamic_complementarity_only_by_default(self):
        robot = SimpleNamespace(dt=0.05)
        robot_spec = {
            "model": "DoubleIntegrator2D",
            "v_max": 1.0,
            "a_max": 1.0,
            "radius": 0.25,
            "sensing_range": 10.0,
            "oa_mpc": {},
        }

        controller = OAMPC(robot, robot_spec)

        self.assertTrue(controller.use_complementarity)
        self.assertTrue(controller.complementarity_for_visible_dynamic)
        self.assertFalse(controller.complementarity_for_circles)

    def test_crowd_keeps_dynamic_occluders_and_uses_exact_di_stop(self):
        runtime = _prepare_crowd_runtime(
            controller_type={"pos": "oa_mpc"},
            model_key="di",
            case_idx=1,
            rand_obs=False,
            n_rand=0,
        )
        robot_spec = runtime["robot_spec"]
        oa_cfg = robot_spec["oa_mpc"]

        self.assertTrue(oa_cfg["dynamic_occluders"])
        self.assertEqual(robot_spec["occlusion_types"], [0, 1])
        self.assertEqual(oa_cfg["visible_reach_mode"], "worst_case")
        self.assertEqual(oa_cfg["di_terminal_stop_mode"], "exact")


if __name__ == "__main__":
    unittest.main()
