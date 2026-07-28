"""Regression tests for the committed Occlusion-CBF parameter profiles."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from examples import test_campus
from examples import test_crowd_narrow
from examples import test_crosswalk
from examples import test_vis_ocbf
from position_control.ocbf.defaults import (
    apply_ocbf_best_parameters,
    load_ocbf_best_parameters,
    merge_ocbf_best_parameters,
)


class OCBFParameterTests(unittest.TestCase):
    def test_double_integrator_profile_matches_tuning_result(self):
        parameters = load_ocbf_best_parameters("di")

        self.assertEqual(parameters["model"], "DoubleIntegrator2D")
        self.assertEqual(
            parameters["backup_cbf"],
            {
                "T_horizon": 0.25,
                "dt_backup": 0.05,
                "max_active_occlusions": 10,
                "vref_scenario_softmax_kappa": 20.0,
                "k_p_occ_di": 2.7134037493952947,
                "k_d_occ_di": 1.0553275095919081,
                "rho_T": "auto",
                "vref_scenario_weight_mode": "barrier_unexpand",
                "qp_failure_fallback_mode": "state_safe",
                "vref_front_mode_occ": "los",
                "vref_mode_occ": "strict",
                "occ_selection_mode": "h_tilde",
                "occ_rollout_mode": "common",
                "terminal_mode": "all",
                "terminal_residual_mode": "off",
                "terminal_slack_weight": 0.0,
            },
        )
        self.assertEqual(parameters["robot_spec"]["occ_kappa"], 10.0)
        self.assertEqual(parameters["robot_spec"]["fov_angle"], 360.0)
        self.assertTrue(
            parameters["robot_spec"]["enable_visible_hocbf_in_occ"]
        )

    def test_unicycle_profile_matches_tuning_result(self):
        parameters = load_ocbf_best_parameters("Unicycle2D")
        backup = parameters["backup_cbf"]

        self.assertEqual(parameters["model"], "Unicycle2D")
        self.assertEqual(backup["T_horizon"], 2.0)
        self.assertEqual(backup["max_active_occlusions"], 5)
        self.assertEqual(backup["vref_scenario_softmax_kappa"], 5.0)
        self.assertEqual(backup["k_theta_occ_uni_p"], 1.0783765065946995)
        self.assertEqual(backup["k_theta_occ_uni_d"], 0.15000000000000002)
        self.assertEqual(backup["k_v_occ_uni_p"], 1.4500000000000002)
        self.assertEqual(backup["k_v_occ_uni_d"], 0.24)
        self.assertEqual(backup["k_turn_boost_occ_uni"], 0.9)
        self.assertEqual(
            backup["turn_boost_angle_occ_uni"],
            0.34007262580098474,
        )
        self.assertEqual(backup["vref_scenario_weight_mode"], "barrier_unexpand")
        self.assertEqual(parameters["robot_spec"]["v_min"], 0.0)

    def test_dynamic_unicycle_remains_untuned(self):
        self.assertIsNone(load_ocbf_best_parameters("du"))
        backup, robot = merge_ocbf_best_parameters(
            "DynamicUnicycle2D",
            backup_defaults={"T_horizon": 1.0},
            robot_defaults={"w_max": 0.8},
        )
        self.assertEqual(backup, {"T_horizon": 1.0})
        self.assertEqual(robot, {"w_max": 0.8})

    def test_explicit_overrides_win_over_yaml(self):
        backup, robot = merge_ocbf_best_parameters(
            "di",
            backup_defaults={"T_horizon": 9.0, "alpha": 1.5},
            robot_defaults={"occ_kappa": 2.0, "radius": 0.25},
            backup_overrides={"T_horizon": 0.75},
            robot_overrides={"occ_kappa": 4.0},
        )

        self.assertEqual(backup["T_horizon"], 0.75)
        self.assertEqual(backup["alpha"], 1.5)
        self.assertEqual(backup["max_active_occlusions"], 10)
        self.assertEqual(robot["occ_kappa"], 4.0)
        self.assertEqual(robot["radius"], 0.25)

    def test_loaded_parameters_are_independent_copies(self):
        first = load_ocbf_best_parameters("di")
        first["backup_cbf"]["T_horizon"] = 99.0

        second = load_ocbf_best_parameters("di")
        self.assertEqual(second["backup_cbf"]["T_horizon"], 0.25)

    def test_controller_level_fallback_preserves_existing_values(self):
        robot_spec = {
            "model": "DoubleIntegrator2D",
            "occ_kappa": 3.0,
            "backup_cbf": {"T_horizon": 0.5},
        }

        apply_ocbf_best_parameters(robot_spec)

        self.assertEqual(robot_spec["occ_kappa"], 3.0)
        self.assertEqual(robot_spec["backup_cbf"]["T_horizon"], 0.5)
        self.assertEqual(robot_spec["backup_cbf"]["max_active_occlusions"], 10)

    def test_crowd_runtime_uses_yaml_for_ocbf_only(self):
        common = {
            "rand_obs": False,
            "n_rand": 0,
            "known_obs_override": np.empty((0, 8), dtype=float),
            "obs_meta_override": [],
            "scenario_diag_override": {},
            "occ_enable_visible_hocbf": None,
        }

        di_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "occlusion_cbf_qp"},
            model_key="di",
            **common,
        )
        di_backup = di_runtime["robot_spec"]["backup_cbf"]
        self.assertEqual(di_backup["T_horizon"], 0.25)
        self.assertEqual(di_backup["max_active_occlusions"], 10)
        self.assertTrue(
            di_runtime["robot_spec"]["enable_visible_hocbf_in_occ"]
        )

        uni_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "occlusion_cbf_qp"},
            model_key="uni",
            **common,
        )
        uni_backup = uni_runtime["robot_spec"]["backup_cbf"]
        self.assertEqual(uni_backup["T_horizon"], 2.0)
        self.assertEqual(uni_backup["max_active_occlusions"], 5)
        self.assertEqual(uni_runtime["robot_spec"]["v_min"], 0.0)

        cbf_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "cbf_qp"},
            model_key="di",
            **common,
        )
        cbf_backup = cbf_runtime["robot_spec"]["backup_cbf"]
        self.assertEqual(cbf_backup["T_horizon"], 0.5)
        self.assertNotIn("k_p_occ_di", cbf_backup)

    def test_campus_default_controller_uses_yaml(self):
        with (
            mock.patch.object(
                test_campus,
                "_build_random_pedestrians",
                return_value=(np.empty((0, 8), dtype=float), [], {}),
            ),
            mock.patch.object(
                test_campus.crowd1,
                "run_crowd_scenario",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            forwarded = test_campus.run_campus_scenario(
                controller_type=None,
                model_key="di",
                show_animation=False,
                n_rand=0,
            )

        self.assertEqual(
            forwarded["controller_type"],
            {"pos": "occlusion_cbf_qp"},
        )
        self.assertEqual(forwarded["backup_cbf_overrides"]["T_horizon"], 0.25)
        self.assertEqual(
            forwarded["backup_cbf_overrides"]["max_active_occlusions"],
            10,
        )
        self.assertEqual(forwarded["robot_spec_overrides"]["occ_kappa"], 10.0)

    def test_crosswalk_uses_yaml_and_explicit_override(self):
        captured_specs = []

        def capture_controller(*args, **kwargs):
            captured_specs.append(args[1])
            raise RuntimeError("configuration captured")

        with mock.patch.object(
            test_crosswalk,
            "LocalTrackingControllerDyn_OCC",
            side_effect=capture_controller,
        ):
            with self.assertRaisesRegex(RuntimeError, "configuration captured"):
                test_crosswalk.crosswalk_scenario_v3(
                    controller_type={"pos": "occlusion_cbf_qp"},
                    model_key="uni",
                    enable_plot=False,
                )
            with self.assertRaisesRegex(RuntimeError, "configuration captured"):
                test_crosswalk.crosswalk_scenario_v3(
                    controller_type={"pos": "occlusion_cbf_qp"},
                    model_key="di",
                    enable_plot=False,
                    occ_T_horizon=0.75,
                )

        uni_spec, di_spec = captured_specs
        self.assertEqual(uni_spec["backup_cbf"]["T_horizon"], 2.0)
        self.assertEqual(uni_spec["backup_cbf"]["max_active_occlusions"], 5)
        self.assertEqual(uni_spec["v_min"], 0.0)
        self.assertEqual(di_spec["backup_cbf"]["T_horizon"], 0.75)
        self.assertEqual(di_spec["backup_cbf"]["max_active_occlusions"], 10)

    def test_visualization_defaults_to_yaml(self):
        parser = test_vis_ocbf.build_arg_parser()
        args = parser.parse_args([])
        robot_spec = test_vis_ocbf._make_robot_spec(args, sensing_range=8.0)

        self.assertEqual(robot_spec["backup_cbf"]["T_horizon"], 0.25)
        self.assertEqual(
            robot_spec["backup_cbf"]["vref_scenario_softmax_kappa"],
            20.0,
        )
        self.assertEqual(robot_spec["occ_kappa"], 10.0)

        override_args = parser.parse_args(["--T-horizon", "0.75"])
        overridden = test_vis_ocbf._make_robot_spec(
            override_args,
            sensing_range=8.0,
        )
        self.assertEqual(overridden["backup_cbf"]["T_horizon"], 0.75)


if __name__ == "__main__":
    unittest.main()
