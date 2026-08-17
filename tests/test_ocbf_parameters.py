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
    OCBF_TERMINAL_RELAX_CONTROLLER,
    OCBF_TERMINAL_RELAX_DEFAULTS,
    apply_ocbf_best_parameters,
    apply_ocbf_method_defaults,
    load_ocbf_best_parameters,
    load_shared_robot_parameters,
    merge_shared_robot_parameters,
    merge_ocbf_best_parameters,
)
from position_control.occlusion_cbf_qp import OcclusionCBFQP


DI_TUNED_KEYS = {
    "T_horizon",
    "max_active_occlusions",
    "vref_scenario_softmax_kappa",
    "k_p_occ_di",
    "k_d_occ_di",
}
UNI_TUNED_KEYS = {
    "T_horizon",
    "max_active_occlusions",
    "vref_scenario_softmax_kappa",
    "k_theta_occ_uni_p",
    "k_theta_occ_uni_d",
    "k_v_occ_uni_p",
    "k_v_occ_uni_d",
    "k_turn_boost_occ_uni",
    "turn_boost_angle_occ_uni",
}


class OCBFParameterTests(unittest.TestCase):
    def test_double_integrator_profile_matches_tuning_result(self):
        parameters = load_ocbf_best_parameters("di")
        backup = parameters["backup_cbf"]

        self.assertEqual(parameters["model"], "DoubleIntegrator2D")
        self.assertEqual(
            {key: value for key, value in backup.items() if key not in DI_TUNED_KEYS},
            {
                "dt_backup": 0.05,
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
        self.assertEqual(
            {key: backup[key] for key in DI_TUNED_KEYS},
            {
                "T_horizon": 0.25,
                "max_active_occlusions": 15,
                "vref_scenario_softmax_kappa": 10,
                "k_p_occ_di": 5.724752905909391,
                "k_d_occ_di": 0.44174267659711325,
            },
        )
        self.assertEqual(
            parameters["shared_robot_spec"],
            {"fov_angle": 360.0},
        )
        self.assertEqual(parameters["robot_spec"]["occ_kappa"], 10.0)
        self.assertTrue(
            parameters["robot_spec"]["enable_visible_hocbf_in_occ"]
        )

    def test_unicycle_profile_combines_tuned_and_shared_values(self):
        parameters = load_ocbf_best_parameters("Unicycle2D")
        backup = parameters["backup_cbf"]

        self.assertEqual(parameters["model"], "Unicycle2D")
        self.assertEqual(
            {key: value for key, value in backup.items() if key not in UNI_TUNED_KEYS},
            {
                "dt_backup": 0.05,
                "rho_T": 0.0,
                "vref_tracking_mode_occ_uni": "gated",
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
        self.assertEqual(
            {key: backup[key] for key in UNI_TUNED_KEYS},
            {
                "T_horizon": 3.0,
                "max_active_occlusions": 8,
                "vref_scenario_softmax_kappa": 1,
                "k_theta_occ_uni_p": 1.8927540185374787,
                "k_theta_occ_uni_d": 0.375,
                "k_v_occ_uni_p": 2.1500000000000004,
                "k_v_occ_uni_d": 0.12000000000000001,
                "k_turn_boost_occ_uni": 0.8,
                "turn_boost_angle_occ_uni": 0.2004007382154965,
            },
        )
        self.assertEqual(
            parameters["shared_robot_spec"],
            {"fov_angle": 360.0, "v_min": 0.0},
        )

    def test_dynamic_unicycle_remains_untuned(self):
        self.assertIsNone(load_ocbf_best_parameters("du"))
        backup, robot = merge_ocbf_best_parameters(
            "DynamicUnicycle2D",
            backup_defaults={"T_horizon": 1.0},
            robot_defaults={"w_max": 0.8},
        )
        self.assertEqual(backup, {"T_horizon": 1.0})
        self.assertEqual(robot, {"w_max": 0.8})
        self.assertIsNone(load_shared_robot_parameters("du"))

    def test_shared_profile_merge_preserves_explicit_overrides(self):
        self.assertEqual(
            load_shared_robot_parameters("di"),
            {"fov_angle": 360.0},
        )
        self.assertEqual(
            load_shared_robot_parameters("uni"),
            {"fov_angle": 360.0, "v_min": 0.0},
        )

        robot = merge_shared_robot_parameters(
            "uni",
            robot_defaults={
                "fov_angle": 70.0,
                "v_min": 0.0,
                "radius": 0.25,
            },
            robot_overrides={"v_min": -0.25},
        )
        self.assertEqual(
            robot,
            {
                "fov_angle": 360.0,
                "v_min": -0.25,
                "radius": 0.25,
            },
        )

    def test_explicit_overrides_win_over_yaml(self):
        tuned = load_ocbf_best_parameters("di")["backup_cbf"]
        backup, robot = merge_ocbf_best_parameters(
            "di",
            backup_defaults={"T_horizon": 9.0, "alpha": 1.5},
            robot_defaults={"occ_kappa": 2.0, "radius": 0.25},
            backup_overrides={"T_horizon": 0.75},
            robot_overrides={"occ_kappa": 4.0},
        )

        self.assertEqual(backup["T_horizon"], 0.75)
        self.assertEqual(backup["alpha"], 1.5)
        self.assertEqual(
            backup["max_active_occlusions"],
            tuned["max_active_occlusions"],
        )
        self.assertEqual(robot["occ_kappa"], 4.0)
        self.assertEqual(robot["radius"], 0.25)

    def test_loaded_parameters_are_independent_copies(self):
        expected = load_ocbf_best_parameters("di")["backup_cbf"]["T_horizon"]
        first = load_ocbf_best_parameters("di")
        first["backup_cbf"]["T_horizon"] = 99.0

        second = load_ocbf_best_parameters("di")
        self.assertEqual(second["backup_cbf"]["T_horizon"], expected)

    def test_controller_level_fallback_preserves_existing_values(self):
        tuned = load_ocbf_best_parameters("di")["backup_cbf"]
        robot_spec = {
            "model": "DoubleIntegrator2D",
            "occ_kappa": 3.0,
            "backup_cbf": {"T_horizon": 0.5},
        }

        apply_ocbf_best_parameters(robot_spec)

        self.assertEqual(robot_spec["occ_kappa"], 3.0)
        self.assertEqual(robot_spec["backup_cbf"]["T_horizon"], 0.5)
        self.assertEqual(
            robot_spec["backup_cbf"]["max_active_occlusions"],
            tuned["max_active_occlusions"],
        )

    def test_crowd_runtime_uses_full_yaml_only_for_ocbf(self):
        di_profile = load_ocbf_best_parameters("di")["backup_cbf"]
        uni_profile = load_ocbf_best_parameters("uni")["backup_cbf"]
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
        self.assertEqual(di_backup["T_horizon"], di_profile["T_horizon"])
        self.assertEqual(
            di_backup["max_active_occlusions"],
            di_profile["max_active_occlusions"],
        )
        self.assertTrue(
            di_runtime["robot_spec"]["enable_visible_hocbf_in_occ"]
        )

        uni_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "occlusion_cbf_qp"},
            model_key="uni",
            **common,
        )
        uni_backup = uni_runtime["robot_spec"]["backup_cbf"]
        self.assertEqual(uni_backup["T_horizon"], uni_profile["T_horizon"])
        self.assertEqual(
            uni_backup["max_active_occlusions"],
            uni_profile["max_active_occlusions"],
        )
        self.assertEqual(uni_runtime["robot_spec"]["v_min"], 0.0)

        cbf_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "cbf_qp"},
            model_key="di",
            **common,
        )
        cbf_backup = cbf_runtime["robot_spec"]["backup_cbf"]
        self.assertEqual(cbf_backup["T_horizon"], 0.5)
        self.assertNotIn("k_p_occ_di", cbf_backup)

    def test_terminal_relax_runtime_changes_only_terminal_slack_settings(self):
        common = {
            "rand_obs": False,
            "n_rand": 0,
            "known_obs_override": np.empty((0, 8), dtype=float),
            "obs_meta_override": [],
            "scenario_diag_override": {},
            "occ_enable_visible_hocbf": None,
        }
        hard_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "occlusion_cbf_qp"},
            model_key="di",
            **common,
        )
        relaxed_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": OCBF_TERMINAL_RELAX_CONTROLLER},
            model_key="di",
            **common,
        )

        hard = dict(hard_runtime["robot_spec"]["backup_cbf"])
        relaxed = dict(relaxed_runtime["robot_spec"]["backup_cbf"])
        ignored = {
            "terminal_slack_weight",
            "terminal_slack_max",
            "obs_hocbf_slack_max",
            "occ_rollout_slack_max",
        }
        self.assertEqual(
            {key: value for key, value in hard.items() if key not in ignored},
            {
                key: value
                for key, value in relaxed.items()
                if key not in ignored
            },
        )
        self.assertEqual(hard["terminal_slack_weight"], 0.0)
        self.assertGreater(relaxed["terminal_slack_weight"], 0.0)
        self.assertGreater(relaxed["terminal_slack_max"], 0.0)
        self.assertEqual(relaxed["obs_hocbf_slack_max"], 0.0)
        self.assertEqual(relaxed["occ_rollout_slack_max"], 0.0)
        self.assertEqual(
            relaxed["terminal_slack_weight"],
            OCBF_TERMINAL_RELAX_DEFAULTS["terminal_slack_weight"],
        )
        self.assertEqual(
            relaxed["terminal_slack_max"],
            OCBF_TERMINAL_RELAX_DEFAULTS["terminal_slack_max"],
        )

    def test_terminal_relax_is_di_only_and_forces_nonterminal_caps_hard(self):
        overrides = apply_ocbf_method_defaults(
            OCBF_TERMINAL_RELAX_CONTROLLER,
            "di",
            {
                "terminal_slack_weight": 20.0,
                "terminal_slack_max": 2.0,
                "obs_hocbf_slack_max": 3.0,
                "occ_rollout_slack_max": 4.0,
            },
        )
        self.assertEqual(overrides["terminal_slack_weight"], 20.0)
        self.assertEqual(overrides["terminal_slack_max"], 2.0)
        self.assertEqual(overrides["obs_hocbf_slack_max"], 0.0)
        self.assertEqual(overrides["occ_rollout_slack_max"], 0.0)

        with self.assertRaisesRegex(ValueError, "only.*DoubleIntegrator2D"):
            apply_ocbf_method_defaults(
                OCBF_TERMINAL_RELAX_CONTROLLER,
                "uni",
            )

    def test_terminal_relax_slack_mask_only_opens_terminal_rows(self):
        controller = object.__new__(OcclusionCBFQP)
        controller._terminal_slack_enabled = True
        controller.terminal_slack_max = 0.5
        controller.obs_hocbf_slack_max = 0.0
        controller.occ_rollout_slack_max = 0.0

        bounds = controller._build_terminal_slack_ub(
            [
                {"kind": "obs_hocbf"},
                {"kind": "occ"},
                {"kind": "terminal"},
                {"kind": "corridor"},
            ]
        )
        np.testing.assert_allclose(bounds[:, 0], [0.0, 0.0, 0.5, 0.0])

    def test_crowd_runtime_shares_only_canonical_model_values(self):
        common = {
            "rand_obs": False,
            "n_rand": 0,
            "known_obs_override": np.empty((0, 8), dtype=float),
            "obs_meta_override": [],
            "scenario_diag_override": {},
            "occ_enable_visible_hocbf": None,
        }
        comparison_controllers = (
            "cbf_qp",
            "oa_mpc",
            "single_risk_mpc",
            "control_tree_mpc",
            "oacp_mpc",
        )

        for controller_name in comparison_controllers:
            with self.subTest(controller=controller_name, model="di"):
                runtime = test_crowd_narrow._prepare_crowd_runtime(
                    controller_type={"pos": controller_name},
                    model_key="di",
                    **common,
                )
                robot_spec = runtime["robot_spec"]
                self.assertEqual(robot_spec["fov_angle"], 360.0)
                self.assertNotIn("occ_kappa", robot_spec)
                self.assertNotIn(
                    "k_p_occ_di",
                    robot_spec["backup_cbf"],
                )

            with self.subTest(controller=controller_name, model="uni"):
                runtime = test_crowd_narrow._prepare_crowd_runtime(
                    controller_type={"pos": controller_name},
                    model_key="uni",
                    **common,
                )
                robot_spec = runtime["robot_spec"]
                self.assertEqual(robot_spec["fov_angle"], 360.0)
                self.assertEqual(robot_spec["v_min"], 0.0)
                self.assertNotIn("occ_kappa", robot_spec)
                self.assertNotIn(
                    "k_theta_occ_uni_p",
                    robot_spec["backup_cbf"],
                )

        for controller_name, config_name in (
            ("single_risk_mpc", "single_risk_mpc"),
            ("control_tree_mpc", "control_tree_mpc"),
            ("oacp_mpc", "oacp_mpc"),
        ):
            for model_key in ("di", "uni"):
                with self.subTest(
                    controller=controller_name,
                    model=model_key,
                    parameter="max_active_occlusions",
                ):
                    runtime = test_crowd_narrow._prepare_crowd_runtime(
                        controller_type={"pos": controller_name},
                        model_key=model_key,
                        **common,
                    )
                    self.assertEqual(
                        runtime["robot_spec"][config_name][
                            "max_active_occlusions"
                        ],
                        2,
                    )

        control_tree_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "control_tree_mpc"},
            model_key="uni",
            **common,
        )
        self.assertFalse(
            control_tree_runtime["robot_spec"]["control_tree_mpc"][
                "forward_only"
            ]
        )
        self.assertEqual(
            control_tree_runtime["robot_spec"]["control_tree_mpc"][
                "v_plan_min"
            ],
            0.0,
        )

        forward_only_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "control_tree_mpc"},
            model_key="uni",
            robot_spec_overrides={"_uni_forward_only": True},
            **common,
        )
        self.assertEqual(forward_only_runtime["robot_spec"]["v_min"], 0.0)
        self.assertTrue(
            forward_only_runtime["robot_spec"]["control_tree_mpc"][
                "forward_only"
            ]
        )
        self.assertEqual(
            forward_only_runtime["robot_spec"]["control_tree_mpc"][
                "v_plan_min"
            ],
            0.0,
        )

        reverse_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "control_tree_mpc"},
            model_key="uni",
            robot_spec_overrides={"_uni_allow_reverse": True},
            **common,
        )
        self.assertEqual(reverse_runtime["robot_spec"]["v_min"], -1.0)
        self.assertEqual(
            reverse_runtime["robot_spec"]["control_tree_mpc"][
                "v_plan_min"
            ],
            -1.0,
        )

        explicit_runtime = test_crowd_narrow._prepare_crowd_runtime(
            controller_type={"pos": "cbf_qp"},
            model_key="uni",
            robot_spec_overrides={"v_min": -0.25},
            **common,
        )
        self.assertEqual(explicit_runtime["robot_spec"]["v_min"], -0.25)

    def test_campus_default_controller_uses_yaml(self):
        tuned = load_ocbf_best_parameters("di")
        with (
            mock.patch.object(
                test_campus,
                "_build_random_pedestrians",
                return_value=(np.empty((0, 8), dtype=float), [], {}),
            ),
            mock.patch.object(
                test_campus.crowd_narrow,
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
        self.assertEqual(
            forwarded["backup_cbf_overrides"]["T_horizon"],
            tuned["backup_cbf"]["T_horizon"],
        )
        self.assertEqual(
            forwarded["backup_cbf_overrides"]["max_active_occlusions"],
            tuned["backup_cbf"]["max_active_occlusions"],
        )
        self.assertEqual(
            forwarded["robot_spec_overrides"]["occ_kappa"],
            tuned["robot_spec"]["occ_kappa"],
        )

    def test_crosswalk_uses_yaml_and_explicit_override(self):
        di_profile = load_ocbf_best_parameters("di")["backup_cbf"]
        uni_profile = load_ocbf_best_parameters("uni")["backup_cbf"]
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
                test_crosswalk.run_crosswalk_scenario(
                    controller_type={"pos": "occlusion_cbf_qp"},
                    model_key="uni",
                    enable_plot=False,
                )
            with self.assertRaisesRegex(RuntimeError, "configuration captured"):
                test_crosswalk.run_crosswalk_scenario(
                    controller_type={"pos": "occlusion_cbf_qp"},
                    model_key="di",
                    enable_plot=False,
                    occ_T_horizon=0.75,
                )

            with self.assertRaisesRegex(RuntimeError, "configuration captured"):
                test_crosswalk.run_crosswalk_scenario(
                    controller_type={"pos": OCBF_TERMINAL_RELAX_CONTROLLER},
                    model_key="di",
                    enable_plot=False,
                )

        uni_spec, di_spec, relaxed_di_spec = captured_specs
        self.assertEqual(
            uni_spec["backup_cbf"]["T_horizon"],
            uni_profile["T_horizon"],
        )
        self.assertEqual(
            uni_spec["backup_cbf"]["max_active_occlusions"],
            uni_profile["max_active_occlusions"],
        )
        self.assertEqual(uni_spec["v_min"], 0.0)
        self.assertEqual(di_spec["backup_cbf"]["T_horizon"], 0.75)
        self.assertEqual(
            di_spec["backup_cbf"]["max_active_occlusions"],
            di_profile["max_active_occlusions"],
        )
        self.assertEqual(
            relaxed_di_spec["backup_cbf"]["terminal_slack_weight"],
            OCBF_TERMINAL_RELAX_DEFAULTS["terminal_slack_weight"],
        )
        self.assertEqual(
            relaxed_di_spec["backup_cbf"]["terminal_slack_max"],
            OCBF_TERMINAL_RELAX_DEFAULTS["terminal_slack_max"],
        )

    def test_visualization_defaults_to_yaml(self):
        tuned = load_ocbf_best_parameters("di")
        parser = test_vis_ocbf.build_arg_parser()
        args = parser.parse_args([])
        robot_spec = test_vis_ocbf._make_robot_spec(args, sensing_range=8.0)

        self.assertEqual(
            robot_spec["backup_cbf"]["T_horizon"],
            tuned["backup_cbf"]["T_horizon"],
        )
        self.assertEqual(
            robot_spec["backup_cbf"]["vref_scenario_softmax_kappa"],
            tuned["backup_cbf"]["vref_scenario_softmax_kappa"],
        )
        self.assertEqual(robot_spec["occ_kappa"], tuned["robot_spec"]["occ_kappa"])

        override_args = parser.parse_args(["--T-horizon", "0.75"])
        overridden = test_vis_ocbf._make_robot_spec(
            override_args,
            sensing_range=8.0,
        )
        self.assertEqual(overridden["backup_cbf"]["T_horizon"], 0.75)


if __name__ == "__main__":
    unittest.main()
