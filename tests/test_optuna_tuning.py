"""Regression tests for the OCBF Optuna tuning contract."""

from __future__ import annotations

import unittest

import numpy as np
import optuna
from optuna.pruners import NopPruner

from position_control.ocbf.defaults import load_ocbf_best_parameters

from tools.tune_ocbf_optuna import (
    ACTIVE_OCCLUSION_CHOICES,
    FIXED_OCBF_CONFIG,
    LatestStepMedianPruner,
    MODEL_DEFAULTS,
    SEARCH_SPACE,
    TERMINAL_RELAX_GRID,
    TrialCaseError,
    build_controller_overrides,
    build_study_pruner,
    compatibility_fingerprint,
    lexicographic_score,
    sample_hyperparameters,
    selected_profile_parameters,
    validate_case_rows,
)


class OptunaTuningTests(unittest.TestCase):
    def test_objective_is_strictly_lexicographic(self):
        faster_but_more_collisions = lexicographic_score(
            collision_count=1,
            success_count=99,
            evaluated_count=100,
            avg_compute_time_ms=0.0,
        )
        slower_but_fewer_collisions = lexicographic_score(
            collision_count=0,
            success_count=0,
            evaluated_count=100,
            avg_compute_time_ms=999999.0,
        )
        self.assertLess(
            faster_but_more_collisions,
            slower_but_fewer_collisions,
        )

        more_success = lexicographic_score(
            collision_count=49,
            success_count=51,
            evaluated_count=100,
            avg_compute_time_ms=999999.0,
        )
        less_success = lexicographic_score(
            collision_count=0,
            success_count=50,
            evaluated_count=100,
            avg_compute_time_ms=0.0,
        )
        self.assertLess(more_success, less_success)

        fewer_collisions = lexicographic_score(
            collision_count=0,
            success_count=50,
            evaluated_count=100,
            avg_compute_time_ms=999999.0,
        )
        more_collisions = lexicographic_score(
            collision_count=1,
            success_count=50,
            evaluated_count=100,
            avg_compute_time_ms=0.0,
        )
        self.assertLess(fewer_collisions, more_collisions)

        faster = lexicographic_score(
            collision_count=0,
            success_count=50,
            evaluated_count=100,
            avg_compute_time_ms=10.0,
        )
        slower = lexicographic_score(
            collision_count=0,
            success_count=50,
            evaluated_count=100,
            avg_compute_time_ms=20.0,
        )
        self.assertLess(faster, slower)

    def test_fixed_safety_choices_are_never_sampled(self):
        self.assertEqual(
            FIXED_OCBF_CONFIG["obstacle_selection_angle_rad"],
            2.0 * np.pi,
        )
        self.assertEqual(
            FIXED_OCBF_CONFIG["vref_scenario_weight_mode"],
            "barrier_unexpand",
        )
        self.assertNotIn("vref_scenario_weight_mode", SEARCH_SPACE["di"])
        self.assertNotIn("vref_scenario_weight_mode", SEARCH_SPACE["uni"])
        self.assertNotIn("barrier_kappa", SEARCH_SPACE["di"])
        self.assertNotIn("barrier_kappa", SEARCH_SPACE["uni"])
        self.assertNotIn("rho_T_mode", SEARCH_SPACE["di"])
        self.assertNotIn(
            "vref_tracking_mode_occ_uni",
            SEARCH_SPACE["uni"],
        )
        self.assertEqual(
            FIXED_OCBF_CONFIG["osqp_cache_update_guard"],
            "checked_rebuild_on_rejection",
        )
        self.assertEqual(
            FIXED_OCBF_CONFIG["qp_solution_certificate"],
            "current_cbf_slack_input_post_clip",
        )

    def test_model_profiles_use_requested_obstacle_counts(self):
        self.assertEqual(MODEL_DEFAULTS["di"]["n_rand"], 50)
        self.assertEqual(MODEL_DEFAULTS["uni"]["n_rand"], 30)

    def test_active_occlusion_choices_match_tuning_contract(self):
        expected = {
            "di": [3, 5, 10, 15, 20, 0],
            "uni": [3, 5, 8, 10, 0],
        }
        self.assertEqual(ACTIVE_OCCLUSION_CHOICES, expected)
        for model in ("di", "uni"):
            self.assertEqual(
                SEARCH_SPACE[model]["max_active_occlusions"]["choices"],
                expected[model],
            )
            for incumbent in MODEL_DEFAULTS[model]["incumbents"]:
                self.assertIn(
                    incumbent["max_active_occlusions"],
                    expected[model],
                )

    def test_search_spaces_match_controller_contract(self):
        self.assertEqual(
            SEARCH_SPACE["di"]["T_horizon"]["choices"],
            [0.10, 0.15, 0.20, 0.25, 0.35, 0.50],
        )
        self.assertEqual(
            SEARCH_SPACE["di"]["vref_scenario_softmax_kappa"]["choices"],
            [5, 10, 15, 20, 30, 40],
        )
        self.assertEqual(
            SEARCH_SPACE["di"]["k_p_occ_di"],
            {"type": "float", "low": 1.0, "high": 6.0, "log": True},
        )
        self.assertEqual(
            SEARCH_SPACE["di"]["k_d_occ_di"],
            {"type": "float", "low": 0.35, "high": 2.5, "log": True},
        )

        self.assertEqual(
            SEARCH_SPACE["uni"]["T_horizon"]["choices"],
            [1.0, 1.5, 2.0, 2.5, 3.0],
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["vref_scenario_softmax_kappa"]["choices"],
            [0, 1, 2, 5, 10, 20],
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["k_theta_occ_uni_p"],
            {"type": "float", "low": 0.25, "high": 2.5, "log": True},
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["k_theta_occ_uni_d"],
            {"type": "float", "low": 0.0, "high": 0.5, "step": 0.025},
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["k_v_occ_uni_p"],
            {"type": "float", "low": 0.8, "high": 2.5, "step": 0.05},
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["k_v_occ_uni_d"],
            {"type": "float", "low": 0.1, "high": 0.4, "step": 0.02},
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["k_turn_boost_occ_uni"],
            {"type": "float", "low": 0.3, "high": 1.8, "step": 0.1},
        )
        self.assertEqual(
            SEARCH_SPACE["uni"]["turn_boost_angle_occ_uni"],
            {
                "type": "float",
                "low": float(np.pi / 36.0),
                "high": float(np.pi / 3.0),
            },
        )

    def test_first_incumbents_are_derived_from_committed_profiles(self):
        for model in ("di", "uni"):
            with self.subTest(model=model):
                backup = load_ocbf_best_parameters(model)["backup_cbf"]
                expected = {
                    name: backup[name]
                    for name in SEARCH_SPACE[model]
                }
                self.assertEqual(selected_profile_parameters(model), expected)
                self.assertEqual(
                    MODEL_DEFAULTS[model]["incumbents"],
                    [expected],
                )

    def test_exception_rows_cannot_receive_an_objective(self):
        valid = [{"idx": 1, "exception": None}]
        validate_case_rows(valid)
        with self.assertRaises(TrialCaseError):
            validate_case_rows(
                [
                    {"idx": 1, "exception": None},
                    {"idx": 2, "exception": "solver setup failed"},
                ]
            )

    def test_compatibility_fingerprint_makes_trial_target_immutable(self):
        base = {
            "model": "di",
            "study_name": "paper_di",
            "n_rand": 50,
            "trials": 40,
            "timeout_s": 100,
        }
        fingerprint = compatibility_fingerprint(base)
        self.assertNotEqual(
            fingerprint,
            compatibility_fingerprint({**base, "trials": 80}),
        )
        self.assertEqual(
            fingerprint,
            compatibility_fingerprint({**base, "timeout_s": 200}),
        )
        self.assertNotEqual(
            fingerprint,
            compatibility_fingerprint({**base, "n_rand": 30}),
        )

    def test_incumbents_materialize_fixed_crowd_configuration(self):
        for model in ("di", "uni"):
            with self.subTest(model=model):
                params = MODEL_DEFAULTS[model]["incumbents"][0]
                sampled = sample_hyperparameters(
                    optuna.trial.FixedTrial(params),
                    model,
                )
                self.assertEqual(sampled, params)
                backup, robot = build_controller_overrides(model, sampled)

                self.assertEqual(
                    backup["vref_scenario_weight_mode"],
                    "barrier_unexpand",
                )
                self.assertEqual(backup["occ_selection_mode"], "h_tilde")
                self.assertEqual(backup["occ_rollout_mode"], "common")
                self.assertEqual(backup["terminal_slack_weight"], 0.0)
                self.assertEqual(robot["occ_kappa"], 10.0)
                if model == "di":
                    self.assertEqual(backup["rho_T"], "auto")
                if model == "uni":
                    self.assertEqual(
                        FIXED_OCBF_CONFIG["unicycle_v_min"],
                        0.0,
                    )
                    self.assertEqual(robot["v_min"], 0.0)
                    self.assertEqual(
                        backup["vref_tracking_mode_occ_uni"],
                        "gated",
                    )

    def test_terminal_relax_mode_samples_only_terminal_slack(self):
        params = {
            "terminal_slack_weight": 10.0,
            "terminal_slack_max": 2.0,
        }
        sampled = sample_hyperparameters(
            optuna.trial.FixedTrial(params),
            "di",
            "terminal_relax",
        )
        self.assertEqual(sampled, params)
        self.assertEqual(
            set(TERMINAL_RELAX_GRID),
            {"terminal_slack_weight", "terminal_slack_max"},
        )

        backup, robot = build_controller_overrides(
            "di",
            sampled,
            "terminal_relax",
        )
        self.assertEqual(backup["terminal_slack_weight"], 10.0)
        self.assertEqual(backup["terminal_slack_max"], 2.0)
        self.assertEqual(backup["obs_hocbf_slack_max"], 0.0)
        self.assertEqual(backup["occ_rollout_slack_max"], 0.0)
        self.assertEqual(robot, {})

    def test_latest_step_pruner_compares_equal_case_prefixes(self):
        pruner = LatestStepMedianPruner(
            n_startup_trials=4,
            n_warmup_steps=20,
            interval_steps=10,
            n_min_trials=4,
        )
        study = optuna.create_study(direction="minimize", pruner=pruner)
        for value in (8000.0, 8500.0, 9000.0, 9500.0):
            trial = study.ask()
            trial.report(value, step=20)
            study.tell(trial, value)

        weak = study.ask()
        weak.report(1000.0, step=10)
        weak.report(10000.0, step=20)
        self.assertTrue(weak.should_prune())

        competitive = study.ask()
        competitive.report(1000.0, step=10)
        competitive.report(8000.0, step=20)
        self.assertFalse(competitive.should_prune())

    def test_all_enabled_studies_use_equal_prefix_pruner(self):
        enabled = build_study_pruner(
            disable_pruning=False,
            startup_trials=12,
            warmup_cases=64,
            interval_cases=32,
        )
        self.assertIsInstance(enabled, LatestStepMedianPruner)
        self.assertEqual(enabled.n_startup_trials, 12)
        self.assertEqual(enabled.n_warmup_steps, 64)
        self.assertEqual(enabled.interval_steps, 32)

        disabled = build_study_pruner(
            disable_pruning=True,
            startup_trials=12,
            warmup_cases=64,
            interval_cases=32,
        )
        self.assertIsInstance(disabled, NopPruner)


if __name__ == "__main__":
    unittest.main()
