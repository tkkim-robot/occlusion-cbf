"""Regression tests for the OCBF Optuna tuning contract."""

from __future__ import annotations

import unittest

import numpy as np
import optuna

from tools.tune_ocbf_optuna import (
    ACTIVE_OCCLUSION_CHOICES,
    FIXED_OCBF_CONFIG,
    LatestStepMedianPruner,
    MODEL_DEFAULTS,
    SEARCH_SPACE,
    TERMINAL_RELAX_GRID,
    TrialCaseError,
    build_controller_overrides,
    compatibility_fingerprint,
    lexicographic_score,
    sample_hyperparameters,
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
            slower_but_fewer_collisions,
            faster_but_more_collisions,
        )

        more_success = lexicographic_score(
            collision_count=0,
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

    def test_model_profiles_use_requested_obstacle_counts(self):
        self.assertEqual(MODEL_DEFAULTS["di"]["n_rand"], 50)
        self.assertEqual(MODEL_DEFAULTS["uni"]["n_rand"], 30)

    def test_active_occlusion_choices_match_tuning_contract(self):
        self.assertEqual(ACTIVE_OCCLUSION_CHOICES, [0, 3, 5, 10])
        for model in ("di", "uni"):
            self.assertEqual(
                SEARCH_SPACE[model]["max_active_occlusions"]["choices"],
                [0, 3, 5, 10],
            )
            for incumbent in MODEL_DEFAULTS[model]["incumbents"]:
                self.assertIn(
                    incumbent["max_active_occlusions"],
                    [0, 3, 5, 10],
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

    def test_compatibility_fingerprint_tracks_result_inputs_only(self):
        base = {
            "model": "di",
            "study_name": "paper_di",
            "n_rand": 50,
            "trials": 40,
            "timeout_s": 100,
        }
        fingerprint = compatibility_fingerprint(base)
        self.assertEqual(
            fingerprint,
            compatibility_fingerprint(
                {**base, "trials": 80, "timeout_s": 200}
            ),
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


if __name__ == "__main__":
    unittest.main()
