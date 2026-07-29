"""Scenario naming and launcher-contract regression tests."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from examples import test_crowd
from examples import test_crowd_narrow
from examples import test_multi_crowd
from examples import run_scenario
from examples._baseline_defs import (
    CROWD_BENCHMARK_DEFAULTS,
    OACP_BENCHMARK_DEFAULTS,
    default_benchmark_workers,
)
from position_control.ocbf.defaults import (
    apply_crowd_ocbf_defaults,
    default_visible_hocbf_for_scenario,
)
from tools import benchmark_crowd_trials
from tools.benchmark_crowd_trials import _load_run_crowd_scenario


class ScenarioContractTests(unittest.TestCase):
    def test_shared_launcher_defaults_to_exact_crowd_replay(self):
        scenario_module = mock.Mock()
        scenario_module.main.return_value = 0
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(
                run_scenario.importlib,
                "import_module",
                return_value=scenario_module,
            ) as import_module,
        ):
            self.assertEqual(
                run_scenario.main(
                    [
                        "--model",
                        "uni",
                        "--idx",
                        "28",
                        "--n-rand",
                        "10",
                        "--save-animation",
                    ]
                ),
                0,
            )
            self.assertEqual(
                {
                    key: run_scenario.os.environ.get(key)
                    for key in run_scenario.BENCHMARK_CPU_ENVIRONMENT
                },
                run_scenario.BENCHMARK_CPU_ENVIRONMENT,
            )

        import_module.assert_called_once_with("examples.test_crowd")
        scenario_module.main.assert_called_once_with(
            [
                "--model",
                "uni",
                "--idx",
                "28",
                "--n-rand",
                "10",
                "--save-animation",
            ]
        )

    def test_crowd_ocbf_defaults_use_parkcart_scenario_weighting(self):
        defaults = apply_crowd_ocbf_defaults(None)
        self.assertEqual(
            defaults["vref_scenario_weight_mode"],
            "barrier_unexpand",
        )

        explicit_override = apply_crowd_ocbf_defaults(
            {"vref_scenario_weight_mode": "barrier_expand"}
        )
        self.assertEqual(
            explicit_override["vref_scenario_weight_mode"],
            "barrier_expand",
        )

    def test_canonical_benchmark_profile_matches_reference_command(self):
        self.assertEqual(
            CROWD_BENCHMARK_DEFAULTS,
            {
                "scenario": "crowd",
                "baseline": "oacp_mpc",
                "model": "di",
                "seed": 42,
                "idx_start": 1,
                "idx_end": 100,
                "n_rand": 10,
                "tf": 500.0,
                "crowd_mode": "forced_emergence",
                "forced_events": 6,
                "forced_hidden_speed": 1.0,
                "forced_occluder_radius_min": 0.8,
                "forced_occluder_radius_max": 1.0,
                "forced_validate_occlusion": True,
                "forced_require_corridor_conflict": True,
            },
        )
        self.assertEqual(
            OACP_BENCHMARK_DEFAULTS,
            {
                "allow_solver_fallback": False,
                "dynamic_occluders": True,
                "visible_reach_mode": "constant_velocity",
                "branch_safety_gate": False,
            },
        )
        self.assertEqual(default_benchmark_workers(20), 8)

    def test_benchmark_cli_uses_canonical_profile_by_default(self):
        with (
            mock.patch("sys.argv", ["benchmark_crowd_trials.py"]),
            mock.patch.object(benchmark_crowd_trials.Path, "mkdir"),
            mock.patch.object(
                benchmark_crowd_trials,
                "_run_baseline_sweep",
            ) as run_sweep,
        ):
            self.assertEqual(benchmark_crowd_trials.main(), 0)

        kwargs = run_sweep.call_args.kwargs
        for key in (
            "scenario",
            "baseline",
            "model",
            "seed",
            "idx_start",
            "idx_end",
            "n_rand",
            "tf",
            "crowd_mode",
            "forced_events",
            "forced_hidden_speed",
            "forced_occluder_radius_min",
            "forced_occluder_radius_max",
            "forced_validate_occlusion",
            "forced_require_corridor_conflict",
        ):
            call_key = {"scenario": "scenario_name", "baseline": "baseline_alias"}.get(key, key)
            self.assertEqual(kwargs[call_key], CROWD_BENCHMARK_DEFAULTS[key])
        self.assertEqual(kwargs["workers"], default_benchmark_workers())
        self.assertEqual(
            kwargs["robot_spec_overrides"]["oacp_mpc"],
            OACP_BENCHMARK_DEFAULTS,
        )

    def test_unicycle_benchmark_relies_on_shared_forward_only_profile(self):
        with (
            mock.patch(
                "sys.argv",
                [
                    "benchmark_crowd_trials.py",
                    "--model",
                    "uni",
                    "--baseline",
                    "cbf_qp",
                ],
            ),
            mock.patch.object(benchmark_crowd_trials.Path, "mkdir"),
            mock.patch.object(
                benchmark_crowd_trials,
                "_run_baseline_sweep",
            ) as run_sweep,
        ):
            self.assertEqual(benchmark_crowd_trials.main(), 0)

        overrides = run_sweep.call_args.kwargs["robot_spec_overrides"]
        self.assertNotIn("_uni_forward_only", overrides)
        self.assertNotIn("v_min", overrides)

    def test_single_and_multi_cli_share_canonical_scenario_defaults(self):
        with mock.patch.object(test_crowd, "run_crowd_scenario") as run_single:
            test_crowd.main(["--baseline", "oacp_mpc", "--disable-plot"])

        single_kwargs = run_single.call_args.kwargs
        self.assertEqual(single_kwargs["model_key"], CROWD_BENCHMARK_DEFAULTS["model"])
        self.assertEqual(single_kwargs["case_idx"], CROWD_BENCHMARK_DEFAULTS["idx_start"])
        self.assertTrue(single_kwargs["return_metrics"])
        for key in (
            "tf",
            "seed",
            "n_rand",
            "crowd_mode",
            "forced_events",
            "forced_hidden_speed",
            "forced_occluder_radius_min",
            "forced_occluder_radius_max",
            "forced_validate_occlusion",
            "forced_require_corridor_conflict",
        ):
            self.assertEqual(single_kwargs[key], CROWD_BENCHMARK_DEFAULTS[key])
        self.assertEqual(
            single_kwargs["robot_spec_overrides"]["oacp_mpc"],
            OACP_BENCHMARK_DEFAULTS,
        )

        with mock.patch.object(test_multi_crowd, "run_multi_crowd") as run_multi:
            test_multi_crowd.main([])

        multi_args = run_multi.call_args.args[0]
        self.assertEqual(multi_args.model, CROWD_BENCHMARK_DEFAULTS["model"])
        self.assertEqual(multi_args.idx, CROWD_BENCHMARK_DEFAULTS["idx_start"])
        for key in (
            "tf",
            "seed",
            "n_rand",
            "crowd_mode",
            "forced_events",
            "forced_hidden_speed",
            "forced_occluder_radius_min",
            "forced_occluder_radius_max",
            "forced_validate_occlusion",
            "forced_require_corridor_conflict",
        ):
            self.assertEqual(getattr(multi_args, key), CROWD_BENCHMARK_DEFAULTS[key])
        self.assertEqual(multi_args.oacp_allow_solver_fallback, False)
        self.assertEqual(multi_args.oacp_dynamic_occluders, True)
        self.assertEqual(multi_args.oacp_visible_reach_mode, "constant_velocity")
        self.assertEqual(multi_args.oacp_branch_safety_gate, False)

    def test_no_argument_single_case_defaults_to_tuned_occlusion_cbf(self):
        with mock.patch.object(test_crowd, "run_crowd_scenario") as run_single:
            self.assertEqual(test_crowd.main([]), 0)

        kwargs = run_single.call_args.kwargs
        self.assertEqual(
            kwargs["controller_type"],
            {"pos": "occlusion_cbf_qp"},
        )
        self.assertEqual(kwargs["model_key"], "di")
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["case_idx"], 1)
        self.assertEqual(kwargs["n_rand"], 10)
        self.assertEqual(kwargs["tf"], 500.0)
        self.assertEqual(kwargs["crowd_mode"], "forced_emergence")
        self.assertEqual(kwargs["rand_obs_setting"], "v2")
        self.assertEqual(kwargs["forced_events"], 6)
        self.assertEqual(kwargs["forced_hidden_speed"], 1.0)
        self.assertEqual(kwargs["forced_occluder_radius_min"], 0.8)
        self.assertEqual(kwargs["forced_occluder_radius_max"], 1.0)
        self.assertTrue(kwargs["forced_validate_occlusion"])
        self.assertTrue(kwargs["forced_require_corridor_conflict"])
        self.assertTrue(kwargs["occ_enable_visible_hocbf"])

    def test_method_alias_selects_an_explicit_comparison_controller(self):
        with mock.patch.object(test_crowd, "run_crowd_scenario") as run_single:
            self.assertEqual(
                test_crowd.main(["--method", "cbf_qp", "--disable-plot"]),
                0,
            )

        self.assertEqual(
            run_single.call_args.kwargs["controller_type"],
            {"pos": "cbf_qp"},
        )

    def test_single_case_cli_forwards_reproducible_animation_settings(self):
        metrics = {
            "outcome": "collision",
            "total_steps": 2561,
            "total_sim_time": 128.05,
            "final_goal_distance": 1.0,
        }
        with (
            mock.patch.object(
                test_crowd,
                "run_crowd_scenario",
                return_value=metrics,
            ) as run_single,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(
                test_crowd.main(
                    [
                        "--model",
                        "uni",
                        "--idx",
                        "28",
                        "--n-rand",
                        "10",
                        "--disable-plot",
                        "--save-animation",
                        "true",
                    ]
                ),
                0,
            )

        kwargs = run_single.call_args.kwargs
        self.assertFalse(kwargs["show_animation"])
        self.assertTrue(kwargs["save_animation"])
        self.assertTrue(kwargs["return_metrics"])
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["case_idx"], 28)
        self.assertEqual(kwargs["n_rand"], 10)
        self.assertEqual(
            kwargs["robot_spec_overrides"]["animation_subdir"],
            "crowd/occlusion_cbf/uni_n10_seed42_idx28",
        )
        self.assertEqual(
            kwargs["robot_spec_overrides"]["animation_frame_stride"],
            5,
        )
        self.assertEqual(
            kwargs["robot_spec_overrides"]["animation_frame_dpi"],
            150,
        )
        self.assertNotIn(
            "_uni_forward_only",
            kwargs["robot_spec_overrides"],
        )

    def test_case_seed_is_the_one_based_rng_draw(self):
        self.assertEqual(test_crowd_narrow._compute_case_seed(42, 1), 191664963)
        self.assertEqual(test_crowd_narrow._compute_case_seed(42, 2), 1662057957)
        self.assertEqual(test_crowd_narrow._compute_case_seed(42, 3), 1405681631)

    def test_headless_animation_export_overwrites_video_and_cleans_on_success(self):
        controller = object.__new__(test_crowd_narrow.LocalTrackingControllerDyn_OCC)
        controller.show_animation = False
        controller.save_animation = True
        controller.save_frame_ext = "png"
        controller.animation_export_video = True
        controller.save_folder = "/tmp/ocbf-animation-test"

        with (
            mock.patch(
                "dynamic_env.main.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ) as run_ffmpeg,
            mock.patch(
                "dynamic_env.main.glob.glob",
                return_value=["/tmp/ocbf-animation-test/t_step_0001.png"],
            ) as glob_frames,
            mock.patch("dynamic_env.main.os.remove") as remove,
            mock.patch("dynamic_env.main.os.replace") as replace,
            mock.patch("builtins.print"),
        ):
            controller.export_video()

        command = run_ffmpeg.call_args.args[0]
        self.assertIn("-y", command)
        self.assertEqual(
            command[-1],
            "/tmp/ocbf-animation-test/tracking.tmp.mp4",
        )
        replace.assert_called_once_with(
            "/tmp/ocbf-animation-test/tracking.tmp.mp4",
            "/tmp/ocbf-animation-test/tracking.mp4",
        )
        glob_frames.assert_called_once_with(
            "/tmp/ocbf-animation-test/t_step_*.png"
        )
        remove.assert_called_once_with(
            "/tmp/ocbf-animation-test/t_step_0001.png"
        )

    def test_required_corridor_conflict_is_enforced_for_every_forced_event(self):
        case_seed = test_crowd_narrow._compute_case_seed(
            CROWD_BENCHMARK_DEFAULTS["seed"],
            1,
        )
        _known_obs, _obs_meta, diagnostics = (
            test_crowd._build_route_forced_emergence_scenario(
                case_seed=case_seed,
                n_rand=CROWD_BENCHMARK_DEFAULTS["n_rand"],
                rand_obs=True,
                static_occluders=False,
                forced_events=CROWD_BENCHMARK_DEFAULTS["forced_events"],
                forced_bg_rand=None,
                forced_hidden_speed=CROWD_BENCHMARK_DEFAULTS["forced_hidden_speed"],
                forced_occluder_radius_min=CROWD_BENCHMARK_DEFAULTS[
                    "forced_occluder_radius_min"
                ],
                forced_occluder_radius_max=CROWD_BENCHMARK_DEFAULTS[
                    "forced_occluder_radius_max"
                ],
                forced_validate_occlusion=True,
                forced_require_corridor_conflict=True,
                rand_obs_setting=test_crowd_narrow.DEFAULT_RAND_OBS_SETTING,
            )
        )

        expected = CROWD_BENCHMARK_DEFAULTS["forced_events"]
        self.assertEqual(diagnostics["n_forced_events"], expected)
        self.assertEqual(diagnostics["n_forced_initially_occluded"], expected)
        self.assertEqual(diagnostics["n_forced_corridor_conflict"], expected)

    def test_canonical_crowd_geometry_and_display_label(self):
        self.assertEqual(test_crowd.ENV_WIDTH, 30.0)
        self.assertEqual(test_crowd.ENV_HEIGHT, 30.0)

        empty_obstacles = np.empty((0, 8), dtype=float)
        with (
            mock.patch.object(
                test_crowd,
                "_build_route_random_scenario",
                return_value=(empty_obstacles, [], {}),
            ),
            mock.patch.object(
                test_crowd.crowd_narrow,
                "run_crowd_scenario",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            forwarded = test_crowd.run_crowd_scenario(
                controller_type={"pos": "cbf_qp"},
                crowd_mode="random",
                rand_obs=False,
                n_rand=0,
                show_animation=False,
            )

        self.assertEqual(forwarded["scenario_name"], "Crowd")
        self.assertEqual(forwarded["env_width_override"], 30.0)
        self.assertEqual(forwarded["env_height_override"], 30.0)

    def test_shared_launcher_exposes_current_scenarios(self):
        self.assertEqual(
            run_scenario.SCENARIO_MODULES,
            {
                "crowd": "examples.test_crowd",
                "crowd_narrow": "examples.test_crowd_narrow",
                "campus": "examples.test_campus",
                "crosswalk": "examples.test_crosswalk",
            },
        )

    def test_benchmark_loader_resolves_current_scenarios(self):
        runner, scenario_name = _load_run_crowd_scenario("crowd")
        self.assertIs(runner, test_crowd.run_crowd_scenario)
        self.assertEqual(scenario_name, "crowd")

        runner, scenario_name = _load_run_crowd_scenario("crowd_narrow")
        self.assertIs(runner, test_crowd_narrow.run_crowd_scenario)
        self.assertEqual(scenario_name, "crowd_narrow")

        with self.assertRaises(ValueError):
            _load_run_crowd_scenario("unknown")

    def test_visible_hocbf_default_is_limited_to_crowd(self):
        self.assertTrue(default_visible_hocbf_for_scenario("crowd"))

        for other in ("crowd_narrow", "campus", "crosswalk", "", None):
            with self.subTest(other=other):
                self.assertFalse(default_visible_hocbf_for_scenario(other))


if __name__ == "__main__":
    unittest.main()
