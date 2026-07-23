"""Scenario naming and compatibility-contract regression tests."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from examples import test_crowd
from examples import test_crowd1
from examples import test_crowd2
from examples import test_crowd_narrow
from position_control.ocbf.defaults import default_visible_hocbf_for_scenario
from tools.benchmark_crowd_trials import _load_run_crowd_scenario


class ScenarioContractTests(unittest.TestCase):
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

    def test_legacy_modules_forward_public_and_private_helpers(self):
        self.assertIs(test_crowd2.run_crowd_scenario, test_crowd.run_crowd_scenario)
        self.assertIs(
            test_crowd2._route_point_at_distance,
            test_crowd._route_point_at_distance,
        )
        self.assertIs(
            test_crowd1.run_crowd_scenario,
            test_crowd_narrow.run_crowd_scenario,
        )
        self.assertIs(
            test_crowd1._safe_normalize,
            test_crowd_narrow._safe_normalize,
        )

    def test_benchmark_loader_normalizes_crowd_aliases(self):
        for alias in ("crowd", "crowd2", "test_crowd", "test_crowd2"):
            with self.subTest(alias=alias):
                runner, canonical_name = _load_run_crowd_scenario(alias)
                self.assertIs(runner, test_crowd.run_crowd_scenario)
                self.assertEqual(canonical_name, "crowd")

        for alias in (
            "crowd_narrow",
            "crowd1",
            "test_crowd_narrow",
            "test_crowd1",
        ):
            with self.subTest(alias=alias):
                runner, canonical_name = _load_run_crowd_scenario(alias)
                self.assertIs(runner, test_crowd_narrow.run_crowd_scenario)
                self.assertEqual(canonical_name, "crowd_narrow")

    def test_visible_hocbf_default_is_limited_to_canonical_crowd_aliases(self):
        for alias in ("crowd", "crowd2", "test_crowd", "test_crowd2"):
            with self.subTest(alias=alias):
                self.assertTrue(default_visible_hocbf_for_scenario(alias))

        for other in (
            "crowd_narrow",
            "crowd1",
            "test_crowd_narrow",
            "test_crowd1",
            "campus",
            "crosswalk",
            "",
            None,
        ):
            with self.subTest(other=other):
                self.assertFalse(default_visible_hocbf_for_scenario(other))


if __name__ == "__main__":
    unittest.main()
