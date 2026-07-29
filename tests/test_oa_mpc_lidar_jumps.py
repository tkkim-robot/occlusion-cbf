"""Regression tests for OA-MPC LiDAR range-jump geometry."""

from __future__ import annotations

import unittest

import numpy as np

from position_control.oa_mpc import OAMPC
from utils.occlusion import OcclusionUtils


class OAMPCLidarJumpTests(unittest.TestCase):
    def _controller(self):
        controller = object.__new__(OAMPC)
        controller.lidar_jump_threshold = 0.45
        controller.min_occ_boundary_len = 0.08
        controller.max_occ_boundary_len = 10.0
        controller.max_occ_boundaries = 12
        controller.v_hidden_max_default = 0.5
        controller.sensing_range = 10.0
        controller._occ_utils = object.__new__(OcclusionUtils)
        return controller

    @staticmethod
    def _scan(ranges, points, hit_mask, *, full_circle=False):
        return {
            "origin": np.zeros(2),
            "ranges": np.asarray(ranges, dtype=float),
            "points": np.asarray(points, dtype=float),
            "hit_mask": np.asarray(hit_mask, dtype=bool),
            "full_circle": bool(full_circle),
        }

    def test_hit_run_produces_two_near_to_far_jump_segments(self):
        points = np.array(
            [
                [10.0, -1.0],
                [3.0, -0.2],
                [3.0, 0.2],
                [10.0, 1.0],
            ]
        )
        scan = self._scan(
            ranges=[10.0, 3.0, 3.0, 10.0],
            points=points,
            hit_mask=[False, True, True, False],
        )

        scenarios = self._controller()._build_occ_scenarios_from_lidar_jumps(scan)

        self.assertEqual(len(scenarios), 2)
        np.testing.assert_allclose(scenarios[0]["t1"], points[1])
        np.testing.assert_allclose(scenarios[0]["t2"], points[0])
        np.testing.assert_allclose(scenarios[1]["t1"], points[2])
        np.testing.assert_allclose(scenarios[1]["t2"], points[3])
        self.assertEqual(scenarios[0]["source"], "lidar_range_jump")
        self.assertEqual(scenarios[1]["source"], "lidar_range_jump")

        # The former bug produced one segment across the two near hit points.
        self.assertFalse(
            any(
                np.allclose(scenario["t1"], points[1])
                and np.allclose(scenario["t2"], points[2])
                for scenario in scenarios
            )
        )

    def test_jump_between_two_hit_surfaces_is_detected(self):
        points = np.array([[2.0, 0.0], [5.0, 0.1]])
        scan = self._scan(
            ranges=[2.0, 5.0],
            points=points,
            hit_mask=[True, True],
        )

        scenarios = self._controller()._build_occ_scenarios_from_lidar_jumps(scan)

        self.assertEqual(len(scenarios), 1)
        np.testing.assert_allclose(scenarios[0]["t1"], points[0])
        np.testing.assert_allclose(scenarios[0]["t2"], points[1])

    def test_partial_fov_does_not_create_wraparound_jump(self):
        points = np.array(
            [
                [2.0, -1.0],
                [2.0, -0.3],
                [5.0, 0.3],
                [5.0, 1.0],
            ]
        )
        partial_scan = self._scan(
            ranges=[2.0, 2.0, 5.0, 5.0],
            points=points,
            hit_mask=[True, True, True, True],
            full_circle=False,
        )
        full_scan = dict(partial_scan, full_circle=True)

        partial_scenarios = self._controller()._build_occ_scenarios_from_lidar_jumps(partial_scan)
        full_scenarios = self._controller()._build_occ_scenarios_from_lidar_jumps(full_scan)

        self.assertEqual(len(partial_scenarios), 1)
        self.assertEqual(len(full_scenarios), 2)


if __name__ == "__main__":
    unittest.main()
