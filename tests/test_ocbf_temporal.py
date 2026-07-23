"""Regression tests for the accepted pure-facet temporal correction."""

from __future__ import annotations

import unittest

import numpy as np

from position_control.occlusion_cbf_qp import OcclusionCBFQP


class PureFacetTemporalTests(unittest.TestCase):
    def setUp(self):
        # These helpers only need kappa and robot radius; bypassing __init__
        # keeps the test independent of QP construction and solver availability.
        self.controller = object.__new__(OcclusionCBFQP)
        self.controller.kappa = 7.5
        self.controller.robot_spec = {"radius": 0.24}
        self.position = np.array([0.63, -0.37])
        self.scenario = {
            "A": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-0.8, 0.6],
                ],
                dtype=float,
            ),
            "b0": np.array([0.12, -0.23, -0.08], dtype=float),
            "v_expand_vec": np.array([0.15, 0.42, 0.27], dtype=float),
        }

    def test_pure_propagation_derivatives_match_weighted_facet_speed(self):
        _, _, lam, dh_ds_from_barrier, _ = self.controller._occ_smax(
            self.position, self.scenario, tau=0.41
        )
        dh_dt, dh_ds = (
            self.controller._occ_explicit_time_derivatives_pure_propagation(
                lam, self.scenario
            )
        )
        expected = -float(lam @ self.scenario["v_expand_vec"])

        self.assertAlmostEqual(dh_dt, expected, places=12)
        self.assertAlmostEqual(dh_ds, expected, places=12)
        self.assertAlmostEqual(dh_ds_from_barrier, expected, places=12)
        self.assertAlmostEqual(dh_dt - dh_ds, 0.0, places=12)

    def test_softmax_barrier_propagation_time_derivative(self):
        tau = 0.41
        eps = 1.0e-6
        _, _, lam, _, _ = self.controller._occ_smax(
            self.position, self.scenario, tau=tau
        )
        expected, _ = (
            self.controller._occ_explicit_time_derivatives_pure_propagation(
                lam, self.scenario
            )
        )
        h_plus = self.controller._occ_smax(
            self.position, self.scenario, tau=tau + eps
        )[0]
        h_minus = self.controller._occ_smax(
            self.position, self.scenario, tau=tau - eps
        )[0]
        numerical = (h_plus - h_minus) / (2.0 * eps)

        self.assertAlmostEqual(numerical, expected, places=7)

    def test_softmax_barrier_real_time_facet_propagation_derivative(self):
        eps = 1.0e-6
        _, _, lam, _, _ = self.controller._occ_smax(
            self.position, self.scenario, tau=0.0
        )
        expected, _ = (
            self.controller._occ_explicit_time_derivatives_pure_propagation(
                lam, self.scenario
            )
        )

        def propagated_barrier(time_s):
            propagated = dict(self.scenario)
            propagated["b0"] = (
                self.scenario["b0"]
                + self.scenario["v_expand_vec"] * float(time_s)
            )
            return self.controller._occ_smax(
                self.position, propagated, tau=0.0
            )[0]

        numerical = (
            propagated_barrier(eps) - propagated_barrier(-eps)
        ) / (2.0 * eps)
        self.assertAlmostEqual(numerical, expected, places=7)


if __name__ == "__main__":
    unittest.main()
