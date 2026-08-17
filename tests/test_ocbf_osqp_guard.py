"""Regression tests for Occlusion-CBF's checked OSQP cache path."""

from __future__ import annotations

import types
import unittest

import cvxpy as cp
import numpy as np
from scipy import sparse

from position_control.occlusion_cbf_qp import (
    OSQPCacheUpdateError,
    OcclusionCBFQP,
    _install_checked_osqp_update,
)


class _RecordingLowLevelSolver:
    def __init__(self, vector_return_code=0, matrix_return_code=0):
        self.vector_return_code = int(vector_return_code)
        self.matrix_return_code = int(matrix_return_code)
        self.vector_calls = []
        self.matrix_calls = []

    def update_data_vec(self, *, q, l, u):
        self.vector_calls.append(
            {
                "q": None if q is None else np.asarray(q, dtype=float).copy(),
                "l": None if l is None else np.asarray(l, dtype=float).copy(),
                "u": None if u is None else np.asarray(u, dtype=float).copy(),
            }
        )
        return self.vector_return_code

    def update_data_mat(self, *, P_x, P_i, A_x, A_i):
        self.matrix_calls.append((P_x, P_i, A_x, A_i))
        return self.matrix_return_code


class _FakeOSQPSolver:
    def __init__(self, low_level):
        self._solver = low_level
        self._derivative_cache = {
            "q": np.array([0.0, 0.0]),
            "l": np.array([-np.inf, 0.25]),
            "u": np.array([1.0, 1.0]),
            "P": sparse.eye(2, format="csc"),
            "A": sparse.eye(2, format="csc"),
            "results": object(),
        }

    @staticmethod
    def constant(name):
        if name != "OSQP_INFTY":
            raise KeyError(name)
        return 1e30


def _make_minimal_unicycle_controller(*, terminal_slack=False):
    controller = object.__new__(OcclusionCBFQP)
    controller.robot = types.SimpleNamespace(u_dim=2)
    controller.robot_spec = {
        "model": "Unicycle2D",
        "v_min": 0.0,
        "v_max": 1.0,
        "w_max": 0.5,
        "cbf_feas_tol": 1e-5,
    }
    controller._terminal_slack_enabled = bool(terminal_slack)
    controller._qp_objective_visible_hocbf = False
    controller._max_constraints = 1
    controller._last_qp_constraint_count = None
    controller.u = cp.Variable((2, 1))
    controller.u_ref = cp.Parameter(
        (2, 1), value=np.array([[0.8], [0.0]], dtype=float)
    )
    if terminal_slack:
        controller.terminal_slack_weight = 1.0
    controller._rebuild_qp_problem(1)
    return controller


class OCBFOSQPGuardTests(unittest.TestCase):
    def test_checked_update_pairs_bounds_and_raises_on_low_level_rejection(self):
        low_level = _RecordingLowLevelSolver(vector_return_code=1)
        solver = _FakeOSQPSolver(low_level)
        original_upper = solver._derivative_cache["u"].copy()

        _install_checked_osqp_update(solver)

        with self.assertRaisesRegex(
            OSQPCacheUpdateError, "update_data_vec return_code=1"
        ):
            # CVXPY can request only an upper-bound update. The guard must send
            # the last accepted lower bounds too, making validation atomic.
            solver.update(u=np.array([0.75, 0.8]))

        self.assertTrue(solver._ocbf_checked_update)
        self.assertEqual(len(low_level.vector_calls), 1)
        call = low_level.vector_calls[0]
        self.assertIsNone(call["q"])
        np.testing.assert_allclose(call["l"], [-1e30, 0.25])
        np.testing.assert_allclose(call["u"], [0.75, 0.8])
        # A rejected update must not advance the guard's accepted-data shadow.
        np.testing.assert_allclose(solver._ocbf_guard_u, original_upper)

    def test_certificate_checks_current_rows_input_bounds_and_actual_slack(self):
        controller = object.__new__(OcclusionCBFQP)
        controller.robot_spec = {
            "model": "Unicycle2D",
            "v_min": 0.0,
            "v_max": 1.0,
            "w_max": 0.5,
            "cbf_feas_tol": 1e-6,
        }
        controller._terminal_slack_enabled = True
        controller.u = types.SimpleNamespace(value=np.array([[0.6], [0.2]]))
        controller.terminal_slack = types.SimpleNamespace(value=np.array([[0.1]]))

        current_A = np.array([[1.0, 0.0]])
        current_b = np.array([[0.5]])
        slack_ub = np.array([[0.2]])
        valid = controller._qp_solution_certificate(
            current_A,
            current_b,
            1,
            terminal_slack_ub_val=slack_ub,
            u_cmd=np.array([[0.6], [0.2]]),
            slack_val=np.array([[0.1]]),
        )
        self.assertTrue(valid["certified"], valid)

        changed_current_rows = controller._qp_solution_certificate(
            current_A,
            np.array([[0.45]]),
            1,
            terminal_slack_ub_val=slack_ub,
            u_cmd=np.array([[0.6], [0.2]]),
            slack_val=np.array([[0.1]]),
        )
        self.assertFalse(changed_current_rows["certified"])
        self.assertAlmostEqual(
            changed_current_rows["max_cbf_violation"], 0.05, places=12
        )

        invalid_input = controller._qp_solution_certificate(
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            1,
            terminal_slack_ub_val=slack_ub,
            u_cmd=np.array([[1.1], [0.0]]),
            slack_val=np.array([[0.0]]),
        )
        self.assertFalse(invalid_input["certified"])
        self.assertAlmostEqual(invalid_input["max_input_violation"], 0.1)

        invalid_slack = controller._qp_solution_certificate(
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            1,
            terminal_slack_ub_val=slack_ub,
            u_cmd=np.array([[0.5], [0.0]]),
            slack_val=np.array([[0.25]]),
        )
        self.assertFalse(invalid_slack["certified"])
        self.assertAlmostEqual(invalid_slack["max_slack_violation"], 0.05)

        malformed_slack = controller._qp_solution_certificate(
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            1,
            terminal_slack_ub_val=slack_ub,
            u_cmd=np.array([[0.5], [0.0]]),
            slack_val=np.empty((0, 1)),
        )
        self.assertFalse(malformed_slack["certified"])
        self.assertIn("undersized", malformed_slack["error"])

    def test_same_size_warm_rejection_rebuilds_and_certifies_current_qp(self):
        controller = _make_minimal_unicycle_controller()
        A = np.array([[1.0, 0.0]])

        first_timings = {}
        first_error = controller._solve_qp_with_fallbacks(
            A, np.array([[0.5]]), 1, first_timings
        )
        self.assertIsNone(first_error)
        np.testing.assert_allclose(controller.u.value[:, 0], [0.5, 0.0], atol=1e-5)

        original_problem = controller.cbf_controller
        controller._detach_cached_osqp_data()
        cached_solver = original_problem._solver_cache["OSQP"][0]
        self.assertTrue(cached_solver._ocbf_checked_update)

        def reject_next_warm_update(self, **kwargs):
            raise OSQPCacheUpdateError("forced warm-cache rejection")

        cached_solver.update = types.MethodType(
            reject_next_warm_update, cached_solver
        )
        controller.u_ref.value = np.array([[0.2], [0.0]])

        second_timings = {}
        second_error = controller._solve_qp_with_fallbacks(
            A, np.array([[0.1]]), 1, second_timings
        )

        self.assertIsNone(second_error)
        self.assertIsNot(controller.cbf_controller, original_problem)
        attempts = second_timings["solver_attempts"]
        self.assertGreaterEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["solver"], "OSQP")
        self.assertFalse(attempts[0]["rebuild"])
        self.assertIn("forced warm-cache rejection", attempts[0]["exception"])
        self.assertEqual(attempts[1]["solver"], "OSQP")
        self.assertTrue(attempts[1]["rebuild"])
        self.assertIsNone(attempts[1]["exception"])
        self.assertTrue(attempts[1]["certificate"]["certified"])

        u_current = np.asarray(controller.u.value, dtype=float).reshape(-1, 1)
        np.testing.assert_allclose(u_current[:, 0], [0.1, 0.0], atol=1e-5)
        current_residual = controller._effective_constraint_violation(
            A, np.array([[0.1]]), u_current
        )
        self.assertLessEqual(float(np.max(current_residual)), 1e-5)
        self.assertLessEqual(controller._input_max_violation(u_current), 1e-5)


if __name__ == "__main__":
    unittest.main()
