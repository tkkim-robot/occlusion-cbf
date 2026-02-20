import time

import cvxpy as cp
import numpy as np

from position_control.backup_controller import OcclusionController
from safe_control.position_control.backup_cbf_qp import BackupCBFQP
from utils.occlusion import OcclusionUtils

"""
Occlusion CBF - BackupCBFQP 기반의 Occlusion-aware 확장 구현.
"""


class OcclusionCBFQP(BackupCBFQP):
    """
    Occlusion-aware Backup CBF-QP.

    - 상속: BackupCBFQP (base backup CBF 메커니즘/공통 상태)
    - 조합: OcclusionUtils (가시성/occlusion scenario 빌드 유틸)
    """

    def __init__(self, robot, robot_spec, num_obs=10, kappa=10.0, ax=None):
        self.num_obs = num_obs
        self.kappa = kappa
        self.occlusion_scenarios = []
        self.last_num_constraints = None
        self.last_qp_solve_time_ms = None
        self.last_solver_solve_time_ms = None
        self.last_total_compute_time_ms = None
        self.last_intervention = None
        self.last_u_ref = None
        self.last_u = None
        self.last_profile = {}

        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))
        self.debug = bool(robot_spec.get("debug_backup_qp", False))

        cfg = robot_spec.setdefault("backup_cbf", {})
        self.T_horizon = float(cfg.get("T_horizon", 2.0))
        self.dt_backup = float(cfg.get("dt_backup", 0.05))
        alpha = float(cfg.get("alpha", 1.0))
        cfg.update({"T_horizon": self.T_horizon, "dt_backup": self.dt_backup, "alpha": alpha})

        super().__init__(
            robot=robot,
            robot_spec=robot_spec,
            dt=self.dt_backup,
            backup_horizon=self.T_horizon,
            ax=ax,
        )
        self.alpha = alpha
        self.occlusion_backup = OcclusionController(robot_spec=robot_spec, dt=self.dt_backup)
        self.occlusion_backup.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=self._occlusion_barrier_smax_curved,
        )

        self.setup_control_problem()

        target = self.robot
        try:
            if hasattr(target, "set_occ_barrier_fn"):
                target.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)
            elif hasattr(target, "robot") and hasattr(target.robot, "set_occ_barrier_fn"):
                target.robot.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)
            else:
                # Expected for safe_control robot models without OCC-specific methods.
                pass
        except Exception as e:
            print(f"[OcclusionCBFQP][WARN] Failed to inject occ barrier callback: {e}")

    def _terminal_rho(self):
        a_lim = float(self.robot_spec.get('a_max', 1.0))
        v_max = float(self.robot_spec.get('v_max', 1.0))
        if a_lim <= 0.0:
            return 0.0
        return (v_max ** 2) / (2.0 * a_lim)

    def _resolve_terminal_rho(self):
        cfg = self.robot_spec.get('backup_cbf', {})
        rho_cfg = cfg.get('rho_T', None)
        if isinstance(rho_cfg, str):
            key = rho_cfg.strip().lower()
            if key in ("auto", "auto_stop", "stop", "stopping_distance"):
                return self._terminal_rho()
            return float(rho_cfg)
        if rho_cfg is None:
            return self._terminal_rho()
        return float(rho_cfg)
    
    def _occ_smax_details(self, pos, scenario, tau=0.0):
        """
        Returns:
            h_tilde (float),
            grad_pos (1x2),
            lam (K,),
            dh_ds (float),
            risk_normal_vec (<=K x 2)
        """
        if scenario is None:
            return None, None, None, None, None

        A = scenario.get("A", None)
        b0 = scenario.get("b0", None)
        if A is None or b0 is None or A.shape[0] < 2:
            return None, None, None, None, None

        kappa = float(self.kappa)
        robot_radius = float(self.robot_spec["radius"])

        pos = np.asarray(pos, float).reshape(2,)
        b0 = np.asarray(b0, float).reshape(-1,)
        K = int(A.shape[0])

        v_expand = scenario.get("v_expand_vec", None)
        if v_expand is None:
            v_adv = float(scenario.get("v_adv_max", 0.0))
            v_expand = np.full((K,), v_adv, dtype=float)
        else:
            v_expand = np.asarray(v_expand, float).reshape(-1,)
            if v_expand.size != K:
                if v_expand.size == 1:
                    v_expand = np.full((K,), float(v_expand.item()), dtype=float)
                else:
                    return None, None, None, None, None

        # h_k(p,tau) = a_k^T p - b0_k - (R + v_expand_k * tau)
        R_occ = robot_radius + v_expand * float(tau)
        h_vec = (A @ pos) - b0 - R_occ
        if not np.all(np.isfinite(h_vec)):
            return None, None, None, None, None

        max_h = float(np.max(h_vec))
        z = np.exp(kappa * (h_vec - max_h))
        Z = float(np.sum(z))
        if (not np.isfinite(Z)) or Z <= 0.0:
            return None, None, None, None, None

        lam = z / Z
        h_tilde = max_h + (np.log(Z) - np.log(K)) / kappa
        grad_pos = (lam[:, None] * A).sum(axis=0, keepdims=True)
        dh_ds = float(lam @ (-v_expand))

        active = h_vec >= 0.0
        risk_normal_vec = A[active] if np.any(active) else np.empty((0, 2))
        return float(h_tilde), grad_pos, lam, dh_ds, risk_normal_vec

    def _occlusion_barrier_smax_curved(self, pos, scenario, tau=0.0):
        h_tilde, grad_pos, _, _, risk_normal_vec = self._occ_smax_details(pos, scenario, tau)
        if h_tilde is None:
            return None, None, None
        return h_tilde, grad_pos, risk_normal_vec

    def setup_control_problem(self):
        u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))

        if not hasattr(self, "u"):
            self.u = cp.Variable((u_dim, 1))
        if not hasattr(self, "u_ref"):
            self.u_ref = cp.Parameter((u_dim, 1), value=np.zeros((u_dim, 1)))

        N_tau = int(self.T_horizon / self.dt_backup) + 2
        max_constraints = int((2 * self.num_obs + 10) * N_tau)
        self._max_constraints = int(max_constraints)

        self.A_cbf = cp.Parameter((self._max_constraints, u_dim), value=np.zeros((self._max_constraints, u_dim)))
        self.b_cbf = cp.Parameter((self._max_constraints, 1), value=np.zeros((self._max_constraints, 1)))

        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
        constraints = [self.A_cbf @ self.u <= self.b_cbf]

        ic_fn = getattr(self.robot, "input_constraints", None)
        if callable(ic_fn):
            constraints.extend(ic_fn(self.u))
        else:
            if u_dim == 2 and "a_max" in self.robot_spec:
                constraints.extend(
                    [
                        cp.abs(self.u[0]) <= self.robot_spec["a_max"],
                        cp.abs(self.u[1]) <= self.robot_spec["a_max"],
                    ]
                )

        self.cbf_controller = cp.Problem(objective, constraints)
        self.status = None

    def set_occlusion_scenarios(self, scenarios):
        self.occlusion_scenarios = scenarios

    def _obs_type(self, obs):
        obs = np.asarray(obs, dtype=float).flatten()
        if obs.size >= 8:
            try:
                return int(obs[7])
            except Exception:
                return None
        return None

    def _dynamic_types(self):
        dyn_types_cfg = self.robot_spec.get("dynamic_obs_types", None)
        if dyn_types_cfg is None:
            return {1}
        return {int(t) for t in dyn_types_cfg}

    def _occlusion_types(self):
        occ_types_cfg = self.robot_spec.get("occlusion_types", None)
        if occ_types_cfg is None:
            return set()
        return {int(t) for t in occ_types_cfg}

    def _is_dynamic_obs_for_di(self, obs):
        obs = np.asarray(obs, dtype=float).flatten()
        obs_type = self._obs_type(obs)
        dyn_types = self._dynamic_types()
        if obs_type is not None:
            return obs_type in dyn_types
        if bool(self.robot_spec.get("dynamic_obs_from_velocity", False)) and obs.size >= 5:
            return float(np.hypot(obs[3], obs[4])) > 1e-9
        return False

    def _build_di_hocbf_constraint(self, robot_state, obs, mode):
        """
        Build one HOCBF inequality row for DoubleIntegrator2D in the form:
            A_row u <= b_row
        compatible with the OcclusionCBFQP stacked QP.
        """
        dt = float(self.robot.dt)
        try:
            f_x = self.robot.f(robot_state)
        except TypeError:
            f_x = self.robot.f()
        try:
            g_x = self.robot.g(robot_state)
        except TypeError:
            g_x = self.robot.g()

        h_dot_t = 0.0
        if self._is_dynamic_obs_for_di(obs):
            dyn_barrier = getattr(self.robot.robot, "dynamic_agent_barrier", None)
            if callable(dyn_barrier):
                res = dyn_barrier(robot_state, obs, self.robot.robot_radius)
            else:
                res = self.robot.agent_barrier(obs)
        else:
            res = self.robot.agent_barrier(obs)

        if isinstance(res, (tuple, list)) and len(res) >= 4:
            h, h_dot, dh_dot_dx, h_dot_t = res[:4]
        else:
            h, h_dot, dh_dot_dx = res[:3]

        Lfh = float((dh_dot_dx @ f_x).item()) + float(h_dot_t)
        Lgh = dh_dot_dx @ g_x

        if mode == "hard":
            rhs = float(h) / (dt ** 2) + 2.0 * float(h_dot) / dt + Lfh
        else:
            alpha1 = float(self.robot_spec.get("hocbf_alpha1", 1.5))
            alpha2 = float(self.robot_spec.get("hocbf_alpha2", 1.5))
            gamma1 = alpha1 + alpha2
            gamma2 = alpha1 * alpha2
            rhs = Lfh + gamma1 * float(h_dot) + gamma2 * float(h)

        return -Lgh, rhs

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()
        timings = {}

        self.u_ref.value = control_ref["u_ref"]
        self.last_u_ref = np.array(self.u_ref.value, dtype=float).reshape(-1, 1)

        t0 = time.perf_counter()
        disable_occ_constraints = bool(self.robot_spec.get("disable_occ_constraints", False))
        occ_types = self._occlusion_types()

        if obs_list is None:
            visible_obs, occlusion_scenarios = [], []
        else:
            obs_arr = np.asarray(obs_list, dtype=float)
            if obs_arr.size == 0:
                visible_obs, occlusion_scenarios = [], []
            else:
                # Visibility filtering (including occlusion geometry) is always active.
                # `disable_occ_constraints` only disables adding occlusion-CBF rows.
                visible_obs, occlusion_scenarios = self._occ_utils._filter_visible_and_build_occ(
                    robot_state, obs_list
                )
        if self.num_obs > 0:
            visible_obs = visible_obs[: self.num_obs]
            occlusion_scenarios = occlusion_scenarios[: self.num_obs]
        occ_for_qp = [] if disable_occ_constraints else occlusion_scenarios
        self.occlusion_scenarios = occ_for_qp

        # HOCBF is applied only to non-occlusion obstacles.
        if disable_occ_constraints:
            hocbf_obs = list(visible_obs)
        else:
            hocbf_obs = [obs for obs in visible_obs if self._obs_type(obs) not in occ_types]

        no_obs = len(hocbf_obs) == 0
        no_occ = len(occ_for_qp) == 0
        timings["filter_occ_ms"] = (time.perf_counter() - t0) * 1000.0

        if no_obs and no_occ:
            self.status = "optimal"
            self.last_num_constraints = 0
            self.last_qp_solve_time_ms = 0.0
            self.last_solver_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_intervention = "u_ref"
            self.last_u = self.last_u_ref
            return self.u_ref.value

        A_list, b_list = [], []
        meta_list = []

        t0 = time.perf_counter()
        phi_b, Phi_b, tau_points, fcl_traj = self.occlusion_backup.simulate_backup_trajectory(
            robot_state,
            self.T_horizon,
            self.dt_backup,
            occlusion_scenarios=None if no_occ else occ_for_qp,
        )
        timings["rollout_stm_ms"] = (time.perf_counter() - t0) * 1000.0

        try:
            f_x = self.robot.f(robot_state)
        except TypeError:
            f_x = self.robot.f()
        try:
            g_x = self.robot.g(robot_state)
        except TypeError:
            g_x = self.robot.g()

        # HOCBF constraints from non-occlusion obstacles
        t0 = time.perf_counter()
        if not no_obs:
            mode = str(self.robot_spec.get("cbf_mode", "cbf")).strip().lower()
            for obs_idx, obs in enumerate(hocbf_obs):
                obs = np.asarray(obs, dtype=float).flatten()

                if self.robot_spec.get("model") == "DoubleIntegrator2D":
                    try:
                        A_row, b_row = self._build_di_hocbf_constraint(robot_state, obs, mode)
                    except Exception:
                        continue
                else:
                    # Keep previous generic behavior for non-DI models.
                    try:
                        h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)
                    except Exception:
                        continue
                    alpha1 = float(self.robot_spec.get("hocbf_alpha1", 1.5))
                    alpha2 = float(self.robot_spec.get("hocbf_alpha2", 1.5))
                    gamma1 = alpha1 + alpha2
                    gamma2 = alpha1 * alpha2
                    A_row = -(dh_dot_dx @ g_x)
                    b_row = float((dh_dot_dx @ f_x).item()) + gamma1 * float(h_dot) + gamma2 * float(h)

                if np.all(np.isfinite(A_row)) and np.isfinite(b_row):
                    A_list.append(A_row)
                    b_list.append(np.array([[b_row]]))
                    meta_list.append({"kind": "obs_hocbf", "obs_idx": obs_idx})
        timings["build_obs_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        # occlusion constraints over backup trajectory
        t0 = time.perf_counter()
        if not no_occ:
            for sc_idx, scenario in enumerate(occ_for_qp):
                for i, tau in enumerate(tau_points):
                    phi_i = phi_b[i].reshape(-1, 1)
                    Phi_i = Phi_b[i]
                    pos_i = phi_i[0:2]

                    h_tilde, grad_pos, _, dh_ds, _ = self._occ_smax_details(pos_i.flatten(), scenario, tau)
                    if h_tilde is None or grad_pos is None:
                        continue

                    state_dim = Phi_i.shape[0]
                    pad_cols = max(0, state_dim - grad_pos.shape[1])
                    grad_h_phi = np.hstack([grad_pos, np.zeros((1, pad_cols))])

                    f_pi_phi = fcl_traj[i].reshape(-1, 1)
                    if f_pi_phi.shape[0] < state_dim:
                        f_pi_pad = np.vstack([f_pi_phi, np.zeros((state_dim - f_pi_phi.shape[0], 1))])
                    else:
                        f_pi_pad = f_pi_phi
                    if f_x.shape[0] < state_dim:
                        f_x_pad = np.vstack([f_x, np.zeros((state_dim - f_x.shape[0], 1))])
                    else:
                        f_x_pad = f_x
                    if g_x.shape[0] < state_dim:
                        g_x_pad = np.vstack([g_x, np.zeros((state_dim - g_x.shape[0], g_x.shape[1]))])
                    else:
                        g_x_pad = g_x

                    time_term = -float(dh_ds)
                    Lfh = grad_h_phi @ (Phi_i @ f_x_pad - f_pi_pad) + time_term
                    Lgh = grad_h_phi @ Phi_i @ g_x_pad

                    if not (np.all(np.isfinite(Lgh)) and np.all(np.isfinite(Lfh))):
                        continue

                    rhs = float(Lfh + self._alpha(h_tilde))
                    if np.linalg.norm(Lgh) < 1e-9 and rhs < 0.0:
                        continue

                    A_list.append(-Lgh)
                    b_list.append(np.array([[rhs]]))
                    meta_list.append({"kind": "occ", "sc_idx": sc_idx, "tau": float(tau)})
        timings["build_occ_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        # terminal backup-set constraints
        t0 = time.perf_counter()
        phi_T = phi_b[-1].reshape(-1, 1)
        Phi_T = Phi_b[-1]
        term_scenarios = list(occ_for_qp) if not no_occ else []

        state_dim_T = Phi_T.shape[0]
        try:
            f_term = self.robot.f(robot_state)
        except TypeError:
            f_term = self.robot.f()
        try:
            g_term = self.robot.g(robot_state)
        except TypeError:
            g_term = self.robot.g()

        if f_term.shape[0] < state_dim_T:
            f_term = np.vstack([f_term, np.zeros((state_dim_T - f_term.shape[0], 1))])
        if g_term.shape[0] < state_dim_T:
            g_term = np.vstack([g_term, np.zeros((state_dim_T - g_term.shape[0], g_term.shape[1]))])

        for sc_idx, term_scn in enumerate(term_scenarios):
            rho_term = self._resolve_terminal_rho()
            self.occlusion_backup.set_terminal_backup_context(
                term_scn, self.T_horizon, kappa=self.kappa, rho_T=rho_term
            )
            h_b_T = self.occlusion_backup._occ_terminal_set(phi_T)
            grad_h_b_T = self.occlusion_backup.grad_occ_terminal(phi_T)

            if h_b_T is None or grad_h_b_T is None:
                continue

            Lfh_b_T = grad_h_b_T @ (Phi_T @ f_term)
            Lgh_b_T = grad_h_b_T @ (Phi_T @ g_term)
            rhs_b = float(Lfh_b_T + self._alpha(h_b_T))

            A_list.append(-Lgh_b_T)
            b_list.append([[rhs_b]])
            meta_list.append({"kind": "terminal", "tau": float(self.T_horizon), "sc_idx": sc_idx})
        timings["build_terminal_ms"] = (time.perf_counter() - t0) * 1000.0

        num_constraints = len(A_list)
        if num_constraints == 0:
            self.status = "optimal"
            self.last_num_constraints = 0
            self.last_qp_solve_time_ms = 0.0
            self.last_solver_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_intervention = "u_ref"
            self.last_u = self.last_u_ref
            return self.u_ref.value

        t0 = time.perf_counter()
        A_cbf_val = np.vstack(A_list).reshape(num_constraints, 2)
        b_cbf_val = np.vstack(b_list).reshape(num_constraints, 1)
        timings["stack_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        tol = float(self.robot_spec.get("intervention_tol", 1e-3))
        u_ref = self.last_u_ref
        violation = (A_cbf_val @ u_ref - b_cbf_val).flatten()
        max_violation = float(np.max(violation))

        input_ok = True
        if u_ref is not None and "a_max" in self.robot_spec:
            a_max = float(self.robot_spec.get("a_max", np.inf))
            input_ok = bool(np.all(np.abs(u_ref.flatten()) <= a_max + tol))

        if max_violation <= tol and input_ok:
            self.status = "optimal"
            self.last_num_constraints = num_constraints
            self.last_qp_solve_time_ms = 0.0
            self.last_solver_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_intervention = "u_ref"
            self.last_u = u_ref
            # if self.debug:
                # print(f"[OcclusionCBFQP] u_ref feasible (max_violation={max_violation:.3e}) -> use u_ref")
            return self.u_ref.value

        t0 = time.perf_counter()
        # Match baseline spirit (build/use constraints each solve): if current
        # preallocated buffers are too small, reallocate and rebuild the QP here.
        current_capacity = int(self.A_cbf.shape[0])
        if num_constraints > current_capacity:
            u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))
            self._max_constraints = int(num_constraints)
            self.A_cbf = cp.Parameter(
                (self._max_constraints, u_dim),
                value=np.zeros((self._max_constraints, u_dim)),
            )
            self.b_cbf = cp.Parameter(
                (self._max_constraints, 1),
                value=np.zeros((self._max_constraints, 1)),
            )

            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A_cbf @ self.u <= self.b_cbf]
            ic_fn = getattr(self.robot, "input_constraints", None)
            if callable(ic_fn):
                constraints.extend(ic_fn(self.u))
            else:
                if u_dim == 2 and "a_max" in self.robot_spec:
                    constraints.extend(
                        [
                            cp.abs(self.u[0]) <= self.robot_spec["a_max"],
                            cp.abs(self.u[1]) <= self.robot_spec["a_max"],
                        ]
                    )
            self.cbf_controller = cp.Problem(objective, constraints)

        self.A_cbf.value[:, :] = 0.0
        self.b_cbf.value[:, :] = 1e6
        self.A_cbf.value[:num_constraints, :] = A_cbf_val
        self.b_cbf.value[:num_constraints, :] = b_cbf_val
        timings["param_assign_ms"] = (time.perf_counter() - t0) * 1000.0

        t_start_qp = time.perf_counter()
        t0 = time.perf_counter()
        solve_exception = None
        try:
            # Baseline policy: OSQP 우선, 실패 시 GUROBI fallback
            try:
                self.cbf_controller.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception:
                self.cbf_controller.solve(solver=cp.GUROBI, verbose=False)
        except Exception as e:
            solve_exception = e
        timings["qp_wall_ms"] = (time.perf_counter() - t0) * 1000.0
        t_end_qp = time.perf_counter()

        stats = self.cbf_controller.solver_stats
        timings["solver_solve_time_s"] = getattr(stats, "solve_time", None)
        timings["solver_setup_time_s"] = getattr(stats, "setup_time", None)
        timings["solver_name"] = getattr(stats, "solver_name", None)
        timings["total_ms"] = (time.perf_counter() - t_all0) * 1000.0
        self.last_profile = timings

        _ = (t_end_qp - t_start_qp) * 1000.0

        self.last_num_constraints = num_constraints
        # QP-only wall time (cvxpy solve call scope, plotting excluded)
        self.last_qp_solve_time_ms = float(timings["qp_wall_ms"])
        solver_s = timings.get("solver_solve_time_s", None)
        self.last_solver_solve_time_ms = (
            float(solver_s) * 1000.0 if solver_s is not None else None
        )
        # Keep full control-cycle compute time separately for profiling.
        self.last_total_compute_time_ms = float(timings["total_ms"])

        qp_status = self.cbf_controller.status
        qp_ok = (
            solve_exception is None
            and qp_status in ["optimal", "optimal_inaccurate"]
            and self.u.value is not None
        )

        if qp_ok:
            u_safe = np.array(self.u.value, dtype=float).reshape(-1, 1)
            self.last_u = u_safe
            if self.last_u_ref is not None:
                delta = float(np.linalg.norm(self.last_u - self.last_u_ref))
            else:
                delta = float("inf")
            self.last_intervention = "u_ref" if delta <= tol else "backup_qp"
            # Baseline은 optimal_inaccurate도 허용하므로 실행 상태는 optimal로 맞춤
            self.status = "optimal"
            self._last_intervention = self.last_intervention != "u_ref"
            self._using_backup = self._last_intervention
            return u_safe

        # Baseline fallback: QP 실패/비최적이면 backup control 사용
        fallback_scenarios = None if no_occ else occ_for_qp
        u_safe = np.array(
            self.occlusion_backup.backup_input_at(
                np.asarray(robot_state, dtype=float).reshape(4, 1),
                scenarios=fallback_scenarios,
                t=0.0,
            ),
            dtype=float,
        ).reshape(-1, 1)
        self.last_u = u_safe
        self.last_intervention = "backup_fallback"
        self.status = "optimal"
        self._last_intervention = True
        self._using_backup = True
        return u_safe
