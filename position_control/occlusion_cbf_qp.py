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

    def _ensure_constraint_capacity(self, required_constraints):
        required = int(required_constraints)
        current = int(getattr(self, "_max_constraints", 0))
        if required <= current:
            return

        u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))
        new_cap = max(required, int(max(1, current) * 1.5))
        self._max_constraints = int(new_cap)

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
        if self.debug:
            print(f"[OcclusionCBFQP] resized constraint buffer: {current} -> {self._max_constraints}")

    def set_occlusion_scenarios(self, scenarios):
        self.occlusion_scenarios = scenarios

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()
        timings = {}

        self.u_ref.value = control_ref["u_ref"]
        self.last_u_ref = np.array(self.u_ref.value, dtype=float).reshape(-1, 1)

        t0 = time.perf_counter()
        if obs_list is None:
            visible_obs, occlusion_scenarios = [], []
        else:
            obs_arr = np.asarray(obs_list)
            if obs_arr.size == 0:
                visible_obs, occlusion_scenarios = [], []
            else:
                visible_obs, occlusion_scenarios = self._occ_utils._filter_visible_and_build_occ(
                    robot_state, obs_list
                )
        if self.num_obs > 0:
            visible_obs = visible_obs[: self.num_obs]
            occlusion_scenarios = occlusion_scenarios[: self.num_obs]
        self.occlusion_scenarios = occlusion_scenarios
        no_obs = len(visible_obs) == 0
        no_occ = len(self.occlusion_scenarios) == 0
        timings["filter_occ_ms"] = (time.perf_counter() - t0) * 1000.0

        if no_obs and no_occ:
            self.status = "optimal"
            self.last_num_constraints = 0
            self.last_qp_solve_time_ms = 0.0
            self.last_intervention = "u_ref"
            self.last_u = self.last_u_ref
            if self.debug:
                print("[OcclusionCBFQP] no detected obstacle/occlusion -> use u_ref")
            return self.u_ref.value

        A_list, b_list = [], []
        meta_list = []

        t0 = time.perf_counter()
        phi_b, Phi_b, tau_points, fcl_traj = self.occlusion_backup.simulate_backup_trajectory(
            robot_state,
            self.T_horizon,
            self.dt_backup,
            occlusion_scenarios=None if no_occ else self.occlusion_scenarios,
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

        # visible obstacle HOCBF (relative-degree 2)
        t0 = time.perf_counter()
        if not no_obs:
            x = float(robot_state[0, 0])
            y = float(robot_state[1, 0])
            vx = float(robot_state[2, 0])
            vy = float(robot_state[3, 0])
            p = np.array([x, y])
            v = np.array([vx, vy])

            robot_radius = self.robot_spec["radius"]
            gamma1 = 1.0
            gamma2 = 1.0

            for obs_idx, obs in enumerate(visible_obs):
                obs = np.asarray(obs, dtype=float).flatten()
                ox, oy, r_obs = obs[:3]
                if len(obs) >= 5:
                    vx_o, vy_o = obs[3:5]
                else:
                    vx_o, vy_o = 0.0, 0.0

                p_obs = np.array([ox, oy])
                v_obs = np.array([vx_o, vy_o])

                p_rel = p - p_obs
                v_rel = v - v_obs
                d_min = r_obs + robot_radius

                h = float(p_rel @ p_rel - d_min**2)
                h_dot = 2.0 * float(p_rel @ v_rel)
                v_rel_norm2 = float(v_rel @ v_rel)
                psi1 = h_dot + gamma1 * h

                A = -2.0 * p_rel.reshape(1, 2)
                b = 2.0 * v_rel_norm2 + gamma2 * psi1

                if np.all(np.isfinite(A)) and np.isfinite(b):
                    A_list.append(A)
                    b_list.append(np.array([[b]]))
                    meta_list.append({"kind": "obs", "obs_idx": obs_idx})
        timings["build_obs_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        # occlusion constraints over backup trajectory
        t0 = time.perf_counter()
        if not no_occ:
            for sc_idx, scenario in enumerate(self.occlusion_scenarios):
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
                        if self.debug:
                            print(f"[occ] skip degenerate (||Lgh||≈0, rhs={rhs:.3e}<0)")
                        continue

                    A_list.append(-Lgh)
                    b_list.append(np.array([[rhs]]))
                    meta_list.append({"kind": "occ", "sc_idx": sc_idx, "tau": float(tau)})
        timings["build_occ_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        # terminal backup-set constraints
        t0 = time.perf_counter()
        phi_T = phi_b[-1].reshape(-1, 1)
        Phi_T = Phi_b[-1]
        term_scenarios = list(self.occlusion_scenarios) if not no_occ else [None]

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
            self.occlusion_backup.set_terminal_backup_context(
                term_scn, self.T_horizon, kappa=self.kappa, rho_T=0.05
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
            self.last_intervention = "u_ref"
            self.last_u = self.last_u_ref
            if self.debug:
                print("[OcclusionCBFQP] no constraints -> use u_ref")
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

        if self.debug:
            max_idx = int(np.argmax(violation))
            _ = meta_list[max_idx] if max_idx < len(meta_list) else None

        if max_violation <= tol and input_ok:
            self.status = "optimal"
            self.last_num_constraints = num_constraints
            self.last_qp_solve_time_ms = 0.0
            self.last_intervention = "u_ref"
            self.last_u = u_ref
            # if self.debug:
                # print(f"[OcclusionCBFQP] u_ref feasible (max_violation={max_violation:.3e}) -> use u_ref")
            return self.u_ref.value

        t0 = time.perf_counter()
        self._ensure_constraint_capacity(num_constraints)
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

        if self.debug:
            print("[PROFILE]", timings)

        _ = (t_end_qp - t_start_qp) * 1000.0

        self.last_num_constraints = num_constraints
        self.last_qp_solve_time_ms = timings["total_ms"]

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
        fallback_scenarios = None if no_occ else self.occlusion_scenarios
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
