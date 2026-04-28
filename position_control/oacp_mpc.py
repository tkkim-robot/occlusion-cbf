import math
import time

import numpy as np

from position_control._mpc_common import MPCCommonUtils
from utils.occlusion import OcclusionUtils

try:
    import casadi as ca
except Exception:
    ca = None

try:
    import cvxpy as cp
except Exception:
    cp = None

try:
    import osqp
except Exception:
    osqp = None

try:
    import scipy.sparse as sp
except Exception:
    sp = None


class OACPMPC(MPCCommonUtils):
    """
    Centralized contingency-MPC adaptation inspired by OACP,
    "Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Driving".

    Implemented features from the paper:
      - a two-branch contingency structure with explicit shared/explore/
        fallback trajectories,
      - receding-horizon execution where the applied control comes from the
        shared prefix,
      - centralized joint optimization of the three trajectories as one coupled
        CasADi IPOPT NLP,
      - active-occlusion selection, SRQ-inspired occlusion scoring, and
        branch-specific stage-wise velocity boundaries,
      - explicit phantom hidden-agent hypotheses generated from occlusion
        tangents and enforced through branch-wise barrier/avoidance
        constraints,
      - model-aware adaptations of the branch tracking / speed-bound logic,
      - route-tracking references shared across the branches, with an optional
        Bezier-smoothed reference scaffold used only for reference generation.

    Still not paper-exact here:
      - the SRQ model and dynamic velocity-boundary equations are adapted
        surrogates for the test_crowd2 benchmark rather than an exact reproduction of
        the paper's road-structured formulation,
      - the optimization is not the paper's biconvex Bezier control-point
        program,
      - the ALM / consensus-ADMM solve procedure is intentionally omitted; this
        repo benchmarks a centralized coupled NLP instead,
      - DoubleIntegrator2D support is a benchmark-specific 2D adaptation rather
        than the paper's original vehicle / lane model.
    """

    def __init__(self, robot, robot_spec, num_obs=30):
        if ca is None:
            raise RuntimeError("CasADi is required for OACPMPC.")

        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = int(num_obs)

        self.model = str(robot_spec.get("model", "")).strip()
        if self.model not in {"DoubleIntegrator2D", "Unicycle2D", "DynamicUnicycle2D"}:
            raise ValueError(
                f"OACPMPC currently supports DoubleIntegrator2D, Unicycle2D and DynamicUnicycle2D, got `{self.model}`"
            )

        self._n_state, self._u_dim = self._dims()
        self.dt = float(getattr(robot, "dt", robot_spec.get("dt", 0.05)))
        self.robot_radius = float(robot_spec.get("radius", 0.25))
        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))

        cfg = robot_spec.setdefault("oacp_mpc", {})
        # Use a planning grid aligned with the crowd benchmark runtime step.
        # Keep the horizon short enough for dense pedestrian emergence scenes.
        self.dt_plan = float(cfg.get("dt_plan", 0.05))
        self.Th = float(cfg.get("Th", 1.0))
        n_from_h = int(np.round(self.Th / max(self.dt_plan, 1e-6)))
        self.N = max(5, int(cfg.get("N", n_from_h)))
        self.n_shared = int(cfg.get("n_shared", min(3, self.N - 1)))
        self.n_shared = max(1, min(self.N - 1, self.n_shared))
        self.n_tail = int(self.N - self.n_shared)
        self.requested_backend = str(cfg.get("backend", "coupled_nlp")).strip().lower()
        # OACP benchmarking in this repo is intentionally ADMM-free. Keep the
        # public backend knob for compatibility, but execute the centralized NLP
        # path as the paper-spirit baseline used in crowd benchmarks.
        self.backend = "coupled_nlp"

        self.forward_only = bool(cfg.get("forward_only", True))
        self.du_nonnegative_speed = bool(cfg.get("du_nonnegative_speed", True))
        self.allow_solver_fallback = bool(cfg.get("allow_solver_fallback", False))
        self.dynamic_occluders = bool(cfg.get("dynamic_occluders", True))
        self.visible_reach_mode = str(cfg.get("visible_reach_mode", "constant_velocity")).strip().lower()
        if self.visible_reach_mode not in {"constant_velocity", "worst_case"}:
            self.visible_reach_mode = "constant_velocity"

        # Keep anticipated obstacle counts close to the structured-road paper regime.
        self.max_visible_obs = max(1, int(cfg.get("max_visible_obs", min(self.num_obs, 30))))
        self.max_occ_scenarios = max(1, int(cfg.get("max_occ_scenarios", 30)))
        self.max_active_occlusions = max(1, int(cfg.get("max_active_occlusions", min(2, self.max_occ_scenarios))))
        self.max_path_points = max(3, int(cfg.get("max_path_points", 10)))
        self.max_phantom_obs = max(1, 2 * self.max_active_occlusions)
        self.num_occ_facets = 4

        self.margin_obs = float(cfg.get("margin_obs", 0.05))
        self.v_ref_default = float(cfg.get("v_ref_default", 0.45))
        self.v_visible_max = float(
            cfg.get(
                "v_visible_max",
                robot_spec.get("v_obs_max", robot_spec.get("v_adv_max_occ", 0.5)),
            )
        )
        self.hidden_agent_radius = float(cfg.get("hidden_agent_radius", 0.4))
        self.hidden_spawn_clearance = float(cfg.get("hidden_spawn_clearance", 0.12))
        self.hidden_speed_scale = float(cfg.get("hidden_speed_scale", 1.0))
        self.hidden_speed = float(
            cfg.get(
                "hidden_speed",
                robot_spec.get("v_adv_max_occ", robot_spec.get("v_obs_max", 0.5)),
            )
        )
        self.active_selection_delta = float(cfg.get("active_selection_delta", 1.0))

        # Keep the SRQ risk horizon tied to the active planning horizon so the
        # risk window matches the receding-horizon optimization window.
        self.srq_horizon = float(self.Th)
        self.srq_confidence_z = float(cfg.get("srq_confidence_z", 1.645))
        self.srq_lane_width_min = float(cfg.get("srq_lane_width_min", 0.8))
        self.srq_lane_width_max = float(cfg.get("srq_lane_width_max", 2.5))
        self.srq_lat_sigma_floor = float(cfg.get("srq_lat_sigma_floor", 0.18))
        self.srq_risk_clip = float(cfg.get("srq_risk_clip", 4.0))
        self.cth_min = float(cfg.get("cth_min", 0.0))
        self.cth_max_explore = float(cfg.get("cth_max_explore", 0.85))
        self.cth_max_fallback = float(cfg.get("cth_max_fallback", 0.55))
        self.v_occ_min_scale = float(cfg.get("v_occ_min_scale", 0.15))
        self.v_occ_min_abs = float(cfg.get("v_occ_min_abs", 0.0))

        self.di_use_progress_speed_cost = bool(cfg.get("di_use_progress_speed_cost", True))
        self.di_lateral_velocity_weight = float(cfg.get("di_lateral_velocity_weight", 1.25))
        self.di_cross_track_scale_shared = float(cfg.get("di_cross_track_scale_shared", 0.50))
        self.di_cross_track_scale_branch = float(cfg.get("di_cross_track_scale_branch", 0.35))
        self.barrier_alpha_start = float(cfg.get("barrier_alpha_start", 0.4))
        self.barrier_alpha_end = float(cfg.get("barrier_alpha_end", 1.0))
        self.ellipse_scale_x = float(cfg.get("ellipse_scale_x", 1.0))
        self.ellipse_scale_y = float(cfg.get("ellipse_scale_y", 1.0))
        self.ellipse_buffer_x = float(cfg.get("ellipse_buffer_x", self.margin_obs))
        self.ellipse_buffer_y = float(cfg.get("ellipse_buffer_y", self.margin_obs))
        self.use_bezier_reference = bool(cfg.get("use_bezier_reference", True))
        self.bezier_ref_order = max(2, int(cfg.get("bezier_ref_order", 10)))
        self.branch_switch_margin = float(cfg.get("branch_switch_margin", 0.05))

        self.w_pos_shared = float(cfg.get("w_pos_shared", 12.0))
        self.w_heading_shared = float(cfg.get("w_heading_shared", 2.5))
        self.w_speed_shared = float(cfg.get("w_speed_shared", 1.5))
        self.w_pos_explore = float(cfg.get("w_pos_explore", 8.0))
        self.w_heading_explore = float(cfg.get("w_heading_explore", 1.5))
        self.w_speed_explore = float(cfg.get("w_speed_explore", 1.0))
        self.w_pos_fallback = float(cfg.get("w_pos_fallback", 10.0))
        self.w_heading_fallback = float(cfg.get("w_heading_fallback", 2.0))
        self.w_speed_fallback = float(cfg.get("w_speed_fallback", 1.5))
        self.w_u = float(cfg.get("w_u", 0.15))
        self.w_du = float(cfg.get("w_du", 0.35))
        self.w_terminal_goal = float(cfg.get("w_terminal_goal", 12.0))
        self.w_barrier_slack_shared = float(
            cfg.get("w_barrier_slack_shared", cfg.get("w_vis_slack_shared", 1200.0))
        )
        self.w_barrier_slack_explore = float(
            cfg.get("w_barrier_slack_explore", cfg.get("w_vis_slack_explore", 900.0))
        )
        self.w_barrier_slack_fallback = float(
            cfg.get("w_barrier_slack_fallback", cfg.get("w_vis_slack_fallback", 1200.0))
        )
        self.di_lateral_speed_cap_scale = float(cfg.get("di_lateral_speed_cap_scale", 0.60))
        self.branch_goal_weight = float(cfg.get("branch_goal_weight", 1.0))
        self.branch_progress_weight = float(cfg.get("branch_progress_weight", 0.25))
        self.branch_slack_weight = float(cfg.get("branch_slack_weight", 6.0))
        self.branch_clearance_weight = float(cfg.get("branch_clearance_weight", 0.15))

        self.solver_tol = float(cfg.get("solver_tol", 1e-3))
        self.solver_acceptable_tol = float(cfg.get("solver_acceptable_tol", 5e-3))
        self.max_iter = int(cfg.get("max_iter", 250))
        self.print_solver = bool(cfg.get("print_solver", False))

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=None,
        )

        self._last_selected_branch = "explore"
        self.branch_switch_count = 0

        self.status = "optimal"
        self.last_num_constraints = 0
        self.last_qp_solve_time_ms = 0.0
        self.last_total_compute_time_ms = 0.0
        self.last_intervention = "u_ref"
        self.last_u_ref = np.zeros((self._u_dim, 1), dtype=float)
        self.last_u = np.zeros((self._u_dim, 1), dtype=float)
        self.last_profile = {}
        self.last_qp_status_raw = ""
        self.last_qp_exception = ""
        self.occlusion_scenarios = []

        self.selected_branch = "explore"
        self.shared_prefix_length = int(self.n_shared)
        self.explore_cost = None
        self.fallback_cost = None
        self.explore_feasible = False
        self.fallback_feasible = False
        self.occlusion_risk_score = 0.0
        self.visible_pressure_score = 0.0
        self.explore_speed_cap = 0.0
        self.fallback_speed_cap = 0.0
        self.shared_risk_slack = 0.0
        self.explore_risk_slack = 0.0
        self.fallback_risk_slack = 0.0
        self.shared_speed_ref_min = 0.0
        self.explore_speed_ref_min = 0.0
        self.fallback_speed_ref_min = 0.0

        self._u_prev_applied = np.zeros((self._u_dim, 1), dtype=float)
        self._sol_prev = None
        self._admm_prev = None
        self.admm_rho = float(cfg.get("admm_rho", 2.0))
        self.admm_max_iter = int(cfg.get("admm_max_iter", 3))
        self.admm_pri_tol = float(cfg.get("admm_pri_tol", 1e-2))
        self.admm_dual_tol = float(cfg.get("admm_dual_tol", 1e-2))
        self.admm_shared_ctrl_pts = max(1, int(cfg.get("admm_shared_ctrl_pts", min(2, self.n_shared))))
        self.admm_tail_ctrl_pts = max(1, int(cfg.get("admm_tail_ctrl_pts", min(3, self.n_tail))))
        self.admm_shared_ctrl_pts = min(self.admm_shared_ctrl_pts, max(1, self.n_shared))
        self.admm_tail_ctrl_pts = min(self.admm_tail_ctrl_pts, max(1, self.n_tail))
        self._basis_shared_np = self._bernstein_basis(self.admm_shared_ctrl_pts, self.n_shared)
        self._basis_tail_np = self._bernstein_basis(self.admm_tail_ctrl_pts, self.n_tail)
        self._basis_shared_ca = ca.DM(self._basis_shared_np)
        self._basis_tail_ca = ca.DM(self._basis_tail_np)
        self._build_backend()

    @staticmethod
    def _angle_diff_ca(a, b):
        return ca.atan2(ca.sin(a - b), ca.cos(a - b))

    @staticmethod
    def _bernstein_basis(n_ctrl, n_steps):
        n_ctrl = int(max(1, n_ctrl))
        n_steps = int(max(0, n_steps))
        if n_steps == 0:
            return np.zeros((n_ctrl, 0), dtype=float)
        if n_ctrl == 1:
            return np.ones((1, n_steps), dtype=float)
        deg = n_ctrl - 1
        tau = np.linspace(0.0, 1.0, n_steps, dtype=float)
        basis = np.zeros((n_ctrl, n_steps), dtype=float)
        for i in range(n_ctrl):
            coeff = float(math.comb(deg, i))
            basis[i, :] = coeff * np.power(tau, i) * np.power(1.0 - tau, deg - i)
        return basis

    @staticmethod
    def _fit_ctrl_points_np(U_seq, basis):
        U_seq = np.asarray(U_seq, dtype=float)
        basis = np.asarray(basis, dtype=float)
        if U_seq.ndim != 2:
            U_seq = U_seq.reshape(2, -1)
        if basis.size == 0:
            return np.zeros((U_seq.shape[0], 0), dtype=float)
        gram = basis @ basis.T
        gram = gram + 1e-6 * np.eye(gram.shape[0], dtype=float)
        return U_seq @ basis.T @ np.linalg.inv(gram)

    def _expand_ctrl_points_np(self, ctrl_pts, basis):
        ctrl_pts = np.asarray(ctrl_pts, dtype=float)
        basis = np.asarray(basis, dtype=float)
        if basis.size == 0:
            return np.zeros((ctrl_pts.shape[0], 0), dtype=float)
        return ctrl_pts @ basis

    def _rollout_controls_np(self, x0, U):
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        U = np.asarray(U, dtype=float)
        X = np.zeros((self._n_state, int(U.shape[1]) + 1), dtype=float)
        X[:, 0] = x0
        for k in range(U.shape[1]):
            X[:, k + 1] = self._discrete_np(X[:, k], U[:, k])
        return X

    def _build_backend(self):
        if self.backend == "admm_lowdim":
            self._build_lowdim_admm_solvers()
        else:
            self._build_coupled_solver()

    def _flatten_lowdim_ctrl_np(self, Csh, Ct):
        Csh = np.asarray(Csh, dtype=float).reshape(self._u_dim, self.admm_shared_ctrl_pts)
        Ct = np.asarray(Ct, dtype=float).reshape(self._u_dim, self.admm_tail_ctrl_pts)
        return np.concatenate([Csh[0, :], Csh[1, :], Ct[0, :], Ct[1, :]], axis=0)

    def _unpack_lowdim_ctrl_np(self, d):
        d = np.asarray(d, dtype=float).reshape(-1)
        m_sh = self.admm_shared_ctrl_pts
        m_t = self.admm_tail_ctrl_pts
        Csh = np.vstack([d[:m_sh], d[m_sh: 2 * m_sh]])
        Ct = np.vstack([d[2 * m_sh: 2 * m_sh + m_t], d[2 * m_sh + m_t: 2 * m_sh + 2 * m_t]])
        return Csh, Ct

    def _rollout_lowdim_ctrl_np(self, x0, Csh, Ct):
        Ush = self._expand_ctrl_points_np(Csh, self._basis_shared_np)
        Ut = self._expand_ctrl_points_np(Ct, self._basis_tail_np)
        Xsh = self._rollout_controls_np(x0, Ush)
        Xt = self._rollout_controls_np(Xsh[:, -1], Ut)
        return Xsh, Xt, Ush, Ut

    def _build_lowdim_affine_maps(self):
        if hasattr(self, "_lowdim_affine_maps"):
            return self._lowdim_affine_maps

        nvar = 2 * (self.admm_shared_ctrl_pts + self.admm_tail_ctrl_pts)
        zero_x = np.zeros((self._n_state,), dtype=float)
        Ush_maps = []
        Ut_maps = []
        Psh_maps = []
        Vsh_maps = []
        Pt_maps = []
        Vt_maps = []

        for k in range(self.n_shared):
            H = np.zeros((self._u_dim, nvar), dtype=float)
            H[0, : self.admm_shared_ctrl_pts] = self._basis_shared_np[:, k]
            H[1, self.admm_shared_ctrl_pts: 2 * self.admm_shared_ctrl_pts] = self._basis_shared_np[:, k]
            Ush_maps.append(H)
        for k in range(self.n_tail):
            H = np.zeros((self._u_dim, nvar), dtype=float)
            off = 2 * self.admm_shared_ctrl_pts
            H[0, off: off + self.admm_tail_ctrl_pts] = self._basis_tail_np[:, k]
            H[1, off + self.admm_tail_ctrl_pts: off + 2 * self.admm_tail_ctrl_pts] = self._basis_tail_np[:, k]
            Ut_maps.append(H)

        Psh_maps = [np.zeros((2, nvar), dtype=float) for _ in range(self.n_shared)]
        Vsh_maps = [np.zeros((2, nvar), dtype=float) for _ in range(self.n_shared)]
        Pt_maps = [np.zeros((2, nvar), dtype=float) for _ in range(self.n_tail)]
        Vt_maps = [np.zeros((2, nvar), dtype=float) for _ in range(self.n_tail)]

        for idx in range(nvar):
            d = np.zeros((nvar,), dtype=float)
            d[idx] = 1.0
            Csh, Ct = self._unpack_lowdim_ctrl_np(d)
            Xsh, Xt, _, _ = self._rollout_lowdim_ctrl_np(zero_x, Csh, Ct)
            for k in range(self.n_shared):
                Psh_maps[k][:, idx] = Xsh[:2, k + 1]
                Vsh_maps[k][:, idx] = Xsh[2:4, k + 1]
            for k in range(self.n_tail):
                Pt_maps[k][:, idx] = Xt[:2, k + 1]
                Vt_maps[k][:, idx] = Xt[2:4, k + 1]

        self._lowdim_affine_maps = {
            "nvar": int(nvar),
            "Ush": Ush_maps,
            "Ut": Ut_maps,
            "Psh": Psh_maps,
            "Vsh": Vsh_maps,
            "Pt": Pt_maps,
            "Vt": Vt_maps,
        }
        return self._lowdim_affine_maps

    def _lowdim_free_rollouts_np(self, x0):
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Ush0 = np.zeros((self._u_dim, self.n_shared), dtype=float)
        Ut0 = np.zeros((self._u_dim, self.n_tail), dtype=float)
        Xsh0 = self._rollout_controls_np(x0, Ush0)
        Xt0 = self._rollout_controls_np(Xsh0[:, -1], Ut0)
        return Xsh0, Xt0

    @staticmethod
    def _heading_frames_np(hdg):
        hdg = np.asarray(hdg, dtype=float).reshape(-1)
        e = np.vstack([np.cos(hdg), np.sin(hdg)])
        n = np.vstack([-np.sin(hdg), np.cos(hdg)])
        return e, n

    def _linearize_visible_constraints_np(self, pos_seed, visible_obs, seg_offset):
        n_steps = int(max(0, pos_seed.shape[1] - 1))
        nx = np.zeros((n_steps, self.max_visible_obs), dtype=float)
        ny = np.zeros((n_steps, self.max_visible_obs), dtype=float)
        rhs = -1e6 * np.ones((n_steps, self.max_visible_obs), dtype=float)
        active = np.zeros((n_steps, self.max_visible_obs), dtype=float)
        default_n = np.array([1.0, 0.0], dtype=float)

        for j, obs in enumerate(list(visible_obs)[: self.max_visible_obs]):
            o = np.asarray(obs, dtype=float).reshape(-1)
            center0 = o[:2]
            vel = o[3:5] if o.size >= 5 else np.zeros((2,), dtype=float)
            rad = float(o[2]) if o.size >= 3 else 0.0
            for k in range(1, n_steps + 1):
                abs_step = int(seg_offset + k)
                center = center0 + float(self.dt_plan * abs_step) * vel
                inflate = 0.0
                if self.visible_reach_mode == "worst_case":
                    inflate = self.v_visible_max * self.dt_plan * abs_step
                r_eff = rad + self.robot_radius + self.margin_obs + inflate
                q = np.asarray(pos_seed[:, k], dtype=float).reshape(2,)
                rel = q - center
                rel_nrm = float(np.linalg.norm(rel))
                if rel_nrm <= 1e-6:
                    n_hat = default_n.copy()
                else:
                    n_hat = rel / rel_nrm
                nx[k - 1, j] = n_hat[0]
                ny[k - 1, j] = n_hat[1]
                rhs[k - 1, j] = float(n_hat @ center + r_eff)
                active[k - 1, j] = 1.0
        return nx, ny, rhs, active

    @staticmethod
    def _point_segment_distance(point_xy, seg_a_xy, seg_b_xy):
        p = np.asarray(point_xy, dtype=float).reshape(2,)
        a = np.asarray(seg_a_xy, dtype=float).reshape(2,)
        b = np.asarray(seg_b_xy, dtype=float).reshape(2,)
        ab = b - a
        den = float(np.dot(ab, ab))
        if den <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / den)
        t = min(1.0, max(0.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _nearest_occ_scenarios(self, occ_scenarios, x0):
        if occ_scenarios is None or len(occ_scenarios) == 0:
            return []
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        scored = []
        for sc in occ_scenarios:
            c = np.asarray(sc.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
            scored.append((float(np.linalg.norm(c - p)), sc))
        scored.sort(key=lambda item: item[0])
        return [it[1] for it in scored[: self.max_occ_scenarios]]

    def _build_srq_frame(self, scenario):
        c = np.asarray(scenario.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
        v_dir = np.asarray(scenario.get("arc_adv", np.zeros(2)), dtype=float).reshape(-1)
        if v_dir.size >= 2:
            e_long = np.asarray(v_dir[:2], dtype=float)
        else:
            e_long = np.array([1.0, 0.0], dtype=float)
        nrm = float(np.linalg.norm(e_long))
        if nrm <= 1e-9:
            e_long = np.array([1.0, 0.0], dtype=float)
        else:
            e_long = e_long / nrm
        n_lat = np.array([-e_long[1], e_long[0]], dtype=float)

        poly = np.asarray(scenario.get("poly", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        if poly.shape[0] == 0:
            t1 = np.asarray(scenario.get("t1", c), dtype=float).reshape(2,)
            t2 = np.asarray(scenario.get("t2", c), dtype=float).reshape(2,)
            poly = np.vstack([t1, t2])
        rel = poly - c[None, :]
        long_coord = rel @ e_long
        lat_coord = rel @ n_lat
        s_s = float(np.min(long_coord))
        s_e = float(np.max(long_coord))
        lane_width = float(np.clip(2.0 * np.max(np.abs(lat_coord)), self.srq_lane_width_min, self.srq_lane_width_max))
        sigma = float(max(self.srq_lat_sigma_floor, lane_width / max(2.0 * self.srq_confidence_z, 1e-6)))
        peak_grid = np.linspace(s_s, s_e + max(0.0, float(scenario.get("v_adv_max", self.hidden_speed))) * self.srq_horizon, 48)
        peak_r = 0.0
        for s in peak_grid:
            peak_r = max(peak_r, float(self._srq_longitudinal_risk(s, s_s, s_e, float(scenario.get("v_adv_max", self.hidden_speed)))))
        peak_r = max(peak_r, 1e-6)
        return {
            "anchor": c,
            "e_long": e_long,
            "n_lat": n_lat,
            "s_s": s_s,
            "s_e": s_e,
            "lane_width": lane_width,
            "sigma_lat": sigma,
            "v_pv_max": float(max(0.0, scenario.get("v_adv_max", self.hidden_speed))),
            "peak_longitudinal_risk": peak_r,
            "scenario": scenario,
        }

    def _srq_longitudinal_risk(self, s, s_s, s_e, v_pv_max):
        s = float(s)
        s_s = float(s_s)
        s_e = float(max(s_e, s_s + 1e-6))
        T = float(max(self.srq_horizon, 1e-6))
        v_pv_max = float(max(0.0, v_pv_max))
        span = float(max(s_e - s_s, 1e-6))
        i2_hi = s_s + v_pv_max * T
        i3_hi = s_e + v_pv_max * T
        if s < s_s or s > i3_hi:
            return 0.0
        if s <= s_e:
            g = 0.5 * (2.0 * v_pv_max - (s - s_s) / T) * (s - s_s)
        elif s <= i2_hi:
            g = 0.5 * (2.0 * v_pv_max - (s - s_s) / T - (s - s_e) / T) * span
        else:
            g = 0.5 * (v_pv_max - (s - s_e) / T) * (s_e - (s - v_pv_max * T))
        return float(max(0.0, span * g))

    def _srq_lateral_risk(self, d_abs, sigma):
        sigma = float(max(self.srq_lat_sigma_floor, sigma))
        z = float(d_abs) / sigma
        return float(np.exp(-0.5 * z * z))

    def _srq_risk_at_position(self, pos_xy, frame):
        pos_xy = np.asarray(pos_xy, dtype=float).reshape(2,)
        anchor = np.asarray(frame["anchor"], dtype=float).reshape(2,)
        rel = pos_xy - anchor
        s = float(rel @ np.asarray(frame["e_long"], dtype=float).reshape(2,))
        d_abs = float(abs(rel @ np.asarray(frame["n_lat"], dtype=float).reshape(2,)))
        r_lon = self._srq_longitudinal_risk(s, frame["s_s"], frame["s_e"], frame["v_pv_max"])
        r_lat = self._srq_lateral_risk(d_abs, frame["sigma_lat"])
        raw = float(r_lon * r_lat)
        norm = float(raw / max(frame["peak_longitudinal_risk"], 1e-6))
        return raw, norm

    def _aggregate_srq_risk(self, pos_xy, srq_frames):
        if srq_frames is None or len(srq_frames) == 0:
            return 0.0, 0.0, None
        raw_total = 0.0
        norm_total = 0.0
        dominant_idx = None
        dominant_val = -np.inf
        for idx, frame in enumerate(srq_frames):
            raw, norm = self._srq_risk_at_position(pos_xy, frame)
            raw_total += float(raw)
            norm_total += float(norm)
            if norm > dominant_val:
                dominant_val = float(norm)
                dominant_idx = idx
        return (
            float(raw_total),
            float(np.clip(norm_total, 0.0, max(self.srq_risk_clip, 1e-6))),
            dominant_idx,
        )

    def _active_occlusion_context(self, x0, goal_xy, occ_scenarios, nominal_points):
        selected_entries, all_entries = self._select_active_occlusion_entries(
            x0=x0,
            goal_xy=goal_xy,
            occ_scenarios=occ_scenarios,
            nominal_points=nominal_points,
            max_active_occlusions=self.max_active_occlusions,
        )
        srq_frames = []
        phantom_obs = []
        for entry in selected_entries:
            scenario = entry.get("scenario", None)
            if scenario is None:
                continue
            frame = self._build_srq_frame(scenario)
            frame["entry"] = entry
            srq_frames.append(frame)
            for key in ("hidden_t1", "hidden_t2"):
                obs = entry.get(key, None)
                if obs is not None:
                    phantom_obs.append(np.asarray(obs, dtype=float).reshape(-1))
        return selected_entries, all_entries, srq_frames, phantom_obs[: self.max_phantom_obs]

    def _velocity_boundary_from_risk(self, v_ref_nom, risk_value, cth_max):
        v_ref_nom = float(max(0.0, v_ref_nom))
        cth_max = float(max(self.cth_min + 1e-6, cth_max))
        v_occ_max = v_ref_nom
        v_occ_min = float(min(v_occ_max, max(self.v_occ_min_abs, self.v_occ_min_scale * v_ref_nom)))
        if risk_value >= cth_max:
            return v_occ_min
        slope = (v_occ_min - v_occ_max) / max(cth_max - self.cth_min, 1e-6)
        v = slope * (float(risk_value) - self.cth_min) + v_occ_max
        return float(np.clip(v, v_occ_min, v_occ_max))

    def _branch_velocity_boundaries(self, nominal_points, srq_frames, v_ref_nom):
        full_explore = np.full((self.N,), float(max(0.0, v_ref_nom)), dtype=float)
        full_fallback = np.full((self.N,), float(max(0.0, v_ref_nom)), dtype=float)
        risk_vec = np.zeros((self.N,), dtype=float)
        risk_raw_vec = np.zeros((self.N,), dtype=float)
        for k in range(1, min(int(self.N), int(len(nominal_points) - 1)) + 1):
            raw_k, norm_k, _ = self._aggregate_srq_risk(nominal_points[k], srq_frames)
            risk_raw_vec[k - 1] = float(raw_k)
            risk_vec[k - 1] = float(norm_k)
            full_explore[k - 1] = self._velocity_boundary_from_risk(v_ref_nom, norm_k, self.cth_max_explore)
            full_fallback[k - 1] = self._velocity_boundary_from_risk(v_ref_nom, norm_k, self.cth_max_fallback)
        shared = np.minimum(full_explore[: self.n_shared], full_fallback[: self.n_shared]).reshape(1, -1)
        explore = full_explore[self.n_shared:].reshape(1, -1)
        fallback = full_fallback[self.n_shared:].reshape(1, -1)
        return shared, explore, fallback, risk_raw_vec, risk_vec

    def _progress_speed_ca(self, xk, ref_heading):
        e = ca.vertcat(ca.cos(ref_heading), ca.sin(ref_heading))
        return ca.dot(xk[2:4], e)

    def _lateral_speed_ca(self, xk, ref_heading):
        n = ca.vertcat(-ca.sin(ref_heading), ca.cos(ref_heading))
        return ca.dot(xk[2:4], n)

    def _extract_path_points(self, control_ref, x0, goal_xy):
        pts = [np.asarray(x0, dtype=float).reshape(-1)[:2].copy()]
        wps = control_ref.get("waypoints", None)
        goal_index = int(control_ref.get("goal_index", 0) or 0)
        if wps is not None:
            arr = np.asarray(wps, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 2:
                start = min(max(goal_index, 0), max(0, arr.shape[0] - 1))
                for row in arr[start: start + self.max_path_points - 1]:
                    pts.append(np.asarray(row[:2], dtype=float).copy())
        if len(pts) == 1:
            pts.append(np.asarray(goal_xy, dtype=float).reshape(2,).copy())
        cleaned = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(np.asarray(p) - np.asarray(cleaned[-1])) > 1e-4:
                cleaned.append(np.asarray(p, dtype=float))
        if len(cleaned) == 1:
            cleaned.append(np.asarray(goal_xy, dtype=float).reshape(2,).copy())
        return cleaned[: self.max_path_points]

    @staticmethod
    def _polyline_arc_data(path_pts):
        pts = [np.asarray(p, dtype=float).reshape(2,) for p in path_pts]
        seg_lens = []
        cum = [0.0]
        for i in range(len(pts) - 1):
            d = float(np.linalg.norm(pts[i + 1] - pts[i]))
            seg_lens.append(d)
            cum.append(cum[-1] + d)
        return pts, np.asarray(seg_lens, dtype=float), np.asarray(cum, dtype=float)

    @staticmethod
    def _closest_progress_on_polyline(p, pts, cum):
        p = np.asarray(p, dtype=float).reshape(2,)
        best_s = 0.0
        best_d = np.inf
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            den = float(np.dot(ab, ab))
            if den <= 1e-12:
                d = float(np.linalg.norm(p - a))
                s = float(cum[i])
            else:
                t = float(np.dot(p - a, ab) / den)
                t = min(1.0, max(0.0, t))
                q = a + t * ab
                d = float(np.linalg.norm(p - q))
                s = float(cum[i] + t * np.linalg.norm(ab))
            if d < best_d:
                best_d = d
                best_s = s
        return best_s

    @staticmethod
    def _sample_polyline(pts, cum, s_query):
        s = float(np.clip(s_query, cum[0], cum[-1]))
        for i in range(len(pts) - 1):
            s0 = float(cum[i])
            s1 = float(cum[i + 1])
            if s <= s1 + 1e-9:
                seg = pts[i + 1] - pts[i]
                seg_len = max(float(np.linalg.norm(seg)), 1e-9)
                alpha = float(np.clip((s - s0) / seg_len, 0.0, 1.0))
                pos = pts[i] + alpha * seg
                heading = float(np.arctan2(seg[1], seg[0]))
                return pos, heading
        seg = pts[-1] - pts[-2]
        heading = float(np.arctan2(seg[1], seg[0]))
        return pts[-1].copy(), heading

    @staticmethod
    def _heading_from_positions(pos):
        pos = np.asarray(pos, dtype=float)
        n = int(pos.shape[1])
        hdg = np.zeros((1, n), dtype=float)
        if n <= 1:
            return hdg
        diffs = pos[:, 1:] - pos[:, :-1]
        for k in range(n - 1):
            d = np.asarray(diffs[:, k], dtype=float).reshape(2,)
            if float(np.linalg.norm(d)) <= 1e-9:
                hdg[0, k] = hdg[0, max(k - 1, 0)]
            else:
                hdg[0, k] = float(np.arctan2(d[1], d[0]))
        hdg[0, -1] = hdg[0, -2]
        return hdg

    def _sample_polyline_speed_profile(self, pts, cum, s_start, speed_profile):
        speed_profile = np.asarray(speed_profile, dtype=float).reshape(-1)
        count = int(speed_profile.size)
        pos = np.zeros((2, count + 1), dtype=float)
        s = float(s_start)
        for k in range(count + 1):
            pk, _ = self._sample_polyline(pts, cum, s)
            pos[:, k] = pk
            if k < count:
                s += float(speed_profile[k]) * float(self.dt_plan)
        hdg = self._heading_from_positions(pos)
        return pos, hdg

    def _bezier_smooth_reference(self, pos):
        pos = np.asarray(pos, dtype=float)
        if (not self.use_bezier_reference) or pos.shape[1] <= 3:
            return pos.copy(), self._heading_from_positions(pos), None
        n_steps = int(pos.shape[1])
        n_ctrl = min(int(self.bezier_ref_order) + 1, n_steps)
        basis = self._bernstein_basis(n_ctrl, n_steps)
        ctrl = self._fit_ctrl_points_np(pos, basis)
        smooth = ctrl @ basis
        smooth[:, 0] = pos[:, 0]
        smooth[:, -1] = pos[:, -1]
        return smooth, self._heading_from_positions(smooth), ctrl

    def _nominal_route_rollout(self, x0, control_ref, goal_xy, v_ref_nom):
        path_pts = self._extract_path_points(control_ref, x0, goal_xy)
        pts, _, cum = self._polyline_arc_data(path_pts)
        s0 = self._closest_progress_on_polyline(np.asarray(x0, dtype=float).reshape(-1)[:2], pts, cum)
        nominal_speed = np.full((self.N,), float(max(0.0, v_ref_nom)), dtype=float)
        nominal_pos, nominal_hdg = self._sample_polyline_speed_profile(pts, cum, s0, nominal_speed)
        return {
            "path_points": path_pts,
            "pts": pts,
            "cum": cum,
            "s0": float(s0),
            "pos": nominal_pos,
            "hdg": nominal_hdg,
            "speed": nominal_speed.reshape(1, -1),
        }

    def _reference_bundle(self, x0, control_ref, goal_xy, speed_s, speed_e, speed_f):
        path_pts = self._extract_path_points(control_ref, x0, goal_xy)
        pts, _, cum = self._polyline_arc_data(path_pts)
        s0 = self._closest_progress_on_polyline(np.asarray(x0, dtype=float).reshape(-1)[:2], pts, cum)

        speed_s = np.asarray(speed_s, dtype=float).reshape(1, -1)
        speed_e = np.asarray(speed_e, dtype=float).reshape(1, -1)
        speed_f = np.asarray(speed_f, dtype=float).reshape(1, -1)

        pos_s_raw, _ = self._sample_polyline_speed_profile(pts, cum, s0, speed_s.reshape(-1))
        s_shared_end = float(s0 + np.sum(speed_s) * float(self.dt_plan))
        pos_e_raw, _ = self._sample_polyline_speed_profile(pts, cum, s_shared_end, speed_e.reshape(-1))
        pos_f_raw, _ = self._sample_polyline_speed_profile(pts, cum, s_shared_end, speed_f.reshape(-1))

        pos_s, hdg_s, ctrl_s = self._bezier_smooth_reference(pos_s_raw)
        pos_e, hdg_e, ctrl_e = self._bezier_smooth_reference(pos_e_raw)
        pos_f, hdg_f, ctrl_f = self._bezier_smooth_reference(pos_f_raw)
        return {
            "path_points": path_pts,
            "shared": (pos_s, hdg_s, speed_s),
            "explore": (pos_e, hdg_e, speed_e),
            "fallback": (pos_f, hdg_f, speed_f),
            "shared_ctrl_pts": ctrl_s,
            "explore_ctrl_pts": ctrl_e,
            "fallback_ctrl_pts": ctrl_f,
        }

    def _build_coupled_solver(self):
        opti = ca.Opti()

        nx = self._n_state
        nu = self._u_dim
        Ns = self.n_shared
        Nt = self.n_tail
        Mv = self.max_visible_obs
        Mh = self.max_phantom_obs

        # Decision variables
        Xs = opti.variable(nx, Ns + 1)
        Us = opti.variable(nu, Ns)
        Xe = opti.variable(nx, Nt + 1)
        Ue = opti.variable(nu, Nt)
        Xf = opti.variable(nx, Nt + 1)
        Uf = opti.variable(nu, Nt)
        Sshared = opti.variable(1, Ns)
        Se = opti.variable(1, Nt)
        Sf = opti.variable(1, Nt)

        # Parameters
        x0_p = opti.parameter(nx, 1)
        u_prev_p = opti.parameter(nu, 1)
        goal_p = opti.parameter(2, 1)

        ref_pos_s = opti.parameter(2, Ns + 1)
        ref_hdg_s = opti.parameter(1, Ns + 1)
        ref_speed_s = opti.parameter(1, Ns)
        ref_pos_e = opti.parameter(2, Nt + 1)
        ref_hdg_e = opti.parameter(1, Nt + 1)
        ref_speed_e = opti.parameter(1, Nt)
        ref_pos_f = opti.parameter(2, Nt + 1)
        ref_hdg_f = opti.parameter(1, Nt + 1)
        ref_speed_f = opti.parameter(1, Nt)

        shared_bound_p = opti.parameter(1, Ns)
        explore_bound_p = opti.parameter(1, Nt)
        fallback_bound_p = opti.parameter(1, Nt)

        obs_center_p = opti.parameter(2, Mv)
        obs_vel_p = opti.parameter(2, Mv)
        obs_rad_p = opti.parameter(1, Mv)
        obs_active_p = opti.parameter(1, Mv)

        hid_center_p = opti.parameter(2, Mh)
        hid_vel_p = opti.parameter(2, Mh)
        hid_rad_p = opti.parameter(1, Mh)
        hid_active_p = opti.parameter(1, Mh)

        lb_u, ub_u = self._input_bounds()
        v_min, v_max = self._speed_bounds()

        big_inactive = 100.0
        objective = 0

        opti.subject_to(Xs[:, 0] == x0_p)
        opti.subject_to(Xe[:, 0] == Xs[:, Ns])
        opti.subject_to(Xf[:, 0] == Xs[:, Ns])
        opti.subject_to(Sshared >= 0)
        opti.subject_to(Se >= 0)
        opti.subject_to(Sf >= 0)

        def stage_speed(X, U, k):
            return self._stage_speed_ca(X, U, k)

        def apply_state_bounds(X):
            if self.model == "DynamicUnicycle2D":
                for k in range(X.shape[1]):
                    opti.subject_to(X[3, k] <= v_max)
                    opti.subject_to(X[3, k] >= (0.0 if self.du_nonnegative_speed else v_min))
            elif self.model == "DoubleIntegrator2D":
                for k in range(X.shape[1]):
                    opti.subject_to(ca.sumsqr(X[2:4, k]) <= (v_max + 1e-6) ** 2)

        def add_input_bounds(U):
            for k in range(U.shape[1]):
                opti.subject_to(U[:, k] <= ub_u)
                opti.subject_to(U[:, k] >= lb_u)

        apply_state_bounds(Xs)
        apply_state_bounds(Xe)
        apply_state_bounds(Xf)
        add_input_bounds(Us)
        add_input_bounds(Ue)
        add_input_bounds(Uf)

        for k in range(Ns):
            opti.subject_to(Xs[:, k + 1] == self._discrete_ca(Xs[:, k], Us[:, k]))
        for k in range(Nt):
            opti.subject_to(Xe[:, k + 1] == self._discrete_ca(Xe[:, k], Ue[:, k]))
            opti.subject_to(Xf[:, k + 1] == self._discrete_ca(Xf[:, k], Uf[:, k]))

        def _barrier_alpha(abs_step):
            if self.N <= 1:
                return float(self.barrier_alpha_end)
            frac = float(np.clip((float(abs_step) - 1.0) / max(float(self.N - 1), 1e-6), 0.0, 1.0))
            return float(self.barrier_alpha_start + frac * (self.barrier_alpha_end - self.barrier_alpha_start))

        def _barrier_h(pos_xy, center_xy, r_eff):
            rx = max(1e-6, float(self.ellipse_scale_x)) * (
                self.robot_radius + r_eff + float(self.ellipse_buffer_x)
            )
            ry = max(1e-6, float(self.ellipse_scale_y)) * (
                self.robot_radius + r_eff + float(self.ellipse_buffer_y)
            )
            dx = (pos_xy[0] - center_xy[0]) / rx
            dy = (pos_xy[1] - center_xy[1]) / ry
            return dx * dx + dy * dy - 1.0

        def add_barrier_bank(prev_pos, curr_pos, abs_prev, abs_curr, slack_stage, center_p, vel_p, rad_p, active_p, n_obs, inflate_visible):
            for j in range(n_obs):
                center_prev = center_p[:, j] + float(self.dt_plan * abs_prev) * vel_p[:, j]
                center_curr = center_p[:, j] + float(self.dt_plan * abs_curr) * vel_p[:, j]
                rad_prev = rad_p[0, j]
                rad_curr = rad_p[0, j]
                if inflate_visible and self.visible_reach_mode == "worst_case":
                    rad_prev = rad_prev + float(self.v_visible_max) * float(self.dt_plan) * float(abs_prev)
                    rad_curr = rad_curr + float(self.v_visible_max) * float(self.dt_plan) * float(abs_curr)
                h_prev = _barrier_h(prev_pos, center_prev, rad_prev)
                h_curr = _barrier_h(curr_pos, center_curr, rad_curr)
                active = active_p[0, j]
                alpha = _barrier_alpha(abs_curr)
                opti.subject_to(h_curr >= -slack_stage - big_inactive * (1.0 - active))
                opti.subject_to(
                    h_curr - (1.0 - alpha) * h_prev >= -slack_stage - big_inactive * (1.0 - active)
                )

        def _pos_track_cost(pos_xy, ref_xy, ref_heading, pos_w, cross_scale):
            pos_err = pos_xy - ref_xy
            if self.model != "DoubleIntegrator2D":
                return pos_w * ca.sumsqr(pos_err)
            e_ref = ca.vertcat(ca.cos(ref_heading), ca.sin(ref_heading))
            n_ref = ca.vertcat(-ca.sin(ref_heading), ca.cos(ref_heading))
            along_err = ca.dot(pos_err, e_ref)
            cross_err = ca.dot(pos_err, n_ref)
            return pos_w * ((along_err ** 2) + float(max(0.0, cross_scale)) * (cross_err ** 2))

        def add_branch_costs(X, U, ref_pos, ref_hdg, ref_speed, pos_w, hdg_w, speed_w, slack_w, speed_bound, slack_var, seg_offset):
            J = 0
            u_prev_local = u_prev_p
            n_steps = U.shape[1]
            for k in range(n_steps):
                xk0 = X[:, k]
                xk1 = X[:, k + 1]
                uk = U[:, k]
                hdg_err = self._heading_error_ca(xk1, ref_hdg[0, k + 1])
                if self.model == "DoubleIntegrator2D":
                    hdg_w_eff = 0.0
                else:
                    hdg_w_eff = hdg_w * self._heading_track_weight_ca(xk1)
                if self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost:
                    v_track = self._progress_speed_ca(xk1, ref_hdg[0, k + 1])
                    v_lat = self._lateral_speed_ca(xk1, ref_hdg[0, k + 1])
                else:
                    v_track = stage_speed(X, U, k)
                    v_lat = 0.0
                v_err = v_track - ref_speed[0, k]
                du = uk - u_prev_local
                J += _pos_track_cost(
                    xk1[:2],
                    ref_pos[:, k + 1],
                    ref_hdg[0, k + 1],
                    pos_w,
                    self.di_cross_track_scale_branch,
                )
                if self.model != "DoubleIntegrator2D":
                    J += hdg_w_eff * (hdg_err ** 2)
                J += speed_w * (v_err ** 2)
                if self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost:
                    J += speed_w * self.di_lateral_velocity_weight * (v_lat ** 2)
                J += self.w_u * ca.sumsqr(uk)
                J += self.w_du * ca.sumsqr(du)
                u_prev_local = uk

                if self.model == "DynamicUnicycle2D":
                    opti.subject_to(X[3, k + 1] <= speed_bound[0, k] + 1e-6)
                elif self.model == "DoubleIntegrator2D":
                    v_progress = self._progress_speed_ca(xk1, ref_hdg[0, k + 1])
                    opti.subject_to(v_progress <= speed_bound[0, k] + 1e-6)
                else:
                    opti.subject_to(U[0, k] <= speed_bound[0, k] + 1e-6)
                    if self.forward_only:
                        opti.subject_to(U[0, k] >= 0.0)

                abs_prev = seg_offset + k
                abs_curr = seg_offset + k + 1
                add_barrier_bank(xk0[:2], xk1[:2], abs_prev, abs_curr, slack_var[0, k], obs_center_p, obs_vel_p, obs_rad_p, obs_active_p, Mv, True)
                add_barrier_bank(xk0[:2], xk1[:2], abs_prev, abs_curr, slack_var[0, k], hid_center_p, hid_vel_p, hid_rad_p, hid_active_p, Mh, False)
                J += slack_w * (slack_var[0, k] ** 2)

            term_pos = X[:2, -1] - goal_p
            J += self.w_terminal_goal * ca.sumsqr(term_pos)
            return J

        def add_shared_costs():
            J = 0
            u_prev_local = u_prev_p
            for k in range(Ns):
                xk1 = Xs[:, k + 1]
                uk = Us[:, k]
                hdg_err = self._heading_error_ca(xk1, ref_hdg_s[0, k + 1])
                if self.model == "DoubleIntegrator2D":
                    hdg_w_eff = 0.0
                else:
                    hdg_w_eff = self.w_heading_shared * self._heading_track_weight_ca(xk1)
                if self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost:
                    v_track = self._progress_speed_ca(xk1, ref_hdg_s[0, k + 1])
                    v_lat = self._lateral_speed_ca(xk1, ref_hdg_s[0, k + 1])
                else:
                    v_track = stage_speed(Xs, Us, k)
                    v_lat = 0.0
                v_err = v_track - ref_speed_s[0, k]
                du = uk - u_prev_local
                J += _pos_track_cost(
                    xk1[:2],
                    ref_pos_s[:, k + 1],
                    ref_hdg_s[0, k + 1],
                    self.w_pos_shared,
                    self.di_cross_track_scale_shared,
                )
                if self.model != "DoubleIntegrator2D":
                    J += hdg_w_eff * (hdg_err ** 2)
                J += self.w_speed_shared * (v_err ** 2)
                if self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost:
                    J += self.w_speed_shared * self.di_lateral_velocity_weight * (v_lat ** 2)
                J += self.w_u * ca.sumsqr(uk)
                J += self.w_du * ca.sumsqr(du)
                u_prev_local = uk

                if self.model == "DynamicUnicycle2D":
                    opti.subject_to(Xs[3, k + 1] <= shared_bound_p[0, k] + 1e-6)
                elif self.model == "DoubleIntegrator2D":
                    v_progress = self._progress_speed_ca(xk1, ref_hdg_s[0, k + 1])
                    opti.subject_to(v_progress <= shared_bound_p[0, k] + 1e-6)
                else:
                    opti.subject_to(Us[0, k] <= shared_bound_p[0, k] + 1e-6)
                    if self.forward_only:
                        opti.subject_to(Us[0, k] >= 0.0)

                add_barrier_bank(Xs[:2, k], xk1[:2], k, k + 1, Sshared[0, k], obs_center_p, obs_vel_p, obs_rad_p, obs_active_p, Mv, True)
                add_barrier_bank(Xs[:2, k], xk1[:2], k, k + 1, Sshared[0, k], hid_center_p, hid_vel_p, hid_rad_p, hid_active_p, Mh, False)
                J += self.w_barrier_slack_shared * (Sshared[0, k] ** 2)
            return J

        objective += add_shared_costs()
        objective += add_branch_costs(
            Xe, Ue, ref_pos_e, ref_hdg_e, ref_speed_e,
            self.w_pos_explore, self.w_heading_explore, self.w_speed_explore,
            self.w_barrier_slack_explore, explore_bound_p, Se, Ns,
        )
        objective += add_branch_costs(
            Xf, Uf, ref_pos_f, ref_hdg_f, ref_speed_f,
            self.w_pos_fallback, self.w_heading_fallback, self.w_speed_fallback,
            self.w_barrier_slack_fallback, fallback_bound_p, Sf, Ns,
        )

        opti.minimize(objective)
        opts = {
            "expand": False,
            "print_time": bool(self.print_solver),
            "ipopt.print_level": 5 if self.print_solver else 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": int(self.max_iter),
            "ipopt.tol": float(self.solver_tol),
            "ipopt.acceptable_tol": float(self.solver_acceptable_tol),
            "ipopt.acceptable_iter": 8,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        self._opti = opti
        self._vars = {
            "Xs": Xs, "Us": Us, "Xe": Xe, "Ue": Ue, "Xf": Xf, "Uf": Uf,
            "Sshared": Sshared, "Se": Se, "Sf": Sf,
        }
        self._pars = {
            "x0": x0_p, "u_prev": u_prev_p, "goal": goal_p,
            "ref_pos_s": ref_pos_s, "ref_hdg_s": ref_hdg_s, "ref_speed_s": ref_speed_s,
            "ref_pos_e": ref_pos_e, "ref_hdg_e": ref_hdg_e, "ref_speed_e": ref_speed_e,
            "ref_pos_f": ref_pos_f, "ref_hdg_f": ref_hdg_f, "ref_speed_f": ref_speed_f,
            "shared_bound": shared_bound_p, "explore_bound": explore_bound_p, "fallback_bound": fallback_bound_p,
            "obs_center": obs_center_p, "obs_vel": obs_vel_p, "obs_rad": obs_rad_p, "obs_active": obs_active_p,
            "hid_center": hid_center_p, "hid_vel": hid_vel_p, "hid_rad": hid_rad_p, "hid_active": hid_active_p,
        }

    def _build_lowdim_branch_solver(self, branch_name):
        return {"branch": branch_name, "cache": {}, "warm_start": None}

    def _build_lowdim_admm_solvers(self):
        self._build_lowdim_affine_maps()
        self._admm_solvers = {
            "explore": self._build_lowdim_branch_solver("explore"),
            "fallback": self._build_lowdim_branch_solver("fallback"),
        }

    def _reference_control_sequence_np(self, x_init, ref_hdg, ref_speed):
        x_init = np.asarray(x_init, dtype=float).reshape(-1)
        n_steps = int(ref_speed.shape[1])
        U = np.zeros((self._u_dim, n_steps), dtype=float)
        if self.model != "DoubleIntegrator2D":
            u0 = np.zeros((self._u_dim,), dtype=float)
            return np.tile(u0.reshape(-1, 1), (1, n_steps))
        v_cur = x_init[2:4].copy()
        lb_u, ub_u = self._input_bounds()
        for k in range(n_steps):
            hdg = float(ref_hdg[0, min(k + 1, ref_hdg.shape[1] - 1)])
            spd = float(ref_speed[0, k])
            v_des = spd * np.array([np.cos(hdg), np.sin(hdg)], dtype=float)
            a = (v_des - v_cur) / max(self.dt_plan, 1e-6)
            a = np.clip(a, lb_u, ub_u)
            U[:, k] = a
            v_cur = v_cur + self.dt_plan * a
            v_mag = float(np.linalg.norm(v_cur))
            _, v_max = self._speed_bounds()
            if np.isfinite(v_max) and v_mag > max(v_max, 1e-9):
                v_cur *= float(v_max) / v_mag
        return U

    def _seed_lowdim_controls(self, x0, u_ref, refs):
        if isinstance(self._admm_prev, dict):
            z0 = np.asarray(self._admm_prev.get("z", np.zeros((self._u_dim, self.admm_shared_ctrl_pts))), dtype=float)
            ce0 = np.asarray(self._admm_prev.get("Ct_explore", np.zeros((self._u_dim, self.admm_tail_ctrl_pts))), dtype=float)
            cf0 = np.asarray(self._admm_prev.get("Ct_fallback", np.zeros((self._u_dim, self.admm_tail_ctrl_pts))), dtype=float)
            if z0.shape == (self._u_dim, self.admm_shared_ctrl_pts) and ce0.shape == (self._u_dim, self.admm_tail_ctrl_pts) and cf0.shape == (self._u_dim, self.admm_tail_ctrl_pts):
                return z0, ce0, cf0

        _, hdg_s, speed_s = refs["shared"]
        _, hdg_e, speed_e = refs["explore"]
        _, hdg_f, speed_f = refs["fallback"]
        U_shared = self._reference_control_sequence_np(x0, hdg_s, speed_s)
        z0 = self._fit_ctrl_points_np(U_shared, self._basis_shared_np)
        Xs0 = self._rollout_controls_np(x0, self._expand_ctrl_points_np(z0, self._basis_shared_np))
        Ue = self._reference_control_sequence_np(Xs0[:, -1], hdg_e, speed_e)
        Uf = self._reference_control_sequence_np(Xs0[:, -1], hdg_f, speed_f)
        Ce0 = self._fit_ctrl_points_np(Ue, self._basis_tail_np)
        Cf0 = self._fit_ctrl_points_np(Uf, self._basis_tail_np)
        return z0, Ce0, Cf0

    def _set_lowdim_branch_params(self, branch_data, x0, u_prev, goal_xy, pos_s, hdg_s, speed_s, pos_t, hdg_t, speed_t, shared_cap, tail_cap, z, y, visible_obs, csh_seed, ct_seed):
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Xsh_free, Xt_free = self._lowdim_free_rollouts_np(x0)
        e_s, n_s = self._heading_frames_np(hdg_s[0, 1:])
        e_t, n_t = self._heading_frames_np(hdg_t[0, 1:])
        Xsh_seed, Xt_seed, _, _ = self._rollout_lowdim_ctrl_np(x0, csh_seed, ct_seed)
        vis_nx_sh, vis_ny_sh, vis_rhs_sh, _ = self._linearize_visible_constraints_np(Xsh_seed[:2], visible_obs, 0)
        vis_nx_t, vis_ny_t, vis_rhs_t, _ = self._linearize_visible_constraints_np(Xt_seed[:2], visible_obs, self.n_shared)
        branch_data["cache"] = {
            "x0": x0.copy(),
            "u_prev": np.asarray(u_prev, dtype=float).reshape(self._u_dim,),
            "goal": np.asarray(goal_xy, dtype=float).reshape(2,),
            "psh_free": np.asarray(Xsh_free[:2, 1:], dtype=float),
            "vsh_free": np.asarray(Xsh_free[2:4, 1:], dtype=float),
            "pt_free": np.asarray(Xt_free[:2, 1:], dtype=float),
            "vt_free": np.asarray(Xt_free[2:4, 1:], dtype=float),
            "ref_pos_s": np.asarray(pos_s[:, 1:], dtype=float),
            "ref_speed_s": np.asarray(speed_s.reshape(-1), dtype=float),
            "e_s": np.asarray(e_s, dtype=float),
            "n_s": np.asarray(n_s, dtype=float),
            "ref_pos_t": np.asarray(pos_t[:, 1:], dtype=float),
            "ref_speed_t": np.asarray(speed_t.reshape(-1), dtype=float),
            "e_t": np.asarray(e_t, dtype=float),
            "n_t": np.asarray(n_t, dtype=float),
            "shared_cap": float(shared_cap),
            "tail_cap": float(tail_cap),
            "z": np.asarray(z, dtype=float).reshape(-1),
            "y": np.asarray(y, dtype=float).reshape(-1),
            "vis_nx_sh": np.asarray(vis_nx_sh, dtype=float),
            "vis_ny_sh": np.asarray(vis_ny_sh, dtype=float),
            "vis_rhs_sh": np.asarray(vis_rhs_sh, dtype=float),
            "vis_nx_t": np.asarray(vis_nx_t, dtype=float),
            "vis_ny_t": np.asarray(vis_ny_t, dtype=float),
            "vis_rhs_t": np.asarray(vis_rhs_t, dtype=float),
        }

    def _solve_lowdim_branch(self, branch_data, init_csh, init_ct):
        if osqp is None or sp is None:
            raise RuntimeError("osqp and scipy.sparse are required for low-dimensional OACP.")

        cache = dict(branch_data.get("cache", {}))
        if not cache:
            raise RuntimeError("Low-dimensional branch cache is empty.")

        maps = self._build_lowdim_affine_maps()
        nvar = int(maps["nvar"])
        n_sh = int(2 * self.admm_shared_ctrl_pts)
        Ns = self.n_shared
        Nt = self.n_tail
        Mv = self.max_visible_obs
        nss = int(Ns * Mv)
        nst = int(Nt * Mv)
        n = int(nvar + nss + nst)
        lb_u, ub_u = self._input_bounds()

        def _slack_sh_idx(k, j):
            return int(nvar + k * Mv + j)

        def _slack_t_idx(k, j):
            return int(nvar + nss + k * Mv + j)

        P = np.zeros((n, n), dtype=float)
        q = np.zeros((n,), dtype=float)

        if branch_data["branch"] == "explore":
            pos_w = self.w_pos_explore
            speed_w = self.w_speed_explore
            vis_slack_w = self.w_vis_slack_explore
        else:
            pos_w = self.w_pos_fallback
            speed_w = self.w_speed_fallback
            vis_slack_w = self.w_vis_slack_fallback

        for k in range(Ns):
            Ud = np.asarray(maps["Ush"][k], dtype=float)
            Pd = np.asarray(maps["Psh"][k], dtype=float)
            Vd = np.asarray(maps["Vsh"][k], dtype=float)
            p_free = np.asarray(cache["psh_free"][:, k], dtype=float)
            v_free = np.asarray(cache["vsh_free"][:, k], dtype=float)
            p_ref = np.asarray(cache["ref_pos_s"][:, k], dtype=float)
            e = np.asarray(cache["e_s"][:, k], dtype=float)
            nvec = np.asarray(cache["n_s"][:, k], dtype=float)
            v_ref = float(cache["ref_speed_s"][k])

            P[:nvar, :nvar] += self.w_pos_shared * (Pd.T @ Pd)
            q[:nvar] += self.w_pos_shared * (Pd.T @ (p_free - p_ref))

            r = e @ Vd
            c = float(e @ v_free - v_ref)
            P[:nvar, :nvar] += self.w_speed_shared * np.outer(r, r)
            q[:nvar] += self.w_speed_shared * c * r

            r = nvec @ Vd
            c = float(nvec @ v_free)
            w_lat = self.w_speed_shared * self.di_lateral_velocity_weight
            P[:nvar, :nvar] += w_lat * np.outer(r, r)
            q[:nvar] += w_lat * c * r

            P[:nvar, :nvar] += self.w_u * (Ud.T @ Ud)

            if k == 0:
                cvec = -np.asarray(cache["u_prev"], dtype=float).reshape(self._u_dim,)
                P[:nvar, :nvar] += self.w_du * (Ud.T @ Ud)
                q[:nvar] += self.w_du * (Ud.T @ cvec)
            else:
                Dd = Ud - np.asarray(maps["Ush"][k - 1], dtype=float)
                P[:nvar, :nvar] += self.w_du * (Dd.T @ Dd)

        for k in range(Nt):
            Ud = np.asarray(maps["Ut"][k], dtype=float)
            Pd = np.asarray(maps["Pt"][k], dtype=float)
            Vd = np.asarray(maps["Vt"][k], dtype=float)
            p_free = np.asarray(cache["pt_free"][:, k], dtype=float)
            v_free = np.asarray(cache["vt_free"][:, k], dtype=float)
            p_ref = np.asarray(cache["ref_pos_t"][:, k], dtype=float)
            e = np.asarray(cache["e_t"][:, k], dtype=float)
            nvec = np.asarray(cache["n_t"][:, k], dtype=float)
            v_ref = float(cache["ref_speed_t"][k])

            P[:nvar, :nvar] += 2.0 * pos_w * (Pd.T @ Pd)
            q[:nvar] += 2.0 * pos_w * (Pd.T @ (p_free - p_ref))

            r = e @ Vd
            c = float(e @ v_free - v_ref)
            P[:nvar, :nvar] += 2.0 * speed_w * np.outer(r, r)
            q[:nvar] += 2.0 * speed_w * c * r

            r = nvec @ Vd
            c = float(nvec @ v_free)
            w_lat = 2.0 * speed_w * self.di_lateral_velocity_weight
            P[:nvar, :nvar] += w_lat * np.outer(r, r)
            q[:nvar] += w_lat * c * r

            P[:nvar, :nvar] += 2.0 * self.w_u * (Ud.T @ Ud)

            if k == 0:
                cvec = -np.asarray(cache["u_prev"], dtype=float).reshape(self._u_dim,)
                P[:nvar, :nvar] += 2.0 * self.w_du * (Ud.T @ Ud)
                q[:nvar] += 2.0 * self.w_du * (Ud.T @ cvec)
            else:
                Dd = Ud - np.asarray(maps["Ut"][k - 1], dtype=float)
                P[:nvar, :nvar] += 2.0 * self.w_du * (Dd.T @ Dd)

        if Nt > 0:
            Pd = np.asarray(maps["Pt"][-1], dtype=float)
            p_free = np.asarray(cache["pt_free"][:, -1], dtype=float)
            goal = np.asarray(cache["goal"], dtype=float).reshape(2,)
            P[:nvar, :nvar] += self.w_terminal_goal * (Pd.T @ Pd)
            q[:nvar] += self.w_terminal_goal * (Pd.T @ (p_free - goal))

        P[:n_sh, :n_sh] += self.admm_rho * np.eye(n_sh, dtype=float)
        q[:n_sh] += self.admm_rho * (np.asarray(cache["y"], dtype=float).reshape(-1) - np.asarray(cache["z"], dtype=float).reshape(-1))

        for idx in range(nvar, nvar + nss):
            P[idx, idx] = self.w_vis_slack_shared
        for idx in range(nvar + nss, n):
            P[idx, idx] = vis_slack_w

        rows = []
        l = []
        u = []

        def _append_row(d_coeff=None, slack_idx=None, lower=-np.inf, upper=np.inf):
            row = np.zeros((n,), dtype=float)
            if d_coeff is not None:
                row[:nvar] = np.asarray(d_coeff, dtype=float).reshape(-1)
            if slack_idx is not None:
                row[int(slack_idx)] = 1.0
            rows.append(row)
            l.append(float(lower))
            u.append(float(upper))

        for k in range(Ns):
            Ud = np.asarray(maps["Ush"][k], dtype=float)
            Vd = np.asarray(maps["Vsh"][k], dtype=float)
            v_free = np.asarray(cache["vsh_free"][:, k], dtype=float)
            e = np.asarray(cache["e_s"][:, k], dtype=float)
            nvec = np.asarray(cache["n_s"][:, k], dtype=float)
            shared_cap = float(cache["shared_cap"])
            vtrack_d = e @ Vd
            vtrack_c = float(e @ v_free)
            vlat_d = nvec @ Vd
            vlat_c = float(nvec @ v_free)
            for r in range(self._u_dim):
                _append_row(Ud[r, :], lower=float(lb_u[r]), upper=float(ub_u[r]))
            _append_row(vtrack_d, lower=-vtrack_c, upper=shared_cap - vtrack_c + 1e-6)
            lat_cap = self.di_lateral_speed_cap_scale * shared_cap + 1e-6
            _append_row(vlat_d, lower=-lat_cap - vlat_c, upper=lat_cap - vlat_c)
            for j in range(Mv):
                d_coeff = cache["vis_nx_sh"][k, j] * np.asarray(maps["Psh"][k][0, :], dtype=float) + cache["vis_ny_sh"][k, j] * np.asarray(maps["Psh"][k][1, :], dtype=float)
                c0 = float(cache["vis_nx_sh"][k, j] * cache["psh_free"][0, k] + cache["vis_ny_sh"][k, j] * cache["psh_free"][1, k])
                rhs = float(cache["vis_rhs_sh"][k, j] - c0)
                _append_row(d_coeff, slack_idx=_slack_sh_idx(k, j), lower=rhs, upper=np.inf)
                _append_row(slack_idx=_slack_sh_idx(k, j), lower=0.0, upper=np.inf)

        for k in range(Nt):
            Ud = np.asarray(maps["Ut"][k], dtype=float)
            Vd = np.asarray(maps["Vt"][k], dtype=float)
            v_free = np.asarray(cache["vt_free"][:, k], dtype=float)
            e = np.asarray(cache["e_t"][:, k], dtype=float)
            nvec = np.asarray(cache["n_t"][:, k], dtype=float)
            tail_cap = float(cache["tail_cap"])
            vtrack_d = e @ Vd
            vtrack_c = float(e @ v_free)
            vlat_d = nvec @ Vd
            vlat_c = float(nvec @ v_free)
            for r in range(self._u_dim):
                _append_row(Ud[r, :], lower=float(lb_u[r]), upper=float(ub_u[r]))
            _append_row(vtrack_d, lower=-vtrack_c, upper=tail_cap - vtrack_c + 1e-6)
            lat_cap = self.di_lateral_speed_cap_scale * tail_cap + 1e-6
            _append_row(vlat_d, lower=-lat_cap - vlat_c, upper=lat_cap - vlat_c)
            for j in range(Mv):
                d_coeff = cache["vis_nx_t"][k, j] * np.asarray(maps["Pt"][k][0, :], dtype=float) + cache["vis_ny_t"][k, j] * np.asarray(maps["Pt"][k][1, :], dtype=float)
                c0 = float(cache["vis_nx_t"][k, j] * cache["pt_free"][0, k] + cache["vis_ny_t"][k, j] * cache["pt_free"][1, k])
                rhs = float(cache["vis_rhs_t"][k, j] - c0)
                _append_row(d_coeff, slack_idx=_slack_t_idx(k, j), lower=rhs, upper=np.inf)
                _append_row(slack_idx=_slack_t_idx(k, j), lower=0.0, upper=np.inf)

        P = sp.csc_matrix(np.triu(P))
        A = sp.csc_matrix(np.vstack(rows))
        l = np.asarray(l, dtype=float)
        u = np.asarray(u, dtype=float)

        d0 = self._flatten_lowdim_ctrl_np(init_csh, init_ct)
        t0 = time.perf_counter()
        solver = osqp.OSQP()
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=bool(self.print_solver),
            polish=True,
            warm_start=True,
            eps_abs=float(max(self.solver_tol, 1e-4)),
            eps_rel=float(max(self.solver_tol, 1e-4)),
            max_iter=4000,
        )
        warm = np.zeros((n,), dtype=float)
        warm[:nvar] = np.asarray(d0, dtype=float).reshape(-1)
        if isinstance(branch_data.get("warm_start"), np.ndarray) and branch_data["warm_start"].shape == (n,):
            warm = np.asarray(branch_data["warm_start"], dtype=float).reshape(-1)
        solver.warm_start(x=warm)
        res = solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        status = str(getattr(res.info, "status", "")).lower()
        if res.x is None or ("solved" not in status):
            raise RuntimeError(f"Low-dimensional QP solve failed with status `{getattr(res.info, 'status', 'unknown')}`")

        x_val = np.asarray(res.x, dtype=float).reshape(-1)
        branch_data["warm_start"] = x_val.copy()
        d_val = x_val[:nvar]
        Csh, Ct = self._unpack_lowdim_ctrl_np(d_val)
        x0 = np.asarray(branch_data.get("cache", {}).get("x0", np.zeros((self._n_state,), dtype=float)), dtype=float).reshape(-1)
        Xsh, Xt, Ush, Ut = self._rollout_lowdim_ctrl_np(x0, Csh, Ct)
        Ssh = x_val[nvar: nvar + nss].reshape(Ns, Mv)
        St = x_val[nvar + nss:].reshape(Nt, Mv)
        return {
            "solve_ms": float(solve_ms),
            "Csh": np.asarray(Csh, dtype=float),
            "Ct": np.asarray(Ct, dtype=float),
            "Ssh": np.asarray(Ssh, dtype=float),
            "St": np.asarray(St, dtype=float),
            "Ush": np.asarray(Ush, dtype=float),
            "Ut": np.asarray(Ut, dtype=float),
            "Xsh": np.asarray(Xsh, dtype=float),
            "Xt": np.asarray(Xt, dtype=float),
            "objective": float(0.5 * x_val @ (P @ x_val) + q @ x_val),
        }

    def _set_obs_params(self, visible_obs):
        centers = np.zeros((2, self.max_visible_obs), dtype=float)
        vels = np.zeros((2, self.max_visible_obs), dtype=float)
        radii = np.zeros((1, self.max_visible_obs), dtype=float)
        active = np.zeros((1, self.max_visible_obs), dtype=float)
        for j, obs in enumerate(list(visible_obs)[: self.max_visible_obs]):
            o = np.asarray(obs, dtype=float).reshape(-1)
            centers[:, j] = o[:2]
            radii[0, j] = float(o[2]) if o.size >= 3 else 0.0
            if o.size >= 5:
                vels[:, j] = o[3:5]
            active[0, j] = 1.0
        return centers, vels, radii, active

    def _set_hidden_params(self, hidden_obs):
        centers = np.zeros((2, self.max_phantom_obs), dtype=float)
        vels = np.zeros((2, self.max_phantom_obs), dtype=float)
        radii = np.zeros((1, self.max_phantom_obs), dtype=float)
        active = np.zeros((1, self.max_phantom_obs), dtype=float)
        for j, obs in enumerate(list(hidden_obs)[: self.max_phantom_obs]):
            o = np.asarray(obs, dtype=float).reshape(-1)
            centers[:, j] = o[:2]
            radii[0, j] = float(o[2]) if o.size >= 3 else 0.0
            if o.size >= 5:
                vels[:, j] = o[3:5]
            active[0, j] = 1.0
        return centers, vels, radii, active

    def _trajectory_min_clearance(self, X, obs_list, seg_offset):
        if obs_list is None or len(obs_list) == 0:
            return None
        X = np.asarray(X, dtype=float)
        min_clear = float("inf")
        n_steps = int(max(0, X.shape[1] - 1))
        for k in range(1, n_steps + 1):
            pos = np.asarray(X[:2, k], dtype=float).reshape(2,)
            abs_step = int(seg_offset + k)
            for obs in obs_list:
                obs = np.asarray(obs, dtype=float).reshape(-1)
                c = self._predict_obs_center(obs, abs_step)
                clear = float(np.linalg.norm(pos - c) - (self.robot_radius + float(obs[2]) + self.margin_obs))
                min_clear = min(min_clear, clear)
        return None if not np.isfinite(min_clear) else float(min_clear)

    def _branch_diag(self, X, U, goal_xy, path_pts, seg_offset, slack, visible_obs, hidden_obs):
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        term = np.asarray(X[:2, -1], dtype=float).reshape(2,)
        goal_err = float(np.linalg.norm(term - goal_xy))
        pts, _, cum = self._polyline_arc_data(path_pts)
        progress = float(self._closest_progress_on_polyline(term, pts, cum))
        effort = float(np.sum(np.square(np.asarray(U, dtype=float))))
        slack = float(slack)
        min_visible_clear = self._trajectory_min_clearance(X, visible_obs, seg_offset)
        min_hidden_clear = self._trajectory_min_clearance(X, hidden_obs, seg_offset)
        min_clear = min(
            float(min_visible_clear) if min_visible_clear is not None else float("inf"),
            float(min_hidden_clear) if min_hidden_clear is not None else float("inf"),
        )
        clear_term = 0.0 if not np.isfinite(min_clear) else float(min_clear)
        score = (
            self.branch_goal_weight * goal_err
            - self.branch_progress_weight * progress
            + self.branch_slack_weight * slack
            - self.branch_clearance_weight * clear_term
            + 0.01 * effort
        )
        return {
            "goal_err": goal_err,
            "progress": progress,
            "effort": effort,
            "slack": slack,
            "min_visible_clearance": min_visible_clear,
            "min_hidden_clearance": min_hidden_clear,
            "score": float(score),
        }

    def _select_branch(self, explore_diag, fallback_diag, feasible):
        if not feasible:
            return None
        prev = str(self._last_selected_branch)
        score_e = float(explore_diag.get("score", np.inf))
        score_f = float(fallback_diag.get("score", np.inf))
        margin = float(max(0.0, self.branch_switch_margin))
        if prev == "fallback":
            return "explore" if (score_e + margin) < score_f else "fallback"
        return "fallback" if (score_f + margin) < score_e else "explore"

    def _solve_control_problem_coupled(self, robot_state, control_ref, obs_list):
        t0 = time.perf_counter()
        x0 = self._state_from_robot_state(robot_state)
        u_ref = np.asarray(control_ref.get("u_ref", np.zeros((self._u_dim, 1))), dtype=float).reshape(-1, 1)
        u_ref = self._clip_input(u_ref)
        self.last_u_ref = u_ref

        goal_xy = self._goal_xy(x0, control_ref.get("goal", None))
        if control_ref.get("goal", None) is None:
            self.status = "optimal"
            self.last_qp_status_raw = "goal_none"
            self.last_qp_exception = ""
            self.last_num_constraints = 0
            self.last_qp_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            self.last_intervention = "u_ref"
            self.last_u = u_ref
            self.last_profile = {"selected_branch": None, "shared_prefix_length": int(self.n_shared)}
            return u_ref

        visible_obs, occ_all = self._occ_utils._filter_visible_and_build_occ(
            np.asarray(robot_state, dtype=float).reshape(-1, 1),
            obs_list,
        )
        visible_obs = self._nearest_visible_obs(visible_obs, x0)
        occ_candidates = self._nearest_occ_scenarios(occ_all, x0)

        v_ref_nom = self._nominal_speed_reference(x0, u_ref, self.v_ref_default)
        nominal_rollout = self._nominal_route_rollout(x0, control_ref, goal_xy, v_ref_nom)
        nominal_points = np.asarray(nominal_rollout["pos"].T, dtype=float)
        active_entries, all_entries, srq_frames, phantom_obs = self._active_occlusion_context(
            x0=x0,
            goal_xy=goal_xy,
            occ_scenarios=occ_candidates,
            nominal_points=nominal_points,
        )
        self.occlusion_scenarios = [dict(entry["scenario"]) for entry in active_entries]

        speed_s_bound, speed_e_bound, speed_f_bound, risk_raw_vec, risk_norm_vec = self._branch_velocity_boundaries(
            nominal_points=nominal_points,
            srq_frames=srq_frames,
            v_ref_nom=v_ref_nom,
        )
        self.occlusion_risk_score = float(risk_norm_vec[0]) if risk_norm_vec.size else 0.0
        self.visible_pressure_score = 0.0
        self.explore_speed_cap = float(np.min(speed_e_bound)) if speed_e_bound.size else float(np.min(speed_s_bound))
        self.fallback_speed_cap = float(np.min(speed_f_bound)) if speed_f_bound.size else float(np.min(speed_s_bound))

        refs = self._reference_bundle(x0, control_ref, goal_xy, speed_s_bound, speed_e_bound, speed_f_bound)
        pos_s, hdg_s, speed_s = refs["shared"]
        pos_e, hdg_e, speed_e = refs["explore"]
        pos_f, hdg_f, speed_f = refs["fallback"]
        self.shared_speed_ref_min = float(np.min(speed_s)) if speed_s.size else float(v_ref_nom)
        self.explore_speed_ref_min = float(np.min(speed_e)) if speed_e.size else float(v_ref_nom)
        self.fallback_speed_ref_min = float(np.min(speed_f)) if speed_f.size else float(v_ref_nom)

        obs_center, obs_vel, obs_rad, obs_active = self._set_obs_params(visible_obs)
        hid_center, hid_vel, hid_rad, hid_active = self._set_hidden_params(phantom_obs)

        p = self._pars
        self._opti.set_value(p["x0"], np.asarray(x0, dtype=float).reshape(-1, 1))
        self._opti.set_value(p["u_prev"], np.asarray(self._u_prev_applied, dtype=float).reshape(-1, 1))
        self._opti.set_value(p["goal"], np.asarray(goal_xy, dtype=float).reshape(2, 1))
        self._opti.set_value(p["ref_pos_s"], pos_s)
        self._opti.set_value(p["ref_hdg_s"], hdg_s)
        self._opti.set_value(p["ref_speed_s"], speed_s)
        self._opti.set_value(p["ref_pos_e"], pos_e)
        self._opti.set_value(p["ref_hdg_e"], hdg_e)
        self._opti.set_value(p["ref_speed_e"], speed_e)
        self._opti.set_value(p["ref_pos_f"], pos_f)
        self._opti.set_value(p["ref_hdg_f"], hdg_f)
        self._opti.set_value(p["ref_speed_f"], speed_f)
        self._opti.set_value(p["shared_bound"], speed_s)
        self._opti.set_value(p["explore_bound"], speed_e)
        self._opti.set_value(p["fallback_bound"], speed_f)
        self._opti.set_value(p["obs_center"], obs_center)
        self._opti.set_value(p["obs_vel"], obs_vel)
        self._opti.set_value(p["obs_rad"], obs_rad)
        self._opti.set_value(p["obs_active"], obs_active)
        self._opti.set_value(p["hid_center"], hid_center)
        self._opti.set_value(p["hid_vel"], hid_vel)
        self._opti.set_value(p["hid_rad"], hid_rad)
        self._opti.set_value(p["hid_active"], hid_active)

        v = self._vars
        if self._sol_prev is not None:
            for key, var in v.items():
                try:
                    self._opti.set_initial(var, self._sol_prev[key])
                except Exception:
                    pass
        else:
            self._opti.set_initial(v["Xs"], np.tile(np.asarray(x0, dtype=float).reshape(-1, 1), (1, self.n_shared + 1)))
            self._opti.set_initial(v["Xe"], np.tile(np.asarray(x0, dtype=float).reshape(-1, 1), (1, self.n_tail + 1)))
            self._opti.set_initial(v["Xf"], np.tile(np.asarray(x0, dtype=float).reshape(-1, 1), (1, self.n_tail + 1)))
            self._opti.set_initial(v["Us"], np.tile(np.asarray(u_ref, dtype=float).reshape(-1, 1), (1, self.n_shared)))
            self._opti.set_initial(v["Ue"], np.tile(np.asarray(u_ref, dtype=float).reshape(-1, 1), (1, self.n_tail)))
            self._opti.set_initial(v["Uf"], np.tile(np.asarray(self._stop_input(), dtype=float).reshape(-1, 1), (1, self.n_tail)))
            self._opti.set_initial(v["Sshared"], 0.0)
            self._opti.set_initial(v["Se"], 0.0)
            self._opti.set_initial(v["Sf"], 0.0)

        feasible = False
        solve_exc = None
        sol = None
        try:
            sol = self._opti.solve()
            feasible = True
        except Exception as exc:
            solve_exc = exc
            feasible = False

        self.last_qp_solve_time_ms = (time.perf_counter() - t0) * 1000.0
        self.last_num_constraints = int(
            2
            * (self.n_shared + 2 * self.n_tail)
            * (
                int(min(len(visible_obs), self.max_visible_obs))
                + int(min(len(phantom_obs), self.max_phantom_obs))
            )
        )

        if feasible and sol is not None:
            def _sol_matrix(var):
                nrow, ncol = int(var.shape[0]), int(var.shape[1])
                return np.asarray(sol.value(var), dtype=float).reshape(nrow, ncol)

            Xs_val = _sol_matrix(v["Xs"])
            Us_val = _sol_matrix(v["Us"])
            Xe_val = _sol_matrix(v["Xe"])
            Ue_val = _sol_matrix(v["Ue"])
            Xf_val = _sol_matrix(v["Xf"])
            Uf_val = _sol_matrix(v["Uf"])
            Sshared_val = _sol_matrix(v["Sshared"])
            Se_val = _sol_matrix(v["Se"])
            Sf_val = _sol_matrix(v["Sf"])

            self._sol_prev = {
                "Xs": Xs_val, "Us": Us_val, "Xe": Xe_val, "Ue": Ue_val,
                "Xf": Xf_val, "Uf": Uf_val, "Sshared": Sshared_val,
                "Se": Se_val, "Sf": Sf_val,
            }

            u_cmd = np.asarray(Us_val[:, 0], dtype=float).reshape(-1, 1)
            u_cmd = self._clip_input(u_cmd)
            delta = float(np.linalg.norm(u_cmd - u_ref))
            tol = float(self.robot_spec.get("intervention_tol", 1e-3))
            self.last_intervention = "u_ref" if delta <= tol else "oacp_mpc"
            self.last_u = u_cmd
            self._u_prev_applied = u_cmd
            self.status = "optimal"
            self.last_qp_status_raw = "optimal"
            self.last_qp_exception = ""
            self.explore_feasible = True
            self.fallback_feasible = True
            self.shared_risk_slack = float(np.max(Sshared_val)) if Sshared_val.size else 0.0
            self.explore_risk_slack = float(np.max(Se_val)) if Se_val.size else 0.0
            self.fallback_risk_slack = float(np.max(Sf_val)) if Sf_val.size else 0.0

            explore_diag = self._branch_diag(
                Xe_val,
                Ue_val,
                goal_xy,
                refs["path_points"],
                self.n_shared,
                self.explore_risk_slack,
                visible_obs,
                phantom_obs,
            )
            fallback_diag = self._branch_diag(
                Xf_val,
                Uf_val,
                goal_xy,
                refs["path_points"],
                self.n_shared,
                self.fallback_risk_slack,
                visible_obs,
                phantom_obs,
            )
            self.explore_cost = float(explore_diag["score"])
            self.fallback_cost = float(fallback_diag["score"])
            selected_branch = self._select_branch(explore_diag, fallback_diag, True)
            self.selected_branch = selected_branch
            if selected_branch is not None and selected_branch != self._last_selected_branch:
                self.branch_switch_count += 1
            if selected_branch is not None:
                self._last_selected_branch = str(selected_branch)

            shared_min_visible = self._trajectory_min_clearance(Xs_val, visible_obs, 0)
            shared_min_hidden = self._trajectory_min_clearance(Xs_val, phantom_obs, 0)

            self.last_profile = {
                "backend": str(self.backend),
                "backend_requested": str(self.requested_backend),
                "total_ms": 0.0,  # filled below
                "solver_ms": float(self.last_qp_solve_time_ms),
                "selected_branch": self.selected_branch,
                "shared_prefix_length": int(self.n_shared),
                "explore_cost": float(self.explore_cost),
                "fallback_cost": float(self.fallback_cost),
                "explore_feasible": True,
                "fallback_feasible": True,
                "occlusion_risk_score": float(self.occlusion_risk_score),
                "visible_pressure_score": float(self.visible_pressure_score),
                "occlusion_risk_raw_max": float(np.max(risk_raw_vec)) if risk_raw_vec.size else 0.0,
                "occlusion_risk_norm_max": float(np.max(risk_norm_vec)) if risk_norm_vec.size else 0.0,
                "explore_speed_cap": float(self.explore_speed_cap),
                "fallback_speed_cap": float(self.fallback_speed_cap),
                "shared_speed_ref_min": float(self.shared_speed_ref_min),
                "explore_speed_ref_min": float(self.explore_speed_ref_min),
                "fallback_speed_ref_min": float(self.fallback_speed_ref_min),
                "speed_cost_mode": (
                    "progress_aligned" if (self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost) else "speed_norm"
                ),
                "branch_switch_count": int(self.branch_switch_count),
                "num_visible_obs": int(min(len(visible_obs), self.max_visible_obs)),
                "num_occ_scenarios": int(min(len(occ_candidates), self.max_occ_scenarios)),
                "num_active_occlusions": int(len(active_entries)),
                "num_phantom_obs": int(min(len(phantom_obs), self.max_phantom_obs)),
                "shared_prefix_feasible": True,
                "shared_prefix_cost": float(np.sum((Xs_val[:2, -1] - pos_s[:, -1]) ** 2)),
                "shared_min_visible_clearance": shared_min_visible,
                "shared_min_hidden_clearance": shared_min_hidden,
                "explore_min_visible_clearance": explore_diag["min_visible_clearance"],
                "explore_min_hidden_clearance": explore_diag["min_hidden_clearance"],
                "fallback_min_visible_clearance": fallback_diag["min_visible_clearance"],
                "fallback_min_hidden_clearance": fallback_diag["min_hidden_clearance"],
                "explore_max_risk": float(np.max(risk_norm_vec)) if risk_norm_vec.size else 0.0,
                "fallback_max_risk": float(np.max(risk_norm_vec)) if risk_norm_vec.size else 0.0,
                "shared_risk_slack": float(self.shared_risk_slack),
                "explore_risk_slack": float(self.explore_risk_slack),
                "fallback_risk_slack": float(self.fallback_risk_slack),
                "explore_goal_err": float(explore_diag["goal_err"]),
                "fallback_goal_err": float(fallback_diag["goal_err"]),
                "explore_progress": float(explore_diag["progress"]),
                "fallback_progress": float(fallback_diag["progress"]),
                "status_raw": str(self.last_qp_status_raw),
            }
        else:
            self._sol_prev = None
            u_cmd = self._stop_input()
            self.last_u = u_cmd
            self._u_prev_applied = u_cmd
            self.explore_feasible = False
            self.fallback_feasible = False
            self.explore_cost = None
            self.fallback_cost = None
            self.selected_branch = None
            self.shared_risk_slack = 0.0
            self.explore_risk_slack = 0.0
            self.fallback_risk_slack = 0.0
            self.last_qp_exception = "" if solve_exc is None else str(solve_exc)
            if self.allow_solver_fallback:
                self.status = "optimal"
                self.last_intervention = "backup_fallback"
                self.last_qp_status_raw = "fallback_stop"
            else:
                self.status = "infeasible"
                self.last_intervention = "infeasible"
                self.last_qp_status_raw = "infeasible"
            self.last_profile = {
                "backend": str(self.backend),
                "backend_requested": str(self.requested_backend),
                "total_ms": 0.0,
                "solver_ms": float(self.last_qp_solve_time_ms),
                "selected_branch": None,
                "shared_prefix_length": int(self.n_shared),
                "explore_cost": None,
                "fallback_cost": None,
                "explore_feasible": False,
                "fallback_feasible": False,
                "occlusion_risk_score": float(self.occlusion_risk_score),
                "visible_pressure_score": float(self.visible_pressure_score),
                "occlusion_risk_raw_max": float(np.max(risk_raw_vec)) if risk_raw_vec.size else 0.0,
                "occlusion_risk_norm_max": float(np.max(risk_norm_vec)) if risk_norm_vec.size else 0.0,
                "explore_speed_cap": float(self.explore_speed_cap),
                "fallback_speed_cap": float(self.fallback_speed_cap),
                "shared_speed_ref_min": float(self.shared_speed_ref_min),
                "explore_speed_ref_min": float(self.explore_speed_ref_min),
                "fallback_speed_ref_min": float(self.fallback_speed_ref_min),
                "speed_cost_mode": (
                    "progress_aligned" if (self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost) else "speed_norm"
                ),
                "branch_switch_count": int(self.branch_switch_count),
                "num_visible_obs": int(min(len(visible_obs), self.max_visible_obs)),
                "num_occ_scenarios": int(min(len(occ_candidates), self.max_occ_scenarios)),
                "num_active_occlusions": int(len(active_entries)),
                "num_phantom_obs": int(min(len(phantom_obs), self.max_phantom_obs)),
                "shared_prefix_feasible": False,
                "shared_prefix_cost": None,
                "shared_min_visible_clearance": None,
                "shared_min_hidden_clearance": None,
                "explore_max_risk": None,
                "fallback_max_risk": None,
                "shared_risk_slack": 0.0,
                "explore_risk_slack": 0.0,
                "fallback_risk_slack": 0.0,
                "status_raw": str(self.last_qp_status_raw),
            }

        self.last_total_compute_time_ms = (time.perf_counter() - t0) * 1000.0
        self.last_profile["total_ms"] = float(self.last_total_compute_time_ms)
        return u_cmd

    def _solve_control_problem_admm_lowdim(self, robot_state, control_ref, obs_list):
        return self._solve_control_problem_coupled(robot_state, control_ref, obs_list)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        if self.backend == "admm_lowdim":
            return self._solve_control_problem_admm_lowdim(robot_state, control_ref, obs_list)
        return self._solve_control_problem_coupled(robot_state, control_ref, obs_list)
