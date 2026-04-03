import time

import numpy as np

from position_control._mpc_common import MPCCommonUtils
from utils.occlusion import OcclusionUtils

try:
    import casadi as ca
except Exception:
    ca = None


class OACPMPC(MPCCommonUtils):
    """
    Pure contingency-MPC baseline inspired by
    "Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Driving".

    This implementation intentionally departs from the earlier rollout-and-select
    adaptation and instead solves one coupled nonlinear program with:
      - an explicit shared prefix,
      - explicit explore/fallback tails,
      - non-anticipativity enforced through shared-prefix variables,
      - route-aware progress / heading tracking costs,
      - direct occlusion-risk constraints with soft slack,
      - no stop-rescue fallback by default.

    It is still not a paper-faithful ADMM / Bezier reproduction. However, unlike
    the prior adaptation, the optimization structure now matches the intended
    contingency-planning formulation much more closely: both branches are solved
    jointly every cycle and the applied control always comes from the common
    shared prefix.
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
        self.dt_plan = float(cfg.get("dt_plan", 0.20))
        self.Th = float(cfg.get("Th", 2.4))
        n_from_h = int(np.round(self.Th / max(self.dt_plan, 1e-6)))
        self.N = max(5, int(cfg.get("N", n_from_h)))
        self.n_shared = int(cfg.get("n_shared", min(3, self.N - 1)))
        self.n_shared = max(1, min(self.N - 1, self.n_shared))
        self.n_tail = int(self.N - self.n_shared)

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
        self.max_path_points = max(3, int(cfg.get("max_path_points", 10)))
        self.num_occ_facets = 4

        self.margin_obs = float(cfg.get("margin_obs", 0.05))
        self.v_ref_default = float(cfg.get("v_ref_default", 0.45))
        self.v_visible_max = float(
            cfg.get(
                "v_visible_max",
                robot_spec.get("v_obs_max", robot_spec.get("v_adv_max_occ", 0.5)),
            )
        )

        self.risk_explore_scale = float(cfg.get("risk_explore_scale", 0.55))
        self.risk_fallback_scale = float(cfg.get("risk_fallback_scale", 1.10))
        self.explore_speed_scale = float(cfg.get("explore_speed_scale", 1.00))
        self.fallback_speed_scale = float(cfg.get("fallback_speed_scale", 0.65))
        self.shared_speed_blend = float(cfg.get("shared_speed_blend", 0.45))
        self.branch_switch_risk_on = float(cfg.get("branch_switch_risk_on", 0.45))
        self.branch_switch_risk_off = float(cfg.get("branch_switch_risk_off", 0.25))

        self.risk_decay = float(cfg.get("risk_decay", 0.35))
        self.risk_distance_scale = float(cfg.get("risk_distance_scale", 3.5))
        self.risk_softplus_k = float(cfg.get("risk_softplus_k", 8.0))
        self.shared_risk_limit = float(cfg.get("shared_risk_limit", 0.75))
        self.explore_risk_limit = float(cfg.get("explore_risk_limit", 0.90))
        self.fallback_risk_limit = float(cfg.get("fallback_risk_limit", 0.60))

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
        self.w_risk_shared = float(cfg.get("w_risk_shared", 10.0))
        self.w_risk_explore = float(cfg.get("w_risk_explore", 7.0))
        self.w_risk_fallback = float(cfg.get("w_risk_fallback", 14.0))
        self.w_risk_slack_shared = float(cfg.get("w_risk_slack_shared", 300.0))
        self.w_risk_slack_explore = float(cfg.get("w_risk_slack_explore", 180.0))
        self.w_risk_slack_fallback = float(cfg.get("w_risk_slack_fallback", 320.0))

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

        self._u_prev_applied = np.zeros((self._u_dim, 1), dtype=float)
        self._sol_prev = None
        self._build_solver()

    @staticmethod
    def _angle_diff_ca(a, b):
        return ca.atan2(ca.sin(a - b), ca.cos(a - b))

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

    def _aggregate_occ_risk(self, pos_xy, goal_xy, occ_scenarios, step_idx=0):
        if occ_scenarios is None or len(occ_scenarios) == 0:
            return 0.0, None
        pos_xy = np.asarray(pos_xy, dtype=float).reshape(2,)
        scores = []
        tau = max(0.0, float(step_idx) * float(self.dt_plan))
        for idx, sc in enumerate(occ_scenarios):
            A = np.asarray(sc.get("A", np.zeros((0, 2))), dtype=float)
            b0 = np.asarray(sc.get("b0", np.zeros((0,))), dtype=float).reshape(-1)
            if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] != 2 or b0.size != A.shape[0]:
                continue
            v_expand = np.asarray(sc.get("v_expand_vec", np.zeros((A.shape[0],))), dtype=float).reshape(-1)
            if v_expand.size != A.shape[0]:
                v_expand = np.zeros((A.shape[0],), dtype=float)
            signed = float(np.max(A @ pos_xy - (b0 + self.robot_radius + v_expand * tau)))
            poly_term = 1.0 / (1.0 + np.exp(float(self.risk_softplus_k) * signed / max(self.risk_decay, 1e-6)))
            c = np.asarray(sc.get("obs_center", pos_xy), dtype=float).reshape(2,)
            dist = float(np.linalg.norm(pos_xy - c))
            center_term = float(np.exp(-(dist ** 2) / max(self.risk_distance_scale ** 2, 1e-6)))
            scores.append((float(poly_term * center_term), idx))
        if len(scores) == 0:
            return 0.0, None
        scores.sort(reverse=True)
        total = sum(s for s, _ in scores[: self.max_occ_scenarios])
        risk = float(np.clip(total, 0.0, 2.0))
        dominant_idx = scores[0][1] if scores[0][0] > 1e-6 else None
        return risk, dominant_idx

    def _branch_speed_caps(self, v_ref_nom, risk_score):
        v_ref_nom = max(0.0, float(v_ref_nom))
        v_explore = self.explore_speed_scale * v_ref_nom * (1.0 - self.risk_explore_scale * risk_score)
        v_fallback = self.fallback_speed_scale * v_ref_nom * (1.0 - self.risk_fallback_scale * risk_score)
        v_explore = float(np.clip(v_explore, 0.0, max(v_ref_nom, 0.0)))
        v_fallback = float(np.clip(v_fallback, 0.0, v_explore))
        shared = float(np.clip(v_fallback + self.shared_speed_blend * (v_explore - v_fallback), 0.0, max(v_explore, 0.0)))
        return shared, v_explore, v_fallback

    def _aggregate_visible_pressure(self, pos_xy, goal_xy, visible_obs):
        if visible_obs is None or len(visible_obs) == 0:
            return 0.0
        p = np.asarray(pos_xy, dtype=float).reshape(2,)
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        dg = g - p
        ng = float(np.linalg.norm(dg))
        if ng <= 1e-9:
            goal_dir = np.array([1.0, 0.0], dtype=float)
        else:
            goal_dir = dg / ng
        scores = []
        for obs in visible_obs:
            o = np.asarray(obs, dtype=float).reshape(-1)
            c = o[:2]
            r = float(o[2]) if o.size >= 3 else 0.0
            rel = c - p
            ahead = float(np.dot(rel, goal_dir))
            if ahead < -(self.robot_radius + r):
                continue
            lateral = rel - ahead * goal_dir
            lat_d = float(np.linalg.norm(lateral))
            lat_scale = max(0.8, self.robot_radius + r + 0.35)
            lat_term = float(np.exp(-((lat_d / lat_scale) ** 2)))
            clear = float(np.linalg.norm(rel) - (self.robot_radius + r + self.margin_obs))
            clear_term = 1.0 / (1.0 + max(clear, 0.0) / 1.5)
            ahead_term = 1.0 / (1.0 + max(ahead, 0.0) / 3.0)
            scores.append(lat_term * clear_term * ahead_term)
        if len(scores) == 0:
            return 0.0
        return float(np.clip(max(scores), 0.0, 1.0))

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

    def _reference_bundle(self, x0, control_ref, goal_xy, shared_cap, explore_cap, fallback_cap):
        path_pts = self._extract_path_points(control_ref, x0, goal_xy)
        pts, _, cum = self._polyline_arc_data(path_pts)
        s0 = self._closest_progress_on_polyline(np.asarray(x0, dtype=float).reshape(-1)[:2], pts, cum)

        def build(count, v_ref, s_start):
            pos = np.zeros((2, count + 1), dtype=float)
            hdg = np.zeros((1, count + 1), dtype=float)
            speed = np.full((1, count), float(v_ref), dtype=float)
            for k in range(count + 1):
                sk = s_start + float(v_ref) * self.dt_plan * float(k)
                pk, hk = self._sample_polyline(pts, cum, sk)
                pos[:, k] = pk
                hdg[0, k] = hk
            return pos, hdg, speed

        pos_s, hdg_s, speed_s = build(self.n_shared, shared_cap, s0)
        s_shared_end = s0 + float(shared_cap) * self.dt_plan * float(self.n_shared)
        pos_e, hdg_e, speed_e = build(self.n_tail, explore_cap, s_shared_end)
        pos_f, hdg_f, speed_f = build(self.n_tail, fallback_cap, s_shared_end)
        return {
            "path_points": path_pts,
            "shared": (pos_s, hdg_s, speed_s),
            "explore": (pos_e, hdg_e, speed_e),
            "fallback": (pos_f, hdg_f, speed_f),
        }

    def _build_solver(self):
        opti = ca.Opti()

        nx = self._n_state
        nu = self._u_dim
        Ns = self.n_shared
        Nt = self.n_tail
        Mv = self.max_visible_obs
        Mo = self.max_occ_scenarios
        Mf = self.num_occ_facets

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

        shared_cap_p = opti.parameter(1, 1)
        explore_cap_p = opti.parameter(1, 1)
        fallback_cap_p = opti.parameter(1, 1)

        obs_center_p = opti.parameter(2, Mv)
        obs_vel_p = opti.parameter(2, Mv)
        obs_rad_p = opti.parameter(1, Mv)
        obs_active_p = opti.parameter(1, Mv)

        occ_A_p = opti.parameter(2 * Mf, Mo)
        occ_b0_p = opti.parameter(Mf, Mo)
        occ_vexp_p = opti.parameter(Mf, Mo)
        occ_center_p = opti.parameter(2, Mo)
        occ_active_p = opti.parameter(1, Mo)

        lb_u, ub_u = self._input_bounds()
        v_min, v_max = self._speed_bounds()

        big_clear = 1e4
        big_risk = 10.0
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

        def softplus(z):
            return ca.log(1.0 + ca.exp(z))

        def risk_sum(pos_xy, step_idx):
            terms = []
            tau = float(step_idx) * float(self.dt_plan)
            for j in range(Mo):
                signed_terms = []
                for m in range(Mf):
                    rhs = occ_b0_p[m, j] + self.robot_radius + occ_vexp_p[m, j] * tau
                    a0 = occ_A_p[2 * m + 0, j]
                    a1 = occ_A_p[2 * m + 1, j]
                    signed_terms.append(a0 * pos_xy[0] + a1 * pos_xy[1] - rhs)
                signed_stack = ca.vertcat(*signed_terms)
                smax = ca.log(ca.sum1(ca.exp(self.risk_softplus_k * signed_stack))) / self.risk_softplus_k
                poly_term = 1.0 / (1.0 + ca.exp(self.risk_softplus_k * smax / max(self.risk_decay, 1e-6)))
                dc = pos_xy - occ_center_p[:, j]
                center_term = ca.exp(-(ca.sumsqr(dc)) / max(self.risk_distance_scale ** 2, 1e-6))
                terms.append(occ_active_p[0, j] * poly_term * center_term)
            if len(terms) == 0:
                return 0
            return ca.sum1(ca.vertcat(*terms))

        def add_visible_constraints(X, seg_offset):
            n_steps = X.shape[1] - 1
            for k in range(1, n_steps + 1):
                pos = X[:2, k]
                abs_step = seg_offset + k
                for j in range(Mv):
                    center = obs_center_p[:, j] + float(self.dt_plan * abs_step) * obs_vel_p[:, j]
                    inflate = 0.0
                    if self.visible_reach_mode == "worst_case":
                        inflate = self.v_visible_max * self.dt_plan * abs_step
                    r_eff = obs_rad_p[0, j] + self.robot_radius + self.margin_obs + inflate
                    dist2 = ca.sumsqr(pos - center)
                    rhs = (r_eff ** 2) - big_clear * (1.0 - obs_active_p[0, j])
                    opti.subject_to(dist2 >= rhs)

        add_visible_constraints(Xs, 0)
        add_visible_constraints(Xe, Ns)
        add_visible_constraints(Xf, Ns)

        def add_branch_costs(X, U, ref_pos, ref_hdg, ref_speed, pos_w, hdg_w, speed_w, risk_w, slack_w, risk_limit, speed_cap, slack_var, seg_offset):
            J = 0
            u_prev_local = u_prev_p
            n_steps = U.shape[1]
            for k in range(n_steps):
                xk1 = X[:, k + 1]
                uk = U[:, k]
                pos_err = xk1[:2] - ref_pos[:, k + 1]
                hdg_err = self._heading_error_ca(xk1, ref_hdg[0, k + 1])
                hdg_w_eff = hdg_w * self._heading_track_weight_ca(xk1)
                v_err = stage_speed(X, U, k) - ref_speed[0, k]
                du = uk - u_prev_local
                J += pos_w * ca.sumsqr(pos_err)
                J += hdg_w_eff * (hdg_err ** 2)
                J += speed_w * (v_err ** 2)
                J += self.w_u * ca.sumsqr(uk)
                J += self.w_du * ca.sumsqr(du)
                u_prev_local = uk

                if self.model == "DynamicUnicycle2D":
                    opti.subject_to(X[3, k] <= speed_cap + 1e-6)
                elif self.model == "DoubleIntegrator2D":
                    opti.subject_to(ca.sumsqr(X[2:4, k + 1]) <= (speed_cap + 1e-6) ** 2)
                else:
                    opti.subject_to(U[0, k] <= speed_cap + 1e-6)
                    if self.forward_only:
                        opti.subject_to(U[0, k] >= 0.0)

                r_sum = risk_sum(xk1[:2], seg_offset + k + 1)
                opti.subject_to(r_sum <= risk_limit + slack_var[0, k] + big_risk * (1.0 - ca.fmin(1.0, ca.sum1(occ_active_p))))
                J += risk_w * (r_sum ** 2)
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
                pos_err = xk1[:2] - ref_pos_s[:, k + 1]
                hdg_err = self._heading_error_ca(xk1, ref_hdg_s[0, k + 1])
                hdg_w_eff = self.w_heading_shared * self._heading_track_weight_ca(xk1)
                v_err = stage_speed(Xs, Us, k) - ref_speed_s[0, k]
                du = uk - u_prev_local
                J += self.w_pos_shared * ca.sumsqr(pos_err)
                J += hdg_w_eff * (hdg_err ** 2)
                J += self.w_speed_shared * (v_err ** 2)
                J += self.w_u * ca.sumsqr(uk)
                J += self.w_du * ca.sumsqr(du)
                u_prev_local = uk

                if self.model == "DynamicUnicycle2D":
                    opti.subject_to(Xs[3, k] <= shared_cap_p[0, 0] + 1e-6)
                elif self.model == "DoubleIntegrator2D":
                    opti.subject_to(ca.sumsqr(Xs[2:4, k + 1]) <= (shared_cap_p[0, 0] + 1e-6) ** 2)
                else:
                    opti.subject_to(Us[0, k] <= shared_cap_p[0, 0] + 1e-6)
                    if self.forward_only:
                        opti.subject_to(Us[0, k] >= 0.0)

                r_sum = risk_sum(xk1[:2], k + 1)
                opti.subject_to(r_sum <= self.shared_risk_limit + Sshared[0, k] + big_risk * (1.0 - ca.fmin(1.0, ca.sum1(occ_active_p))))
                J += self.w_risk_shared * (r_sum ** 2)
                J += self.w_risk_slack_shared * (Sshared[0, k] ** 2)
            return J

        objective += add_shared_costs()
        objective += add_branch_costs(
            Xe, Ue, ref_pos_e, ref_hdg_e, ref_speed_e,
            self.w_pos_explore, self.w_heading_explore, self.w_speed_explore,
            self.w_risk_explore, self.w_risk_slack_explore, self.explore_risk_limit,
            explore_cap_p[0, 0], Se, Ns,
        )
        objective += add_branch_costs(
            Xf, Uf, ref_pos_f, ref_hdg_f, ref_speed_f,
            self.w_pos_fallback, self.w_heading_fallback, self.w_speed_fallback,
            self.w_risk_fallback, self.w_risk_slack_fallback, self.fallback_risk_limit,
            fallback_cap_p[0, 0], Sf, Ns,
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
            "shared_cap": shared_cap_p, "explore_cap": explore_cap_p, "fallback_cap": fallback_cap_p,
            "obs_center": obs_center_p, "obs_vel": obs_vel_p, "obs_rad": obs_rad_p, "obs_active": obs_active_p,
            "occ_A": occ_A_p, "occ_b0": occ_b0_p, "occ_vexp": occ_vexp_p,
            "occ_center": occ_center_p, "occ_active": occ_active_p,
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

    def _set_occ_params(self, occ_scenarios):
        A = np.zeros((2 * self.num_occ_facets, self.max_occ_scenarios), dtype=float)
        b0 = np.zeros((self.num_occ_facets, self.max_occ_scenarios), dtype=float)
        vexp = np.zeros((self.num_occ_facets, self.max_occ_scenarios), dtype=float)
        centers = np.zeros((2, self.max_occ_scenarios), dtype=float)
        active = np.zeros((1, self.max_occ_scenarios), dtype=float)
        for j, sc in enumerate(list(occ_scenarios)[: self.max_occ_scenarios]):
            A_j = np.asarray(sc.get("A", np.zeros((0, 2))), dtype=float)
            b_j = np.asarray(sc.get("b0", np.zeros((0,))), dtype=float).reshape(-1)
            v_j = np.asarray(sc.get("v_expand_vec", np.zeros((0,))), dtype=float).reshape(-1)
            m = min(self.num_occ_facets, A_j.shape[0], b_j.size)
            if m <= 0:
                continue
            for row in range(m):
                A[2 * row: 2 * row + 2, j] = A_j[row, :]
            b0[:m, j] = b_j[:m]
            if v_j.size >= m:
                vexp[:m, j] = v_j[:m]
            centers[:, j] = np.asarray(sc.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
            active[0, j] = 1.0
        return A, b0, vexp, centers, active

    def _select_branch(self, risk_score, explore_cost, fallback_cost, feasible):
        if not feasible:
            return None
        prev = str(self._last_selected_branch)
        if prev == "fallback":
            return "fallback" if risk_score >= self.branch_switch_risk_off else "explore"
        return "fallback" if risk_score >= self.branch_switch_risk_on else "explore"

    def solve_control_problem(self, robot_state, control_ref, obs_list):
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
        occ_scenarios = self._nearest_occ_scenarios(occ_all, x0)
        self.occlusion_scenarios = list(occ_scenarios)

        occ_risk_score, _ = self._aggregate_occ_risk(x0[:2], goal_xy, occ_scenarios, 0)
        visible_pressure = self._aggregate_visible_pressure(x0[:2], goal_xy, visible_obs)
        risk_score = float(max(occ_risk_score, visible_pressure))
        self.occlusion_risk_score = float(occ_risk_score)
        self.visible_pressure_score = float(visible_pressure)

        v_ref_nom = self._nominal_speed_reference(x0, u_ref, self.v_ref_default)
        shared_cap, explore_cap, fallback_cap = self._branch_speed_caps(v_ref_nom, risk_score)
        self.explore_speed_cap = float(explore_cap)
        self.fallback_speed_cap = float(fallback_cap)

        refs = self._reference_bundle(x0, control_ref, goal_xy, shared_cap, explore_cap, fallback_cap)
        pos_s, hdg_s, speed_s = refs["shared"]
        pos_e, hdg_e, speed_e = refs["explore"]
        pos_f, hdg_f, speed_f = refs["fallback"]

        obs_center, obs_vel, obs_rad, obs_active = self._set_obs_params(visible_obs)
        occ_A, occ_b0, occ_vexp, occ_center, occ_active = self._set_occ_params(occ_scenarios)

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
        self._opti.set_value(p["shared_cap"], np.array([[shared_cap]], dtype=float))
        self._opti.set_value(p["explore_cap"], np.array([[explore_cap]], dtype=float))
        self._opti.set_value(p["fallback_cap"], np.array([[fallback_cap]], dtype=float))
        self._opti.set_value(p["obs_center"], obs_center)
        self._opti.set_value(p["obs_vel"], obs_vel)
        self._opti.set_value(p["obs_rad"], obs_rad)
        self._opti.set_value(p["obs_active"], obs_active)
        self._opti.set_value(p["occ_A"], occ_A)
        self._opti.set_value(p["occ_b0"], occ_b0)
        self._opti.set_value(p["occ_vexp"], occ_vexp)
        self._opti.set_value(p["occ_center"], occ_center)
        self._opti.set_value(p["occ_active"], occ_active)

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
            (self.N + self.n_shared) * min(len(visible_obs), self.max_visible_obs)
            + self.N * min(len(occ_scenarios), self.max_occ_scenarios)
        )

        if feasible and sol is not None:
            Xs_val = np.asarray(sol.value(v["Xs"]), dtype=float)
            Us_val = np.asarray(sol.value(v["Us"]), dtype=float)
            Xe_val = np.asarray(sol.value(v["Xe"]), dtype=float)
            Ue_val = np.asarray(sol.value(v["Ue"]), dtype=float)
            Xf_val = np.asarray(sol.value(v["Xf"]), dtype=float)
            Uf_val = np.asarray(sol.value(v["Uf"]), dtype=float)
            Sshared_val = np.asarray(sol.value(v["Sshared"]), dtype=float)
            Se_val = np.asarray(sol.value(v["Se"]), dtype=float)
            Sf_val = np.asarray(sol.value(v["Sf"]), dtype=float)

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

            # branch diagnostics are used for interpretability only; control is always shared-prefix.
            explore_cost = float(np.sum((Xe_val[:2, -1] - np.asarray(goal_xy).reshape(2,)) ** 2) + 50.0 * self.explore_risk_slack)
            fallback_cost = float(np.sum((Xf_val[:2, -1] - np.asarray(goal_xy).reshape(2,)) ** 2) + 50.0 * self.fallback_risk_slack)
            self.explore_cost = explore_cost
            self.fallback_cost = fallback_cost
            selected_branch = self._select_branch(risk_score, explore_cost, fallback_cost, True)
            self.selected_branch = selected_branch
            if selected_branch is not None and selected_branch != self._last_selected_branch:
                self.branch_switch_count += 1
            if selected_branch is not None:
                self._last_selected_branch = str(selected_branch)

            self.last_profile = {
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
                "explore_speed_cap": float(self.explore_speed_cap),
                "fallback_speed_cap": float(self.fallback_speed_cap),
                "branch_switch_count": int(self.branch_switch_count),
                "num_visible_obs": int(min(len(visible_obs), self.max_visible_obs)),
                "num_occ_scenarios": int(min(len(occ_scenarios), self.max_occ_scenarios)),
                "shared_prefix_feasible": True,
                "shared_prefix_cost": float(np.sum((Xs_val[:2, -1] - pos_s[:, -1]) ** 2)),
                "shared_min_visible_clearance": None,
                "explore_max_risk": float(self.explore_risk_slack + risk_score),
                "fallback_max_risk": float(self.fallback_risk_slack + risk_score),
                "shared_risk_slack": float(self.shared_risk_slack),
                "explore_risk_slack": float(self.explore_risk_slack),
                "fallback_risk_slack": float(self.fallback_risk_slack),
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
                "explore_speed_cap": float(self.explore_speed_cap),
                "fallback_speed_cap": float(self.fallback_speed_cap),
                "branch_switch_count": int(self.branch_switch_count),
                "num_visible_obs": int(min(len(visible_obs), self.max_visible_obs)),
                "num_occ_scenarios": int(min(len(occ_scenarios), self.max_occ_scenarios)),
                "shared_prefix_feasible": False,
                "shared_prefix_cost": None,
                "shared_min_visible_clearance": None,
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
