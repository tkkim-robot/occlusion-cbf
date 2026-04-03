import time

import numpy as np

from position_control._mpc_common import MPCCommonUtils
from utils.occlusion import OcclusionUtils

try:
    import casadi as ca

    _CASADI_AVAILABLE = True
except Exception:
    ca = None
    _CASADI_AVAILABLE = False


class SingleRiskMPC(MPCCommonUtils):
    """
    Single-hypothesis worst-case risk-region MPC baseline.

    This controller intentionally keeps one baseline behavior only:
    - DoubleIntegrator2D / Unicycle2D / DynamicUnicycle2D model
    - hidden speed bound fixed to 0.5
    - single NMPC (no multi-branch / no consensus)
    """

    def __init__(self, robot, robot_spec, num_obs=30):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = int(num_obs)

        self.model = str(robot_spec.get("model", "")).strip()
        if self.model not in {"DoubleIntegrator2D", "Unicycle2D", "DynamicUnicycle2D"}:
            raise ValueError(
                f"SingleRiskMPC currently supports DoubleIntegrator2D, Unicycle2D and DynamicUnicycle2D, got `{self.model}`"
            )
        self._n_state, self._u_dim = self._dims()

        self.dt = float(getattr(robot, "dt", robot_spec.get("dt", 0.05)))
        self.robot_radius = float(robot_spec.get("radius", 0.25))
        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))

        cfg = robot_spec.setdefault("single_risk_mpc", {})
        self.dt_plan = float(cfg.get("dt_plan", 0.25))
        self.Th = float(cfg.get("Th", 6.0))
        n_from_h = int(np.round(self.Th / max(self.dt_plan, 1e-6)))
        self.N = max(2, int(cfg.get("N", n_from_h)))

        # Baseline identity: fixed hidden obstacle speed bound.
        self.hidden_speed = 0.5
        # Use tracking-level `num_obs` as the default visible/occlusion cap.
        # This keeps visible and occlusion scenario selection scales aligned
        # unless the user explicitly overrides either cap in config.
        self.max_visible_obs = int(cfg.get("max_visible_obs", self.num_obs))
        self.max_visible_obs = max(1, self.max_visible_obs)
        self.max_occ_regions = int(cfg.get("max_occ_regions", self.max_visible_obs))
        self.max_occ_regions = max(1, self.max_occ_regions)
        self.risk_regions_per_tangent = int(cfg.get("risk_regions_per_tangent", 2))
        self.drisk = float(cfg.get("drisk", 0.7))
        self.risk_sigma = float(cfg.get("risk_sigma", 1e-4))
        self.min_v_for_risk = float(cfg.get("min_v_for_risk", 0.3))
        self.risk_time_model = str(cfg.get("risk_time_model", "distance_over_vref")).strip().lower()
        if self.risk_time_model not in {"distance_over_vref", "nominal_rollout"}:
            self.risk_time_model = "distance_over_vref"
        self.nominal_k_heading = float(cfg.get("nominal_k_heading", 2.0))
        self.rrisk_max = cfg.get("rrisk_max", 1.5)
        if self.rrisk_max is not None:
            self.rrisk_max = float(self.rrisk_max)

        self.margin_obs = float(cfg.get("margin_obs", 0.05))
        self.margin_risk = float(cfg.get("margin_risk", 0.05))
        self.forward_only = bool(cfg.get("forward_only", False))

        self.wguide = float(cfg.get("wguide", 3.5))
        self.wgoal = float(cfg.get("wgoal", self.wguide))
        self.wvel = float(cfg.get("wvel", 5.0))
        self.wacc = float(cfg.get("wacc", 1.8))
        self.wtrack = float(cfg.get("wtrack", 0.5))
        self.lambda_w = float(cfg.get("lambda_w", 1.0))
        self.v_ref_default = float(cfg.get("v_ref_default", 0.5))
        self.n_split = int(cfg.get("n_split", max(1, int(np.floor(0.5 * self.N)))))
        self.n_split = max(1, min(self.N, self.n_split))

        self.use_guidance_point = bool(cfg.get("use_guidance_point", True))
        self.guidance_mode = str(cfg.get("guidance_mode", "gap")).strip().lower()
        if self.guidance_mode not in {"goal", "gap", "external"}:
            self.guidance_mode = "goal"
        self.guidance_lookahead = float(cfg.get("guidance_lookahead", 2.5))
        self.guidance_side_clearance = float(cfg.get("guidance_side_clearance", 0.5))
        self.guidance_forward_fov_deg = float(cfg.get("guidance_forward_fov_deg", 180.0))
        self.guidance_obs_max_dist = cfg.get("guidance_obs_max_dist", None)
        if self.guidance_obs_max_dist is not None:
            self.guidance_obs_max_dist = float(self.guidance_obs_max_dist)
        self.guidance_min_gap_width = cfg.get("guidance_min_gap_width", None)
        if self.guidance_min_gap_width is not None:
            self.guidance_min_gap_width = float(self.guidance_min_gap_width)
        self.tau_guidance = float(cfg.get("tau_guidance", 0.75))

        self.max_iter = int(cfg.get("max_iter", 200))
        self.solver_tol = float(cfg.get("solver_tol", 1e-4))
        self.solver_acceptable_tol = float(cfg.get("solver_acceptable_tol", 1e-2))
        self.solver_acceptable_iter = int(cfg.get("solver_acceptable_iter", 8))
        self.print_solver = bool(cfg.get("print_solver", False))
        self.solver_expand = bool(cfg.get("solver_expand", False))
        self.backend = str(cfg.get("backend", "persistent_casadi")).strip().lower()
        if self.backend not in {"persistent_casadi", "opti"}:
            self.backend = "persistent_casadi"
        self.persistent_fallback_opti = bool(cfg.get("persistent_fallback_opti", False))
        self.warm_start_dual = bool(cfg.get("warm_start_dual", True))
        self.ipopt_linear_solver = str(cfg.get("ipopt_linear_solver", "mumps")).strip()
        self.max_risk_regions_total = int(
            cfg.get(
                "max_risk_regions_total",
                max(1, int(self.max_occ_regions) * 2 * int(self.risk_regions_per_tangent)),
            )
        )

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=None,
        )

        self._x_prev_plan = None
        self._u_prev_plan = None
        self._u_prev_applied = np.zeros((2, 1), dtype=float)

        # Diagnostics
        self.status = "optimal"
        self.last_num_constraints = 0
        self.last_qp_solve_time_ms = 0.0
        self.last_total_compute_time_ms = 0.0
        self.last_intervention = "u_ref"
        self.last_u_ref = np.zeros((2, 1), dtype=float)
        self.last_u = np.zeros((2, 1), dtype=float)
        self.last_profile = {}
        self.last_qp_status_raw = ""
        self.last_qp_exception = ""
        self.occlusion_scenarios = []

        # Persistent CasADi NLP backend caches.
        self._persistent_setup_done = False
        self._persistent_setup_ms = None
        self._persistent_solver = None
        self._persistent_lbx = None
        self._persistent_ubx = None
        self._persistent_lbg = None
        self._persistent_ubg = None
        self._persistent_p_dim = None
        self._persistent_nx = None
        self._persistent_nu = None
        self._persistent_z_prev = None
        self._persistent_lam_x_prev = None
        self._persistent_lam_g_prev = None

    def _shift_or_init_plan(self, x0, u_ref):
        N = self.N
        x0 = np.asarray(x0, dtype=float).reshape(self._n_state)
        u_ref = self._clip_input(u_ref).reshape(2)

        if self._x_prev_plan is None or self._u_prev_plan is None:
            X = np.zeros((self._n_state, N + 1), dtype=float)
            U = np.tile(u_ref.reshape(2, 1), (1, N))
            X[:, 0] = x0
            for k in range(N):
                X[:, k + 1] = self._discrete_np(X[:, k], U[:, k])
            return X, U

        Xp = np.asarray(self._x_prev_plan, dtype=float)
        Up = np.asarray(self._u_prev_plan, dtype=float)
        X = np.zeros((self._n_state, N + 1), dtype=float)
        U = np.zeros((2, N), dtype=float)
        X[:, :-1] = Xp[:, 1:]
        X[:, -1] = Xp[:, -1]
        U[:, :-1] = Up[:, 1:]
        U[:, -1] = Up[:, -1]
        X[:, 0] = x0
        return X, U

    def _nearest_occ_scenarios(self, occ_scenarios, x0):
        if occ_scenarios is None or len(occ_scenarios) == 0:
            return []
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        scored = []
        for sc in occ_scenarios:
            c = np.asarray(sc.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
            scored.append((float(np.linalg.norm(c - p)), sc))
        scored.sort(key=lambda x: x[0])
        return [it[1] for it in scored[: max(0, int(self.max_occ_regions))]]

    def _gap_guidance(self, x0, goal_xy, visible_obs):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        goal_heading = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
        half_fov = np.deg2rad(max(10.0, self.guidance_forward_fov_deg)) * 0.5
        clearance = self.robot_radius + max(0.0, self.guidance_side_clearance)
        obs_max_dist = (
            max(4.0, 2.0 * self.guidance_lookahead)
            if self.guidance_obs_max_dist is None
            else max(0.1, float(self.guidance_obs_max_dist))
        )

        blocked = []
        n_obs_used = 0
        for obs in visible_obs:
            obs = np.asarray(obs, dtype=float).reshape(-1)
            c = np.asarray(obs[:2], dtype=float).reshape(2,)
            r = float(obs[2]) + clearance

            vec = c - p
            d = float(np.linalg.norm(vec))
            if d < 1e-6:
                blocked = [(-half_fov, half_fov)]
                n_obs_used += 1
                break

            alpha = self._angle_wrap(np.arctan2(vec[1], vec[0]) - goal_heading)
            if abs(alpha) > half_fov or d > obs_max_dist:
                continue

            if d <= r:
                h = half_fov
            else:
                h = float(np.arcsin(np.clip(r / d, 0.0, 1.0)))
            lo, hi = alpha - h, alpha + h
            if hi < -half_fov or lo > half_fov:
                continue
            blocked.append((max(-half_fov, lo), min(half_fov, hi)))
            n_obs_used += 1

        blocked = self._merge_intervals(blocked)
        gaps = []
        cursor = -half_fov
        for lo, hi in blocked:
            if lo > cursor:
                gaps.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < half_fov:
            gaps.append((cursor, half_fov))

        if self.guidance_min_gap_width is None:
            min_gap = 2.0 * (self.robot_radius + self.margin_obs) / max(self.guidance_lookahead, 1e-3)
        else:
            min_gap = max(0.0, float(self.guidance_min_gap_width))

        candidates = []
        for lo, hi in gaps:
            width = hi - lo
            if width >= min_gap:
                ctr = 0.5 * (lo + hi)
                candidates.append((ctr, width))

        if len(candidates) == 0:
            return g, {
                "selected_gap_angle": None,
                "selected_gap_width": None,
                "n_gap_candidates": 0,
                "n_guidance_obs_used": int(n_obs_used),
                "goal_heading": float(goal_heading),
                "guidance_heading": float(goal_heading),
            }

        best_idx = int(np.argmin([abs(c[0]) for c in candidates]))
        gap_ang, gap_w = candidates[best_idx]
        world_ang = goal_heading + gap_ang
        guide = p + self.guidance_lookahead * np.array([np.cos(world_ang), np.sin(world_ang)], dtype=float)
        return guide, {
            "selected_gap_angle": float(gap_ang),
            "selected_gap_width": float(gap_w),
            "n_gap_candidates": int(len(candidates)),
            "n_guidance_obs_used": int(n_obs_used),
            "goal_heading": float(goal_heading),
            "guidance_heading": float(world_ang),
        }

    def _select_guidance_point(self, x0, control_ref, goal_xy, visible_obs):
        if not self.use_guidance_point:
            return goal_xy, {
                "guidance_source": "goal",
                "selected_gap_angle": None,
                "selected_gap_width": None,
                "n_guidance_obs_used": 0,
            }

        mode = self.guidance_mode
        if mode == "external":
            gp = control_ref.get("guidance_point", None)
            if gp is not None:
                arr = np.asarray(gp, dtype=float).reshape(-1)
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return arr[:2].copy(), {
                        "guidance_source": "external",
                        "selected_gap_angle": None,
                        "selected_gap_width": None,
                        "n_guidance_obs_used": 0,
                    }
            return goal_xy, {
                "guidance_source": "goal_fallback_external",
                "selected_gap_angle": None,
                "selected_gap_width": None,
                "n_guidance_obs_used": 0,
            }

        if mode == "gap":
            guide, meta = self._gap_guidance(x0, goal_xy, visible_obs)
            meta["guidance_source"] = (
                "gap" if meta.get("selected_gap_angle") is not None else "goal_fallback_gap"
            )
            return np.asarray(guide, dtype=float).reshape(2,), meta

        return goal_xy, {
            "guidance_source": "goal",
            "selected_gap_angle": None,
            "selected_gap_width": None,
            "n_guidance_obs_used": 0,
        }

    def _nearest_visible_distance(self, x0, visible_obs):
        if visible_obs is None or len(visible_obs) == 0:
            return None
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        best = np.inf
        for obs in visible_obs:
            obs = np.asarray(obs, dtype=float).reshape(-1)
            c = obs[:2]
            clear = self.robot_radius + float(obs[2]) + self.margin_obs
            d = float(np.linalg.norm(p - c) - clear)
            best = min(best, d)
        return None if np.isinf(best) else float(best)

    def _nearest_risk_distance(self, x0, risk_regions):
        if risk_regions is None or len(risk_regions) == 0:
            return None
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        best = np.inf
        for center, rr in risk_regions:
            center = np.asarray(center, dtype=float).reshape(2,)
            clear = self.robot_radius + float(rr) + self.margin_risk
            d = float(np.linalg.norm(p - center) - clear)
            best = min(best, d)
        return None if np.isinf(best) else float(best)

    def _guidance_heading_error(self, x0, goal_xy, guidance_xy):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        gd = np.asarray(guidance_xy, dtype=float).reshape(2,)
        ang_goal = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
        ang_guid = float(np.arctan2(gd[1] - p[1], gd[0] - p[0]))
        return abs(self._angle_wrap(ang_guid - ang_goal))

    def _is_guidance_active(self, mode, guidance_meta):
        if mode == "goal":
            return False, "goal_mode"
        if mode == "external":
            src = str(guidance_meta.get("guidance_source", ""))
            if src == "external":
                return True, "external"
            return False, "goal_fallback_external"

        src = str(guidance_meta.get("guidance_source", ""))
        if src.startswith("gap"):
            return True, "gap"
        return False, "goal_fallback_gap"

    def _nominal_rollout_positions(self, x0, guidance_xy, v_ref_nom):
        lb_u, ub_u = self._input_bounds()
        x = np.asarray(x0, dtype=float).reshape(self._n_state,)
        points = np.zeros((self.N + 1, 2), dtype=float)
        points[0] = x[:2]
        for k in range(self.N):
            u_cmd = self._guidance_input_np(x, guidance_xy, v_ref_nom, k_heading=self.nominal_k_heading)
            x = self._discrete_np(x, u_cmd)
            points[k + 1] = x[:2]
        return points

    def _risk_regions_from_scenario(self, scenario, v_ref_nom, nominal_points):
        if self.hidden_speed <= 1e-9:
            return []

        p = np.asarray(scenario.get("robot_pos", np.zeros(2)), dtype=float).reshape(2,)
        c = np.asarray(scenario.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
        r_obs = float(scenario.get("obs_radius", 0.0))

        t1 = scenario.get("t1", None)
        t2 = scenario.get("t2", None)
        if t1 is None or t2 is None:
            t1, t2 = self._occ_utils._circle_tangents(p, c, r_obs)
        if t1 is None or t2 is None:
            return []

        t1 = np.asarray(t1, dtype=float).reshape(2,)
        t2 = np.asarray(t2, dtype=float).reshape(2,)

        rays = []
        for t in (t1, t2):
            d = t - p
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                rays.append((t, d / n))
        if len(rays) == 0:
            return []

        v_nom = max(float(v_ref_nom), float(self.min_v_for_risk), 1e-3)
        regions = []
        for t, d in rays:
            for i in range(1, self.risk_regions_per_tangent + 1):
                center = t + d * (self.drisk * (i - 1))
                if self.risk_time_model == "nominal_rollout" and nominal_points is not None:
                    dists = np.linalg.norm(nominal_points - center[None, :], axis=1)
                    idx = int(np.argmin(dists))
                    travel_t = idx * self.dt_plan + float(dists[idx]) / v_nom
                else:
                    travel_t = float(np.linalg.norm(center - p)) / v_nom
                rr = travel_t * float(self.hidden_speed) + r_obs + self.risk_sigma
                if self.rrisk_max is not None:
                    rr = min(rr, self.rrisk_max)
                if np.isfinite(rr) and rr > 0.0:
                    regions.append((center, float(rr)))
        return regions

    def _build_risk_regions(self, occ_scenarios, v_ref_nom, nominal_points):
        out = []
        for sc in occ_scenarios:
            out.extend(self._risk_regions_from_scenario(sc, v_ref_nom, nominal_points))
        return out

    def _guidance_track_point(self, k, guidance_xy, goal_xy, n_split_eff):
        if k <= int(n_split_eff):
            return guidance_xy
        return goal_xy

    def _plan_cost_numpy(self, X, U, guidance_xy, goal_xy, v_ref_nom, wtrack_eff, n_split_eff):
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        guidance_xy = np.asarray(guidance_xy, dtype=float).reshape(2,)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        up = np.asarray(self._u_prev_applied, dtype=float).reshape(2,)
        J = 0.0
        for k in range(self.N):
            vk = self._stage_speed_np(X, U, k)
            wk = float(U[1, k])
            if k == 0:
                dv = vk - up[0]
                dw = wk - up[1]
            else:
                dv = vk - float(U[0, k - 1])
                dw = wk - float(U[1, k - 1])
            J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
            J += self.wvel * ((vk - v_ref_nom) ** 2)
            gk = self._guidance_track_point(k, guidance_xy, goal_xy, n_split_eff)
            errk = X[0:2, k] - gk
            J += float(wtrack_eff) * float(errk @ errk)
        terr = X[0:2, self.N] - goal_xy
        J += self.wgoal * float(terr @ terr)
        return float(J)

    def _feasibility_stats(self, X, visible_obs, risk_regions):
        X = np.asarray(X, dtype=float)
        n_ok = 0
        min_vis = np.inf
        min_risk = np.inf
        tol = 1e-7
        for k in range(1, self.N + 1):
            pos = X[:2, k]
            step_min = np.inf
            for obs in visible_obs:
                c = self._predict_obs_center(obs, k)
                clear = self.robot_radius + float(obs[2]) + self.margin_obs
                m = float(np.linalg.norm(pos - c) - clear)
                step_min = min(step_min, m)
                min_vis = min(min_vis, m)
            for center, rr in risk_regions:
                clear = self.robot_radius + float(rr) + self.margin_risk
                m = float(np.linalg.norm(pos - center) - clear)
                step_min = min(step_min, m)
                min_risk = min(min_risk, m)
            if np.isinf(step_min) or step_min >= -tol:
                n_ok += 1
        frac = float(n_ok) / float(max(1, self.N))
        if np.isinf(min_vis):
            min_vis = None
        if np.isinf(min_risk):
            min_risk = None
        return frac, min_vis, min_risk

    def _init_persistent_backend(self):
        if self._persistent_setup_done:
            return True, ""
        if not _CASADI_AVAILABLE:
            return False, "casadi_missing"

        t0 = time.perf_counter()
        try:
            N = int(self.N)
            M = int(self.max_visible_obs)
            R = int(self.max_risk_regions_total)
            lb_u, ub_u = self._input_bounds()

            X = ca.SX.sym("X", self._n_state, N + 1)
            U = ca.SX.sym("U", 2, N)
            Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

            p_dim = self._n_state + 2 + 2 * N + 1 + 1 + 2 + 6 * M + 4 * R
            P = ca.SX.sym("P", p_dim)
            idx = 0
            p_x0 = P[idx : idx + self._n_state]
            idx += self._n_state
            p_goal = P[idx : idx + 2]
            idx += 2
            p_track = ca.reshape(P[idx : idx + 2 * N], 2, N)
            idx += 2 * N
            p_vref = P[idx]
            idx += 1
            p_wtrack = P[idx]
            idx += 1
            p_up = P[idx : idx + 2]
            idx += 2

            vis_params = []
            for _ in range(M):
                vis_params.append(
                    (
                        P[idx + 0],  # ox
                        P[idx + 1],  # oy
                        P[idx + 2],  # r
                        P[idx + 3],  # vx
                        P[idx + 4],  # vy
                        P[idx + 5],  # active
                    )
                )
                idx += 6

            risk_params = []
            for _ in range(R):
                risk_params.append(
                    (
                        P[idx + 0],  # cx
                        P[idx + 1],  # cy
                        P[idx + 2],  # r
                        P[idx + 3],  # active
                    )
                )
                idx += 4

            g = []
            lbg = []
            ubg = []

            g.append(X[:, 0] - p_x0)
            lbg.extend([0.0] * self._n_state)
            ubg.extend([0.0] * self._n_state)

            J = 0
            for k in range(N):
                xk = X[:, k]
                uk = U[:, k]
                g.append(X[:, k + 1] - self._discrete_ca(xk, uk))
                lbg.extend([0.0] * self._n_state)
                ubg.extend([0.0] * self._n_state)

                if k == 0:
                    dv = uk[0] - p_up[0]
                    dw = uk[1] - p_up[1]
                else:
                    dv = uk[0] - U[0, k - 1]
                    dw = uk[1] - U[1, k - 1]

                J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
                J += self.wvel * ((self._stage_speed_ca(X, U, k) - p_vref) ** 2)
                J += p_wtrack * ca.sumsqr(X[0:2, k] - p_track[:, k])

            J += self.wgoal * ca.sumsqr(X[0:2, N] - p_goal)

            for k in range(1, N + 1):
                tk = float(k) * float(self.dt_plan)
                for (ox, oy, rr, vx, vy, active) in vis_params:
                    cx = ox + vx * tk
                    cy = oy + vy * tk
                    safe2 = (self.robot_radius + rr + self.margin_obs) ** 2
                    expr = active * (safe2 - ((X[0, k] - cx) ** 2 + (X[1, k] - cy) ** 2))
                    g.append(ca.vertcat(expr))
                    lbg.append(-np.inf)
                    ubg.append(0.0)

                for (cx, cy, rr, active) in risk_params:
                    safe2 = (self.robot_radius + rr + self.margin_risk) ** 2
                    expr = active * (safe2 - ((X[0, k] - cx) ** 2 + (X[1, k] - cy) ** 2))
                    g.append(ca.vertcat(expr))
                    lbg.append(-np.inf)
                    ubg.append(0.0)

            g_cat = ca.vertcat(*g)
            nlp = {"x": Z, "f": J, "g": g_cat, "p": P}
            opts = {
                "ipopt.print_level": 5 if self.print_solver else 0,
                "ipopt.max_iter": int(self.max_iter),
                "ipopt.tol": float(self.solver_tol),
                "ipopt.acceptable_tol": float(self.solver_acceptable_tol),
                "ipopt.acceptable_iter": int(self.solver_acceptable_iter),
                "ipopt.sb": "yes",
                "print_time": bool(self.print_solver),
                "expand": bool(self.solver_expand),
            }
            if self.ipopt_linear_solver:
                opts["ipopt.linear_solver"] = str(self.ipopt_linear_solver)
            if self.warm_start_dual:
                opts["ipopt.warm_start_init_point"] = "yes"

            solver = ca.nlpsol("single_risk_persistent", "ipopt", nlp, opts)

            z_size = int(Z.shape[0])
            nx = self._n_state * (N + 1)
            nu = 2 * N
            lbx = np.full((z_size,), -np.inf, dtype=float)
            ubx = np.full((z_size,), np.inf, dtype=float)
            for k in range(N):
                base = nx + 2 * k
                lbx[base] = float(lb_u[0])
                ubx[base] = float(ub_u[0])
                lbx[base + 1] = float(lb_u[1])
                ubx[base + 1] = float(ub_u[1])

            self._persistent_solver = solver
            self._persistent_lbx = lbx
            self._persistent_ubx = ubx
            self._persistent_lbg = np.asarray(lbg, dtype=float)
            self._persistent_ubg = np.asarray(ubg, dtype=float)
            self._persistent_p_dim = int(p_dim)
            self._persistent_nx = int(nx)
            self._persistent_nu = int(nu)
            self._persistent_setup_done = True
            self._persistent_setup_ms = (time.perf_counter() - t0) * 1000.0
            return True, ""
        except Exception as exc:
            self._persistent_setup_done = False
            self._persistent_setup_ms = (time.perf_counter() - t0) * 1000.0
            return False, str(exc)

    def _pack_persistent_params(
        self,
        x0,
        goal_xy,
        guidance_xy,
        visible_obs,
        risk_regions,
        v_ref_nom,
        wtrack_eff,
        n_split_eff,
    ):
        p = np.zeros((int(self._persistent_p_dim),), dtype=float)
        idx = 0
        p[idx : idx + self._n_state] = np.asarray(x0, dtype=float).reshape(self._n_state,)
        idx += self._n_state
        p[idx : idx + 2] = np.asarray(goal_xy, dtype=float).reshape(2,)
        idx += 2

        guidance_xy = np.asarray(guidance_xy, dtype=float).reshape(2,)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        for k in range(self.N):
            ref = guidance_xy if k <= int(n_split_eff) else goal_xy
            p[idx : idx + 2] = ref
            idx += 2

        p[idx] = float(v_ref_nom)
        idx += 1
        p[idx] = float(wtrack_eff)
        idx += 1
        p[idx : idx + 2] = np.asarray(self._u_prev_applied, dtype=float).reshape(2,)
        idx += 2

        n_visible_active = int(min(len(visible_obs), int(self.max_visible_obs)))
        for j in range(int(self.max_visible_obs)):
            if j < n_visible_active:
                obs = np.asarray(visible_obs[j], dtype=float).reshape(-1)
                p[idx + 0] = float(obs[0])
                p[idx + 1] = float(obs[1])
                p[idx + 2] = float(obs[2])
                p[idx + 3] = float(obs[3]) if obs.size >= 4 else 0.0
                p[idx + 4] = float(obs[4]) if obs.size >= 5 else 0.0
                p[idx + 5] = 1.0
            idx += 6

        n_risk_active = int(min(len(risk_regions), int(self.max_risk_regions_total)))
        for j in range(int(self.max_risk_regions_total)):
            if j < n_risk_active:
                c, rr = risk_regions[j]
                c = np.asarray(c, dtype=float).reshape(2,)
                p[idx + 0] = float(c[0])
                p[idx + 1] = float(c[1])
                p[idx + 2] = float(rr)
                p[idx + 3] = 1.0
            idx += 4

        return p, n_visible_active, n_risk_active

    def _pack_persistent_guess(self, x_init, u_init):
        x0 = np.asarray(x_init, dtype=float).reshape(self._n_state, self.N + 1, order="F")
        u0 = np.asarray(u_init, dtype=float).reshape(2, self.N, order="F")
        return np.concatenate(
            [
                np.reshape(x0, (-1,), order="F"),
                np.reshape(u0, (-1,), order="F"),
            ]
        )

    def _unpack_persistent_solution(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        x_sol = z[: int(self._persistent_nx)].reshape((self._n_state, self.N + 1), order="F")
        u_sol = z[int(self._persistent_nx) :].reshape((2, self.N), order="F")
        return x_sol, u_sol

    def _solve_nmpc_persistent(
        self,
        x0,
        goal_xy,
        guidance_xy,
        visible_obs,
        risk_regions,
        x_init,
        u_init,
        v_ref_nom,
        wtrack_eff,
        n_split_eff,
    ):
        ok_setup, setup_err = self._init_persistent_backend()
        if not ok_setup:
            return False, None, None, "persistent_setup_failed", setup_err, 0.0, 0

        p, n_visible_active, n_risk_active = self._pack_persistent_params(
            x0=x0,
            goal_xy=goal_xy,
            guidance_xy=guidance_xy,
            visible_obs=visible_obs,
            risk_regions=risk_regions,
            v_ref_nom=v_ref_nom,
            wtrack_eff=wtrack_eff,
            n_split_eff=n_split_eff,
        )
        z0 = self._pack_persistent_guess(x_init, u_init)

        kwargs = {
            "x0": z0,
            "p": p,
            "lbg": self._persistent_lbg,
            "ubg": self._persistent_ubg,
            "lbx": self._persistent_lbx,
            "ubx": self._persistent_ubx,
        }
        if self.warm_start_dual:
            if self._persistent_lam_x_prev is not None and len(self._persistent_lam_x_prev) == len(z0):
                kwargs["lam_x0"] = self._persistent_lam_x_prev
            if self._persistent_lam_g_prev is not None and len(self._persistent_lam_g_prev) == len(self._persistent_lbg):
                kwargs["lam_g0"] = self._persistent_lam_g_prev

        t0 = time.perf_counter()
        try:
            sol = self._persistent_solver(**kwargs)
            solve_ms = (time.perf_counter() - t0) * 1000.0

            z_sol = np.array(sol["x"]).reshape(-1)
            x_sol, u_sol = self._unpack_persistent_solution(z_sol)
            self._persistent_z_prev = z_sol

            if self.warm_start_dual:
                self._persistent_lam_x_prev = np.array(sol["lam_x"]).reshape(-1)
                self._persistent_lam_g_prev = np.array(sol["lam_g"]).reshape(-1)

            raw_status = "optimal"
            try:
                stats = self._persistent_solver.stats()
                if isinstance(stats, dict):
                    raw_status = str(stats.get("return_status", raw_status))
            except Exception:
                pass

            n_constraints = int(self.N * (n_visible_active + n_risk_active))
            return True, x_sol, u_sol, raw_status, "", solve_ms, n_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            n_constraints = int(self.N * (n_visible_active + n_risk_active))
            self._persistent_lam_x_prev = None
            self._persistent_lam_g_prev = None
            return False, None, None, "infeasible", str(exc), solve_ms, n_constraints

    def _solve_nmpc(
        self,
        x0,
        goal_xy,
        guidance_xy,
        visible_obs,
        risk_regions,
        x_init,
        u_init,
        v_ref_nom,
        wtrack_eff,
        n_split_eff,
    ):
        if not _CASADI_AVAILABLE:
            return False, None, None, "casadi_missing", "CasADi is not installed", 0.0, 0

        N = self.N
        lb_u, ub_u = self._input_bounds()

        opti = ca.Opti()
        X = opti.variable(self._n_state, N + 1)
        U = opti.variable(2, N)

        x0_dm = ca.DM(np.asarray(x0, dtype=float).reshape(self._n_state))
        goal_dm = ca.DM(np.asarray(goal_xy, dtype=float).reshape(2))
        guide_dm = ca.DM(np.asarray(guidance_xy, dtype=float).reshape(2))
        up_dm = ca.DM(np.asarray(self._u_prev_applied, dtype=float).reshape(2))

        J = 0
        opti.subject_to(X[:, 0] == x0_dm)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._discrete_ca(X[:, k], U[:, k]))
            opti.subject_to(opti.bounded(lb_u, U[:, k], ub_u))
            if k == 0:
                dv = U[0, k] - up_dm[0]
                dw = U[1, k] - up_dm[1]
            else:
                dv = U[0, k] - U[0, k - 1]
                dw = U[1, k] - U[1, k - 1]
            J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
            J += self.wvel * ((self._stage_speed_ca(X, U, k) - float(v_ref_nom)) ** 2)
            gk = guide_dm if k <= int(n_split_eff) else goal_dm
            J += float(wtrack_eff) * ca.sumsqr(X[0:2, k] - gk)
        terr = X[0:2, N] - goal_dm
        J += self.wgoal * ca.sumsqr(terr)
        opti.minimize(J)

        n_constraints = 0
        for k in range(1, N + 1):
            for obs in visible_obs:
                c = self._predict_obs_center(obs, k)
                clear = self.robot_radius + float(obs[2]) + self.margin_obs
                dx = X[0, k] - float(c[0])
                dy = X[1, k] - float(c[1])
                opti.subject_to(dx * dx + dy * dy >= clear * clear)
                n_constraints += 1
            for center, rr in risk_regions:
                clear = self.robot_radius + float(rr) + self.margin_risk
                dx = X[0, k] - float(center[0])
                dy = X[1, k] - float(center[1])
                opti.subject_to(dx * dx + dy * dy >= clear * clear)
                n_constraints += 1

        opti.set_initial(X, np.asarray(x_init, dtype=float))
        opti.set_initial(U, np.asarray(u_init, dtype=float))

        p_opts = {"expand": self.solver_expand, "print_time": self.print_solver}
        s_opts = {
            "print_level": 5 if self.print_solver else 0,
            "max_iter": self.max_iter,
            "tol": self.solver_tol,
            "acceptable_tol": self.solver_acceptable_tol,
            "acceptable_iter": self.solver_acceptable_iter,
            "sb": "yes",
        }
        opti.solver("ipopt", p_opts, s_opts)

        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            solve_ms = (time.perf_counter() - t0) * 1000.0
            x_sol = np.array(sol.value(X), dtype=float)
            u_sol = np.array(sol.value(U), dtype=float)
            return True, x_sol, u_sol, "optimal", "", solve_ms, n_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            return False, None, None, "infeasible", str(exc), solve_ms, n_constraints

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()

        x0 = self._state_from_robot_state(robot_state)

        u_ref = np.asarray(control_ref.get("u_ref", np.zeros((2, 1))), dtype=float).reshape(-1, 1)
        u_ref = self._clip_input(u_ref)
        self.last_u_ref = u_ref

        goal_xy = self._goal_xy(x0, control_ref.get("goal", None))
        v_ref_floor_eff = float(self.v_ref_default)
        v_ref_nom = self._nominal_speed_reference(x0, u_ref, v_ref_floor_eff)

        visible_obs, occ_scenarios_all = self._occ_utils._filter_visible_and_build_occ(
            np.asarray(robot_state, dtype=float).reshape(-1, 1),
            obs_list,
        )
        visible_obs = self._nearest_visible_obs(visible_obs, x0)
        guidance_xy, guidance_meta = self._select_guidance_point(x0, control_ref, goal_xy, visible_obs)

        nominal_points = None
        if self.risk_time_model == "nominal_rollout":
            nominal_points = self._nominal_rollout_positions(x0, guidance_xy, v_ref_nom)

        occ_scenarios = self._nearest_occ_scenarios(occ_scenarios_all, x0)
        self.occlusion_scenarios = list(occ_scenarios)
        risk_regions = self._build_risk_regions(occ_scenarios, v_ref_nom, nominal_points)

        nearest_risk_distance = self._nearest_risk_distance(x0, risk_regions)
        nearest_visible_distance = self._nearest_visible_distance(x0, visible_obs)

        mode = str(self.guidance_mode if self.use_guidance_point else "goal")
        guidance_active, guidance_reason = self._is_guidance_active(mode, guidance_meta)
        if not guidance_active:
            guidance_xy = np.asarray(goal_xy, dtype=float).reshape(2,)

        # Keep reference baseline schedule fixed.
        effective_wtrack = float(self.wtrack) if guidance_active else 0.0
        effective_n_split = int(self.n_split) if guidance_active else 0

        # In nominal-rollout mode, risk timing depends on guidance trajectory.
        if self.risk_time_model == "nominal_rollout":
            nominal_points = self._nominal_rollout_positions(x0, guidance_xy, v_ref_nom)
            risk_regions = self._build_risk_regions(occ_scenarios, v_ref_nom, nominal_points)
            nearest_risk_distance = self._nearest_risk_distance(x0, risk_regions)

        guidance_heading_error = self._guidance_heading_error(x0, goal_xy, guidance_xy)

        x_guess, u_guess = self._shift_or_init_plan(x0, u_ref)
        active_backend = str(self.backend)
        if active_backend == "persistent_casadi":
            ok, x_sol, u_sol, qp_status, qp_exc, solve_ms, n_constraints = self._solve_nmpc_persistent(
                x0=x0,
                goal_xy=goal_xy,
                guidance_xy=guidance_xy,
                visible_obs=visible_obs,
                risk_regions=risk_regions,
                x_init=x_guess,
                u_init=u_guess,
                v_ref_nom=v_ref_nom,
                wtrack_eff=effective_wtrack,
                n_split_eff=effective_n_split,
            )
            if (not ok) and self.persistent_fallback_opti and qp_status == "persistent_setup_failed":
                active_backend = "opti_fallback"
                ok, x_sol, u_sol, qp_status, qp_exc, solve_ms, n_constraints = self._solve_nmpc(
                    x0=x0,
                    goal_xy=goal_xy,
                    guidance_xy=guidance_xy,
                    visible_obs=visible_obs,
                    risk_regions=risk_regions,
                    x_init=x_guess,
                    u_init=u_guess,
                    v_ref_nom=v_ref_nom,
                    wtrack_eff=effective_wtrack,
                    n_split_eff=effective_n_split,
                )
        else:
            active_backend = "opti"
            ok, x_sol, u_sol, qp_status, qp_exc, solve_ms, n_constraints = self._solve_nmpc(
                x0=x0,
                goal_xy=goal_xy,
                guidance_xy=guidance_xy,
                visible_obs=visible_obs,
                risk_regions=risk_regions,
                x_init=x_guess,
                u_init=u_guess,
                v_ref_nom=v_ref_nom,
                wtrack_eff=effective_wtrack,
                n_split_eff=effective_n_split,
            )

        self.last_qp_status_raw = qp_status
        self.last_qp_exception = qp_exc
        self.last_num_constraints = int(n_constraints)
        self.last_qp_solve_time_ms = float(solve_ms)

        if ok:
            u_arr = np.asarray(u_sol, dtype=float)
            if u_arr.ndim == 1:
                u_arr = u_arr.reshape(2, 1)
            u_cmd = self._clip_input(u_arr[:, 0].reshape(-1, 1))
            if x_sol is not None:
                self._x_prev_plan = np.asarray(x_sol, dtype=float)
            if u_sol is not None:
                self._u_prev_plan = np.asarray(u_sol, dtype=float)
            self._u_prev_applied = u_cmd
            self.last_u = u_cmd
            self.status = "optimal"
            self.last_intervention = "single_risk_mpc"
            if x_sol is not None and u_sol is not None:
                plan_cost = self._plan_cost_numpy(
                    x_sol,
                    u_sol,
                    guidance_xy,
                    goal_xy,
                    v_ref_nom,
                    effective_wtrack,
                    effective_n_split,
                )
                feasible_frac, min_vis_margin, min_risk_margin = self._feasibility_stats(
                    x_sol, visible_obs, risk_regions
                )
            else:
                plan_cost = None
                feasible_frac, min_vis_margin, min_risk_margin = None, None, None
        else:
            self.status = "infeasible"
            self.last_intervention = "single_risk_mpc"
            self.last_u = self._stop_input()
            plan_cost = None
            feasible_frac, min_vis_margin, min_risk_margin = 0.0, None, None

        self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
        rr = np.asarray([float(r[1]) for r in risk_regions], dtype=float)
        rr_stats = {
            "min": (None if rr.size == 0 else float(np.min(rr))),
            "max": (None if rr.size == 0 else float(np.max(rr))),
            "mean": (None if rr.size == 0 else float(np.mean(rr))),
        }
        self.last_profile = {
            "backend": str(active_backend),
            "total_ms": float(self.last_total_compute_time_ms),
            "solve_ms": float(self.last_qp_solve_time_ms),
            "setup_ms": (None if self._persistent_setup_ms is None else float(self._persistent_setup_ms)),
            "raw_solver_status": self.last_qp_status_raw,
            "n_visible_obs": int(len(visible_obs)),
            "n_visible_obs_active": int(min(len(visible_obs), int(self.max_visible_obs))),
            "n_occ_regions_total": int(len(occ_scenarios_all)),
            "n_occ_regions_active": int(len(occ_scenarios)),
            "n_risk_regions_total": int(len(risk_regions)),
            "n_risk_regions_active": int(min(len(risk_regions), int(self.max_risk_regions_total))),
            "risk_circle_radii_stats": rr_stats,
            "guidance_mode": str(self.guidance_mode if self.use_guidance_point else "goal"),
            "guidance_point_xy": [float(guidance_xy[0]), float(guidance_xy[1])],
            "guidance_source": guidance_meta.get("guidance_source", "goal"),
            "guidance_active": bool(guidance_active),
            "guidance_activation_reason": str(guidance_reason),
            "selected_gap_angle": guidance_meta.get("selected_gap_angle", None),
            "selected_gap_width": guidance_meta.get("selected_gap_width", None),
            "n_guidance_obs_used": int(guidance_meta.get("n_guidance_obs_used", 0)),
            "tau_guidance": float(self.tau_guidance),
            "guidance_prediction_horizon_s": float(self.tau_guidance),
            "risk_time_model": str(self.risk_time_model),
            "nearest_visible_distance": (None if nearest_visible_distance is None else float(nearest_visible_distance)),
            "nearest_risk_distance": (None if nearest_risk_distance is None else float(nearest_risk_distance)),
            "guidance_goal_heading_error": float(guidance_heading_error),
            "effective_wtrack": float(effective_wtrack),
            "effective_n_split": int(effective_n_split),
            "v_ref_floor_eff": float(v_ref_floor_eff),
            "v_ref_nom": float(v_ref_nom),
            "feasible_horizon_fraction": (
                None if feasible_frac is None else float(feasible_frac)
            ),
            "min_visible_margin": (None if min_vis_margin is None else float(min_vis_margin)),
            "min_risk_margin": (None if min_risk_margin is None else float(min_risk_margin)),
            "num_constraints": int(self.last_num_constraints),
            "plan_cost": (None if plan_cost is None else float(plan_cost)),
        }

        if ok:
            return self.last_u
        return None
