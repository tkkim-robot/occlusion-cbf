import time
from collections import deque

import numpy as np

from position_control._mpc_common import MPCCommonUtils
from utils.occlusion import OcclusionUtils

try:
    import casadi as ca

    _CASADI_AVAILABLE = True
except Exception:
    ca = None
    _CASADI_AVAILABLE = False


class ControlTreeMPC(MPCCommonUtils):
    """
    Control-Tree-inspired, framework-adapted baseline for `test_crowd`.

    Finalized shared runtime scope:
    - Sequential backend only.
    - Visibility-aware branch MPC (clustered visible obstacles -> free gaps -> branches).
    - Near-goal strong-only handoff retained as finalized orbiting mitigation logic.
    - Experimental backends/modes were removed from mainline runtime path.
    - Solver graph reuse (`solver_backend=persistent_casadi`) accelerates compute
      without changing the MPC objective/constraints.

    Notes:
    - This is not an exact reproduction of the original Control-Tree paper stack.
    - Hidden/occluded risk regions are intentionally out of scope for this baseline.
    """

    def __init__(self, robot, robot_spec, num_obs=30):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = int(num_obs)

        self.model = str(robot_spec.get("model", "")).strip()
        if self.model != "Unicycle2D":
            raise ValueError(
                f"ControlTreeMPC currently supports Unicycle2D only, got `{self.model}`"
            )

        self.dt = float(getattr(robot, "dt", robot_spec.get("dt", 0.05)))
        self.robot_radius = float(robot_spec.get("radius", 0.25))
        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))

        cfg = robot_spec.setdefault("control_tree_mpc", {})

        # Finalized shared baseline config.
        self.backend = "sequential"
        self.dt_plan = float(cfg.get("dt_plan", 0.25))
        self.Th = float(cfg.get("Th", 6.0))
        n_from_h = int(np.round(self.Th / max(self.dt_plan, 1e-6)))
        self.N = max(2, int(cfg.get("N", n_from_h)))

        self.n_branches = max(1, int(cfg.get("n_branches", 3)))
        self.gap_lookahead = float(cfg.get("gap_lookahead", 2.5))
        self.cluster_merge_distance = float(cfg.get("cluster_merge_distance", 0.8))
        self.forward_fov_deg_for_branching = float(cfg.get("forward_fov_deg_for_branching", 180.0))
        self.min_gap_width = cfg.get("min_gap_width", None)
        if self.min_gap_width is not None:
            self.min_gap_width = float(self.min_gap_width)

        self.n_split = int(cfg.get("n_split", max(1, int(np.floor(0.5 * self.N)))))
        self.n_split = max(1, min(self.N, self.n_split))

        # Visible-only baseline: use up to `num_obs` visible obstacles by default.
        # If `max_visible_obs` is explicitly provided, honor that override.
        self.max_visible_obs = int(cfg.get("max_visible_obs", self.num_obs))
        self.max_visible_obs = max(1, self.max_visible_obs)
        self.margin_obs = float(cfg.get("margin_obs", 0.05))
        self.forward_only = bool(cfg.get("forward_only", False))

        self.wgoal = float(cfg.get("wgoal", cfg.get("wguide", 3.5)))
        self.wvel = float(cfg.get("wvel", 5.0))
        self.wacc = float(cfg.get("wacc", 1.8))
        self.wtrack = float(cfg.get("wtrack", 2.0))
        self.lambda_w = float(cfg.get("lambda_w", 1.0))
        self.v_ref_default = float(cfg.get("v_ref_default", 0.5))

        # Finalized near-goal strong-only handoff knobs.
        self.goal_handover_radius = float(cfg.get("goal_handover_radius", 1.5))
        self.direct_goal_clearance_margin = float(cfg.get("direct_goal_clearance_margin", 0.08))
        self.goal_handover_hysteresis = float(cfg.get("goal_handover_hysteresis", 0.2))
        self.near_goal_min_speed = float(cfg.get("near_goal_min_speed", 0.2))
        self.near_goal_speed_scale_radius = float(
            cfg.get("near_goal_speed_scale_radius", self.goal_handover_radius)
        )
        self.near_goal_progress_window_steps = int(cfg.get("near_goal_progress_window_steps", 30))
        self.near_goal_progress_window_steps = max(5, self.near_goal_progress_window_steps)
        self.near_goal_progress_min_drop = float(cfg.get("near_goal_progress_min_drop", 0.03))
        self.near_goal_force_radius = float(cfg.get("near_goal_force_radius", 0.6))
        self.near_goal_mode_strategy = "strong_only"

        self.max_iter = int(cfg.get("max_iter", 200))
        self.solver_tol = float(cfg.get("solver_tol", 1e-4))
        self.solver_acceptable_tol = float(cfg.get("solver_acceptable_tol", 1e-2))
        self.solver_acceptable_iter = int(cfg.get("solver_acceptable_iter", 8))
        self.print_solver = bool(cfg.get("print_solver", False))
        self.solver_expand = bool(cfg.get("solver_expand", False))
        self.solve_backend = str(cfg.get("solver_backend", "persistent_casadi")).strip().lower()
        if self.solve_backend not in {"persistent_casadi", "opti"}:
            self.solve_backend = "persistent_casadi"
        self.persistent_fallback_opti = bool(cfg.get("persistent_fallback_opti", False))
        self.warm_start_dual = bool(cfg.get("warm_start_dual", True))
        self.ipopt_linear_solver = str(cfg.get("ipopt_linear_solver", "mumps")).strip()

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=None,
        )

        self._x_prev_plans = None
        self._u_prev_plans = None
        self._u_prev_applied = np.zeros((2, 1), dtype=float)
        self._near_goal_mode_active = False
        self._goal_dist_hist = deque(maxlen=self.near_goal_progress_window_steps)

        # Framework diagnostics contract.
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

        # Persistent CasADi NLP backend caches (algorithm unchanged, graph reuse only).
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
        self._persistent_z_prev = []
        self._persistent_lam_x_prev = []
        self._persistent_lam_g_prev = []

    def _cluster_visible_obs(self, visible_obs, x0, goal_heading):
        """
        Aggregate visible obstacle discs into coarse angular clusters.

        Branching then happens over free angular gaps between these blocked spans,
        which keeps this baseline visibility-aware in a Control-Tree-inspired way.
        """
        if visible_obs is None or len(visible_obs) == 0:
            return []
        obs = np.asarray(visible_obs, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        centers = obs[:, :2]
        radii = obs[:, 2] + self.robot_radius + self.margin_obs
        n = centers.shape[0]

        used = np.zeros(n, dtype=bool)
        clusters = []
        for i in range(n):
            if used[i]:
                continue
            stack = [i]
            used[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if used[v]:
                        continue
                    d = float(np.linalg.norm(centers[u] - centers[v]))
                    if d <= float(radii[u] + radii[v] + self.cluster_merge_distance):
                        used[v] = True
                        stack.append(v)

            pts = centers[comp]
            rs = radii[comp]
            centroid = np.mean(pts, axis=0)
            r_bound = 0.0
            for j in range(len(comp)):
                r_bound = max(r_bound, float(np.linalg.norm(pts[j] - centroid) + rs[j]))

            p = np.asarray(x0, dtype=float).reshape(-1)[:2]
            vec = centroid - p
            d = float(np.linalg.norm(vec))
            alpha = self._angle_wrap(np.arctan2(vec[1], vec[0]) - goal_heading)
            if d <= 1e-6 or d <= r_bound:
                half = np.pi
            else:
                half = float(np.arcsin(np.clip(r_bound / d, 0.0, 1.0)))
            clusters.append(
                {
                    "indices": list(comp),
                    "centroid": centroid,
                    "radius": float(r_bound),
                    "angle_span": (float(alpha - half), float(alpha + half)),
                }
            )
        return clusters

    def _gap_candidates(self, x0, goal_xy, clusters):
        """
        Extract free angular gaps in a forward field-of-view and return candidate
        local guidance points from each gap center.
        """
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        goal_heading = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
        half_fov = np.deg2rad(max(10.0, self.forward_fov_deg_for_branching)) * 0.5

        blocked = []
        for c in clusters:
            lo, hi = c["angle_span"]
            if hi < -half_fov or lo > half_fov:
                continue
            blocked.append((max(-half_fov, lo), min(half_fov, hi)))
        blocked = self._merge_intervals(blocked)

        gaps = []
        cursor = -half_fov
        for lo, hi in blocked:
            if lo > cursor:
                gaps.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < half_fov:
            gaps.append((cursor, half_fov))

        if self.min_gap_width is None:
            min_gap = 2.0 * (self.robot_radius + self.margin_obs) / max(self.gap_lookahead, 1e-3)
        else:
            min_gap = max(0.0, float(self.min_gap_width))

        candidates = []
        for lo, hi in gaps:
            width = hi - lo
            if width < min_gap:
                continue
            ctr = 0.5 * (lo + hi)
            world_ang = goal_heading + ctr
            local_point = p + self.gap_lookahead * np.array([np.cos(world_ang), np.sin(world_ang)], dtype=float)
            candidates.append(
                {
                    "gap_angle": float(ctr),
                    "gap_width": float(width),
                    "local_point": local_point,
                }
            )
        return goal_heading, candidates

    def _build_branches(self, x0, goal_xy, clusters):
        """
        Build branch hypotheses from visible gaps:
        - goal-most-aligned,
        - nearest left alternative,
        - nearest right alternative,
        then fill remaining slots by alignment order.
        """
        goal_heading, cands = self._gap_candidates(x0, goal_xy, clusters)
        branches = []
        used = set()

        if len(cands) > 0:
            i0 = int(np.argmin([abs(c["gap_angle"]) for c in cands]))
            used.add(i0)
            branches.append(cands[i0])

            left = [i for i, c in enumerate(cands) if c["gap_angle"] > 0 and i not in used]
            if len(left) > 0:
                il = min(left, key=lambda i: abs(cands[i]["gap_angle"]))
                used.add(il)
                branches.append(cands[il])

            right = [i for i, c in enumerate(cands) if c["gap_angle"] < 0 and i not in used]
            if len(right) > 0:
                ir = min(right, key=lambda i: abs(cands[i]["gap_angle"]))
                used.add(ir)
                branches.append(cands[ir])

            rest = [i for i in range(len(cands)) if i not in used]
            rest = sorted(rest, key=lambda i: abs(cands[i]["gap_angle"]))
            for i in rest:
                if len(branches) >= self.n_branches:
                    break
                branches.append(cands[i])

        if len(branches) == 0:
            p = np.asarray(x0, dtype=float).reshape(-1)[:2]
            local_point = p + self.gap_lookahead * np.array([np.cos(goal_heading), np.sin(goal_heading)], dtype=float)
            branches.append(
                {
                    "gap_angle": 0.0,
                    "gap_width": 2.0 * np.deg2rad(max(10.0, self.forward_fov_deg_for_branching)) * 0.5,
                    "local_point": local_point,
                }
            )

        branches = branches[: max(1, int(self.n_branches))]
        return branches

    def _shift_or_init_branch_plan(self, x0, u_ref, branch_idx):
        N = self.N
        x0 = np.asarray(x0, dtype=float).reshape(3)
        u_ref = self._clip_input(u_ref).reshape(2)

        if self._x_prev_plans is None or self._u_prev_plans is None:
            X = np.zeros((3, N + 1), dtype=float)
            U = np.tile(u_ref.reshape(2, 1), (1, N))
            X[:, 0] = x0
            for k in range(N):
                X[:, k + 1] = self._discrete_np(X[:, k], U[:, k])
            return X, U

        if (
            branch_idx >= len(self._x_prev_plans)
            or self._x_prev_plans[branch_idx] is None
            or self._u_prev_plans[branch_idx] is None
        ):
            X = np.zeros((3, N + 1), dtype=float)
            U = np.tile(u_ref.reshape(2, 1), (1, N))
            X[:, 0] = x0
            for k in range(N):
                X[:, k + 1] = self._discrete_np(X[:, k], U[:, k])
            return X, U

        Xp = np.asarray(self._x_prev_plans[branch_idx], dtype=float)
        Up = np.asarray(self._u_prev_plans[branch_idx], dtype=float)
        X = np.zeros((3, N + 1), dtype=float)
        U = np.zeros((2, N), dtype=float)
        X[:, :-1] = Xp[:, 1:]
        X[:, -1] = Xp[:, -1]
        U[:, :-1] = Up[:, 1:]
        U[:, -1] = Up[:, -1]
        X[:, 0] = x0
        return X, U

    def _guidance_track_point(self, k, local_xy, goal_xy, n_split=None):
        split = self.n_split if n_split is None else int(n_split)
        if k <= split:
            return local_xy
        return goal_xy

    def _plan_cost_numpy(self, X, U, local_xy, goal_xy, v_ref_nom, n_split=None):
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        local_xy = np.asarray(local_xy, dtype=float).reshape(2,)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        up = np.asarray(self._u_prev_applied, dtype=float).reshape(2,)
        J = 0.0
        for k in range(self.N):
            vk = float(U[0, k])
            wk = float(U[1, k])
            if k == 0:
                dv = vk - up[0]
                dw = wk - up[1]
            else:
                dv = vk - float(U[0, k - 1])
                dw = wk - float(U[1, k - 1])
            J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
            J += self.wvel * ((vk - v_ref_nom) ** 2)
            gk = self._guidance_track_point(k, local_xy, goal_xy, n_split=n_split)
            errk = X[0:2, k] - gk
            J += self.wtrack * float(errk @ errk)
        terr = X[0:2, self.N] - goal_xy
        J += self.wgoal * float(terr @ terr)
        return float(J)

    @staticmethod
    def _dist_point_segment(point_xy, seg_a_xy, seg_b_xy):
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

    def _direct_goal_clearance(self, x0, goal_xy, visible_obs):
        p0 = np.asarray(x0, dtype=float).reshape(-1)[:2]
        pg = np.asarray(goal_xy, dtype=float).reshape(2,)
        min_clear = np.inf
        for obs in visible_obs:
            o = np.asarray(obs, dtype=float).reshape(-1)
            c = o[:2]
            r = float(o[2]) + self.robot_radius + self.margin_obs + self.direct_goal_clearance_margin
            d = self._dist_point_segment(c, p0, pg)
            min_clear = min(min_clear, float(d - r))
        return float(min_clear) if np.isfinite(min_clear) else np.inf

    def _nearest_visible_obs_clearance(self, x0, visible_obs):
        p0 = np.asarray(x0, dtype=float).reshape(-1)[:2]
        min_clear = np.inf
        for obs in visible_obs:
            o = np.asarray(obs, dtype=float).reshape(-1)
            c = o[:2]
            r = float(o[2]) + self.robot_radius + self.margin_obs
            d = float(np.linalg.norm(c - p0)) - r
            min_clear = min(min_clear, d)
        return float(min_clear) if np.isfinite(min_clear) else np.inf

    def _update_near_goal_mode(self, goal_dist, direct_goal_admissible, progress_stalled):
        # Finalized mode: strong-only handoff.
        near_goal = bool(goal_dist <= self.goal_handover_radius)
        force_on = bool(goal_dist <= self.near_goal_force_radius)
        off_dist = bool(goal_dist >= (self.goal_handover_radius + self.goal_handover_hysteresis))
        admissible = bool(direct_goal_admissible)
        stalled = bool(progress_stalled)

        prev_on = bool(self._near_goal_mode_active)
        if prev_on:
            if off_dist:
                level, reason = "off", "handover_off_distance"
            elif not admissible:
                level, reason = "off", "handover_off_blocked"
            else:
                level, reason = "strong", "handover_active_strong"
        else:
            if near_goal and admissible and (stalled or force_on):
                level = "strong"
                reason = "strong_on_force_radius" if force_on and not stalled else "strong_on_stall"
            elif near_goal and not admissible:
                level, reason = "off", "blocked_direct_goal"
            elif near_goal and not stalled:
                level, reason = "off", "near_goal_progressing"
            else:
                level, reason = "off", "far_field"

        self._near_goal_mode_active = bool(level == "strong")
        return str(level), str(reason)

    def _strong_v_ref(self, v_ref_nom, goal_dist):
        v_nom = float(v_ref_nom)
        d_hi = max(float(self.near_goal_speed_scale_radius), float(self.goal_handover_radius))
        d_lo = 0.3
        if d_hi <= d_lo + 1e-6:
            d_hi = d_lo + 1e-3
        alpha = float(np.clip((goal_dist - d_lo) / (d_hi - d_lo), 0.0, 1.0))
        v_eff = float(self.near_goal_min_speed + alpha * (v_nom - self.near_goal_min_speed))
        return float(min(v_nom, max(float(self.near_goal_min_speed), v_eff)))

    def _effective_v_ref(self, v_ref_nom, goal_dist, near_goal_mode_level):
        if str(near_goal_mode_level) != "strong":
            return float(v_ref_nom)
        return self._strong_v_ref(v_ref_nom, goal_dist)

    def _ensure_persistent_branch_cache_size(self, n_branches):
        n = int(max(0, n_branches))
        while len(self._persistent_z_prev) < n:
            self._persistent_z_prev.append(None)
        while len(self._persistent_lam_x_prev) < n:
            self._persistent_lam_x_prev.append(None)
        while len(self._persistent_lam_g_prev) < n:
            self._persistent_lam_g_prev.append(None)
        if len(self._persistent_z_prev) > n:
            self._persistent_z_prev = self._persistent_z_prev[:n]
        if len(self._persistent_lam_x_prev) > n:
            self._persistent_lam_x_prev = self._persistent_lam_x_prev[:n]
        if len(self._persistent_lam_g_prev) > n:
            self._persistent_lam_g_prev = self._persistent_lam_g_prev[:n]

    def _build_track_points(self, local_xy, goal_xy, n_split_eff):
        local = np.asarray(local_xy, dtype=float).reshape(2,)
        goal = np.asarray(goal_xy, dtype=float).reshape(2,)
        track = np.zeros((2, self.N), dtype=float)
        split = int(max(0, min(self.N, int(n_split_eff))))
        for k in range(self.N):
            track[:, k] = local if k <= split else goal
        return track

    def _init_persistent_backend(self):
        if self._persistent_setup_done:
            return True, ""
        if not _CASADI_AVAILABLE:
            return False, "casadi_missing"

        t0 = time.perf_counter()
        try:
            N = int(self.N)
            M = int(self.max_visible_obs)
            lb_u, ub_u = self._input_bounds()

            X = ca.SX.sym("X", 3, N + 1)
            U = ca.SX.sym("U", 2, N)
            Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

            # P = [x0(3), goal(2), track(2*N), v_ref(1), u_prev(2), vis_slots(M*6)]
            # vis_slot = [ox, oy, r, vx, vy, active]
            p_dim = 3 + 2 + 2 * N + 1 + 2 + 6 * M
            P = ca.SX.sym("P", p_dim)
            idx = 0
            p_x0 = P[idx : idx + 3]
            idx += 3
            p_goal = P[idx : idx + 2]
            idx += 2
            p_track = ca.reshape(P[idx : idx + 2 * N], 2, N)
            idx += 2 * N
            p_vref = P[idx]
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

            g = []
            lbg = []
            ubg = []

            g.append(X[:, 0] - p_x0)
            lbg.extend([0.0, 0.0, 0.0])
            ubg.extend([0.0, 0.0, 0.0])

            J = 0
            for k in range(N):
                g.append(X[:, k + 1] - self._discrete_ca(X[:, k], U[:, k]))
                lbg.extend([0.0, 0.0, 0.0])
                ubg.extend([0.0, 0.0, 0.0])

                if k == 0:
                    dv = U[0, k] - p_up[0]
                    dw = U[1, k] - p_up[1]
                else:
                    dv = U[0, k] - U[0, k - 1]
                    dw = U[1, k] - U[1, k - 1]
                J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
                J += self.wvel * ((U[0, k] - p_vref) ** 2)
                J += self.wtrack * ca.sumsqr(X[0:2, k] - p_track[:, k])

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

            solver = ca.nlpsol("control_tree_persistent", "ipopt", nlp, opts)

            z_size = int(Z.shape[0])
            nx = 3 * (N + 1)
            nu = 2 * N
            lbx = np.full((z_size,), -np.inf, dtype=float)
            ubx = np.full((z_size,), np.inf, dtype=float)
            for k in range(N):
                lbx[nx + 2 * k] = float(lb_u[0])
                ubx[nx + 2 * k] = float(ub_u[0])
                lbx[nx + 2 * k + 1] = float(lb_u[1])
                ubx[nx + 2 * k + 1] = float(ub_u[1])

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

    def _pack_persistent_params(self, x0, goal_xy, track_points, v_ref_nom, visible_obs):
        M = int(self.max_visible_obs)
        p = np.zeros((int(self._persistent_p_dim),), dtype=float)
        idx = 0

        x0a = np.asarray(x0, dtype=float).reshape(3,)
        goal = np.asarray(goal_xy, dtype=float).reshape(2,)
        track = np.asarray(track_points, dtype=float).reshape(2, self.N, order="F")
        up = np.asarray(self._u_prev_applied, dtype=float).reshape(2,)

        p[idx : idx + 3] = x0a
        idx += 3
        p[idx : idx + 2] = goal
        idx += 2
        p[idx : idx + 2 * self.N] = track.reshape(-1, order="F")
        idx += 2 * self.N
        p[idx] = float(v_ref_nom)
        idx += 1
        p[idx : idx + 2] = up
        idx += 2

        n_vis_active = 0
        for i in range(M):
            if i < len(visible_obs):
                obs = np.asarray(visible_obs[i], dtype=float).reshape(-1)
                ox = float(obs[0])
                oy = float(obs[1])
                rr = float(obs[2])
                vx = float(obs[3]) if obs.shape[0] >= 4 else 0.0
                vy = float(obs[4]) if obs.shape[0] >= 5 else 0.0
                active = 1.0
                n_vis_active += 1
            else:
                ox, oy, rr, vx, vy, active = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            p[idx : idx + 6] = [ox, oy, rr, vx, vy, active]
            idx += 6

        return p, int(n_vis_active)

    def _pack_persistent_guess(self, x_init, u_init):
        xg = np.asarray(x_init, dtype=float).reshape((3, self.N + 1), order="F")
        ug = np.asarray(u_init, dtype=float).reshape((2, self.N), order="F")
        return np.concatenate([xg.reshape(-1, order="F"), ug.reshape(-1, order="F")], axis=0)

    def _unpack_persistent_solution(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        x_sol = z[: int(self._persistent_nx)].reshape((3, self.N + 1), order="F")
        u_sol = z[int(self._persistent_nx) :].reshape((2, self.N), order="F")
        return x_sol, u_sol

    def _solve_branch_persistent(
        self,
        x0,
        local_xy,
        goal_xy,
        visible_obs,
        x_init,
        u_init,
        v_ref_nom,
        branch_idx,
        n_split=None,
    ):
        ok_setup, setup_err = self._init_persistent_backend()
        if not ok_setup:
            return (
                False,
                None,
                None,
                "persistent_setup_failed",
                setup_err,
                0.0,
                0,
                None,
            )

        n_split_eff = self.n_split if n_split is None else int(n_split)
        track_points = self._build_track_points(local_xy, goal_xy, n_split_eff)
        p, n_vis_active = self._pack_persistent_params(
            x0=x0,
            goal_xy=goal_xy,
            track_points=track_points,
            v_ref_nom=v_ref_nom,
            visible_obs=visible_obs,
        )

        if (
            branch_idx is not None
            and 0 <= int(branch_idx) < len(self._persistent_z_prev)
            and self._persistent_z_prev[int(branch_idx)] is not None
        ):
            z0 = np.asarray(self._persistent_z_prev[int(branch_idx)], dtype=float).reshape(-1)
        else:
            z0 = self._pack_persistent_guess(x_init, u_init)

        kwargs = {
            "x0": z0,
            "p": p,
            "lbg": self._persistent_lbg,
            "ubg": self._persistent_ubg,
            "lbx": self._persistent_lbx,
            "ubx": self._persistent_ubx,
        }
        bi = None if branch_idx is None else int(branch_idx)
        if self.warm_start_dual and bi is not None and 0 <= bi < len(self._persistent_lam_x_prev):
            lam_x_prev = self._persistent_lam_x_prev[bi]
            lam_g_prev = self._persistent_lam_g_prev[bi]
            if lam_x_prev is not None and len(lam_x_prev) == len(z0):
                kwargs["lam_x0"] = lam_x_prev
            if lam_g_prev is not None and len(lam_g_prev) == len(self._persistent_lbg):
                kwargs["lam_g0"] = lam_g_prev

        t0 = time.perf_counter()
        try:
            sol = self._persistent_solver(**kwargs)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            z_sol = np.array(sol["x"]).reshape(-1)
            x_sol, u_sol = self._unpack_persistent_solution(z_sol)

            if bi is not None and 0 <= bi < len(self._persistent_z_prev):
                self._persistent_z_prev[bi] = z_sol
                if self.warm_start_dual:
                    self._persistent_lam_x_prev[bi] = np.array(sol["lam_x"]).reshape(-1)
                    self._persistent_lam_g_prev[bi] = np.array(sol["lam_g"]).reshape(-1)

            cost = self._plan_cost_numpy(
                x_sol,
                u_sol,
                local_xy,
                goal_xy,
                v_ref_nom,
                n_split=n_split_eff,
            )
            n_constraints = int(self.N * n_vis_active)
            return True, x_sol, u_sol, "optimal", "", solve_ms, n_constraints, cost
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            if bi is not None and 0 <= bi < len(self._persistent_z_prev):
                self._persistent_z_prev[bi] = None
                self._persistent_lam_x_prev[bi] = None
                self._persistent_lam_g_prev[bi] = None
            return False, None, None, "infeasible", str(exc), solve_ms, int(self.N * n_vis_active), None

    def _solve_branch_opti(self, x0, local_xy, goal_xy, visible_obs, x_init, u_init, v_ref_nom, n_split=None):
        if not _CASADI_AVAILABLE:
            return False, None, None, "casadi_missing", "CasADi is not installed", 0.0, 0, None

        N = self.N
        lb_u, ub_u = self._input_bounds()
        n_split_eff = self.n_split if n_split is None else int(n_split)

        opti = ca.Opti()
        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)

        x0_dm = ca.DM(np.asarray(x0, dtype=float).reshape(3))
        local_dm = ca.DM(np.asarray(local_xy, dtype=float).reshape(2))
        goal_dm = ca.DM(np.asarray(goal_xy, dtype=float).reshape(2))
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
            J += self.wvel * ((U[0, k] - float(v_ref_nom)) ** 2)
            gk = local_dm if k <= int(n_split_eff) else goal_dm
            J += self.wtrack * ca.sumsqr(X[0:2, k] - gk)
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
            cost = self._plan_cost_numpy(
                x_sol,
                u_sol,
                local_xy,
                goal_xy,
                v_ref_nom,
                n_split=n_split_eff,
            )
            return True, x_sol, u_sol, "optimal", "", solve_ms, n_constraints, cost
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            return False, None, None, "infeasible", str(exc), solve_ms, n_constraints, None

    def _solve_branch(
        self,
        x0,
        local_xy,
        goal_xy,
        visible_obs,
        x_init,
        u_init,
        v_ref_nom,
        n_split=None,
        branch_idx=None,
    ):
        if str(self.solve_backend) == "persistent_casadi":
            out = self._solve_branch_persistent(
                x0=x0,
                local_xy=local_xy,
                goal_xy=goal_xy,
                visible_obs=visible_obs,
                x_init=x_init,
                u_init=u_init,
                v_ref_nom=v_ref_nom,
                branch_idx=branch_idx,
                n_split=n_split,
            )
            if (not out[0]) and self.persistent_fallback_opti and out[3] == "persistent_setup_failed":
                return self._solve_branch_opti(
                    x0=x0,
                    local_xy=local_xy,
                    goal_xy=goal_xy,
                    visible_obs=visible_obs,
                    x_init=x_init,
                    u_init=u_init,
                    v_ref_nom=v_ref_nom,
                    n_split=n_split,
                )
            return out

        return self._solve_branch_opti(
            x0=x0,
            local_xy=local_xy,
            goal_xy=goal_xy,
            visible_obs=visible_obs,
            x_init=x_init,
            u_init=u_init,
            v_ref_nom=v_ref_nom,
            n_split=n_split,
        )

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()

        x = np.asarray(robot_state, dtype=float).reshape(-1)
        x0 = np.array([x[0], x[1], self._normalize_angle(x[2])], dtype=float)

        u_ref = np.asarray(control_ref.get("u_ref", np.zeros((2, 1))), dtype=float).reshape(-1, 1)
        u_ref = self._clip_input(u_ref)
        self.last_u_ref = u_ref
        v_ref_nom = max(abs(float(u_ref[0, 0])), self.v_ref_default)

        goal_xy = self._goal_xy(x0, control_ref.get("goal", None))

        # Reuse occlusion front-end for fair visibility filtering.
        visible_obs, occ_scenarios = self._occ_utils._filter_visible_and_build_occ(
            np.asarray(robot_state, dtype=float).reshape(-1, 1),
            obs_list,
        )
        visible_obs = self._nearest_visible_obs(visible_obs, x0)
        self.occlusion_scenarios = list(occ_scenarios)

        goal_heading = float(np.arctan2(goal_xy[1] - x0[1], goal_xy[0] - x0[0]))
        clusters = self._cluster_visible_obs(visible_obs, x0, goal_heading)
        branches = self._build_branches(x0, goal_xy, clusters)
        B = len(branches)
        self._ensure_persistent_branch_cache_size(B)

        goal_dist = float(np.linalg.norm(np.asarray(x0[:2], dtype=float) - np.asarray(goal_xy, dtype=float)))
        self._goal_dist_hist.append(goal_dist)
        progress_window_drop = 0.0
        progress_stalled = False
        if len(self._goal_dist_hist) >= 2:
            progress_window_drop = float(self._goal_dist_hist[0] - self._goal_dist_hist[-1])
            progress_stalled = bool(progress_window_drop < self.near_goal_progress_min_drop)

        direct_goal_clearance = self._direct_goal_clearance(x0, goal_xy, visible_obs)
        direct_goal_admissible = bool(direct_goal_clearance > 0.0)
        nearest_visible_obs_clearance = self._nearest_visible_obs_clearance(x0, visible_obs)

        near_goal_mode_level, goal_handover_reason = self._update_near_goal_mode(
            goal_dist=goal_dist,
            direct_goal_admissible=direct_goal_admissible,
            progress_stalled=progress_stalled,
        )
        near_goal_mode_active = bool(near_goal_mode_level == "strong")
        effective_n_split = 0 if near_goal_mode_active else int(self.n_split)
        effective_v_ref = self._effective_v_ref(
            v_ref_nom=v_ref_nom,
            goal_dist=goal_dist,
            near_goal_mode_level=near_goal_mode_level,
        )

        if self._x_prev_plans is None or self._u_prev_plans is None:
            self._x_prev_plans = [None] * B
            self._u_prev_plans = [None] * B
        elif len(self._x_prev_plans) != B or len(self._u_prev_plans) != B:
            self._x_prev_plans = [None] * B
            self._u_prev_plans = [None] * B

        branch_statuses = []
        branch_costs = []
        branch_gap_angles = []
        branch_gap_widths = []
        branch_guidance_points = []
        solve_ms_per_branch = []
        num_constraints_per_branch = []
        x_sols = [None] * B
        u_sols = [None] * B

        total_solver_ms = 0.0
        selected_branch = None
        selected_cost = None
        selected_u = None

        for bi in range(B):
            x_guess, u_guess = self._shift_or_init_branch_plan(x0, u_ref, bi)
            local_xy_nom = np.asarray(branches[bi]["local_point"], dtype=float).reshape(2,)
            local_xy = np.asarray(goal_xy, dtype=float).reshape(2,) if near_goal_mode_active else local_xy_nom

            branch_guidance_points.append([float(local_xy[0]), float(local_xy[1])])
            branch_gap_angles.append(float(branches[bi]["gap_angle"]))
            branch_gap_widths.append(float(branches[bi]["gap_width"]))

            ok, x_sol, u_sol, st, ex, solve_ms, ncons, cost = self._solve_branch(
                x0=x0,
                local_xy=local_xy,
                goal_xy=goal_xy,
                visible_obs=visible_obs,
                x_init=x_guess,
                u_init=u_guess,
                v_ref_nom=effective_v_ref,
                n_split=effective_n_split,
                branch_idx=bi,
            )

            total_solver_ms += float(solve_ms)
            branch_statuses.append(st)
            branch_costs.append(None if cost is None else float(cost))
            solve_ms_per_branch.append(float(solve_ms))
            num_constraints_per_branch.append(int(ncons))

            if ok:
                x_sols[bi] = x_sol
                u_sols[bi] = u_sol
                if (selected_branch is None) or (cost < selected_cost):
                    selected_branch = bi
                    selected_cost = float(cost)
                    selected_u = self._clip_input(u_sol[:, 0].reshape(-1, 1))
            else:
                self.last_qp_exception = str(ex)

        self.last_qp_solve_time_ms = float(total_solver_ms)
        self.last_num_constraints = int(max(num_constraints_per_branch) if num_constraints_per_branch else 0)
        self.last_qp_status_raw = "optimal" if selected_branch is not None else "infeasible"

        goal_heading = float(np.arctan2(goal_xy[1] - x0[1], goal_xy[0] - x0[0]))
        goal_heading_error = float(self._angle_wrap(goal_heading - float(x0[2])))

        if selected_branch is None:
            self.status = "infeasible"
            self.last_intervention = "control_tree_mpc"
            self.last_u = self._stop_input()
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_profile = {
                "backend": "sequential",
                "solver_backend": str(self.solve_backend),
                "setup_ms": (None if self._persistent_setup_ms is None else float(self._persistent_setup_ms)),
                "total_ms": float(self.last_total_compute_time_ms),
                "wall_clock_total_ms": float(self.last_total_compute_time_ms),
                "solve_ms_total": float(self.last_qp_solve_time_ms),
                "solve_ms_per_branch": [float(x) for x in solve_ms_per_branch],
                "solve_ms_total_branch_sum": float(np.sum(solve_ms_per_branch)),
                "n_visible_obs": int(len(visible_obs)),
                "n_branches_generated": int(B),
                "n_branches_feasible": 0,
                "branch_statuses": list(branch_statuses),
                "branch_costs": list(branch_costs),
                "branch_guidance_points": list(branch_guidance_points),
                "selected_branch": None,
                "raw_solver_status": self.last_qp_status_raw,
                "n_clusters": int(len(clusters)),
                "cluster_centroids": [[float(c["centroid"][0]), float(c["centroid"][1])] for c in clusters],
                "cluster_angular_spans": [[float(c["angle_span"][0]), float(c["angle_span"][1])] for c in clusters],
                "goal_dist": float(goal_dist),
                "near_goal_mode_active": bool(near_goal_mode_active),
                "near_goal_mode_level": str(near_goal_mode_level),
                "goal_handover_radius": float(self.goal_handover_radius),
                "goal_handover_hysteresis": float(self.goal_handover_hysteresis),
                "direct_goal_admissible": bool(direct_goal_admissible),
                "direct_goal_clearance": float(direct_goal_clearance),
                "nearest_visible_obs_clearance": float(nearest_visible_obs_clearance),
                "progress_stalled": bool(progress_stalled),
                "progress_window_drop": float(progress_window_drop),
                "effective_n_split": int(effective_n_split),
                "effective_v_ref": float(effective_v_ref),
                "u_ref_0": float(u_ref[0, 0]),
                "u_ref_1": float(u_ref[1, 0]),
                "u_cmd_0": float(self.last_u[0, 0]),
                "u_cmd_1": float(self.last_u[1, 0]),
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "local_xy": None,
                "goal_heading_error": float(goal_heading_error),
                "selected_local_heading_error": None,
                "selected_local_xy": None,
                "local_goal_deviation": None,
                "goal_handover_reason": str(goal_handover_reason),
                "goal_handover_reason_detailed": str(goal_handover_reason),
                "num_constraints": int(self.last_num_constraints),
            }
            return None

        self.status = "optimal"
        self.last_intervention = "control_tree_mpc"
        self.last_u = selected_u
        self._u_prev_applied = selected_u
        self._x_prev_plans = x_sols
        self._u_prev_plans = u_sols

        selected_local_xy = np.asarray(branch_guidance_points[selected_branch], dtype=float).reshape(2,)
        local_goal_deviation = float(np.linalg.norm(selected_local_xy - np.asarray(goal_xy, dtype=float).reshape(2,)))
        local_heading = float(np.arctan2(selected_local_xy[1] - x0[1], selected_local_xy[0] - x0[0]))
        selected_local_heading_error = float(self._angle_wrap(local_heading - float(x0[2])))

        self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
        self.last_profile = {
            "backend": "sequential",
            "solver_backend": str(self.solve_backend),
            "setup_ms": (None if self._persistent_setup_ms is None else float(self._persistent_setup_ms)),
            "total_ms": float(self.last_total_compute_time_ms),
            "wall_clock_total_ms": float(self.last_total_compute_time_ms),
            "solve_ms_total": float(self.last_qp_solve_time_ms),
            "solve_ms_per_branch": [float(x) for x in solve_ms_per_branch],
            "solve_ms_total_branch_sum": float(np.sum(solve_ms_per_branch)),
            "n_visible_obs": int(len(visible_obs)),
            "n_branches_generated": int(B),
            "n_branches_feasible": int(sum(1 for s in branch_statuses if s == "optimal")),
            "branch_statuses": list(branch_statuses),
            "branch_costs": list(branch_costs),
            "branch_guidance_points": list(branch_guidance_points),
            "branch_gap_angles": list(branch_gap_angles),
            "branch_gap_widths": list(branch_gap_widths),
            "selected_branch": int(selected_branch),
            "selected_cost": float(selected_cost),
            "selected_gap_angle": float(branch_gap_angles[selected_branch]),
            "selected_gap_width": float(branch_gap_widths[selected_branch]),
            "raw_solver_status": self.last_qp_status_raw,
            "n_clusters": int(len(clusters)),
            "cluster_centroids": [[float(c["centroid"][0]), float(c["centroid"][1])] for c in clusters],
            "cluster_angular_spans": [[float(c["angle_span"][0]), float(c["angle_span"][1])] for c in clusters],
            "goal_dist": float(goal_dist),
            "near_goal_mode_active": bool(near_goal_mode_active),
            "near_goal_mode_level": str(near_goal_mode_level),
            "goal_handover_radius": float(self.goal_handover_radius),
            "goal_handover_hysteresis": float(self.goal_handover_hysteresis),
            "direct_goal_admissible": bool(direct_goal_admissible),
            "direct_goal_clearance": float(direct_goal_clearance),
            "nearest_visible_obs_clearance": float(nearest_visible_obs_clearance),
            "progress_stalled": bool(progress_stalled),
            "progress_window_drop": float(progress_window_drop),
            "effective_n_split": int(effective_n_split),
            "effective_v_ref": float(effective_v_ref),
            "u_ref_0": float(u_ref[0, 0]),
            "u_ref_1": float(u_ref[1, 0]),
            "u_cmd_0": float(selected_u[0, 0]),
            "u_cmd_1": float(selected_u[1, 0]),
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "local_xy": [float(selected_local_xy[0]), float(selected_local_xy[1])],
            "goal_heading_error": float(goal_heading_error),
            "selected_local_heading_error": float(selected_local_heading_error),
            "selected_local_xy": [float(selected_local_xy[0]), float(selected_local_xy[1])],
            "local_goal_deviation": float(local_goal_deviation),
            "goal_handover_reason": str(goal_handover_reason),
            "goal_handover_reason_detailed": str(goal_handover_reason),
            "num_constraints": int(self.last_num_constraints),
        }
        return selected_u
