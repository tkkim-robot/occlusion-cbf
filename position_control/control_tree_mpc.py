import time
from itertools import product

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
    Literature-like, framework-adapted Control-Tree MPC baseline.

    This controller replaces the earlier gap-branch baseline with an explicit
    hypothesis tree that is closer in spirit to Phiquepal & Toussaint (ICRA 2021):
      - a shared control trunk,
      - multiple branch tails,
      - belief-weighted branch costs,
      - robust trunk safety against all active hypotheses.

    Benchmark adaptation choices:
      - Discrete hypotheses are built from the top-K occlusion scenarios rather than
        from a dedicated perception classifier over symbolic states.
      - Each selected occlusion scenario contributes one binary latent variable:
        hidden agent present / absent. Branches enumerate all binary combinations.
      - Branch-specific hidden-agent risk is encoded with simple tangent-ray risk
        regions, not the original KOMO / distributed ADMM stack.
      - The optimization is solved as one joint nonlinear program with an explicit
        shared trunk. This preserves the control-tree structure without requiring
        the original C++ stack.

    This is still an adaptation, not a claim of exact paper reproduction.
    """

    def __init__(self, robot, robot_spec, num_obs=30):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = int(num_obs)

        self.model = str(robot_spec.get("model", "")).strip()
        if self.model not in {"DoubleIntegrator2D", "Unicycle2D", "DynamicUnicycle2D"}:
            raise ValueError(
                f"ControlTreeMPC currently supports DoubleIntegrator2D, Unicycle2D and DynamicUnicycle2D, got `{self.model}`"
            )
        self._n_state, self._u_dim = self._dims()

        self.dt = float(getattr(robot, "dt", robot_spec.get("dt", 0.05)))
        self.robot_radius = float(robot_spec.get("radius", 0.25))
        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))

        cfg = robot_spec.setdefault("control_tree_mpc", {})

        self.dt_plan = float(cfg.get("dt_plan", 0.25))
        self.Th = float(cfg.get("Th", 3.0))
        n_from_h = int(np.round(self.Th / max(self.dt_plan, 1e-6)))
        self.N = max(4, int(cfg.get("N", n_from_h)))

        self.n_split = int(cfg.get("n_split", 3))
        self.n_split = max(1, min(self.N - 1, self.n_split))
        self.tail_horizon = int(self.N - self.n_split)

        self.forward_only = bool(cfg.get("forward_only", False))
        self.v_plan_min = float(cfg.get("v_plan_min", 0.15 if self.forward_only else 0.0))
        self.max_visible_obs = int(cfg.get("max_visible_obs", self.num_obs))
        self.max_visible_obs = max(1, self.max_visible_obs)
        self.margin_obs = float(cfg.get("margin_obs", 0.05))
        self.margin_risk = float(cfg.get("margin_risk", 0.05))

        # Shared-guidance heuristic. This is only used to generate a common trunk
        # reference and is not the branch semantics itself.
        self.gap_lookahead = float(cfg.get("gap_lookahead", 2.5))
        self.cluster_merge_distance = float(cfg.get("cluster_merge_distance", 0.8))
        self.forward_fov_deg_for_guidance = float(cfg.get("forward_fov_deg_for_guidance", 180.0))
        self.min_gap_width = cfg.get("min_gap_width", 0.25)
        if self.min_gap_width is not None:
            self.min_gap_width = float(self.min_gap_width)
        self.goal_handover_radius = float(cfg.get("goal_handover_radius", 1.0))

        # Hypothesis-tree parameters.
        self.n_occ_hypotheses = int(cfg.get("n_occ_hypotheses", 2))
        self.n_occ_hypotheses = max(0, min(3, self.n_occ_hypotheses))
        self.max_branches = max(1, 2 ** self.n_occ_hypotheses)
        self.hidden_speed = float(
            cfg.get(
                "hidden_speed",
                robot_spec.get("v_adv_max_occ", robot_spec.get("v_obs_max", 0.5)),
            )
        )
        self.risk_regions_per_tangent = int(cfg.get("risk_regions_per_tangent", 2))
        self.risk_regions_per_tangent = max(1, self.risk_regions_per_tangent)
        self.drisk = float(cfg.get("drisk", 0.7))
        self.risk_sigma = float(cfg.get("risk_sigma", 1e-4))
        self.rrisk_max = cfg.get("rrisk_max", 1.5)
        if self.rrisk_max is not None:
            self.rrisk_max = float(self.rrisk_max)
        self.min_v_for_risk = float(cfg.get("min_v_for_risk", 0.3))
        self.risk_time_model = str(cfg.get("risk_time_model", "distance_over_vref")).strip().lower()
        if self.risk_time_model not in {"distance_over_vref", "nominal_rollout"}:
            self.risk_time_model = "distance_over_vref"
        self.nominal_k_heading = float(cfg.get("nominal_k_heading", 2.0))
        self.max_branch_risk_regions = int(
            cfg.get(
                "max_branch_risk_regions",
                max(1, self.n_occ_hypotheses * 2 * self.risk_regions_per_tangent),
            )
        )

        # Belief heuristic for ranking / weighting hypotheses.
        self.belief_prob_scale = float(cfg.get("belief_prob_scale", 0.45))
        self.belief_prob_min = float(cfg.get("belief_prob_min", 0.05))
        self.belief_prob_max = float(cfg.get("belief_prob_max", 0.55))
        self.belief_dist_scale = float(cfg.get("belief_dist_scale", 4.0))
        self.belief_path_scale = float(cfg.get("belief_path_scale", 1.0))
        self.belief_align_floor = float(cfg.get("belief_align_floor", 0.25))
        self.belief_align_power = float(cfg.get("belief_align_power", 1.5))
        self.hypothesis_score_min = float(cfg.get("hypothesis_score_min", 0.02))

        # Cost shaping.
        self.wgoal = float(cfg.get("wgoal", 3.5))
        self.wvel = float(cfg.get("wvel", 5.0))
        self.wacc = float(cfg.get("wacc", 1.8))
        self.wtrack_shared = float(cfg.get("wtrack_shared", 2.0))
        self.wtrack_tail = float(cfg.get("wtrack_tail", 0.5))
        self.branch_align_weight = float(cfg.get("branch_align_weight", 1.0))
        self.branch_width_weight = float(cfg.get("branch_width_weight", 0.35))
        self.branch_clearance_weight = float(cfg.get("branch_clearance_weight", 0.9))
        self.lambda_w = float(cfg.get("lambda_w", 1.0))
        self.v_ref_default = float(cfg.get("v_ref_default", 0.5))
        self.branch_zero_prob_reg = float(cfg.get("branch_zero_prob_reg", 1e-3))

        self.max_iter = int(cfg.get("max_iter", 150))
        self.solver_tol = float(cfg.get("solver_tol", 1e-4))
        self.solver_acceptable_tol = float(cfg.get("solver_acceptable_tol", 1e-2))
        self.solver_acceptable_iter = int(cfg.get("solver_acceptable_iter", 8))
        self.print_solver = bool(cfg.get("print_solver", False))
        self.solver_expand = bool(cfg.get("solver_expand", False))
        self.solve_backend = str(cfg.get("solver_backend", "joint_persistent")).strip().lower()
        if self.solve_backend in {"persistent_casadi", "joint_persistent", "persistent"}:
            self.solve_backend = "joint_persistent"
        elif self.solve_backend not in {"joint_persistent", "opti"}:
            self.solve_backend = "joint_persistent"
        self.persistent_fallback_opti = bool(cfg.get("persistent_fallback_opti", False))
        self.warm_start_dual = bool(cfg.get("warm_start_dual", True))
        self.ipopt_linear_solver = str(cfg.get("ipopt_linear_solver", "mumps")).strip()

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=None,
        )

        # Runtime state.
        self._u_prev_applied = np.zeros((2, 1), dtype=float)
        self._z_prev = None
        self._lam_x_prev = None
        self._lam_g_prev = None

        # Diagnostics contract.
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

        # Persistent solver caches.
        self._persistent_setup_done = False
        self._persistent_setup_ms = None
        self._persistent_solver = None
        self._persistent_lbx = None
        self._persistent_ubx = None
        self._persistent_lbg = None
        self._persistent_ubg = None
        self._persistent_p_dim = None
        self._z_layout = None

    def _input_bounds(self):
        lb, ub = super()._input_bounds()
        if self.model == "Unicycle2D":
            lb = np.asarray(lb, dtype=float).copy()
            ub = np.asarray(ub, dtype=float).copy()
            lb[0] = max(float(lb[0]), float(self.v_plan_min))
        return lb, ub

    # ---------------------------------------------------------------------
    # Shared guidance helper
    # ---------------------------------------------------------------------
    def _cluster_visible_obs(self, visible_obs, x0, goal_heading):
        if visible_obs is None or len(visible_obs) == 0:
            return []
        obs = np.asarray(visible_obs, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        centers = obs[:, :2]
        radii = obs[:, 2] + self.robot_radius + self.margin_obs
        used = np.zeros(centers.shape[0], dtype=bool)
        clusters = []
        for i in range(centers.shape[0]):
            if used[i]:
                continue
            stack = [i]
            used[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(centers.shape[0]):
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
                    "centroid": centroid,
                    "radius": float(r_bound),
                    "angle_span": (float(alpha - half), float(alpha + half)),
                }
            )
        return clusters

    def _gap_candidates(self, x0, goal_xy, clusters):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        goal_heading = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
        half_fov = np.deg2rad(max(10.0, self.forward_fov_deg_for_guidance)) * 0.5

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

        min_gap = max(0.0, float(self.min_gap_width)) if self.min_gap_width is not None else 0.0
        candidates = []
        for lo, hi in gaps:
            width = hi - lo
            if width < min_gap:
                continue
            ctr = 0.5 * (lo + hi)
            world_ang = goal_heading + ctr
            local_point = p + self.gap_lookahead * np.array([np.cos(world_ang), np.sin(world_ang)], dtype=float)
            candidates.append({
                "gap_angle": float(ctr),
                "gap_width": float(width),
                "local_point": local_point,
            })
        return goal_heading, candidates

    def _select_shared_guidance(self, x0, goal_xy, visible_obs):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        if float(np.linalg.norm(goal_xy - p)) <= float(self.goal_handover_radius):
            return goal_xy, {"guidance_source": "goal_near"}
        goal_heading = float(np.arctan2(goal_xy[1] - p[1], goal_xy[0] - p[0]))
        clusters = self._cluster_visible_obs(visible_obs, x0, goal_heading)
        _, cands = self._gap_candidates(x0, goal_xy, clusters)
        if len(cands) == 0:
            return goal_xy, {"guidance_source": "goal_fallback", "n_clusters": int(len(clusters))}
        i0 = int(np.argmin([abs(c["gap_angle"]) for c in cands]))
        return np.asarray(cands[i0]["local_point"], dtype=float).reshape(2,), {
            "guidance_source": "gap",
            "n_clusters": int(len(clusters)),
            "gap_angle": float(cands[i0]["gap_angle"]),
            "gap_width": float(cands[i0]["gap_width"]),
        }

    def _select_branch_target(self, x0, goal_xy, visible_obs, risk_regions):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        if float(np.linalg.norm(goal_xy - p)) <= float(self.goal_handover_radius):
            return goal_xy, {"target_source": "goal_near"}

        goal_heading = float(np.arctan2(goal_xy[1] - p[1], goal_xy[0] - p[0]))
        clusters = self._cluster_visible_obs(visible_obs, x0, goal_heading)
        _, cands = self._gap_candidates(x0, goal_xy, clusters)
        if len(cands) == 0:
            return goal_xy, {"target_source": "goal_fallback", "n_clusters": int(len(clusters))}

        regs = list(risk_regions or [])
        best_i = 0
        best_score = -np.inf
        for i, cand in enumerate(cands):
            align = max(0.0, float(np.cos(float(cand["gap_angle"]))))
            width = float(cand["gap_width"])
            clear_norm = 1.0
            if len(regs) > 0:
                lp = np.asarray(cand["local_point"], dtype=float).reshape(2,)
                clear = min(
                    float(np.linalg.norm(lp - np.asarray(c, dtype=float).reshape(2,)) - float(rr))
                    for c, rr in regs
                )
                clear_norm = float(np.clip(clear / max(self.gap_lookahead, 1e-3), 0.0, 1.5))
            score = (
                self.branch_align_weight * align
                + self.branch_width_weight * width
                + self.branch_clearance_weight * clear_norm
            )
            if score > best_score:
                best_score = float(score)
                best_i = int(i)

        chosen = cands[best_i]
        return np.asarray(chosen["local_point"], dtype=float).reshape(2,), {
            "target_source": "gap_branch",
            "n_clusters": int(len(clusters)),
            "gap_angle": float(chosen["gap_angle"]),
            "gap_width": float(chosen["gap_width"]),
            "score": float(best_score),
        }

    # ---------------------------------------------------------------------
    # Hypothesis construction
    # ---------------------------------------------------------------------
    def _nominal_rollout_positions(self, x0, guidance_xy, v_ref_nom):
        x = np.asarray(x0, dtype=float).reshape(self._n_state,)
        points = np.zeros((self.N + 1, 2), dtype=float)
        points[0] = x[:2]
        for k in range(self.N):
            u_cmd = self._guidance_input_np(x, guidance_xy, v_ref_nom, k_heading=self.nominal_k_heading)
            x = self._discrete_np(x, u_cmd)
            points[k + 1] = x[:2]
        return points

    def _risk_regions_from_scenario(self, scenario, v_ref_nom, nominal_points):
        hidden_speed = float(scenario.get("v_adv_max", self.hidden_speed))
        if hidden_speed <= 1e-9:
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

        rays = []
        for t in (np.asarray(t1, dtype=float).reshape(2,), np.asarray(t2, dtype=float).reshape(2,)):
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
                center = t + d * (self.drisk * float(i - 1))
                if self.risk_time_model == "nominal_rollout" and nominal_points is not None:
                    dists = np.linalg.norm(nominal_points - center[None, :], axis=1)
                    idx = int(np.argmin(dists))
                    travel_t = idx * self.dt_plan + float(dists[idx]) / v_nom
                else:
                    travel_t = float(np.linalg.norm(center - p)) / v_nom
                rr = travel_t * hidden_speed + r_obs + self.risk_sigma
                if self.rrisk_max is not None:
                    rr = min(rr, self.rrisk_max)
                if np.isfinite(rr) and rr > 0.0:
                    regions.append((center, float(rr)))
        return regions

    def _scenario_score(self, x0, goal_xy, scenario, risk_regions, nominal_points):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        c = np.asarray(scenario.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
        r_obs = float(scenario.get("obs_radius", 0.0))

        d = max(0.0, float(np.linalg.norm(c - p) - r_obs))
        dist_factor = np.exp(-d / max(self.belief_dist_scale, 1e-6))

        goal_vec = g - p
        goal_n = float(np.linalg.norm(goal_vec))
        if goal_n > 1e-9:
            goal_vec = goal_vec / goal_n
        scen_vec = c - p
        scen_n = float(np.linalg.norm(scen_vec))
        if scen_n > 1e-9:
            scen_vec = scen_vec / scen_n
        align = 0.0 if (goal_n <= 1e-9 or scen_n <= 1e-9) else float(np.clip(goal_vec @ scen_vec, -1.0, 1.0))
        align = float(self.belief_align_floor + (1.0 - self.belief_align_floor) * max(0.0, align))
        align = float(align ** self.belief_align_power)

        if risk_regions and nominal_points is not None:
            dmins = []
            for center, rr in risk_regions:
                dmins.append(float(np.min(np.linalg.norm(nominal_points - center[None, :], axis=1)) - rr))
            path_clear = min(dmins) if len(dmins) > 0 else np.inf
        else:
            path_clear = np.inf
        if np.isfinite(path_clear):
            path_factor = np.exp(-max(0.0, path_clear) / max(self.belief_path_scale, 1e-6))
        else:
            path_factor = 0.5

        width = 0.0
        if scenario.get("front1", None) is not None and scenario.get("front2", None) is not None:
            width = float(np.linalg.norm(np.asarray(scenario["front2"]) - np.asarray(scenario["front1"])))
        width_factor = float(np.clip(width / max(2.0 * self.robot_radius, 1e-3), 0.5, 2.0))

        score = float(dist_factor * align * (0.25 + 0.75 * path_factor) * width_factor)
        p_i = float(np.clip(self.belief_prob_scale * score, self.belief_prob_min, self.belief_prob_max))
        return score, p_i

    def _select_hypothesis_scenarios(self, x0, goal_xy, occ_scenarios, v_ref_nom, nominal_points):
        scored = []
        for sc in occ_scenarios:
            risk_regions = self._risk_regions_from_scenario(sc, v_ref_nom, nominal_points)
            score, p_i = self._scenario_score(x0, goal_xy, sc, risk_regions, nominal_points)
            scored.append({
                "scenario": sc,
                "risk_regions": risk_regions,
                "score": float(score),
                "p": float(p_i),
            })
        scored = [s for s in scored if s["score"] >= float(self.hypothesis_score_min)]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: int(self.n_occ_hypotheses)]

    @staticmethod
    def _branch_masks(k):
        if k <= 0:
            return [()]
        return [tuple(int(v) for v in bits) for bits in product([0, 1], repeat=int(k))]

    def _build_branch_hypotheses(self, selected_scenarios):
        k = len(selected_scenarios)
        masks = self._branch_masks(k)
        branches = []
        for mask in masks:
            prob = 1.0
            risk_regions = []
            active_ids = []
            for i, bit in enumerate(mask):
                p_i = float(selected_scenarios[i]["p"])
                prob *= p_i if bit else (1.0 - p_i)
                if bit:
                    active_ids.append(int(i))
                    risk_regions.extend(selected_scenarios[i]["risk_regions"])
            branches.append({
                "mask": tuple(mask),
                "prob": float(prob),
                "risk_regions": list(risk_regions),
                "active_ids": list(active_ids),
            })
        total = float(sum(b["prob"] for b in branches))
        if total > 1e-9:
            for b in branches:
                b["prob"] = float(b["prob"] / total)
        if len(branches) == 0:
            branches = [{"mask": tuple(), "prob": 1.0, "risk_regions": [], "active_ids": []}]
        return branches

    # ---------------------------------------------------------------------
    # Joint NLP backend
    # ---------------------------------------------------------------------
    def _layout_sizes(self):
        nx = self._n_state
        L = int(self.n_split)
        Nt = int(self.tail_horizon)
        B = int(self.max_branches)
        return {
            "nx": nx,
            "L": L,
            "Nt": Nt,
            "B": B,
            "Xs": nx * (L + 1),
            "Us": 2 * L,
            "Xt": nx * (Nt + 1),
            "Ut": 2 * Nt,
        }

    def _init_joint_persistent_backend(self):
        if self._persistent_setup_done:
            return True, ""
        if not _CASADI_AVAILABLE:
            return False, "casadi_missing"

        t0 = time.perf_counter()
        try:
            sz = self._layout_sizes()
            nx = int(sz["nx"])
            L = int(sz["L"])
            Nt = int(sz["Nt"])
            B = int(sz["B"])
            M = int(self.max_visible_obs)
            R = int(self.max_branch_risk_regions)
            lb_u, ub_u = self._input_bounds()

            Xs = ca.SX.sym("Xs", nx, L + 1)
            Us = ca.SX.sym("Us", 2, L)
            Xt = [ca.SX.sym(f"Xt_{b}", nx, Nt + 1) for b in range(B)]
            Ut = [ca.SX.sym(f"Ut_{b}", 2, Nt) for b in range(B)]

            z_parts = [ca.reshape(Xs, -1, 1), ca.reshape(Us, -1, 1)]
            for b in range(B):
                z_parts.append(ca.reshape(Xt[b], -1, 1))
                z_parts.append(ca.reshape(Ut[b], -1, 1))
            Z = ca.vertcat(*z_parts)

            # P = [x0(nx), goal(2), guidance(2), branch_targets(B*2), vref(1), up_prev(2), branch_probs(B), vis(M*6), risk(B*R*4)]
            p_dim = nx + 2 + 2 + 2 * B + 1 + 2 + B + 6 * M + 4 * B * R
            P = ca.SX.sym("P", p_dim)
            idx = 0
            p_x0 = P[idx : idx + nx]
            idx += nx
            p_goal = P[idx : idx + 2]
            idx += 2
            p_guidance = P[idx : idx + 2]
            idx += 2
            p_branch_targets = [P[idx + 2 * b : idx + 2 * (b + 1)] for b in range(B)]
            idx += 2 * B
            p_vref = P[idx]
            idx += 1
            p_up = P[idx : idx + 2]
            idx += 2
            p_branch_probs = P[idx : idx + B]
            idx += B

            vis_params = []
            for _ in range(M):
                vis_params.append((P[idx + 0], P[idx + 1], P[idx + 2], P[idx + 3], P[idx + 4], P[idx + 5]))
                idx += 6

            risk_params = []
            for _b in range(B):
                branch_slots = []
                for _r in range(R):
                    branch_slots.append((P[idx + 0], P[idx + 1], P[idx + 2], P[idx + 3]))
                    idx += 4
                risk_params.append(branch_slots)

            g = []
            lbg = []
            ubg = []
            J = 0

            # Shared trunk.
            g.append(Xs[:, 0] - p_x0)
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)
            for k in range(L):
                g.append(Xs[:, k + 1] - self._discrete_ca(Xs[:, k], Us[:, k]))
                lbg.extend([0.0] * nx)
                ubg.extend([0.0] * nx)

                if k == 0:
                    dv = Us[0, k] - p_up[0]
                    dw = Us[1, k] - p_up[1]
                else:
                    dv = Us[0, k] - Us[0, k - 1]
                    dw = Us[1, k] - Us[1, k - 1]
                J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
                J += self.wvel * ((self._stage_speed_ca(Xs, Us, k) - p_vref) ** 2)
                J += self.wtrack_shared * ca.sumsqr(Xs[0:2, k] - p_guidance)

            J += 0.5 * self.wtrack_shared * ca.sumsqr(Xs[0:2, L] - p_guidance)

            # Visible constraints on shared trunk.
            for k in range(1, L + 1):
                tk = float(k) * float(self.dt_plan)
                for (ox, oy, rr, vx, vy, active) in vis_params:
                    cx = ox + vx * tk
                    cy = oy + vy * tk
                    safe2 = (self.robot_radius + rr + self.margin_obs) ** 2
                    expr = active * (safe2 - ((Xs[0, k] - cx) ** 2 + (Xs[1, k] - cy) ** 2))
                    g.append(ca.vertcat(expr))
                    lbg.append(-np.inf)
                    ubg.append(0.0)

            # Branch tails.
            for b in range(B):
                weight = p_branch_probs[b] + float(self.branch_zero_prob_reg)
                g.append(Xt[b][:, 0] - Xs[:, L])
                lbg.extend([0.0] * nx)
                ubg.extend([0.0] * nx)

                for k in range(Nt):
                    g.append(Xt[b][:, k + 1] - self._discrete_ca(Xt[b][:, k], Ut[b][:, k]))
                    lbg.extend([0.0] * nx)
                    ubg.extend([0.0] * nx)

                    if k == 0:
                        dv = Ut[b][0, k] - Us[0, L - 1]
                        dw = Ut[b][1, k] - Us[1, L - 1]
                    else:
                        dv = Ut[b][0, k] - Ut[b][0, k - 1]
                        dw = Ut[b][1, k] - Ut[b][1, k - 1]
                    J += weight * self.wacc * (dv * dv + self.lambda_w * dw * dw)
                    J += weight * self.wvel * ((self._stage_speed_ca(Xt[b], Ut[b], k) - p_vref) ** 2)
                    J += weight * self.wtrack_tail * ca.sumsqr(Xt[b][0:2, k] - p_branch_targets[b])

                J += weight * self.wgoal * ca.sumsqr(Xt[b][0:2, Nt] - p_goal)

                # Visible constraints on branch tail with global time index offset by L.
                for k in range(1, Nt + 1):
                    tk = float(L + k) * float(self.dt_plan)
                    for (ox, oy, rr, vx, vy, active) in vis_params:
                        cx = ox + vx * tk
                        cy = oy + vy * tk
                        safe2 = (self.robot_radius + rr + self.margin_obs) ** 2
                        expr = active * (safe2 - ((Xt[b][0, k] - cx) ** 2 + (Xt[b][1, k] - cy) ** 2))
                        g.append(ca.vertcat(expr))
                        lbg.append(-np.inf)
                        ubg.append(0.0)

                # Branch-specific hidden risk regions apply to both shared trunk and tail.
                for (cx, cy, rr, active) in risk_params[b]:
                    safe2 = (self.robot_radius + rr + self.margin_risk) ** 2
                    for k in range(1, Nt + 1):
                        expr = active * (safe2 - ((Xt[b][0, k] - cx) ** 2 + (Xt[b][1, k] - cy) ** 2))
                        g.append(ca.vertcat(expr))
                        lbg.append(-np.inf)
                        ubg.append(0.0)

            nlp = {"x": Z, "f": J, "g": ca.vertcat(*g), "p": P}
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

            solver = ca.nlpsol("control_tree_joint", "ipopt", nlp, opts)

            z_size = int(Z.shape[0])
            lbx = np.full((z_size,), -np.inf, dtype=float)
            ubx = np.full((z_size,), np.inf, dtype=float)

            offsets = {}
            cursor = 0
            offsets["Xs"] = (cursor, cursor + sz["Xs"])
            cursor += sz["Xs"]
            offsets["Us"] = (cursor, cursor + sz["Us"])
            for k in range(L):
                base = offsets["Us"][0] + 2 * k
                lbx[base] = float(lb_u[0])
                ubx[base] = float(ub_u[0])
                lbx[base + 1] = float(lb_u[1])
                ubx[base + 1] = float(ub_u[1])

            xt_off = []
            ut_off = []
            for b in range(B):
                xt_off.append((cursor, cursor + sz["Xt"]))
                cursor += sz["Xt"]
                ut_off.append((cursor, cursor + sz["Ut"]))
                for k in range(Nt):
                    base = cursor + 2 * k
                    lbx[base] = float(lb_u[0])
                    ubx[base] = float(ub_u[0])
                    lbx[base + 1] = float(lb_u[1])
                    ubx[base + 1] = float(ub_u[1])
                cursor += sz["Ut"]
            offsets["Xt"] = xt_off
            offsets["Ut"] = ut_off

            self._persistent_solver = solver
            self._persistent_lbx = lbx
            self._persistent_ubx = ubx
            self._persistent_lbg = np.asarray(lbg, dtype=float)
            self._persistent_ubg = np.asarray(ubg, dtype=float)
            self._persistent_p_dim = int(p_dim)
            self._z_layout = offsets
            self._persistent_setup_done = True
            self._persistent_setup_ms = (time.perf_counter() - t0) * 1000.0
            return True, ""
        except Exception as exc:
            self._persistent_setup_done = False
            self._persistent_setup_ms = (time.perf_counter() - t0) * 1000.0
            return False, str(exc)

    def _pack_branch_risk_regions(self, branches):
        out = []
        B = int(self.max_branches)
        R = int(self.max_branch_risk_regions)
        for b in range(B):
            if b < len(branches):
                regs = list(branches[b].get("risk_regions", []))
            else:
                regs = []
            regs = regs[:R]
            while len(regs) < R:
                regs.append((np.zeros(2, dtype=float), 0.0, 0.0))
            branch_slots = []
            for reg in regs:
                if len(reg) == 3:
                    center, rr, active = reg
                else:
                    center, rr = reg
                    active = 1.0 if float(rr) > 0.0 else 0.0
                center = np.asarray(center, dtype=float).reshape(2,)
                branch_slots.append((center, float(rr), float(active)))
            out.append(branch_slots)
        return out

    def _pack_joint_params(self, x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom):
        P = np.zeros((int(self._persistent_p_dim),), dtype=float)
        idx = 0
        nx = int(self._n_state)
        B = int(self.max_branches)
        M = int(self.max_visible_obs)
        R = int(self.max_branch_risk_regions)

        P[idx : idx + nx] = np.asarray(x0, dtype=float).reshape(nx,)
        idx += nx
        P[idx : idx + 2] = np.asarray(goal_xy, dtype=float).reshape(2,)
        idx += 2
        P[idx : idx + 2] = np.asarray(guidance_xy, dtype=float).reshape(2,)
        idx += 2
        for b in range(B):
            if b < len(branch_targets):
                P[idx : idx + 2] = np.asarray(branch_targets[b], dtype=float).reshape(2,)
            idx += 2
        P[idx] = float(v_ref_nom)
        idx += 1
        P[idx : idx + 2] = np.asarray(self._u_prev_applied, dtype=float).reshape(2,)
        idx += 2

        branch_probs = np.zeros((B,), dtype=float)
        for b in range(min(B, len(branches))):
            branch_probs[b] = float(branches[b].get("prob", 0.0))
        if float(np.sum(branch_probs)) <= 1e-9:
            branch_probs[0] = 1.0
        P[idx : idx + B] = branch_probs
        idx += B

        n_vis_active = int(min(len(visible_obs), M))
        for j in range(M):
            if j < n_vis_active:
                obs = np.asarray(visible_obs[j], dtype=float).reshape(-1)
                P[idx + 0] = float(obs[0])
                P[idx + 1] = float(obs[1])
                P[idx + 2] = float(obs[2])
                P[idx + 3] = float(obs[3]) if obs.size >= 4 else 0.0
                P[idx + 4] = float(obs[4]) if obs.size >= 5 else 0.0
                P[idx + 5] = 1.0
            idx += 6

        branch_risk_slots = self._pack_branch_risk_regions(branches)
        n_risk_active_total = 0
        for b in range(B):
            for r in range(R):
                center, rr, active = branch_risk_slots[b][r]
                P[idx + 0] = float(center[0])
                P[idx + 1] = float(center[1])
                P[idx + 2] = float(rr)
                P[idx + 3] = float(active)
                if active > 0.5:
                    n_risk_active_total += 1
                idx += 4

        return P, branch_probs, n_vis_active, n_risk_active_total

    def _unpack_joint_solution(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        nx = int(self._n_state)
        L = int(self.n_split)
        Nt = int(self.tail_horizon)
        Xs = z[self._z_layout["Xs"][0] : self._z_layout["Xs"][1]].reshape((nx, L + 1), order="F")
        Us = z[self._z_layout["Us"][0] : self._z_layout["Us"][1]].reshape((2, L), order="F")
        Xt = []
        Ut = []
        n_layout_branches = int(len(self._z_layout.get("Xt", [])))
        for b in range(n_layout_branches):
            Xt.append(z[self._z_layout["Xt"][b][0] : self._z_layout["Xt"][b][1]].reshape((nx, Nt + 1), order="F"))
            Ut.append(z[self._z_layout["Ut"][b][0] : self._z_layout["Ut"][b][1]].reshape((2, Nt), order="F"))
        return Xs, Us, Xt, Ut

    def _initial_joint_guess(self, x0, guidance_xy, branch_targets, goal_xy, v_ref_nom):
        if self._z_prev is not None and len(self._z_prev) == len(self._persistent_lbx):
            return np.asarray(self._z_prev, dtype=float).reshape(-1)

        nx = int(self._n_state)
        L = int(self.n_split)
        Nt = int(self.tail_horizon)
        B = int(self.max_branches)
        Xs = np.zeros((nx, L + 1), dtype=float)
        Us = np.zeros((2, L), dtype=float)
        Xt = [np.zeros((nx, Nt + 1), dtype=float) for _ in range(B)]
        Ut = [np.zeros((2, Nt), dtype=float) for _ in range(B)]

        x = np.asarray(x0, dtype=float).reshape(nx,)
        Xs[:, 0] = x
        for k in range(L):
            u_cmd = self._guidance_input_np(x, guidance_xy, v_ref_nom, k_heading=self.nominal_k_heading)
            Us[:, k] = u_cmd
            x = self._discrete_np(x, u_cmd)
            Xs[:, k + 1] = x

        for b in range(B):
            x = Xs[:, -1].copy()
            Xt[b][:, 0] = x
            target_xy = goal_xy if b >= len(branch_targets) else np.asarray(branch_targets[b], dtype=float).reshape(2,)
            for k in range(Nt):
                u_cmd = self._guidance_input_np(x, target_xy, v_ref_nom, k_heading=self.nominal_k_heading)
                Ut[b][:, k] = u_cmd
                x = self._discrete_np(x, u_cmd)
                Xt[b][:, k + 1] = x

        z_parts = [Xs.reshape(-1, order="F"), Us.reshape(-1, order="F")]
        for b in range(B):
            z_parts.append(Xt[b].reshape(-1, order="F"))
            z_parts.append(Ut[b].reshape(-1, order="F"))
        return np.concatenate(z_parts, axis=0)

    def _solve_joint_persistent(self, x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom):
        ok_setup, setup_err = self._init_joint_persistent_backend()
        if not ok_setup:
            return False, None, None, None, None, "persistent_setup_failed", setup_err, 0.0, 0

        p, branch_probs, n_vis_active, n_risk_active_total = self._pack_joint_params(
            x0=x0,
            goal_xy=goal_xy,
            guidance_xy=guidance_xy,
            branch_targets=branch_targets,
            visible_obs=visible_obs,
            branches=branches,
            v_ref_nom=v_ref_nom,
        )
        z0 = self._initial_joint_guess(x0, guidance_xy, branch_targets, goal_xy, v_ref_nom)
        kwargs = {
            "x0": z0,
            "p": p,
            "lbg": self._persistent_lbg,
            "ubg": self._persistent_ubg,
            "lbx": self._persistent_lbx,
            "ubx": self._persistent_ubx,
        }
        if self.warm_start_dual:
            if self._lam_x_prev is not None and len(self._lam_x_prev) == len(z0):
                kwargs["lam_x0"] = self._lam_x_prev
            if self._lam_g_prev is not None and len(self._lam_g_prev) == len(self._persistent_lbg):
                kwargs["lam_g0"] = self._lam_g_prev

        t0 = time.perf_counter()
        try:
            sol = self._persistent_solver(**kwargs)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            z_sol = np.array(sol["x"]).reshape(-1)
            self._z_prev = z_sol
            if self.warm_start_dual:
                self._lam_x_prev = np.array(sol["lam_x"]).reshape(-1)
                self._lam_g_prev = np.array(sol["lam_g"]).reshape(-1)
            Xs, Us, Xt, Ut = self._unpack_joint_solution(z_sol)
            raw_status = "optimal"
            try:
                stats = self._persistent_solver.stats()
                if isinstance(stats, dict):
                    raw_status = str(stats.get("return_status", raw_status))
            except Exception:
                pass
            n_constraints = int((self.n_split + self.tail_horizon * self.max_branches) * n_vis_active)
            n_constraints += int((self.n_split + self.tail_horizon) * n_risk_active_total)
            return True, Xs, Us, Xt, Ut, raw_status, "", solve_ms, n_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            self._lam_x_prev = None
            self._lam_g_prev = None
            n_constraints = int((self.n_split + self.tail_horizon * self.max_branches) * n_vis_active)
            n_constraints += int((self.n_split + self.tail_horizon) * n_risk_active_total)
            return False, None, None, None, None, "infeasible", str(exc), solve_ms, n_constraints

    def _solve_joint_opti(self, x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom):
        if not _CASADI_AVAILABLE:
            return False, None, None, None, None, "casadi_missing", "CasADi is not installed", 0.0, 0

        nx = int(self._n_state)
        L = int(self.n_split)
        Nt = int(self.tail_horizon)
        B = max(1, len(branches))
        lb_u, ub_u = self._input_bounds()

        opti = ca.Opti()
        Xs = opti.variable(nx, L + 1)
        Us = opti.variable(2, L)
        Xt = [opti.variable(nx, Nt + 1) for _ in range(B)]
        Ut = [opti.variable(2, Nt) for _ in range(B)]

        x0_dm = ca.DM(np.asarray(x0, dtype=float).reshape(nx,))
        goal_dm = ca.DM(np.asarray(goal_xy, dtype=float).reshape(2,))
        guidance_dm = ca.DM(np.asarray(guidance_xy, dtype=float).reshape(2,))
        branch_targets_dm = [
            ca.DM(np.asarray(branch_targets[b] if b < len(branch_targets) else goal_xy, dtype=float).reshape(2,))
            for b in range(B)
        ]
        up_dm = ca.DM(np.asarray(self._u_prev_applied, dtype=float).reshape(2,))

        branch_probs = np.asarray([float(b.get("prob", 0.0)) for b in branches], dtype=float)
        if branch_probs.sum() <= 1e-9:
            branch_probs = np.zeros((B,), dtype=float)
            branch_probs[0] = 1.0

        J = 0
        opti.subject_to(Xs[:, 0] == x0_dm)
        for k in range(L):
            opti.subject_to(Xs[:, k + 1] == self._discrete_ca(Xs[:, k], Us[:, k]))
            opti.subject_to(opti.bounded(lb_u, Us[:, k], ub_u))
            if k == 0:
                dv = Us[0, k] - up_dm[0]
                dw = Us[1, k] - up_dm[1]
            else:
                dv = Us[0, k] - Us[0, k - 1]
                dw = Us[1, k] - Us[1, k - 1]
            J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
            J += self.wvel * ((self._stage_speed_ca(Xs, Us, k) - float(v_ref_nom)) ** 2)
            J += self.wtrack_shared * ca.sumsqr(Xs[0:2, k] - guidance_dm)

        for k in range(1, L + 1):
            for obs in visible_obs:
                c = self._predict_obs_center(obs, k)
                clear = self.robot_radius + float(obs[2]) + self.margin_obs
                dx = Xs[0, k] - float(c[0])
                dy = Xs[1, k] - float(c[1])
                opti.subject_to(dx * dx + dy * dy >= clear * clear)

        for b in range(B):
            weight = float(branch_probs[b]) + float(self.branch_zero_prob_reg)
            opti.subject_to(Xt[b][:, 0] == Xs[:, L])
            for k in range(Nt):
                opti.subject_to(Xt[b][:, k + 1] == self._discrete_ca(Xt[b][:, k], Ut[b][:, k]))
                opti.subject_to(opti.bounded(lb_u, Ut[b][:, k], ub_u))
                if k == 0:
                    dv = Ut[b][0, k] - Us[0, L - 1]
                    dw = Ut[b][1, k] - Us[1, L - 1]
                else:
                    dv = Ut[b][0, k] - Ut[b][0, k - 1]
                    dw = Ut[b][1, k] - Ut[b][1, k - 1]
                J += weight * self.wacc * (dv * dv + self.lambda_w * dw * dw)
                J += weight * self.wvel * ((self._stage_speed_ca(Xt[b], Ut[b], k) - float(v_ref_nom)) ** 2)
                J += weight * self.wtrack_tail * ca.sumsqr(Xt[b][0:2, k] - branch_targets_dm[b])
            J += weight * self.wgoal * ca.sumsqr(Xt[b][0:2, Nt] - goal_dm)

            regs = list(branches[b].get("risk_regions", []))
            for k in range(1, Nt + 1):
                for obs in visible_obs:
                    c = self._predict_obs_center(obs, L + k)
                    clear = self.robot_radius + float(obs[2]) + self.margin_obs
                    dx = Xt[b][0, k] - float(c[0])
                    dy = Xt[b][1, k] - float(c[1])
                    opti.subject_to(dx * dx + dy * dy >= clear * clear)
                for center, rr in regs:
                    clear = self.robot_radius + float(rr) + self.margin_risk
                    dx = Xt[b][0, k] - float(center[0])
                    dy = Xt[b][1, k] - float(center[1])
                    opti.subject_to(dx * dx + dy * dy >= clear * clear)

        opti.minimize(J)
        x0_guess = self._initial_joint_guess(x0, guidance_xy, branch_targets, goal_xy, v_ref_nom)
        if self._z_layout is not None and self._z_prev is not None and len(self._z_prev) == len(x0_guess):
            x0_guess = np.asarray(self._z_prev, dtype=float).reshape(-1)
        # Initializing all matrices via unpacked guess keeps the joint solve numerically stable.
        # Build a temporary layout for unpacking this one-shot solve.
        self._z_layout = {
            "Xs": (0, nx * (L + 1)),
            "Us": (nx * (L + 1), nx * (L + 1) + 2 * L),
            "Xt": [],
            "Ut": [],
        }
        cursor = self._z_layout["Us"][1]
        for _ in range(B):
            self._z_layout["Xt"].append((cursor, cursor + nx * (Nt + 1)))
            cursor += nx * (Nt + 1)
            self._z_layout["Ut"].append((cursor, cursor + 2 * Nt))
            cursor += 2 * Nt
        Xs_g, Us_g, Xt_g, Ut_g = self._unpack_joint_solution(x0_guess)
        opti.set_initial(Xs, Xs_g)
        opti.set_initial(Us, Us_g)
        for b in range(B):
            opti.set_initial(Xt[b], Xt_g[b])
            opti.set_initial(Ut[b], Ut_g[b])

        p_opts = {"expand": self.solver_expand, "print_time": self.print_solver}
        s_opts = {
            "print_level": 5 if self.print_solver else 0,
            "max_iter": self.max_iter,
            "tol": self.solver_tol,
            "acceptable_tol": self.solver_acceptable_tol,
            "acceptable_iter": self.solver_acceptable_iter,
            "sb": "yes",
        }
        if self.ipopt_linear_solver:
            s_opts["linear_solver"] = str(self.ipopt_linear_solver)
        opti.solver("ipopt", p_opts, s_opts)

        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            solve_ms = (time.perf_counter() - t0) * 1000.0
            Xs_sol = np.asarray(sol.value(Xs), dtype=float).reshape((nx, L + 1), order="F")
            Us_sol = np.asarray(sol.value(Us), dtype=float).reshape((2, L), order="F")
            Xt_sol = [
                np.asarray(sol.value(xb), dtype=float).reshape((nx, Nt + 1), order="F")
                for xb in Xt
            ]
            Ut_sol = [
                np.asarray(sol.value(ub), dtype=float).reshape((2, Nt), order="F")
                for ub in Ut
            ]
            raw_status = "optimal"
            n_constraints = 0
            return True, Xs_sol, Us_sol, Xt_sol, Ut_sol, raw_status, "", solve_ms, n_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            return False, None, None, None, None, "infeasible", str(exc), solve_ms, 0

    def _solve_joint(self, x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom):
        if self.solve_backend == "joint_persistent":
            out = self._solve_joint_persistent(x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom)
            if (not out[0]) and self.persistent_fallback_opti and out[5] == "persistent_setup_failed":
                return self._solve_joint_opti(x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom)
            return out
        return self._solve_joint_opti(x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom)

    # ---------------------------------------------------------------------
    # Diagnostics helpers
    # ---------------------------------------------------------------------
    def _shared_cost_numpy(self, Xs, Us, guidance_xy, v_ref_nom):
        Xs = np.asarray(Xs, dtype=float)
        Us = np.asarray(Us, dtype=float)
        guidance_xy = np.asarray(guidance_xy, dtype=float).reshape(2,)
        up = np.asarray(self._u_prev_applied, dtype=float).reshape(2,)
        J = 0.0
        for k in range(int(self.n_split)):
            vk = self._stage_speed_np(Xs, Us, k)
            wk = float(Us[1, k])
            if k == 0:
                dv = vk - up[0]
                dw = wk - up[1]
            else:
                dv = vk - float(Us[0, k - 1])
                dw = wk - float(Us[1, k - 1])
            J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
            J += self.wvel * ((vk - v_ref_nom) ** 2)
            err = Xs[0:2, k] - guidance_xy
            J += self.wtrack_shared * float(err @ err)
        J += 0.5 * self.wtrack_shared * float(np.sum((Xs[0:2, int(self.n_split)] - guidance_xy) ** 2))
        return float(J)

    def _tail_cost_numpy(self, Xt, Ut, branch_target_xy, goal_xy, v_ref_nom, prev_u):
        Xt = np.asarray(Xt, dtype=float)
        Ut = np.asarray(Ut, dtype=float)
        branch_target_xy = np.asarray(branch_target_xy, dtype=float).reshape(2,)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        prev_u = np.asarray(prev_u, dtype=float).reshape(2,)
        J = 0.0
        for k in range(int(self.tail_horizon)):
            vk = self._stage_speed_np(Xt, Ut, k)
            wk = float(Ut[1, k])
            if k == 0:
                dv = vk - prev_u[0]
                dw = wk - prev_u[1]
            else:
                dv = vk - float(Ut[0, k - 1])
                dw = wk - float(Ut[1, k - 1])
            J += self.wacc * (dv * dv + self.lambda_w * dw * dw)
            J += self.wvel * ((vk - v_ref_nom) ** 2)
            err = Xt[0:2, k] - branch_target_xy
            J += self.wtrack_tail * float(err @ err)
        terr = Xt[0:2, int(self.tail_horizon)] - goal_xy
        J += self.wgoal * float(terr @ terr)
        return float(J)

    def _feasibility_stats(self, Xs, Xt_list, visible_obs, branches):
        tol = 1e-7
        n_ok = 0
        n_total = 0
        min_vis = np.inf
        min_risk = np.inf

        Xs = np.asarray(Xs, dtype=float)
        for k in range(1, int(self.n_split) + 1):
            pos = Xs[:2, k]
            step_min = np.inf
            for obs in visible_obs:
                c = self._predict_obs_center(obs, k)
                clear = self.robot_radius + float(obs[2]) + self.margin_obs
                m = float(np.linalg.norm(pos - c) - clear)
                step_min = min(step_min, m)
                min_vis = min(min_vis, m)
            if np.isinf(step_min) or step_min >= -tol:
                n_ok += 1
            n_total += 1

        for bi, Xt in enumerate(Xt_list[: len(branches)]):
            Xt = np.asarray(Xt, dtype=float)
            regs = branches[bi].get("risk_regions", [])
            for k in range(1, int(self.tail_horizon) + 1):
                pos = Xt[:2, k]
                step_min = np.inf
                for obs in visible_obs:
                    c = self._predict_obs_center(obs, int(self.n_split) + k)
                    clear = self.robot_radius + float(obs[2]) + self.margin_obs
                    m = float(np.linalg.norm(pos - c) - clear)
                    step_min = min(step_min, m)
                    min_vis = min(min_vis, m)
                for center, rr in regs:
                    clear = self.robot_radius + float(rr) + self.margin_risk
                    m = float(np.linalg.norm(pos - center) - clear)
                    step_min = min(step_min, m)
                    min_risk = min(min_risk, m)
                if np.isinf(step_min) or step_min >= -tol:
                    n_ok += 1
                n_total += 1

        frac = float(n_ok) / float(max(1, n_total))
        if np.isinf(min_vis):
            min_vis = None
        if np.isinf(min_risk):
            min_risk = None
        return frac, min_vis, min_risk

    # ---------------------------------------------------------------------
    # Main solve
    # ---------------------------------------------------------------------
    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()

        x0 = self._state_from_robot_state(robot_state)
        u_ref = np.asarray(control_ref.get("u_ref", np.zeros((2, 1))), dtype=float).reshape(-1, 1)
        u_ref = self._clip_input(u_ref)
        self.last_u_ref = u_ref
        v_ref_nom = self._nominal_speed_reference(x0, u_ref, self.v_ref_default)
        goal_xy = self._goal_xy(x0, control_ref.get("goal", None))

        visible_obs, occ_scenarios = self._occ_utils._filter_visible_and_build_occ(
            np.asarray(robot_state, dtype=float).reshape(-1, 1),
            obs_list,
        )
        visible_obs = self._nearest_visible_obs(visible_obs, x0)
        self.occlusion_scenarios = list(occ_scenarios)

        guidance_xy, guidance_meta = self._select_shared_guidance(x0, goal_xy, visible_obs)
        nominal_points = self._nominal_rollout_positions(x0, guidance_xy, v_ref_nom)
        selected_scenarios = self._select_hypothesis_scenarios(x0, goal_xy, occ_scenarios, v_ref_nom, nominal_points)
        branches = self._build_branch_hypotheses(selected_scenarios)
        branch_targets = []
        branch_target_meta = []
        for br in branches:
            target_xy, meta = self._select_branch_target(x0, goal_xy, visible_obs, br.get("risk_regions", []))
            branch_targets.append(np.asarray(target_xy, dtype=float).reshape(2,))
            branch_target_meta.append(dict(meta))

        ok, Xs, Us, Xt, Ut, raw_status, ex, solve_ms, ncons = self._solve_joint(
            x0=x0,
            goal_xy=goal_xy,
            guidance_xy=guidance_xy,
            branch_targets=branch_targets,
            visible_obs=visible_obs,
            branches=branches,
            v_ref_nom=v_ref_nom,
        )

        self.last_qp_solve_time_ms = float(solve_ms)
        self.last_num_constraints = int(ncons)
        self.last_qp_status_raw = str(raw_status)
        self.last_qp_exception = str(ex) if ex else ""

        branch_probs = [float(b.get("prob", 0.0)) for b in branches]
        branch_masks = [list(map(int, b.get("mask", tuple()))) for b in branches]
        map_branch = int(np.argmax(branch_probs)) if len(branch_probs) > 0 else 0

        if not ok:
            self.status = "infeasible"
            self.last_intervention = "control_tree_mpc"
            self.last_u = self._stop_input()
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_profile = {
                "backend": "control_tree_joint",
                "solver_backend": str(self.solve_backend),
                "setup_ms": (None if self._persistent_setup_ms is None else float(self._persistent_setup_ms)),
                "total_ms": float(self.last_total_compute_time_ms),
                "wall_clock_total_ms": float(self.last_total_compute_time_ms),
                "solve_ms_total": float(self.last_qp_solve_time_ms),
                "solve_ms_total_branch_sum": float(self.last_qp_solve_time_ms),
                "n_visible_obs": int(len(visible_obs)),
                "n_occ_regions_total": int(len(occ_scenarios)),
                "n_occ_hypotheses_selected": int(len(selected_scenarios)),
                "n_branches_generated": int(len(branches)),
                "n_branches_feasible": 0,
                "branch_probabilities": list(branch_probs),
                "branch_masks": list(branch_masks),
                "branch_guidance_points": [[float(bt[0]), float(bt[1])] for bt in branch_targets],
                "branch_guidance_meta": branch_target_meta,
                "selected_branch": int(map_branch),
                "guidance_source": str(guidance_meta.get("guidance_source", "")),
                "guidance_xy": [float(guidance_xy[0]), float(guidance_xy[1])],
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "shared_prefix_length": int(self.n_split),
                "belief_scores": [float(s["score"]) for s in selected_scenarios],
                "belief_state": [float(s["p"]) for s in selected_scenarios],
                "raw_solver_status": self.last_qp_status_raw,
                "num_constraints": int(self.last_num_constraints),
            }
            return None

        u_cmd = self._clip_input(Us[:, 0].reshape(-1, 1))
        self.last_u = u_cmd
        self._u_prev_applied = u_cmd
        self.status = "optimal"
        self.last_intervention = "control_tree_mpc"

        shared_cost = self._shared_cost_numpy(Xs, Us, guidance_xy, v_ref_nom)
        prev_u_tail = np.asarray(Us[:, int(self.n_split) - 1], dtype=float).reshape(2,)
        branch_tail_costs = [
            self._tail_cost_numpy(Xt[b], Ut[b], branch_targets[b], goal_xy, v_ref_nom, prev_u_tail)
            for b in range(len(branches))
        ]
        explore_cost = float(shared_cost + sum(float(branch_probs[b]) * float(branch_tail_costs[b]) for b in range(len(branches))))
        frac, min_vis_margin, min_risk_margin = self._feasibility_stats(Xs, Xt, visible_obs, branches)

        self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
        self.last_profile = {
            "backend": "control_tree_joint",
            "solver_backend": str(self.solve_backend),
            "setup_ms": (None if self._persistent_setup_ms is None else float(self._persistent_setup_ms)),
            "total_ms": float(self.last_total_compute_time_ms),
            "wall_clock_total_ms": float(self.last_total_compute_time_ms),
            "solve_ms_total": float(self.last_qp_solve_time_ms),
            "solve_ms_total_branch_sum": float(self.last_qp_solve_time_ms),
            "n_visible_obs": int(len(visible_obs)),
            "n_occ_regions_total": int(len(occ_scenarios)),
            "n_occ_hypotheses_selected": int(len(selected_scenarios)),
            "n_branches_generated": int(len(branches)),
            "n_branches_feasible": int(len(branches)),
            "branch_probabilities": list(branch_probs),
            "branch_masks": list(branch_masks),
            "branch_costs": [float(c) for c in branch_tail_costs],
            "branch_guidance_points": [[float(bt[0]), float(bt[1])] for bt in branch_targets],
            "branch_guidance_meta": branch_target_meta,
            "selected_branch": int(map_branch),
            "selected_cost": float(branch_tail_costs[map_branch]) if len(branch_tail_costs) > 0 else None,
            "belief_scores": [float(s["score"]) for s in selected_scenarios],
            "belief_state": [float(s["p"]) for s in selected_scenarios],
            "guidance_source": str(guidance_meta.get("guidance_source", "")),
            "guidance_xy": [float(guidance_xy[0]), float(guidance_xy[1])],
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "shared_prefix_length": int(self.n_split),
            "effective_v_ref": float(v_ref_nom),
            "u_ref_0": float(u_ref[0, 0]),
            "u_ref_1": float(u_ref[1, 0]),
            "u_cmd_0": float(u_cmd[0, 0]),
            "u_cmd_1": float(u_cmd[1, 0]),
            "raw_solver_status": self.last_qp_status_raw,
            "num_constraints": int(self.last_num_constraints),
            "feasible_horizon_fraction": float(frac),
            "min_visible_margin": (None if min_vis_margin is None else float(min_vis_margin)),
            "min_risk_margin": (None if min_risk_margin is None else float(min_risk_margin)),
            # Branch-level diagnostics expected by this project.
            "selected_branch_map": int(map_branch),
            "explore_cost": float(explore_cost),
            "fallback_cost": float(branch_tail_costs[map_branch]) if len(branch_tail_costs) > 0 else None,
            "explore_feasible": True,
            "fallback_feasible": True,
            "occlusion_risk_score": float(sum(s["score"] for s in selected_scenarios)),
            "explore_speed_cap": float(v_ref_nom),
            "fallback_speed_cap": float(v_ref_nom),
            "branch_switch_count": 0,
        }
        return u_cmd
