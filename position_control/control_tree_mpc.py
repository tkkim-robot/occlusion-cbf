import itertools
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


class ControlTreeMPC(MPCCommonUtils):
    """
    Centralized, ADMM-free Control-Tree MPC baseline inspired by ICRA 2021,
    "Control-Tree Optimization: an approach to MPC under discrete Partial
    Observability", adapted here to crowd-style occlusion scenes.

    Implemented here from the ICRA 2021 control-tree spirit:
      - one explicit control tree with a shared trunk and branch-specific tails,
      - non-anticipativity through a common optimized prefix shared by all
        branches,
      - belief-weighted branch costs,
      - deterministic active-occlusion selection before tree construction so
        the planner reasons over a small set of geometrically relevant
        occlusions rather than a single top-ranked occlusion only,
      - local discrete branch states of the form
        {no_hidden, hidden_t1, hidden_t2} for each active occlusion, yielding
        joint hidden-world branches,
      - branch-specific direct hidden-obstacle constraints after the split,
      - robust shared-trunk safety against the active hidden-world constraints
        from all branches,
      - receding-horizon execution where only the first shared-trunk action is
        applied each MPC cycle.

    Not implemented paper-faithfully here:
      - the original augmented-Lagrangian / ADMM decomposition, consensus
        updates, dual updates, or distributed branch subproblem solves,
      - the paper's original discrete symbolic-state construction and belief
        estimation pipeline; here branch worlds are synthesized from the repo's
        occlusion geometry by generating direct hidden-obstacle candidates on
        tangent rays of active occlusions,
      - the paper's original application-specific experiment stack; this file is
        a crowd2-style centralized adaptation with benchmark-specific horizon,
        branch caps, and conservative hidden-agent assumptions.
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

        self.dt_plan = float(cfg.get("dt_plan", 0.05))
        self.Th = float(cfg.get("Th", 1.0))
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

        # Hypothesis-tree parameters.
        self.hypothesis_model = str(cfg.get("hypothesis_model", "direct_hidden_obstacle")).strip().lower()
        if self.hypothesis_model != "direct_hidden_obstacle":
            self.hypothesis_model = "direct_hidden_obstacle"
        self.max_active_occlusions = int(cfg.get("max_active_occlusions", cfg.get("n_occ_hypotheses", 2)))
        self.max_active_occlusions = max(0, min(2, self.max_active_occlusions))
        self.n_occ_hypotheses = int(self.max_active_occlusions)
        self.active_selection_delta = float(cfg.get("active_selection_delta", 1.0))
        self.max_branches = int(3**self.max_active_occlusions) if self.max_active_occlusions > 0 else 1
        self.hidden_speed = float(
            cfg.get(
                "hidden_speed",
                robot_spec.get("v_adv_max_occ", robot_spec.get("v_obs_max", 0.5)),
            )
        )
        self.hidden_agent_radius = float(cfg.get("hidden_agent_radius", 0.4))
        self.hidden_spawn_clearance = float(cfg.get("hidden_spawn_clearance", 0.12))
        self.hidden_speed_scale = float(cfg.get("hidden_speed_scale", 1.0))
        self.max_branch_hidden_obs = max(
            1,
            int(cfg.get("max_branch_hidden_obs", max(1, self.max_active_occlusions))),
        )

        # Cost shaping.
        self.wgoal = float(cfg.get("wgoal", 3.5))
        self.wvel = float(cfg.get("wvel", 5.0))
        self.wacc = float(cfg.get("wacc", 1.8))
        self.wtrack_shared = float(cfg.get("wtrack_shared", 2.0))
        self.wtrack_tail = float(cfg.get("wtrack_tail", 0.5))
        self.lambda_w = float(cfg.get("lambda_w", 1.0))
        self.v_des = float(cfg.get("v_des", robot_spec.get("v_max", 0.5)))
        self.branch_zero_prob_reg = float(cfg.get("branch_zero_prob_reg", 1e-3))
        v_min_bound, v_max_bound = self._speed_bounds()
        self.v_des = float(np.clip(self.v_des, v_min_bound, v_max_bound))
        self.v_state_bound = float(v_max_bound)

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
    # Hypothesis construction
    # ---------------------------------------------------------------------
    def _nominal_rollout_positions(self, x0, goal_xy, v_plan):
        x = np.asarray(x0, dtype=float).reshape(self._n_state,)
        points = np.zeros((self.N + 1, 2), dtype=float)
        points[0] = x[:2]
        for k in range(self.N):
            u_cmd = self._guidance_input_np(x, goal_xy, v_plan)
            x = self._discrete_np(x, u_cmd)
            points[k + 1] = x[:2]
        return points

    def _hidden_obstacle_from_tangent(self, scenario, tangent_key):
        return self._direct_hidden_obstacle_from_tangent(scenario, tangent_key)

    def _scenario_score(self, x0, goal_xy, scenario, nominal_points):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        g = np.asarray(goal_xy, dtype=float).reshape(2,)
        c = np.asarray(scenario.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
        r_obs = float(scenario.get("obs_radius", 0.0))

        dist_to_robot = max(0.0, float(np.linalg.norm(c - p) - r_obs))
        if nominal_points is not None and nominal_points.size > 0:
            path_clear = max(0.0, float(np.min(np.linalg.norm(nominal_points - c[None, :], axis=1)) - r_obs))
        else:
            path_clear = max(0.0, float(np.linalg.norm(c - g) - r_obs))
        return 1.0 / max(0.25, dist_to_robot + path_clear)

    def _select_hypothesis_scenarios(self, x0, goal_xy, occ_scenarios, v_plan, nominal_points):
        scored = [
            {
                "scenario": sc,
                "score": float(self._scenario_score(x0, goal_xy, sc, nominal_points)),
            }
            for sc in occ_scenarios
        ]
        if len(scored) == 0:
            return []
        score_ref = float(np.mean([item["score"] for item in scored]))
        scored.sort(key=lambda x: x["score"], reverse=True)
        selected = scored[: int(self.n_occ_hypotheses)]
        if len(selected) == 0:
            return []

        out = []
        for item in selected:
            sc = item["scenario"]
            out.append(
                {
                    "scenario": sc,
                    "score": float(item["score"]),
                    "p": float(item["score"] / (item["score"] + score_ref + 1e-9)),
                }
            )
        return out

    def _local_hidden_states(self, entry):
        if entry is None:
            return [{"label": "none", "prob": 1.0, "hidden_obs": [], "active_ids": [], "local_state": "none"}]

        p_occ = float(np.clip(entry.get("p_occ", 0.0), 0.0, 1.0))
        scenario_index = int(entry.get("scenario_index", -1))
        occupied_states = []
        if entry.get("hidden_t1", None) is not None:
            occupied_states.append(
                {
                    "label": f"occ{scenario_index}_t1",
                    "prob": 0.0,
                    "hidden_obs": [np.asarray(entry["hidden_t1"], dtype=float)],
                    "active_ids": [scenario_index],
                    "local_state": "hidden_t1",
                }
            )
        if entry.get("hidden_t2", None) is not None:
            occupied_states.append(
                {
                    "label": f"occ{scenario_index}_t2",
                    "prob": 0.0,
                    "hidden_obs": [np.asarray(entry["hidden_t2"], dtype=float)],
                    "active_ids": [scenario_index],
                    "local_state": "hidden_t2",
                }
            )

        states = [
            {
                "label": f"occ{scenario_index}_none",
                "prob": 1.0 if len(occupied_states) == 0 else max(0.0, 1.0 - p_occ),
                "hidden_obs": [],
                "active_ids": [],
                "local_state": "no_hidden",
            }
        ]
        if len(occupied_states) > 0:
            occupied_prob = p_occ / float(len(occupied_states))
            for st in occupied_states:
                st["prob"] = float(max(0.0, occupied_prob))
                states.append(st)
        return states

    def _build_hidden_world_branches(self, active_entries):
        if len(active_entries) == 0 or int(self.max_branches) <= 1:
            return [{"label": "no_hidden", "prob": 1.0, "hidden_obs": [], "active_ids": []}]

        entries = list(active_entries[: int(self.max_active_occlusions)])
        if len(entries) == 0:
            return [{"label": "no_hidden", "prob": 1.0, "hidden_obs": [], "active_ids": []}]

        local_state_sets = [self._local_hidden_states(entry) for entry in entries]
        branches = []
        for combo in itertools.product(*local_state_sets):
            prob = 1.0
            hidden_obs = []
            active_ids = []
            branch_labels = []
            branch_state = []
            for local in combo:
                prob *= float(local.get("prob", 0.0))
                hidden_obs.extend(local.get("hidden_obs", []))
                active_ids.extend(local.get("active_ids", []))
                branch_labels.append(str(local.get("label", "state")))
                branch_state.append(str(local.get("local_state", "none")))
            branches.append(
                {
                    "label": "__".join(branch_labels) if len(branch_labels) > 0 else "no_hidden",
                    "prob": float(prob),
                    "hidden_obs": list(hidden_obs[: int(self.max_branch_hidden_obs)]),
                    "active_ids": [int(idx) for idx in active_ids],
                    "joint_state": list(branch_state),
                    "active_count": int(len(active_ids)),
                }
            )

        total = float(sum(b["prob"] for b in branches))
        if total <= 1e-9:
            branches = [{"label": "no_hidden", "prob": 1.0, "hidden_obs": [], "active_ids": []}]
        else:
            for branch in branches:
                branch["prob"] = float(branch["prob"] / total)
            branches.sort(key=lambda b: (-float(b.get("prob", 0.0)), str(b.get("label", ""))))
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
            H = int(self.max_branch_hidden_obs)
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

            # P = [x0(nx), goal(2), guidance(2), branch_targets(B*2), vref(1), up_prev(2), branch_probs(B), vis(M*6), hidden(B*H*6)]
            p_dim = nx + 2 + 2 + 2 * B + 1 + 2 + B + 6 * M + 6 * B * H
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

            hidden_params = []
            for _b in range(B):
                branch_slots = []
                for _h in range(H):
                    branch_slots.append((P[idx + 0], P[idx + 1], P[idx + 2], P[idx + 3], P[idx + 4], P[idx + 5]))
                    idx += 6
                hidden_params.append(branch_slots)

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
                if self.model == "DoubleIntegrator2D":
                    g.append(ca.vertcat(ca.sumsqr(Xs[2:4, k + 1]) - self.v_state_bound * self.v_state_bound))
                    lbg.append(-np.inf)
                    ubg.append(0.0)

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

            # Robust hidden-obstacle constraints on the shared trunk: the
            # common prefix must remain safe for every active branch hypothesis.
            for k in range(1, L + 1):
                tk = float(k) * float(self.dt_plan)
                for b in range(B):
                    for (ox, oy, rr, vx, vy, active) in hidden_params[b]:
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
                    if self.model == "DoubleIntegrator2D":
                        g.append(ca.vertcat(ca.sumsqr(Xt[b][2:4, k + 1]) - self.v_state_bound * self.v_state_bound))
                        lbg.append(-np.inf)
                        ubg.append(0.0)

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

                # Branch-specific hidden obstacles remain enforced on branch tails.
                for (ox, oy, rr, vx, vy, active) in hidden_params[b]:
                    safe2 = (self.robot_radius + rr + self.margin_obs) ** 2
                    for k in range(1, Nt + 1):
                        tk = float(L + k) * float(self.dt_plan)
                        cx = ox + vx * tk
                        cy = oy + vy * tk
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

    def _pack_branch_hidden_obs(self, branches):
        out = []
        B = int(self.max_branches)
        H = int(self.max_branch_hidden_obs)
        for b in range(B):
            if b < len(branches):
                hidden_obs = list(branches[b].get("hidden_obs", []))
            else:
                hidden_obs = []
            hidden_obs = hidden_obs[:H]
            while len(hidden_obs) < H:
                hidden_obs.append(np.zeros((6,), dtype=float))
            branch_slots = []
            for obs in hidden_obs:
                arr = np.asarray(obs, dtype=float).reshape(-1)
                slot = np.zeros((6,), dtype=float)
                n_copy = min(6, arr.size)
                if n_copy > 0:
                    slot[:n_copy] = arr[:n_copy]
                if arr.size < 6:
                    slot[5] = 1.0 if float(slot[2]) > 0.0 else 0.0
                branch_slots.append((float(slot[0]), float(slot[1]), float(slot[2]), float(slot[3]), float(slot[4]), float(slot[5])))
            out.append(branch_slots)
        return out

    def _pack_joint_params(self, x0, goal_xy, guidance_xy, branch_targets, visible_obs, branches, v_ref_nom):
        P = np.zeros((int(self._persistent_p_dim),), dtype=float)
        idx = 0
        nx = int(self._n_state)
        B = int(self.max_branches)
        M = int(self.max_visible_obs)
        H = int(self.max_branch_hidden_obs)

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

        branch_hidden_slots = self._pack_branch_hidden_obs(branches)
        n_hidden_active_total = 0
        for b in range(B):
            for h in range(H):
                ox, oy, rr, vx, vy, active = branch_hidden_slots[b][h]
                P[idx + 0] = float(ox)
                P[idx + 1] = float(oy)
                P[idx + 2] = float(rr)
                P[idx + 3] = float(vx)
                P[idx + 4] = float(vy)
                P[idx + 5] = float(active)
                if active > 0.5:
                    n_hidden_active_total += 1
                idx += 6

        return P, branch_probs, n_vis_active, n_hidden_active_total

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

    def _joint_constraint_counts(self, n_visible_obs, n_hidden_active_total, n_branches):
        counts = {
            "shared_visible": int(self.n_split) * int(n_visible_obs),
            "shared_hidden_obs": int(self.n_split) * int(n_hidden_active_total),
            "tail_visible": int(self.tail_horizon) * int(n_branches) * int(n_visible_obs),
            "tail_hidden_obs": int(self.tail_horizon) * int(n_hidden_active_total),
        }
        counts["speed_state_bound"] = 0
        if self.model == "DoubleIntegrator2D":
            counts["speed_state_bound"] = int(self.n_split + self.tail_horizon * int(n_branches))
        counts["total"] = int(sum(counts.values()))
        return counts

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
            u_cmd = self._guidance_input_np(x, guidance_xy, v_ref_nom)
            Us[:, k] = u_cmd
            x = self._discrete_np(x, u_cmd)
            Xs[:, k + 1] = x

        for b in range(B):
            x = Xs[:, -1].copy()
            Xt[b][:, 0] = x
            target_xy = goal_xy if b >= len(branch_targets) else np.asarray(branch_targets[b], dtype=float).reshape(2,)
            for k in range(Nt):
                u_cmd = self._guidance_input_np(x, target_xy, v_ref_nom)
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

        p, branch_probs, n_vis_active, n_hidden_active_total = self._pack_joint_params(
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
            n_constraints = self._joint_constraint_counts(
                n_visible_obs=n_vis_active,
                n_hidden_active_total=n_hidden_active_total,
                n_branches=self.max_branches,
            )["total"]
            return True, Xs, Us, Xt, Ut, raw_status, "", solve_ms, n_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            self._lam_x_prev = None
            self._lam_g_prev = None
            n_constraints = self._joint_constraint_counts(
                n_visible_obs=n_vis_active,
                n_hidden_active_total=n_hidden_active_total,
                n_branches=self.max_branches,
            )["total"]
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
        branch_hidden_obs = [list(branches[b].get("hidden_obs", [])) for b in range(B)]
        n_hidden_active_total = int(sum(len(obs_list) for obs_list in branch_hidden_obs))

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
            if self.model == "DoubleIntegrator2D":
                opti.subject_to(ca.sumsqr(Xs[2:4, k + 1]) <= self.v_state_bound * self.v_state_bound)

        for k in range(1, L + 1):
            for obs in visible_obs:
                c = self._predict_obs_center(obs, k)
                clear = self.robot_radius + float(obs[2]) + self.margin_obs
                dx = Xs[0, k] - float(c[0])
                dy = Xs[1, k] - float(c[1])
                opti.subject_to(dx * dx + dy * dy >= clear * clear)
            # The shared trunk must be safe for every active branch hypothesis.
            for hidden_obs_list in branch_hidden_obs:
                for obs in hidden_obs_list:
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
                if self.model == "DoubleIntegrator2D":
                    opti.subject_to(ca.sumsqr(Xt[b][2:4, k + 1]) <= self.v_state_bound * self.v_state_bound)
            J += weight * self.wgoal * ca.sumsqr(Xt[b][0:2, Nt] - goal_dm)

            hidden_obs_list = branch_hidden_obs[b]
            for k in range(1, Nt + 1):
                for obs in visible_obs:
                    c = self._predict_obs_center(obs, L + k)
                    clear = self.robot_radius + float(obs[2]) + self.margin_obs
                    dx = Xt[b][0, k] - float(c[0])
                    dy = Xt[b][1, k] - float(c[1])
                    opti.subject_to(dx * dx + dy * dy >= clear * clear)
                for obs in hidden_obs_list:
                    c = self._predict_obs_center(obs, L + k)
                    clear = self.robot_radius + float(obs[2]) + self.margin_obs
                    dx = Xt[b][0, k] - float(c[0])
                    dy = Xt[b][1, k] - float(c[1])
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
            n_constraints = self._joint_constraint_counts(
                n_visible_obs=len(visible_obs),
                n_hidden_active_total=n_hidden_active_total,
                n_branches=B,
            )["total"]
            return True, Xs_sol, Us_sol, Xt_sol, Ut_sol, raw_status, "", solve_ms, n_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            n_constraints = self._joint_constraint_counts(
                n_visible_obs=len(visible_obs),
                n_hidden_active_total=n_hidden_active_total,
                n_branches=B,
            )["total"]
            return False, None, None, None, None, "infeasible", str(exc), solve_ms, n_constraints

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
        min_hidden = np.inf
        shared_hidden_obs = []
        for br in branches:
            shared_hidden_obs.extend(list(br.get("hidden_obs", [])))

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
            for obs in shared_hidden_obs:
                c = self._predict_obs_center(obs, k)
                clear = self.robot_radius + float(obs[2]) + self.margin_obs
                m = float(np.linalg.norm(pos - c) - clear)
                step_min = min(step_min, m)
                min_hidden = min(min_hidden, m)
            if np.isinf(step_min) or step_min >= -tol:
                n_ok += 1
            n_total += 1

        for bi, Xt in enumerate(Xt_list[: len(branches)]):
            Xt = np.asarray(Xt, dtype=float)
            hidden_obs_list = branches[bi].get("hidden_obs", [])
            for k in range(1, int(self.tail_horizon) + 1):
                pos = Xt[:2, k]
                step_min = np.inf
                for obs in visible_obs:
                    c = self._predict_obs_center(obs, int(self.n_split) + k)
                    clear = self.robot_radius + float(obs[2]) + self.margin_obs
                    m = float(np.linalg.norm(pos - c) - clear)
                    step_min = min(step_min, m)
                    min_vis = min(min_vis, m)
                for obs in hidden_obs_list:
                    c = self._predict_obs_center(obs, int(self.n_split) + k)
                    clear = self.robot_radius + float(obs[2]) + self.margin_obs
                    m = float(np.linalg.norm(pos - c) - clear)
                    step_min = min(step_min, m)
                    min_hidden = min(min_hidden, m)
                if np.isinf(step_min) or step_min >= -tol:
                    n_ok += 1
                n_total += 1

        frac = float(n_ok) / float(max(1, n_total))
        if np.isinf(min_vis):
            min_vis = None
        if np.isinf(min_hidden):
            min_hidden = None
        return frac, min_vis, min_hidden

    # ---------------------------------------------------------------------
    # Main solve
    # ---------------------------------------------------------------------
    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()

        x0 = self._state_from_robot_state(robot_state)
        u_ref = np.asarray(control_ref.get("u_ref", np.zeros((2, 1))), dtype=float).reshape(-1, 1)
        u_ref = self._clip_input(u_ref)
        self.last_u_ref = u_ref
        goal_xy = self._goal_xy(x0, control_ref.get("goal", None))
        v_plan = float(self.v_des)

        visible_obs, occ_scenarios = self._occ_utils._filter_visible_and_build_occ(
            np.asarray(robot_state, dtype=float).reshape(-1, 1),
            obs_list,
        )
        visible_obs = self._nearest_visible_obs(visible_obs, x0)
        self.occlusion_scenarios = list(occ_scenarios)

        guidance_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        guidance_meta = {"guidance_source": "goal"}
        nominal_points = self._nominal_rollout_positions(x0, goal_xy, v_plan)
        active_entries, occ_candidates = self._select_active_occlusion_entries(
            x0=x0,
            goal_xy=goal_xy,
            occ_scenarios=occ_scenarios,
            nominal_points=nominal_points,
            max_active_occlusions=self.max_active_occlusions,
        )
        branches = self._build_hidden_world_branches(active_entries)
        branch_targets = [np.asarray(goal_xy, dtype=float).reshape(2,) for _ in branches]
        branch_target_meta = [{"target_source": "goal"} for _ in branches]

        ok, Xs, Us, Xt, Ut, raw_status, ex, solve_ms, ncons = self._solve_joint(
            x0=x0,
            goal_xy=goal_xy,
            guidance_xy=guidance_xy,
            branch_targets=branch_targets,
            visible_obs=visible_obs,
            branches=branches,
            v_ref_nom=v_plan,
        )

        self.last_qp_solve_time_ms = float(solve_ms)
        self.last_num_constraints = int(ncons)
        self.last_qp_status_raw = str(raw_status)
        self.last_qp_exception = str(ex) if ex else ""

        branch_probs = [float(b.get("prob", 0.0)) for b in branches]
        branch_labels = [str(b.get("label", f"branch_{i}")) for i, b in enumerate(branches)]
        branch_joint_states = [list(b.get("joint_state", [])) for b in branches]
        branch_hidden_counts = [int(len(b.get("hidden_obs", []))) for b in branches]
        active_occlusion_indices = [int(e["scenario_index"]) for e in active_entries]
        active_occlusion_scores = [float(e["score"]) for e in active_entries]
        active_occlusion_probabilities = [float(e["p_occ"]) for e in active_entries]
        active_occlusion_min_clearances = [float(e["min_clearance"]) for e in active_entries]
        active_occlusion_critical_steps = [
            None if not np.isfinite(float(e["critical_step"])) else float(e["critical_step"])
            for e in active_entries
        ]
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
                "n_occ_regions_active": int(len(active_entries)),
                "n_occ_hypotheses_selected": int(len(active_entries)),
                "n_branches_generated": int(len(branches)),
                "n_branches_feasible": 0,
                "branch_probabilities": list(branch_probs),
                "branch_labels": list(branch_labels),
                "branch_joint_states": branch_joint_states,
                "branch_hidden_counts": branch_hidden_counts,
                "branch_guidance_points": [[float(bt[0]), float(bt[1])] for bt in branch_targets],
                "branch_guidance_meta": branch_target_meta,
                "selected_branch": int(map_branch),
                "selected_branch_label": (None if len(branch_labels) == 0 else str(branch_labels[map_branch])),
                "guidance_source": str(guidance_meta.get("guidance_source", "")),
                "guidance_xy": [float(guidance_xy[0]), float(guidance_xy[1])],
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "shared_prefix_length": int(self.n_split),
                "belief_scores": active_occlusion_scores,
                "belief_state": active_occlusion_probabilities,
                "active_occlusion_indices": active_occlusion_indices,
                "active_occlusion_scores": active_occlusion_scores,
                "active_occlusion_probabilities": active_occlusion_probabilities,
                "active_occlusion_min_clearances": active_occlusion_min_clearances,
                "active_occlusion_critical_steps": active_occlusion_critical_steps,
                "n_occ_candidates_considered": int(len(occ_candidates)),
                "effective_v_ref": float(v_plan),
                "speed_cost_mode": "speed_norm",
                "raw_solver_status": self.last_qp_status_raw,
                "num_constraints": int(self.last_num_constraints),
            }
            return None

        u_cmd = self._clip_input(Us[:, 0].reshape(-1, 1))
        self.last_u = u_cmd
        self._u_prev_applied = u_cmd
        self.status = "optimal"
        self.last_intervention = "control_tree_mpc"

        shared_cost = self._shared_cost_numpy(Xs, Us, guidance_xy, v_plan)
        prev_u_tail = np.asarray(Us[:, int(self.n_split) - 1], dtype=float).reshape(2,)
        branch_tail_costs = [
            self._tail_cost_numpy(Xt[b], Ut[b], branch_targets[b], goal_xy, v_plan, prev_u_tail)
            for b in range(len(branches))
        ]
        explore_cost = float(shared_cost + sum(float(branch_probs[b]) * float(branch_tail_costs[b]) for b in range(len(branches))))
        frac, min_vis_margin, min_hidden_margin = self._feasibility_stats(Xs, Xt, visible_obs, branches)

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
            "n_occ_regions_active": int(len(active_entries)),
            "n_occ_hypotheses_selected": int(len(active_entries)),
            "n_branches_generated": int(len(branches)),
            "n_branches_feasible": int(len(branches)),
            "branch_probabilities": list(branch_probs),
            "branch_labels": list(branch_labels),
            "branch_joint_states": branch_joint_states,
            "branch_hidden_counts": branch_hidden_counts,
            "branch_costs": [float(c) for c in branch_tail_costs],
            "branch_guidance_points": [[float(bt[0]), float(bt[1])] for bt in branch_targets],
            "branch_guidance_meta": branch_target_meta,
            "selected_branch": int(map_branch),
            "selected_branch_label": (None if len(branch_labels) == 0 else str(branch_labels[map_branch])),
            "selected_cost": float(branch_tail_costs[map_branch]) if len(branch_tail_costs) > 0 else None,
            "belief_scores": active_occlusion_scores,
            "belief_state": active_occlusion_probabilities,
            "active_occlusion_indices": active_occlusion_indices,
            "active_occlusion_scores": active_occlusion_scores,
            "active_occlusion_probabilities": active_occlusion_probabilities,
            "active_occlusion_min_clearances": active_occlusion_min_clearances,
            "active_occlusion_critical_steps": active_occlusion_critical_steps,
            "n_occ_candidates_considered": int(len(occ_candidates)),
            "guidance_source": str(guidance_meta.get("guidance_source", "")),
            "guidance_xy": [float(guidance_xy[0]), float(guidance_xy[1])],
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "shared_prefix_length": int(self.n_split),
            "effective_v_ref": float(v_plan),
            "speed_cost_mode": "speed_norm",
            "u_ref_0": float(u_ref[0, 0]),
            "u_ref_1": float(u_ref[1, 0]),
            "u_cmd_0": float(u_cmd[0, 0]),
            "u_cmd_1": float(u_cmd[1, 0]),
            "raw_solver_status": self.last_qp_status_raw,
            "num_constraints": int(self.last_num_constraints),
            "feasible_horizon_fraction": float(frac),
            "min_visible_margin": (None if min_vis_margin is None else float(min_vis_margin)),
            "min_hidden_margin": (None if min_hidden_margin is None else float(min_hidden_margin)),
            "min_risk_margin": (None if min_hidden_margin is None else float(min_hidden_margin)),
            # Branch-level diagnostics expected by this project.
            "selected_branch_map": int(map_branch),
            "explore_cost": float(explore_cost),
            "fallback_cost": float(branch_tail_costs[map_branch]) if len(branch_tail_costs) > 0 else None,
            "explore_feasible": True,
            "fallback_feasible": True,
            "occlusion_risk_score": float(sum(active_occlusion_scores)),
            "speed_state_bound": float(self.v_state_bound),
            "branch_switch_count": 0,
        }
        return u_cmd
