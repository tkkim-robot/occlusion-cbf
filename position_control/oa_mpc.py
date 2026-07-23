import time

import numpy as np

from utils.occlusion import OcclusionUtils

try:
    import casadi as ca

    _CASADI_AVAILABLE = True
except Exception:
    ca = None
    _CASADI_AVAILABLE = False


class OAMPC:
    """
    OA-MPC baseline inspired by:
      "OA-MPC: Occlusion-Aware MPC for Guaranteed Safe Robot Navigation
      with Unseen Dynamic Obstacles", adapted here to crowd-style occlusion
      scenes.

    Implemented features from the paper:
      - forward reachable sets for dynamic agents with no intention prediction:
        hidden agents from occlusion boundaries and visible dynamic agents from
        bounded-speed worst-case expansion,
      - LiDAR-jump-style occlusion-boundary extraction and capsule-like hidden
        reachable sets that grow over the MPC horizon,
      - projected-point collision constraints against the shifted open-loop
        trajectory followed by one CasADi/IPOPT NMPC solve,
      - safe-distance defaults, effective 1.0 s horizon, input /
        delta-input / terminal costs, and terminal stopping logic,
      - visible dynamic agents can also act as occluders in test cases; this is a
        benchmark adaptation because the paper experiments mostly use static
        occluders and treat pedestrian-induced occlusion as negligible,

    Not implemented here:
      - theorem-exact recursive feasibility is not claimed in dense dynamic
        crowd scenes; full complementarity handling for every collision row is
        solver-fragile, so visible/static circle rows are hard constraints by
        default and terminal complementarity is not forced for dynamic-occluder
        crowd runs,
      - the full numerical Algorithm-2 fixed-point alternation is simplified by
        default to one projection-target construction followed by one NMPC solve
        per MPC step, though additional alternations can be enabled by config,
      - this implementation preserves the main OA-MPC planning structure, but uses a
        simplified single IPOPT NMPC solve rather than reproducing every numerical
        detail of the original algorithm.
    """

    def __init__(self, robot, robot_spec, num_obs=30):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = int(num_obs)

        self.model = str(robot_spec.get("model", "")).strip()
        if self.model not in {"DoubleIntegrator2D", "Unicycle2D", "DynamicUnicycle2D"}:
            raise ValueError(f"OA-MPC currently supports DI/UNI/DU only, got `{self.model}`")

        self.dt = float(getattr(robot, "dt", robot_spec.get("dt", 0.05)))
        self.robot_radius = float(robot_spec.get("radius", 0.25))
        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))

        cfg = robot_spec.setdefault("oa_mpc", {})
        # Keep OA-MPC in paper-faithful mode for horizon/cost/safety defaults,
        # while allowing reverse motion by default for fairness-matched
        # benchmark comparisons in this codebase.
        self.paper_mode = True
        self.N = int(cfg.get("N", 10))
        # Paper setting uses N=10 at dt=0.1 (~1.0 s horizon).
        # Preserve this effective horizon when running with smaller dt.
        self.paper_horizon_time = float(cfg.get("paper_horizon_time", 1.0))
        self.auto_scale_N_with_dt = bool(cfg.get("auto_scale_N_with_dt", self.paper_mode))
        if self.auto_scale_N_with_dt and self.dt > 1e-9:
            n_from_time = int(np.ceil(self.paper_horizon_time / self.dt))
            self.N = max(self.N, n_from_time)
        # Paper setting uses d_safe=0.5 (can be overridden per scenario).
        default_dsafe = 0.5 if self.paper_mode else 0.02
        self.dsafe = float(cfg.get("dsafe", default_dsafe))
        default_hidden_agent_radius = 0.0 if self.paper_mode else 0.2
        self.hidden_agent_radius = float(cfg.get("hidden_agent_radius", default_hidden_agent_radius))
        hidden_speed_default = robot_spec.get("v_adv_max_occ", robot_spec.get("v_obs_max", 0.5))
        self.v_hidden_max_default = float(cfg.get("v_hidden_max", hidden_speed_default))
        visible_speed_default = robot_spec.get("v_obs_max", self.v_hidden_max_default)
        self.v_visible_max_default = float(cfg.get("v_visible_max", visible_speed_default))
        self.visible_reach_mode = str(cfg.get("visible_reach_mode", "worst_case")).strip().lower()
        if self.visible_reach_mode not in {"worst_case", "constant_velocity"}:
            self.visible_reach_mode = "worst_case"
        self.use_nominal_tracking_cost = bool(cfg.get("use_nominal_tracking_cost", False))
        default_max_constraints = 32 if self.paper_mode else 64
        self.max_constraints_per_step = int(cfg.get("max_constraints_per_step", default_max_constraints))
        default_prune_far = (not self.paper_mode)
        self.prune_far_constraints = bool(cfg.get("prune_far_constraints", default_prune_far))
        self.prune_far_margin = float(cfg.get("prune_far_margin", 0.0))
        # Practical robustness guard: IPOPT may occasionally return
        # Restoration/Step_Computation failure on this nonconvex NMPC even when
        # a safe stop action exists. Keep it opt-in so benchmark defaults report
        # solver failure as OA-MPC infeasibility rather than silently stopping.
        default_fallback = False
        self.allow_solver_fallback = bool(cfg.get("allow_solver_fallback", default_fallback))
        self.occluder_speed_max = float(cfg.get("occluder_speed_max", 1e-3))
        self.dynamic_occluders = bool(cfg.get("dynamic_occluders", False))
        # Keep the paper's occlusion projected-set complementarity active by
        # default. Visible/static circle rows remain hard distance constraints
        # unless explicitly overridden, which is the crowd adaptation used here.
        default_use_comp = True
        self.use_complementarity = bool(cfg.get("use_complementarity", default_use_comp))
        self.complementarity_smax_kappa = float(cfg.get("complementarity_smax_kappa", 20.0))
        self.aggregate_complementarity = bool(cfg.get("aggregate_complementarity", False))
        self.complementarity_for_circles = bool(
            cfg.get("complementarity_for_circles", False)
        )
        # Paper recursive-feasibility argument relies on a terminal stopping
        # contingency. In practice, encoding bilinear complementarity at every
        # step is solver-fragile in this benchmark. Instead, keep hard
        # collision-avoidance rows over the horizon and realize the terminal
        # complementarity only at the final prediction step by dropping the
        # corresponding hard row there. Since X[:,N] == X[:,N-1] is enforced,
        # this captures the intended "stopped => safe contingency" logic
        # without introducing another nonconvex product term.
        default_terminal_comp = self.paper_mode and (not self.dynamic_occluders)
        self.use_terminal_complementarity = bool(
            cfg.get("use_terminal_complementarity", default_terminal_comp)
        )
        self.terminal_complementarity_for_occlusion = bool(
            cfg.get("terminal_complementarity_for_occlusion", self.use_terminal_complementarity)
        )
        self.terminal_complementarity_for_circles = bool(
            cfg.get("terminal_complementarity_for_circles", False)
        )
        self.terminal_relax_weight = float(cfg.get("terminal_relax_weight", 250.0))
        # For DI, the paper's generic terminal equality z_N = z_{N-1}
        # translates to an exact full stop at the end of the horizon.
        # In this benchmark's acceleration-limited crowd setting, that can be
        # overly restrictive. Keep the exact interpretation as the default, but
        # allow a DI-specific relaxed terminal set in which the terminal state
        # only needs to be brake-reachable within a small number of future
        # control intervals.
        self.di_terminal_stop_mode = str(cfg.get("di_terminal_stop_mode", "exact")).strip().lower()
        if self.di_terminal_stop_mode not in {"exact", "brake_reachable", "input_zero_only", "none"}:
            self.di_terminal_stop_mode = "exact"
        self.di_terminal_brake_steps = max(1, int(cfg.get("di_terminal_brake_steps", 1)))
        # Paper Algorithm 2 alternates projected-distance construction and
        # NMPC solves. The benchmark default uses one projection/NMPC pass per
        # MPC step for robustness, while keeping extra alternations configurable.
        default_alt_iters = 1 if self.paper_mode else 1
        self.alternation_iters = max(1, int(cfg.get("alternation_iters", default_alt_iters)))
        self.alternation_tol = float(cfg.get("alternation_tol", 1e-3))

        # OA-MPC paper-style sensing geometry (LiDAR jump + point-cloud circles).
        self.use_lidar_jump_occ = bool(cfg.get("use_lidar_jump_occ", self.paper_mode))
        self.use_static_pointcloud_constraints = bool(
            cfg.get("use_static_pointcloud_constraints", self.paper_mode)
        )
        default_lidar_rays = 91 if self.paper_mode else 181
        self.lidar_num_rays = int(cfg.get("lidar_num_rays", default_lidar_rays))
        self.lidar_fov_deg = float(cfg.get("lidar_fov_deg", robot_spec.get("fov_angle", 360.0)))
        self.lidar_jump_threshold = float(cfg.get("lidar_jump_threshold", 0.45))
        default_max_occ_boundaries = 12 if self.paper_mode else 48
        self.max_occ_boundaries = int(cfg.get("max_occ_boundaries", default_max_occ_boundaries))
        self.min_occ_boundary_len = float(cfg.get("min_occ_boundary_len", 0.08))
        self.max_occ_boundary_len = float(
            cfg.get("max_occ_boundary_len", max(1.0, 0.35 * self.sensing_range))
        )
        default_static_downsample = 8 if self.paper_mode else 3
        self.static_pc_downsample = int(cfg.get("static_pc_downsample", default_static_downsample))
        self.static_pc_min_radius = float(cfg.get("static_pc_min_radius", 0.10))
        self.static_pc_max_radius = float(cfg.get("static_pc_max_radius", 0.40))
        default_static_max_circles = 24 if self.paper_mode else 128
        self.static_pc_max_circles = int(cfg.get("static_pc_max_circles", default_static_max_circles))

        # Stage/terminal costs
        self.v_ref_default = float(cfg.get("v_ref_default", 0.5))
        self.w_pos = float(cfg.get("w_pos", 15.0))
        self.w_u = float(cfg.get("w_u", 1.0))
        self.w_du = float(cfg.get("w_du", 0.2))
        self.w_terminal = float(cfg.get("w_terminal", 25.0))
        self.di_use_progress_speed_cost = bool(cfg.get("di_use_progress_speed_cost", True))
        # Pressure scores are retained as diagnostics, but the default DI
        # adaptation does not add an extra risk-style speed governor. OA-MPC
        # should slow through reachable-set constraints, complementarity, and
        # terminal stopping rather than a separate pressure-based cap.
        self.di_use_speed_schedule = bool(cfg.get("di_use_speed_schedule", False))
        self.di_use_speed_cap = bool(cfg.get("di_use_speed_cap", True))
        self.di_w_speed = float(cfg.get("di_w_speed", 6.0))
        self.di_w_lateral_velocity = float(cfg.get("di_w_lateral_velocity", 2.0))
        self.di_circle_speed_scale = float(cfg.get("di_circle_speed_scale", 0.55))
        self.di_proj_speed_scale = float(cfg.get("di_proj_speed_scale", 0.90))
        self.di_speed_floor_scale = float(cfg.get("di_speed_floor_scale", 0.25))
        self.di_pressure_clip = float(cfg.get("di_pressure_clip", 4.0))
        self.di_pressure_margin_scale = float(
            cfg.get("di_pressure_margin_scale", max(self.dsafe + self.robot_radius, 1e-3))
        )

        self.solver_name = str(cfg.get("solver", "ipopt")).lower()
        # In paper-faithful one-shot alternation, overly large IPOPT iteration
        # budgets can drift into poor local branches in this benchmark setup.
        # A moderate budget is empirically more stable.
        default_max_iter = 200 if self.paper_mode else 120
        self.max_iter = int(cfg.get("max_iter", default_max_iter))
        self.print_solver = bool(cfg.get("print_solver", False))
        self.solver_expand = bool(cfg.get("solver_expand", False))
        self.solver_tol = float(cfg.get("solver_tol", 1e-4))
        default_acc_tol = 1e-2 if self.paper_mode else 5e-4
        self.solver_acceptable_tol = float(cfg.get("solver_acceptable_tol", default_acc_tol))
        self.solver_acceptable_iter = int(cfg.get("solver_acceptable_iter", 8))

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=None,
        )

        self._n, self._m = self._dims()
        self._x_prev_plan = None  # (n, N+1)
        self._u_prev_plan = None  # (m, N)
        self._u_prev_applied = np.zeros((self._m, 1), dtype=float)

        # Diagnostics exposed to dynamic_env/main.py and logging tools.
        self.status = "optimal"
        self.last_num_constraints = 0
        self.last_qp_solve_time_ms = 0.0
        self.last_total_compute_time_ms = 0.0
        self.last_intervention = "u_ref"
        self.last_u_ref = np.zeros((self._m, 1), dtype=float)
        self.last_u = np.zeros((self._m, 1), dtype=float)
        self.last_profile = {}
        self.last_qp_status_raw = ""
        self.last_qp_exception = ""
        self.last_v_ref_nom_raw = 0.0
        self.last_v_ref_schedule_min = 0.0
        self.last_v_ref_schedule_mean = 0.0
        self.last_circle_pressure_score = 0.0
        self.last_proj_pressure_score = 0.0

        self.occlusion_scenarios = []
        self._last_projection_points = []
        self._last_static_pc_circles = []

    def _stop_input(self):
        try:
            u_stop = np.asarray(self.robot.stop(), dtype=float).reshape(-1, 1)
        except Exception:
            u_stop = np.zeros((self._m, 1), dtype=float)
        return self._clip_input(u_stop)

    def _dims(self):
        if self.model == "DoubleIntegrator2D":
            return 4, 2
        if self.model == "Unicycle2D":
            return 3, 2
        return 4, 2  # DynamicUnicycle2D

    def _input_bounds(self):
        if self.model == "DoubleIntegrator2D":
            a_max = float(self.robot_spec.get("a_max", 1.0))
            lb = np.array([-a_max, -a_max], dtype=float)
            ub = np.array([a_max, a_max], dtype=float)
            return lb, ub
        if self.model == "Unicycle2D":
            v_max = float(self.robot_spec.get("v_max", 1.0))
            w_max = float(self.robot_spec.get("w_max", 0.8))
            # The original paper uses forward speed bound v in [0, v_max].
            # In this benchmark codebase we allow reverse motion by default
            # unless `v_min` is explicitly provided in `robot_spec`.
            if "v_min" in self.robot_spec:
                v_min = float(self.robot_spec.get("v_min", 0.0))
            else:
                v_min = -v_max
            if (not np.isfinite(v_min)) or v_min > v_max:
                v_min = -v_max
            lb = np.array([v_min, -w_max], dtype=float)
            ub = np.array([v_max, w_max], dtype=float)
            return lb, ub
        a_max = float(self.robot_spec.get("a_max", 1.0))
        w_max = float(self.robot_spec.get("w_max", 0.8))
        lb = np.array([-a_max, -w_max], dtype=float)
        ub = np.array([a_max, w_max], dtype=float)
        return lb, ub

    def _clip_input(self, u):
        lb, ub = self._input_bounds()
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.clip(u, lb, ub).reshape(-1, 1)

    def _normalize_state(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.model in {"Unicycle2D", "DynamicUnicycle2D"}:
            x = x.copy()
            x[2] = ((x[2] + np.pi) % (2 * np.pi)) - np.pi
        return x

    def _discrete_dynamics_np(self, x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        dt = self.dt

        if self.model == "DoubleIntegrator2D":
            xn = np.array(
                [
                    x[0] + dt * x[2],
                    x[1] + dt * x[3],
                    x[2] + dt * u[0],
                    x[3] + dt * u[1],
                ],
                dtype=float,
            )
            return xn

        if self.model == "Unicycle2D":
            xn = np.array(
                [
                    x[0] + dt * u[0] * np.cos(x[2]),
                    x[1] + dt * u[0] * np.sin(x[2]),
                    x[2] + dt * u[1],
                ],
                dtype=float,
            )
            xn[2] = ((xn[2] + np.pi) % (2 * np.pi)) - np.pi
            return xn

        # DynamicUnicycle2D
        xn = np.array(
            [
                x[0] + dt * x[3] * np.cos(x[2]),
                x[1] + dt * x[3] * np.sin(x[2]),
                x[2] + dt * u[1],
                x[3] + dt * u[0],
            ],
            dtype=float,
        )
        xn[2] = ((xn[2] + np.pi) % (2 * np.pi)) - np.pi
        return xn

    def _discrete_dynamics_ca(self, xk, uk):
        dt = self.dt
        if self.model == "DoubleIntegrator2D":
            return ca.vertcat(
                xk[0] + dt * xk[2],
                xk[1] + dt * xk[3],
                xk[2] + dt * uk[0],
                xk[3] + dt * uk[1],
            )

        if self.model == "Unicycle2D":
            return ca.vertcat(
                xk[0] + dt * uk[0] * ca.cos(xk[2]),
                xk[1] + dt * uk[0] * ca.sin(xk[2]),
                xk[2] + dt * uk[1],
            )

        return ca.vertcat(
            xk[0] + dt * xk[3] * ca.cos(xk[2]),
            xk[1] + dt * xk[3] * ca.sin(xk[2]),
            xk[2] + dt * uk[1],
            xk[3] + dt * uk[0],
        )

    def _obs_array(self, obs_list):
        if obs_list is None:
            return np.zeros((0, 8), dtype=float)
        arr = np.asarray(obs_list, dtype=float)
        if arr.size == 0:
            return np.zeros((0, 8), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _closest_point_segment(p, a, b):
        p = np.asarray(p, dtype=float).reshape(2)
        a = np.asarray(a, dtype=float).reshape(2)
        b = np.asarray(b, dtype=float).reshape(2)
        d = b - a
        den = float(d @ d)
        if den < 1e-12:
            return a
        t = float(np.clip(((p - a) @ d) / den, 0.0, 1.0))
        return a + t * d

    def _project_to_circle_set(self, p, c, r):
        p = np.asarray(p, dtype=float).reshape(2)
        c = np.asarray(c, dtype=float).reshape(2)
        r = max(float(r), 0.0)
        d = p - c
        n = float(np.linalg.norm(d))
        if n <= r:
            return p
        if n < 1e-12:
            return c + np.array([r, 0.0], dtype=float)
        return c + (r / n) * d

    def _project_to_capsule_set(self, p, a, b, r):
        c = self._closest_point_segment(p, a, b)
        return self._project_to_circle_set(p, c, r)

    def _heading_from_state(self, x):
        if self.model in {"Unicycle2D", "DynamicUnicycle2D"} and x.size >= 3:
            return float(x[2])
        return 0.0

    @staticmethod
    def _ray_circle_intersection(origin, direction, center, radius):
        # Solve ||origin + t*direction - center||^2 = radius^2 for t >= 0.
        o = np.asarray(origin, dtype=float).reshape(2)
        d = np.asarray(direction, dtype=float).reshape(2)
        c = np.asarray(center, dtype=float).reshape(2)
        r = float(radius)
        oc = o - c
        b = 2.0 * float(d @ oc)
        c0 = float(oc @ oc) - r * r
        disc = b * b - 4.0 * c0
        if disc < 0.0:
            return None
        sdisc = float(np.sqrt(max(0.0, disc)))
        t1 = (-b - sdisc) / 2.0
        t2 = (-b + sdisc) / 2.0
        cand = []
        if t1 >= 0.0:
            cand.append(t1)
        if t2 >= 0.0:
            cand.append(t2)
        if not cand:
            return None
        return float(min(cand))

    def _simulate_lidar_scan(self, robot_state, occluder_obs):
        x = np.asarray(robot_state, dtype=float).reshape(-1)
        p = x[0:2]
        heading = self._heading_from_state(x)

        n_rays = max(8, int(self.lidar_num_rays))
        fov = float(self.lidar_fov_deg)
        if (not np.isfinite(fov)) or fov <= 0.0:
            fov = 360.0
        full_circle = fov >= 359.0
        if full_circle:
            rel = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        else:
            hf = np.deg2rad(0.5 * fov)
            rel = np.linspace(-hf, hf, n_rays)
        angles = heading + rel

        ranges = np.full((n_rays,), float(self.sensing_range), dtype=float)
        hit_mask = np.zeros((n_rays,), dtype=bool)
        points = np.zeros((n_rays, 2), dtype=float)

        for i in range(n_rays):
            a = float(angles[i])
            d = np.array([np.cos(a), np.sin(a)], dtype=float)
            best = float(self.sensing_range)
            hit = False
            for obs in occluder_obs:
                obs = np.asarray(obs, dtype=float).reshape(-1)
                c = obs[0:2]
                r = float(obs[2]) if obs.size >= 3 else 0.0
                t_hit = self._ray_circle_intersection(p, d, c, r)
                if t_hit is None:
                    continue
                if t_hit <= best:
                    best = float(t_hit)
                    hit = True
            ranges[i] = best
            hit_mask[i] = hit
            points[i] = p + best * d

        return {
            "origin": p.copy(),
            "angles": angles,
            "ranges": ranges,
            "hit_mask": hit_mask,
            "points": points,
            "full_circle": full_circle,
        }

    def _build_static_pointcloud_circles(self, scan):
        if scan is None:
            return []
        hit = np.asarray(scan.get("hit_mask", []), dtype=bool).reshape(-1)
        points = np.asarray(scan.get("points", []), dtype=float)
        if hit.size == 0 or points.size == 0:
            return []
        hit_idx = np.flatnonzero(hit)
        if hit_idx.size == 0:
            return []

        stride = max(1, int(self.static_pc_downsample))
        kept_idx = hit_idx[::stride]
        circles = []
        for idx in kept_idx:
            p = np.asarray(points[idx], dtype=float).reshape(2)
            # Radius selected to cover nearby scanned surface points.
            local = []
            prev_idx = idx - 1
            while prev_idx >= 0 and (not hit[prev_idx]):
                prev_idx -= 1
            next_idx = idx + 1
            while next_idx < hit.size and (not hit[next_idx]):
                next_idx += 1
            if prev_idx >= 0 and hit[prev_idx]:
                local.append(float(np.linalg.norm(points[idx] - points[prev_idx])))
            if next_idx < hit.size and hit[next_idx]:
                local.append(float(np.linalg.norm(points[idx] - points[next_idx])))
            if local:
                r = 0.55 * min(local)
            else:
                r = float(self.static_pc_min_radius)
            r = float(np.clip(r, self.static_pc_min_radius, self.static_pc_max_radius))
            circles.append((p, r))
            if len(circles) >= self.static_pc_max_circles:
                break
        return circles

    def _build_occ_scenarios_from_lidar_jumps(self, scan):
        if scan is None:
            return []
        ranges = np.asarray(scan.get("ranges", []), dtype=float).reshape(-1)
        points = np.asarray(scan.get("points", []), dtype=float)
        hit = np.asarray(scan.get("hit_mask", []), dtype=bool).reshape(-1)
        origin = np.asarray(scan.get("origin", np.zeros(2)), dtype=float).reshape(2)
        full_circle = bool(scan.get("full_circle", False))
        if ranges.size < 2 or points.shape[0] != ranges.size or hit.size != ranges.size:
            return []

        n = int(ranges.size)
        scenarios = []
        jump_thr = max(1e-6, float(self.lidar_jump_threshold))
        max_len = float(self.max_occ_boundary_len)
        if (not np.isfinite(max_len)) or max_len <= 0.0:
            max_len = float("inf")

        # Build contiguous hit-runs. Each run approximates a visible obstacle
        # silhouette; use the two run edges as one occlusion-boundary segment.
        hit_idx = np.flatnonzero(hit)
        if hit_idx.size == 0:
            return []

        runs = []
        run_start = int(hit_idx[0])
        prev = int(hit_idx[0])
        for idx in hit_idx[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
                continue
            runs.append((run_start, prev))
            run_start = idx
            prev = idx
        runs.append((run_start, prev))

        # Merge wrap-around run for 360-degree scans.
        if full_circle and len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == (n - 1):
            merged = (runs[-1][0], runs[0][1] + n)
            runs = [merged] + runs[1:-1]

        for rs, re in runs:
            i = int(rs % n)
            j = int(re % n)
            # Require jump evidence at both run edges.
            i_prev = (i - 1) % n
            j_next = (j + 1) % n
            left_jump = abs(float(ranges[i]) - float(ranges[i_prev])) >= jump_thr
            right_jump = abs(float(ranges[j]) - float(ranges[j_next])) >= jump_thr
            if not (left_jump or right_jump):
                continue
            p1 = np.asarray(points[i], dtype=float).reshape(2)
            p2 = np.asarray(points[j], dtype=float).reshape(2)
            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < self.min_occ_boundary_len or seg_len > max_len:
                continue
            d1 = p1 - origin
            d2 = p2 - origin
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))
            if n1 <= 1e-9 or n2 <= 1e-9:
                continue
            far1 = origin + (float(self.sensing_range) / n1) * d1
            far2 = origin + (float(self.sensing_range) / n2) * d2
            poly = np.vstack([p1, p2, far2, far1])
            try:
                A, b0 = self._occ_utils._polygon_to_halfspaces(poly)
            except Exception:
                A, b0 = None, None
            if A is None or b0 is None:
                continue
            v_expand_vec = np.full((len(b0),), float(self.v_hidden_max_default), dtype=float)
            if v_expand_vec.size > 0:
                # The boundary segment itself is the immediate visible frontier.
                # Keep only the "behind the frontier" facets expanding over time.
                v_expand_vec[0] = 0.0
            scenarios.append(
                {
                    "poly": poly,
                    "A": A,
                    "b0": b0,
                    "v_expand_vec": v_expand_vec,
                    "robot_pos": origin.copy(),
                    "obs_center": 0.5 * (p1 + p2),
                    "obs_radius": 0.5 * seg_len,
                    "front1": p1,
                    "front2": p2,
                    "t1": p1,
                    "t2": p2,
                    "v_adv_max": float(self.v_hidden_max_default),
                    "source": "lidar_jump_run",
                }
            )
            if len(scenarios) >= self.max_occ_boundaries:
                break
        return scenarios

    def _shift_or_init_plan(self, x0, u_ref):
        x0 = np.asarray(x0, dtype=float).reshape(self._n)
        u_ref = self._clip_input(u_ref).reshape(self._m)

        use_prev = (
            isinstance(self._x_prev_plan, np.ndarray)
            and isinstance(self._u_prev_plan, np.ndarray)
            and self._x_prev_plan.shape == (self._n, self.N + 1)
            and self._u_prev_plan.shape == (self._m, self.N)
        )

        if use_prev:
            x_bar = np.hstack([self._x_prev_plan[:, 1:], self._x_prev_plan[:, -1:]])
            u_bar = np.hstack([self._u_prev_plan[:, 1:], self._u_prev_plan[:, -1:]])
            x_bar[:, 0] = x0
        else:
            x_bar = np.zeros((self._n, self.N + 1), dtype=float)
            u_bar = np.tile(u_ref.reshape(-1, 1), (1, self.N))
            x_bar[:, 0] = x0

        for k in range(self.N):
            u_bar[:, k] = self._clip_input(u_bar[:, k]).reshape(self._m)
            x_bar[:, k] = self._normalize_state(x_bar[:, k])
            x_bar[:, k + 1] = self._discrete_dynamics_np(x_bar[:, k], u_bar[:, k])

        x_bar[:, 0] = x0
        return x_bar, u_bar

    def _nominal_speed_reference(self, x0, u_ref, floor_speed):
        floor_speed = float(floor_speed)
        if self.model == "DoubleIntegrator2D":
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
            v_cur = x0[2:4] if x0.size >= 4 else np.zeros((2,), dtype=float)
            v_des = v_cur + self.dt * u_ref[:2]
            v_max = float(self.robot_spec.get("v_max", 1.0))
            v_mag = float(np.linalg.norm(v_des))
            if np.isfinite(v_max) and v_mag > max(v_max, 1e-9):
                v_des = v_des * (float(v_max) / v_mag)
            return max(float(np.linalg.norm(v_des)), floor_speed)
        if self.model == "DynamicUnicycle2D":
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
            v_max = float(self.robot_spec.get("v_max", 1.0))
            v_min = float(self.robot_spec.get("v_min", -v_max))
            v_cur = float(x0[3]) if x0.size >= 4 else 0.0
            v_des = float(np.clip(v_cur + float(u_ref[0]), v_min, v_max))
            return max(abs(v_des), floor_speed)
        return max(abs(float(np.asarray(u_ref, dtype=float).reshape(-1)[0])), floor_speed)

    @staticmethod
    def _safe_goal_dir_np(pos_xy, goal_xy):
        pos_xy = np.asarray(pos_xy, dtype=float).reshape(2,)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
        d = goal_xy - pos_xy
        dn = float(np.linalg.norm(d))
        if dn <= 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return d / dn

    def _di_progress_components_np(self, xk, goal_xy):
        xk = np.asarray(xk, dtype=float).reshape(-1)
        e = self._safe_goal_dir_np(xk[:2], goal_xy)
        n = np.array([-e[1], e[0]], dtype=float)
        v = xk[2:4]
        return float(v @ e), float(v @ n)

    @staticmethod
    def _di_progress_components_ca(xk, goal_xy):
        d = goal_xy - xk[0:2]
        dn = ca.sqrt(ca.sumsqr(d) + 1e-12)
        e = d / dn
        n = ca.vertcat(-e[1], e[0])
        v = xk[2:4]
        return ca.dot(v, e), ca.dot(v, n)

    def _di_speed_schedule(self, x0, u_ref, x_bar, proj_targets):
        v_ref_raw = float(self._nominal_speed_reference(x0, u_ref, self.v_ref_default))
        schedule = np.full((self.N,), v_ref_raw, dtype=float)
        if self.model != "DoubleIntegrator2D":
            return v_ref_raw, schedule, 0.0, 0.0

        margin_scale = max(float(self.di_pressure_margin_scale), 1e-3)
        v0 = float(np.linalg.norm(np.asarray(x0, dtype=float).reshape(-1)[2:4]))
        a_max = float(self.robot_spec.get("a_max", 1.0))
        decel_rate = max(np.sqrt(2.0) * max(a_max, 0.0), 1e-6)
        circle_pressures = []
        proj_pressures = []
        for k in range(1, self.N + 1):
            p = np.asarray(x_bar[0:2, k], dtype=float).reshape(2,)
            circle_pressure = 0.0
            proj_pressure = 0.0
            for cst in proj_targets[k]:
                kind = str(cst.get("kind", "proj")).strip().lower()
                if kind == "circle":
                    cxy = np.asarray(cst.get("center", np.zeros(2)), dtype=float).reshape(2,)
                    margin = float(np.linalg.norm(p - cxy) - float(cst.get("min_dist", 0.0)))
                    circle_pressure += float(np.exp(-max(0.0, margin) / margin_scale))
                else:
                    zxy = np.asarray(cst.get("point", np.zeros(2)), dtype=float).reshape(2,)
                    margin = float(np.linalg.norm(p - zxy) - float(cst.get("clear", 0.0)))
                    proj_pressure += float(np.exp(-max(0.0, margin) / margin_scale))
            circle_pressure = float(np.clip(circle_pressure, 0.0, max(self.di_pressure_clip, 0.0)))
            proj_pressure = float(np.clip(proj_pressure, 0.0, max(self.di_pressure_clip, 0.0)))
            circle_pressures.append(circle_pressure)
            proj_pressures.append(proj_pressure)
            if self.di_use_speed_schedule:
                denom = 1.0 + self.di_circle_speed_scale * circle_pressure + self.di_proj_speed_scale * proj_pressure
                scale = float(np.clip(1.0 / max(denom, 1e-6), self.di_speed_floor_scale, 1.0))
                v_sched_k = float(v_ref_raw * scale)
                reachable_floor = max(0.0, v0 - decel_rate * self.dt * float(k))
                schedule[k - 1] = float(max(v_sched_k, reachable_floor))

        mean_circle = float(np.mean(circle_pressures)) if circle_pressures else 0.0
        mean_proj = float(np.mean(proj_pressures)) if proj_pressures else 0.0
        return v_ref_raw, schedule, mean_circle, mean_proj

    def _build_projection_targets(self, x_bar, visible_obs, occ_scenarios, static_pc_circles):
        # targets[k] contains per-step collision constraints:
        #   {"kind":"circle", "center": c_xy, "min_dist": d_min}
        #     -> hard distance to circle center (paper-like static/visible handling)
        #   {"kind":"proj", "point": z_proj_xy, "clear": clear, "use_comp": True}
        #     -> complementarity-relaxed projected-point distance (occlusion capsules)
        targets = [[] for _ in range(self.N + 1)]
        p0 = np.asarray(x_bar[0:2, 0], dtype=float).reshape(2,)
        dyn_types_cfg = self.robot_spec.get("dynamic_obs_types", [1])
        try:
            dyn_types = {int(t) for t in dyn_types_cfg}
        except Exception:
            dyn_types = {1}
        if self.model in {"Unicycle2D", "DynamicUnicycle2D"}:
            v_ego_max = float(self.robot_spec.get("v_max", 1.0))
        else:
            v_ego_max = float(self.robot_spec.get("v_max", 1.0))
        if (not np.isfinite(v_ego_max)) or (v_ego_max < 0.0):
            v_ego_max = 1.0

        for k in range(1, self.N + 1):
            tau = float(k) * self.dt
            p = x_bar[0:2, k]
            cand = []

            # Static obstacle constraints from downsampled LiDAR point-cloud circles.
            for c_obs, r_obs in static_pc_circles:
                c_pred = np.asarray(c_obs, dtype=float).reshape(2)
                r_pred = max(0.0, float(r_obs))
                clear = self.dsafe + self.robot_radius
                if self.prune_far_constraints:
                    d0 = float(np.linalg.norm(p0 - c_pred))
                    if d0 - v_ego_max * tau - r_pred > clear + self.prune_far_margin:
                        continue
                cand.append(
                    {
                        "kind": "circle",
                        "center": c_pred,
                        "min_dist": float(r_pred + clear),
                    }
                )

            # Visible obstacle reachable sets (circles translated with measured velocity).
            for obs in visible_obs:
                obs = np.asarray(obs, dtype=float).reshape(-1)
                ox, oy = float(obs[0]), float(obs[1])
                r_obs = float(obs[2]) if obs.size >= 3 else 0.0
                vx = float(obs[3]) if obs.size >= 4 else 0.0
                vy = float(obs[4]) if obs.size >= 5 else 0.0
                obs_type = int(obs[7]) if obs.size >= 8 else 0
                is_dynamic_agent = obs_type in dyn_types
                if is_dynamic_agent:
                    if self.visible_reach_mode == "constant_velocity":
                        c_pred = np.array([ox + vx * tau, oy + vy * tau], dtype=float)
                        r_pred = r_obs
                    else:
                        # Paper-aligned worst-case reachable set for visible dynamic agents:
                        # no intention assumption, bounded speed in any direction.
                        speed_bound = float(np.hypot(vx, vy))
                        if (not np.isfinite(speed_bound)) or speed_bound < 1e-9:
                            speed_bound = self.v_visible_max_default
                        c_pred = np.array([ox, oy], dtype=float)
                        r_pred = r_obs + speed_bound * tau
                else:
                    # Static obstacle handling (paper Eq. (3a)-style counterpart):
                    # static obstacles are not dynamic reachable sets.
                    c_pred = np.array([ox, oy], dtype=float)
                    r_pred = r_obs
                clear = self.dsafe + self.robot_radius
                if self.prune_far_constraints:
                    # If the robot cannot physically reach the clearance boundary
                    # within tau, this constraint is guaranteed inactive.
                    d0 = float(np.linalg.norm(p0 - c_pred))
                    if d0 - v_ego_max * tau - r_pred > clear + self.prune_far_margin:
                        continue
                cand.append(
                    {
                        "kind": "circle",
                        "center": c_pred,
                        "min_dist": float(r_pred + clear),
                    }
                )

            # Occlusion hidden-agent reachable sets (capsules grown over horizon).
            for sc in occ_scenarios:
                t1 = sc.get("t1", None)
                t2 = sc.get("t2", None)
                if t1 is None or t2 is None:
                    continue
                v_hidden = float(sc.get("v_adv_max", self.v_hidden_max_default))
                if not np.isfinite(v_hidden) or v_hidden < 0.0:
                    v_hidden = self.v_hidden_max_default
                r_capsule = self.hidden_agent_radius + v_hidden * tau
                clear = self.dsafe + self.robot_radius
                if self.prune_far_constraints:
                    c0 = self._closest_point_segment(p0, t1, t2)
                    d0 = float(np.linalg.norm(p0 - c0))
                    if d0 - v_ego_max * tau - r_capsule > clear + self.prune_far_margin:
                        continue
                z_proj = self._project_to_capsule_set(p, t1, t2, r_capsule)
                # Occlusion constraints retain complementarity relaxation.
                cand.append(
                    {
                        "kind": "proj",
                        "point": z_proj,
                        "clear": float(clear),
                        "use_comp": bool(self.use_complementarity),
                    }
                )

            # Keep nearest constraints for runtime stability.
            if len(cand) > self.max_constraints_per_step:
                def _cand_key(it):
                    if str(it.get("kind", "proj")) == "circle":
                        c0 = np.asarray(it.get("center", np.zeros(2)), dtype=float).reshape(2,)
                        d0 = float(np.linalg.norm(p - c0)) - float(it.get("min_dist", 0.0))
                        return max(0.0, d0)
                    z0 = np.asarray(it.get("point", np.zeros(2)), dtype=float).reshape(2,)
                    return float(np.linalg.norm(p - z0))
                cand.sort(key=_cand_key)
                cand = cand[: self.max_constraints_per_step]

            targets[k] = cand

        return targets

    def _build_visible_and_occ_for_oa(self, robot_state, obs_list):
        """
        OA-MPC baseline visibility/reachability construction.

        - First, apply the same LOS-style visibility filtering used by the
          occlusion-CBF pipeline so only truly visible obstacles remain.
        - From those visible obstacles, build:
            1) visible dynamic-agent reachable sets, and
            2) occlusion boundaries generated by selected occluders.

        When `dynamic_occluders=True`, a truly visible moving obstacle can act
        both as a visible reachable-set source and as an occlusion-boundary
        source. This matches the intended "visible body + hidden emergence from
        behind it" interpretation used in the crowd benchmarks.
        """
        arr_full = self._obs_array(obs_list)
        if arr_full.shape[0] == 0:
            return [], [], []

        # Reuse the occlusion-CBF visibility logic so OA-MPC only reasons about
        # obstacles that are actually visible after LOS/occlusion filtering.
        try:
            rs = np.asarray(robot_state, dtype=float).reshape(-1, 1)
            _, _, visible_indices = self._occ_utils._filter_visible_and_build_occ(
                rs, arr_full, return_indices=True
            )
            vis_idx = np.asarray(visible_indices, dtype=int).reshape(-1)
            if vis_idx.size == 0:
                return [], [], []
            arr = arr_full[vis_idx]
        except Exception:
            # Fallback to simple range gating only if the shared visibility
            # filter path fails unexpectedly.
            arr = arr_full
            x = np.asarray(robot_state, dtype=float).reshape(-1)
            p = x[0:2]
            d2 = np.sum((arr[:, 0:2] - p[None, :]) ** 2, axis=1)
            keep = d2 <= (self.sensing_range * self.sensing_range)
            arr = arr[keep]
            if arr.shape[0] == 0:
                return [], [], []

        # Classify sensed obstacles into visible dynamic agents and occluders.
        dyn_types_cfg = self.robot_spec.get("dynamic_obs_types", [1])
        try:
            dyn_types = {int(t) for t in dyn_types_cfg}
        except Exception:
            dyn_types = {1}
        occ_types_cfg = self.robot_spec.get("occlusion_types", None)
        occ_types = None if occ_types_cfg is None else {int(t) for t in occ_types_cfg}

        visible_dyn_obs = []
        occluder_obs = []
        static_occluder_obs = []
        for obs in arr:
            obs = np.asarray(obs, dtype=float).reshape(-1)
            obs_type = int(obs[7]) if obs.size >= 8 else 0
            vx = float(obs[3]) if obs.size >= 4 else 0.0
            vy = float(obs[4]) if obs.size >= 5 else 0.0
            vmag = float(np.hypot(vx, vy))
            is_dynamic_agent = obs_type in dyn_types
            if is_dynamic_agent:
                visible_dyn_obs.append(obs)
            type_ok = (obs_type == 0) if occ_types is None else (obs_type in occ_types)
            static_like = (not np.isfinite(vmag)) or (vmag <= self.occluder_speed_max)
            if type_ok and (self.dynamic_occluders or static_like):
                occluder_obs.append(obs)
            if type_ok and static_like:
                static_occluder_obs.append(obs)

        occ_scenarios = []
        static_pc_circles = []
        if self.use_lidar_jump_occ or self.use_static_pointcloud_constraints:
            scan_for_occ = self._simulate_lidar_scan(robot_state, occluder_obs)
            if self.use_static_pointcloud_constraints:
                # Paper point-cloud circles approximate the visible portion of
                # static obstacles. When moving agents are also allowed to act
                # as occluders, reusing the same scan for static circles
                # double-counts them as:
                #   1) visible dynamic reachable set
                #   2) static LiDAR circle
                #   3) occlusion boundary source
                # This is overly conservative and is the main source of the
                # spurious early infeasibility seen in dense crowd cases.
                # Keep static-circle constraints only for genuinely static-like
                # occluders; dynamic occluders still generate occlusion
                # boundaries via `scan_for_occ`.
                scan_for_static = self._simulate_lidar_scan(robot_state, static_occluder_obs)
                static_pc_circles = self._build_static_pointcloud_circles(scan_for_static)
            if self.use_lidar_jump_occ:
                occ_scenarios = self._build_occ_scenarios_from_lidar_jumps(scan_for_occ)
        else:
            # Legacy fallback path (occlusion utils geometry).
            rs = np.asarray(robot_state, dtype=float).reshape(-1, 1)
            for obs in occluder_obs:
                sc = self._occ_utils._build_occlusion_scenario(
                    rs,
                    obs,
                    is_static=(not self.dynamic_occluders),
                )
                if sc is not None:
                    occ_scenarios.append(sc)

        return visible_dyn_obs, occ_scenarios, static_pc_circles

    def _build_goal_vec(self, x0, goal):
        # Important: do not alias x0 in-place.
        # np.asarray(x0) can return a view of the same array, which would
        # overwrite x0 when filling goal components and corrupt NMPC x0.
        x_goal = np.array(x0, dtype=float, copy=True).reshape(-1)
        if goal is None:
            return x_goal

        g = np.asarray(goal, dtype=float).reshape(-1)
        if g.size >= 2:
            x_goal[0] = g[0]
            x_goal[1] = g[1]
        if self.model in {"Unicycle2D", "DynamicUnicycle2D"} and g.size >= 3:
            x_goal[2] = g[2]
        return x_goal

    def _solve_nmpc(self, x0, u_ref, x_goal, x_init, u_init, proj_targets):
        if not _CASADI_AVAILABLE:
            return False, None, None, "casadi_missing", "CasADi is not installed"

        n, m, N = self._n, self._m, self.N
        lb_u, ub_u = self._input_bounds()

        v_ref_raw, v_ref_schedule, circle_pressure, proj_pressure = self._di_speed_schedule(
            x0=x0,
            u_ref=u_ref,
            x_bar=x_init,
            proj_targets=proj_targets,
        )
        self.last_v_ref_nom_raw = float(v_ref_raw)
        self.last_v_ref_schedule_min = float(np.min(v_ref_schedule)) if v_ref_schedule.size > 0 else float(v_ref_raw)
        self.last_v_ref_schedule_mean = float(np.mean(v_ref_schedule)) if v_ref_schedule.size > 0 else float(v_ref_raw)
        self.last_circle_pressure_score = float(circle_pressure)
        self.last_proj_pressure_score = float(proj_pressure)

        opti = ca.Opti()
        X = opti.variable(n, N + 1)
        U = opti.variable(m, N)

        x0_dm = ca.DM(np.asarray(x0, dtype=float).reshape(n))
        uref_dm = ca.DM(np.asarray(u_ref, dtype=float).reshape(m))
        xgoal_dm = ca.DM(np.asarray(x_goal, dtype=float).reshape(n))
        uprev_dm = ca.DM(np.asarray(self._u_prev_applied, dtype=float).reshape(m))
        vref_sched_dm = ca.DM(np.asarray(v_ref_schedule, dtype=float).reshape(N))

        # Objective:
        # - paper_mode default: ||z-z_goal||^2 + ||u||^2 + ||Delta u||^2
        # - optional tracking extension: ||u-u_ref||^2
        J = 0
        for k in range(N):
            pos_err = X[0:2, k] - xgoal_dm[0:2]
            J += self.w_pos * ca.sumsqr(pos_err)
            if self.use_nominal_tracking_cost:
                J += self.w_u * ca.sumsqr(U[:, k] - uref_dm)
            else:
                J += self.w_u * ca.sumsqr(U[:, k])
            if k == 0:
                J += self.w_du * ca.sumsqr(U[:, k] - uprev_dm)
            else:
                J += self.w_du * ca.sumsqr(U[:, k] - U[:, k - 1])
            if self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost:
                v_par, v_lat = self._di_progress_components_ca(X[:, k], xgoal_dm[0:2])
                J += self.di_w_speed * ((v_par - vref_sched_dm[k]) ** 2)
                J += self.di_w_lateral_velocity * (v_lat ** 2)

        J += self.w_terminal * ca.sumsqr(X[0:2, N] - xgoal_dm[0:2])

        # Dynamics and bounds.
        opti.subject_to(X[:, 0] == x0_dm)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._discrete_dynamics_ca(X[:, k], U[:, k]))
            opti.subject_to(opti.bounded(lb_u, U[:, k], ub_u))
            if self.model == "DoubleIntegrator2D" and self.di_use_speed_cap:
                v_cap = float(self.robot_spec.get("v_max", 1.0))
                if (not np.isfinite(v_cap)) or v_cap <= 0.0:
                    v_cap = 1.0
                opti.subject_to(ca.sumsqr(X[2:4, k + 1]) <= (v_cap + 1e-6) ** 2)

        # State bounds (model-specific).
        if self.model == "DoubleIntegrator2D":
            v_max = float(self.robot_spec.get("v_max", 1.0))
            for k in range(N + 1):
                opti.subject_to(opti.bounded(-v_max, X[2, k], v_max))
                opti.subject_to(opti.bounded(-v_max, X[3, k], v_max))
        elif self.model == "DynamicUnicycle2D":
            v_max = float(self.robot_spec.get("v_max", 1.0))
            v_min = float(self.robot_spec.get("v_min", -v_max))
            for k in range(N + 1):
                opti.subject_to(opti.bounded(v_min, X[3, k], v_max))

        # Terminal stopping constraint for recursive-feasibility condition.
        #
        # Writing this as X[:,N] == X[:,N-1] is algebraically valid, but for
        # UNI/DU it introduces redundant stop equalities through the dynamics
        # (e.g. x_N-x_{N-1}=dt*v*cos(theta), y_N-y_{N-1}=dt*v*sin(theta)),
        # which become rank-deficient at v=0 and can trigger IPOPT
        # Restoration_Failed even when a feasible warm start exists.
        #
        # Use exact, model-specific stop conditions instead:
        # - UNI: terminal input is zero -> x,y,theta do not change
        # - DU: terminal speed is zero and terminal input is zero
        # - DI exact: terminal velocity is zero and terminal acceleration is zero
        # - DI brake_reachable: terminal speed can be canceled within a small
        #   number of additional control intervals under bounded acceleration
        if self.model == "Unicycle2D":
            opti.subject_to(U[:, N - 1] == 0.0)
        elif self.model == "DynamicUnicycle2D":
            opti.subject_to(X[3, N - 1] == 0.0)
            opti.subject_to(U[:, N - 1] == 0.0)
        else:
            if self.di_terminal_stop_mode == "exact":
                opti.subject_to(X[2, N - 1] == 0.0)
                opti.subject_to(X[3, N - 1] == 0.0)
                opti.subject_to(U[:, N - 1] == 0.0)
            elif self.di_terminal_stop_mode == "brake_reachable":
                a_max = float(self.robot_spec.get("a_max", 1.0))
                v_tol = float(max(0.0, a_max * self.dt * self.di_terminal_brake_steps))
                opti.subject_to(opti.bounded(-v_tol, X[2, N], v_tol))
                opti.subject_to(opti.bounded(-v_tol, X[3, N], v_tol))
            elif self.di_terminal_stop_mode == "input_zero_only":
                opti.subject_to(U[:, N - 1] == 0.0)

        collision_constraints = 0
        terminal_relaxed_constraints = 0
        terminal_relax_cost = 0
        for k in range(1, N + 1):
            move_dx = X[0, k] - X[0, k - 1]
            move_dy = X[1, k] - X[1, k - 1]
            # Paper-style complementarity:
            #   ||z_k-z_{k-1}|| * g_t(z_k) <= 0
            # To preserve "stopped => zero multiplier" exactly (and avoid
            # infeasibility caused by epsilon-bias), use squared norm:
            #   (||z_k-z_{k-1}||^2) * g_t(z_k) <= 0
            # This is equivalent for sign logic and numerically smoother.
            move_sq = move_dx * move_dx + move_dy * move_dy
            g_terms = []
            for cst in proj_targets[k]:
                kind = str(cst.get("kind", "proj")).strip().lower()
                is_terminal_step = (k == N)
                if kind == "circle":
                    cxy = np.asarray(cst.get("center", np.zeros(2)), dtype=float).reshape(2,)
                    min_dist = float(cst.get("min_dist", 0.0))
                    dx = X[0, k] - float(cxy[0])
                    dy = X[1, k] - float(cxy[1])
                    dist_sq = dx * dx + dy * dy
                    min_dist_sq = min_dist * min_dist
                    if self.use_complementarity and self.complementarity_for_circles:
                        # Paper complementarity condition can be applied to all
                        # collision rows: if already inside safety margin, allow
                        # only zero-motion at that prediction step.
                        # Use squared-distance form for numerical robustness.
                        # sign(min_dist - dist) == sign(min_dist^2 - dist^2)
                        # for nonnegative distances.
                        g_terms.append(min_dist_sq - dist_sq)
                    elif (
                        is_terminal_step
                        and self.use_terminal_complementarity
                        and self.terminal_complementarity_for_circles
                    ):
                        slack = opti.variable()
                        opti.subject_to(slack >= 0.0)
                        opti.subject_to(slack >= (min_dist_sq - dist_sq))
                        terminal_relax_cost += self.terminal_relax_weight * slack * slack
                        terminal_relaxed_constraints += 1
                    else:
                        # Optional hard static/visible distance constraint.
                        opti.subject_to(dist_sq >= min_dist_sq)
                    collision_constraints += 1
                    continue

                z_proj = np.asarray(cst.get("point", np.zeros(2)), dtype=float).reshape(2,)
                clear = float(cst.get("clear", 0.0))
                use_comp = bool(cst.get("use_comp", self.use_complementarity))
                dx = X[0, k] - float(z_proj[0])
                dy = X[1, k] - float(z_proj[1])
                dist_sq = dx * dx + dy * dy
                clear_sq = clear * clear
                if self.use_complementarity and use_comp:
                    # Complementarity relaxation for occlusion constraints.
                    # Use squared-distance form for numerical robustness.
                    g_violation = clear_sq - dist_sq
                    g_terms.append(g_violation)
                elif (
                    is_terminal_step
                    and self.use_terminal_complementarity
                    and self.terminal_complementarity_for_occlusion
                ):
                    slack = opti.variable()
                    opti.subject_to(slack >= 0.0)
                    opti.subject_to(slack >= (clear_sq - dist_sq))
                    terminal_relax_cost += self.terminal_relax_weight * slack * slack
                    terminal_relaxed_constraints += 1
                else:
                    opti.subject_to(dist_sq >= clear_sq)
                collision_constraints += 1
            if g_terms:
                if self.aggregate_complementarity and len(g_terms) > 1:
                    kappa = max(1e-6, self.complementarity_smax_kappa)
                    stack = ca.vertcat(*g_terms)
                    gk = ca.log(ca.sum1(ca.exp(kappa * stack)) + 1e-12) / kappa
                    opti.subject_to(move_sq * gk <= 0.0)
                else:
                    # Numerically stable complementarity:
                    # g_pos = max(0, g_1, ..., g_m), then move_sq * g_pos <= 0
                    # This preserves the intended logic:
                    # - if all g_i <= 0 -> g_pos = 0 (free motion)
                    # - if any g_i > 0 -> g_pos > 0 -> move_sq = 0 (stop)
                    g_pos = opti.variable()
                    opti.subject_to(g_pos >= 0.0)
                    for gk in g_terms:
                        opti.subject_to(g_pos >= gk)
                    opti.subject_to(move_sq * g_pos <= 0.0)

        opti.minimize(J + terminal_relax_cost)

        # Warm start.
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

        if self.solver_name == "ipopt":
            opti.solver("ipopt", p_opts, s_opts)
        else:
            # fallback: ipopt is the main tested path for this OA-MPC baseline.
            opti.solver("ipopt", p_opts, s_opts)

        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            solve_ms = (time.perf_counter() - t0) * 1000.0
            x_sol = np.array(sol.value(X), dtype=float)
            u_sol = np.array(sol.value(U), dtype=float)
            return True, x_sol, u_sol, "optimal", "", solve_ms, collision_constraints, terminal_relaxed_constraints
        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            return False, None, None, "infeasible", str(exc), solve_ms, collision_constraints, terminal_relaxed_constraints

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()

        x0 = self._normalize_state(np.asarray(robot_state, dtype=float).reshape(-1))
        u_ref = np.asarray(control_ref.get("u_ref", np.zeros((self._m, 1))), dtype=float).reshape(-1, 1)
        u_ref = self._clip_input(u_ref)
        goal = control_ref.get("goal", None)

        self.last_u_ref = u_ref
        self.last_qp_exception = ""

        # Build visible dynamic agents + occlusion scenarios + static point-cloud circles.
        visible_obs, occ_scenarios, static_pc_circles = self._build_visible_and_occ_for_oa(
            robot_state, obs_list
        )
        self.occlusion_scenarios = occ_scenarios
        self._last_static_pc_circles = static_pc_circles

        x_goal = self._build_goal_vec(x0, goal)
        x_guess, u_guess = self._shift_or_init_plan(x0, u_ref)
        best_sol = None
        best_proj_targets = None
        total_solve_ms = 0.0
        max_n_coll = 0
        max_n_terminal_relaxed = 0
        qp_status = "infeasible"
        qp_exc = ""
        alt_iter_count = 0

        tol = max(0.0, float(self.alternation_tol))
        for alt_idx in range(self.alternation_iters):
            alt_iter_count = alt_idx + 1
            proj_targets = self._build_projection_targets(
                x_guess,
                visible_obs,
                occ_scenarios,
                static_pc_circles,
            )
            (
                ok_i,
                x_sol_i,
                u_sol_i,
                status_i,
                exc_i,
                solve_ms_i,
                n_coll_i,
                n_term_relaxed_i,
            ) = self._solve_nmpc(
                x0=x0,
                u_ref=u_ref,
                x_goal=x_goal,
                x_init=x_guess,
                u_init=u_guess,
                proj_targets=proj_targets,
            )
            total_solve_ms += float(solve_ms_i)
            max_n_coll = max(max_n_coll, int(n_coll_i))
            max_n_terminal_relaxed = max(max_n_terminal_relaxed, int(n_term_relaxed_i))
            qp_status = status_i
            qp_exc = exc_i

            if not ok_i:
                # Keep the latest feasible iterate if we have one.
                break

            best_sol = (x_sol_i, u_sol_i)
            best_proj_targets = proj_targets

            # Fixed-point update for projection/NMPC alternation.
            dx = float(np.max(np.abs(x_sol_i - x_guess)))
            du = float(np.max(np.abs(u_sol_i - u_guess)))
            x_guess = x_sol_i
            u_guess = u_sol_i
            if max(dx, du) <= tol:
                break

        ok = best_sol is not None
        if ok:
            x_sol, u_sol = best_sol
            proj_targets = best_proj_targets if best_proj_targets is not None else []
        else:
            x_sol, u_sol = None, None
            proj_targets = []
        self._last_projection_points = proj_targets

        if ok:
            u_cmd = self._clip_input(u_sol[:, 0].reshape(-1, 1))
            self._x_prev_plan = x_sol
            self._u_prev_plan = u_sol
            self.status = "optimal"
            if self.use_nominal_tracking_cost:
                self.last_intervention = (
                    "u_ref"
                    if float(np.linalg.norm(u_cmd - u_ref)) <= float(self.robot_spec.get("intervention_tol", 1e-3))
                    else "backup_qp"
                )
            else:
                self.last_intervention = "oa_mpc"
            self.last_qp_status_raw = qp_status
            self.last_qp_exception = ""
        else:
            u_cmd = self._stop_input()
            if self.allow_solver_fallback:
                x_fallback, u_fallback = self._shift_or_init_plan(x0, u_cmd)
                self._x_prev_plan = x_fallback
                self._u_prev_plan = u_fallback
                self.status = "optimal"
                self.last_intervention = "backup_fallback"
            else:
                self.status = "infeasible"
                self.last_intervention = "infeasible"
            self.last_qp_status_raw = qp_status
            self.last_qp_exception = qp_exc

        self.last_u = u_cmd
        self._u_prev_applied = u_cmd

        # Reported constraint count focuses on collision-avoidance rows in NMPC.
        self.last_num_constraints = int(max_n_coll)
        self.last_qp_solve_time_ms = float(total_solve_ms)
        self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
        self.last_profile = {
            "total_ms": self.last_total_compute_time_ms,
            "solver_ms": self.last_qp_solve_time_ms,
            "num_collision_constraints": self.last_num_constraints,
            "num_visible_obs": int(len(visible_obs)),
            "num_occ_scenarios": int(len(occ_scenarios)),
            "num_static_pc_circles": int(len(static_pc_circles)),
            "num_terminal_relaxed_constraints": int(max_n_terminal_relaxed),
            "use_terminal_complementarity": bool(self.use_terminal_complementarity),
            "di_terminal_stop_mode": str(self.di_terminal_stop_mode),
            "di_terminal_brake_steps": int(self.di_terminal_brake_steps),
            "raw_v_ref_nom": float(self.last_v_ref_nom_raw),
            "speed_schedule_min": float(self.last_v_ref_schedule_min),
            "speed_schedule_mean": float(self.last_v_ref_schedule_mean),
            "circle_pressure_score": float(self.last_circle_pressure_score),
            "proj_pressure_score": float(self.last_proj_pressure_score),
            "speed_cost_mode": (
                "progress_aligned" if (self.model == "DoubleIntegrator2D" and self.di_use_progress_speed_cost) else "input_only"
            ),
            "alternation_iters_run": int(alt_iter_count),
            "alternation_iters_cfg": int(self.alternation_iters),
        }

        return u_cmd
