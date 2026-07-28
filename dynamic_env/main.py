from base_control.dynamic_env.main import LocalTrackingControllerDyn
from base_control.position_control.cbf_qp import CBFQP as BaseCBFQP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cvxpy as cp
import os
import glob
import subprocess
import csv
import time

class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)


class LocalCBFQP(BaseCBFQP):

    def __init__(self, robot, robot_spec, num_obs=1):
        self.status = "optimal"
        self.last_u = None
        self.last_u_ref = None
        self.last_intervention = "u_ref"
        self.last_num_constraints = 0
        self.last_qp_solve_time_ms = 0.0
        self.last_total_compute_time_ms = 0.0
        self.last_profile = {}
        super().__init__(robot, robot_spec, num_obs=num_obs)

    def _dynamic_obstacle_types(self):
        configured = self.robot_spec.get("dynamic_obs_types", (1,))
        if np.isscalar(configured):
            configured = (configured,)
        return {int(value) for value in configured}

    def _is_dynamic_obs(self, obs):
        row = np.asarray(obs, dtype=float).reshape(-1)
        return row.size >= 8 and int(round(float(row[7]))) in self._dynamic_obstacle_types()

    def _circle_barrier(self, robot_state, obs):
        """Return the moving-circle CBF terms used by the retained robot models."""
        row = np.asarray(obs, dtype=float).reshape(-1)
        if row.size < 5:
            row = np.pad(row, (0, 5 - row.size))
        if not self._is_dynamic_obs(row):
            row = row.copy()
            row[3:5] = 0.0
        return self.robot.robot.dynamic_agent_barrier(
            robot_state,
            row,
            self.robot.robot_radius,
            beta=float(self.robot_spec.get("cbf_beta", 1.01)),
        )

    def _get_position_corridor(self):
        cfg = self.robot_spec.get("position_corridor", None)
        if not isinstance(cfg, dict):
            return None
        if not bool(cfg.get("enabled", True)):
            return None
        x_min = cfg.get("x_min", None)
        x_max = cfg.get("x_max", None)
        if x_min is None or x_max is None:
            return None
        try:
            x_min = float(x_min)
            x_max = float(x_max)
        except Exception:
            return None
        if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or x_min >= x_max:
            return None
        buffer = cfg.get("buffer", self.robot_spec.get("radius", 0.0))
        try:
            buffer = float(buffer)
        except Exception:
            buffer = 0.0
        if (not np.isfinite(buffer)) or buffer < 0.0:
            buffer = 0.0
        x_min_eff = x_min + buffer
        x_max_eff = x_max - buffer
        if x_min_eff >= x_max_eff:
            return None
        return {
            "x_min": x_min_eff,
            "x_max": x_max_eff,
            "alpha": float(cfg.get("alpha", self.cbf_param.get("alpha", 1.0))),
            "alpha1": float(cfg.get("alpha1", self.cbf_param.get("alpha1", 1.5))),
            "alpha2": float(cfg.get("alpha2", self.cbf_param.get("alpha2", 1.5))),
        }

    def _build_position_corridor_constraints(self, robot_state):
        cfg = self._get_position_corridor()
        if cfg is None:
            return []
        X = np.asarray(robot_state, dtype=float).reshape(-1, 1)
        model_name = str(self.robot_spec.get("model", "")).strip()
        rows = []

        def _append_row(A_row, b_row):
            A_row = np.asarray(A_row, dtype=float).reshape(1, -1)
            b_row = float(b_row)
            if np.all(np.isfinite(A_row)) and np.isfinite(b_row):
                rows.append((A_row, np.array([[b_row]])))

        if model_name == 'DoubleIntegrator2D':
            x = float(X[0, 0])
            vx = float(X[2, 0])
            gamma1 = float(cfg["alpha1"]) + float(cfg["alpha2"])
            gamma2 = float(cfg["alpha1"]) * float(cfg["alpha2"])
            _append_row(np.array([[1.0, 0.0]]), gamma1 * vx + gamma2 * (x - float(cfg["x_min"])))
            _append_row(np.array([[-1.0, 0.0]]), gamma1 * (-vx) + gamma2 * (float(cfg["x_max"]) - x))
            return rows

        if model_name == 'Unicycle2D':
            c = float(np.cos(X[2, 0]))
            alpha = float(cfg["alpha"])
            _append_row(np.array([[c, 0.0]]), alpha * (float(X[0, 0]) - float(cfg["x_min"])))
            _append_row(np.array([[-c, 0.0]]), alpha * (float(cfg["x_max"]) - float(X[0, 0])))
            return rows

        if model_name == 'DynamicUnicycle2D':
            x = float(X[0, 0])
            theta = float(X[2, 0])
            v = float(X[3, 0])
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            gamma1 = float(cfg["alpha1"]) + float(cfg["alpha2"])
            gamma2 = float(cfg["alpha1"]) * float(cfg["alpha2"])
            _append_row(np.array([[c, -v * s]]), gamma1 * (v * c) + gamma2 * (x - float(cfg["x_min"])))
            _append_row(np.array([[-c, v * s]]), gamma1 * (-v * c) + gamma2 * (float(cfg["x_max"]) - x))
            return rows

        return []

    def _count_active_constraint_rows(self):
        try:
            A = np.asarray(self.A1.value, dtype=float)
            b = np.asarray(self.b1.value, dtype=float).reshape(-1)
        except Exception:
            return 0
        if A.ndim != 2 or A.shape[0] == 0:
            return 0
        used = np.any(np.abs(A) > 1e-12, axis=1)
        if b.shape[0] == A.shape[0]:
            used = np.logical_or(used, np.abs(b) > 1e-12)
        return int(np.count_nonzero(used))

    def _finalize_runtime_stats(self, t_all0, u_ref, u_out, solve_ms=None, num_constraints=None):
        self.last_u_ref = np.asarray(u_ref, dtype=float).reshape(-1, 1)
        self.last_u = self.last_u_ref if u_out is None else np.asarray(u_out, dtype=float).reshape(-1, 1)
        if num_constraints is None:
            num_constraints = self._count_active_constraint_rows()
        self.last_num_constraints = int(num_constraints)
        if solve_ms is None:
            solve_ms = (time.perf_counter() - t_all0) * 1000.0
        self.last_qp_solve_time_ms = float(solve_ms)
        self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
        tol = float(self.robot_spec.get("intervention_tol", 1e-3))
        m = min(self.last_u.shape[0], self.last_u_ref.shape[0])
        delta = float(np.linalg.norm(self.last_u[:m] - self.last_u_ref[:m])) if m > 0 else 0.0
        self.last_intervention = "u_ref" if delta <= tol else "backup_qp"
        self.last_profile = {
            "total_ms": float(self.last_total_compute_time_ms),
            "solve_ms": float(self.last_qp_solve_time_ms),
            "num_constraints": int(self.last_num_constraints),
        }

    def setup_control_problem(self):
        if self._get_position_corridor() is None:
            self._corridor_num_rows = 0
            return super().setup_control_problem()

        corridor_rows = 2
        self._corridor_num_rows = int(corridor_rows)
        total_rows = int(max(1, self.num_obs + self._corridor_num_rows))
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((total_rows, 2), value=np.zeros((total_rows, 2)))
        self.b1 = cp.Parameter((total_rows, 1), value=np.zeros((total_rows, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

        if self.robot_spec['model'] == 'SingleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <=  self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <=  self.robot_spec['v_max']]
        elif self.robot_spec['model'] == 'Unicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['a_max']]
        elif 'KinematicBicycle2D' in self.robot_spec['model']:
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['beta_max']]
        elif self.robot_spec['model'] == 'Quad2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.robot_spec["f_min"] <= self.u[0],
                           self.u[0] <= self.robot_spec["f_max"],
                           self.robot_spec["f_min"] <= self.u[1],
                           self.u[1] <= self.robot_spec["f_max"]]
        elif self.robot_spec['model'] == 'Quad3D':
            self.u = cp.Variable((4, 1))
            self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
            self.A1 = cp.Parameter((1, 4), value=np.zeros((1, 4)))
            self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.u[0] <= self.robot_spec['u_max'],
                            self.u[0] >= 0.0,
                           cp.abs(self.u[1]) <= self.robot_spec['u_max'],
                           cp.abs(self.u[2]) <= self.robot_spec['u_max'],
                           cp.abs(self.u[3]) <= self.robot_spec['u_max']]
        elif self.robot_spec['model'] == 'Manipulator2D':
            self.u = cp.Variable((3, 1))
            self.u_ref = cp.Parameter((3, 1), value=np.zeros((3, 1)))
            self.A1 = cp.Parameter((self.num_obs, 3), value=np.zeros((self.num_obs, 3)))
            self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u) <= self.robot_spec['w_max']]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()
        self.A1.value[:] = 0
        self.b1.value[:] = 0
        corridor_rows = self._build_position_corridor_constraints(robot_state)

        if obs_list is None and len(corridor_rows) == 0:
            self.u_ref.value = control_ref['u_ref']
            if self.robot_spec['model'] in ['Quad3D']:
                self.u_ref.value = np.vstack((self.u_ref.value, self.u_ref.value))
            # A no-obstacle step still needs the QP's actuator bounds. Returning
            # the nominal reference directly can violate them (notably for the
            # kinematic unicycle when its waypoint is far away).
            t_solve0 = time.perf_counter()
            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
            self.status = self.cbf_controller.status
            solve_ms = (time.perf_counter() - t_solve0) * 1000.0
            u_out = self.u.value
            self._finalize_runtime_stats(
                t_all0,
                self.u_ref.value,
                u_out,
                solve_ms=solve_ms,
                num_constraints=0,
            )
            return u_out

        mode = self.robot_spec.get('cbf_mode', 'cbf')
        obs_iter = [] if obs_list is None else obs_list
        max_obs_rows = int(self.A1.shape[0]) - int(getattr(self, "_corridor_num_rows", 0))
        row_idx = 0
        for obs in obs_iter:
            if obs is None:
                continue
            if row_idx >= max_obs_rows:
                break

            dt = self.robot.dt
            if self.robot_spec['model'] == 'Unicycle2D':
                h, dh_dx, h_t = self._circle_barrier(robot_state, obs)
                if mode == 'hard':
                    self.A1.value[row_idx, :] = dh_dx @ self.robot.g()
                    self.b1.value[row_idx, :] = h / dt + dh_dx @ self.robot.f() + h_t
                else:
                    self.A1.value[row_idx, :] = dh_dx @ self.robot.g()
                    self.b1.value[row_idx, :] = dh_dx @ self.robot.f() + h_t + self.cbf_param['alpha'] * h
            elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D']:
                h, h_dot, dh_dot_dx, h_dot_t = self._circle_barrier(robot_state, obs)
                if mode == 'hard':
                    self.A1.value[row_idx, :] = dh_dot_dx @ self.robot.g()
                    self.b1.value[row_idx, :] = h / (dt**2) + 2 * h_dot / dt + dh_dot_dx @ self.robot.f() + h_dot_t
                else:
                    gamma1 = self.cbf_param['alpha1'] + self.cbf_param['alpha2']
                    gamma2 = self.cbf_param['alpha1'] * self.cbf_param['alpha2']
                    self.A1.value[row_idx, :] = dh_dot_dx @ self.robot.g()
                    self.b1.value[row_idx, :] = dh_dot_dx @ self.robot.f() + h_dot_t + gamma1 * h_dot + gamma2 * h
            row_idx += 1

        for A_row, b_row in corridor_rows:
            if row_idx >= int(self.A1.shape[0]):
                break
            self.A1.value[row_idx, :] = np.asarray(A_row, dtype=float).reshape(1, -1)
            self.b1.value[row_idx, :] = np.asarray(b_row, dtype=float).reshape(1, -1)
            row_idx += 1

        self.u_ref.value = control_ref['u_ref']
        t_solve0 = time.perf_counter()
        self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        self.status = self.cbf_controller.status
        solve_ms = (time.perf_counter() - t_solve0) * 1000.0
        u_out = self.u.value
        self._finalize_runtime_stats(t_all0, self.u_ref.value, u_out, solve_ms=solve_ms, num_constraints=row_idx)
        return u_out

class LocalTrackingControllerDyn_OCC(LocalTrackingControllerDyn):

    def setup_animation_plot(self):
        if self.ax is None:
            self.ax = plt.axes()
        if self.fig is None:
            self.fig = plt.figure()
        if self.show_animation:
            plt.ion()
        else:
            plt.ioff()
        self.ax.set_xlabel("X [m]")
        if self.robot_spec['model'] in ['Quad2D', 'VTOL2D']:
            self.ax.set_ylabel("Z [m]")
        else:
            self.ax.set_ylabel("Y [m]")
        self.ax.set_aspect(1)
        if len(getattr(self.fig, "axes", [])) <= 1:
            self.fig.tight_layout()
        self.waypoints_scatter = self.ax.scatter(
            [], [], s=10, facecolors='g', edgecolors='g', alpha=0.5)

    def __init__(self, X0, robot_spec,
                 controller_type=None,
                 dt=0.05,
                 show_animation=False, save_animation=False, show_mpc_traj=False,
                 enable_rotation=True, raise_error=False,
                 ax=None, fig=None, env=None, rand_seed=42,
                 tracking_view_ax=None, tracking_view_window_size=None):
        requested_pos_type = str(
            (controller_type or {}).get("pos", "cbf_qp")
        ).strip().lower() or "cbf_qp"
        if requested_pos_type in {"occlusion_cbf", "occlusion_cbf_qp"}:
            from position_control.ocbf.defaults import apply_ocbf_best_parameters

            apply_ocbf_best_parameters(robot_spec)
        # Build the shared tracking state with the in-tree baseline controller,
        # then replace it with the explicitly requested project controller.
        bootstrap_controller_type = dict(controller_type or {})
        bootstrap_controller_type["pos"] = "cbf_qp"
        super().__init__(X0, robot_spec,
                         controller_type=bootstrap_controller_type,
                         dt=dt,
                         show_animation=show_animation, save_animation=save_animation, show_mpc_traj=show_mpc_traj,
                         enable_rotation=enable_rotation, raise_error=raise_error,
                         ax=ax, fig=fig, env=env)

        self.pos_controller_type = requested_pos_type
        if requested_pos_type == 'cbf_qp':
            self.pos_controller = LocalCBFQP(self.robot, self.robot_spec, num_obs=self.num_constraints)
        elif requested_pos_type == 'occlusion_cbf_qp':
            from position_control.occlusion_cbf_qp import OcclusionCBFQP
            self.pos_controller = OcclusionCBFQP(self.robot, self.robot_spec, num_obs=30)
        elif requested_pos_type == 'oa_mpc':
            from position_control.oa_mpc import OAMPC
            self.pos_controller = OAMPC(self.robot, self.robot_spec, num_obs=30)
        elif requested_pos_type == 'single_risk_mpc':
            from position_control.single_risk_mpc import SingleRiskMPC
            self.pos_controller = SingleRiskMPC(self.robot, self.robot_spec, num_obs=30)
        elif requested_pos_type == 'control_tree_mpc':
            from position_control.control_tree_mpc import ControlTreeMPC
            self.pos_controller = ControlTreeMPC(self.robot, self.robot_spec, num_obs=30)
        elif requested_pos_type == 'oacp_mpc':
            from position_control.oacp_mpc import OACPMPC
            self.pos_controller = OACPMPC(self.robot, self.robot_spec, num_obs=30)
        else:
            raise ValueError(f"Unknown position controller: {requested_pos_type}")

        self.qp_stats_text = None
        self.backup_rollout_line = None
        self.backup_rollout_end_marker = None
        self.backup_safe_contour = []
        self._backup_vis_counter = 0
        self.show_backup_rollout = bool(self.robot_spec.get('show_backup_rollout', False))
        self.backup_rollout_every = int(self.robot_spec.get('backup_rollout_every', 1))

        self.plot_dyn_obs = bool(self.robot_spec.get('plot_dyn_obs', True))
        self.plot_dyn_obs_arrows = bool(self.robot_spec.get('plot_dyn_obs_arrows', True))
        self.plot_dyn_obs_occlusion = bool(self.robot_spec.get('plot_dyn_obs_occlusion', True))
        self.plot_dyn_obs_adaptive_arrow = bool(self.robot_spec.get('plot_dyn_obs_adaptive_arrow', False))
        self.plot_dyn_obs_arrow_len_min = float(self.robot_spec.get('plot_dyn_obs_arrow_len_min', 0.3))
        self.plot_dyn_obs_arrow_len_max = float(self.robot_spec.get('plot_dyn_obs_arrow_len_max', 0.8))
        self.plot_dyn_obs_arrow_head = float(self.robot_spec.get('plot_dyn_obs_arrow_head', 6.0))
        self.plot_dyn_obs_visible_color = self.robot_spec.get('plot_dyn_obs_visible_color', 'gray')
        self.plot_dyn_obs_hidden_color = self.robot_spec.get('plot_dyn_obs_hidden_color', 'orange')
        self.plot_dyn_obs_arrow_color = self.robot_spec.get('plot_dyn_obs_arrow_color', 'deepskyblue')
        self.plot_occ_polygons = bool(
            self.robot_spec.get(
                'plot_occ_polygons',
                self.show_animation or self.save_animation,
            )
        )
        self.continue_on_infeasible = bool(self.robot_spec.get('continue_on_infeasible', False))
        self._infeasible_active = False
        self._infeasible_text = None
        self._tracking_infeasible_text = None
        self._infeasible_seen = False
        self.plot_every = int(self.robot_spec.get('plot_every', 1))
        if self.plot_every < 1:
            self.plot_every = 1
        self.plot_pause = float(self.robot_spec.get('plot_pause', 0.01))
        if self.plot_pause < 0.0:
            self.plot_pause = 0.0
        self._plot_counter = 0
        self.occlusion_mask_every = int(self.robot_spec.get('occlusion_mask_every', self.plot_every))
        if self.occlusion_mask_every < 1:
            self.occlusion_mask_every = 1
        self._occ_mask_counter = 0
        self._cached_occluded_mask = None
        self._last_occluded_mask = None
        self.last_terminal_event = None

        self._rng = np.random.default_rng(rand_seed)
        self.obs_meta = None
        self._dyn_behavior_vis_mask_cache = None
        self._dyn_behavior_fov_mask_cache = None
        self._dyn_behavior_vis_cache_step = -1
        self._dyn_behavior_step_counter = 0
        self.tracking_view_ax = tracking_view_ax
        self.tracking_view_window_size = float(tracking_view_window_size) if tracking_view_window_size is not None else 5.0
        if self.tracking_view_window_size <= 0.0:
            self.tracking_view_window_size = 5.0
        self.tracking_view_enabled = bool(
            (self.show_animation or self.save_animation)
            and (self.tracking_view_ax is not None)
        )
        self.tracking_view_waypoints_scatter = None
        self.tracking_view_robot_patch = None
        self.tracking_view_heading_line = None
        self.tracking_view_fov_line = None
        self.tracking_view_fov_fill = None
        self.tracking_view_backup_rollout_line = None
        self.tracking_view_backup_rollout_end_marker = None
        self.plot_robot_trajectory = bool(self.robot_spec.get('plot_robot_trajectory', True))
        self.plot_robot_trajectory_cmap = self.robot_spec.get(
            'plot_robot_trajectory_cmap', 'trajectory_red_pink'
        )
        self.plot_robot_trajectory_linewidth = float(
            self.robot_spec.get('plot_robot_trajectory_linewidth', 2.8)
        )
        self.plot_robot_trajectory_alpha = float(
            self.robot_spec.get('plot_robot_trajectory_alpha', 0.95)
        )
        self.plot_robot_trajectory_zorder = float(
            self.robot_spec.get('plot_robot_trajectory_zorder', 9.0)
        )
        self.plot_robot_trajectory_speed_max = self.robot_spec.get(
            'plot_robot_trajectory_speed_max', None
        )
        self.robot_trajectory_points = []
        self.robot_trajectory_speeds = []
        self.robot_trajectory_collection = None
        self._trajectory_cmap_obj = None
        self._trajectory_cache_dirty = True
        self._trajectory_cached_segments = np.empty((0, 2, 2), dtype=float)
        self._trajectory_cached_segment_speeds = np.empty((0,), dtype=float)
        self._trajectory_cached_vmax = 1.0
        self.tracking_view_trajectory_collection = None
        self.tracking_view_occ_patches = []
        self.tracking_view_obs_patches = []
        self.tracking_view_obs_arrows = []
        self._record_trajectory_sample()
        if self.tracking_view_enabled:
            self._setup_tracking_view()

    def setup_animation_saving(self):
        self.current_directory_path = os.getcwd()
        frame_ext = str(self.robot_spec.get("animation_frame_ext", "png")).strip().lower().lstrip(".")
        if frame_ext not in {"png", "svg"}:
            frame_ext = "png"
        subdir = str(self.robot_spec.get("animation_subdir", "")).strip()
        save_root = os.path.join(self.current_directory_path, "output", "animations")
        if subdir:
            save_root = os.path.join(save_root, subdir)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        self.save_folder = save_root
        self.save_frame_ext = frame_ext
        for stale_frame in glob.glob(
            os.path.join(self.save_folder, f"t_step_*.{self.save_frame_ext}")
        ):
            os.remove(stale_frame)
        for stale_video_name in ("tracking.mp4", "tracking.tmp.mp4"):
            stale_video_path = os.path.join(self.save_folder, stale_video_name)
            if os.path.isfile(stale_video_path):
                os.remove(stale_video_path)
        self.save_frame_dpi = int(self.robot_spec.get("animation_frame_dpi", 300))
        self.animation_export_video = bool(
            self.robot_spec.get("animation_export_video", self.save_frame_ext == "png")
        )
        self.save_per_frame = int(
            self.robot_spec.get("animation_frame_stride", 1)
        )
        if self.save_per_frame < 1:
            self.save_per_frame = 1
        self.ani_idx = 0

    def _frame_output_path(self, frame_idx):
        return os.path.join(
            self.save_folder,
            f"t_step_{int(frame_idx):04d}.{self.save_frame_ext}",
        )

    def export_video(self):
        if not self.save_animation:
            return
        if self.save_frame_ext != "png" or not self.animation_export_video:
            return
        video_path = os.path.join(self.save_folder, 'tracking.mp4')
        temporary_video_path = os.path.join(
            self.save_folder, 'tracking.tmp.mp4'
        )
        command = [
            'ffmpeg',
            '-y',
            '-framerate', '30',
            '-i', os.path.join(self.save_folder, 't_step_%04d.png'),
            '-vf', 'scale=1920:982,fps=60',
            '-pix_fmt', 'yuv420p',
            temporary_video_path,
        ]
        try:
            completed = subprocess.run(command, check=False)
        except OSError as exc:
            print(f"[ANIMATION] ffmpeg could not run; PNG frames were kept: {exc}")
            return
        if completed.returncode != 0:
            if os.path.isfile(temporary_video_path):
                os.remove(temporary_video_path)
            print(
                f"[ANIMATION] ffmpeg exited with code {completed.returncode}; "
                f"PNG frames were kept in {self.save_folder}"
            )
            return

        os.replace(temporary_video_path, video_path)
        for file_name in glob.glob(os.path.join(self.save_folder, "t_step_*.png")):
            os.remove(file_name)
        print(f"[ANIMATION] Saved video: {video_path}")

    @staticmethod
    def _wrap_angle(theta):
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    @staticmethod
    def _safe_normalize(vec, eps=1e-9):
        n = float(np.linalg.norm(vec))
        if n <= eps:
            return None
        return vec / n

    def _compute_robot_speed_scalar(self):
        model = str(self.robot_spec.get("model", ""))
        X = np.asarray(getattr(self.robot, "X", np.zeros((0, 1))), dtype=float).reshape(-1)
        u_pos = getattr(self, "u_pos", None)
        if u_pos is None:
            u_arr = np.asarray(getattr(self.robot, "U", np.zeros((0, 1))), dtype=float).reshape(-1)
        else:
            u_arr = np.asarray(u_pos, dtype=float).reshape(-1)

        if model == "SingleIntegrator2D":
            if u_arr.size >= 2:
                return float(np.linalg.norm(u_arr[:2]))
        elif model in ["DoubleIntegrator2D", "DoubleIntegrator2D_DPCBF"]:
            if X.size >= 4:
                return float(np.linalg.norm(X[2:4]))
        elif model == "Unicycle2D":
            if u_arr.size >= 1:
                return float(abs(u_arr[0]))
        elif model in [
            "DynamicUnicycle2D",
            "DynamicUnicycle2D_C3BF",
            "DynamicUnicycle2D_DPCBF",
            "KinematicBicycle2D",
            "KinematicBicycle2D_C3BF",
            "KinematicBicycle2D_DPCBF",
            "KinematicBicycle2D_CBFVO",
            "KinematicBicycle2D_OVVO",
            "KinematicBicycle2D_CVAR",
        ]:
            if X.size >= 4:
                return float(abs(X[3]))
            if u_arr.size >= 1:
                return float(abs(u_arr[0]))
        elif model in ["Quad2D", "VTOL2D"]:
            if X.size >= 5:
                return float(np.linalg.norm(X[3:5]))
        elif model == "Quad3D":
            if X.size >= 9:
                return float(np.linalg.norm(X[6:9]))

        if hasattr(self.robot, "get_velocity"):
            try:
                vel = np.asarray(self.robot.get_velocity(), dtype=float).reshape(-1)
                if vel.size == 1:
                    return float(abs(vel[0]))
                if vel.size > 1:
                    return float(np.linalg.norm(vel))
            except Exception:
                pass

        if u_arr.size > 0:
            return float(np.linalg.norm(u_arr))
        return 0.0

    def _trajectory_speed_limit(self, segment_speeds):
        vmax = self.plot_robot_trajectory_speed_max
        if vmax is not None:
            try:
                return max(float(vmax), 1e-6)
            except (TypeError, ValueError):
                pass

        spec_vmax = self.robot_spec.get("v_max", None)
        if spec_vmax is not None:
            try:
                return max(float(spec_vmax), 1e-6)
            except (TypeError, ValueError):
                pass

        if segment_speeds.size > 0:
            return max(float(np.max(segment_speeds)), 1e-6)
        return 1.0

    def _get_trajectory_cmap(self):
        cmap_name = str(self.plot_robot_trajectory_cmap)
        if self._trajectory_cmap_obj is not None and getattr(
            self._trajectory_cmap_obj, "name", None
        ) == cmap_name:
            return self._trajectory_cmap_obj

        if cmap_name in {"trajectory_red_pink", "red_pink", "pink_red"}:
            self._trajectory_cmap_obj = LinearSegmentedColormap.from_list(
                "trajectory_red_pink",
                [
                    (0.00, "#ffd6e5"),
                    (0.35, "#ff9fbc"),
                    (0.70, "#ff4f7a"),
                    (1.00, "#ff0000"),
                ],
            )
        elif cmap_name in {"trajectory_teal_cyan", "teal_cyan", "cyan_teal"}:
            self._trajectory_cmap_obj = LinearSegmentedColormap.from_list(
                "trajectory_teal_cyan",
                [
                    (0.00, "#dcfbff"),
                    (0.35, "#96edf7"),
                    (0.70, "#24b7c9"),
                    (1.00, "#006d77"),
                ],
            )
        elif cmap_name in {"trajectory_yellow", "yellow_trajectory", "pure_yellow"}:
            self._trajectory_cmap_obj = LinearSegmentedColormap.from_list(
                "trajectory_yellow",
                [
                    (0.00, "#fffde7"),
                    (0.35, "#fff59d"),
                    (0.70, "#ffeb3b"),
                    (1.00, "#ffd400"),
                ],
            )
        else:
            self._trajectory_cmap_obj = plt.colormaps.get_cmap(cmap_name)
        return self._trajectory_cmap_obj

    def _record_trajectory_sample(self):
        if not self.plot_robot_trajectory or not (self.show_animation or self.save_animation):
            return

        pos = np.asarray(self.robot.get_position(), dtype=float).reshape(-1)
        if pos.size < 2:
            return

        point = pos[:2].copy()
        speed = self._compute_robot_speed_scalar()

        if self.robot_trajectory_points and np.allclose(
            self.robot_trajectory_points[-1], point, atol=1e-9, rtol=0.0
        ):
            self.robot_trajectory_speeds[-1] = speed
        else:
            self.robot_trajectory_points.append(point)
            self.robot_trajectory_speeds.append(speed)
        self._trajectory_cache_dirty = True

    def _refresh_trajectory_cache(self):
        if not self._trajectory_cache_dirty:
            return

        self._trajectory_cache_dirty = False
        self._trajectory_cached_segments = np.empty((0, 2, 2), dtype=float)
        self._trajectory_cached_segment_speeds = np.empty((0,), dtype=float)
        self._trajectory_cached_vmax = 1.0

        if len(self.robot_trajectory_points) < 2:
            return

        points = np.asarray(self.robot_trajectory_points, dtype=float)
        speeds = np.asarray(self.robot_trajectory_speeds, dtype=float)
        self._trajectory_cached_segments = np.stack([points[:-1], points[1:]], axis=1)
        self._trajectory_cached_segment_speeds = 0.5 * (speeds[:-1] + speeds[1:])
        self._trajectory_cached_vmax = self._trajectory_speed_limit(
            self._trajectory_cached_segment_speeds
        )

    def _update_trajectory_artist(self, ax, attr_name, linewidth=None, zorder=None):
        artist = getattr(self, attr_name, None)
        if not self.plot_robot_trajectory:
            if artist is not None:
                artist.set_visible(False)
            return

        self._refresh_trajectory_cache()
        if self._trajectory_cached_segments.shape[0] == 0:
            if artist is not None:
                artist.set_visible(False)
            return

        if linewidth is None:
            linewidth = self.plot_robot_trajectory_linewidth
        if zorder is None:
            zorder = self.plot_robot_trajectory_zorder

        norm = Normalize(vmin=0.0, vmax=self._trajectory_cached_vmax)
        cmap = self._get_trajectory_cmap()

        if artist is None:
            artist = LineCollection(
                self._trajectory_cached_segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidth,
                alpha=self.plot_robot_trajectory_alpha,
                zorder=zorder,
                capstyle='round',
                joinstyle='round',
            )
            ax.add_collection(artist)
            setattr(self, attr_name, artist)
        else:
            artist.set_segments(self._trajectory_cached_segments)
            artist.set_cmap(cmap)
            artist.set_norm(norm)
            artist.set_linewidth(linewidth)
            artist.set_alpha(self.plot_robot_trajectory_alpha)
            artist.set_zorder(zorder)
            artist.set_capstyle('round')
            artist.set_joinstyle('round')

        artist.set_array(self._trajectory_cached_segment_speeds)
        artist.set_visible(True)

    def _setup_tracking_view(self):
        ax = self.tracking_view_ax
        ax.set_title("Tracking View")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect('equal', adjustable='box')

        self.tracking_view_waypoints_scatter = ax.scatter(
            [], [],
            s=10, facecolors='g', edgecolors='g',
            alpha=0.5, zorder=2
        )
        colors = plt.colormaps.get_cmap('Pastel1').colors
        robot_id = int(self.robot_spec.get('robot_id', 0))
        body_color = colors[robot_id % len(colors) + 1]
        self.tracking_view_robot_patch = ax.add_patch(
            patches.Circle(
                (0.0, 0.0), self.robot.robot_radius,
                edgecolor='black', facecolor=body_color,
                fill=True, linewidth=1.0, zorder=7
            )
        )
        (self.tracking_view_heading_line,) = ax.plot(
            [], [],
            color='r', linewidth=2.0, zorder=8
        )
        (self.tracking_view_fov_line,) = ax.plot([], [], 'k--', zorder=5)
        self.tracking_view_fov_fill = ax.fill([], [], 'k', alpha=0.1, zorder=4)[0]
        (self.tracking_view_backup_rollout_line,) = ax.plot(
            [], [],
            linestyle='-', color='tab:green',
            linewidth=2.4, alpha=1.0, zorder=20
        )
        self.tracking_view_backup_rollout_line.set_path_effects([
            path_effects.Stroke(linewidth=4.2, foreground='white'),
            path_effects.Normal(),
        ])
        self.tracking_view_backup_rollout_line.set_visible(False)
        self.tracking_view_backup_rollout_end_marker = ax.scatter(
            [], [],
            s=26, facecolors='tab:green', edgecolors='white',
            linewidths=0.8, alpha=1.0, zorder=21
        )
        self._update_tracking_view()

    def _sync_tracking_view_waypoints(self):
        if not self.tracking_view_enabled:
            return
        if getattr(self, "waypoints", None) is None or len(self.waypoints) == 0:
            self.tracking_view_waypoints_scatter.set_offsets(np.empty((0, 2)))
            return
        pts = np.asarray(self.waypoints[:, :2], dtype=float)
        self.tracking_view_waypoints_scatter.set_offsets(pts)

    def _ensure_tracking_view_obs_artists(self, n_obs):
        ax = self.tracking_view_ax
        while len(self.tracking_view_obs_patches) < n_obs:
            patch = patches.Circle(
                (0.0, 0.0), 0.0,
                edgecolor='black', facecolor=self.plot_dyn_obs_visible_color,
                alpha=0.85, linewidth=1.0, zorder=4
            )
            ax.add_patch(patch)
            self.tracking_view_obs_patches.append(patch)

            arrow = patches.FancyArrowPatch(
                (0.0, 0.0), (0.0, 0.0),
                arrowstyle='-|>',
                mutation_scale=self.plot_dyn_obs_arrow_head,
                color=self.plot_dyn_obs_arrow_color,
                linewidth=1.0,
                zorder=5,
            )
            arrow.set_visible(False)
            ax.add_patch(arrow)
            self.tracking_view_obs_arrows.append(arrow)

        while len(self.tracking_view_obs_patches) > n_obs:
            patch = self.tracking_view_obs_patches.pop()
            patch.remove()
        while len(self.tracking_view_obs_arrows) > n_obs:
            arrow = self.tracking_view_obs_arrows.pop()
            arrow.remove()

    def _ensure_tracking_view_occ_artists(self, n_occ):
        ax = self.tracking_view_ax
        while len(self.tracking_view_occ_patches) < n_occ:
            patch = patches.Polygon(
                np.zeros((3, 2)),
                closed=True,
                fill=True,
                facecolor='gray',
                edgecolor='none',
                alpha=0.25,
                zorder=1.5,
            )
            patch.set_visible(False)
            ax.add_patch(patch)
            self.tracking_view_occ_patches.append(patch)

        while len(self.tracking_view_occ_patches) > n_occ:
            patch = self.tracking_view_occ_patches.pop()
            patch.remove()

    def _update_tracking_view_occlusion_polygons(self, center_xy, half_window):
        if not self.tracking_view_enabled:
            return
        # For fair visualization across baselines, always build plotting
        # occlusion geometry from the shared visibility/occlusion pipeline
        # first. Some MPC baselines intentionally keep only a truncated
        # nearest-scenario subset for optimization, which should not change
        # the displayed occlusion geometry.
        scenarios = self._get_occlusion_scenarios_for_plot(self.obs)
        if not scenarios:
            pos_controller = getattr(self, "pos_controller", None)
            scenarios = getattr(pos_controller, "occlusion_scenarios", None) if pos_controller is not None else None
        if not scenarios:
            for patch in self.tracking_view_occ_patches:
                patch.set_visible(False)
            return

        xmin = float(center_xy[0] - half_window)
        xmax = float(center_xy[0] + half_window)
        ymin = float(center_xy[1] - half_window)
        ymax = float(center_xy[1] + half_window)

        visible_polys = []
        for sc in scenarios:
            poly = sc.get('visibility_poly', sc.get('poly', None))
            if poly is None:
                continue
            pts = np.asarray(poly, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue
            pxmin = float(np.min(pts[:, 0]))
            pxmax = float(np.max(pts[:, 0]))
            pymin = float(np.min(pts[:, 1]))
            pymax = float(np.max(pts[:, 1]))
            if pxmax < xmin or pxmin > xmax or pymax < ymin or pymin > ymax:
                continue
            visible_polys.append(pts)

        self._ensure_tracking_view_occ_artists(len(visible_polys))
        for i, pts in enumerate(visible_polys):
            patch = self.tracking_view_occ_patches[i]
            patch.set_xy(pts)
            patch.set_visible(True)
        for j in range(len(visible_polys), len(self.tracking_view_occ_patches)):
            self.tracking_view_occ_patches[j].set_visible(False)

    def _update_tracking_view(self):
        if not self.tracking_view_enabled:
            return

        ax = self.tracking_view_ax
        pos = np.asarray(self.robot.get_position(), dtype=float).reshape(-1)
        theta = float(self.robot.get_orientation())
        half = 0.5 * float(self.tracking_view_window_size)

        ax.set_xlim(float(pos[0] - half), float(pos[0] + half))
        ax.set_ylim(float(pos[1] - half), float(pos[1] + half))
        self._update_tracking_view_occlusion_polygons(pos, half)
        self._update_trajectory_artist(
            ax,
            "tracking_view_trajectory_collection",
            linewidth=0.9 * self.plot_robot_trajectory_linewidth,
            zorder=max(2.5, self.plot_robot_trajectory_zorder),
        )

        self.tracking_view_robot_patch.center = (float(pos[0]), float(pos[1]))
        heading_len = float(getattr(self.robot, "vis_orient_len", 0.5))
        hx = float(pos[0] + heading_len * np.cos(theta))
        hy = float(pos[1] + heading_len * np.sin(theta))
        self.tracking_view_heading_line.set_data([float(pos[0]), hx], [float(pos[1]), hy])

        if 'sensor' in self.robot_spec and self.robot_spec['sensor'] == 'rgbd':
            fov_left, fov_right = self.robot.calculate_fov_points()
            fov_x_points = [float(pos[0]), float(fov_left[0]), float(fov_right[0]), float(pos[0])]
            fov_y_points = [float(pos[1]), float(fov_left[1]), float(fov_right[1]), float(pos[1])]
            self.tracking_view_fov_line.set_data(fov_x_points, fov_y_points)
            self.tracking_view_fov_fill.set_xy(np.array([fov_x_points, fov_y_points]).T)
            self.tracking_view_fov_line.set_visible(True)
            self.tracking_view_fov_fill.set_visible(True)
        else:
            self.tracking_view_fov_line.set_visible(False)
            self.tracking_view_fov_fill.set_visible(False)

        if self.obs is None:
            obs_arr = np.empty((0, 8), dtype=float)
        else:
            obs_arr = np.asarray(self.obs, dtype=float)
        if obs_arr.size == 0:
            obs_arr = np.empty((0, 8), dtype=float)
        elif obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        occluded_mask = self._cached_occluded_mask
        if occluded_mask is None or len(occluded_mask) != obs_arr.shape[0]:
            occluded_mask = None

        if obs_arr.size:
            margin = max(
                0.75,
                float(np.max(obs_arr[:, 2])) + self.plot_dyn_obs_arrow_len_max
            )
            in_view_mask = (
                (obs_arr[:, 0] >= float(pos[0] - half - margin)) &
                (obs_arr[:, 0] <= float(pos[0] + half + margin)) &
                (obs_arr[:, 1] >= float(pos[1] - half - margin)) &
                (obs_arr[:, 1] <= float(pos[1] + half + margin))
            )
            obs_plot = obs_arr[in_view_mask]
            occ_plot = occluded_mask[in_view_mask] if occluded_mask is not None else None
        else:
            obs_plot = obs_arr
            occ_plot = None

        self._ensure_tracking_view_obs_artists(int(obs_plot.shape[0]))

        speeds = np.hypot(obs_plot[:, 3], obs_plot[:, 4]) if obs_plot.size else np.array([])
        v_ref = float(np.max(speeds)) if speeds.size else 1.0
        if v_ref < 1e-9:
            v_ref = 1.0

        for i, obs_info in enumerate(obs_plot):
            ox, oy, r = float(obs_info[0]), float(obs_info[1]), float(obs_info[2])
            patch = self.tracking_view_obs_patches[i]
            patch.center = (ox, oy)
            patch.set_radius(r)
            patch.set_visible(True)
            is_occ = bool(occ_plot[i]) if occ_plot is not None else False
            patch.set_facecolor(self.plot_dyn_obs_hidden_color if is_occ else self.plot_dyn_obs_visible_color)

            arrow = self.tracking_view_obs_arrows[i]
            if not self.plot_dyn_obs_arrows or obs_plot.shape[1] < 5:
                arrow.set_visible(False)
                continue

            vx, vy = float(obs_info[3]), float(obs_info[4])
            speed = float(np.hypot(vx, vy))
            if speed < 1e-9:
                arrow.set_visible(False)
                continue

            ux = vx / speed
            uy = vy / speed
            t = min(1.0, speed / v_ref)
            length = self.plot_dyn_obs_arrow_len_min + (
                self.plot_dyn_obs_arrow_len_max - self.plot_dyn_obs_arrow_len_min
            ) * np.sqrt(t)
            arrow.set_positions((ox, oy), (ox + ux * length, oy + uy * length))
            arrow.set_mutation_scale(self.plot_dyn_obs_arrow_head)
            arrow.set_visible(True)

    def set_waypoints(self, waypoints):
        super().set_waypoints(waypoints)
        self._sync_tracking_view_waypoints()

    def is_collide_unknown(self):
        robot_radius = self.robot.robot_radius
        px, py = float(self.robot.X[0, 0]), float(self.robot.X[1, 0])
        use_rect = bool(self.robot_spec.get('dynamic_obs_rect_collision', False))

        if self.unknown_obs is not None:
            for obs in self.unknown_obs:
                distance = np.linalg.norm(self.robot.X[:2, 0] - obs[:2])
                if distance < (obs[2] + robot_radius):
                    print("Collision with unknown obstacle detected!")
                    return True

        if self.obs is not None:
            for obs in self.obs:
                obs_type = int(obs[7]) if len(obs) >= 8 else None
                if use_rect and obs_type == 2:
                    ox, oy, r = float(obs[0]), float(obs[1]), float(obs[2])
                    half_len = 2.0 * r
                    half_w = r
                    dx = max(abs(px - ox) - half_len, 0.0)
                    dy = max(abs(py - oy) - half_w, 0.0)
                    if (dx * dx + dy * dy) <= (robot_radius * robot_radius):
                        print(f"Collision with known obstacle detected! Obs: {obs}, Robot: {self.robot.X[:2, 0]} {robot_radius}, RectCollision: True")
                        return True
                    continue

                distance = np.linalg.norm(self.robot.X[:2, 0] - obs[:2])
                if distance < (obs[2] + robot_radius):
                    print(f"Collision with known obstacle detected! Obs: {obs}, Robot: {self.robot.X[:2, 0]} {robot_radius}, Distance: {distance}, {distance < (obs[2] + robot_radius)}")
                    return True

        if self.robot_spec['model'] in ['VTOL2D']:
            if self.robot.X[1, 0] < 0:
                return True
            if np.abs(self.robot.X[2, 0]) > self.robot_spec['pitch_max']:
                return True
        return False

    def setup_robot(self, X0):
        # Load local dynamic_env/robot.py explicitly to avoid name collisions
        import importlib.util
        import os
        robot_path = os.path.join(os.path.dirname(__file__), "robot.py")
        spec = importlib.util.spec_from_file_location("dynamic_env_robot_local", robot_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local robot module at {robot_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BaseRobotDyn_OCC = getattr(mod, "BaseRobotDyn_OCC")
        self.robot = BaseRobotDyn_OCC(
            X0.reshape(-1, 1), self.robot_spec, self.dt, self.ax)

    def set_obs_meta(self, meta_list):
        """
        meta_list: list of length N.
        Each element is a dict, e.g. {'mode':0/1, 'v_max':0.35, 'theta':initial heading (rad)}
        - mode=0: constant velocity (default behavior)
        - mode=1: random walker (heading jitter)
        """
        if not isinstance(meta_list, list):
            raise ValueError("meta_list must be a list")
        if len(meta_list) != len(self.obs):
            raise ValueError("meta_list length must match self.obs rows")
        self.obs_meta = meta_list

    def _ensure_obs_meta(self):
        """Populate constant-velocity defaults when metadata is missing."""
        if self.obs_meta and len(self.obs_meta) == len(self.obs):
            return
        self.obs_meta = []
        for i in range(len(self.obs)):
            vx, vy = float(self.obs[i,3]), float(self.obs[i,4])
            vmag = float(np.hypot(vx, vy))
            theta0 = float(np.arctan2(vy, vx)) if vmag > 1e-9 else float(self._rng.uniform(-np.pi, np.pi))
            self.obs_meta.append({'mode': 0, 'v_max': vmag, 'theta': theta0})

    @staticmethod
    def make_random_obstacles7(
        n_rand,
        v_obs_max,
        x_range,
        y_spawn_range,
        r_range,
        y_bounds,
        seed=42,
        rand_obs=True,
        v_obs_min=0.0,
    ):
        if not rand_obs:
            return np.empty((0, 7), dtype=float) , []
        rng = np.random.default_rng(seed)
        x_min, x_max = x_range
        y_min_spawn, y_max_spawn = y_spawn_range
        r_min, r_max = r_range
        y_min_g, y_max_g = y_bounds
        v_obs_min = float(max(0.0, v_obs_min))
        v_obs_max = float(max(0.0, v_obs_max))

        rows = []
        metas = []
        for _ in range(n_rand):
            x0 = rng.uniform(x_min, x_max)
            y0 = rng.uniform(y_min_spawn, y_max_spawn)
            r  = rng.uniform(r_min, r_max)
            theta0 = rng.uniform(-np.pi, np.pi)
            if np.cos(theta0) >= 0.0:
                theta0 += np.pi
                if theta0 > np.pi:
                    theta0 -= 2*np.pi
            speed0 = v_obs_max
            if v_obs_max > 1e-9 and v_obs_min < v_obs_max:
                speed0 = float(rng.uniform(v_obs_min, v_obs_max))
            vx0, vy0 = speed0*np.cos(theta0), speed0*np.sin(theta0)
            rows.append([x0, y0, r, vx0, vy0, y_min_g, y_max_g])
            metas.append({'mode': 1, 'v_max': speed0, 'theta': theta0})
        return np.array(rows, dtype=float), metas

    def step_dyn_obs(self):
        """
        self.obs: (N,7) = [x, y, r, vx, vy, y_min, y_max]
        self.obs_meta[i] = {'mode':0/1, 'v_max':..., 'theta':...}
        - mode=0: constant velocity
        - mode=1: random agents
        """
        if len(self.obs) == 0:
            return

        if not (isinstance(self.obs, np.ndarray) and self.obs.ndim == 2 and self.obs.shape[1] == 8):
            try:
                self.obs = np.array(self.obs, dtype=float).reshape(-1, 8)
            except ValueError as e:
                print(f"[Error] self.obs size is {np.array(self.obs).size}, cannot reshape to (-1, 8). Check data consistency.")
                raise e

        self._ensure_obs_meta()
        crowd_cfg = self.robot_spec.get("crowd_dyn_obs", {})
        if not isinstance(crowd_cfg, dict):
            crowd_cfg = {}

        base_jitter_std = float(crowd_cfg.get("heading_jitter_std", 0.0))
        large_turn_prob = float(crowd_cfg.get("large_turn_prob", 0.05))
        large_turn_std = float(crowd_cfg.get("large_turn_std", 0.2))

        # Pedestrian-aware near-robot interaction:
        # if a visible obstacle is close to the robot and heading toward it,
        # bias heading to sidestep instead of charging into the robot.
        ped_aware_enable = bool(crowd_cfg.get("ped_aware_enable", False))
        ped_aware_visible_only = bool(crowd_cfg.get("ped_aware_visible_only", True))
        ped_aware_radius = float(crowd_cfg.get("ped_aware_radius", 2.0))
        ped_aware_approach_cos = float(crowd_cfg.get("ped_aware_approach_cos", 0.0))
        ped_aware_turn_gain = float(crowd_cfg.get("ped_aware_turn_gain", 0.45))
        ped_aware_side_weight = float(crowd_cfg.get("ped_aware_side_weight", 0.85))
        ped_aware_away_weight = float(crowd_cfg.get("ped_aware_away_weight", 0.25))
        ped_aware_noise_std = float(crowd_cfg.get("ped_aware_noise_std", 0.07))
        ped_aware_side_flip_prob = float(crowd_cfg.get("ped_aware_side_flip_prob", 0.04))

        # Occlusion-emergence bias:
        # for currently occluded agents near the robot-forward region,
        # occasionally steer toward a path-crossing target to increase natural
        # emergence events without forcing deterministic collisions.
        emergence_enable = bool(crowd_cfg.get("occlusion_emergence_enable", False))
        emergence_occluded_only = bool(crowd_cfg.get("occlusion_emergence_occluded_only", True))
        emergence_zone_radius = float(crowd_cfg.get("occlusion_emergence_zone_radius", 8.0))
        emergence_forward_only = bool(crowd_cfg.get("occlusion_emergence_forward_only", True))
        emergence_forward_min_proj = float(crowd_cfg.get("occlusion_emergence_forward_min_proj", -0.2))
        emergence_turn_prob = float(crowd_cfg.get("occlusion_emergence_turn_prob", 0.07))
        emergence_turn_gain = float(crowd_cfg.get("occlusion_emergence_turn_gain", 0.28))
        emergence_target_ahead = float(crowd_cfg.get("occlusion_emergence_target_ahead", 2.8))
        emergence_lateral_jitter = float(crowd_cfg.get("occlusion_emergence_lateral_jitter", 0.9))
        emergence_noise_std = float(crowd_cfg.get("occlusion_emergence_noise_std", 0.04))
        occluded_speed_boost_enable = bool(crowd_cfg.get("occluded_speed_boost_enable", False))
        occluded_speed_boost_vmax = float(crowd_cfg.get("occluded_speed_boost_vmax", 1.0))
        occluded_speed_boost_exact = bool(crowd_cfg.get("occluded_speed_boost_exact", False))
        if (not np.isfinite(occluded_speed_boost_vmax)) or occluded_speed_boost_vmax <= 1e-9:
            occluded_speed_boost_enable = False
        occluded_speed_boost_fov_only = bool(crowd_cfg.get("occluded_speed_boost_fov_only", True))
        occluded_speed_boost_on_hys = int(crowd_cfg.get("occluded_speed_boost_on_hysteresis_steps", 1))
        hold_time_s = crowd_cfg.get("occluded_speed_boost_hold_time_s", None)
        if hold_time_s is not None:
            try:
                hold_time_s = float(hold_time_s)
            except Exception:
                hold_time_s = None
        if hold_time_s is not None and np.isfinite(hold_time_s) and hold_time_s > 0.0:
            # Keep the max-speed mode active for this long after the obstacle
            # leaves the occluded set. The counter logic below deactivates when
            # off_count reaches this threshold, so add one step to include the
            # requested hold duration.
            occluded_speed_boost_off_hys = int(np.ceil(hold_time_s / max(float(self.dt), 1e-9))) + 1
        else:
            occluded_speed_boost_off_hys = int(crowd_cfg.get("occluded_speed_boost_off_hysteresis_steps", 1))
        occluded_speed_boost_on_hys = max(1, occluded_speed_boost_on_hys)
        occluded_speed_boost_off_hys = max(1, occluded_speed_boost_off_hys)

        robot_xy = np.asarray(self.robot.X[:2, 0], dtype=float).reshape(2,)
        goal_xy = None
        if self.goal is not None:
            goal_xy = np.asarray(self.goal[:2], dtype=float).reshape(2,)
        if goal_xy is not None:
            goal_vec = goal_xy - robot_xy
            goal_dir = self._safe_normalize(goal_vec)
        else:
            goal_dir = None
        if goal_dir is None:
            yaw = float(self.robot.X[2, 0]) if self.robot.X.shape[0] >= 3 else 0.0
            goal_dir = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        goal_lat = np.array([-goal_dir[1], goal_dir[0]], dtype=float)

        use_visibility_logic = bool(
            (ped_aware_enable and ped_aware_visible_only)
            or (emergence_enable and emergence_occluded_only)
            or occluded_speed_boost_enable
        )
        visible_mask = None
        fov_range_mask = None
        if use_visibility_logic:
            vis_refresh_steps = int(crowd_cfg.get("visibility_refresh_steps", 1))
            if vis_refresh_steps < 1:
                vis_refresh_steps = 1
            if (
                self._dyn_behavior_vis_mask_cache is None
                or self._dyn_behavior_step_counter - self._dyn_behavior_vis_cache_step >= vis_refresh_steps
                or len(self._dyn_behavior_vis_mask_cache) != int(self.obs.shape[0])
            ):
                n_obs = int(self.obs.shape[0])
                visible_mask = np.ones(n_obs, dtype=bool)
                fov_range_mask = np.zeros(n_obs, dtype=bool)
                obs_xy = np.asarray(self.obs[:, :2], dtype=float)
                rel = obs_xy - robot_xy[None, :]
                dist = np.linalg.norm(rel, axis=1)
                mode_arr = np.array([int(m.get("mode", 0)) for m in self.obs_meta], dtype=int)
                dyn_mask = mode_arr == 1
                candidate_mask = np.zeros(n_obs, dtype=bool)

                # Visibility is needed only where behavior logic consumes it.
                if ped_aware_enable and ped_aware_visible_only:
                    candidate_mask |= dyn_mask & (dist <= ped_aware_radius)

                if emergence_enable and emergence_occluded_only:
                    in_front = np.ones(n_obs, dtype=bool)
                    if emergence_forward_only:
                        proj = rel @ goal_dir
                        in_front = proj >= emergence_forward_min_proj
                    candidate_mask |= dyn_mask & (dist <= emergence_zone_radius) & in_front

                if occluded_speed_boost_enable:
                    for j in range(n_obs):
                        try:
                            if occluded_speed_boost_fov_only:
                                fov_range_mask[j] = bool(self.robot.is_in_fov(obs_xy[j], is_in_cam_range=True))
                            else:
                                fov_range_mask[j] = dist[j] <= float(self.robot.cam_range)
                        except Exception:
                            fov_range_mask[j] = dist[j] <= float(self.robot.robot_spec.get("sensing_range", np.inf))
                    candidate_mask |= dyn_mask & fov_range_mask

                candidate_idx = np.nonzero(candidate_mask)[0]
                if candidate_idx.size > 0:
                    occ_filter_fn = self._resolve_occ_filter_fn()
                    if occ_filter_fn is not None:
                        # For candidate visibility, only obstacles up to the farthest
                        # candidate distance can act as occluders (distance-ordered logic).
                        d_max = float(np.max(dist[candidate_idx]))
                        support_idx = np.nonzero(dist <= (d_max + 1e-9))[0]
                        try:
                            _, _, vis_local_idx = occ_filter_fn(
                                self.robot.X,
                                self.obs[support_idx],
                                return_indices=True,
                            )
                            vis_local_idx = np.asarray(vis_local_idx, dtype=int).reshape(-1)
                            vis_local_idx = vis_local_idx[
                                (vis_local_idx >= 0) & (vis_local_idx < support_idx.shape[0])
                            ]
                            support_visible = np.zeros(support_idx.shape[0], dtype=bool)
                            if vis_local_idx.size > 0:
                                support_visible[vis_local_idx] = True
                            support_visible_global = np.zeros(n_obs, dtype=bool)
                            support_visible_global[support_idx] = support_visible
                            visible_mask[candidate_idx] = support_visible_global[candidate_idx]
                        except Exception:
                            # Keep conservative fallback behavior-neutral for dynamic logic.
                            pass

                self._dyn_behavior_vis_mask_cache = visible_mask
                self._dyn_behavior_fov_mask_cache = fov_range_mask
                self._dyn_behavior_vis_cache_step = self._dyn_behavior_step_counter
            else:
                visible_mask = self._dyn_behavior_vis_mask_cache
                fov_range_mask = self._dyn_behavior_fov_mask_cache

        for i in range(self.obs.shape[0]):
            x, y, r, vx, vy, y_min, y_max = self.obs[i, :7]
            meta = self.obs_meta[i]
            mode   = int(meta.get('mode', 0))
            v_max  = float(meta.get('v_max', np.hypot(vx, vy)))
            forced_occluder = bool(meta.get("forced_occluder", False))
            base_v_max = float(meta.get("base_v_max", v_max))
            meta["base_v_max"] = float(base_v_max)
            if 'theta' not in meta:
                meta['theta'] = np.arctan2(vy, vx) if v_max > 1e-9 else 0.0
            theta  = float(meta['theta'])
            is_visible = True if visible_mask is None else bool(visible_mask[i])
            in_fov_range = True if fov_range_mask is None else bool(fov_range_mask[i])
            speed_boost_desired = bool(
                (mode == 1)
                and (not forced_occluder)
                and occluded_speed_boost_enable
                and (base_v_max > 1e-9)
                and (not is_visible)
                and ((not occluded_speed_boost_fov_only) or in_fov_range)
            )
            prev_speed_boost_active = bool(meta.get("occluded_speed_boost_latched", False))
            speed_boost_on_count = int(meta.get("occluded_speed_boost_on_count", 0) or 0)
            speed_boost_off_count = int(meta.get("occluded_speed_boost_off_count", 0) or 0)
            if speed_boost_desired:
                speed_boost_on_count += 1
                speed_boost_off_count = 0
                if prev_speed_boost_active or speed_boost_on_count >= occluded_speed_boost_on_hys:
                    speed_boost_active = True
                else:
                    speed_boost_active = False
            else:
                speed_boost_off_count += 1
                speed_boost_on_count = 0
                if prev_speed_boost_active and speed_boost_off_count < occluded_speed_boost_off_hys:
                    speed_boost_active = True
                else:
                    speed_boost_active = False
            if speed_boost_active:
                if occluded_speed_boost_exact:
                    eff_v_max = float(occluded_speed_boost_vmax)
                else:
                    eff_v_max = float(max(base_v_max, occluded_speed_boost_vmax))
            else:
                eff_v_max = float(base_v_max)
            meta["occluded_speed_boost_latched"] = bool(speed_boost_active)
            meta["occluded_speed_boost_on_count"] = int(speed_boost_on_count)
            meta["occluded_speed_boost_off_count"] = int(speed_boost_off_count)
            meta["occluded_speed_boost_desired"] = bool(speed_boost_desired)
            meta["occluded_speed_boost_active"] = bool(speed_boost_active)
            meta["occluded_speed_boost_exact"] = bool(occluded_speed_boost_exact)
            meta["effective_v_max"] = float(eff_v_max)

            if mode == 1:
                # --- Random walker: heading noise + occasional large turn ---
                jitter_std = float(meta.get("heading_jitter_std", base_jitter_std))
                turn_prob = float(meta.get("large_turn_prob", large_turn_prob))
                turn_std = float(meta.get("large_turn_std", large_turn_std))
                dtheta = self._rng.normal(0.0, jitter_std)
                if self._rng.random() < turn_prob:
                    dtheta += self._rng.normal(0.0, turn_std)
                theta += dtheta

                # Occluded-agent emergence bias (kept stochastic).
                if (not forced_occluder) and emergence_enable and base_v_max > 1e-9:
                    rel_from_robot = np.array([x - robot_xy[0], y - robot_xy[1]], dtype=float)
                    dist_from_robot = float(np.linalg.norm(rel_from_robot))
                    in_front = float(np.dot(rel_from_robot, goal_dir)) >= emergence_forward_min_proj
                    trigger = (dist_from_robot <= emergence_zone_radius) and (in_front or (not emergence_forward_only))
                    if emergence_occluded_only:
                        trigger = trigger and (not is_visible)
                    if trigger and self._rng.random() < emergence_turn_prob:
                        ahead = emergence_target_ahead + float(self._rng.normal(0.0, 0.5))
                        lat = float(self._rng.normal(0.0, emergence_lateral_jitter))
                        target = robot_xy + ahead * goal_dir + lat * goal_lat
                        to_target = target - np.array([x, y], dtype=float)
                        dir_target = self._safe_normalize(to_target)
                        if dir_target is not None:
                            theta_target = float(np.arctan2(dir_target[1], dir_target[0]))
                            d = self._wrap_angle(theta_target - theta)
                            theta += emergence_turn_gain * d + float(self._rng.normal(0.0, emergence_noise_std))

                # Near-robot pedestrian-aware sidestep/avoidance.
                if (not forced_occluder) and ped_aware_enable and base_v_max > 1e-9:
                    if (not ped_aware_visible_only) or is_visible:
                        rel_to_robot = robot_xy - np.array([x, y], dtype=float)
                        dist = float(np.linalg.norm(rel_to_robot))
                        if dist <= ped_aware_radius and dist > 1e-9:
                            u_to_robot = rel_to_robot / dist
                            u_heading = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                            approach_cos = float(np.dot(u_heading, u_to_robot))
                            if approach_cos > ped_aware_approach_cos:
                                side_sign = float(meta.get("pass_side_sign", 0.0))
                                if side_sign == 0.0:
                                    side_sign = 1.0 if self._rng.random() < 0.5 else -1.0
                                if self._rng.random() < ped_aware_side_flip_prob:
                                    side_sign = -side_sign
                                meta["pass_side_sign"] = float(side_sign)

                                side_vec = side_sign * np.array([-u_to_robot[1], u_to_robot[0]], dtype=float)
                                avoid_vec = (
                                    ped_aware_side_weight * side_vec
                                    - ped_aware_away_weight * u_to_robot
                                    + 0.2 * u_heading
                                )
                                dir_avoid = self._safe_normalize(avoid_vec)
                                if dir_avoid is not None:
                                    theta_avoid = float(np.arctan2(dir_avoid[1], dir_avoid[0]))
                                    closeness = float(np.clip((ped_aware_radius - dist) / max(ped_aware_radius, 1e-6), 0.0, 1.0))
                                    approach_scale = float(
                                        np.clip(
                                            (approach_cos - ped_aware_approach_cos)
                                            / max(1.0 - ped_aware_approach_cos, 1e-6),
                                            0.0,
                                            1.0,
                                        )
                                    )
                                    alpha = ped_aware_turn_gain * (0.35 + 0.65 * closeness) * approach_scale
                                    d = self._wrap_angle(theta_avoid - theta)
                                    theta += alpha * d + float(self._rng.normal(0.0, ped_aware_noise_std * closeness))

                if forced_occluder and base_v_max > 1e-9:
                    sep_margin = float(meta.get("forced_occluder_sep_margin", 0.35))
                    repel = np.zeros(2, dtype=float)
                    p_i = np.array([x, y], dtype=float)
                    for j in range(self.obs.shape[0]):
                        if j == i:
                            continue
                        meta_j = self.obs_meta[j] if j < len(self.obs_meta) else {}
                        if not bool(meta_j.get("forced_occluder", False)):
                            continue
                        p_j = np.asarray(self.obs[j, :2], dtype=float)
                        diff = p_i - p_j
                        dist = float(np.linalg.norm(diff))
                        desired = float(r + self.obs[j, 2] + sep_margin)
                        if dist <= 1e-6:
                            ang = float(self._rng.uniform(-np.pi, np.pi))
                            repel += np.array([np.cos(ang), np.sin(ang)], dtype=float)
                        elif dist < desired:
                            repel += ((desired - dist) / max(desired, 1e-6)) * (diff / dist)
                    repel_dir = self._safe_normalize(repel)
                    if repel_dir is not None:
                        theta_repel = float(np.arctan2(repel_dir[1], repel_dir[0]))
                        theta += 0.45 * self._wrap_angle(theta_repel - theta)
                    drift_x = meta.get("forced_drift_dir_x", None)
                    drift_y = meta.get("forced_drift_dir_y", None)
                    if drift_x is not None and drift_y is not None:
                        drift_dir = self._safe_normalize(np.array([float(drift_x), float(drift_y)], dtype=float))
                        if drift_dir is not None:
                            theta_drift = float(np.arctan2(drift_dir[1], drift_dir[0]))
                            drift_gain = float(meta.get("forced_drift_gain", 0.25))
                            theta += drift_gain * self._wrap_angle(theta_drift - theta)

                theta = self._wrap_angle(theta)
                vx, vy = eff_v_max * np.cos(theta), eff_v_max * np.sin(theta)
                meta['theta'] = theta   # store latest heading in meta

            # Update position
            x_new = x + vx * self.dt
            y_new = y + vy * self.dt

            x_min = meta.get("x_min", None)
            x_max = meta.get("x_max", None)
            if x_min is not None and x_max is not None:
                x_min = float(x_min)
                x_max = float(x_max)
                if x_new >= x_max:
                    x_new = x_max
                    vx = -abs(vx)
                    if mode == 1:
                        meta['theta'] = self._wrap_angle(np.pi - float(meta['theta']))
                        vx, vy = eff_v_max*np.cos(meta['theta']), eff_v_max*np.sin(meta['theta'])
                elif x_new <= x_min:
                    x_new = x_min
                    vx = abs(vx)
                    if mode == 1:
                        meta['theta'] = self._wrap_angle(np.pi - float(meta['theta']))
                        vx, vy = eff_v_max*np.cos(meta['theta']), eff_v_max*np.sin(meta['theta'])

            # Reflect at y bounds
            if y_new >= y_max:
                y_new = y_max
                vy = -abs(vy)
                if mode == 1:
                    meta['theta'] = self._wrap_angle(-float(meta['theta']))  # mirror across x-axis
                    vx, vy = eff_v_max*np.cos(meta['theta']), eff_v_max*np.sin(meta['theta'])
            elif y_new <= y_min:
                y_new = y_min
                vy =  abs(vy)
                if mode == 1:
                    meta['theta'] = self._wrap_angle(-float(meta['theta']))
                    vx, vy = eff_v_max*np.cos(meta['theta']), eff_v_max*np.sin(meta['theta'])

            # Write back
            self.obs[i, 0] = x_new
            self.obs[i, 1] = y_new
            self.obs[i, 3] = vx
            self.obs[i, 4] = vy
        self._dyn_behavior_step_counter += 1

    def render_dyn_obs(self):
        if not self.plot_dyn_obs:
            if self.dyn_obs_patch is not None:
                for patch in self.dyn_obs_patch:
                    patch.set_visible(False)
                for arrow in self.obs_vel_arrows:
                    arrow.set_visible(False)
                return

        if len(self.obs_vel_arrows) != len(self.obs):
            for arrow in self.obs_vel_arrows:
                arrow.remove()
            self.obs_vel_arrows = []

            if self.plot_dyn_obs_arrows and self.obs.shape[0] > 0 and self.obs.shape[1] >= 5:
                for _ in range(len(self.obs)):
                    arrow = patches.FancyArrowPatch(
                        (0, 0), (0, 0),
                        arrowstyle='-|>',
                        mutation_scale=self.plot_dyn_obs_arrow_head,
                        color=self.plot_dyn_obs_arrow_color,
                        linewidth=1.0,
                        zorder=5
                    )
                    arrow.set_visible(False)
                    self.ax.add_patch(arrow)
                    self.obs_vel_arrows.append(arrow)

        occluded_mask = None
        if self.plot_dyn_obs_occlusion:
            occluded_mask = self._cached_occluded_mask
            if (self._occ_mask_counter % self.occlusion_mask_every) == 0 or occluded_mask is None:
                occluded_mask = self._get_occluded_obs_mask()
                self._cached_occluded_mask = occluded_mask
            self._occ_mask_counter += 1
        if occluded_mask is None:
            self._last_occluded_mask = None
        elif self._last_occluded_mask is None or len(self._last_occluded_mask) != len(occluded_mask):
            self._last_occluded_mask = np.full(len(occluded_mask), -1, dtype=np.int8)

        speeds = np.hypot(self.obs[:, 3], self.obs[:, 4]) if len(self.obs) else np.array([])
        v_ref = float(np.max(speeds)) if speeds.size else 1.0
        if v_ref < 1e-9:
            v_ref = 1.0

        for i, obs_info in enumerate(self.obs):
            # obs: [x, y, r, vx, vy]
            ox, oy, r = obs_info[:3]
            patch = self.dyn_obs_patch[i]
            patch.set_visible(True)
            patch.center = ox, oy
            patch.set_radius(r)
            if occluded_mask is not None:
                is_occ = int(bool(occluded_mask[i]))
                if self._last_occluded_mask is None or self._last_occluded_mask[i] != is_occ:
                    patch.set_facecolor(self.plot_dyn_obs_hidden_color if is_occ else self.plot_dyn_obs_visible_color)
                    if self._last_occluded_mask is not None:
                        self._last_occluded_mask[i] = is_occ
            else:
                patch.set_facecolor(self.plot_dyn_obs_visible_color)

            # Check if there are arrows to update
            if self.plot_dyn_obs_arrows and i < len(self.obs_vel_arrows):
                vx, vy = obs_info[3], obs_info[4]

                arrow = self.obs_vel_arrows[i]
                speed = float(np.hypot(vx, vy))
                if speed < 1e-9:
                    arrow.set_visible(False)
                    continue

                ux = float(vx / speed)
                uy = float(vy / speed)
                min_len = self.plot_dyn_obs_arrow_len_min
                max_len = self.plot_dyn_obs_arrow_len_max
                t = min(1.0, speed / v_ref)
                length = min_len + (max_len - min_len) * np.sqrt(t)
                dx = ux * length
                dy = uy * length
                arrow.set_mutation_scale(self.plot_dyn_obs_arrow_head)
                if self.plot_dyn_obs_adaptive_arrow:
                    p0 = self.ax.transData.transform((ox, oy))
                    p1 = self.ax.transData.transform((ox + dx, oy + dy))
                    pix_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                    dpi = float(self.fig.dpi)
                    head_scale = max(2.0, min(8.0, pix_len * 72.0 / dpi * 0.25))
                    arrow.set_mutation_scale(head_scale)
                arrow.set_positions((ox, oy), (ox + dx, oy + dy))
                arrow.set_visible(True)

    def _resolve_occ_filter_fn(self):
        if not hasattr(self, "pos_controller") or self.pos_controller is None:
            return None
        occ_filter_fn = None
        if hasattr(self.pos_controller, "_filter_visible_and_build_occ"):
            occ_filter_fn = self.pos_controller._filter_visible_and_build_occ
        elif hasattr(self.pos_controller, "_occ_utils") and \
                hasattr(self.pos_controller._occ_utils, "_filter_visible_and_build_occ"):
            occ_filter_fn = self.pos_controller._occ_utils._filter_visible_and_build_occ
        elif bool(self.robot_spec.get("plot_dyn_obs_occlusion_fallback", True)):
            # Keep occlusion-aware visibility filter for non-occlusion controllers.
            try:
                from utils.occlusion import OcclusionUtils
                vis_utils = getattr(self, "_plot_occ_utils", None)
                if vis_utils is None:
                    sensing_range = self.robot_spec.get('sensing_range', 10.0)
                    vis_utils = OcclusionUtils(
                        robot=self.robot,
                        robot_spec=self.robot_spec,
                        sensing_range=float(sensing_range),
                        barrier_fn=None,
                    )
                    self._plot_occ_utils = vis_utils
                occ_filter_fn = vis_utils._filter_visible_and_build_occ
            except Exception:
                occ_filter_fn = None
        return occ_filter_fn

    def _get_occluded_obs_mask(self):
        occ_filter_fn = self._resolve_occ_filter_fn()
        if occ_filter_fn is None:
            return None
        occ_types = self.robot_spec.get('occlusion_types', None)
        if occ_types is not None and not occ_types:
            return None
        if not isinstance(self.obs, np.ndarray) or self.obs.ndim != 2 or self.obs.shape[0] == 0:
            return None

        sensing_range = getattr(self.pos_controller, "sensing_range", None)
        if sensing_range is None:
            sensing_range = self.robot_spec.get('sensing_range', None)
        if sensing_range is None:
            return None

        obs_arr = np.asarray(self.obs, dtype=float)
        visible_mask = np.zeros(obs_arr.shape[0], dtype=bool)

        # Fast path: request visible indices directly from occlusion filter.
        try:
            _, _, vis_idx = occ_filter_fn(self.robot.X, self.obs, return_indices=True)
            if vis_idx is None:
                vis_idx = []
            vis_idx = np.asarray(vis_idx, dtype=int).reshape(-1)
            if vis_idx.size > 0:
                vis_idx = vis_idx[(vis_idx >= 0) & (vis_idx < visible_mask.shape[0])]
                visible_mask[vis_idx] = True
        except TypeError:
            # Backward-compatible path for filters that do not support return_indices.
            try:
                visible_obs, _ = occ_filter_fn(self.robot.X, self.obs)
            except Exception:
                return None

            row_to_indices = {}
            for i, row in enumerate(obs_arr):
                key = tuple(row.tolist())
                if key in row_to_indices:
                    row_to_indices[key].append(i)
                else:
                    row_to_indices[key] = [i]

            unresolved = []
            for vis in visible_obs:
                vis_row = np.asarray(vis, dtype=float).reshape(-1)
                idxs = row_to_indices.get(tuple(vis_row.tolist()), None)
                if idxs is not None:
                    visible_mask[idxs] = True
                else:
                    unresolved.append(vis_row)

            if unresolved:
                for vis_row in unresolved:
                    matches = np.all(np.isclose(obs_arr, vis_row, rtol=0.0, atol=1e-9), axis=1)
                    if np.any(matches):
                        visible_mask |= matches
        except Exception:
            return None

        # Orange for all non-visible obstacles:
        # - occluded in sensing range
        # - outside sensing range
        return ~visible_mask

    def _get_occlusion_scenarios_for_plot(self, obs_candidates=None):
        """
        Build occlusion scenarios for plotting overlays, even when the active
        controller does not manage occlusion_scenarios itself (e.g., cbf_qp).
        """
        occ_filter_fn = self._resolve_occ_filter_fn()
        if occ_filter_fn is None:
            return None
        if obs_candidates is None:
            obs_candidates = self.obs
        if obs_candidates is None:
            return None

        obs_arr = np.asarray(obs_candidates, dtype=float)
        if obs_arr.size == 0:
            return None
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        try:
            _, scenarios = occ_filter_fn(self.robot.X, obs_arr)
        except Exception:
            return None
        if scenarios is None or len(scenarios) == 0:
            return None
        return scenarios

    def _get_visible_obs_subset_for_control(self, obs_candidates):
        """
        Build visible subset for control constraints using the same occlusion/FOV
        visibility filter used by plotting.
        """
        if obs_candidates is None:
            return None
        obs_arr = np.asarray(obs_candidates, dtype=float)
        if obs_arr.size == 0:
            return None
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        if not hasattr(self, "pos_controller") or self.pos_controller is None:
            return obs_arr

        occ_filter_fn = self._resolve_occ_filter_fn()

        if occ_filter_fn is None:
            return obs_arr

        try:
            visible_obs, _ = occ_filter_fn(self.robot.X, obs_arr)
        except Exception:
            return obs_arr

        if visible_obs is None or len(visible_obs) == 0:
            return None

        vis_arr = np.asarray(visible_obs, dtype=float)
        if vis_arr.ndim == 1:
            vis_arr = vis_arr.reshape(1, -1)
        return vis_arr

    def draw_plot(self, pause=0.01, force_save=False):
        if not (self.show_animation or self.save_animation):
            return

        if self.dyn_obs_patch is None:
            # Initialize moving obstacles
            self.dyn_obs_patch = [self.ax.add_patch(plt.Circle(
                (0, 0), 0, edgecolor='black', facecolor=self.plot_dyn_obs_visible_color, fill=True)) for _ in range(len(self.obs))]
            self.dyn_obs_labels = [None] * len(self.obs)
            self.init_obs_info = self.obs.copy()

        self._plot_counter += 1
        if self.plot_every > 1 and (self._plot_counter % self.plot_every) != 0:
            return

        self.render_dyn_obs()
        self._update_trajectory_artist(self.ax, "robot_trajectory_collection")
        self._update_infeasible_marker()
        self._update_qp_stats_text()
        self._update_backup_rollout_plot()
        self._update_tracking_view()

        if self.show_animation:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(pause)

        if self.save_animation:
            self.ani_idx += 1
            if force_save or self.ani_idx % self.save_per_frame == 0:
                plt.savefig(
                    self._frame_output_path(self.ani_idx // self.save_per_frame),
                    dpi=self.save_frame_dpi,
                    format=self.save_frame_ext,
                )

    def _update_infeasible_marker(self):
        if not (self.show_animation or self.save_animation):
            return
        if not self._infeasible_active:
            if self._infeasible_text is not None:
                self._infeasible_text.set_visible(False)
            if self._tracking_infeasible_text is not None:
                self._tracking_infeasible_text.set_visible(False)
            return
        current_position = self.robot.get_position()
        pos = (current_position[0] + 0.5, current_position[1] + 0.5)
        if self._infeasible_text is None:
            self._infeasible_text = self.ax.text(
                pos[0], pos[1], '!',
                color='red', weight='bold', fontsize=22, zorder=50
            )
        else:
            self._infeasible_text.set_position(pos)
            self._infeasible_text.set_visible(True)
        if self.tracking_view_enabled:
            tracking_pos = (current_position[0] + 0.35, current_position[1] + 0.35)
            if self._tracking_infeasible_text is None:
                self._tracking_infeasible_text = self.tracking_view_ax.text(
                    tracking_pos[0], tracking_pos[1], '!',
                    color='red', weight='bold', fontsize=22, zorder=50
                )
            else:
                self._tracking_infeasible_text.set_position(tracking_pos)
                self._tracking_infeasible_text.set_visible(True)

    def draw_infeasible(self):
        """
        Draw/refresh infeasible marker with high z-order so it stays visible
        above scenario overlays (bus/car patches), then force one render.
        """
        self._infeasible_active = True
        self._infeasible_seen = True
        if not (self.show_animation or self.save_animation):
            return
        self.robot.render_plot()
        self._update_infeasible_marker()
        pause = float(self.robot_spec.get("infeasible_pause", 0.5))
        if pause < 0.0:
            pause = 0.0
        self.draw_plot(pause=pause, force_save=True)

    def _update_backup_rollout_plot(self):
        if not (self.show_animation or self.save_animation) or not self.show_backup_rollout:
            return
        pos_controller = getattr(self, "pos_controller", None)
        if pos_controller is None:
            return
        scenarios = getattr(pos_controller, "occlusion_scenarios", None)
        if not scenarios:
            scenarios = self._get_occlusion_scenarios_for_plot(self.obs)
        if not scenarios:
            if self.backup_rollout_line is not None:
                self.backup_rollout_line.set_visible(False)
            if self.backup_rollout_end_marker is not None:
                self.backup_rollout_end_marker.set_offsets(np.empty((0, 2)))
            if self.tracking_view_backup_rollout_line is not None:
                self.tracking_view_backup_rollout_line.set_visible(False)
            if self.tracking_view_backup_rollout_end_marker is not None:
                self.tracking_view_backup_rollout_end_marker.set_offsets(np.empty((0, 2)))
            if self.backup_safe_contour:
                self.robot._clear_artists(self.backup_safe_contour)
            return

        self._backup_vis_counter += 1
        if self.backup_rollout_every > 1 and (self._backup_vis_counter % self.backup_rollout_every) != 0:
            return

        T = float(getattr(pos_controller, "T_horizon", 0.0))
        if T <= 0.0:
            return
        dt_b = float(getattr(pos_controller, "dt_backup", self.dt))

        backup_solver = getattr(pos_controller, "occlusion_backup", None)
        if backup_solver is None or not hasattr(backup_solver, "simulate_backup_trajectory"):
            return

        try:
            traj = backup_solver.simulate_backup_trajectory(
                self.robot.X, T, dt_b, occlusion_scenarios=scenarios
            )[0]
        except TypeError:
            try:
                traj = backup_solver.simulate_backup_trajectory(
                    self.robot.X, T, dt_b, scenarios
                )[0]
            except Exception:
                return
        except Exception:
            return

        xy = traj[:, 0:2]
        if xy.size == 0:
            if self.backup_rollout_line is not None:
                self.backup_rollout_line.set_visible(False)
            if self.backup_rollout_end_marker is not None:
                self.backup_rollout_end_marker.set_offsets(np.empty((0, 2)))
            if self.tracking_view_backup_rollout_line is not None:
                self.tracking_view_backup_rollout_line.set_visible(False)
            if self.tracking_view_backup_rollout_end_marker is not None:
                self.tracking_view_backup_rollout_end_marker.set_offsets(np.empty((0, 2)))
            return

        xy_plot = xy
        end_xy = xy[-1, :].reshape(1, 2)

        if self.backup_rollout_line is None:
            (line,) = self.ax.plot(
                xy_plot[:, 0], xy_plot[:, 1],
                linestyle='-', color='tab:green',
                linewidth=2.4, alpha=1.0, zorder=20
            )
            line.set_path_effects([
                path_effects.Stroke(linewidth=4.2, foreground='white'),
                path_effects.Normal(),
            ])
            self.backup_rollout_line = line
            self.backup_rollout_end_marker = self.ax.scatter(
                end_xy[:, 0], end_xy[:, 1],
                s=26, facecolors='tab:green', edgecolors='white',
                linewidths=0.8, alpha=1.0, zorder=21
            )
        else:
            self.backup_rollout_line.set_data(xy_plot[:, 0], xy_plot[:, 1])
            self.backup_rollout_line.set_visible(True)
            if self.backup_rollout_end_marker is not None:
                self.backup_rollout_end_marker.set_offsets(end_xy)

        if self.tracking_view_backup_rollout_line is not None:
            self.tracking_view_backup_rollout_line.set_data(xy_plot[:, 0], xy_plot[:, 1])
            self.tracking_view_backup_rollout_line.set_visible(True)
        if self.tracking_view_backup_rollout_end_marker is not None:
            self.tracking_view_backup_rollout_end_marker.set_offsets(end_xy)

        # Safe-harbor contour is intentionally disabled for now.

    def _update_qp_stats_text(self):
        if not (self.show_animation or self.save_animation):
            return
        if not bool(self.robot_spec.get("show_qp_stats_text", True)):
            if self.qp_stats_text is not None:
                try:
                    self.qp_stats_text.remove()
                except Exception:
                    pass
                self.qp_stats_text = None
            return
        pos_controller = getattr(self, "pos_controller", None)
        if pos_controller is None:
            return
        total_ms = getattr(pos_controller, "last_total_compute_time_ms", None)
        qp_ms = getattr(pos_controller, "last_qp_solve_time_ms", None)

        # Compute time excluding plotting:
        # prefer controller-reported total (preprocess + solver), fallback to profile/qp only.
        if total_ms is None:
            prof = getattr(pos_controller, "last_profile", None)
            if isinstance(prof, dict):
                total_ms = prof.get("total_ms", None)
        if total_ms is None:
            total_ms = qp_ms

        intervention = getattr(pos_controller, "last_intervention", None)
        if intervention is None:
            # Fallback for controllers that only expose boolean intervention flags.
            int_flag = getattr(pos_controller, "_last_intervention", None)
            if isinstance(int_flag, (bool, np.bool_)):
                intervention = "backup_qp" if bool(int_flag) else "u_ref"

        controller_type = str(getattr(self, "pos_controller_type", "")).strip().lower()
        controller_label_map = {
            "occlusion_cbf_qp": "Occlusion CBF-QP",
            "cbf_qp": "CBF-QP",
            "oa_mpc": "OA-MPC",
            "single_risk_mpc": "Single-Hypothesis MPC",
            "control_tree_mpc": "Control-Tree MPC",
            "oacp_mpc": "OACP",
        }
        controller_label = controller_label_map.get(controller_type, controller_type or "controller")

        if intervention == "u_ref":
            policy_text = "Nominal (u_ref)"
        elif intervention == "backup_qp":
            if controller_type in {"occlusion_cbf_qp", "cbf_qp"}:
                policy_text = f"Safety Filter ({controller_label})"
            elif controller_type in {"oa_mpc", "single_risk_mpc", "control_tree_mpc", "oacp_mpc"}:
                policy_text = f"MPC ({controller_label})"
            else:
                policy_text = controller_label
        elif intervention == "backup_fallback":
            policy_text = f"Fallback ({controller_label})"
        elif intervention in controller_label_map:
            if intervention in {"occlusion_cbf_qp", "cbf_qp"}:
                policy_text = f"Safety Filter ({controller_label_map.get(intervention, str(intervention))})"
            else:
                policy_text = f"MPC ({controller_label_map.get(intervention, str(intervention))})"
        elif intervention == "infeasible":
            policy_text = "Infeasible"
        elif intervention is None:
            policy_text = "Unknown"
        else:
            policy_text = str(intervention)

        time_text = f"{float(total_ms):.3f} ms" if total_ms is not None else "n/a"
        value_width = 18
        text = (
            f"Computation   : {time_text:<{value_width}}\n"
            f"Control Mode  : {policy_text:<{value_width}}"
        )
        text_ax = self.tracking_view_ax if self.tracking_view_enabled else self.ax
        if self.qp_stats_text is not None and getattr(self.qp_stats_text, "axes", None) is not text_ax:
            try:
                self.qp_stats_text.remove()
            except Exception:
                pass
            self.qp_stats_text = None
        if self.qp_stats_text is None:
            self.qp_stats_text = text_ax.text(
                0.02, 0.98, text, transform=text_ax.transAxes,
                ha='left', va='top', fontsize=9, family='monospace',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2.0))
        else:
            self.qp_stats_text.set_text(text)

    def control_step(self):
        '''
        Simulate one step of tracking control with CBF-QP with the given waypoints.
        Output:
            - -2 or QPError: if the QP is infeasible or the robot collides with the obstacle
            - -1: all waypoints reached
            - 0: normal
            - 1: visibility violation
        '''
        # update state machine
        self.last_terminal_event = None
        if self.state_machine == 'stop':
            if self.robot.has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # 1. Update moving obstacles before selecting constraints
        self.step_dyn_obs()

        # 2. Update the detected obstacles (unknown obs)
        detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        # self.nearest_obs = self.get_nearest_obs(detected_obs)
        self.nearest_multi_obs = self.get_nearest_unpassed_obs(detected_obs, obs_num=self.num_constraints)
        if self.nearest_multi_obs is not None:
            self.nearest_obs = self.nearest_multi_obs[0].reshape(-1, 1)

        # 3. Compuite nominal control input, pre-defined in the robot class
        if self.state_machine == 'rotate':
            goal_angle = np.arctan2(self.goal[1] - self.robot.X[1, 0],
                                    self.goal[0] - self.robot.X[0, 0])
            if self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
                self.u_att = self.robot.rotate_to(goal_angle)
                u_ref = self.robot.stop()
            elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'Quad2D', 'VTOL2D']:
                u_ref = self.robot.rotate_to(goal_angle)
        elif self.goal is None:
            u_ref = self.robot.stop()
        else:
            # Normal waypoint tracking
            if self.pos_controller_type == 'optimal_decay_cbf_qp':
                u_ref = self.robot.nominal_input(self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
            else:
                u_ref = self.robot.nominal_input(self.goal)

        # 4. Update the CBF constraints & 5. Solve the control problem
        control_ref = {'state_machine': self.state_machine,
                       'u_ref': u_ref,
                       'goal': self.goal,
                       'waypoints': None if getattr(self, 'waypoints', None) is None else np.asarray(self.waypoints, dtype=float),
                       'goal_index': int(getattr(self, 'current_goal_index', 0))}
        try:
            # Common bookkeeping for benchmark diagnostics across controllers.
            self.pos_controller.last_u_ref = np.asarray(u_ref, dtype=float).reshape(-1, 1)
        except Exception:
            pass

        control_obs = self.nearest_multi_obs
        # For pure baseline CBF-QP, apply visibility filtering so only currently
        # visible obstacles generate CBF constraints.
        if self.pos_controller_type == 'cbf_qp' and bool(self.robot_spec.get('cbf_visible_only', True)):
            control_obs = self._get_visible_obs_subset_for_control(self.nearest_multi_obs)

        if self.pos_controller_type in ['optimal_decay_cbf_qp', 'cbf_qp']:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, control_obs)
        else:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs)

        # Guard against None/NaN control output; mark infeasible so tracking stops.
        try:
            invalid_u = (u is None) or (not np.all(np.isfinite(u)))
        except Exception:
            invalid_u = True
        if invalid_u:
            try:
                self.pos_controller.status = 'infeasible'
            except Exception:
                pass
            prev_u = getattr(self, "u_pos", None)
            if self.continue_on_infeasible and prev_u is not None:
                u = prev_u
                self._infeasible_active = True
                self._infeasible_seen = True
            else:
                self._infeasible_seen = True
                u = self.robot.stop()
        try:
            self.pos_controller.last_u = np.asarray(u, dtype=float).reshape(-1, 1)
        except Exception:
            pass

        if (self.show_animation or self.save_animation) and self.fig is not None:
            plt.figure(self.fig.number)

        # 6. Draw collision cones/parabolas for C3BF/DPCBF
        if self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.robot.draw_collision_cone(self.robot.X, self.nearest_multi_obs, self.ax)
        elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
            self.robot.draw_collision_parabola(self.robot.X, self.nearest_multi_obs, self.ax)

        # 7. Update the attitude controller
        if self.state_machine == 'track' and self.att_controller is not None:
            # att_controller is only defined for integrators
            self.u_att = self.att_controller.solve_control_problem(
                    self.robot.X, self.robot.yaw, u)

        # 8. Raise an error if the QP is infeasible, or the robot collides with the obstacle
        collide = self.is_collide_unknown()

        if self.pos_controller.status != 'optimal' or collide:
            if collide:
                # Keep latest command for logging paths that read get_control_input()
                # before loop break in run_all_steps.
                self.u_pos = u
                self._infeasible_active = True
                self._infeasible_seen = True
                self.draw_infeasible()
                print("Collision detected !!")
                self.last_terminal_event = "collision"
                if self.raise_error:
                    raise InfeasibleError("Collision detected !!")
                return -2
            if self.continue_on_infeasible:
                prev_u = getattr(self, "u_pos", None)
                if prev_u is not None:
                    u = prev_u
                self._infeasible_active = True
                self._infeasible_seen = True
            else:
                # Keep latest command for logging paths that read get_control_input()
                # before loop break in run_all_steps.
                self.u_pos = u
                self.draw_infeasible()
                print("Infeasible detected!!")
                self.last_terminal_event = "infeasible"
                self._infeasible_seen = True
                if self.raise_error:
                    raise InfeasibleError("Infeasible detected !!")
                return -2

        # 9. Step the robot
        self.robot.step(u, self.u_att)
        self.u_pos = u
        self._record_trajectory_sample()

        collide = self.is_collide_unknown()
        if collide:
            self._infeasible_active = True
            self._infeasible_seen = True
            self.draw_infeasible()
            print("Collision detected !!")
            self.last_terminal_event = "collision"
            if self.raise_error:
                raise InfeasibleError("Collision detected !!")
            return -2

        if self.plot_occ_polygons and hasattr(self.robot, "update_occlusion_polygons"):
            # Use a baseline-invariant plotting geometry. Controller-internal
            # scenario pruning is allowed for optimization, but should not alter
            # the occlusion regions shown in the benchmark videos.
            scenarios = self._get_occlusion_scenarios_for_plot(self.obs)
            if scenarios is None or len(scenarios) == 0:
                scenarios = getattr(self.pos_controller, "occlusion_scenarios", None)

            kappa = getattr(self.pos_controller, "kappa", 10.0)
            T_rollout = getattr(self.pos_controller, "T_horizon", 3.0)

            self.robot.update_occlusion_polygons(
                scenarios,
                kappa=kappa,
                show_true_occ=True,
                show_true_occ_T=False,
                show_smax_occ_T=False,
                T_rollout=T_rollout,
                grid_res=0.05,
            )

        if self.show_animation or self.save_animation:
            self.robot.render_plot()

        # 10. Update sensing information
        if 'sensor' in self.robot_spec and self.robot_spec['sensor'] == 'rgbd':
            self.robot.update_sensing_footprints()
            self.robot.update_safety_area()

            beyond_flag = self.robot.is_beyond_sensing_footprints()
        else:
            beyond_flag = 0

        if self.goal is None and self.state_machine != 'stop':
            self.last_terminal_event = "success"
            return -1  # all waypoints reached
        return beyond_flag

    def run_all_steps(self, tf=30, write_csv=False):
        print("===================================")
        print("============ Tracking =============")
        print("Start following the generated path.")
        unexpected_beh = 0
        compute_ms_sum = 0.0
        compute_ms_count = 0

        if write_csv:
            with open('output.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['states', 'control_inputs', 'alpha1', 'alpha2'])

        for _ in range(int(tf / self.dt)):
            ret = self.control_step()
            self.draw_plot()
            unexpected_beh += ret

            pos_controller = getattr(self, "pos_controller", None)
            step_compute_ms = None
            if pos_controller is not None:
                step_compute_ms = getattr(pos_controller, "last_total_compute_time_ms", None)
                if step_compute_ms is None:
                    profile = getattr(pos_controller, "last_profile", None)
                    if isinstance(profile, dict):
                        step_compute_ms = profile.get("total_ms", None)
                if step_compute_ms is None:
                    step_compute_ms = getattr(pos_controller, "last_qp_solve_time_ms", None)
            if step_compute_ms is not None and np.isfinite(step_compute_ms):
                compute_ms_sum += float(step_compute_ms)
                compute_ms_count += 1

            robot_state = self.robot.X[:, 0].flatten()
            control_input = self.get_control_input().flatten()

            if write_csv:
                with open('output.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        np.append(
                            robot_state,
                            np.append(
                                control_input,
                                [
                                    self.pos_controller.cbf_param['alpha1'],
                                    self.pos_controller.cbf_param['alpha2'],
                                ],
                            ),
                        )
                    )

            if ret == -1 or ret == -2:
                break

        self.export_video()

        self.avg_compute_time_ms = (
            (compute_ms_sum / compute_ms_count) if compute_ms_count > 0 else None
        )
        self.compute_time_sample_count = int(compute_ms_count)
        if self.avg_compute_time_ms is not None:
            print(
                f"[STATS] Avg computation time (preprocess+solver, no plotting): "
                f"{self.avg_compute_time_ms:.3f} ms over {self.compute_time_sample_count} steps"
            )

        print("=====   Tracking finished    =====")
        print("===================================\n")
        if self.show_animation or self.save_animation:
            plt.ioff()
            plt.close()

        return unexpected_beh
