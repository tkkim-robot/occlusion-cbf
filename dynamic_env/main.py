from safe_control.dynamic_env.main import LocalTrackingControllerDyn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import subprocess
import csv

class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)

class LocalTrackingControllerDyn_OCC(LocalTrackingControllerDyn):

    def __init__(self, X0, robot_spec,
                 controller_type=None,
                 dt=0.05,
                 show_animation=False, save_animation=False, show_mpc_traj=False,
                 enable_rotation=True, raise_error=False,
                 ax=None, fig=None, env=None, rand_seed=42):
        super().__init__(X0, robot_spec,
                         controller_type=controller_type,
                         dt=dt,
                         show_animation=show_animation, save_animation=save_animation, show_mpc_traj=show_mpc_traj,
                         enable_rotation=enable_rotation, raise_error=raise_error,
                         ax=ax, fig=fig, env=env)
        
        if self.pos_controller_type == 'occlusion_cbf_qp':
            from position_control.occlusion_cbf_qp import OcclusionCBFQP
            self.pos_controller = OcclusionCBFQP(self.robot, self.robot_spec, num_obs=30)
        
        self.qp_stats_text = None
        self.backup_rollout_line = None
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
        self.plot_occ_polygons = bool(self.robot_spec.get('plot_occ_polygons', self.show_animation))
        self.continue_on_infeasible = bool(self.robot_spec.get('continue_on_infeasible', False))
        self._infeasible_active = False
        self._infeasible_text = None
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
        
        self._rng = np.random.default_rng(rand_seed)
        self.obs_meta = None

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
    def make_random_obstacles7(n_rand, v_obs_max, x_range, y_spawn_range, r_range, y_bounds, seed=42, rand_obs=True):
        if not rand_obs:
            return np.empty((0, 7), dtype=float) , []
        rng = np.random.default_rng(seed)
        x_min, x_max = x_range
        y_min_spawn, y_max_spawn = y_spawn_range
        r_min, r_max = r_range
        y_min_g, y_max_g = y_bounds

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
            vx0, vy0 = v_obs_max*np.cos(theta0), v_obs_max*np.sin(theta0)
            rows.append([x0, y0, r, vx0, vy0, y_min_g, y_max_g])
            metas.append({'mode': 1, 'v_max': v_obs_max, 'theta': theta0})
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

        for i in range(self.obs.shape[0]):
            x, y, r, vx, vy, y_min, y_max = self.obs[i, :7]
            meta = self.obs_meta[i]
            mode   = int(meta.get('mode', 0))
            v_max  = float(meta.get('v_max', np.hypot(vx, vy)))
            if 'theta' not in meta:
                meta['theta'] = np.arctan2(vy, vx) if v_max > 1e-9 else 0.0 
            theta  = float(meta['theta'])

            if mode == 1:
                # --- Random walker: heading noise + occasional large turn ---
                dtheta = self._rng.normal(0.0, 0.0)          # small heading jitter
                if self._rng.random() < 0.05:                 # 5% chance of a large turn
                    dtheta += self._rng.normal(0.0, 0.2)
                theta += dtheta
                vx, vy = v_max * np.cos(theta), v_max * np.sin(theta)
                meta['theta'] = theta   # store latest heading in meta

            # Update position
            x_new = x + vx * self.dt
            y_new = y + vy * self.dt

            # Reflect at y bounds
            if y_new >= y_max:
                y_new = y_max
                vy = -abs(vy)
                if mode == 1:
                    meta['theta'] = -meta['theta']            # mirror across x-axis
                    vx, vy = v_max*np.cos(meta['theta']), v_max*np.sin(meta['theta'])
            elif y_new <= y_min:
                y_new = y_min
                vy =  abs(vy)
                if mode == 1:
                    meta['theta'] = -meta['theta']
                    vx, vy = v_max*np.cos(meta['theta']), v_max*np.sin(meta['theta'])

            # Write back
            self.obs[i, 0] = x_new
            self.obs[i, 1] = y_new
            self.obs[i, 3] = vx
            self.obs[i, 4] = vy
        
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
                        color='orange',
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
                    patch.set_facecolor('orange' if is_occ else 'gray')
                    if self._last_occluded_mask is not None:
                        self._last_occluded_mask[i] = is_occ
            else:
                patch.set_facecolor('gray')

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
    
    def _get_occluded_obs_mask(self):
        if not hasattr(self, "pos_controller") or self.pos_controller is None:
            return None
        occ_filter_fn = None
        if hasattr(self.pos_controller, "_filter_visible_and_build_occ"):
            occ_filter_fn = self.pos_controller._filter_visible_and_build_occ
        elif hasattr(self.pos_controller, "_occ_utils") and \
                hasattr(self.pos_controller._occ_utils, "_filter_visible_and_build_occ"):
            occ_filter_fn = self.pos_controller._occ_utils._filter_visible_and_build_occ
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

        try:
            visible_obs, _ = occ_filter_fn(self.robot.X, self.obs)
        except Exception:
            return None

        obs_arr = np.asarray(self.obs, dtype=float)
        visible_mask = np.zeros(obs_arr.shape[0], dtype=bool)
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

        # Orange for all non-visible obstacles:
        # - occluded in sensing range
        # - outside sensing range
        return ~visible_mask

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            if self.dyn_obs_patch is None:
                # Initialize moving obstacles
                self.dyn_obs_patch = [self.ax.add_patch(plt.Circle(
                    (0, 0), 0, edgecolor='black', facecolor='gray', fill=True)) for _ in range(len(self.obs))]
                self.dyn_obs_labels = [None] * len(self.obs)
                self.init_obs_info = self.obs.copy()
                
            self._plot_counter += 1
            if self.plot_every > 1 and (self._plot_counter % self.plot_every) != 0:
                return
            
            self.render_dyn_obs()
            self._update_infeasible_marker()
            self._update_qp_stats_text()
            self._update_backup_rollout_plot()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(pause)
            if self.save_animation:
                self.ani_idx += 1
                if force_save or self.ani_idx % self.save_per_frame == 0:
                    plt.savefig(self.current_directory_path +
                                "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".png", dpi=300)
                    # plt.savefig(self.current_directory_path +
                    #             "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".svg")

    def _update_infeasible_marker(self):
        if not self.show_animation:
            return
        if not self._infeasible_active:
            if self._infeasible_text is not None:
                self._infeasible_text.set_visible(False)
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

    def draw_infeasible(self):
        """
        Draw/refresh infeasible marker with high z-order so it stays visible
        above scenario overlays (bus/car patches), then force one render.
        """
        self._infeasible_active = True
        self._infeasible_seen = True
        if not self.show_animation:
            return
        self.robot.render_plot()
        self._update_infeasible_marker()
        pause = float(self.robot_spec.get("infeasible_pause", 0.5))
        if pause < 0.0:
            pause = 0.0
        self.draw_plot(pause=pause, force_save=True)

    def _update_backup_rollout_plot(self):
        if not self.show_animation or not self.show_backup_rollout:
            return
        pos_controller = getattr(self, "pos_controller", None)
        if pos_controller is None:
            return
        scenarios = getattr(pos_controller, "occlusion_scenarios", None)
        if not scenarios:
            if self.backup_rollout_line is not None:
                self.backup_rollout_line.set_visible(False)
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
            return

        if self.backup_rollout_line is None:
            (line,) = self.ax.plot(
                xy[:, 0], xy[:, 1],
                linestyle='--', color='tab:green',
                linewidth=2.0, alpha=0.9, zorder=2
            )
            self.backup_rollout_line = line
        else:
            self.backup_rollout_line.set_data(xy[:, 0], xy[:, 1])
            self.backup_rollout_line.set_visible(True)

        # Safe-harbor contour is intentionally disabled for now.

    def _update_qp_stats_text(self):
        if not self.show_animation:
            return
        pos_controller = getattr(self, "pos_controller", None)
        if pos_controller is None:
            return
        num_constraints = getattr(pos_controller, "last_num_constraints", None)
        total_ms = getattr(pos_controller, "last_total_compute_time_ms", None)
        qp_ms = getattr(pos_controller, "last_qp_solve_time_ms", None)
        if num_constraints is None:
            return

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

        if intervention == "u_ref":
            policy_text = "u_ref (nominal)"
        elif intervention == "backup_qp":
            policy_text = "backup_qp (QP)"
        elif intervention == "backup_fallback":
            policy_text = "backup_fallback"
        elif intervention is None:
            policy_text = "unknown"
        else:
            policy_text = str(intervention)

        constraints_text = f"{int(num_constraints)}"
        time_text = f"{float(total_ms):.3f} ms" if total_ms is not None else "n/a"
        value_width = 18
        text = (
            f"QP Constraints: {constraints_text:<{value_width}}\n"
            f"Computation   : {time_text:<{value_width}}\n"
            f"Control Policy: {policy_text:<{value_width}}"
        )
        if self.qp_stats_text is None:
            self.qp_stats_text = self.ax.text(
                0.02, 0.98, text, transform=self.ax.transAxes,
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
                       'goal': self.goal}
        
        if self.pos_controller_type in ['optimal_decay_cbf_qp', 'cbf_qp']:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs) 
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

        if self.show_animation and self.fig is not None:
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
                self._infeasible_active = True
                self._infeasible_seen = True
                self.draw_infeasible()
                # cause = "Collision" if collide else "Infeasible"
                # self.draw_infeasible()
                print("Collision detected !!")
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
                self.draw_infeasible()
                print("Infeasible detected!!")
                self._infeasible_seen = True
                if self.raise_error:
                    raise InfeasibleError("Infeasible detected !!")
                return -2

        # 9. Step the robot
        self.robot.step(u, self.u_att)
        self.u_pos = u

        if self.plot_occ_polygons and hasattr(self.robot, "update_occlusion_polygons") and \
           hasattr(self.pos_controller, "occlusion_scenarios"):

            kappa = getattr(self.pos_controller, "kappa", 10.0)

            self.robot.update_occlusion_polygons(
                self.pos_controller.occlusion_scenarios,
                kappa=kappa,
                show_true_occ=True,
                show_true_occ_T=False,
                show_smax_occ_T=False,
                T_rollout=getattr(self.pos_controller, "T_horizon", 3.0),
                grid_res=0.05,
            )

        if self.show_animation:
            self.robot.render_plot()

        # 10. Update sensing information
        if 'sensor' in self.robot_spec and self.robot_spec['sensor'] == 'rgbd':
            self.robot.update_sensing_footprints()
            self.robot.update_safety_area()

            beyond_flag = self.robot.is_beyond_sensing_footprints()
            if beyond_flag and self.show_animation:
                pass
                # print("Visibility Violation")
        else:
            beyond_flag = 0 # not checking sensing footprint

        if self.goal is None and self.state_machine != 'stop':
            return -1  # all waypoints reached
        return beyond_flag