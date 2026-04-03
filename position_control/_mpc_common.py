import numpy as np

try:
    import casadi as ca
except Exception:
    ca = None


class MPCCommonUtils:
    """
    Shared helpers for MPC-style baselines.

    This module intentionally keeps only pure utility methods so that
    each controller can preserve its own optimization logic.
    """

    def _model_name(self):
        return str(getattr(self, "model", self.robot_spec.get("model", ""))).strip()

    def _dims(self):
        model = self._model_name()
        if model == "DoubleIntegrator2D":
            return 4, 2
        if model == "DynamicUnicycle2D":
            return 4, 2
        return 3, 2

    def _speed_bounds(self):
        v_max = float(self.robot_spec.get("v_max", 1.0))
        if self._model_name() == "DoubleIntegrator2D":
            return 0.0, float(v_max)
        if "v_min" in self.robot_spec:
            v_min = float(self.robot_spec.get("v_min", 0.0))
        else:
            v_min = 0.0 if bool(getattr(self, "forward_only", False)) else -v_max
        if (not np.isfinite(v_min)) or v_min > v_max:
            v_min = 0.0 if bool(getattr(self, "forward_only", False)) else -v_max
        return float(v_min), float(v_max)

    def _input_bounds(self):
        model = self._model_name()
        if model == "DoubleIntegrator2D":
            ax_max = float(self.robot_spec.get("ax_max", self.robot_spec.get("a_max", 1.0)))
            ay_max = float(self.robot_spec.get("ay_max", self.robot_spec.get("a_max", 1.0)))
            lb = np.array([-ax_max, -ay_max], dtype=float)
            ub = np.array([ax_max, ay_max], dtype=float)
            return lb, ub
        w_max = float(self.robot_spec.get("w_max", 0.8))
        if model == "DynamicUnicycle2D":
            a_max = float(self.robot_spec.get("a_max", 1.0))
            lb = np.array([-a_max, -w_max], dtype=float)
            ub = np.array([a_max, w_max], dtype=float)
            return lb, ub
        v_min, v_max = self._speed_bounds()
        lb = np.array([v_min, -w_max], dtype=float)
        ub = np.array([v_max, w_max], dtype=float)
        return lb, ub

    def _clip_input(self, u):
        lb, ub = self._input_bounds()
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.clip(u, lb, ub).reshape(-1, 1)

    @staticmethod
    def _normalize_angle(theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def _normalize_state(self, x):
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        if self._model_name() in {"Unicycle2D", "DynamicUnicycle2D"} and x.size >= 3:
            x[2] = self._normalize_angle(x[2])
        return x

    def _state_from_robot_state(self, robot_state):
        x = self._normalize_state(robot_state)
        nx, _ = self._dims()
        if self._model_name() == "DoubleIntegrator2D":
            vx = float(x[2]) if x.size >= 3 else 0.0
            vy = float(x[3]) if x.size >= 4 else 0.0
            return np.array([x[0], x[1], vx, vy], dtype=float)
        if self._model_name() == "DynamicUnicycle2D":
            v0 = float(x[3]) if x.size >= 4 else 0.0
            return np.array([x[0], x[1], x[2], v0], dtype=float)
        return np.array([x[0], x[1], x[2]], dtype=float)

    def _discrete_np(self, x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if self._model_name() == "DoubleIntegrator2D":
            xn = np.array(
                [
                    x[0] + self.dt_plan * x[2],
                    x[1] + self.dt_plan * x[3],
                    x[2] + self.dt_plan * u[0],
                    x[3] + self.dt_plan * u[1],
                ],
                dtype=float,
            )
            _, v_max = self._speed_bounds()
            v_mag = float(np.linalg.norm(xn[2:4]))
            if np.isfinite(v_max) and v_mag > max(v_max, 1e-9):
                xn[2:4] *= float(v_max) / v_mag
            return xn
        if self._model_name() == "DynamicUnicycle2D":
            xn = np.array(
                [
                    x[0] + self.dt_plan * x[3] * np.cos(x[2]),
                    x[1] + self.dt_plan * x[3] * np.sin(x[2]),
                    self._normalize_angle(x[2] + self.dt_plan * u[1]),
                    x[3] + self.dt_plan * u[0],
                ],
                dtype=float,
            )
            v_min, v_max = self._speed_bounds()
            xn[3] = float(np.clip(xn[3], v_min, v_max))
            return xn
        xn = np.array(
            [
                x[0] + self.dt_plan * u[0] * np.cos(x[2]),
                x[1] + self.dt_plan * u[0] * np.sin(x[2]),
                self._normalize_angle(x[2] + self.dt_plan * u[1]),
            ],
            dtype=float,
        )
        return xn

    def _discrete_ca(self, xk, uk):
        if ca is None:
            raise RuntimeError("CasADi is not available.")
        if self._model_name() == "DoubleIntegrator2D":
            return ca.vertcat(
                xk[0] + self.dt_plan * xk[2],
                xk[1] + self.dt_plan * xk[3],
                xk[2] + self.dt_plan * uk[0],
                xk[3] + self.dt_plan * uk[1],
            )
        if self._model_name() == "DynamicUnicycle2D":
            return ca.vertcat(
                xk[0] + self.dt_plan * xk[3] * ca.cos(xk[2]),
                xk[1] + self.dt_plan * xk[3] * ca.sin(xk[2]),
                xk[2] + self.dt_plan * uk[1],
                xk[3] + self.dt_plan * uk[0],
            )
        return ca.vertcat(
            xk[0] + self.dt_plan * uk[0] * ca.cos(xk[2]),
            xk[1] + self.dt_plan * uk[0] * ca.sin(xk[2]),
            xk[2] + self.dt_plan * uk[1],
        )

    def _stop_input(self):
        try:
            u_stop = np.asarray(self.robot.stop(), dtype=float).reshape(-1, 1)
        except Exception:
            u_stop = np.zeros((2, 1), dtype=float)
        return self._clip_input(u_stop)

    @staticmethod
    def _goal_xy(x0, goal):
        p = np.asarray(x0, dtype=float).reshape(-1)[:2].copy()
        if goal is None:
            return p
        g = np.asarray(goal, dtype=float).reshape(-1)
        if g.size >= 2:
            p[0], p[1] = g[0], g[1]
        return p

    def _stage_speed_np(self, X, U, k):
        if self._model_name() == "DoubleIntegrator2D":
            vel = np.asarray(X, dtype=float)[2:4, k]
            return float(np.linalg.norm(vel))
        if self._model_name() == "DynamicUnicycle2D":
            return float(np.asarray(X, dtype=float)[3, k])
        return float(np.asarray(U, dtype=float)[0, k])

    def _stage_speed_ca(self, X, U, k):
        if self._model_name() == "DoubleIntegrator2D":
            return ca.sqrt(ca.sumsqr(X[2:4, k]) + 1e-12)
        if self._model_name() == "DynamicUnicycle2D":
            return X[3, k]
        return U[0, k]

    def _nominal_speed_reference(self, x0, u_ref, floor_speed):
        floor_speed = float(floor_speed)
        if self._model_name() == "DoubleIntegrator2D":
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
            v_cur = x0[2:4] if x0.size >= 4 else np.zeros((2,), dtype=float)
            v_des = v_cur + self.dt_plan * u_ref[:2]
            _, v_max = self._speed_bounds()
            v_mag = float(np.linalg.norm(v_des))
            if np.isfinite(v_max) and v_mag > max(v_max, 1e-9):
                v_des = v_des * (float(v_max) / v_mag)
            return max(float(np.linalg.norm(v_des)), floor_speed)
        if self._model_name() == "DynamicUnicycle2D":
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
            v_min, v_max = self._speed_bounds()
            v_cur = float(x0[3]) if x0.size >= 4 else 0.0
            # DynamicUnicycle2D.nominal_input uses a = k_a * (v_des - v_cur),
            # and the default nominal path in this repo uses k_a = 1.0.
            v_des = float(np.clip(v_cur + float(u_ref[0]), v_min, v_max))
            return max(abs(v_des), floor_speed)
        return max(abs(float(np.asarray(u_ref, dtype=float).reshape(-1)[0])), floor_speed)

    def _guidance_input_np(self, x, guidance_xy, v_ref_nom, k_heading=2.0):
        x = np.asarray(x, dtype=float).reshape(-1)
        guidance_xy = np.asarray(guidance_xy, dtype=float).reshape(2,)
        lb_u, ub_u = self._input_bounds()
        if self._model_name() == "DoubleIntegrator2D":
            to_g = guidance_xy - x[:2]
            dist = float(np.linalg.norm(to_g))
            if dist <= 1e-9:
                return self._clip_input(np.zeros((2, 1), dtype=float)).reshape(2,)
            dir_g = to_g / dist
            v_des_mag = float(np.clip(v_ref_nom, 0.0, self._speed_bounds()[1]))
            if dist < max(0.2, 0.5 * self.dt_plan * max(v_des_mag, 1e-3)):
                v_des_mag = min(v_des_mag, dist / max(self.dt_plan, 1e-6))
            v_des = v_des_mag * dir_g
            v_cur = x[2:4] if x.size >= 4 else np.zeros((2,), dtype=float)
            gain = max(0.5, float(k_heading))
            a_cmd = gain * (v_des - v_cur)
            return np.clip(a_cmd, lb_u, ub_u).reshape(2,)
        to_g = guidance_xy - x[:2]
        hdg_des = float(np.arctan2(to_g[1], to_g[0]))
        err = self._angle_wrap(hdg_des - x[2])
        w_cmd = float(np.clip(k_heading * err, lb_u[1], ub_u[1]))
        if self._model_name() == "DynamicUnicycle2D":
            v_min, v_max = self._speed_bounds()
            v_des = float(np.clip(v_ref_nom, v_min, v_max))
            v_cur = float(x[3]) if x.size >= 4 else 0.0
            a_cmd = float(np.clip(v_des - v_cur, lb_u[0], ub_u[0]))
            return np.array([a_cmd, w_cmd], dtype=float)
        v_cmd = float(np.clip(v_ref_nom, lb_u[0], ub_u[0]))
        return np.array([v_cmd, w_cmd], dtype=float)

    @staticmethod
    def _angle_wrap(a):
        return ((float(a) + np.pi) % (2.0 * np.pi)) - np.pi

    @staticmethod
    def _merge_intervals(intervals):
        if len(intervals) == 0:
            return []
        intervals = sorted(intervals, key=lambda ab: ab[0])
        merged = [list(intervals[0])]
        for a, b in intervals[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(float(a), float(b)) for a, b in merged]

    def _predict_obs_center(self, obs, k):
        obs = np.asarray(obs, dtype=float).reshape(-1)
        c0 = obs[:2]
        if obs.size >= 5:
            vx, vy = float(obs[3]), float(obs[4])
        else:
            vx, vy = 0.0, 0.0
        dt = self.dt_plan * float(k)
        return c0 + dt * np.array([vx, vy], dtype=float)

    def _nearest_visible_obs(self, visible_obs, x0):
        if visible_obs is None or len(visible_obs) == 0:
            return []
        p = np.asarray(x0, dtype=float).reshape(-1)[:2]
        obs_arr = np.asarray(visible_obs, dtype=float)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        d = np.linalg.norm(obs_arr[:, :2] - p[None, :], axis=1)
        max_n = int(getattr(self, "max_visible_obs", obs_arr.shape[0]))
        idx = np.argsort(d)[: max(0, max_n)]
        return [obs_arr[i].copy() for i in idx]

    def _state_heading_np(self, x, ref_heading=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        model = self._model_name()
        if model in {"Unicycle2D", "DynamicUnicycle2D"}:
            return float(x[2])
        if model == "DoubleIntegrator2D":
            vx = float(x[2]) if x.size >= 3 else 0.0
            vy = float(x[3]) if x.size >= 4 else 0.0
            speed = float(np.hypot(vx, vy))
            if speed <= 1e-6:
                return 0.0 if ref_heading is None else float(ref_heading)
            return float(np.arctan2(vy, vx))
        return 0.0 if ref_heading is None else float(ref_heading)

    def _state_heading_ca(self, xk, ref_heading=0.0):
        model = self._model_name()
        if model in {"Unicycle2D", "DynamicUnicycle2D"}:
            return xk[2]
        if model == "DoubleIntegrator2D":
            speed2 = ca.sumsqr(xk[2:4])
            hdg_vel = ca.atan2(xk[3], xk[2] + 1e-12)
            return ca.if_else(speed2 > 1e-8, hdg_vel, ref_heading)
        return ref_heading

    def _heading_track_weight_ca(self, xk):
        if self._model_name() == "DoubleIntegrator2D":
            speed = ca.sqrt(ca.sumsqr(xk[2:4]) + 1e-12)
            return speed / (speed + 0.05)
        return 1.0

    def _heading_error_ca(self, xk, ref_heading):
        hdg = self._state_heading_ca(xk, ref_heading)
        return ca.atan2(ca.sin(hdg - ref_heading), ca.cos(hdg - ref_heading))
