import numpy as np

try:
    import casadi as ca
except Exception:
    ca = None


class MPCCommonUtils:
    """
    Shared helpers for unicycle MPC-style baselines.

    This module intentionally keeps only pure utility methods so that
    each controller can preserve its own optimization logic.
    """

    def _input_bounds(self):
        v_max = float(self.robot_spec.get("v_max", 1.0))
        w_max = float(self.robot_spec.get("w_max", 0.8))
        if "v_min" in self.robot_spec:
            v_min = float(self.robot_spec.get("v_min", 0.0))
        else:
            v_min = 0.0 if bool(getattr(self, "forward_only", False)) else -v_max
        if (not np.isfinite(v_min)) or v_min > v_max:
            v_min = 0.0 if bool(getattr(self, "forward_only", False)) else -v_max
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

    def _discrete_np(self, x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
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
