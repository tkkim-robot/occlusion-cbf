# from safe_control.robots.double_integrator2D import DoubleIntegrator2D
# import numpy as np
# import casadi as ca
# import math

# # try:
# #     from numba import njit
# #     _NUMBA_AVAILABLE = True
# # except Exception:
# #     njit = None
# #     _NUMBA_AVAILABLE = False

# # if _NUMBA_AVAILABLE:
# #     @njit(cache=True)
# #     def _occ_vel_ref_jit(p, v, A, b0, v_expand, v_adv, K_counts, R, tau):
# #         S = K_counts.shape[0]
# #         if S == 0:
# #             return np.zeros(2, dtype=np.float64)

# #         v_sum = np.zeros(2, dtype=np.float64)
# #         count = 0
# #         for s in range(S):
# #             K = K_counts[s]
# #             if K <= 0:
# #                 continue
# #             avg_n0 = 0.0
# #             avg_n1 = 0.0
# #             active_count = 0
# #             max_h = -1e18
# #             best_idx = 0

# #             for k in range(K):
# #                 h = A[s, k, 0] * p[0] + A[s, k, 1] * p[1] - b0[s, k] - (R + v_expand[s, k] * tau)
# #                 if h >= 0.0:
# #                     avg_n0 += A[s, k, 0]
# #                     avg_n1 += A[s, k, 1]
# #                     active_count += 1
# #                 if h > max_h:
# #                     max_h = h
# #                     best_idx = k

# #             if active_count > 0:
# #                 avg_n0 /= active_count
# #                 avg_n1 /= active_count
# #             else:
# #                 avg_n0 = A[s, best_idx, 0]
# #                 avg_n1 = A[s, best_idx, 1]

# #             norm = math.sqrt(avg_n0 * avg_n0 + avg_n1 * avg_n1)
# #             if norm > 1e-9:
# #                 v_sum[0] += (avg_n0 / norm) * v_adv[s]
# #                 v_sum[1] += (avg_n1 / norm) * v_adv[s]
# #                 count += 1

# #         if count == 0:
# #             return np.zeros(2, dtype=np.float64)
# #         return v_sum / count

# #     @njit(cache=True)
# #     def _backup_input_occ_jit(x, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t, k_d):
# #         tau = t
# #         if tau < 0.0:
# #             tau = 0.0
# #         elif tau > T:
# #             tau = T

# #         p = np.array([x[0], x[1]], dtype=np.float64)
# #         v = np.array([x[2], x[3]], dtype=np.float64)
# #         v_ref = _occ_vel_ref_jit(p, v, A, b0, v_expand, v_adv, K_counts, R, tau)

# #         e0 = v[0] - v_ref[0]
# #         e1 = v[1] - v_ref[1]
# #         u0 = -Kp * e0 - k_d * v[0]
# #         u1 = -Kp * e1 - k_d * v[1]

# #         if u0 > a_lim:
# #             u0 = a_lim
# #         elif u0 < -a_lim:
# #             u0 = -a_lim
# #         if u1 > a_lim:
# #             u1 = a_lim
# #         elif u1 < -a_lim:
# #             u1 = -a_lim

# #         eps = 1e-6
# #         if v[0] >= (v_max - eps) and u0 > 0.0:
# #             u0 = 0.0
# #         if v[0] <= (-v_max + eps) and u0 < 0.0:
# #             u0 = 0.0
# #         if v[1] >= (v_max - eps) and u1 > 0.0:
# #             u1 = 0.0
# #         if v[1] <= (-v_max + eps) and u1 < 0.0:
# #             u1 = 0.0

# #         return u0, u1

# #     @njit(cache=True)
# #     def _f_cl_jit(x, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t):
# #         if K_counts.shape[0] > 0:
# #             u0, u1 = _backup_input_occ_jit(
# #                 x, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t, 1.0
# #             )
# #         else:
# #             u0 = -x[2]
# #             u1 = -x[3]
# #         return np.array([x[2], x[3], u0, u1], dtype=np.float64)

# #     @njit(cache=True)
# #     def _jac_f_cl_fd_jit(x, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t, eps):
# #         J = np.zeros((4, 4), dtype=np.float64)
# #         Xp = x.copy()
# #         Xm = x.copy()
# #         for j in range(4):
# #             xj = x[j]
# #             eps_j = eps * (1.0 + abs(xj))
# #             Xp[j] = xj + eps_j
# #             Xm[j] = xj - eps_j
# #             fp = _f_cl_jit(Xp, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t)
# #             fm = _f_cl_jit(Xm, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t)
# #             J[:, j] = (fp - fm) / (2.0 * eps_j)
# #             Xp[j] = xj
# #             Xm[j] = xj
# #         return J

# #     @njit(cache=True)
# #     def _simulate_backup_trajectory_jit(x0, T, dt, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, eps_A):
# #         N = int(np.floor(T / dt)) + 1
# #         t_grid = dt * np.arange(N, dtype=np.float64)

# #         backup_traj = np.zeros((N, 4), dtype=np.float64)
# #         stm_traj = np.zeros((N, 4, 4), dtype=np.float64)
# #         fcl_traj = np.zeros((N, 4), dtype=np.float64)

# #         x = x0.copy()
# #         Phi = np.eye(4, dtype=np.float64)
# #         backup_traj[0] = x
# #         stm_traj[0] = Phi

# #         I = np.eye(4, dtype=np.float64)
# #         half_dt = 0.5 * dt
# #         sixth_dt = dt / 6.0

# #         for k in range(N - 1):
# #             t = t_grid[k]

# #             k1 = _f_cl_jit(x, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t)
# #             fcl_traj[k] = k1
# #             k2 = _f_cl_jit(x + half_dt * k1, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t + half_dt)
# #             k3 = _f_cl_jit(x + half_dt * k2, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t + half_dt)
# #             k4 = _f_cl_jit(x + dt * k3,      A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t + dt)
# #             x_next = x + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# #             x_mid = x + half_dt * k2
# #             A_mat = _jac_f_cl_fd_jit(x_mid, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t + half_dt, eps_A)
# #             Phi_next = (I + dt * A_mat) @ Phi

# #             x = x_next
# #             Phi = Phi_next
# #             backup_traj[k + 1] = x
# #             stm_traj[k + 1] = Phi

# #         fcl_traj[-1] = _f_cl_jit(x, A, b0, v_expand, v_adv, K_counts, a_lim, v_max, Kp, R, T, t_grid[-1])
# #         return backup_traj, stm_traj, t_grid, fcl_traj

# # def _pack_occ_scenarios(occlusion_scenarios):
# #     if not occlusion_scenarios:
# #         A = np.zeros((0, 1, 2), dtype=np.float64)
# #         b0 = np.zeros((0, 1), dtype=np.float64)
# #         v_expand = np.zeros((0, 1), dtype=np.float64)
# #         v_adv = np.zeros((0,), dtype=np.float64)
# #         K_counts = np.zeros((0,), dtype=np.int64)
# #         return A, b0, v_expand, v_adv, K_counts

# #     K_max = 0
# #     for sc in occlusion_scenarios:
# #         A_i = np.asarray(sc["A"], dtype=np.float64)
# #         if A_i.shape[0] > K_max:
# #             K_max = A_i.shape[0]

# #     S = len(occlusion_scenarios)
# #     A = np.zeros((S, K_max, 2), dtype=np.float64)
# #     b0 = np.zeros((S, K_max), dtype=np.float64)
# #     v_expand = np.zeros((S, K_max), dtype=np.float64)
# #     v_adv = np.zeros((S,), dtype=np.float64)
# #     K_counts = np.zeros((S,), dtype=np.int64)

# #     for i, sc in enumerate(occlusion_scenarios):
# #         A_i = np.asarray(sc["A"], dtype=np.float64)
# #         b0_i = np.asarray(sc["b0"], dtype=np.float64).reshape(-1,)
# #         K = A_i.shape[0]
# #         A[i, :K, :] = A_i
# #         b0[i, :K] = b0_i
# #         if "v_expand_vec" in sc and sc["v_expand_vec"] is not None:
# #             v_exp = np.asarray(sc["v_expand_vec"], dtype=np.float64).reshape(-1,)
# #             if v_exp.size != K:
# #                 v_exp = np.full((K,), float(sc.get("v_adv_max", 0.0)), dtype=np.float64)
# #         else:
# #             v_exp = np.full((K,), float(sc.get("v_adv_max", 0.0)), dtype=np.float64)
# #         v_expand[i, :K] = v_exp
# #         v_adv[i] = float(sc.get("v_adv_max", 0.0))
# #         K_counts[i] = K

# #     return A, b0, v_expand, v_adv, K_counts

# """
# Created on July 15th, 2024
# @author: Taekyung Kim

# @description: 
# Double Integrator model for CBF-QP and MPC-CBF (casadi) with separated position and attitude states
# """

# def angle_normalize(x):
#     if isinstance(x, (np.ndarray, float, int)):
#         # NumPy implementation
#         return (((x + np.pi) % (2 * np.pi)) - np.pi)
#     elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
#         # CasADi implementation
#         return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
#     else:
#         raise TypeError(f"Unsupported input type: {type(x)}")


# class DoubleIntegrator2D_OCC(DoubleIntegrator2D):
#     def __init__(self, dt, robot_spec):
#         super().__init__(dt, robot_spec)
        
#         self.pid_occ_gains = {
#             "Kp": 1.0,
#             "Ki": 0.2,
#             "Kd": 0.1,
#             "aw_limit": 1.0
#         }
#         cfg = self.robot_spec.setdefault("backup_cbf", {})
#         self.T_horizon = float(cfg.get("T_horizon", 2.0))
#         cfg["T_horizon"] = self.T_horizon
#         # self._use_numba = bool(self.robot_spec.get("use_numba", True))
#         # self._occ_cache = None

#     # ---- Occlusion-Aware Backup CBF for Double Integrator ----
#     def get_backup_horizon(self):
#         return float(getattr(self, "T_horizon", 2.0))

#     def clamp_tau(self, tau):
#         if tau is None:
#             return None
#         T = self.get_backup_horizon()
#         tau_f = float(tau)
#         return max(0.0, min(tau_f, T))

#     def set_occ_barrier_fn(self, fn):
#         self._occ_barrier_fn = fn

#     def _occ_barrier(self, pos, scenario, tau=None):
#         fn = getattr(self, "_occ_barrier_fn", None)
#         if fn is None:
#             raise AttributeError("Occlusion barrier function not set.")
#         if not isinstance(scenario, dict):
#             raise TypeError(f"scenario must be dict, got {type(scenario)}")
        
#         if tau is None:
#             tau = self.get_backup_horizon()
#         else:
#             tau = self.clamp_tau(tau)

#         return fn(pos, scenario, tau)
    
#     def _occ_vel_ref(self, X, scenarios, t):

#         if (scenarios is None) or (len(scenarios) == 0):
#             return np.zeros(2, dtype=float)
        
#         p = np.array([float(X[0,0]), float(X[1,0])], dtype=float)
#         v = np.array([float(X[2,0]), float(X[3,0])], dtype=float)

#         a_lim = float(self.robot_spec.get('a_max', 1.0))
#         R     = self.robot_spec['radius']

#         tau = float(t) if t is not None else 0.0
#         tau = self.clamp_tau(tau)

#         packed = self._get_occ_packed(scenarios)
#         if packed is not None:
#             A_pack, b0_pack, v_expand_pack, v_adv_pack, K_counts = packed
#             A_stack = []
#             for i in range(K_counts.shape[0]):
#                 K = int(K_counts[i])
#                 if K <= 0:
#                     continue
#                 A = A_pack[i, :K, :]
#                 b0 = b0_pack[i, :K]
#                 v_expand = v_expand_pack[i, :K]
#                 vadv = float(v_adv_pack[i])

#                 delta = R + v_expand * tau
#                 h_vec = (A @ p) - b0 - delta

#                 active = (h_vec >= 0.0)
#                 if np.any(active):
#                     avg_normal = A[active].mean(axis=0)
#                     norm = np.linalg.norm(avg_normal)
#                     if norm > 1e-9:
#                         direction = avg_normal / norm
#                         v_target = direction * vadv
#                         A_stack.append(v_target)
#                 else:
#                     best_idx = int(np.argmax(h_vec))
#                     selected_normal = A[best_idx]
#                     norm = np.linalg.norm(selected_normal)
#                     if norm > 1e-9:
#                         direction = selected_normal / norm
#                         v_target = direction * vadv
#                         A_stack.append(v_target)
#         else:
#             A_stack = []
#             for sc in scenarios:
#                 A    = sc['A']
#                 b0   = sc['b0']
#                 vadv = sc['v_adv_max']

#                 if 'v_expand_vec' in sc:
#                     v_expand = sc['v_expand_vec']  # (K,)
#                 else:
#                     v_expand = np.full(len(b0), sc['v_adv_max'])

#                 delta = R + v_expand * tau
#                 h_vec = (A @ p) - b0 - delta

#                 active = (h_vec >= 0.0)
#                 if np.any(active):
#                     avg_normal = A[active].mean(axis=0)
#                     norm = np.linalg.norm(avg_normal)
#                     if norm > 1e-9:
#                         direction = avg_normal / norm
#                         v_target = direction * vadv
#                         A_stack.append(v_target)
#                 else:
#                     best_idx = np.argmax(h_vec)
#                     selected_normal = A[best_idx]
#                     norm = np.linalg.norm(selected_normal)
#                     if norm > 1e-9:
#                         direction = selected_normal / norm
#                         v_target = direction * vadv
#                         A_stack.append(v_target)

#         if len(A_stack) == 0:
#             return np.zeros(2, dtype=float)
        
#         A_all = np.vstack(A_stack)
#         # v_ref = A_all.mean(axis=0)

#         v_avg = A_all.mean(axis=0)
#         # v_norm = np.linalg.norm(v_avg)
        
#         # if v_norm > 1e-9:
#         #     max_speed = np.max([np.linalg.norm(v) for v in A_stack])
#         #     # avg_speed = np.mean([np.linalg.norm(v) for v in A_stack])
#         #     v_ref = (v_avg / v_norm) * max_speed
#         # else:
#         #     v_ref = v_avg
#         # # print(f"DEBUG: v_ref from occlusion backup: {v_ref}")
        
#         return v_avg

#     def _get_occ_packed(self, scenarios):
#         if scenarios is None or len(scenarios) == 0:
#             return None
#         cache = self._occ_cache
#         if cache is not None and cache["scenarios"] is scenarios:
#             return cache["packed"]
#         packed = _pack_occ_scenarios(scenarios)
#         self._occ_cache = {"scenarios": scenarios, "packed": packed}
#         return packed
        
#     def backup_input(self, X, k_a=1.0):
#         """
#         Using stop() function as backup policy
#         """
#         return self.stop(X, k_a=k_a)

#     def backup_input_occlusion(self, X, occlusion_scenarios, t=None,
#                             k_d=1.0, k_occ=1.0):
#         # print(f"DEBUG: backup_input_occlusion CALLED with t={t}")

#         a_lim  = float(self.robot_spec.get('a_max', 1.0))
#         v_max  = float(self.robot_spec.get('v_max', 1.0))
#         Kp = float(self.pid_occ_gains.get("Kp", 1.0))

#         v = np.array([float(X[2,0]), float(X[3,0])], dtype=float)
#         v_ref = self._occ_vel_ref(X, occlusion_scenarios, t)
        
#         e = v - v_ref
#         u_unsat = -Kp * e - k_d * v
#         u = np.clip(u_unsat, -a_lim, a_lim)

#         # limite acceleration when near v_max
#         eps = 1e-6
#         for i in range(2):
#             if v[i] >= (v_max - eps) and u[i] > 0.0: u[i] = 0.0
#             if v[i] <= (-v_max + eps) and u[i] < 0.0: u[i] = 0.0

#         return u.reshape(2,1)

#     def f_cl(self, X, occlusion_scenarios=None, t=None):
#         """
#         System dynamics as using backup policy u_b (Closed-Loop)
#         """
#         if occlusion_scenarios:
#             u_b = self.backup_input_occlusion(X, occlusion_scenarios, t)
#         else:
#             u_b = self.backup_input(X)

#         return np.array([X[2,0], X[3,0], u_b[0,0], u_b[1,0]]).reshape(4,1)
    
#     def _dvref_dp_fd(self, X, scenarios, t, eps=1e-3):
#         p = X[0:2,0].astype(float)
#         vref = self._occ_vel_ref(X, scenarios, t)
#         J = np.zeros((2,2))
#         for j in range(2):
#             Xp = X.copy(); Xp[j,0] = p[j] + eps
#             vr = self._occ_vel_ref(Xp, scenarios, t)
#             J[:,j] = (vr - vref)/eps
#         return J
    
#     def F_cl(self, X, occlusion_scenarios=None, t=None):
#         Kp = float(self.pid_occ_gains.get("Kp", 1.0))
#         k_d = 1.0
#         A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], float)
#         Bv = -(Kp + k_d) * np.eye(2)
#         if occlusion_scenarios is not None and t is not None:
#             Jp = self._dvref_dp_fd(X, occlusion_scenarios, t)
#         else:
#             Jp = np.zeros((2,2))
#         lower_left = Kp * Jp
#         F = np.block([[np.zeros((2,2)), np.eye(2)],
#                     [lower_left,      Bv      ]])
#         return F
        
#     def set_terminal_backup_context(self, occlusion_scenario, T, kappa=None, rho_T=0.05):
#         self._term_occ_scenario = occlusion_scenario
#         self._term_T = float(T)
#         if kappa is not None:
#             self.kappa = kappa
#         self._term_rho = float(rho_T)
#         self._term_grad_cache = None
    
#     def _occ_terminal_set(self, X):

#         p_T = X[0:2, 0].astype(float)

#         scenario = getattr(self, "_term_occ_scenario", None)
#         T = float(getattr(self, "_term_T", self.get_backup_horizon()))
#         rho_T = float(getattr(self, "_term_rho", 0.05))

#         if scenario is not None:
#             # print("loop in h_b_stop")
#             h_tilde, grad_pos, _ = self._occ_barrier(p_T.reshape(2, 1), scenario, tau=T)
#             # print(f"h_tilde: {h_tilde} | rho_T: {rho_T}")
#             self._term_grad_cache = (grad_pos.reshape(1,2) if grad_pos is not None else None)
#             return float(h_tilde) - rho_T

#     def grad_occ_terminal(self, X):
#         gp = getattr(self, "_term_grad_cache", None)
#         if gp is not None:
#             # print("loop in grad_h_b_stop")
#             if gp.shape == (1,2):
#                 # Expand to state dimension: [grad_pos, 0, 0]
#                 return np.hstack([gp, np.array([[0.0, 0.0]])])
#             return gp
    
#     # def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None):
#     #     """
#     #     Compute the future trajectory (phi_b) and sensitivity matrix (Phi_b, STM) by following the backup controller from the current state x0.
#     #     """
#     #     from scipy.integrate import solve_ivp

#     #     if hasattr(self, "pid_occ"):
#     #         self.pid_occ["t_prev"] = None
#     #         self.pid_occ["v_prev"] = None
        
#     #     def augmented_dynamics(t, y):
#     #         x = y[0:4]
#     #         Phi = y[4:].reshape((4, 4))
            
#     #         x_col = x.reshape(-1, 1)
#     #         x_dot = self.f_cl(x_col, occlusion_scenarios, t).flatten()
#     #         Phi_dot = self.F_cl(x_col, occlusion_scenarios, t) @ Phi
            
#     #         return np.concatenate([x_dot, Phi_dot.flatten()])

#     #     y0 = np.concatenate([x0.flatten(), np.eye(4).flatten()])
#     #     t_eval = np.arange(0.0, T + 1e-9, dt)
        
#     #     sol = solve_ivp(
#     #         augmented_dynamics,
#     #         [0, T],
#     #         y0,
#     #         t_eval=t_eval,
#     #         dense_output=True
#     #     )
        
#     #     backup_traj = sol.y[0:4, :].T       # (N,4)
#     #     stm_traj = sol.y[4:, :].T.reshape(-1, 4, 4)
        
#     #     return backup_traj, stm_traj, t_eval

#     def jac_f_cl_fd(self, X, occlusion_scenarios=None, t=None, eps=1e-4):
#         """
#         Central-difference approximation of A = ∂f_cl/∂x at (X,t).
#         X: (4,1)
#         returns: (4,4)
#         """
#         X = np.asarray(X, dtype=float).reshape(4, 1)
#         f_cl = self.f_cl
#         f0 = f_cl(X, occlusion_scenarios, t).reshape(4,)

#         J = np.zeros((4, 4), dtype=float)

#         # scale-aware eps (optional): helps when states are large/small
#         # eps_j = eps * (1.0 + abs(X[j,0]))
#         Xp = X.copy()
#         Xm = X.copy()
#         for j in range(4):
#             xj = float(X[j, 0])
#             eps_j = eps * (1.0 + abs(xj))
#             Xp[j, 0] = xj + eps_j
#             Xm[j, 0] = xj - eps_j

#             fp = f_cl(Xp, occlusion_scenarios, t).reshape(4,)
#             fm = f_cl(Xm, occlusion_scenarios, t).reshape(4,)

#             J[:, j] = (fp - fm) / (2.0 * eps_j)

#             Xp[j, 0] = xj
#             Xm[j, 0] = xj

#         return J
    
#     # def _aug_rhs(self, t, y, occlusion_scenarios=None, eps_A=1e-4):
#     #     """
#     #     RHS for augmented state y = [x; vec(Phi)].
#     #     y: (4 + 16,)
#     #     returns ydot: (4 + 16,)
#     #     """
#     #     x = y[:4].reshape(4, 1)
#     #     Phi = y[4:].reshape(4, 4)

#     #     xdot = self.f_cl(x, occlusion_scenarios, t).reshape(4,)

#     #     A = self.jac_f_cl_fd(x, occlusion_scenarios, t, eps=eps_A)  # (4,4)
#     #     Phidot = (A @ Phi).reshape(-1)

#     #     return np.concatenate([xdot, Phidot])

#     def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None, eps_A=1e-4):
#         if _NUMBA_AVAILABLE and self._use_numba:
#             A_pack, b0_pack, v_expand_pack, v_adv_pack, K_counts = _pack_occ_scenarios(occlusion_scenarios)
#             x0_v = np.asarray(x0, dtype=np.float64).reshape(4,)
#             a_lim = float(self.robot_spec.get('a_max', 1.0))
#             v_max = float(self.robot_spec.get('v_max', 1.0))
#             Kp = float(self.pid_occ_gains.get("Kp", 1.0))
#             R = float(self.robot_spec['radius'])
#             return _simulate_backup_trajectory_jit(
#                 x0_v, float(T), float(dt),
#                 A_pack, b0_pack, v_expand_pack, v_adv_pack, K_counts,
#                 a_lim, v_max, Kp, R, float(eps_A)
#             )
#         x = np.asarray(x0, float).reshape(4,1)
#         Phi = np.eye(4)
#         N = int(np.floor(T/dt)) + 1
#         t_grid = dt*np.arange(N)

#         backup_traj = np.zeros((N,4))
#         stm_traj = np.zeros((N,4,4))
#         backup_traj[0] = x.ravel()
#         stm_traj[0] = Phi
#         fcl_traj = np.zeros((N,4))

#         I = np.eye(4)
#         f_cl = self.f_cl
#         jac_fd = self.jac_f_cl_fd
#         half_dt = 0.5 * dt
#         sixth_dt = dt / 6.0

#         for k in range(N-1):
#             t = float(t_grid[k])

#             # 1) RK4 for x (NO Jacobian inside)
#             k1 = f_cl(x, occlusion_scenarios, t)
#             fcl_traj[k] = k1.ravel()
#             k2 = f_cl(x + half_dt*k1, occlusion_scenarios, t+half_dt)
#             k3 = f_cl(x + half_dt*k2, occlusion_scenarios, t+half_dt)
#             k4 = f_cl(x + dt*k3,      occlusion_scenarios, t+dt)
#             x_next = x + sixth_dt*(k1 + 2*k2 + 2*k3 + k4)

#             # 2) A once per step (use midpoint)
#             x_mid = x + half_dt*k2
#             A = jac_fd(x_mid, occlusion_scenarios, t+half_dt, eps=eps_A)

#             # 3) Phi update (cheap)
#             Phi_next = (I + dt*A) @ Phi

#             x, Phi = x_next, Phi_next
#             backup_traj[k+1] = x.ravel()
#             stm_traj[k+1] = Phi
        
#         fcl_traj[-1] = f_cl(x, occlusion_scenarios, float(t_grid[-1])).ravel()

#         return backup_traj, stm_traj, t_grid, fcl_traj

#     # def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None, eps_A=1e-4):
#     #     """
#     #     Fixed-step RK4 rollout + FD-Jacobian STM propagation.

#     #     Returns:
#     #     backup_traj: (N,4)
#     #     stm_traj:    (N,4,4)
#     #     t_grid:      (N,)
#     #     """
#     #     x0 = np.asarray(x0, dtype=float).reshape(4, 1)

#     #     # fixed time grid
#     #     N = int(np.floor(T / dt)) + 1
#     #     t_grid = dt * np.arange(N, dtype=float)
#     #     # ensure final time hits T (optional)
#     #     if abs(t_grid[-1] - T) > 1e-12:
#     #         t_grid = np.append(t_grid, T)
#     #         N = len(t_grid)

#     #     # initial augmented state
#     #     Phi0 = np.eye(4, dtype=float)
#     #     y = np.concatenate([x0.reshape(-1), Phi0.reshape(-1)])

#     #     backup_traj = np.zeros((N, 4), dtype=float)
#     #     stm_traj = np.zeros((N, 4, 4), dtype=float)

#     #     backup_traj[0, :] = x0.reshape(-1)
#     #     stm_traj[0, :, :] = Phi0

#     #     for k in range(N - 1):
#     #         t = float(t_grid[k])
#     #         h = float(t_grid[k + 1] - t_grid[k])

#     #         # RK4 on augmented system
#     #         k1 = self._aug_rhs(t,         y,               occlusion_scenarios, eps_A)
#     #         k2 = self._aug_rhs(t + 0.5*h, y + 0.5*h*k1,    occlusion_scenarios, eps_A)
#     #         k3 = self._aug_rhs(t + 0.5*h, y + 0.5*h*k2,    occlusion_scenarios, eps_A)
#     #         k4 = self._aug_rhs(t + h,     y + h*k3,        occlusion_scenarios, eps_A)

#     #         y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

#     #         xk = y[:4]
#     #         Phik = y[4:].reshape(4, 4)

#     #         backup_traj[k + 1, :] = xk
#     #         stm_traj[k + 1, :, :] = Phik

#     #     return backup_traj, stm_traj, t_grid
