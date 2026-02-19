"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Backup Controller - Abstract base class and implementations for backup control strategies.
These controllers provide alternative control behaviors that can be used for safety guarantees
or emergency maneuvers. The controllers are designed to be simple feedback controllers
that can be forward-simulated to predict trajectories.

@required-scripts: None (standalone module)
"""

import numpy as np
from abc import ABC, abstractmethod


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class BackupController(ABC):
    """
    Abstract base class for backup controllers.
    
    Backup controllers are simple feedback controllers that can be used to
    predict closed-loop trajectories for safety analysis or emergency maneuvers.
    """
    
    def __init__(self, robot_spec, dt):
        """
        Initialize the backup controller.
        
        Args:
            robot_spec: Dictionary with robot specifications
            dt: Time step for simulation
        """
        self.robot_spec = robot_spec
        self.dt = dt
        
    @abstractmethod
    def compute_control(self, state, target):
        """
        Compute control input given current state and target.
        
        Args:
            state: Current state vector (model-specific)
            target: Target for the controller (behavior-specific)
            
        Returns:
            Control input vector
        """
        pass
    
    @abstractmethod
    def simulate_trajectory(self, initial_state, target, horizon, friction=1.0):
        """
        Forward simulate the closed-loop trajectory.
        
        Args:
            initial_state: Initial state vector
            target: Target for the controller
            horizon: Number of steps to simulate
            friction: Friction coefficient for simulation
            
        Returns:
            trajectory: Array of states over the horizon (n_states x horizon)
        """
        pass
    
    def get_behavior_name(self):
        """Return the name of the backup behavior."""
        return self.__class__.__name__


class OcclusionController(BackupController):
    """
    Lane change backup controller using cascaded PD control.
    
    This controller steers the vehicle to change to a target lane Y position
    and stabilize there. Uses a cascaded control structure:
    1. Outer loop: Lateral position (y) → desired heading (theta_des)
    2. Inner loop: Heading error → desired steering angle (delta_des)
    3. Actuator loop: Steering error → steering rate (delta_dot)
    """
    def __init__(self, robot_spec, dt):
        """
        Initialize the stopping controller.
        
        Args:
            robot_spec: Dictionary with robot specifications
            dt: Time step for simulation
        """
        super().__init__(robot_spec, dt)
        if self.robot_spec.get('model') != 'DoubleIntegrator2D':
            raise NotImplementedError("OcclusionController is only implemented for DoubleIntegrator2D model.")

        self.pid_occ_gains = {
            "Kp": 1.0,
            "Ki": 0.2,
            "Kd": 0.1,
            "aw_limit": 1.0
        }
        cfg = self.robot_spec.setdefault("backup_cbf", {})
        self.T_horizon = float(cfg.get("T_horizon", 2.0))
        cfg["T_horizon"] = self.T_horizon
    
    def get_backup_horizon(self):
        return float(getattr(self, "T_horizon", 2.0))

    def backup_input(self, X, k_a=1.0):
        """
        Default backup action: stop-like damping for double integrator.
        """
        X = np.asarray(X, dtype=float).reshape(4, 1)
        vx = float(X[2, 0])
        vy = float(X[3, 0])
        return np.array([[-k_a * vx], [-k_a * vy]], dtype=float)

    def backup_input_at(self, X, scenarios=None, t=0.0):
        if scenarios is not None and len(scenarios) > 0:
            return self.backup_input_occlusion(X, scenarios, t=t)
        return self.backup_input(X)

    def compute_control(self, state, target):
        """
        BackupController interface.
        target can be:
          - dict with keys {'scenarios', 't'}
          - list/tuple of scenarios
          - None (fallback stop policy)
        """
        t = 0.0
        scenarios = None
        if isinstance(target, dict):
            scenarios = target.get("scenarios", None)
            t = float(target.get("t", 0.0))
        elif isinstance(target, (list, tuple)):
            scenarios = target
        return self.backup_input_at(state, scenarios=scenarios, t=t)

    def simulate_trajectory(self, initial_state, target, horizon, friction=1.0):
        """
        BackupController interface.
        """
        del friction
        scenarios = None
        if isinstance(target, dict):
            scenarios = target.get("scenarios", None)
        elif isinstance(target, (list, tuple)):
            scenarios = target
        T = float(horizon) * float(self.dt)
        return self.simulate_backup_trajectory(initial_state, T, self.dt, occlusion_scenarios=scenarios)

    def clamp_tau(self, tau):
        if tau is None:
            return None
        T = self.get_backup_horizon()
        tau_f = float(tau)
        return max(0.0, min(tau_f, T))

    def set_occ_barrier_fn(self, fn):
        self._occ_barrier_fn = fn

    def _occ_barrier(self, pos, scenario, tau=None):
        fn = getattr(self, "_occ_barrier_fn", None)
        if fn is None:
            raise AttributeError("Occlusion barrier function not set.")
        if not isinstance(scenario, dict):
            raise TypeError(f"scenario must be dict, got {type(scenario)}")
        
        if tau is None:
            tau = self.get_backup_horizon()
        else:
            tau = self.clamp_tau(tau)

        return fn(pos, scenario, tau)
    
    def _occ_safe_velocity_reference_rollout(self, X, scenarios, t):

        if (scenarios is None) or (len(scenarios) == 0):
            return np.zeros(2, dtype=float)
        
        p = np.array([float(X[0,0]), float(X[1,0])], dtype=float)
        v = np.array([float(X[2,0]), float(X[3,0])], dtype=float)

        a_lim = float(self.robot_spec.get('a_max', 1.0))
        R     = self.robot_spec['radius']

        A_stack =[]
        for sc in scenarios:
            A    = sc['A']
            b0   = sc['b0']
            vadv = sc['v_adv_max']

            # classify static or dynamic obstacles
            if 'v_expand_vec' in sc:
                v_expand = sc['v_expand_vec'] # Shape (K,)
            else:
                v_expand = np.full(len(b0), sc['v_adv_max'])

            tau = float(t) if t is not None else 0.0
            tau = self.clamp_tau(tau)

            delta = R + v_expand * tau
            h_vec = (A @ p) - b0 - delta

            active = (h_vec >= 0.0)  # outside wrt inflated poly
            if np.any(active):
                # calculate selected normal vectors
                avg_normal = A[active].mean(axis=0)

                # Calculate the magnitude normal vector
                norm = np.linalg.norm(avg_normal)

                if norm > 1e-9:
                    direction = avg_normal / norm

                    v_target = direction * vadv

                    A_stack.append(v_target)
            else:
                best_idx = np.argmax(h_vec)
                selected_normal = A[best_idx]
                norm = np.linalg.norm(selected_normal)
                if norm > 1e-9:
                    direction = selected_normal / norm
                    v_target = direction * vadv
                    A_stack.append(v_target)

        if len(A_stack) == 0:
            return np.zeros(2, dtype=float)
        
        A_all = np.vstack(A_stack)

        v_avg = A_all.mean(axis=0)
        
        return v_avg

    def f_cl(self, X, occlusion_scenarios=None, t=None):
        """
        System dynamics as using backup policy u_b (Closed-Loop)
        """
        if occlusion_scenarios:
            u_b = self.backup_input_occlusion(X, occlusion_scenarios, t)
        else:
            u_b = self.backup_input(X)

        return np.array([X[2,0], X[3,0], u_b[0,0], u_b[1,0]]).reshape(4,1)
    
    def _dvref_dp_fd(self, X, scenarios, t, eps=1e-3):
        p = X[0:2,0].astype(float)
        vref = self._occ_safe_velocity_reference_rollout(X, scenarios, t)
        J = np.zeros((2,2))
        for j in range(2):
            Xp = X.copy(); Xp[j,0] = p[j] + eps
            vr = self._occ_safe_velocity_reference_rollout(Xp, scenarios, t)
            J[:,j] = (vr - vref)/eps
        return J
    
    def F_cl(self, X, occlusion_scenarios=None, t=None):
        Kp = float(self.pid_occ_gains.get("Kp", 1.0))
        k_d = 1.0
        A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], float)
        Bv = -(Kp + k_d) * np.eye(2)
        if occlusion_scenarios is not None and t is not None:
            Jp = self._dvref_dp_fd(X, occlusion_scenarios, t)
        else:
            Jp = np.zeros((2,2))
        lower_left = Kp * Jp
        F = np.block([[np.zeros((2,2)), np.eye(2)],
                    [lower_left,      Bv      ]])
        return F
        
    def set_terminal_backup_context(self, occlusion_scenario, T, kappa=None, rho_T=0.05):
        self._term_occ_scenario = occlusion_scenario
        self._term_T = float(T)
        if kappa is not None:
            self.kappa = kappa
        self._term_rho = float(rho_T)
        self._term_grad_cache = None
    
    def h_b_stop(self, X):

        p_T = X[0:2, 0].astype(float)

        scenario = getattr(self, "_term_occ_scenario", None)
        T = float(getattr(self, "_term_T", self.get_backup_horizon()))
        rho_T = getattr(self, "_term_rho", None)
        if rho_T is None:
            a_lim = float(self.robot_spec.get('a_max', 1.0))
            v_max = float(self.robot_spec.get('v_max', 1.0))
            if a_lim <= 0.0:
                rho_T = 0.0
            else:
                rho_T = v_max**2 / (2.0 * a_lim)  # stopping distance
                # rho_T = 0.05  # fixed small value for terminal constraint

        if scenario is not None:
            # print("loop in h_b_stop")
            h_tilde, grad_pos, _ = self._occ_barrier(p_T.reshape(2, 1), scenario, tau=T)
            # print(f"h_tilde: {h_tilde} | rho_T: {rho_T}")
            self._term_grad_cache = (grad_pos.reshape(1,2) if grad_pos is not None else None)
            return float(h_tilde) - float(rho_T)

    def _occ_terminal_set(self, X):
        return self.h_b_stop(X)

    def grad_h_b_stop(self, X):
        gp = getattr(self, "_term_grad_cache", None)
        if gp is not None:
            # print("loop in grad_h_b_stop")
            if gp.shape == (1,2):
                # Expand to state dimension: [grad_pos, 0, 0]
                return np.hstack([gp, np.array([[0.0, 0.0]])])
            return gp

    def grad_occ_terminal(self, X):
        return self.grad_h_b_stop(X)

    def backup_input_occlusion(self, X, occlusion_scenarios, t=None,
                            k_d=1.0, k_occ=1.0):
        # print(f"DEBUG: backup_input_occlusion CALLED with t={t}")

        a_lim  = float(self.robot_spec.get('a_max', 1.0))
        v_max  = float(self.robot_spec.get('v_max', 1.0))
        Kp = float(self.pid_occ_gains.get("Kp", 1.0))

        v = np.array([float(X[2,0]), float(X[3,0])], dtype=float)
        v_ref = self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, t)
        
        e = v - v_ref
        u_unsat = -Kp * e - k_d * v
        u = np.clip(u_unsat, -a_lim, a_lim)

        # limite acceleration when near v_max
        eps = 1e-6
        for i in range(2):
            if v[i] >= (v_max - eps) and u[i] > 0.0: u[i] = 0.0
            if v[i] <= (-v_max + eps) and u[i] < 0.0: u[i] = 0.0

        return u.reshape(2,1)
    
    def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None, eps_A=1e-4):
        """
        RK4 rollout for x with a stable STM update.
        Uses a midpoint linearization F_cl and a trapezoidal (Cayley) step for Phi.
        """
        x = np.asarray(x0, float).reshape(4, 1)
        Phi = np.eye(4)
        N = int(np.floor(T / dt)) + 1
        t_grid = dt * np.arange(N)

        backup_traj = np.zeros((N, 4))
        stm_traj = np.zeros((N, 4, 4))
        backup_traj[0] = x.ravel()
        stm_traj[0] = Phi
        fcl_traj = np.zeros((N, 4))

        I = np.eye(4)

        for k in range(N - 1):
            t = float(t_grid[k])

            # RK4 for x
            k1 = self.f_cl(x, occlusion_scenarios, t)
            fcl_traj[k] = k1.ravel()
            k2 = self.f_cl(x + 0.5 * dt * k1, occlusion_scenarios, t + 0.5 * dt)
            k3 = self.f_cl(x + 0.5 * dt * k2, occlusion_scenarios, t + 0.5 * dt)
            k4 = self.f_cl(x + dt * k3, occlusion_scenarios, t + dt)
            x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # STM update: midpoint linearization + trapezoidal integration
            x_mid = x + 0.5 * dt * k2
            A_mid = self.F_cl(x_mid, occlusion_scenarios, t + 0.5 * dt)
            A_mid = np.asarray(A_mid, dtype=float)
            M1 = I - 0.5 * dt * A_mid
            M2 = I + 0.5 * dt * A_mid
            try:
                Phi_next = np.linalg.solve(M1, M2 @ Phi)
            except np.linalg.LinAlgError:
                Phi_next = (I + dt * A_mid) @ Phi

            x, Phi = x_next, Phi_next
            backup_traj[k + 1] = x.ravel()
            stm_traj[k + 1] = Phi

        fcl_traj[-1] = self.f_cl(x, occlusion_scenarios, float(t_grid[-1])).ravel()

        return backup_traj, stm_traj, t_grid, fcl_traj
