import time

import cvxpy as cp
import numpy as np

from position_control.backup_controller import OcclusionController
from position_control.ocbf.defaults import (
    OCBF_DEFAULT_BARRIER_KAPPA,
    OCBF_DEFAULT_QP_FAILURE_FALLBACK_MODE,
    OCBF_DEFAULT_ROLLOUT_MODE,
    OCBF_DEFAULT_SELECTION_MODE,
    OCBF_DEFAULT_TERMINAL_MODE,
    OCBF_DEFAULT_TERMINAL_RESIDUAL_MODE,
    OCBF_QP_FAILURE_FALLBACK_MODES,
    OCBF_ROLLOUT_MODES,
    OCBF_SELECTION_MODES,
    OCBF_TERMINAL_MODES,
    OCBF_TERMINAL_RESIDUAL_MODES,
    normalize_choice,
)
from position_control.ocbf.jax_constraints import (
    JAX_AVAILABLE as _JAX_AVAILABLE,
    jnp,
    jax_occ_constraints_kernel_di as _jax_occ_constraints_kernel_di,
    jax_occ_constraints_kernel_uni as _jax_occ_constraints_kernel_uni,
)
from safe_control.position_control.backup_cbf_qp import BackupCBFQP
from utils.occlusion import OcclusionUtils

_JAX_OCC_WARNED_MISSING = False

class OcclusionCBFQP(BackupCBFQP):
    """Backup-CBF QP with time-expanded occlusion constraints.

    The class is scenario-agnostic. Paper benchmark defaults that are specific
    to crowd2, such as state-safe QP fallback, are applied by the example and
    benchmark wrappers before this class is instantiated.
    """

    def __init__(self, robot, robot_spec, num_obs=10, kappa=OCBF_DEFAULT_BARRIER_KAPPA, ax=None):
        self.num_obs = num_obs
        kappa_cfg = robot_spec.get("occ_kappa", kappa)
        try:
            self.kappa = float(kappa_cfg)
        except Exception:
            self.kappa = float(kappa)
        if (not np.isfinite(self.kappa)) or self.kappa <= 0.0:
            self.kappa = float(kappa)
        self.occlusion_scenarios = []
        self.last_num_constraints = None
        self.last_qp_solve_time_ms = None
        self.last_solver_solve_time_ms = None
        self.last_total_compute_time_ms = None
        self.last_intervention = None
        self.last_u_ref = None
        self.last_u = None
        self.last_profile = {}
        self.last_fallback_min_h = None
        self.last_fallback_allowed = None
        self.last_fallback_cmd_max_violation = None
        self.last_fallback_cmd_feasible = None
        self.last_constraint_meta = []
        self.last_A_cbf_val = None
        self.last_b_cbf_val = None
        self.last_terminal_slack_val = None
        self.last_terminal_slack_l1 = 0.0
        self.last_terminal_slack_max = 0.0
        self.last_terminal_slack_active_count = 0
        self.last_occ_candidate_count = 0
        self.last_occ_selected_indices = []
        self.last_occ_selected_scores = []
        self.last_occ_selection_mode = None
        self.last_occ_max_active = 0
        self.last_terminal_mode = None
        self.last_terminal_candidate_count = 0
        self.last_terminal_selected_indices = []
        self.last_terminal_residual_mode = None
        self._last_qp_constraint_count = None

        self.sensing_range = float(robot_spec.get("sensing_range", 10.0))
        self.debug = bool(robot_spec.get("debug_backup_qp", False))

        cfg = robot_spec.setdefault("backup_cbf", {})
        self.T_horizon = float(cfg.get("T_horizon", 2.0))
        self.dt_backup = float(cfg.get("dt_backup", 0.05))
        alpha = float(cfg.get("alpha", 1.0))
        self.occ_rollout_mode = normalize_choice(
            cfg.get("occ_rollout_mode", OCBF_DEFAULT_ROLLOUT_MODE),
            OCBF_ROLLOUT_MODES,
            OCBF_DEFAULT_ROLLOUT_MODE,
        )
        self.terminal_slack_weight = float(cfg.get("terminal_slack_weight", 0.0))
        term_slack_max_cfg = cfg.get("terminal_slack_max", None)
        self.terminal_slack_max = None if term_slack_max_cfg is None else float(term_slack_max_cfg)
        obs_slack_max_cfg = cfg.get("obs_hocbf_slack_max", None)
        occ_slack_max_cfg = cfg.get("occ_rollout_slack_max", None)
        self.obs_hocbf_slack_max = None if obs_slack_max_cfg is None else float(obs_slack_max_cfg)
        self.occ_rollout_slack_max = None if occ_slack_max_cfg is None else float(occ_slack_max_cfg)
        self._terminal_slack_enabled = self.terminal_slack_weight > 0.0
        self.max_active_occlusions = int(cfg.get("max_active_occlusions", 0))
        if self.max_active_occlusions < 0:
            self.max_active_occlusions = 0
        self.occ_selection_mode = normalize_choice(
            cfg.get("occ_selection_mode", OCBF_DEFAULT_SELECTION_MODE),
            OCBF_SELECTION_MODES,
            OCBF_DEFAULT_SELECTION_MODE,
        )
        self.terminal_mode = normalize_choice(
            cfg.get("terminal_mode", OCBF_DEFAULT_TERMINAL_MODE),
            OCBF_TERMINAL_MODES,
            OCBF_DEFAULT_TERMINAL_MODE,
        )
        self.terminal_active_count = int(cfg.get("terminal_active_count", 0))
        if self.terminal_active_count < 0:
            self.terminal_active_count = 0
        self.terminal_residual_mode = normalize_choice(
            cfg.get("terminal_residual_mode", OCBF_DEFAULT_TERMINAL_RESIDUAL_MODE),
            OCBF_TERMINAL_RESIDUAL_MODES,
            OCBF_DEFAULT_TERMINAL_RESIDUAL_MODE,
        )
        self.terminal_visibility_reaction_margin = float(
            cfg.get("terminal_visibility_reaction_margin", 0.0)
        )
        if not np.isfinite(self.terminal_visibility_reaction_margin):
            self.terminal_visibility_reaction_margin = 0.0
        self.qp_failure_fallback_mode = normalize_choice(
            cfg.get("qp_failure_fallback_mode", OCBF_DEFAULT_QP_FAILURE_FALLBACK_MODE),
            OCBF_QP_FAILURE_FALLBACK_MODES,
            OCBF_DEFAULT_QP_FAILURE_FALLBACK_MODE,
        )
        cfg.update(
            {
                "T_horizon": self.T_horizon,
                "dt_backup": self.dt_backup,
                "alpha": alpha,
                "occ_rollout_mode": self.occ_rollout_mode,
                "max_active_occlusions": self.max_active_occlusions,
                "occ_selection_mode": self.occ_selection_mode,
                "terminal_mode": self.terminal_mode,
                "terminal_active_count": self.terminal_active_count,
                "terminal_residual_mode": self.terminal_residual_mode,
                "terminal_visibility_reaction_margin": self.terminal_visibility_reaction_margin,
                "terminal_slack_weight": self.terminal_slack_weight,
                "terminal_slack_max": self.terminal_slack_max,
                "obs_hocbf_slack_max": self.obs_hocbf_slack_max,
                "occ_rollout_slack_max": self.occ_rollout_slack_max,
                "qp_failure_fallback_mode": self.qp_failure_fallback_mode,
            }
        )
        model_name = str(robot_spec.get("model", "")).strip()
        self._model_name = model_name

        # Optional JAX acceleration for occlusion-constraint construction.
        self.use_jax_occ_constraints = bool(cfg.get("use_jax_occ_constraints", True))
        # JAX occ kernels support DI/DU(4-state) and Unicycle(3-state) rollouts.
        self._jax_occ_enabled = bool(
            self.use_jax_occ_constraints
            and _JAX_AVAILABLE
            and model_name in {"DoubleIntegrator2D", "DynamicUnicycle2D", "Unicycle2D"}
        )
        self._jax_occ_max_scenarios = int(cfg.get("jax_occ_max_scenarios", 64))
        self._jax_occ_max_halfspaces = int(cfg.get("jax_occ_max_halfspaces", 16))
        self._jax_occ_warm_n_tau = int(np.floor(self.T_horizon / self.dt_backup)) + 1
        self._jax_occ_warmed = False

        cfg["use_jax_occ_constraints"] = self.use_jax_occ_constraints
        cfg["jax_occ_max_scenarios"] = self._jax_occ_max_scenarios
        cfg["jax_occ_max_halfspaces"] = self._jax_occ_max_halfspaces

        global _JAX_OCC_WARNED_MISSING
        if self.use_jax_occ_constraints and (not _JAX_AVAILABLE) and (not _JAX_OCC_WARNED_MISSING):
            print("[OcclusionCBFQP][WARN] JAX not installed; falling back to NumPy occ-constraint build.")
            _JAX_OCC_WARNED_MISSING = True

        super().__init__(
            robot=robot,
            robot_spec=robot_spec,
            dt=self.dt_backup,
            backup_horizon=self.T_horizon,
            ax=ax,
        )
        self.alpha = alpha
        self.occlusion_backup = OcclusionController(robot_spec=robot_spec, dt=self.dt_backup)
        self.occlusion_backup.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)

        self._occ_utils = OcclusionUtils(
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=self._occlusion_barrier_smax_curved,
        )

        if self._jax_occ_enabled:
            self._warm_start_jax_occ_constraints(self._jax_occ_warm_n_tau)

        self.setup_control_problem()

        target = self.robot
        try:
            if hasattr(target, "set_occ_barrier_fn"):
                target.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)
            elif hasattr(target, "robot") and hasattr(target.robot, "set_occ_barrier_fn"):
                target.robot.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)
            else:
                # Expected for safe_control robot models without OCC-specific methods.
                pass
        except Exception as e:
            print(f"[OcclusionCBFQP][WARN] Failed to inject occ barrier callback: {e}")

    def _pack_occ_scenarios_for_jax(self, occ_for_qp):
        S = int(self._jax_occ_max_scenarios)
        K = int(self._jax_occ_max_halfspaces)

        if len(occ_for_qp) > S:
            return None

        A_pad = np.zeros((S, K, 2), dtype=np.float32)
        b0_pad = np.zeros((S, K), dtype=np.float32)
        v_expand_pad = np.zeros((S, K), dtype=np.float32)
        valid_mask = np.zeros((S, K), dtype=np.float32)
        sc_present = np.zeros((S,), dtype=bool)

        for s, sc in enumerate(occ_for_qp):
            if not isinstance(sc, dict):
                continue
            A = np.asarray(sc.get("A", []), dtype=np.float32)
            b0 = np.asarray(sc.get("b0", []), dtype=np.float32).reshape(-1)
            if A.ndim != 2 or A.shape[1] != 2:
                continue
            if A.shape[0] < 2 or b0.size < 2:
                continue
            if A.shape[0] > K or b0.size > K:
                # Preserve exact behavior: overflow falls back to NumPy path.
                return None
            m = min(int(A.shape[0]), int(b0.size))
            if m < 2:
                continue

            v_expand = sc.get("v_expand_vec", None)
            if v_expand is None:
                v_adv = float(sc.get("v_adv_max", 0.0))
                v_expand_arr = np.full((m,), v_adv, dtype=np.float32)
            else:
                v_expand_arr = np.asarray(v_expand, dtype=np.float32).reshape(-1)
                if v_expand_arr.size == 1:
                    v_expand_arr = np.full((m,), float(v_expand_arr.item()), dtype=np.float32)
                elif v_expand_arr.size != m:
                    return None

            A_pad[s, :m, :] = A[:m, :]
            b0_pad[s, :m] = b0[:m]
            v_expand_pad[s, :m] = v_expand_arr[:m]
            valid_mask[s, :m] = 1.0
            sc_present[s] = True

        return A_pad, b0_pad, v_expand_pad, valid_mask, sc_present

    def _simulate_occ_rollouts(self, robot_state, occ_for_qp, no_occ):
        if no_occ or self.occ_rollout_mode == "common":
            return {
                "common": self.occlusion_backup.simulate_backup_trajectory(
                    robot_state,
                    self.T_horizon,
                    self.dt_backup,
                    occlusion_scenarios=None if no_occ else occ_for_qp,
                ),
                "per_scenario": None,
            }

        per_scenario = []
        for scenario in occ_for_qp:
            per_scenario.append(
                self.occlusion_backup.simulate_backup_trajectory(
                    robot_state,
                    self.T_horizon,
                    self.dt_backup,
                    occlusion_scenarios=[scenario],
                )
            )
        return {"common": None, "per_scenario": per_scenario}

    def _warm_start_jax_occ_constraints(self, n_tau):
        if not self._jax_occ_enabled:
            return
        try:
            N = int(n_tau)
            S = int(self._jax_occ_max_scenarios)
            K = int(self._jax_occ_max_halfspaces)
            state_dim = 4 if self._model_name in {"DoubleIntegrator2D", "DynamicUnicycle2D"} else 3
            phi = np.zeros((N, state_dim), dtype=np.float32)
            Phi = np.zeros((N, state_dim, state_dim), dtype=np.float32)
            fcl = np.zeros((N, state_dim), dtype=np.float32)
            tau = (self.dt_backup * np.arange(N)).astype(np.float32)
            A = np.zeros((S, K, 2), dtype=np.float32)
            b0 = np.zeros((S, K), dtype=np.float32)
            v_expand = np.zeros((S, K), dtype=np.float32)
            valid = np.zeros((S, K), dtype=np.float32)
            sc_present = np.zeros((S,), dtype=bool)
            f_x = np.zeros((state_dim,), dtype=np.float32)
            g_x = np.zeros((state_dim, 2), dtype=np.float32)

            if self._model_name in {"DoubleIntegrator2D", "DynamicUnicycle2D"}:
                A_out, _, _ = _jax_occ_constraints_kernel_di(
                    jnp.asarray(phi),
                    jnp.asarray(Phi),
                    jnp.asarray(fcl),
                    jnp.asarray(tau),
                    jnp.asarray(A),
                    jnp.asarray(b0),
                    jnp.asarray(v_expand),
                    jnp.asarray(valid),
                    jnp.asarray(sc_present),
                    jnp.asarray(f_x),
                    jnp.asarray(g_x),
                    np.float32(self.kappa),
                    np.float32(self.alpha),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    int(N),
                )
            else:
                A_out, _, _ = _jax_occ_constraints_kernel_uni(
                    jnp.asarray(phi),
                    jnp.asarray(Phi),
                    jnp.asarray(fcl),
                    jnp.asarray(tau),
                    jnp.asarray(A),
                    jnp.asarray(b0),
                    jnp.asarray(v_expand),
                    jnp.asarray(valid),
                    jnp.asarray(sc_present),
                    jnp.asarray(f_x),
                    jnp.asarray(g_x),
                    np.float32(self.kappa),
                    np.float32(self.alpha),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    int(N),
                )
            _ = np.asarray(A_out[0, 0, 0])
            self._jax_occ_warm_n_tau = N
            self._jax_occ_warmed = True
        except Exception as e:
            print(f"[OcclusionCBFQP][WARN] JAX occ warm-start failed, fallback to NumPy: {e}")
            self._jax_occ_enabled = False

    def _terminal_rho(self):
        if str(self.robot_spec.get("model", "")).strip() == "Unicycle2D":
            return 0.0
        a_lim = float(self.robot_spec.get('a_max', 1.0))
        v_max = float(self.robot_spec.get('v_max', 1.0))
        if a_lim <= 0.0:
            return 0.0
        return (v_max ** 2) / (2.0 * a_lim)

    def _resolve_terminal_rho(self):
        cfg = self.robot_spec.get('backup_cbf', {})
        rho_cfg = cfg.get('rho_T', None)
        if isinstance(rho_cfg, str):
            key = rho_cfg.strip().lower()
            if key in ("auto", "auto_stop", "stop", "stopping_distance"):
                return self._terminal_rho()
            return float(rho_cfg)
        if rho_cfg is None:
            return self._terminal_rho()
        return float(rho_cfg)
    
    def _occ_smax(self, pos, scenario, tau=0.0):
        """
        Returns:
            h_tilde (float),
            grad_pos (1x2),
            lam (K,),
            dh_ds (float),
            risk_normal_vec (<=K x 2)
        """
        if scenario is None:
            return None, None, None, None, None

        A = scenario.get("A", None)
        b0 = scenario.get("b0", None)
        if A is None or b0 is None or A.shape[0] < 2:
            return None, None, None, None, None

        kappa = float(self.kappa)
        robot_radius = float(self.robot_spec["radius"])

        pos = np.asarray(pos, float).reshape(2,)
        b0 = np.asarray(b0, float).reshape(-1,)
        K = int(A.shape[0])

        v_expand = scenario.get("v_expand_vec", None)
        if v_expand is None:
            v_adv = float(scenario.get("v_adv_max", 0.0))
            v_expand = np.full((K,), v_adv, dtype=float)
        else:
            v_expand = np.asarray(v_expand, float).reshape(-1,)
            if v_expand.size != K:
                if v_expand.size == 1:
                    v_expand = np.full((K,), float(v_expand.item()), dtype=float)
                else:
                    return None, None, None, None, None

        # h_k(p,tau) = a_k^T p - b0_k - (R + v_expand_k * tau)
        R_occ = robot_radius + v_expand * float(tau)
        h_vec = (A @ pos) - b0 - R_occ
        if not np.all(np.isfinite(h_vec)):
            return None, None, None, None, None

        max_h = float(np.max(h_vec))
        z = np.exp(kappa * (h_vec - max_h))
        Z = float(np.sum(z))
        if (not np.isfinite(Z)) or Z <= 0.0:
            return None, None, None, None, None

        lam = z / Z
        h_tilde = max_h + (np.log(Z) - np.log(K)) / kappa
        grad_pos = (lam[:, None] * A).sum(axis=0, keepdims=True)
        dh_ds = float(lam @ (-v_expand))

        active = h_vec >= 0.0
        risk_normal_vec = A[active] if np.any(active) else np.empty((0, 2))
        return float(h_tilde), grad_pos, lam, dh_ds, risk_normal_vec

    def _occ_explicit_time_derivatives_pure_propagation(self, lam, scenario):
        """
        Compute partial h_C/partial t and partial h_C/partial s under the
        pure facet propagation model (a_dot=0, nu_dot=0, b_dot=nu).

        Eq. (24b):  dh_ds = -sum_l lambda_l * nu_l
        Eq. (24c):  dh_dt = -sum_l lambda_l * nu_l   (same, since b_dot=nu)
        Eq. (32):   time_residual = dh_dt - dh_ds = 0

        Returns:
            dh_dt (float), dh_ds (float)
        """
        A = scenario.get("A", None)
        K = int(A.shape[0]) if A is not None else 0
        v_expand = scenario.get("v_expand_vec", None)
        if v_expand is None:
            v_adv = float(scenario.get("v_adv_max", 0.0))
            v_expand = np.full((K,), v_adv, dtype=float)
        else:
            v_expand = np.asarray(v_expand, float).reshape(-1,)
            if v_expand.size != K:
                if v_expand.size == 1:
                    v_expand = np.full((K,), float(v_expand.item()), dtype=float)
                else:
                    raise ValueError(
                        f"v_expand size {v_expand.size} does not match K={K}"
                    )

        lam = np.asarray(lam, dtype=float).reshape(-1,)
        assert lam.shape[0] == K, f"lam length {lam.shape[0]} != K={K}"
        assert np.all(np.isfinite(lam)), "lam contains non-finite values"

        weighted_nu = float(lam @ v_expand)
        dh_ds = -weighted_nu
        dh_dt = -weighted_nu   # pure propagation: b_dot = nu => same as dh_ds
        return dh_dt, dh_ds

    def _occlusion_barrier_smax_curved(self, pos, scenario, tau=0.0):
        h_tilde, grad_pos, _, _, risk_normal_vec = self._occ_smax(pos, scenario, tau)
        if h_tilde is None:
            return None, None, None
        return h_tilde, grad_pos, risk_normal_vec

    def _build_qp_objective(self, use_visible_hocbf_objective=False):
        """
        Build QP objective.
        Use isotropic tracking cost by default. For DoubleIntegrator2D, an
        anisotropic cost can be enabled when visible-obstacle HOCBF rows are
        active, so occlusion-only constraints keep the default unit weighting.
        """
        err = self.u - self.u_ref
        model_name = str(self.robot_spec.get("model", "")).strip()
        u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))
        def _valid_weight(val, default=1.0):
            try:
                val = float(val)
            except Exception:
                return float(default)
            if (not np.isfinite(val)) or val <= 0.0:
                return float(default)
            return float(val)

        if bool(use_visible_hocbf_objective) and model_name == "DoubleIntegrator2D" and u_dim == 2:
            w_ax = _valid_weight(
                self.robot_spec.get(
                    "qp_weight_ax_visible_hocbf",
                    self.robot_spec.get("qp_weight_ax", 1.0),
                )
            )
            w_ay = _valid_weight(
                self.robot_spec.get(
                    "qp_weight_ay_visible_hocbf",
                    self.robot_spec.get("qp_weight_ay", 1.0),
                )
            )
            w_sqrt = np.array([[np.sqrt(w_ax)], [np.sqrt(w_ay)]], dtype=float)
            track_cost = cp.sum_squares(cp.multiply(w_sqrt, err))
        elif model_name == "Unicycle2D" and u_dim == 2:
            w_v = _valid_weight(self.robot_spec.get("qp_weight_v", 1.0))
            w_w = _valid_weight(self.robot_spec.get("qp_weight_w", 1.0))
            if abs(w_v - 1.0) > 1e-12 or abs(w_w - 1.0) > 1e-12:
                w_sqrt = np.array([[np.sqrt(w_v)], [np.sqrt(w_w)]], dtype=float)
                track_cost = cp.sum_squares(cp.multiply(w_sqrt, err))
            else:
                track_cost = cp.sum_squares(err)
        else:
            track_cost = cp.sum_squares(err)

        if self._terminal_slack_enabled and hasattr(self, "terminal_slack"):
            track_cost += self.terminal_slack_weight * cp.sum_squares(self.terminal_slack)
        return cp.Minimize(track_cost)

    def _rebuild_qp_problem(self, num_constraints=None, use_visible_hocbf_objective=None):
        u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))
        if not hasattr(self, "u"):
            self.u = cp.Variable((u_dim, 1))
        if not hasattr(self, "u_ref"):
            self.u_ref = cp.Parameter((u_dim, 1), value=np.zeros((u_dim, 1)))

        if num_constraints is None:
            capacity = int(max(1, getattr(self, "_max_constraints", 1)))
        else:
            capacity = int(max(1, num_constraints))
            self._max_constraints = capacity

        self.A_cbf = cp.Parameter((capacity, u_dim), value=np.zeros((capacity, u_dim)))
        self.b_cbf = cp.Parameter((capacity, 1), value=np.zeros((capacity, 1)))
        if self._terminal_slack_enabled:
            self.terminal_slack = cp.Variable((capacity, 1), nonneg=True)
            self.terminal_slack_ub = cp.Parameter((capacity, 1), value=np.zeros((capacity, 1)))
        else:
            self.terminal_slack = None
            self.terminal_slack_ub = None

        if use_visible_hocbf_objective is None:
            use_visible_hocbf_objective = bool(getattr(self, "_qp_objective_visible_hocbf", False))
        else:
            use_visible_hocbf_objective = bool(use_visible_hocbf_objective)
        self._qp_objective_visible_hocbf = use_visible_hocbf_objective
        objective = self._build_qp_objective(use_visible_hocbf_objective=use_visible_hocbf_objective)
        if self._terminal_slack_enabled:
            constraints = [
                self.A_cbf @ self.u - self.terminal_slack <= self.b_cbf,
                self.terminal_slack <= self.terminal_slack_ub,
            ]
        else:
            constraints = [self.A_cbf @ self.u <= self.b_cbf]

        ic_fn = getattr(self.robot, "input_constraints", None)
        if callable(ic_fn):
            constraints.extend(ic_fn(self.u))
        else:
            model_name = str(self.robot_spec.get("model", "")).strip()
            if model_name == "Unicycle2D" and u_dim == 2:
                v_max = float(self.robot_spec.get("v_max", 1.0))
                v_min = float(self.robot_spec.get("v_min", -v_max))
                if (not np.isfinite(v_min)) or v_min > v_max:
                    v_min = -v_max
                constraints.extend(
                    [
                        self.u[0] >= v_min,
                        self.u[0] <= v_max,
                        cp.abs(self.u[1]) <= float(self.robot_spec.get("w_max", 0.5)),
                    ]
                )
            elif model_name == "DynamicUnicycle2D" and u_dim == 2:
                constraints.extend(
                    [
                        cp.abs(self.u[0]) <= float(self.robot_spec.get("a_max", 1.0)),
                        cp.abs(self.u[1]) <= float(self.robot_spec.get("w_max", 0.5)),
                    ]
                )
            elif u_dim == 2 and "a_max" in self.robot_spec:
                constraints.extend(
                    [
                        cp.abs(self.u[0]) <= self.robot_spec["a_max"],
                        cp.abs(self.u[1]) <= self.robot_spec["a_max"],
                    ]
                )

        self.cbf_controller = cp.Problem(objective, constraints)

    def _assign_qp_data(self, A_cbf_val, b_cbf_val, terminal_slack_ub_val=None):
        num_constraints = int(np.asarray(A_cbf_val).shape[0])
        capacity = int(self.A_cbf.shape[0])
        if num_constraints > capacity:
            raise ValueError(
                f"QP buffer too small: num_constraints={num_constraints}, capacity={capacity}"
            )
        self.A_cbf.value[:, :] = 0.0
        self.b_cbf.value[:, :] = 1e6
        self.A_cbf.value[:num_constraints, :] = A_cbf_val
        self.b_cbf.value[:num_constraints, :] = b_cbf_val
        if self._terminal_slack_enabled and self.terminal_slack_ub is not None:
            self.terminal_slack_ub.value[:, :] = 0.0
            if terminal_slack_ub_val is not None:
                terminal_slack_ub_val = np.asarray(terminal_slack_ub_val, dtype=float).reshape(-1, 1)
                self.terminal_slack_ub.value[:num_constraints, :] = terminal_slack_ub_val[:num_constraints, :]

    def _terminal_slack_cap(self):
        if self.terminal_slack_max is None:
            return 1e6
        return max(float(self.terminal_slack_max), 0.0)

    def _row_slack_cap(self, cap_cfg):
        if cap_cfg is None:
            return 0.0
        return max(float(cap_cfg), 0.0)

    def _build_terminal_slack_ub(self, meta_list):
        if not self._terminal_slack_enabled:
            return None
        slack_ub = np.zeros((len(meta_list), 1), dtype=float)
        term_cap = self._terminal_slack_cap()
        obs_cap = self._row_slack_cap(self.obs_hocbf_slack_max)
        occ_cap = self._row_slack_cap(self.occ_rollout_slack_max)
        for i, meta in enumerate(meta_list):
            if not isinstance(meta, dict):
                continue
            kind = meta.get("kind")
            if kind == "terminal":
                slack_ub[i, 0] = term_cap
            elif kind == "obs_hocbf":
                slack_ub[i, 0] = obs_cap
            elif kind == "occ":
                slack_ub[i, 0] = occ_cap
        return slack_ub

    def _effective_constraint_violation(self, A_cbf_val, b_cbf_val, u_cmd, slack_val=None):
        A = np.asarray(A_cbf_val, dtype=float)
        b = np.asarray(b_cbf_val, dtype=float).reshape(-1, 1)
        u = np.asarray(u_cmd, dtype=float).reshape(-1, 1)
        violation = A @ u - b
        if slack_val is not None:
            slack_val = np.asarray(slack_val, dtype=float).reshape(-1, 1)
            if slack_val.shape[0] == violation.shape[0]:
                violation = violation - slack_val
        return violation

    def _solve_qp_with_fallbacks(self, A_cbf_val, b_cbf_val, num_constraints, timings, terminal_slack_ub_val=None):
        solve_exception = None
        solver_attempts = []

        def _status_bad(status, exc):
            if exc is not None:
                return True
            if status is None:
                return True
            return status in {
                "infeasible",
                "infeasible_inaccurate",
                "infeasible_or_unbounded",
                "unbounded",
                "unbounded_inaccurate",
                "solver_error",
            }

        def _run_attempt(solver, solver_name, *, rebuild=False, warm_start=False, extra_kwargs=None):
            nonlocal solve_exception
            extra_kwargs = dict(extra_kwargs or {})
            if rebuild:
                self._rebuild_qp_problem(num_constraints)
                t_assign0 = time.perf_counter()
                self._assign_qp_data(A_cbf_val, b_cbf_val, terminal_slack_ub_val=terminal_slack_ub_val)
                timings["param_assign_ms"] = timings.get("param_assign_ms", 0.0) + (
                    (time.perf_counter() - t_assign0) * 1000.0
                )
            exc = None
            t_attempt0 = time.perf_counter()
            try:
                self.cbf_controller.solve(solver=solver, warm_start=warm_start, verbose=False, **extra_kwargs)
            except TypeError:
                try:
                    self.cbf_controller.solve(solver=solver, verbose=False, **extra_kwargs)
                except Exception as e:
                    exc = e
            except Exception as e:
                exc = e
            solve_exception = exc
            stats = getattr(self.cbf_controller, "solver_stats", None)
            status = getattr(self.cbf_controller, "status", None)
            solver_attempts.append(
                {
                    "solver": solver_name,
                    "rebuild": bool(rebuild),
                    "status": status,
                    "exception": None if exc is None else str(exc),
                    "solve_time_s": None if stats is None else getattr(stats, "solve_time", None),
                    "setup_time_s": None if stats is None else getattr(stats, "setup_time", None),
                    "wall_ms": (time.perf_counter() - t_attempt0) * 1000.0,
                }
            )
            return status, exc

        needs_rebuild = (
            num_constraints > int(self.A_cbf.shape[0])
            or self._last_qp_constraint_count is None
            or int(self._last_qp_constraint_count) != int(num_constraints)
        )
        if needs_rebuild:
            self._rebuild_qp_problem(num_constraints)
        t_assign0 = time.perf_counter()
        self._assign_qp_data(A_cbf_val, b_cbf_val, terminal_slack_ub_val=terminal_slack_ub_val)
        timings["param_assign_ms"] = (time.perf_counter() - t_assign0) * 1000.0

        total_t0 = time.perf_counter()
        status, exc = _run_attempt(cp.OSQP, "OSQP", rebuild=False, warm_start=not needs_rebuild)
        if _status_bad(status, exc):
            status, exc = _run_attempt(cp.OSQP, "OSQP", rebuild=True, warm_start=False)
        if _status_bad(status, exc):
            status, exc = _run_attempt(
                cp.GUROBI,
                "GUROBI",
                rebuild=True,
                warm_start=False,
                extra_kwargs={"reoptimize": True},
            )
        if _status_bad(status, exc):
            status, exc = _run_attempt(
                cp.SCS,
                "SCS",
                rebuild=True,
                warm_start=False,
                extra_kwargs={"eps": 1e-5, "max_iters": 5000},
            )

        timings["qp_wall_ms"] = (time.perf_counter() - total_t0) * 1000.0
        stats = getattr(self.cbf_controller, "solver_stats", None)
        timings["solver_solve_time_s"] = getattr(stats, "solve_time", None) if stats is not None else None
        timings["solver_setup_time_s"] = getattr(stats, "setup_time", None) if stats is not None else None
        timings["solver_name"] = getattr(stats, "solver_name", None) if stats is not None else None
        timings["solver_attempts"] = solver_attempts
        self._last_qp_constraint_count = int(num_constraints)
        if not _status_bad(status, solve_exception):
            solve_exception = None
        return solve_exception

    def setup_control_problem(self):
        u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))

        if not hasattr(self, "u"):
            self.u = cp.Variable((u_dim, 1))
        if not hasattr(self, "u_ref"):
            self.u_ref = cp.Parameter((u_dim, 1), value=np.zeros((u_dim, 1)))

        N_tau = int(self.T_horizon / self.dt_backup) + 2
        max_constraints = int((2 * self.num_obs + 10) * N_tau)
        self._max_constraints = int(max_constraints)
        self._qp_objective_visible_hocbf = False
        self._rebuild_qp_problem(self._max_constraints, use_visible_hocbf_objective=False)
        self.status = None

    def set_occlusion_scenarios(self, scenarios):
        self.occlusion_scenarios = scenarios

    def _obs_type(self, obs):
        obs = np.asarray(obs, dtype=float).flatten()
        if obs.size >= 8:
            try:
                return int(obs[7])
            except Exception:
                return None
        return None

    def _dynamic_types(self):
        dyn_types_cfg = self.robot_spec.get("dynamic_obs_types", None)
        if dyn_types_cfg is None:
            return {1}
        return {int(t) for t in dyn_types_cfg}

    def _occlusion_types(self):
        occ_types_cfg = self.robot_spec.get("occlusion_types", None)
        if occ_types_cfg is None:
            return set()
        return {int(t) for t in occ_types_cfg}

    def _occ_scenario_distance(self, scenario):
        p = np.asarray(scenario.get("robot_pos", np.zeros(2)), dtype=float).reshape(2,)
        c = np.asarray(scenario.get("obs_center", np.zeros(2)), dtype=float).reshape(2,)
        r = float(scenario.get("obs_radius", 0.0))
        return float(max(0.0, np.linalg.norm(c - p) - r))

    def _select_active_occlusion_scenarios(self, robot_state, occlusion_scenarios):
        scenarios = list(occlusion_scenarios or [])
        candidate_count = len(scenarios)
        self.last_occ_candidate_count = int(candidate_count)
        self.last_occ_selection_mode = self.occ_selection_mode
        self.last_occ_max_active = int(self.max_active_occlusions)
        if candidate_count == 0:
            self.last_occ_selected_indices = []
            self.last_occ_selected_scores = []
            return scenarios

        max_active = int(self.max_active_occlusions)
        if max_active <= 0 or candidate_count <= max_active:
            self.last_occ_selected_indices = list(range(candidate_count))
            self.last_occ_selected_scores = [None] * candidate_count
            return scenarios

        try:
            pos_now = np.asarray(robot_state, dtype=float).reshape(-1, 1)[0:2, 0]
        except Exception:
            pos_now = np.asarray(robot_state, dtype=float).reshape(-1)[:2]

        scored = []
        for idx, scenario in enumerate(scenarios):
            dist = self._occ_scenario_distance(scenario)
            if self.occ_selection_mode == "distance":
                score = dist
            else:
                h_occ_now, _, _, _, _ = self._occ_smax(pos_now, scenario, tau=0.0)
                score = float(h_occ_now) if h_occ_now is not None and np.isfinite(h_occ_now) else float("inf")
            scored.append((float(score), float(dist), int(idx), scenario))

        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        selected = scored[:max_active]
        self.last_occ_selected_indices = [idx for _, _, idx, _ in selected]
        self.last_occ_selected_scores = [score for score, _, _, _ in selected]
        return [scenario for _, _, _, scenario in selected]

    def _rank_terminal_scenarios(self, robot_state, scenarios):
        try:
            pos_now = np.asarray(robot_state, dtype=float).reshape(-1, 1)[0:2, 0]
        except Exception:
            pos_now = np.asarray(robot_state, dtype=float).reshape(-1)[:2]

        ranked = []
        for idx, scenario in enumerate(list(scenarios or [])):
            dist = self._occ_scenario_distance(scenario) if isinstance(scenario, dict) else float("inf")
            h_occ_now, _, _, _, _ = self._occ_smax(pos_now, scenario, tau=0.0)
            score = float(h_occ_now) if h_occ_now is not None and np.isfinite(h_occ_now) else float("inf")
            ranked.append((score, dist, int(idx), scenario))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        return ranked

    def _select_terminal_scenarios(self, robot_state, scenarios):
        scenarios = list(scenarios or [])
        self.last_terminal_mode = self.terminal_mode
        self.last_terminal_residual_mode = self.terminal_residual_mode
        self.last_terminal_candidate_count = int(len(scenarios))
        self.last_terminal_selected_indices = []
        if not scenarios or self.terminal_mode == "none":
            return []

        if self.terminal_mode == "all":
            self.last_terminal_selected_indices = list(range(len(scenarios)))
            return [(idx, scenario) for idx, scenario in enumerate(scenarios)]

        if self.terminal_mode == "dominant":
            count = 1
        else:
            count = int(self.terminal_active_count)
            if count <= 0:
                count = len(scenarios)
        count = int(np.clip(count, 0, len(scenarios)))
        ranked = self._rank_terminal_scenarios(robot_state, scenarios)
        selected = ranked[:count]
        self.last_terminal_selected_indices = [idx for _, _, idx, _ in selected]
        return [(idx, scenario) for _, _, idx, scenario in selected]

    def _build_visibility_terminal_scenario(self, term_scn, phi_T):
        if not isinstance(term_scn, dict):
            return None
        obs_state = term_scn.get("source_obs_state", None)
        obs_vel = term_scn.get("source_obs_vel", None)
        if obs_state is None or obs_vel is None:
            return None
        try:
            obs_pred = np.asarray(obs_state, dtype=float).reshape(-1).copy()
            vel = np.asarray(obs_vel, dtype=float).reshape(2,)
        except Exception:
            return None
        if obs_pred.size < 3 or not np.all(np.isfinite(obs_pred[:3])) or not np.all(np.isfinite(vel)):
            return None
        obs_pred[0:2] = obs_pred[0:2] + vel * float(self.T_horizon)

        robot_pred = np.asarray(phi_T, dtype=float).reshape(-1, 1).copy()
        if robot_pred.shape[0] < 2:
            return None
        obs_type = int(obs_pred[7]) if obs_pred.size >= 8 else None
        try:
            pred_scn = self._occ_utils._build_occlusion_scenario(
                robot_pred,
                obs_pred,
                is_static=(obs_type == 0),
            )
        except Exception:
            return None
        if pred_scn is None:
            return None

        dyn_A = np.asarray(pred_scn.get("A", []), dtype=float)
        dyn_b0 = np.asarray(pred_scn.get("b0", []), dtype=float).reshape(-1)
        if dyn_A.ndim != 2 or dyn_A.shape[1] != 2 or dyn_A.shape[0] < 2 or dyn_b0.size < 2:
            return None

        dyn_m = min(int(dyn_A.shape[0]), int(dyn_b0.size))
        dyn_A = dyn_A[:dyn_m, :]
        dyn_b0 = dyn_b0[:dyn_m].copy()
        if self.terminal_visibility_reaction_margin > 0.0:
            dyn_b0 = dyn_b0 + float(self.terminal_visibility_reaction_margin)
        dyn_v = np.zeros((dyn_m,), dtype=float)

        out = dict(pred_scn)
        out["A"] = dyn_A
        out["b0"] = dyn_b0
        out["v_expand_vec"] = dyn_v
        out["v_adv_max"] = float(term_scn.get("v_adv_max", pred_scn.get("v_adv_max", 0.0)))
        out["terminal_residual_mode"] = self.terminal_residual_mode
        out["source_visible_index"] = term_scn.get("source_visible_index", None)
        out["source_obs_state"] = obs_pred
        out["source_obs_vel"] = vel
        return out

    def _build_terminal_constraint_scenario(self, term_scn, phi_T):
        if self.terminal_residual_mode == "off":
            return term_scn

        dyn_scn = self._build_visibility_terminal_scenario(term_scn, phi_T)
        if dyn_scn is None:
            return None
        if self.terminal_residual_mode == "visibility_only":
            return dyn_scn

        try:
            A0 = np.asarray(term_scn.get("A", []), dtype=float)
            b0 = np.asarray(term_scn.get("b0", []), dtype=float).reshape(-1)
            A1 = np.asarray(dyn_scn.get("A", []), dtype=float)
            b1 = np.asarray(dyn_scn.get("b0", []), dtype=float).reshape(-1)
        except Exception:
            return dyn_scn
        if A0.ndim != 2 or A0.shape[1] != 2 or A0.shape[0] < 2 or b0.size < 2:
            return dyn_scn
        if A1.ndim != 2 or A1.shape[1] != 2 or A1.shape[0] < 2 or b1.size < 2:
            return term_scn

        m0 = min(int(A0.shape[0]), int(b0.size))
        m1 = min(int(A1.shape[0]), int(b1.size))
        v0 = term_scn.get("v_expand_vec", None)
        if v0 is None:
            v0_arr = np.full((m0,), float(term_scn.get("v_adv_max", 0.0)), dtype=float)
        else:
            v0_arr = np.asarray(v0, dtype=float).reshape(-1)
            if v0_arr.size == 1:
                v0_arr = np.full((m0,), float(v0_arr.item()), dtype=float)
            elif v0_arr.size != m0:
                v0_arr = np.full((m0,), float(term_scn.get("v_adv_max", 0.0)), dtype=float)
        residual_v_expand = np.zeros((m1,), dtype=float)

        residual = dict(term_scn)
        residual["A"] = np.vstack([A0[:m0, :], A1[:m1, :]])
        residual["b0"] = np.concatenate([b0[:m0], b1[:m1]])
        residual["v_expand_vec"] = np.concatenate([v0_arr[:m0], residual_v_expand])
        residual["terminal_residual_mode"] = self.terminal_residual_mode
        residual["terminal_visibility_scenario"] = dyn_scn
        return residual

    def _attach_occlusion_score_context(self, robot_state, scenarios, visible_obs):
        del robot_state
        visible = list(visible_obs or [])
        enriched = []
        for scenario in list(scenarios or []):
            if not isinstance(scenario, dict):
                enriched.append(scenario)
                continue
            sc = dict(scenario)
            src_idx = sc.get("source_visible_index", None)
            obs_vel = None
            if src_idx is not None:
                try:
                    obs = np.asarray(visible[int(src_idx)], dtype=float).reshape(-1)
                    sc["source_obs_state"] = obs.copy()
                    if obs.size >= 5 and np.all(np.isfinite(obs[3:5])):
                        obs_vel = obs[3:5].astype(float)
                except Exception:
                    obs_vel = None
            if obs_vel is None:
                obs_vel = np.zeros(2, dtype=float)
            sc["source_obs_vel"] = np.asarray(obs_vel, dtype=float).reshape(2,)
            enriched.append(sc)
        return enriched

    def _is_dynamic_obs_for_di(self, obs):
        obs = np.asarray(obs, dtype=float).flatten()
        obs_type = self._obs_type(obs)
        dyn_types = self._dynamic_types()
        if obs_type is not None:
            return obs_type in dyn_types
        if bool(self.robot_spec.get("dynamic_obs_from_velocity", False)) and obs.size >= 5:
            return float(np.hypot(obs[3], obs[4])) > 1e-9
        return False

    def _build_di_hocbf_constraint(self, robot_state, obs, mode):
        """
        Build one HOCBF inequality row for DoubleIntegrator2D in the form:
            A_row u <= b_row
        compatible with the OcclusionCBFQP stacked QP.
        """
        dt = float(self.robot.dt)
        try:
            f_x = self.robot.f(robot_state)
        except TypeError:
            f_x = self.robot.f()
        try:
            g_x = self.robot.g(robot_state)
        except TypeError:
            g_x = self.robot.g()

        h_dot_t = 0.0
        if self._is_dynamic_obs_for_di(obs):
            dyn_barrier = getattr(self.robot.robot, "dynamic_agent_barrier", None)
            if callable(dyn_barrier):
                res = dyn_barrier(robot_state, obs, self.robot.robot_radius)
            else:
                res = self.robot.agent_barrier(obs)
        else:
            res = self.robot.agent_barrier(obs)

        if isinstance(res, (tuple, list)) and len(res) >= 4:
            h, h_dot, dh_dot_dx, h_dot_t = res[:4]
        else:
            h, h_dot, dh_dot_dx = res[:3]

        Lfh = float((dh_dot_dx @ f_x).item()) + float(h_dot_t)
        Lgh = dh_dot_dx @ g_x

        if mode == "hard":
            rhs = float(h) / (dt ** 2) + 2.0 * float(h_dot) / dt + Lfh
        else:
            alpha1 = float(self.robot_spec.get("hocbf_alpha1", 1.0))
            alpha2 = float(self.robot_spec.get("hocbf_alpha2", 1.0))
            gamma1 = alpha1 + alpha2
            gamma2 = alpha1 * alpha2
            rhs = Lfh + gamma1 * float(h_dot) + gamma2 * float(h)

        return -Lgh, rhs, float(h)

    def _build_visible_obs_constraint(self, robot_state, obs, mode):
        """
        Build one visible-obstacle CBF/HOCBF inequality row using the same
        dynamic-obstacle barrier selection logic as the baseline CBF-QP.
        """
        model_name = str(self.robot_spec.get("model", "")).strip()
        if model_name == "DoubleIntegrator2D":
            return self._build_di_hocbf_constraint(robot_state, obs, mode)

        try:
            f_x = self.robot.f(robot_state)
        except TypeError:
            f_x = self.robot.f()
        try:
            g_x = self.robot.g(robot_state)
        except TypeError:
            g_x = self.robot.g()

        dt = float(self.robot.dt)

        if model_name in {"SingleIntegrator2D", "Unicycle2D", "KinematicBicycle2D_C3BF", "KinematicBicycle2D_DPCBF", "Quad3D"}:
            h_t = 0.0
            dyn_barrier = getattr(self.robot.robot, "dynamic_agent_barrier", None)
            if callable(dyn_barrier) and self._is_dynamic_obs_for_di(obs):
                res = dyn_barrier(robot_state, obs, self.robot.robot_radius)
                if isinstance(res, (tuple, list)) and len(res) >= 3:
                    h, dh_dx, h_t = res[:3]
                else:
                    h, dh_dx = self.robot.agent_barrier(obs)
            else:
                h, dh_dx = self.robot.agent_barrier(obs)

            A_row = -(dh_dx @ g_x)
            if mode == "hard":
                b_row = float(h) / dt + float((dh_dx @ f_x).item()) + float(h_t)
            else:
                alpha_cbf = float(self.robot_spec.get("cbf_alpha", 1.5))
                b_row = float((dh_dx @ f_x).item()) + float(h_t) + alpha_cbf * float(h)
            return A_row, b_row, float(h)

        if model_name in {"DynamicUnicycle2D", "KinematicBicycle2D", "Quad2D"}:
            h_dot_t = 0.0
            dyn_barrier = getattr(self.robot.robot, "dynamic_agent_barrier", None)
            if callable(dyn_barrier) and self._is_dynamic_obs_for_di(obs):
                res = dyn_barrier(robot_state, obs, self.robot.robot_radius)
            else:
                res = self.robot.agent_barrier(obs)

            if isinstance(res, (tuple, list)) and len(res) >= 4:
                h, h_dot, dh_dot_dx, h_dot_t = res[:4]
            else:
                h, h_dot, dh_dot_dx = res[:3]

            A_row = -(dh_dot_dx @ g_x)
            if mode == "hard":
                b_row = float(h) / (dt ** 2) + 2.0 * float(h_dot) / dt + float((dh_dot_dx @ f_x).item()) + float(h_dot_t)
            else:
                alpha1 = float(self.robot_spec.get("hocbf_alpha1", 1.5))
                alpha2 = float(self.robot_spec.get("hocbf_alpha2", 1.5))
                gamma1 = alpha1 + alpha2
                gamma2 = alpha1 * alpha2
                b_row = float((dh_dot_dx @ f_x).item()) + float(h_dot_t) + gamma1 * float(h_dot) + gamma2 * float(h)
            return A_row, b_row, float(h)

        raise ValueError(f"Unsupported visible obstacle barrier model `{model_name}`.")

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
            "alpha": float(cfg.get("alpha", 1.0)),
            "alpha1": float(cfg.get("alpha1", self.robot_spec.get("hocbf_alpha1", 1.0))),
            "alpha2": float(cfg.get("alpha2", self.robot_spec.get("hocbf_alpha2", 1.0))),
        }

    def _build_position_corridor_constraints(self, robot_state):
        cfg = self._get_position_corridor()
        if cfg is None:
            return []

        X = np.asarray(robot_state, dtype=float).reshape(-1, 1)
        model_name = str(self.robot_spec.get("model", "")).strip()
        rows = []

        def _append_row(A_row, b_row, side):
            A_row = np.asarray(A_row, dtype=float).reshape(1, -1)
            b_row = float(b_row)
            if np.all(np.isfinite(A_row)) and np.isfinite(b_row):
                rows.append((A_row, np.array([[b_row]]), {"kind": "corridor", "axis": "x", "side": side}))

        if model_name == "DoubleIntegrator2D":
            x = float(X[0, 0])
            vx = float(X[2, 0])
            gamma1 = float(cfg["alpha1"]) + float(cfg["alpha2"])
            gamma2 = float(cfg["alpha1"]) * float(cfg["alpha2"])

            h_left = x - float(cfg["x_min"])
            h_dot_left = vx
            _append_row(np.array([[-1.0, 0.0]]), gamma1 * h_dot_left + gamma2 * h_left, "xmin")

            h_right = float(cfg["x_max"]) - x
            h_dot_right = -vx
            _append_row(np.array([[1.0, 0.0]]), gamma1 * h_dot_right + gamma2 * h_right, "xmax")
            return rows

        if model_name == "Unicycle2D":
            theta = float(X[2, 0])
            c = float(np.cos(theta))
            alpha = float(cfg["alpha"])

            h_left = float(X[0, 0]) - float(cfg["x_min"])
            _append_row(np.array([[-c, 0.0]]), alpha * h_left, "xmin")

            h_right = float(cfg["x_max"]) - float(X[0, 0])
            _append_row(np.array([[c, 0.0]]), alpha * h_right, "xmax")
            return rows

        if model_name == "DynamicUnicycle2D":
            x = float(X[0, 0])
            theta = float(X[2, 0])
            v = float(X[3, 0])
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            gamma1 = float(cfg["alpha1"]) + float(cfg["alpha2"])
            gamma2 = float(cfg["alpha1"]) * float(cfg["alpha2"])

            h_left = x - float(cfg["x_min"])
            h_dot_left = v * c
            Lgh_left = np.array([[c, -v * s]], dtype=float)
            _append_row(-Lgh_left, gamma1 * h_dot_left + gamma2 * h_left, "xmin")

            h_right = float(cfg["x_max"]) - x
            h_dot_right = -v * c
            Lgh_right = np.array([[-c, v * s]], dtype=float)
            _append_row(-Lgh_right, gamma1 * h_dot_right + gamma2 * h_right, "xmax")
            return rows

        return []

    def _fallback_h_tol(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        if "fallback_h_tol" in cfg:
            return float(cfg["fallback_h_tol"])
        return float(self.robot_spec.get("fallback_h_tol", 0.0))

    def _is_fallback_safe(self, min_barrier_now, has_barrier_value):
        if not has_barrier_value:
            return True
        if min_barrier_now is None or not np.isfinite(min_barrier_now):
            return True
        return float(min_barrier_now) >= -self._fallback_h_tol()

    def _fallback_cmd_feasible(self, u_cmd, A_cbf_val, b_cbf_val, tol_feas, slack_ub_val=None):
        try:
            u = np.asarray(u_cmd, dtype=float).reshape(-1, 1)
            A = np.asarray(A_cbf_val, dtype=float)
            b = np.asarray(b_cbf_val, dtype=float).reshape(-1, 1)
            if A.ndim != 2 or b.shape[0] != A.shape[0] or u.shape[0] != A.shape[1]:
                return False, None
            violation = self._effective_constraint_violation(A, b, u, slack_val=slack_ub_val)
            max_violation = float(np.max(violation)) if violation.size > 0 else -np.inf
            return bool(max_violation <= float(tol_feas)), max_violation
        except Exception:
            return False, None

    def _clip_reference_input(self, robot_state, u_val):
        model_name = str(self.robot_spec.get("model", "")).strip()
        u = np.asarray(u_val, dtype=float).reshape(-1, 1).copy()
        if u.shape[0] < 2:
            return u

        if model_name == "Unicycle2D":
            v_max = float(self.robot_spec.get("v_max", 1.0))
            v_min = float(self.robot_spec.get("v_min", -v_max))
            if not np.isfinite(v_min):
                v_min = -v_max
            if v_min > v_max:
                v_min = v_max
            w_max = float(self.robot_spec.get("w_max", 0.5))
            u[0, 0] = float(np.clip(u[0, 0], v_min, v_max))
            u[1, 0] = float(np.clip(u[1, 0], -w_max, w_max))
            return u

        if model_name != "DynamicUnicycle2D":
            return u

        try:
            v_cur = float(np.asarray(robot_state, dtype=float).reshape(-1, 1)[3, 0])
        except Exception:
            return u

        dt = float(getattr(self.robot, "dt", self.dt_backup))
        dt = max(dt, 1e-6)
        a_max = float(self.robot_spec.get("a_max", 1.0))
        w_max = float(self.robot_spec.get("w_max", 0.5))
        v_max = float(self.robot_spec.get("v_max", 1.0))
        v_min = float(self.robot_spec.get("v_min", -v_max))
        if not np.isfinite(v_min):
            v_min = -v_max
        if v_min > v_max:
            v_min = v_max

        a_lb = max(-a_max, (v_min - v_cur) / dt)
        a_ub = min(a_max, (v_max - v_cur) / dt)
        if a_lb > a_ub:
            a_mid = 0.5 * (a_lb + a_ub)
            a_lb, a_ub = a_mid, a_mid

        u[0, 0] = float(np.clip(u[0, 0], a_lb, a_ub))
        u[1, 0] = float(np.clip(u[1, 0], -w_max, w_max))
        return u

    def _clip_du_input_for_speed(self, robot_state, u_val):
        return self._clip_reference_input(robot_state, u_val)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        t_all0 = time.perf_counter()
        timings = {}

        bounded_u_ref = self._clip_reference_input(robot_state, control_ref["u_ref"])
        self.u_ref.value = bounded_u_ref
        self.last_u_ref = np.array(self.u_ref.value, dtype=float).reshape(-1, 1)

        t0 = time.perf_counter()
        disable_occ_constraints = bool(self.robot_spec.get("disable_occ_constraints", False))
        occ_types = self._occlusion_types()

        if obs_list is None:
            visible_obs, occlusion_scenarios = [], []
        else:
            obs_arr = np.asarray(obs_list, dtype=float)
            if obs_arr.size == 0:
                visible_obs, occlusion_scenarios = [], []
            else:
                # Visibility filtering (including occlusion geometry) is always active.
                # `disable_occ_constraints` only disables adding occlusion-CBF rows.
                visible_obs, occlusion_scenarios = self._occ_utils._filter_visible_and_build_occ(
                    robot_state, obs_list
                )
        if self.num_obs > 0:
            occlusion_scenarios = occlusion_scenarios[: self.num_obs]
            if not bool(self.robot_spec.get("enable_visible_hocbf_in_occ", False)):
                visible_obs = visible_obs[: self.num_obs]
        occ_for_qp = [] if disable_occ_constraints else self._select_active_occlusion_scenarios(
            robot_state, occlusion_scenarios
        )
        occ_for_qp = self._attach_occlusion_score_context(robot_state, occ_for_qp, visible_obs)
        self.occlusion_scenarios = occ_for_qp

        enable_visible_hocbf = bool(self.robot_spec.get("enable_visible_hocbf_in_occ", False))
        selected_visible_idx = {
            int(sc.get("source_visible_index"))
            for sc in occ_for_qp
            if isinstance(sc, dict) and sc.get("source_visible_index", None) is not None
        }
        if disable_occ_constraints:
            hocbf_obs = list(visible_obs)
        elif enable_visible_hocbf:
            hocbf_obs = list(visible_obs)
        else:
            # Keep visible-obstacle HOCBF for all visible obstacles except the
            # selected active occluders, which are already handled by the
            # occlusion rollout / terminal rows and occlusion-derived v_ref.
            hocbf_obs = [
                obs for obs_idx, obs in enumerate(visible_obs)
                if int(obs_idx) not in selected_visible_idx
            ]
        use_visible_hocbf_objective = len(hocbf_obs) > 0
        if use_visible_hocbf_objective != bool(getattr(self, "_qp_objective_visible_hocbf", False)):
            self._rebuild_qp_problem(
                int(max(1, getattr(self, "_max_constraints", 1))),
                use_visible_hocbf_objective=use_visible_hocbf_objective,
            )
            self._last_qp_constraint_count = None

        corridor_rows = self._build_position_corridor_constraints(robot_state)

        no_obs = len(hocbf_obs) == 0
        no_occ = len(occ_for_qp) == 0
        no_corridor = len(corridor_rows) == 0
        min_barrier_now = np.inf
        has_barrier_value = False

        if not no_occ:
            try:
                pos_now = np.asarray(robot_state, dtype=float).reshape(-1, 1)[0:2, 0]
            except Exception:
                pos_now = np.asarray(robot_state, dtype=float).reshape(-1)[:2]
            for scenario in occ_for_qp:
                h_occ_now, _, _, _, _ = self._occ_smax(pos_now, scenario, tau=0.0)
                if h_occ_now is None:
                    continue
                h_occ_now = float(h_occ_now)
                if np.isfinite(h_occ_now):
                    min_barrier_now = min(min_barrier_now, h_occ_now)
                    has_barrier_value = True
            try:
                vref_debug = self.occlusion_backup.occ_vref_scenario_weight_debug(
                    robot_state, occ_for_qp, t=0.0
                )
            except Exception:
                vref_debug = None
            if isinstance(vref_debug, dict):
                timings["occ_vref_weight_mode"] = vref_debug.get("weight_mode")
                timings["occ_vref_avg_unexpanded_margin"] = vref_debug.get("avg_unexpanded_margin")
                timings["occ_vref_max_softmax_weight"] = vref_debug.get("max_softmax_weight")
                timings["occ_vref_scenario_debug"] = vref_debug.get("scenarios", [])
        timings["filter_occ_ms"] = (time.perf_counter() - t0) * 1000.0

        if no_obs and no_occ and no_corridor:
            self.status = "optimal"
            self.last_num_constraints = 0
            self.last_qp_solve_time_ms = 0.0
            self.last_solver_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_intervention = "u_ref"
            u_out = self._clip_du_input_for_speed(robot_state, self.u_ref.value)
            self.last_u = u_out
            return u_out

        A_list, b_list = [], []
        meta_list = []

        t0 = time.perf_counter()
        rollout_data = self._simulate_occ_rollouts(robot_state, occ_for_qp, no_occ)
        common_rollout = rollout_data["common"]
        per_scenario_rollouts = rollout_data["per_scenario"]
        if common_rollout is not None:
            phi_b, Phi_b, tau_points, fcl_traj = common_rollout
        else:
            phi_b = Phi_b = fcl_traj = None
            tau_points = (
                per_scenario_rollouts[0][2]
                if per_scenario_rollouts and len(per_scenario_rollouts) > 0
                else np.zeros((0,), dtype=float)
            )
        timings["rollout_stm_ms"] = (time.perf_counter() - t0) * 1000.0

        try:
            f_x = self.robot.f(robot_state)
        except TypeError:
            f_x = self.robot.f()
        try:
            g_x = self.robot.g(robot_state)
        except TypeError:
            g_x = self.robot.g()

        # HOCBF constraints from non-occlusion obstacles
        t0 = time.perf_counter()
        if not no_obs:
            mode = str(self.robot_spec.get("cbf_mode", "cbf")).strip().lower()
            for obs_idx, obs in enumerate(hocbf_obs):
                obs = np.asarray(obs, dtype=float).flatten()
                try:
                    A_row, b_row, h_now = self._build_visible_obs_constraint(robot_state, obs, mode)
                except Exception:
                    continue
                if np.isfinite(h_now):
                    min_barrier_now = min(min_barrier_now, float(h_now))
                    has_barrier_value = True

                if np.all(np.isfinite(A_row)) and np.isfinite(b_row):
                    A_list.append(A_row)
                    b_list.append(np.array([[b_row]]))
                    meta_list.append({"kind": "obs_hocbf", "obs_idx": obs_idx})
        timings["build_obs_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if not no_corridor:
            for A_row, b_row, meta in corridor_rows:
                A_list.append(A_row)
                b_list.append(b_row)
                meta_list.append(dict(meta))
        timings["build_corridor_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        # occlusion constraints over backup trajectory
        t0 = time.perf_counter()
        if not no_occ:
            used_jax_occ = False
            if (
                self.occ_rollout_mode == "common"
                and self._jax_occ_enabled
                and self._jax_occ_warmed
                and len(tau_points) == self._jax_occ_warm_n_tau
            ):
                packed = self._pack_occ_scenarios_for_jax(occ_for_qp)
                if packed is not None:
                    try:
                        phi_arr = np.asarray(phi_b, dtype=np.float32)
                        Phi_arr = np.asarray(Phi_b, dtype=np.float32)
                        fcl_arr = np.asarray(fcl_traj, dtype=np.float32)
                        tau_arr = np.asarray(tau_points, dtype=np.float32).reshape(-1)
                        f_x_vec = np.asarray(f_x, dtype=np.float32).reshape(-1)
                        g_x_mat = np.asarray(g_x, dtype=np.float32)
                        state_dim = 4 if self._model_name in {"DoubleIntegrator2D", "DynamicUnicycle2D"} else 3

                        shape_ok = (
                            phi_arr.ndim == 2 and phi_arr.shape[1] == state_dim
                            and Phi_arr.ndim == 3 and Phi_arr.shape[1:] == (state_dim, state_dim)
                            and fcl_arr.ndim == 2 and fcl_arr.shape[1] == state_dim
                            and f_x_vec.shape[0] == state_dim
                            and g_x_mat.shape == (state_dim, 2)
                        )
                    except Exception:
                        shape_ok = False

                    if shape_ok:
                        A_pad, b0_pad, v_expand_pad, valid_mask, sc_present = packed
                        try:
                            if self._model_name in {"DoubleIntegrator2D", "DynamicUnicycle2D"}:
                                A_occ, b_occ, keep_mask = _jax_occ_constraints_kernel_di(
                                    jnp.asarray(phi_arr),
                                    jnp.asarray(Phi_arr),
                                    jnp.asarray(fcl_arr),
                                    jnp.asarray(tau_arr),
                                    jnp.asarray(A_pad),
                                    jnp.asarray(b0_pad),
                                    jnp.asarray(v_expand_pad),
                                    jnp.asarray(valid_mask),
                                    jnp.asarray(sc_present),
                                    jnp.asarray(f_x_vec),
                                    jnp.asarray(g_x_mat),
                                    np.float32(self.kappa),
                                    np.float32(self.alpha),
                                    np.float32(self.robot_spec.get("radius", 0.25)),
                                    int(len(tau_arr)),
                                )
                            else:
                                A_occ, b_occ, keep_mask = _jax_occ_constraints_kernel_uni(
                                    jnp.asarray(phi_arr),
                                    jnp.asarray(Phi_arr),
                                    jnp.asarray(fcl_arr),
                                    jnp.asarray(tau_arr),
                                    jnp.asarray(A_pad),
                                    jnp.asarray(b0_pad),
                                    jnp.asarray(v_expand_pad),
                                    jnp.asarray(valid_mask),
                                    jnp.asarray(sc_present),
                                    jnp.asarray(f_x_vec),
                                    jnp.asarray(g_x_mat),
                                    np.float32(self.kappa),
                                    np.float32(self.alpha),
                                    np.float32(self.robot_spec.get("radius", 0.25)),
                                    int(len(tau_arr)),
                                )

                            A_occ_np = np.asarray(A_occ, dtype=float)
                            b_occ_np = np.asarray(b_occ, dtype=float)
                            keep_np = np.asarray(keep_mask, dtype=bool)

                            if np.any(keep_np):
                                idx = np.argwhere(keep_np)
                                A_valid = A_occ_np[idx[:, 0], idx[:, 1], :]
                                b_valid = b_occ_np[idx[:, 0], idx[:, 1]].reshape(-1, 1)
                                A_list.append(A_valid)
                                b_list.append(b_valid)
                                meta_list.extend([{"kind": "occ"}] * int(idx.shape[0]))
                            used_jax_occ = True
                        except Exception:
                            used_jax_occ = False

            if not used_jax_occ:
                for sc_idx, scenario in enumerate(occ_for_qp):
                    if per_scenario_rollouts is not None:
                        phi_sc, Phi_sc, tau_points_sc, fcl_sc = per_scenario_rollouts[sc_idx]
                    else:
                        phi_sc, Phi_sc, tau_points_sc, fcl_sc = phi_b, Phi_b, tau_points, fcl_traj
                    for i, tau in enumerate(tau_points_sc):
                        phi_i = phi_sc[i].reshape(-1, 1)
                        Phi_i = Phi_sc[i]
                        pos_i = phi_i[0:2]

                        h_tilde, grad_pos, lam, dh_ds_from_barrier, _ = self._occ_smax(
                            pos_i.flatten(), scenario, tau
                        )
                        if h_tilde is None or grad_pos is None:
                            continue

                        # Eq. (24b)/(24c): compute dh_dt and dh_ds explicitly
                        dh_dt, dh_ds = self._occ_explicit_time_derivatives_pure_propagation(
                            lam, scenario
                        )

                        if self.debug:
                            if not np.isclose(
                                dh_ds,
                                float(dh_ds_from_barrier),
                                rtol=1e-7,
                                atol=1e-9,
                            ):
                                raise RuntimeError(
                                    "Inconsistent occlusion derivative: "
                                    f"dh_ds from helper ({dh_ds:.6g}) does not match "
                                    f"dh_ds_from_barrier ({float(dh_ds_from_barrier):.6g})."
                                )

                        # Eq. (32)/(35b): explicit-time residual (= 0 under pure propagation)
                        time_residual = float(dh_dt - dh_ds)

                        state_dim = Phi_i.shape[0]
                        pad_cols = max(0, state_dim - grad_pos.shape[1])
                        grad_h_phi = np.hstack([grad_pos, np.zeros((1, pad_cols))])

                        f_pi_phi = fcl_sc[i].reshape(-1, 1)
                        if f_pi_phi.shape[0] < state_dim:
                            f_pi_pad = np.vstack([f_pi_phi, np.zeros((state_dim - f_pi_phi.shape[0], 1))])
                        else:
                            f_pi_pad = f_pi_phi
                        if f_x.shape[0] < state_dim:
                            f_x_pad = np.vstack([f_x, np.zeros((state_dim - f_x.shape[0], 1))])
                        else:
                            f_x_pad = f_x
                        if g_x.shape[0] < state_dim:
                            g_x_pad = np.vstack([g_x, np.zeros((state_dim - g_x.shape[0], g_x.shape[1]))])
                        else:
                            g_x_pad = g_x

                        # Eq. (35b): c_occ = grad_h @ (Phi @ f(x) - f_b(y)) + dh_dt - dh_ds
                        c_occ = grad_h_phi @ (Phi_i @ f_x_pad - f_pi_pad) + time_residual
                        A_occ = grad_h_phi @ Phi_i @ g_x_pad

                        if not (np.all(np.isfinite(A_occ)) and np.all(np.isfinite(c_occ))):
                            continue

                        rhs = float(c_occ + self._alpha(h_tilde))
                        if np.linalg.norm(A_occ) < 1e-9 and rhs < 0.0:
                            continue

                        A_list.append(-A_occ)
                        b_list.append(np.array([[rhs]]))
                        meta_list.append({"kind": "occ", "sc_idx": sc_idx, "tau": float(tau)})
        timings["build_occ_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        # terminal backup-set constraints
        t0 = time.perf_counter()
        term_scenarios = list(occ_for_qp) if not no_occ else []
        term_scenarios = self._select_terminal_scenarios(robot_state, term_scenarios)
        try:
            f_term = self.robot.f(robot_state)
        except TypeError:
            f_term = self.robot.f()
        try:
            g_term = self.robot.g(robot_state)
        except TypeError:
            g_term = self.robot.g()

        for term_pos, (sc_idx, term_scn_raw) in enumerate(term_scenarios):
            if per_scenario_rollouts is not None:
                phi_T = per_scenario_rollouts[sc_idx][0][-1].reshape(-1, 1)
                Phi_T = per_scenario_rollouts[sc_idx][1][-1]
            else:
                phi_T = phi_b[-1].reshape(-1, 1)
                Phi_T = Phi_b[-1]
            term_scn = self._build_terminal_constraint_scenario(term_scn_raw, phi_T)
            if term_scn is None:
                continue
            state_dim_T = Phi_T.shape[0]
            f_term_use = f_term
            g_term_use = g_term
            if f_term_use.shape[0] < state_dim_T:
                f_term_use = np.vstack([f_term_use, np.zeros((state_dim_T - f_term_use.shape[0], 1))])
            if g_term_use.shape[0] < state_dim_T:
                g_term_use = np.vstack([g_term_use, np.zeros((state_dim_T - g_term_use.shape[0], g_term_use.shape[1]))])
            rho_term = self._resolve_terminal_rho()
            self.occlusion_backup.set_terminal_backup_context(
                term_scn, self.T_horizon, kappa=self.kappa, rho_T=rho_term
            )
            h_b_T = self.occlusion_backup._occ_terminal_set(phi_T)
            grad_h_b_T = self.occlusion_backup.grad_occ_terminal(phi_T)

            if h_b_T is None or grad_h_b_T is None:
                continue

            Lfh_b_T = grad_h_b_T @ (Phi_T @ f_term_use)
            Lgh_b_T = grad_h_b_T @ (Phi_T @ g_term_use)
            rhs_b = float(Lfh_b_T + self._alpha(h_b_T))

            A_list.append(-Lgh_b_T)
            b_list.append([[rhs_b]])
            meta_list.append(
                {
                    "kind": "terminal",
                    "tau": float(self.T_horizon),
                    "sc_idx": int(sc_idx),
                    "term_pos": int(term_pos),
                    "terminal_mode": self.terminal_mode,
                    "terminal_residual_mode": self.terminal_residual_mode,
                }
            )
        timings["build_terminal_ms"] = (time.perf_counter() - t0) * 1000.0

        if len(A_list) == 0:
            self.status = "optimal"
            self.last_num_constraints = 0
            self.last_qp_solve_time_ms = 0.0
            self.last_solver_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_terminal_slack_val = None
            self.last_terminal_slack_l1 = 0.0
            self.last_terminal_slack_max = 0.0
            self.last_terminal_slack_active_count = 0
            self.last_intervention = "u_ref"
            u_out = self._clip_du_input_for_speed(robot_state, self.u_ref.value)
            self.last_u = u_out
            return u_out

        t0 = time.perf_counter()
        A_cbf_val = np.vstack(A_list).reshape(-1, 2)
        b_cbf_val = np.vstack(b_list).reshape(-1, 1)
        num_constraints = int(A_cbf_val.shape[0])
        terminal_slack_ub_val = self._build_terminal_slack_ub(meta_list)
        self.last_constraint_meta = list(meta_list)
        self.last_A_cbf_val = np.array(A_cbf_val, dtype=float, copy=True)
        self.last_b_cbf_val = np.array(b_cbf_val, dtype=float, copy=True)
        self.last_terminal_slack_val = None
        self.last_terminal_slack_l1 = 0.0
        self.last_terminal_slack_max = 0.0
        self.last_terminal_slack_active_count = 0
        timings["stack_constraints_ms"] = (time.perf_counter() - t0) * 1000.0

        tol = float(self.robot_spec.get("intervention_tol", 1e-3))
        u_ref = self.last_u_ref
        violation = (A_cbf_val @ u_ref - b_cbf_val).flatten()
        max_violation = float(np.max(violation))

        input_ok = True
        if u_ref is not None:
            model_name = str(self.robot_spec.get("model", "")).strip()
            u_flat = np.asarray(u_ref, dtype=float).reshape(-1)
            if model_name == "Unicycle2D" and u_flat.size >= 2:
                v_max = float(self.robot_spec.get("v_max", np.inf))
                v_min = float(self.robot_spec.get("v_min", -v_max))
                if (not np.isfinite(v_min)) or v_min > v_max:
                    v_min = -v_max
                w_max = float(self.robot_spec.get("w_max", np.inf))
                input_ok = bool((v_min - tol) <= float(u_flat[0]) <= (v_max + tol) and abs(u_flat[1]) <= w_max + tol)
            elif model_name == "DynamicUnicycle2D" and u_flat.size >= 2:
                a_max = float(self.robot_spec.get("a_max", np.inf))
                w_max = float(self.robot_spec.get("w_max", np.inf))
                try:
                    v_cur = float(np.asarray(robot_state, dtype=float).reshape(-1, 1)[3, 0])
                except Exception:
                    v_cur = 0.0
                dt_du = max(float(getattr(self.robot, "dt", self.dt_backup)), 1e-6)
                v_max = float(self.robot_spec.get("v_max", np.inf))
                v_min = float(self.robot_spec.get("v_min", -v_max))
                if not np.isfinite(v_min):
                    v_min = -v_max
                if v_min > v_max:
                    v_min = v_max
                a_lb = max(-a_max, (v_min - v_cur) / dt_du)
                a_ub = min(a_max, (v_max - v_cur) / dt_du)
                input_ok = bool(
                    (a_lb - tol) <= float(u_flat[0]) <= (a_ub + tol)
                    and abs(u_flat[1]) <= w_max + tol
                )
            elif "a_max" in self.robot_spec:
                a_max = float(self.robot_spec.get("a_max", np.inf))
                input_ok = bool(np.all(np.abs(u_flat) <= a_max + tol))

        if max_violation <= tol and input_ok:
            self.status = "optimal"
            self.last_num_constraints = num_constraints
            self.last_qp_solve_time_ms = 0.0
            self.last_solver_solve_time_ms = 0.0
            self.last_total_compute_time_ms = (time.perf_counter() - t_all0) * 1000.0
            self.last_intervention = "u_ref"
            u_out = self._clip_du_input_for_speed(robot_state, self.u_ref.value)
            self.last_u = u_out
            # if self.debug:
                # print(f"[OcclusionCBFQP] u_ref feasible (max_violation={max_violation:.3e}) -> use u_ref")
            return u_out

        solve_exception = self._solve_qp_with_fallbacks(
            A_cbf_val,
            b_cbf_val,
            num_constraints,
            timings,
            terminal_slack_ub_val=terminal_slack_ub_val,
        )
        timings["total_ms"] = (time.perf_counter() - t_all0) * 1000.0
        timings["occ_candidate_count"] = int(self.last_occ_candidate_count)
        timings["occ_max_active"] = int(self.last_occ_max_active)
        timings["occ_selection_mode"] = self.last_occ_selection_mode
        timings["occ_selected_count"] = int(len(self.last_occ_selected_indices))
        timings["occ_selected_indices"] = list(self.last_occ_selected_indices)
        timings["occ_selected_scores"] = list(self.last_occ_selected_scores)
        self.last_profile = timings

        self.last_num_constraints = num_constraints
        # QP-only wall time (cvxpy solve call scope, plotting excluded)
        self.last_qp_solve_time_ms = float(timings["qp_wall_ms"])
        solver_s = timings.get("solver_solve_time_s", None)
        self.last_solver_solve_time_ms = (
            float(solver_s) * 1000.0 if solver_s is not None else None
        )
        # Keep full control-cycle compute time separately for profiling.
        self.last_total_compute_time_ms = float(timings["total_ms"])

        qp_status = self.cbf_controller.status
        self.last_qp_status_raw = qp_status
        self.last_qp_exception = None if solve_exception is None else str(solve_exception)
        qp_ok = False
        u_raw = None
        slack_raw = None
        if solve_exception is None and self.u.value is not None:
            u_raw = np.array(self.u.value, dtype=float).reshape(-1, 1)
            if self._terminal_slack_enabled and self.terminal_slack is not None and self.terminal_slack.value is not None:
                slack_raw = np.array(self.terminal_slack.value, dtype=float).reshape(-1, 1)[:num_constraints, :]
            tol_feas = float(self.robot_spec.get("cbf_feas_tol", 1e-4))
            raw_violation = self._effective_constraint_violation(
                A_cbf_val, b_cbf_val, u_raw, slack_val=slack_raw
            ).flatten()
            max_raw_violation = float(np.max(raw_violation)) if raw_violation.size > 0 else -np.inf
            if qp_status == "optimal":
                qp_ok = True
            elif qp_status in {"optimal_inaccurate", "user_limit"} and max_raw_violation <= tol_feas:
                qp_ok = True

        if qp_ok:
            u_safe = np.array(u_raw, dtype=float).reshape(-1, 1)
            u_safe = self._clip_du_input_for_speed(robot_state, u_safe)
            self.last_u = u_safe
            if slack_raw is not None:
                self.last_terminal_slack_val = np.array(slack_raw, dtype=float, copy=True)
                self.last_terminal_slack_l1 = float(np.sum(np.abs(slack_raw)))
                self.last_terminal_slack_max = float(np.max(slack_raw)) if slack_raw.size > 0 else 0.0
                self.last_terminal_slack_active_count = int(np.count_nonzero(slack_raw > 1e-9))
            if self.last_u_ref is not None:
                delta = float(np.linalg.norm(self.last_u - self.last_u_ref))
            else:
                delta = float("inf")
            self.last_intervention = "u_ref" if delta <= tol else "backup_qp"
            self.status = "optimal"
            self._last_intervention = self.last_intervention != "u_ref"
            self._using_backup = self._last_intervention
            return u_safe

        fallback_scenarios = None if no_occ else occ_for_qp
        u_safe = np.array(
            self.occlusion_backup.backup_input_at(
                np.asarray(robot_state, dtype=float).reshape(-1, 1),
                scenarios=fallback_scenarios,
                t=0.0,
            ),
            dtype=float,
        ).reshape(-1, 1)
        u_safe = self._clip_du_input_for_speed(robot_state, u_safe)
        model_name = str(self.robot_spec.get("model", "")).strip()
        if model_name == "DynamicUnicycle2D":
            try:
                v_cur = float(np.asarray(robot_state, dtype=float).reshape(-1, 1)[3, 0])
            except Exception:
                v_cur = 0.0
            v_spin_eps = float(self.robot_spec.get("du_fallback_spin_v_thresh", 0.05))
            a_eps = float(self.robot_spec.get("du_fallback_spin_a_thresh", 1e-3))
            if abs(v_cur) <= v_spin_eps and abs(float(u_safe[0, 0])) <= a_eps:
                u_safe[1, 0] = 0.0
        self.last_u = u_safe
        self.last_fallback_min_h = float(min_barrier_now) if has_barrier_value else None
        tol_feas = float(self.robot_spec.get("cbf_feas_tol", 1e-4))
        fallback_state_safe = self._is_fallback_safe(min_barrier_now, has_barrier_value)
        fallback_cmd_feasible, fallback_cmd_max_violation = self._fallback_cmd_feasible(
            u_safe, A_cbf_val, b_cbf_val, tol_feas, slack_ub_val=terminal_slack_ub_val
        )
        self.last_fallback_cmd_max_violation = fallback_cmd_max_violation
        self.last_fallback_cmd_feasible = bool(fallback_cmd_feasible)
        fallback_safe = bool(fallback_state_safe and fallback_cmd_feasible)
        self.last_fallback_allowed = bool(fallback_safe)
        relaxed_fallback = False
        if self.qp_failure_fallback_mode == "state_safe":
            relaxed_fallback = bool(fallback_state_safe)
        elif self.qp_failure_fallback_mode == "always":
            relaxed_fallback = True
        if fallback_safe or relaxed_fallback:
            self.last_intervention = "backup_fallback" if fallback_safe else "backup_fallback_relaxed"
            self.status = "optimal"
        else:
            self.last_intervention = "infeasible"
            self.status = "infeasible"
        self._last_intervention = True
        self._using_backup = True
        return u_safe
