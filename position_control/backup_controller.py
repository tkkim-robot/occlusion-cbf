"""
Created on February 19th, 2026
@author: Hun Kuk Park

@description:
Backup Controller - Abstract base class and implementations for backup control strategies.
These controllers provide alternative control behaviors that can be used for safety guarantees
or emergency maneuvers. The controllers are designed to be simple feedback controllers
that can be forward-simulated to predict trajectories.

@required-scripts: None (standalone module)
"""

import numpy as np
from abc import ABC, abstractmethod
from functools import partial

from position_control.ocbf.defaults import (
    OCBF_DEFAULT_BARRIER_KAPPA,
    OCBF_DEFAULT_VREF_FRONT_MODE,
    OCBF_DEFAULT_VREF_SCENARIO_WEIGHT_MODE,
    OCBF_DEFAULT_VREF_TRACKING_MODE,
    OCBF_VREF_FRONT_MODES,
    OCBF_VREF_SCENARIO_WEIGHT_MODES,
    OCBF_VREF_TRACKING_MODES,
    normalize_choice,
)

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    _JAX_AVAILABLE = False

_JAX_WARNED_MISSING = False


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


if _JAX_AVAILABLE:
    def _jax_occ_target_normals(
        p,
        A_pad,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
    ):
        use_los = jnp.logical_and(front_mode_los, obs_center_valid > 0.5)
        los = p[None, :] - obs_center_pad
        los_norm = jnp.linalg.norm(los, axis=1, keepdims=True)
        los_dir = los / jnp.maximum(los_norm, 1e-9)
        replace_front = jnp.logical_and(use_los, los_norm[:, 0] > 1e-9)
        front_normal = jnp.where(replace_front[:, None], los_dir, A_pad[:, 0, :])
        return A_pad.at[:, 0, :].set(front_normal)

    def _jax_occ_weighted_scenario_average(targets, scores, use_target, kappa):
        valid_scores = jnp.where(use_target, scores, -jnp.inf)
        max_score = jnp.max(valid_scores)
        max_score_safe = jnp.where(jnp.isfinite(max_score), max_score, 0.0)
        z = jnp.where(
            use_target,
            jnp.exp(kappa * (valid_scores - max_score_safe)),
            0.0,
        )
        z_sum = jnp.sum(z)
        weights = z / jnp.maximum(z_sum, 1e-9)
        weighted = jnp.sum(weights[:, None] * targets, axis=0)
        return jnp.where(z_sum > 0.0, weighted, jnp.zeros((2,), dtype=targets.dtype))

    def _jax_occ_scenario_threat_scores(h_valid, valid_bool, barrier_kappa):
        barrier_kappa = jnp.maximum(jnp.asarray(barrier_kappa, dtype=h_valid.dtype), 1e-6)
        max_h = jnp.max(h_valid, axis=1)
        finite_row = jnp.isfinite(max_h)
        max_h_safe = jnp.where(finite_row, max_h, 0.0)
        z = jnp.where(
            valid_bool,
            jnp.exp(barrier_kappa * (h_valid - max_h_safe[:, None])),
            0.0,
        )
        Z = jnp.sum(z, axis=1)
        Z_safe = jnp.maximum(Z, 1e-12)
        K_count = jnp.sum(valid_bool.astype(h_valid.dtype), axis=1)
        K_safe = jnp.maximum(K_count, 1.0)
        h_tilde = max_h_safe + (jnp.log(Z_safe) - jnp.log(K_safe)) / barrier_kappa
        scenario_present = jnp.any(valid_bool, axis=1)
        threat = -h_tilde
        return jnp.where(jnp.logical_and(scenario_present, finite_row), threat, -jnp.inf)

    def _jax_occ_scenario_scores_with_unexpanded_margin(
        h_valid,
        valid_bool,
        barrier_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        p_score,
        radius,
        use_unexpanded_score,
    ):
        barrier_scores = _jax_occ_scenario_threat_scores(h_valid, valid_bool, barrier_kappa)
        score_valid_bool = score_valid_mask > 0.5
        h_score = jnp.einsum("skd,d->sk", score_A_pad, p_score) - score_b0_pad - radius
        h_score_valid = jnp.where(score_valid_bool, h_score, -jnp.inf)
        unexpanded_scores = _jax_occ_scenario_threat_scores(
            h_score_valid, score_valid_bool, barrier_kappa
        )
        use_unexpand = jnp.logical_and(use_unexpanded_score, score_present > 0.5)
        return jnp.where(use_unexpand, unexpanded_scores, barrier_scores)

    def _jax_occ_vref_from_pos_strict(
        p,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        radius,
        barrier_kappa,
        scenario_kappa,
    ):
        """
        Strict-active occlusion v_ref:
        - Per scenario: average active facets (h_k >= 0)
        - If no active facet: use argmax(h_k)
        - Then threat-weight per-scenario v_target using -h_s^C
        """
        h = (
            jnp.einsum("skd,d->sk", A_pad, p)
            - b0_pad
            - (radius + v_expand_pad * tau)
        )

        valid_bool = valid_mask > 0.5
        neg_inf = jnp.full_like(h, -jnp.inf)
        h_valid = jnp.where(valid_bool, h, neg_inf)
        A_target = _jax_occ_target_normals(
            p, A_pad, obs_center_pad, obs_center_valid, front_mode_los
        )

        active_bool = jnp.logical_and(valid_bool, h_valid >= 0.0)
        active_count = jnp.sum(active_bool.astype(A_pad.dtype), axis=1, keepdims=True)
        active_sum = jnp.einsum("sk,skd->sd", active_bool.astype(A_pad.dtype), A_target)
        active_mean = active_sum / jnp.maximum(active_count, 1.0)

        best_idx = jnp.argmax(h_valid, axis=1)
        best_normal = A_target[jnp.arange(A_pad.shape[0]), best_idx, :]

        use_active = active_count[:, 0] > 0.0
        normal = jnp.where(use_active[:, None], active_mean, best_normal)

        norm = jnp.linalg.norm(normal, axis=1)
        direction = normal / jnp.maximum(norm[:, None], 1e-9)
        scenario_present = jnp.any(valid_bool, axis=1)
        use_target = jnp.logical_and(scenario_present, norm > 1e-9)
        targets = jnp.where(use_target[:, None], direction * v_adv_vec[:, None], 0.0)
        scores = _jax_occ_scenario_threat_scores(h_valid, valid_bool, barrier_kappa)
        return _jax_occ_weighted_scenario_average(
            targets,
            scores,
            use_target,
            scenario_kappa,
        )

    def _jax_occ_vref_from_pos(
        p,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        radius,
        barrier_kappa,
        scenario_kappa,
    ):
        """Vectorized occlusion velocity reference from position only."""
        h = (
            jnp.einsum("skd,d->sk", A_pad, p)
            - b0_pad
            - (radius + v_expand_pad * tau)
        )

        valid_bool = valid_mask > 0.5
        neg_inf = jnp.full_like(h, -jnp.inf)
        h_valid = jnp.where(valid_bool, h, neg_inf)
        A_target = _jax_occ_target_normals(
            p, A_pad, obs_center_pad, obs_center_valid, front_mode_los
        )

        # Smooth facet blend to avoid discontinuous facet switching,
        # which is particularly harmful for non-holonomic heading control.
        kappa_vref = jnp.asarray(8.0, dtype=A_pad.dtype)
        max_h = jnp.max(h_valid, axis=1, keepdims=True)
        max_h_safe = jnp.where(jnp.isfinite(max_h), max_h, 0.0)
        z = jnp.where(
            valid_bool,
            jnp.exp(kappa_vref * (h_valid - max_h_safe)),
            0.0,
        )
        z_sum = jnp.sum(z, axis=1, keepdims=True)
        lam = z / jnp.maximum(z_sum, 1e-9)
        normal = jnp.einsum("sk,skd->sd", lam, A_target)

        norm = jnp.linalg.norm(normal, axis=1)
        direction = normal / jnp.maximum(norm[:, None], 1e-9)

        scenario_present = jnp.any(valid_bool, axis=1)
        use_target = jnp.logical_and(scenario_present, norm > 1e-9)
        targets = jnp.where(use_target[:, None], direction * v_adv_vec[:, None], 0.0)
        scores = _jax_occ_scenario_threat_scores(h_valid, valid_bool, barrier_kappa)
        return _jax_occ_weighted_scenario_average(
            targets,
            scores,
            use_target,
            scenario_kappa,
        )


    def _jax_occ_vref_from_pos_strict_scored(
        p,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        radius,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        p_score,
        use_unexpanded_score,
    ):
        h = (
            jnp.einsum("skd,d->sk", A_pad, p)
            - b0_pad
            - (radius + v_expand_pad * tau)
        )

        valid_bool = valid_mask > 0.5
        h_valid = jnp.where(valid_bool, h, -jnp.inf)
        A_target = _jax_occ_target_normals(
            p, A_pad, obs_center_pad, obs_center_valid, front_mode_los
        )

        active_bool = jnp.logical_and(valid_bool, h_valid >= 0.0)
        active_count = jnp.sum(active_bool.astype(A_pad.dtype), axis=1, keepdims=True)
        active_sum = jnp.einsum("sk,skd->sd", active_bool.astype(A_pad.dtype), A_target)
        active_mean = active_sum / jnp.maximum(active_count, 1.0)

        best_idx = jnp.argmax(h_valid, axis=1)
        best_normal = A_target[jnp.arange(A_pad.shape[0]), best_idx, :]

        use_active = active_count[:, 0] > 0.0
        normal = jnp.where(use_active[:, None], active_mean, best_normal)

        norm = jnp.linalg.norm(normal, axis=1)
        direction = normal / jnp.maximum(norm[:, None], 1e-9)
        scenario_present = jnp.any(valid_bool, axis=1)
        use_target = jnp.logical_and(scenario_present, norm > 1e-9)
        targets = jnp.where(use_target[:, None], direction * v_adv_vec[:, None], 0.0)
        scores = _jax_occ_scenario_scores_with_unexpanded_margin(
            h_valid,
            valid_bool,
            barrier_kappa,
            score_A_pad,
            score_b0_pad,
            score_valid_mask,
            score_present,
            p_score,
            radius,
            use_unexpanded_score,
        )
        return _jax_occ_weighted_scenario_average(
            targets,
            scores,
            use_target,
            scenario_kappa,
        )


    def _jax_occ_vref_from_pos_scored(
        p,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        radius,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        p_score,
        use_unexpanded_score,
    ):
        h = (
            jnp.einsum("skd,d->sk", A_pad, p)
            - b0_pad
            - (radius + v_expand_pad * tau)
        )

        valid_bool = valid_mask > 0.5
        h_valid = jnp.where(valid_bool, h, -jnp.inf)
        A_target = _jax_occ_target_normals(
            p, A_pad, obs_center_pad, obs_center_valid, front_mode_los
        )

        kappa_vref = jnp.asarray(8.0, dtype=A_pad.dtype)
        max_h = jnp.max(h_valid, axis=1, keepdims=True)
        max_h_safe = jnp.where(jnp.isfinite(max_h), max_h, 0.0)
        z = jnp.where(
            valid_bool,
            jnp.exp(kappa_vref * (h_valid - max_h_safe)),
            0.0,
        )
        z_sum = jnp.sum(z, axis=1, keepdims=True)
        lam = z / jnp.maximum(z_sum, 1e-9)
        normal = jnp.einsum("sk,skd->sd", lam, A_target)

        norm = jnp.linalg.norm(normal, axis=1)
        direction = normal / jnp.maximum(norm[:, None], 1e-9)

        scenario_present = jnp.any(valid_bool, axis=1)
        use_target = jnp.logical_and(scenario_present, norm > 1e-9)
        targets = jnp.where(use_target[:, None], direction * v_adv_vec[:, None], 0.0)
        scores = _jax_occ_scenario_scores_with_unexpanded_margin(
            h_valid,
            valid_bool,
            barrier_kappa,
            score_A_pad,
            score_b0_pad,
            score_valid_mask,
            score_present,
            p_score,
            radius,
            use_unexpanded_score,
        )
        return _jax_occ_weighted_scenario_average(
            targets,
            scores,
            use_target,
            scenario_kappa,
        )


    _jax_occ_vref_pos_jac_strict = jax.jacfwd(_jax_occ_vref_from_pos_strict, argnums=0)
    _jax_occ_vref_pos_jac = jax.jacfwd(_jax_occ_vref_from_pos, argnums=0)


    def _jax_occ_vref_from_di_state_strict(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        radius,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
    ):
        p = x[:2]
        h = (
            jnp.einsum("skd,d->sk", A_pad, p)
            - b0_pad
            - (radius + v_expand_pad * tau)
        )

        valid_bool = valid_mask > 0.5
        h_valid = jnp.where(valid_bool, h, -jnp.inf)
        A_target = _jax_occ_target_normals(
            p, A_pad, obs_center_pad, obs_center_valid, front_mode_los
        )

        active_bool = jnp.logical_and(valid_bool, h_valid >= 0.0)
        active_count = jnp.sum(active_bool.astype(A_pad.dtype), axis=1, keepdims=True)
        active_sum = jnp.einsum("sk,skd->sd", active_bool.astype(A_pad.dtype), A_target)
        active_mean = active_sum / jnp.maximum(active_count, 1.0)

        best_idx = jnp.argmax(h_valid, axis=1)
        best_normal = A_target[jnp.arange(A_pad.shape[0]), best_idx, :]

        use_active = active_count[:, 0] > 0.0
        normal = jnp.where(use_active[:, None], active_mean, best_normal)

        norm = jnp.linalg.norm(normal, axis=1)
        direction = normal / jnp.maximum(norm[:, None], 1e-9)
        scenario_present = jnp.any(valid_bool, axis=1)
        use_target = jnp.logical_and(scenario_present, norm > 1e-9)
        targets = jnp.where(use_target[:, None], direction * v_adv_vec[:, None], 0.0)

        p_score = p
        use_unexpanded_score = vref_weight_mode_code == 1
        scores = _jax_occ_scenario_scores_with_unexpanded_margin(
            h_valid,
            valid_bool,
            barrier_kappa,
            score_A_pad,
            score_b0_pad,
            score_valid_mask,
            score_present,
            p_score,
            radius,
            use_unexpanded_score,
        )
        weighted_vref = _jax_occ_weighted_scenario_average(
            targets,
            scores,
            use_target,
            scenario_kappa,
        )
        return weighted_vref


    def _jax_di_f_cl(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        Kp,
        k_d,
        a_lim,
        v_max,
        radius,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
    ):
        v = x[2:4]
        v_ref = _jax_occ_vref_from_di_state_strict(
            x, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            tau, radius, barrier_kappa, scenario_kappa,
            score_A_pad, score_b0_pad, score_valid_mask, score_present,
            vref_weight_mode_code,
        )
        e = v - v_ref
        u_unsat = -Kp * e - k_d * v
        u = jnp.clip(u_unsat, -a_lim, a_lim)

        eps = 1e-6
        u = jnp.where(jnp.logical_and(v >= (v_max - eps), u > 0.0), 0.0, u)
        u = jnp.where(jnp.logical_and(v <= (-v_max + eps), u < 0.0), 0.0, u)

        return jnp.array([v[0], v[1], u[0], u[1]], dtype=x.dtype)


    _jax_di_f_cl_state_jac = jax.jacfwd(_jax_di_f_cl, argnums=0)


    def _jax_di_F_cl_jac(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        Kp,
        k_d,
        a_lim,
        v_max,
        radius,
        barrier_kappa,
        scenario_kappa,
        eps_fd,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
    ):
        del eps_fd
        def _unexpanded_jac(_):
            return _jax_di_f_cl_state_jac(
                x,
                A_pad,
                b0_pad,
                v_expand_pad,
                v_adv_vec,
                valid_mask,
                obs_center_pad,
                obs_center_valid,
                front_mode_los,
                tau,
                Kp,
                k_d,
                a_lim,
                v_max,
                radius,
                barrier_kappa,
                scenario_kappa,
                score_A_pad,
                score_b0_pad,
                score_valid_mask,
                score_present,
                vref_weight_mode_code,
            )

        def _barrier_jac(_):
            Jp = _jax_occ_vref_pos_jac_strict(
                x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                tau, radius, barrier_kappa, scenario_kappa
            )
            Bv = -(Kp + k_d) * jnp.eye(2, dtype=x.dtype)
            lower_left = Kp * Jp
            top = jnp.concatenate([jnp.zeros((2, 2), dtype=x.dtype), jnp.eye(2, dtype=x.dtype)], axis=1)
            bottom = jnp.concatenate([lower_left, Bv], axis=1)
            return jnp.concatenate([top, bottom], axis=0)

        return jax.lax.cond(
            vref_weight_mode_code > 0,
            _unexpanded_jac,
            _barrier_jac,
            operand=None,
        )


    @partial(jax.jit, static_argnums=(10,))
    def _jax_rollout_kernel_di(
        x0,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        dt,
        n_steps,
        Kp,
        k_d,
        a_lim,
        v_max,
        radius,
        barrier_kappa,
        scenario_kappa,
        eps_fd,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
    ):
        """
        JIT-compiled rollout kernel.
        static_argnums=(7,) fixes n_steps at compile time to avoid re-tracing.
        """
        I = jnp.eye(4, dtype=x0.dtype)

        def step_fn(carry, idx):
            x, Phi = carry
            t = dt * idx.astype(x0.dtype)

            k1 = _jax_di_f_cl(
                x, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t, Kp, k_d, a_lim, v_max, radius, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code,
            )
            k2 = _jax_di_f_cl(
                x + 0.5 * dt * k1, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, Kp, k_d, a_lim, v_max, radius, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code,
            )
            k3 = _jax_di_f_cl(
                x + 0.5 * dt * k2, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, Kp, k_d, a_lim, v_max, radius, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code,
            )
            k4 = _jax_di_f_cl(
                x + dt * k3, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + dt, Kp, k_d, a_lim, v_max, radius, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code,
            )
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_mid = x + 0.5 * dt * k2
            A_mid = _jax_di_F_cl_jac(
                x_mid, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, Kp, k_d, a_lim, v_max, radius, barrier_kappa, scenario_kappa, eps_fd,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code,
            )
            M1 = I - 0.5 * dt * A_mid
            M2 = I + 0.5 * dt * A_mid
            Phi_next = jnp.linalg.solve(M1 + 1e-8 * I, M2 @ Phi)

            return (x_next, Phi_next), (x_next, Phi_next, k1)

        idxs = jnp.arange(n_steps - 1, dtype=jnp.int32)
        (x_final, _), (x_seq, Phi_seq, k1_seq) = jax.lax.scan(
            step_fn, (x0, I), idxs
        )

        t_last = dt * (n_steps - 1)
        f_last = _jax_di_f_cl(
            x_final, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            t_last, Kp, k_d, a_lim, v_max, radius, barrier_kappa, scenario_kappa,
            score_A_pad, score_b0_pad, score_valid_mask, score_present,
            vref_weight_mode_code,
        )

        traj = jnp.concatenate([x0[None, :], x_seq], axis=0)
        stm = jnp.concatenate([I[None, :, :], Phi_seq], axis=0)
        fcl = jnp.concatenate([k1_seq, f_last[None, :]], axis=0)
        t_grid = dt * jnp.arange(n_steps, dtype=x0.dtype)
        return traj, stm, t_grid, fcl


    def _jax_angle_normalize(x):
        return ((x + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi


    def _jax_scale_occ_vref(
        pos,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        ref_speed,
        radius,
        use_strict,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
    ):
        v_ref = jax.lax.cond(
            use_strict,
            lambda _: _jax_occ_vref_from_pos_strict(
                pos, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                tau, radius, barrier_kappa, scenario_kappa
            ),
            lambda _: _jax_occ_vref_from_pos(
                pos, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                tau, radius, barrier_kappa, scenario_kappa
            ),
            operand=None,
        )
        del ref_speed, use_norm_mode
        return v_ref, jnp.linalg.norm(v_ref)

    def _jax_scale_occ_vref_scored(
        pos,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        ref_speed,
        radius,
        use_strict,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        p_score,
        use_unexpanded_score,
    ):
        v_ref = jax.lax.cond(
            use_strict,
            lambda _: _jax_occ_vref_from_pos_strict_scored(
                pos, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                tau, radius, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                p_score, use_unexpanded_score,
            ),
            lambda _: _jax_occ_vref_from_pos_scored(
                pos, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                tau, radius, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                p_score, use_unexpanded_score,
            ),
            operand=None,
        )
        del ref_speed, use_norm_mode
        return v_ref, jnp.linalg.norm(v_ref)

    def _jax_uni_virtual_cmd_from_vref(
        theta,
        v_ref,
        v_ref_prev,
        dt,
        v_min,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        v_min_cmd_rev,
        turn_crawl_speed,
        turn_crawl_angle,
        reverse_bias,
        reverse_gate_angle,
        reverse_gate_power,
        tracking_mode_code,
    ):
        v_ref_norm = jnp.linalg.norm(v_ref)
        theta_ref = jnp.arctan2(v_ref[1], v_ref[0])
        theta_ref_prev = jnp.arctan2(v_ref_prev[1], v_ref_prev[0])

        e_fwd = _jax_angle_normalize(theta_ref - theta)
        e_rev = _jax_angle_normalize(theta_ref + jnp.pi - theta)
        allow_reverse = v_min < -1e-9
        use_reverse = allow_reverse & ((jnp.abs(e_rev) + reverse_bias) < jnp.abs(e_fwd))
        e_track = jnp.where(use_reverse, e_rev, e_fwd)

        e_fwd_prev = _jax_angle_normalize(theta_ref_prev - theta)
        e_rev_prev = _jax_angle_normalize(theta_ref_prev + jnp.pi - theta)
        use_reverse_prev = allow_reverse & ((jnp.abs(e_rev_prev) + reverse_bias) < jnp.abs(e_fwd_prev))
        e_track_prev = jnp.where(use_reverse_prev, e_rev_prev, e_fwd_prev)

        dt_safe = jnp.maximum(dt, 1e-6)
        e_track_dot = _jax_angle_normalize(e_track - e_track_prev) / dt_safe

        boost = 1.0 + k_turn_boost * jnp.minimum(
            1.0, jnp.abs(e_track) / jnp.maximum(turn_boost_angle, 1e-3)
        )
        omega_unsat = boost * (k_theta_p * e_track + k_theta_d * e_track_dot)
        omega_gated = jnp.clip(omega_unsat, -w_max, w_max)

        v_ref_prev_norm = jnp.linalg.norm(v_ref_prev)
        fwd_gate = jnp.maximum(0.0, jnp.cos(e_fwd))
        fwd_gate_prev = jnp.maximum(0.0, jnp.cos(e_fwd_prev))
        reverse_gate_angle = jnp.maximum(reverse_gate_angle, 1e-3)
        reverse_gate_power = jnp.maximum(reverse_gate_power, 1.0)
        rev_gate = jnp.clip(1.0 - jnp.abs(e_rev) / reverse_gate_angle, 0.0, 1.0) ** reverse_gate_power
        rev_gate_prev = jnp.clip(1.0 - jnp.abs(e_rev_prev) / reverse_gate_angle, 0.0, 1.0) ** reverse_gate_power
        rev_cap = jnp.maximum(0.0, jnp.abs(jnp.minimum(v_min, 0.0)))

        v_des_fwd = jnp.clip(v_ref_norm * fwd_gate, 0.0, v_max)
        v_des_fwd_prev = jnp.clip(v_ref_prev_norm * fwd_gate_prev, 0.0, v_max)
        turn_crawl_speed = jnp.clip(turn_crawl_speed, 0.0, v_max)
        turn_crawl_angle = jnp.maximum(turn_crawl_angle, 1e-3)
        crawl_fwd = jnp.logical_and(v_ref_norm > 1e-9, jnp.abs(e_fwd) <= turn_crawl_angle)
        crawl_fwd_prev = jnp.logical_and(v_ref_prev_norm > 1e-9, jnp.abs(e_fwd_prev) <= turn_crawl_angle)
        v_des_fwd = jnp.where(crawl_fwd, jnp.maximum(v_des_fwd, turn_crawl_speed), v_des_fwd)
        v_des_fwd_prev = jnp.where(crawl_fwd_prev, jnp.maximum(v_des_fwd_prev, turn_crawl_speed), v_des_fwd_prev)
        v_des_rev = -jnp.clip(v_ref_norm * rev_gate, 0.0, rev_cap)
        v_des_rev_prev = -jnp.clip(v_ref_prev_norm * rev_gate_prev, 0.0, rev_cap)
        v_des = jnp.where(use_reverse, v_des_rev, v_des_fwd)
        v_des_prev = jnp.where(use_reverse_prev, v_des_rev_prev, v_des_fwd_prev)
        v_des_dot = (v_des - v_des_prev) / dt_safe

        v_cmd_unsat = k_v_p * v_des + k_v_d * v_des_dot
        v_cmd = jnp.clip(v_cmd_unsat, v_min, v_max)

        fwd_floor = jnp.minimum(v_min_cmd, v_max)
        rev_floor = jnp.minimum(v_min_cmd_rev, jnp.abs(v_min))
        v_cmd = jnp.where(v_cmd > 1e-9, jnp.maximum(v_cmd, fwd_floor), v_cmd)
        v_cmd = jnp.where(v_cmd < -1e-9, jnp.minimum(v_cmd, -rev_floor), v_cmd)

        # Projected mode: least-squares map from inertial v_ref to the
        # instantaneous velocity subspace span{[cos(theta), sin(theta)]}.
        heading = jnp.array([jnp.cos(theta), jnp.sin(theta)], dtype=v_ref.dtype)
        v_proj = jnp.dot(v_ref, heading)
        v_proj_prev = jnp.dot(v_ref_prev, heading)
        v_des_proj = jnp.where(
            allow_reverse,
            jnp.clip(v_proj, v_min, v_max),
            jnp.clip(jnp.maximum(v_proj, 0.0), 0.0, v_max),
        )
        v_des_proj_prev = jnp.where(
            allow_reverse,
            jnp.clip(v_proj_prev, v_min, v_max),
            jnp.clip(jnp.maximum(v_proj_prev, 0.0), 0.0, v_max),
        )
        use_reverse_proj = allow_reverse & (v_des_proj < -1e-9)
        use_reverse_proj_prev = allow_reverse & (v_des_proj_prev < -1e-9)
        e_proj = jnp.where(use_reverse_proj, e_rev, e_fwd)
        e_proj_prev = jnp.where(use_reverse_proj_prev, e_rev_prev, e_fwd_prev)
        crawl_proj = jnp.logical_and(
            jnp.logical_and(~use_reverse_proj, v_ref_norm > 1e-9),
            jnp.abs(e_proj) <= turn_crawl_angle,
        )
        crawl_proj_prev = jnp.logical_and(
            jnp.logical_and(~use_reverse_proj_prev, v_ref_prev_norm > 1e-9),
            jnp.abs(e_proj_prev) <= turn_crawl_angle,
        )
        v_des_proj = jnp.where(crawl_proj, jnp.maximum(v_des_proj, turn_crawl_speed), v_des_proj)
        v_des_proj_prev = jnp.where(crawl_proj_prev, jnp.maximum(v_des_proj_prev, turn_crawl_speed), v_des_proj_prev)
        e_proj_dot = _jax_angle_normalize(e_proj - e_proj_prev) / dt_safe
        boost_proj = 1.0 + k_turn_boost * jnp.minimum(
            1.0, jnp.abs(e_proj) / jnp.maximum(turn_boost_angle, 1e-3)
        )
        omega_proj = jnp.clip(
            boost_proj * (k_theta_p * e_proj + k_theta_d * e_proj_dot),
            -w_max,
            w_max,
        )
        v_des_proj_dot = (v_des_proj - v_des_proj_prev) / dt_safe
        v_cmd_proj = jnp.clip(k_v_p * v_des_proj + k_v_d * v_des_proj_dot, v_min, v_max)
        v_cmd_proj = jnp.where(v_cmd_proj > 1e-9, jnp.maximum(v_cmd_proj, fwd_floor), v_cmd_proj)
        v_cmd_proj = jnp.where(v_cmd_proj < -1e-9, jnp.minimum(v_cmd_proj, -rev_floor), v_cmd_proj)

        use_projected = tracking_mode_code == 1
        omega = jnp.where(use_projected, omega_proj, omega_gated)
        v_cmd = jnp.where(use_projected, v_cmd_proj, v_cmd)

        omega = jnp.where(v_ref_norm < 1e-9, 0.0, omega)
        v_cmd = jnp.where(v_ref_norm < 1e-9, 0.0, v_cmd)
        return jnp.array([v_cmd, omega], dtype=v_ref.dtype)

    def _jax_uni_backup_input_occ(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        dt,
        v_min,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        v_min_cmd_rev,
        turn_crawl_speed,
        turn_crawl_angle,
        reverse_bias,
        reverse_gate_angle,
        reverse_gate_power,
        tracking_mode_code,
        uni_vref_speed,
        radius,
        use_strict_vref,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
        last_speed,
    ):
        del last_speed
        theta = x[2]
        tau_prev = jnp.maximum(tau - dt, 0.0)
        p_score = x[:2]
        use_unexpanded_score = vref_weight_mode_code == 1
        v_ref, _ = _jax_scale_occ_vref_scored(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            tau, uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
            score_A_pad, score_b0_pad, score_valid_mask, score_present,
            p_score, use_unexpanded_score,
        )
        v_ref_prev, _ = _jax_scale_occ_vref_scored(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            tau_prev, uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
            score_A_pad, score_b0_pad, score_valid_mask, score_present,
            p_score, use_unexpanded_score,
        )
        return _jax_uni_virtual_cmd_from_vref(
            theta,
            v_ref,
            v_ref_prev,
            dt,
            v_min,
            v_max,
            w_max,
            k_theta_p,
            k_theta_d,
            k_v_p,
            k_v_d,
            k_turn_boost,
            turn_boost_angle,
            v_min_cmd,
            v_min_cmd_rev,
            turn_crawl_speed,
            turn_crawl_angle,
            reverse_bias,
            reverse_gate_angle,
            reverse_gate_power,
            tracking_mode_code,
        )

    def _jax_du_backup_input_occ(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        dt,
        v_min,
        v_max,
        w_max,
        a_max,
        k_theta_p,
        k_theta_d,
        k_turn_boost,
        turn_boost_angle,
        k_a_p,
        k_a_d,
        k_brake,
        k_vemu_p,
        k_vemu_d,
        v_min_cmd_emu,
        du_nonnegative_speed,
        du_vref_speed,
        radius,
        use_strict_vref,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
    ):
        v_cur = x[3]
        tau_prev = jnp.maximum(tau - dt, 0.0)
        tau_prev2 = jnp.maximum(tau_prev - dt, 0.0)
        dt_safe = jnp.maximum(dt, 1e-6)
        v_floor = jnp.where(du_nonnegative_speed > 0.5, 0.0, v_min)
        a_lb_base = jnp.maximum(-a_max, (v_floor - v_cur) / dt_safe)
        a_ub_base = jnp.minimum(a_max, (v_max - v_cur) / dt_safe)
        a_stop = jnp.clip(-k_brake * v_cur, a_lb_base, a_ub_base)

        v_ref, v_ref_norm = _jax_scale_occ_vref(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            tau, du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
        )
        v_ref_prev, _ = _jax_scale_occ_vref(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            tau_prev, du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
        )
        v_ref_prev2, _ = _jax_scale_occ_vref(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            tau_prev2, du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
        )

        theta = x[2]
        # Build UNI-like virtual (v, omega), then DU tracks v via acceleration.
        u_uni = _jax_uni_virtual_cmd_from_vref(
            theta,
            v_ref,
            v_ref_prev,
            dt,
            v_min,
            v_max,
            w_max,
            k_theta_p,
            k_theta_d,
            k_vemu_p,
            k_vemu_d,
            k_turn_boost,
            turn_boost_angle,
            v_min_cmd_emu,
            v_min_cmd_emu,
            0.0,
            jnp.pi / 2.0,
            0.05,
            0.45,
            2.0,
            0,
        )
        u_uni_prev = _jax_uni_virtual_cmd_from_vref(
            theta,
            v_ref_prev,
            v_ref_prev2,
            dt,
            v_min,
            v_max,
            w_max,
            k_theta_p,
            k_theta_d,
            k_vemu_p,
            k_vemu_d,
            k_turn_boost,
            turn_boost_angle,
            v_min_cmd_emu,
            v_min_cmd_emu,
            0.0,
            jnp.pi / 2.0,
            0.05,
            0.45,
            2.0,
            0,
        )
        v_cmd = u_uni[0]
        omega = u_uni[1]
        v_cmd_prev = u_uni_prev[0]
        v_cmd_dot = (v_cmd - v_cmd_prev) / dt_safe

        a_cmd_unsat = k_a_p * (v_cmd - v_cur) + k_a_d * v_cmd_dot
        a_cmd = jnp.clip(a_cmd_unsat, a_lb_base, a_ub_base)

        no_ref = v_ref_norm < 1e-9
        a_cmd = jnp.where(no_ref, a_stop, a_cmd)
        omega = jnp.where(no_ref, 0.0, omega)
        return jnp.array([a_cmd, omega], dtype=x.dtype)

    def _jax_uni_f_cl(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        dt,
        v_min,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        v_min_cmd_rev,
        turn_crawl_speed,
        turn_crawl_angle,
        reverse_bias,
        reverse_gate_angle,
        reverse_gate_power,
        tracking_mode_code,
        uni_vref_speed,
        radius,
        use_strict_vref,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
        last_speed,
    ):
        u = _jax_uni_backup_input_occ(
            x,
            A_pad,
            b0_pad,
            v_expand_pad,
            v_adv_vec,
            valid_mask,
            obs_center_pad,
            obs_center_valid,
            front_mode_los,
            tau,
            dt,
            v_min,
            v_max,
            w_max,
            k_theta_p,
            k_theta_d,
            k_v_p,
            k_v_d,
            k_turn_boost,
            turn_boost_angle,
            v_min_cmd,
            v_min_cmd_rev,
            turn_crawl_speed,
            turn_crawl_angle,
            reverse_bias,
            reverse_gate_angle,
            reverse_gate_power,
            tracking_mode_code,
            uni_vref_speed,
            radius,
            use_strict_vref,
            use_norm_mode,
            barrier_kappa,
            scenario_kappa,
            score_A_pad,
            score_b0_pad,
            score_valid_mask,
            score_present,
            vref_weight_mode_code,
            last_speed,
        )
        theta = x[2]
        v_cmd = u[0]
        w_cmd = u[1]
        return jnp.array(
            [v_cmd * jnp.cos(theta), v_cmd * jnp.sin(theta), w_cmd],
            dtype=x.dtype,
        )

    def _jax_du_f_cl(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        tau,
        dt,
        v_min,
        v_max,
        w_max,
        a_max,
        k_theta_p,
        k_theta_d,
        k_turn_boost,
        turn_boost_angle,
        k_a_p,
        k_a_d,
        k_brake,
        k_vemu_p,
        k_vemu_d,
        v_min_cmd_emu,
        du_nonnegative_speed,
        du_vref_speed,
        radius,
        use_strict_vref,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
    ):
        u = _jax_du_backup_input_occ(
            x,
            A_pad,
            b0_pad,
            v_expand_pad,
            v_adv_vec,
            valid_mask,
            obs_center_pad,
            obs_center_valid,
            front_mode_los,
            tau,
            dt,
            v_min,
            v_max,
            w_max,
            a_max,
            k_theta_p,
            k_theta_d,
            k_turn_boost,
            turn_boost_angle,
            k_a_p,
            k_a_d,
            k_brake,
            k_vemu_p,
            k_vemu_d,
            v_min_cmd_emu,
            du_nonnegative_speed,
            du_vref_speed,
            radius,
            use_strict_vref,
            use_norm_mode,
            barrier_kappa,
            scenario_kappa,
        )
        theta = x[2]
        v = x[3]
        a_cmd = u[0]
        w_cmd = u[1]
        return jnp.array(
            [v * jnp.cos(theta), v * jnp.sin(theta), w_cmd, a_cmd],
            dtype=x.dtype,
        )


    _jax_uni_F_cl_jac = jax.jacfwd(_jax_uni_f_cl, argnums=0)
    _jax_du_F_cl_jac = jax.jacfwd(_jax_du_f_cl, argnums=0)


    @partial(jax.jit, static_argnums=(10,))
    def _jax_rollout_kernel_uni(
        x0,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        dt,
        n_steps,
        v_min,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        v_min_cmd_rev,
        turn_crawl_speed,
        turn_crawl_angle,
        reverse_bias,
        reverse_gate_angle,
        reverse_gate_power,
        tracking_mode_code,
        uni_vref_speed,
        radius,
        use_strict_vref,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
        score_A_pad,
        score_b0_pad,
        score_valid_mask,
        score_present,
        vref_weight_mode_code,
        last_speed,
    ):
        I = jnp.eye(3, dtype=x0.dtype)

        def step_fn(carry, idx):
            x, Phi = carry
            t = dt * idx.astype(x0.dtype)

            k1 = _jax_uni_f_cl(
                x, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t, dt, v_min, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, v_min_cmd_rev,
                turn_crawl_speed, turn_crawl_angle, reverse_bias,
                reverse_gate_angle, reverse_gate_power,
                tracking_mode_code,
                uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code, last_speed,
            )
            k2 = _jax_uni_f_cl(
                x + 0.5 * dt * k1, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, dt, v_min, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, v_min_cmd_rev,
                turn_crawl_speed, turn_crawl_angle, reverse_bias,
                reverse_gate_angle, reverse_gate_power,
                tracking_mode_code,
                uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code, last_speed,
            )
            k3 = _jax_uni_f_cl(
                x + 0.5 * dt * k2, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, dt, v_min, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, v_min_cmd_rev,
                turn_crawl_speed, turn_crawl_angle, reverse_bias,
                reverse_gate_angle, reverse_gate_power,
                tracking_mode_code,
                uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code, last_speed,
            )
            k4 = _jax_uni_f_cl(
                x + dt * k3, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + dt, dt, v_min, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, v_min_cmd_rev,
                turn_crawl_speed, turn_crawl_angle, reverse_bias,
                reverse_gate_angle, reverse_gate_power,
                tracking_mode_code,
                uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code, last_speed,
            )
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_mid = x + 0.5 * dt * k2
            A_mid = _jax_uni_F_cl_jac(
                x_mid, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, dt, v_min, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, v_min_cmd_rev,
                turn_crawl_speed, turn_crawl_angle, reverse_bias,
                reverse_gate_angle, reverse_gate_power,
                tracking_mode_code,
                uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
                score_A_pad, score_b0_pad, score_valid_mask, score_present,
                vref_weight_mode_code, last_speed,
            )
            M1 = I - 0.5 * dt * A_mid
            M2 = I + 0.5 * dt * A_mid
            Phi_next = jnp.linalg.solve(M1 + 1e-8 * I, M2 @ Phi)

            return (x_next, Phi_next), (x_next, Phi_next, k1)

        idxs = jnp.arange(n_steps - 1, dtype=jnp.int32)
        (x_final, _), (x_seq, Phi_seq, k1_seq) = jax.lax.scan(
            step_fn, (x0, I), idxs
        )

        t_last = dt * (n_steps - 1)
        f_last = _jax_uni_f_cl(
            x_final, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            t_last, dt, v_min, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
            k_turn_boost, turn_boost_angle, v_min_cmd, v_min_cmd_rev,
            turn_crawl_speed, turn_crawl_angle, reverse_bias,
            reverse_gate_angle, reverse_gate_power,
            tracking_mode_code,
            uni_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa,
            score_A_pad, score_b0_pad, score_valid_mask, score_present,
            vref_weight_mode_code, last_speed,
        )

        traj = jnp.concatenate([x0[None, :], x_seq], axis=0)
        stm = jnp.concatenate([I[None, :, :], Phi_seq], axis=0)
        fcl = jnp.concatenate([k1_seq, f_last[None, :]], axis=0)
        t_grid = dt * jnp.arange(n_steps, dtype=x0.dtype)
        return traj, stm, t_grid, fcl


    @partial(jax.jit, static_argnums=(10,))
    def _jax_rollout_kernel_du(
        x0,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        obs_center_pad,
        obs_center_valid,
        front_mode_los,
        dt,
        n_steps,
        v_min,
        v_max,
        w_max,
        a_max,
        k_theta_p,
        k_theta_d,
        k_turn_boost,
        turn_boost_angle,
        k_a_p,
        k_a_d,
        k_brake,
        k_vemu_p,
        k_vemu_d,
        v_min_cmd_emu,
        du_nonnegative_speed,
        du_vref_speed,
        radius,
        use_strict_vref,
        use_norm_mode,
        barrier_kappa,
        scenario_kappa,
    ):
        I = jnp.eye(4, dtype=x0.dtype)

        def step_fn(carry, idx):
            x, Phi = carry
            t = dt * idx.astype(x0.dtype)

            k1 = _jax_du_f_cl(
                x, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t, dt, v_min, v_max, w_max, a_max,
                k_theta_p, k_theta_d, k_turn_boost, turn_boost_angle,
                k_a_p, k_a_d, k_brake, k_vemu_p, k_vemu_d, v_min_cmd_emu, du_nonnegative_speed,
                du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
            )
            k2 = _jax_du_f_cl(
                x + 0.5 * dt * k1, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, dt, v_min, v_max, w_max, a_max,
                k_theta_p, k_theta_d, k_turn_boost, turn_boost_angle,
                k_a_p, k_a_d, k_brake, k_vemu_p, k_vemu_d, v_min_cmd_emu, du_nonnegative_speed,
                du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
            )
            k3 = _jax_du_f_cl(
                x + 0.5 * dt * k2, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, dt, v_min, v_max, w_max, a_max,
                k_theta_p, k_theta_d, k_turn_boost, turn_boost_angle,
                k_a_p, k_a_d, k_brake, k_vemu_p, k_vemu_d, v_min_cmd_emu, du_nonnegative_speed,
                du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
            )
            k4 = _jax_du_f_cl(
                x + dt * k3, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + dt, dt, v_min, v_max, w_max, a_max,
                k_theta_p, k_theta_d, k_turn_boost, turn_boost_angle,
                k_a_p, k_a_d, k_brake, k_vemu_p, k_vemu_d, v_min_cmd_emu, du_nonnegative_speed,
                du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
            )
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_mid = x + 0.5 * dt * k2
            A_mid = _jax_du_F_cl_jac(
                x_mid, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                obs_center_pad, obs_center_valid, front_mode_los,
                t + 0.5 * dt, dt, v_min, v_max, w_max, a_max,
                k_theta_p, k_theta_d, k_turn_boost, turn_boost_angle,
                k_a_p, k_a_d, k_brake, k_vemu_p, k_vemu_d, v_min_cmd_emu, du_nonnegative_speed,
                du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
            )
            M1 = I - 0.5 * dt * A_mid
            M2 = I + 0.5 * dt * A_mid
            Phi_next = jnp.linalg.solve(M1 + 1e-8 * I, M2 @ Phi)

            return (x_next, Phi_next), (x_next, Phi_next, k1)

        idxs = jnp.arange(n_steps - 1, dtype=jnp.int32)
        (x_final, _), (x_seq, Phi_seq, k1_seq) = jax.lax.scan(
            step_fn, (x0, I), idxs
        )

        t_last = dt * (n_steps - 1)
        f_last = _jax_du_f_cl(
            x_final, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
            obs_center_pad, obs_center_valid, front_mode_los,
            t_last, dt, v_min, v_max, w_max, a_max,
            k_theta_p, k_theta_d, k_turn_boost, turn_boost_angle,
            k_a_p, k_a_d, k_brake, k_vemu_p, k_vemu_d, v_min_cmd_emu, du_nonnegative_speed,
            du_vref_speed, radius, use_strict_vref, use_norm_mode, barrier_kappa, scenario_kappa
        )

        traj = jnp.concatenate([x0[None, :], x_seq], axis=0)
        stm = jnp.concatenate([I[None, :, :], Phi_seq], axis=0)
        fcl = jnp.concatenate([k1_seq, f_last[None, :]], axis=0)
        t_grid = dt * jnp.arange(n_steps, dtype=x0.dtype)
        return traj, stm, t_grid, fcl


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
    """Backup policy used by Occlusion-CBF rollout constraints.

    Each control step receives selected occlusion scenarios from
    ``OcclusionCBFQP`` and generates a local escape velocity reference. DI
    tracks that reference with acceleration feedback; Uni/DU map it to their
    admissible velocity/turn-rate controls.
    """
    def __init__(self, robot_spec, dt):
        """Initialize the occlusion backup controller."""
        super().__init__(robot_spec, dt)
        model = str(self.robot_spec.get("model", "DoubleIntegrator2D")).strip()
        if model not in {"DoubleIntegrator2D", "Unicycle2D", "DynamicUnicycle2D"}:
            raise NotImplementedError(
                "OcclusionController is implemented for DoubleIntegrator2D, Unicycle2D, and DynamicUnicycle2D models."
            )
        self.model = model
        self._n_state = 3 if self.model == "Unicycle2D" else 4
        self._u_dim = 2

        self.pid_occ_gains = {
            "Kp": 1.0,
            "Ki": 0.2,
            "Kd": 1.0,
            "aw_limit": 1.0
        }
        cfg = self.robot_spec.setdefault("backup_cbf", {})
        # DI occlusion backup uses a proportional-plus-damping acceleration law.
        # Keep these gains configurable through backup_cbf overrides so DI sweeps
        # can tune them without changing the controller structure.
        self.pid_occ_gains["Kp"] = float(cfg.get("k_p_occ_di", self.pid_occ_gains["Kp"]))
        self.pid_occ_gains["Kd"] = float(cfg.get("k_d_occ_di", self.pid_occ_gains["Kd"]))
        cfg["k_p_occ_di"] = float(self.pid_occ_gains["Kp"])
        cfg["k_d_occ_di"] = float(self.pid_occ_gains["Kd"])
        self.T_horizon = float(cfg.get("T_horizon", 2.0))
        cfg["T_horizon"] = self.T_horizon
        self.last_vref_scenario_debug = None
        self.last_vref_scenario_summary = None

        weight_mode = normalize_choice(
            cfg.get("vref_scenario_weight_mode", OCBF_DEFAULT_VREF_SCENARIO_WEIGHT_MODE),
            OCBF_VREF_SCENARIO_WEIGHT_MODES,
            OCBF_DEFAULT_VREF_SCENARIO_WEIGHT_MODE,
        )
        cfg["vref_scenario_weight_mode"] = weight_mode

        # Optional JAX acceleration path for rollout/STM.
        self.use_jax_rollout = bool(cfg.get("use_jax_rollout", True))
        # JAX rollout kernels support DoubleIntegrator2D, Unicycle2D, DynamicUnicycle2D.
        self._jax_enabled = bool(
            self.use_jax_rollout
            and _JAX_AVAILABLE
            and self.model in {"DoubleIntegrator2D", "Unicycle2D", "DynamicUnicycle2D"}
        )
        self._jax_max_scenarios = int(cfg.get("jax_max_scenarios", 32))
        self._jax_max_halfspaces = int(cfg.get("jax_max_halfspaces", 8))
        self._jax_eps_fd = float(cfg.get("jax_eps_fd", 1e-3))
        self._jax_warm_n_steps = int(np.floor(self.T_horizon / self.dt)) + 1
        self._jax_warmed = False

        cfg["use_jax_rollout"] = self.use_jax_rollout
        cfg["jax_max_scenarios"] = self._jax_max_scenarios
        cfg["jax_max_halfspaces"] = self._jax_max_halfspaces
        cfg["jax_eps_fd"] = self._jax_eps_fd

        global _JAX_WARNED_MISSING
        if self.use_jax_rollout and not _JAX_AVAILABLE and not _JAX_WARNED_MISSING:
            print("[OcclusionController][WARN] JAX not installed; falling back to NumPy rollout.")
            _JAX_WARNED_MISSING = True

        if self._jax_enabled:
            self._warm_start_jax_rollout(self._jax_warm_n_steps)

    def get_backup_horizon(self):
        return float(getattr(self, "T_horizon", 2.0))

    def backup_input(self, X, k_a=1.0):
        """
        Default backup action:
        - DoubleIntegrator2D: stop-like damping in acceleration space.
        - Unicycle2D: immediate stop (v=0, w=0).
        """
        X = np.asarray(X, dtype=float).reshape(self._n_state, 1)
        if self.model == "DoubleIntegrator2D":
            vx = float(X[2, 0])
            vy = float(X[3, 0])
            return np.array([[-k_a * vx], [-k_a * vy]], dtype=float)
        if self.model == "DynamicUnicycle2D":
            v = float(X[3, 0])
            return np.array([[-k_a * v], [0.0]], dtype=float)
        if self.model == "Unicycle2D":
            return np.zeros((2, 1), dtype=float)
        raise NotImplementedError(f"Unsupported model: {self.model}")

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

    def _uni_occ_gains(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        tracking_mode = str(
            cfg.get("vref_tracking_mode_occ_uni", cfg.get("uni_vref_tracking_mode", "gated"))
        ).strip().lower()
        if tracking_mode in {"projection", "body_projection", "projected"}:
            tracking_mode = "projected"
        else:
            tracking_mode = "gated"
        k_theta_p = float(cfg.get("k_theta_occ_uni_p", cfg.get("k_theta_occ_uni", 2.5)))
        k_theta_d = float(cfg.get("k_theta_occ_uni_d", 0.4))
        k_v_p = float(cfg.get("k_v_occ_uni_p", 1.0))
        k_v_d = float(cfg.get("k_v_occ_uni_d", 0.2))
        k_turn_boost = float(cfg.get("k_turn_boost_occ_uni", 1.0))
        turn_boost_angle = float(cfg.get("turn_boost_angle_occ_uni", np.pi / 6.0))
        v_min_cmd = float(cfg.get("v_min_occ_uni", 0.0))
        v_min_cmd_rev = float(cfg.get("v_min_cmd_rev_occ_uni", v_min_cmd))
        turn_crawl_speed = float(cfg.get("turn_crawl_speed_occ_uni", 0.0))
        turn_crawl_angle = float(cfg.get("turn_crawl_angle_occ_uni", np.pi / 2.0))
        reverse_bias = float(cfg.get("reverse_bias_occ_uni", 0.05))
        reverse_gate_angle = float(cfg.get("reverse_speed_gate_angle_occ_uni", 0.45))
        reverse_gate_power = float(cfg.get("reverse_speed_gate_power_occ_uni", 2.0))
        if not np.isfinite(k_theta_p):
            k_theta_p = 2.5
        if not np.isfinite(k_theta_d):
            k_theta_d = 0.4
        if not np.isfinite(k_v_p):
            k_v_p = 1.0
        if not np.isfinite(k_v_d):
            k_v_d = 0.2
        if not np.isfinite(k_turn_boost):
            k_turn_boost = 1.0
        if (not np.isfinite(turn_boost_angle)) or (turn_boost_angle <= 1e-3):
            turn_boost_angle = np.pi / 6.0
        if (not np.isfinite(v_min_cmd)) or (v_min_cmd < 0.0):
            v_min_cmd = 0.0
        if (not np.isfinite(v_min_cmd_rev)) or (v_min_cmd_rev < 0.0):
            v_min_cmd_rev = v_min_cmd
        if (not np.isfinite(turn_crawl_speed)) or (turn_crawl_speed < 0.0):
            turn_crawl_speed = 0.0
        if (not np.isfinite(turn_crawl_angle)) or (turn_crawl_angle <= 1e-3):
            turn_crawl_angle = np.pi / 2.0
        if not np.isfinite(reverse_bias):
            reverse_bias = 0.05
        if (not np.isfinite(reverse_gate_angle)) or (reverse_gate_angle <= 1e-3):
            reverse_gate_angle = 0.45
        if (not np.isfinite(reverse_gate_power)) or (reverse_gate_power < 1.0):
            reverse_gate_power = 2.0
        return {
            "k_theta_p": k_theta_p,
            "k_theta_d": k_theta_d,
            "k_v_p": k_v_p,
            "k_v_d": k_v_d,
            "k_turn_boost": k_turn_boost,
            "turn_boost_angle": turn_boost_angle,
            "v_min_cmd": v_min_cmd,
            "v_min_cmd_rev": v_min_cmd_rev,
            "turn_crawl_speed": turn_crawl_speed,
            "turn_crawl_angle": turn_crawl_angle,
            "reverse_bias": reverse_bias,
            "reverse_gate_angle": reverse_gate_angle,
            "reverse_gate_power": reverse_gate_power,
            "tracking_mode": tracking_mode,
            "tracking_mode_code": 1 if tracking_mode == "projected" else 0,
        }

    def _uni_occ_vref_speed(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        s = cfg.get(
            "v_ref_norm_occ_uni",
            self.robot_spec.get("v_adv_max_occ", self.robot_spec.get("v_obs_max", 0.0)),
        )
        try:
            s = float(s)
        except Exception:
            return 0.0
        if (not np.isfinite(s)) or s <= 1e-9:
            return 0.0
        return s

    def _du_occ_gains(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        # Global DU emulation defaults (rounded to 2 decimals).
        k_a_p = float(cfg.get("k_a_occ_du_p", 2.99))
        k_a_d = float(cfg.get("k_a_occ_du_d", 0.98))
        k_brake = float(cfg.get("k_brake_occ_du", 0.31))
        # In emulation mode, default heading gains are synchronized to UNI defaults.
        # Optional *_emu_* keys allow explicit override.
        k_theta_emu_p = float(cfg.get("k_theta_emu_occ_du_p", 2.96))
        k_theta_emu_d = float(cfg.get("k_theta_emu_occ_du_d", 0.42))
        k_turn_boost_emu = float(cfg.get("k_turn_boost_emu_occ_du", 1.67))
        turn_boost_angle_emu = float(cfg.get("turn_boost_angle_emu_occ_du", 0.54))
        # UNI-emulation virtual-speed gains (for generating v_cmd_uni).
        k_vemu_p = float(cfg.get("k_v_emu_occ_du_p", 1.07))
        k_vemu_d = float(cfg.get("k_v_emu_occ_du_d", 0.22))
        # DU acceleration-tracking gains (track v_cmd_uni with acceleration).
        k_a_track_p = float(cfg.get("k_a_track_occ_du_p", k_a_p))
        k_a_track_d = float(cfg.get("k_a_track_occ_du_d", k_a_d))
        # UNI-equivalent minimum crawl speed at virtual-command level.
        v_min_cmd_emu = float(cfg.get("v_min_occ_du_cmd", 0.03))
        # Comparison-friendly option: keep DU speed nonnegative in emulation mode.
        du_nonnegative_speed = bool(cfg.get("du_nonnegative_speed_occ_du", False))

        if not np.isfinite(k_a_p):
            k_a_p = 2.99
        if not np.isfinite(k_a_d):
            k_a_d = 0.98
        if (not np.isfinite(k_brake)) or (k_brake < 0.0):
            k_brake = 0.31
        if not np.isfinite(k_theta_emu_p):
            k_theta_emu_p = 2.96
        if not np.isfinite(k_theta_emu_d):
            k_theta_emu_d = 0.42
        if not np.isfinite(k_turn_boost_emu):
            k_turn_boost_emu = 1.67
        if (not np.isfinite(turn_boost_angle_emu)) or (turn_boost_angle_emu <= 1e-3):
            turn_boost_angle_emu = 0.54
        if not np.isfinite(k_vemu_p):
            k_vemu_p = 1.07
        if not np.isfinite(k_vemu_d):
            k_vemu_d = 0.22
        if not np.isfinite(k_a_track_p):
            k_a_track_p = k_a_p
        if not np.isfinite(k_a_track_d):
            k_a_track_d = k_a_d
        if (not np.isfinite(v_min_cmd_emu)) or (v_min_cmd_emu < 0.0):
            v_min_cmd_emu = 0.0

        return {
            "k_theta_emu_p": k_theta_emu_p,
            "k_theta_emu_d": k_theta_emu_d,
            "k_a_track_p": k_a_track_p,
            "k_a_track_d": k_a_track_d,
            "k_turn_boost_emu": k_turn_boost_emu,
            "turn_boost_angle_emu": turn_boost_angle_emu,
            "k_vemu_p": k_vemu_p,
            "k_vemu_d": k_vemu_d,
            "v_min_cmd_emu": v_min_cmd_emu,
            "du_nonnegative_speed": du_nonnegative_speed,
            "k_brake": k_brake,
        }

    def _du_occ_vref_speed(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        # Emulation-first default:
        # 1) explicit DU override: v_ref_norm_occ_du
        # 2) default (when no compatibility flag is set): follow UNI speed policy
        # 3) compatibility / legacy override:
        #    - use_uni_vref_norm_default_occ_du=True  -> follow UNI speed policy
        #    - use_uni_vref_norm_default_occ_du=False -> use environment speed hints
        if "v_ref_norm_occ_du" in cfg:
            s = cfg.get("v_ref_norm_occ_du", 0.0)
        else:
            compat_key = "use_uni_vref_norm_default_occ_du"
            if compat_key in cfg:
                use_uni_default = bool(cfg.get(compat_key, True))
            else:
                use_uni_default = True

            if use_uni_default:
                # Match UNI behavior exactly (including fallback handling).
                s = self._uni_occ_vref_speed()
            else:
                # Legacy DU behavior: environment speed hint fallback.
                s = self.robot_spec.get("v_adv_max_occ", self.robot_spec.get("v_obs_max", 0.0))
        try:
            s = float(s)
        except Exception:
            return 0.0
        if (not np.isfinite(s)) or s <= 1e-9:
            return 0.0
        return s

    def _du_speed_bounds(self):
        v_max = float(self.robot_spec.get("v_max", 1.0))
        if not np.isfinite(v_max):
            v_max = 1.0
        # If v_min is not explicitly set, allow symmetric reverse speed by default.
        if "v_min" in self.robot_spec:
            v_min = float(self.robot_spec.get("v_min", -v_max))
        else:
            v_min = -v_max
        if not np.isfinite(v_min):
            v_min = -v_max
        if v_min > v_max:
            v_min = v_max
        return v_min, v_max

    def _use_occ_vref_normalization(self):
        # Final v_ref magnitude normalization has been retired.
        # Keep this helper so existing call-sites stay simple.
        return False

    def _occ_vref_scenario_softmax_kappa(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        kappa = float(cfg.get("vref_scenario_softmax_kappa", 8.0))
        if not np.isfinite(kappa):
            return 8.0
        return max(kappa, 0.0)

    def _occ_vref_mode(self):
        if self.model == "DoubleIntegrator2D":
            return OCBF_DEFAULT_VREF_TRACKING_MODE
        cfg = self.robot_spec.get("backup_cbf", {})
        if self.model == "Unicycle2D":
            mode = cfg.get("vref_mode_occ_uni", cfg.get("vref_mode_occ", OCBF_DEFAULT_VREF_TRACKING_MODE))
        elif self.model == "DynamicUnicycle2D":
            mode = cfg.get("vref_mode_occ_du", cfg.get("vref_mode_occ", OCBF_DEFAULT_VREF_TRACKING_MODE))
        else:
            mode = cfg.get("vref_mode_occ", OCBF_DEFAULT_VREF_TRACKING_MODE)
        return normalize_choice(mode, OCBF_VREF_TRACKING_MODES, OCBF_DEFAULT_VREF_TRACKING_MODE)

    def _occ_vref_front_mode(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        if self.model == "Unicycle2D":
            mode = cfg.get("vref_front_mode_occ_uni", cfg.get("vref_front_mode_occ", OCBF_DEFAULT_VREF_FRONT_MODE))
        elif self.model == "DynamicUnicycle2D":
            mode = cfg.get("vref_front_mode_occ_du", cfg.get("vref_front_mode_occ", OCBF_DEFAULT_VREF_FRONT_MODE))
        else:
            mode = cfg.get("vref_front_mode_occ", OCBF_DEFAULT_VREF_FRONT_MODE)
        return normalize_choice(mode, OCBF_VREF_FRONT_MODES, OCBF_DEFAULT_VREF_FRONT_MODE)

    def _use_strict_occ_vref(self):
        return self._occ_vref_mode() == "strict"

    def _occ_target_normals(self, scenario, p):
        A = np.asarray(scenario["A"], dtype=float)
        if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] == 0:
            return A
        if self._occ_vref_front_mode() != "los":
            return A
        c = scenario.get("obs_center", None)
        if c is None:
            return A
        c = np.asarray(c, dtype=float).reshape(2,)
        p = np.asarray(p, dtype=float).reshape(2,)
        los = p - c
        n = float(np.linalg.norm(los))
        if (not np.isfinite(n)) or n <= 1e-9:
            return A
        A_eff = A.copy()
        A_eff[0] = los / n
        return A_eff

    def _scale_occ_vref_rollout(self, X, occlusion_scenarios, tau, ref_speed):
        del ref_speed
        return np.asarray(
            self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, tau),
            dtype=float,
        ).reshape(2,)
    def _uni_virtual_cmd_from_vref(self, theta, v_ref, v_ref_prev, gains=None, v_min=None, v_max=None, w_max=None):
        v_ref = np.asarray(v_ref, dtype=float).reshape(2,)
        v_ref_prev = np.asarray(v_ref_prev, dtype=float).reshape(2,)
        v_ref_norm = float(np.linalg.norm(v_ref))
        v_ref_prev_norm = float(np.linalg.norm(v_ref_prev))
        if v_ref_norm < 1e-9:
            return 0.0, 0.0

        if gains is None:
            gains = self._uni_occ_gains()

        if v_max is None:
            v_max = float(self.robot_spec.get("v_max", 1.0))
        if v_min is None:
            v_min = float(self.robot_spec.get("v_min", -v_max))
        if w_max is None:
            w_max = float(self.robot_spec.get("w_max", 0.8))

        reverse_bias = float(gains.get("reverse_bias", 0.05))
        v_min_cmd_rev = float(gains.get("v_min_cmd_rev", gains["v_min_cmd"]))
        reverse_gate_angle = float(gains.get("reverse_gate_angle", 0.45))
        reverse_gate_power = float(gains.get("reverse_gate_power", 2.0))
        if (not np.isfinite(reverse_gate_angle)) or (reverse_gate_angle <= 1e-3):
            reverse_gate_angle = 0.45
        if (not np.isfinite(reverse_gate_power)) or (reverse_gate_power < 1.0):
            reverse_gate_power = 2.0

        theta_ref = float(np.arctan2(v_ref[1], v_ref[0]))
        theta_ref_prev = float(np.arctan2(v_ref_prev[1], v_ref_prev[0]))

        e_fwd = angle_normalize(theta_ref - float(theta))
        e_rev = angle_normalize(theta_ref + np.pi - float(theta))
        allow_reverse = v_min < -1e-9
        use_reverse = allow_reverse and ((abs(e_rev) + reverse_bias) < abs(e_fwd))
        e_track = e_rev if use_reverse else e_fwd

        e_fwd_prev = angle_normalize(theta_ref_prev - float(theta))
        e_rev_prev = angle_normalize(theta_ref_prev + np.pi - float(theta))
        use_reverse_prev = allow_reverse and ((abs(e_rev_prev) + reverse_bias) < abs(e_fwd_prev))
        e_track_prev = e_rev_prev if use_reverse_prev else e_fwd_prev

        dt_safe = max(float(self.dt), 1e-6)
        e_track_dot = angle_normalize(e_track - e_track_prev) / dt_safe

        boost = 1.0 + gains["k_turn_boost"] * min(
            1.0, abs(e_track) / max(gains["turn_boost_angle"], 1e-3)
        )
        omega_unsat = boost * (gains["k_theta_p"] * e_track + gains["k_theta_d"] * e_track_dot)
        omega = float(np.clip(omega_unsat, -w_max, w_max))

        fwd_gate = max(0.0, np.cos(e_fwd))
        fwd_gate_prev = max(0.0, np.cos(e_fwd_prev))
        rev_gate = max(0.0, 1.0 - abs(e_rev) / reverse_gate_angle) ** reverse_gate_power
        rev_gate_prev = max(0.0, 1.0 - abs(e_rev_prev) / reverse_gate_angle) ** reverse_gate_power
        rev_cap = max(0.0, abs(min(v_min, 0.0)))
        turn_crawl_speed = float(np.clip(gains.get("turn_crawl_speed", 0.0), 0.0, v_max))
        turn_crawl_angle = max(float(gains.get("turn_crawl_angle", np.pi / 2.0)), 1e-3)

        if use_reverse:
            v_des = -float(np.clip(v_ref_norm * rev_gate, 0.0, rev_cap))
        else:
            v_des = float(np.clip(v_ref_norm * fwd_gate, 0.0, v_max))
            if v_ref_norm > 1e-9 and abs(e_fwd) <= turn_crawl_angle:
                v_des = max(v_des, turn_crawl_speed)

        if use_reverse_prev:
            v_des_prev = -float(np.clip(v_ref_prev_norm * rev_gate_prev, 0.0, rev_cap))
        else:
            v_des_prev = float(np.clip(v_ref_prev_norm * fwd_gate_prev, 0.0, v_max))
            if v_ref_prev_norm > 1e-9 and abs(e_fwd_prev) <= turn_crawl_angle:
                v_des_prev = max(v_des_prev, turn_crawl_speed)
        v_des_dot = (v_des - v_des_prev) / dt_safe

        v_cmd_unsat = gains["k_v_p"] * v_des + gains["k_v_d"] * v_des_dot
        v_cmd = float(np.clip(v_cmd_unsat, v_min, v_max))

        # optional crawl floor
        if v_cmd > 1e-9:
            v_cmd = max(v_cmd, min(gains["v_min_cmd"], v_max))
        elif v_cmd < -1e-9:
            v_cmd = min(v_cmd, -min(v_min_cmd_rev, abs(v_min)))

        if str(gains.get("tracking_mode", "gated")).strip().lower() == "projected":
            heading = np.array([np.cos(float(theta)), np.sin(float(theta))], dtype=float)
            v_des_proj = float(np.dot(v_ref, heading))
            v_des_proj_prev = float(np.dot(v_ref_prev, heading))
            if v_min < -1e-9:
                v_des_proj = float(np.clip(v_des_proj, v_min, v_max))
                v_des_proj_prev = float(np.clip(v_des_proj_prev, v_min, v_max))
            else:
                v_des_proj = float(np.clip(max(v_des_proj, 0.0), 0.0, v_max))
                v_des_proj_prev = float(np.clip(max(v_des_proj_prev, 0.0), 0.0, v_max))

            use_reverse_proj = (v_min < -1e-9) and (v_des_proj < -1e-9)
            use_reverse_proj_prev = (v_min < -1e-9) and (v_des_proj_prev < -1e-9)
            e_proj = e_rev if use_reverse_proj else e_fwd
            e_proj_prev = e_rev_prev if use_reverse_proj_prev else e_fwd_prev
            if (not use_reverse_proj) and v_ref_norm > 1e-9 and abs(e_proj) <= turn_crawl_angle:
                v_des_proj = max(v_des_proj, turn_crawl_speed)
            if (not use_reverse_proj_prev) and v_ref_prev_norm > 1e-9 and abs(e_proj_prev) <= turn_crawl_angle:
                v_des_proj_prev = max(v_des_proj_prev, turn_crawl_speed)
            e_proj_dot = angle_normalize(e_proj - e_proj_prev) / dt_safe

            boost_proj = 1.0 + gains["k_turn_boost"] * min(
                1.0, abs(e_proj) / max(gains["turn_boost_angle"], 1e-3)
            )
            omega_proj = float(
                np.clip(
                    boost_proj * (gains["k_theta_p"] * e_proj + gains["k_theta_d"] * e_proj_dot),
                    -w_max,
                    w_max,
                )
            )
            v_cmd_proj = float(
                np.clip(
                    gains["k_v_p"] * v_des_proj + gains["k_v_d"] * ((v_des_proj - v_des_proj_prev) / dt_safe),
                    v_min,
                    v_max,
                )
            )
            if v_cmd_proj > 1e-9:
                v_cmd_proj = max(v_cmd_proj, min(gains["v_min_cmd"], v_max))
            elif v_cmd_proj < -1e-9:
                v_cmd_proj = min(v_cmd_proj, -min(v_min_cmd_rev, abs(v_min)))
            return v_cmd_proj, omega_proj
        return v_cmd, omega

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

    def _occ_barrier_smoothing_kappa(self):
        kappa = getattr(self, "kappa", None)
        if kappa is None:
            fn = getattr(self, "_occ_barrier_fn", None)
            owner = getattr(fn, "__self__", None)
            if owner is not None:
                kappa = getattr(owner, "kappa", None)
        if kappa is None:
            kappa = self.robot_spec.get("occ_kappa", OCBF_DEFAULT_BARRIER_KAPPA)
        try:
            kappa = float(kappa)
        except Exception:
            kappa = OCBF_DEFAULT_BARRIER_KAPPA
        if (not np.isfinite(kappa)) or kappa <= 0.0:
            kappa = OCBF_DEFAULT_BARRIER_KAPPA
        return kappa

    def _occ_scenario_barrier_margin_from_hvec(self, h_vec):
        h_vec = np.asarray(h_vec, dtype=float).reshape(-1)
        if h_vec.size == 0 or (not np.all(np.isfinite(h_vec))):
            return None
        kappa = float(self._occ_barrier_smoothing_kappa())
        max_h = float(np.max(h_vec))
        z = np.exp(kappa * (h_vec - max_h))
        Z = float(np.sum(z))
        if (not np.isfinite(Z)) or Z <= 0.0:
            return None
        K = float(max(1, h_vec.size))
        return max_h + (np.log(Z) - np.log(K)) / kappa

    def _occ_scenario_threat_score_from_hvec(self, h_vec):
        h_tilde = self._occ_scenario_barrier_margin_from_hvec(h_vec)
        if h_tilde is None:
            return None
        return -float(h_tilde)

    def _occ_vref_scenario_weight_mode(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        return normalize_choice(
            cfg.get("vref_scenario_weight_mode", OCBF_DEFAULT_VREF_SCENARIO_WEIGHT_MODE),
            OCBF_VREF_SCENARIO_WEIGHT_MODES,
            OCBF_DEFAULT_VREF_SCENARIO_WEIGHT_MODE,
        )

    def _occ_vref_weight_mode_code(self):
        mode = self._occ_vref_scenario_weight_mode()
        if mode == "barrier_unexpand":
            return 1
        return 0

    def _scenario_unexpanded_margin(self, scenario, p=None):
        if self._occ_vref_scenario_weight_mode() != "barrier_unexpand":
            return None
        score_scenario = scenario.get("score_scenario", scenario)
        if not isinstance(score_scenario, dict):
            return None
        A = np.asarray(score_scenario.get("A", []), dtype=float)
        b0 = np.asarray(score_scenario.get("b0", []), dtype=float).reshape(-1)
        if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] == 0 or b0.size == 0:
            return None
        if p is None:
            return None
        p_eval = np.asarray(p, dtype=float).reshape(2,)
        R = float(self.robot_spec.get("radius", 0.25))
        m = min(int(A.shape[0]), int(b0.size))
        h_vec_score = (A[:m, :] @ p_eval) - b0[:m] - R
        return self._occ_scenario_barrier_margin_from_hvec(h_vec_score)

    def _occ_scenario_score_from_hvec(self, h_vec, X, scenario, p=None):
        h_tilde = self._occ_scenario_barrier_margin_from_hvec(h_vec)
        if h_tilde is None:
            return None, None
        expanded_score = -float(h_tilde)
        unexpanded_h = self._scenario_unexpanded_margin(scenario, p=p)
        score = -float(unexpanded_h) if unexpanded_h is not None else expanded_score
        return float(score), {
            "h_tilde": float(h_tilde),
            "expanded_score": float(expanded_score),
            "unexpanded_h_tilde": None if unexpanded_h is None else float(unexpanded_h),
            "unexpanded_score": None if unexpanded_h is None else -float(unexpanded_h),
            "combined_score": float(score),
        }

    def occ_vref_scenario_weight_debug(self, X, scenarios, t=0.0):
        if (scenarios is None) or (len(scenarios) == 0):
            return {
                "weight_mode": self._occ_vref_scenario_weight_mode(),
                "scenarios": [],
                "avg_unexpanded_margin": None,
                "max_softmax_weight": None,
            }

        X_arr = np.asarray(X, dtype=float).reshape(self._n_state, 1)
        p = X_arr[0:2, 0]
        R = float(self.robot_spec.get("radius", 0.25))
        tau = self.clamp_tau(float(t) if t is not None else 0.0)
        scores = []
        entries = []
        for idx, sc in enumerate(list(scenarios)):
            A = np.asarray(sc.get("A", []), dtype=float)
            b0 = np.asarray(sc.get("b0", []), dtype=float).reshape(-1)
            if A.ndim != 2 or A.shape[1] != 2 or b0.size == 0:
                continue
            vadv = float(sc.get("v_adv_max", 1.0))
            v_expand = np.asarray(sc.get("v_expand_vec", np.full(len(b0), vadv)), dtype=float).reshape(-1)
            if v_expand.size < b0.size:
                v_expand = np.pad(v_expand, (0, b0.size - v_expand.size), constant_values=vadv)
            h_vec = (A @ p) - b0 - (R + v_expand[: b0.size] * tau)
            score, dbg = self._occ_scenario_score_from_hvec(h_vec, X_arr, sc, p=p)
            if score is None or dbg is None:
                continue
            dbg["scenario_index"] = int(idx)
            dbg["source_visible_index"] = (
                None if sc.get("source_visible_index", None) is None else int(sc.get("source_visible_index"))
            )
            entries.append(dbg)
            scores.append(float(score))

        if len(scores) > 0:
            scores_arr = np.asarray(scores, dtype=float)
            kappa = self._occ_vref_scenario_softmax_kappa()
            max_score = float(np.max(scores_arr))
            z = np.exp(kappa * (scores_arr - max_score))
            z_sum = float(np.sum(z))
            weights = z / z_sum if np.isfinite(z_sum) and z_sum > 1e-12 else np.zeros_like(scores_arr)
            for entry, weight in zip(entries, weights):
                entry["softmax_weight"] = float(weight)
            unexpanded_margins = [
                float(e["unexpanded_h_tilde"])
                for e in entries
                if e.get("unexpanded_h_tilde", None) is not None
            ]
            avg_unexpanded_margin = float(np.mean(unexpanded_margins)) if unexpanded_margins else None
            max_weight = float(np.max(weights)) if weights.size else None
        else:
            avg_unexpanded_margin = None
            max_weight = None

        return {
            "weight_mode": self._occ_vref_scenario_weight_mode(),
            "scenarios": entries,
            "avg_unexpanded_margin": avg_unexpanded_margin,
            "max_softmax_weight": max_weight,
        }

    def _occ_safe_velocity_reference_rollout(self, X, scenarios, t):
        return self._occ_barrier_velocity_reference_rollout(X, scenarios, t)

    def _occ_barrier_velocity_reference_rollout(self, X, scenarios, t):

        if (scenarios is None) or (len(scenarios) == 0):
            return np.zeros(2, dtype=float)

        p = np.array([float(X[0,0]), float(X[1,0])], dtype=float)
        R     = self.robot_spec['radius']

        use_strict = self._use_strict_occ_vref()
        scenario_kappa = self._occ_vref_scenario_softmax_kappa()

        if use_strict:
            v_targets = []
            scenario_scores = []
            for sc in scenarios:
                A = sc["A"]
                b0 = sc["b0"]
                vadv = float(sc["v_adv_max"])
                A_target = self._occ_target_normals(sc, p)

                if "v_expand_vec" in sc:
                    v_expand = sc["v_expand_vec"]
                else:
                    v_expand = np.full(len(b0), vadv)

                tau = float(t) if t is not None else 0.0
                tau = self.clamp_tau(tau)

                delta = R + v_expand * tau
                h_vec = (A @ p) - b0 - delta
                if h_vec.size == 0:
                    continue

                active = h_vec >= 0.0
                if np.any(active):
                    avg_normal = A_target[active].mean(axis=0)
                else:
                    best_idx = int(np.argmax(h_vec))
                    avg_normal = A_target[best_idx]

                n = float(np.linalg.norm(avg_normal))
                if n <= 1e-9:
                    continue
                direction = avg_normal / n
                threat_score, _ = self._occ_scenario_score_from_hvec(h_vec, X, sc, p=p)
                if threat_score is None:
                    continue
                v_targets.append(direction * vadv)
                scenario_scores.append(threat_score)

            if len(v_targets) == 0:
                return np.zeros(2, dtype=float)

            scores = np.asarray(scenario_scores, dtype=float)
            max_score = float(np.max(scores))
            z = np.exp(scenario_kappa * (scores - max_score))
            z_sum = float(np.sum(z))
            if (not np.isfinite(z_sum)) or z_sum <= 1e-12:
                return np.zeros(2, dtype=float)
            weights = z / z_sum
            v_avg = (weights[:, None] * np.vstack(v_targets)).sum(axis=0)
        else:
            kappa_vref = float(
                self.robot_spec.get("backup_cbf", {}).get("vref_softmax_kappa", 8.0)
            )
            if (not np.isfinite(kappa_vref)) or (kappa_vref <= 1e-6):
                kappa_vref = 8.0

            v_targets = []
            scenario_scores = []
            for sc in scenarios:
                A    = sc['A']
                b0   = sc['b0']
                vadv = sc['v_adv_max']
                A_target = self._occ_target_normals(sc, p)

                if 'v_expand_vec' in sc:
                    v_expand = sc['v_expand_vec']
                else:
                    v_expand = np.full(len(b0), sc['v_adv_max'])

                tau = float(t) if t is not None else 0.0
                tau = self.clamp_tau(tau)

                delta = R + v_expand * tau
                h_vec = (A @ p) - b0 - delta

                if h_vec.size == 0:
                    continue
                max_h = float(np.max(h_vec))
                z = np.exp(kappa_vref * (h_vec - max_h))
                z_sum = float(np.sum(z))
                if (not np.isfinite(z_sum)) or z_sum <= 1e-12:
                    continue
                lam = z / z_sum
                avg_normal = (lam[:, None] * A_target).sum(axis=0)
                norm = float(np.linalg.norm(avg_normal))
                if norm > 1e-9:
                    direction = avg_normal / norm
                    threat_score, _ = self._occ_scenario_score_from_hvec(h_vec, X, sc, p=p)
                    if threat_score is None:
                        continue
                    v_targets.append(direction * vadv)
                    scenario_scores.append(threat_score)

            if len(v_targets) == 0:
                return np.zeros(2, dtype=float)

            scores = np.asarray(scenario_scores, dtype=float)
            max_score = float(np.max(scores))
            z = np.exp(scenario_kappa * (scores - max_score))
            z_sum = float(np.sum(z))
            if (not np.isfinite(z_sum)) or z_sum <= 1e-12:
                return np.zeros(2, dtype=float)
            weights = z / z_sum
            v_avg = (weights[:, None] * np.vstack(v_targets)).sum(axis=0)

        return v_avg

    def f_cl(self, X, occlusion_scenarios=None, t=None):
        """
        System dynamics as using backup policy u_b (Closed-Loop)
        """
        X = np.asarray(X, dtype=float).reshape(self._n_state, 1)
        if occlusion_scenarios:
            u_b = self.backup_input_occlusion(X, occlusion_scenarios, t)
        else:
            u_b = self.backup_input(X)

        if self.model == "DoubleIntegrator2D":
            return np.array([X[2,0], X[3,0], u_b[0,0], u_b[1,0]], dtype=float).reshape(4,1)

        if self.model == "Unicycle2D":
            theta = float(X[2, 0])
            v_cmd = float(u_b[0, 0])
            w_cmd = float(u_b[1, 0])
            return np.array(
                [v_cmd * np.cos(theta), v_cmd * np.sin(theta), w_cmd],
                dtype=float,
            ).reshape(3, 1)

        if self.model == "DynamicUnicycle2D":
            theta = float(X[2, 0])
            v = float(X[3, 0])
            a_cmd = float(u_b[0, 0])
            w_cmd = float(u_b[1, 0])
            return np.array(
                [v * np.cos(theta), v * np.sin(theta), w_cmd, a_cmd],
                dtype=float,
            ).reshape(4, 1)

        raise NotImplementedError(f"Unsupported model: {self.model}")

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
        X = np.asarray(X, dtype=float).reshape(self._n_state, 1)
        if self.model == "DoubleIntegrator2D":
            Kp = float(self.pid_occ_gains.get("Kp", 1.0))
            k_d = float(self.pid_occ_gains.get("Kd", 1.0))
            Bv = -(Kp + k_d) * np.eye(2)
            if occlusion_scenarios is not None and t is not None:
                Jp = self._dvref_dp_fd(X, occlusion_scenarios, t)
            else:
                Jp = np.zeros((2, 2))
            lower_left = Kp * Jp
            return np.block(
                [
                    [np.zeros((2, 2)), np.eye(2)],
                    [lower_left, Bv],
                ]
            )

        # For non-holonomic backup dynamics, use a local FD Jacobian of f_cl.
        eps = float(self.robot_spec.get("backup_cbf", {}).get("fd_eps_fcl", 1e-4))
        f0 = self.f_cl(X, occlusion_scenarios, t)
        A = np.zeros((self._n_state, self._n_state), dtype=float)
        for j in range(self._n_state):
            Xp = X.copy()
            Xp[j, 0] += eps
            fp = self.f_cl(Xp, occlusion_scenarios, t)
            A[:, j] = ((fp - f0) / eps).reshape(-1)
        return A

    def set_terminal_backup_context(self, occlusion_scenario, T, kappa=None, rho_T=0.05):
        self._term_occ_scenario = occlusion_scenario
        self._term_T = float(T)
        if kappa is not None:
            self.kappa = kappa
        self._term_rho = float(rho_T)
        self._term_grad_cache = None

    def _terminal_rho(self):
        # Unicycle directly commands speed, so stopping-distance margin can be 0 by default.
        if self.model == "Unicycle2D":
            return 0.0
        a_lim = float(self.robot_spec.get("a_max", 1.0))
        v_max = float(self.robot_spec.get("v_max", 1.0))
        if a_lim <= 0.0:
            return 0.0
        return v_max**2 / (2.0 * a_lim)

    def h_b_stop(self, X):

        p_T = X[0:2, 0].astype(float)

        scenario = getattr(self, "_term_occ_scenario", None)
        T = float(getattr(self, "_term_T", self.get_backup_horizon()))
        rho_T = getattr(self, "_term_rho", None)
        if rho_T is None:
            rho_T = self._terminal_rho()

        if scenario is not None:
            h_tilde, grad_pos, _ = self._occ_barrier(p_T.reshape(2, 1), scenario, tau=T)
            self._term_grad_cache = (grad_pos.reshape(1,2) if grad_pos is not None else None)
            return float(h_tilde) - float(rho_T)

    def _occ_terminal_set(self, X):
        return self.h_b_stop(X)

    def grad_h_b_stop(self, X):
        del X
        gp = getattr(self, "_term_grad_cache", None)
        if gp is not None:
            if gp.shape == (1,2):
                # Expand to full backup-state dimension: [grad_pos, 0, ..., 0]
                pad = np.zeros((1, max(0, self._n_state - 2)), dtype=float)
                return np.hstack([gp, pad])
            return gp

    def grad_occ_terminal(self, X):
        return self.grad_h_b_stop(X)

    def backup_input_occlusion(self, X, occlusion_scenarios, t=None,
                            k_d=1.0, k_occ=1.0):
        del k_occ

        X = np.asarray(X, dtype=float).reshape(self._n_state, 1)

        if self.model == "DoubleIntegrator2D":
            v_ref = self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, t)
            a_lim = float(self.robot_spec.get("a_max", 1.0))
            v_max = float(self.robot_spec.get("v_max", 1.0))
            Kp = float(self.pid_occ_gains.get("Kp", 1.0))
            k_d = float(self.pid_occ_gains.get("Kd", k_d))

            v = np.array([float(X[2, 0]), float(X[3, 0])], dtype=float)
            e = v - v_ref
            u_unsat = -Kp * e - k_d * v
            u = np.clip(u_unsat, -a_lim, a_lim)

            # limit acceleration when near v_max
            eps = 1e-6
            for i in range(2):
                if v[i] >= (v_max - eps) and u[i] > 0.0:
                    u[i] = 0.0
                if v[i] <= (-v_max + eps) and u[i] < 0.0:
                    u[i] = 0.0
            return u.reshape(2, 1)

        if self.model == "Unicycle2D":
            # Unicycle2D: map facet-normal velocity reference to (v, omega)
            theta = float(X[2, 0])
            tau_now = self.clamp_tau(float(t) if t is not None else 0.0)
            tau_prev = self.clamp_tau(max(0.0, float(tau_now) - float(self.dt)))
            v_ref = self._scale_occ_vref_rollout(X, occlusion_scenarios, tau_now, self._uni_occ_vref_speed())
            v_ref_prev = self._scale_occ_vref_rollout(X, occlusion_scenarios, tau_prev, self._uni_occ_vref_speed())
            v_cmd, omega = self._uni_virtual_cmd_from_vref(theta, v_ref, v_ref_prev)
            return np.array([[v_cmd], [omega]], dtype=float)

        if self.model == "DynamicUnicycle2D":
            v_cur = float(X[3, 0])
            tau_now = self.clamp_tau(float(t) if t is not None else 0.0)
            tau_prev = self.clamp_tau(max(0.0, float(tau_now) - float(self.dt)))
            tau_prev2 = self.clamp_tau(max(0.0, float(tau_prev) - float(self.dt)))

            a_max = float(self.robot_spec.get("a_max", 1.0))
            v_min, v_max = self._du_speed_bounds()
            gains = self._du_occ_gains()
            dt_safe = max(float(self.dt), 1e-6)
            v_floor = 0.0 if gains.get("du_nonnegative_speed", True) else v_min
            a_lb_base = max(-a_max, (v_floor - v_cur) / dt_safe)
            a_ub_base = min(a_max, (v_max - v_cur) / dt_safe)
            if a_lb_base > a_ub_base:
                a_mid = 0.5 * (a_lb_base + a_ub_base)
                a_lb_base, a_ub_base = a_mid, a_mid

            v_ref = self._scale_occ_vref_rollout(X, occlusion_scenarios, tau_now, self._du_occ_vref_speed())
            v_ref_prev = self._scale_occ_vref_rollout(X, occlusion_scenarios, tau_prev, self._du_occ_vref_speed())
            v_ref_prev2 = self._scale_occ_vref_rollout(X, occlusion_scenarios, tau_prev2, self._du_occ_vref_speed())
            v_ref_norm = float(np.linalg.norm(v_ref))

            if v_ref_norm < 1e-9:
                a_stop = float(np.clip(-gains["k_brake"] * v_cur, a_lb_base, a_ub_base))
                return np.array([[a_stop], [0.0]], dtype=float)

            theta = float(X[2, 0])
            emu_gains = {
                "k_theta_p": gains["k_theta_emu_p"],
                "k_theta_d": gains["k_theta_emu_d"],
                "k_v_p": gains["k_vemu_p"],
                "k_v_d": gains["k_vemu_d"],
                "k_turn_boost": gains["k_turn_boost_emu"],
                "turn_boost_angle": gains["turn_boost_angle_emu"],
                "v_min_cmd": gains["v_min_cmd_emu"],
            }
            v_cmd, omega = self._uni_virtual_cmd_from_vref(
                theta, v_ref, v_ref_prev, gains=emu_gains, v_max=v_max, w_max=float(self.robot_spec.get("w_max", 0.8))
            )
            v_cmd_prev, _ = self._uni_virtual_cmd_from_vref(
                theta, v_ref_prev, v_ref_prev2, gains=emu_gains, v_max=v_max, w_max=float(self.robot_spec.get("w_max", 0.8))
            )
            v_cmd_dot = (v_cmd - v_cmd_prev) / dt_safe
            a_cmd_unsat = gains["k_a_track_p"] * (v_cmd - v_cur) + gains["k_a_track_d"] * v_cmd_dot
            a_cmd = float(np.clip(a_cmd_unsat, a_lb_base, a_ub_base))
            return np.array([[a_cmd], [omega]], dtype=float)

        raise NotImplementedError(f"Unsupported model: {self.model}")

    def _pack_scenarios_for_jax(self, occlusion_scenarios, x_ref=None):
        del x_ref
        S = int(self._jax_max_scenarios)
        K = int(self._jax_max_halfspaces)

        A_pad = np.zeros((S, K, 2), dtype=np.float32)
        b0_pad = np.zeros((S, K), dtype=np.float32)
        v_expand_pad = np.zeros((S, K), dtype=np.float32)
        v_adv_vec = np.zeros((S,), dtype=np.float32)
        valid_mask = np.zeros((S, K), dtype=np.float32)
        obs_center_pad = np.zeros((S, 2), dtype=np.float32)
        obs_center_valid = np.zeros((S,), dtype=np.float32)

        if not occlusion_scenarios:
            return A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, obs_center_pad, obs_center_valid

        for i, sc in enumerate(list(occlusion_scenarios)[:S]):
            if not isinstance(sc, dict):
                continue
            A = np.asarray(sc.get("A", []), dtype=np.float32)
            b0 = np.asarray(sc.get("b0", []), dtype=np.float32).reshape(-1)
            if A.ndim != 2 or A.shape[1] != 2 or b0.size == 0:
                continue
            m = min(K, int(A.shape[0]), int(b0.size))
            if m <= 0:
                continue

            v_adv = float(sc.get("v_adv_max", 0.0))
            v_expand = sc.get("v_expand_vec", None)
            if v_expand is None:
                v_expand_arr = np.full((m,), v_adv, dtype=np.float32)
            else:
                v_expand_arr = np.asarray(v_expand, dtype=np.float32).reshape(-1)
                if v_expand_arr.size == 1:
                    v_expand_arr = np.full((m,), float(v_expand_arr.item()), dtype=np.float32)
                elif v_expand_arr.size < m:
                    pad = np.full((m - v_expand_arr.size,), v_adv, dtype=np.float32)
                    v_expand_arr = np.concatenate([v_expand_arr, pad], axis=0)
                else:
                    v_expand_arr = v_expand_arr[:m]

            A_pad[i, :m, :] = A[:m, :]
            b0_pad[i, :m] = b0[:m]
            v_expand_pad[i, :m] = v_expand_arr[:m]
            v_adv_vec[i] = v_adv
            valid_mask[i, :m] = 1.0
            obs_center = sc.get("obs_center", None)
            if obs_center is not None:
                obs_center_arr = np.asarray(obs_center, dtype=np.float32).reshape(-1)
                if obs_center_arr.size >= 2 and np.all(np.isfinite(obs_center_arr[:2])):
                    obs_center_pad[i, :] = obs_center_arr[:2]
                    obs_center_valid[i] = 1.0

        return A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, obs_center_pad, obs_center_valid

    def _pack_score_scenarios_for_jax(self, occlusion_scenarios):
        S = int(self._jax_max_scenarios)
        K = int(self._jax_max_halfspaces)

        A_pad = np.zeros((S, K, 2), dtype=np.float32)
        b0_pad = np.zeros((S, K), dtype=np.float32)
        valid_mask = np.zeros((S, K), dtype=np.float32)
        present = np.zeros((S,), dtype=np.float32)

        if not occlusion_scenarios:
            return A_pad, b0_pad, valid_mask, present

        for i, sc in enumerate(list(occlusion_scenarios)[:S]):
            if not isinstance(sc, dict):
                continue
            score_sc = sc.get("score_scenario", sc)
            if not isinstance(score_sc, dict):
                continue
            A = np.asarray(score_sc.get("A", []), dtype=np.float32)
            b0 = np.asarray(score_sc.get("b0", []), dtype=np.float32).reshape(-1)
            if A.ndim != 2 or A.shape[1] != 2 or b0.size == 0:
                continue
            m = min(K, int(A.shape[0]), int(b0.size))
            if m <= 0:
                continue
            A_pad[i, :m, :] = A[:m, :]
            b0_pad[i, :m] = b0[:m]
            valid_mask[i, :m] = 1.0
            present[i] = 1.0

        return A_pad, b0_pad, valid_mask, present

    def _warm_start_jax_rollout(self, n_steps):
        if not self._jax_enabled:
            return
        try:
            A_pad = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces, 2), dtype=np.float32)
            b0_pad = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces), dtype=np.float32)
            v_expand_pad = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces), dtype=np.float32)
            v_adv_vec = np.zeros((self._jax_max_scenarios,), dtype=np.float32)
            valid_mask = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces), dtype=np.float32)
            obs_center_pad = np.zeros((self._jax_max_scenarios, 2), dtype=np.float32)
            obs_center_valid = np.zeros((self._jax_max_scenarios,), dtype=np.float32)
            score_A_pad = np.zeros_like(A_pad)
            score_b0_pad = np.zeros_like(b0_pad)
            score_valid_mask = np.zeros_like(valid_mask)
            score_present = np.zeros((self._jax_max_scenarios,), dtype=np.float32)
            front_mode_los = np.bool_(self._occ_vref_front_mode() == "los")
            barrier_kappa = np.float32(self._occ_barrier_smoothing_kappa())

            # Compile ahead of control loop; no JIT should happen inside step loop.
            if self.model == "DoubleIntegrator2D":
                x0 = np.zeros((4,), dtype=np.float32)
                scenario_kappa = np.float32(self._occ_vref_scenario_softmax_kappa())
                kd_occ = np.float32(self.pid_occ_gains.get("Kd", 1.0))
                traj, _, _, _ = _jax_rollout_kernel_di(
                    jnp.asarray(x0),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    jnp.asarray(obs_center_pad),
                    jnp.asarray(obs_center_valid),
                    front_mode_los,
                    np.float32(self.dt),
                    int(n_steps),
                    np.float32(self.pid_occ_gains.get("Kp", 1.0)),
                    kd_occ,
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    barrier_kappa,
                    scenario_kappa,
                    np.float32(self._jax_eps_fd),
                    jnp.asarray(score_A_pad),
                    jnp.asarray(score_b0_pad),
                    jnp.asarray(score_valid_mask),
                    jnp.asarray(score_present),
                    np.int32(self._occ_vref_weight_mode_code()),
                )
            elif self.model == "Unicycle2D":
                x0 = np.zeros((3,), dtype=np.float32)
                ug = self._uni_occ_gains()
                v_max = float(self.robot_spec.get("v_max", 1.0))
                v_min = float(self.robot_spec.get("v_min", -v_max))
                scenario_kappa = np.float32(self._occ_vref_scenario_softmax_kappa())
                traj, _, _, _ = _jax_rollout_kernel_uni(
                    jnp.asarray(x0),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    jnp.asarray(obs_center_pad),
                    jnp.asarray(obs_center_valid),
                    front_mode_los,
                    np.float32(self.dt),
                    int(n_steps),
                    np.float32(v_min),
                    np.float32(v_max),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(ug["k_theta_p"]),
                    np.float32(ug["k_theta_d"]),
                    np.float32(ug["k_v_p"]),
                    np.float32(ug["k_v_d"]),
                    np.float32(ug["k_turn_boost"]),
                    np.float32(ug["turn_boost_angle"]),
                    np.float32(ug["v_min_cmd"]),
                    np.float32(ug.get("v_min_cmd_rev", ug["v_min_cmd"])),
                    np.float32(ug.get("turn_crawl_speed", 0.0)),
                    np.float32(ug.get("turn_crawl_angle", np.pi / 2.0)),
                    np.float32(ug.get("reverse_bias", 0.05)),
                    np.float32(ug.get("reverse_gate_angle", 0.45)),
                    np.float32(ug.get("reverse_gate_power", 2.0)),
                    np.int32(ug.get("tracking_mode_code", 0)),
                    np.float32(self._uni_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    np.bool_(self._use_strict_occ_vref()),
                    np.bool_(self._use_occ_vref_normalization()),
                    barrier_kappa,
                    scenario_kappa,
                    jnp.asarray(score_A_pad),
                    jnp.asarray(score_b0_pad),
                    jnp.asarray(score_valid_mask),
                    jnp.asarray(score_present),
                    np.int32(self._occ_vref_weight_mode_code()),
                    np.float32(getattr(self, "last_speed", 0.0)),
                )
            else:
                x0 = np.zeros((4,), dtype=np.float32)
                dg = self._du_occ_gains()
                v_min, v_max = self._du_speed_bounds()
                scenario_kappa = np.float32(self._occ_vref_scenario_softmax_kappa())
                traj, _, _, _ = _jax_rollout_kernel_du(
                    jnp.asarray(x0),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    jnp.asarray(obs_center_pad),
                    jnp.asarray(obs_center_valid),
                    front_mode_los,
                    np.float32(self.dt),
                    int(n_steps),
                    np.float32(v_min),
                    np.float32(v_max),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(dg["k_theta_emu_p"]),
                    np.float32(dg["k_theta_emu_d"]),
                    np.float32(dg["k_turn_boost_emu"]),
                    np.float32(dg["turn_boost_angle_emu"]),
                    np.float32(dg["k_a_track_p"]),
                    np.float32(dg["k_a_track_d"]),
                    np.float32(dg["k_brake"]),
                    np.float32(dg["k_vemu_p"]),
                    np.float32(dg["k_vemu_d"]),
                    np.float32(dg["v_min_cmd_emu"]),
                    np.float32(1.0 if dg.get("du_nonnegative_speed", True) else 0.0),
                    np.float32(self._du_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    np.bool_(self._use_strict_occ_vref()),
                    np.bool_(self._use_occ_vref_normalization()),
                    barrier_kappa,
                    scenario_kappa,
                )
            _ = np.asarray(traj[0])
            self._jax_warm_n_steps = int(n_steps)
            self._jax_warmed = True
        except Exception as e:
            print(f"[OcclusionController][WARN] JAX warm-start failed, fallback to NumPy rollout: {e}")
            self._jax_enabled = False

    def _simulate_backup_trajectory_numpy(self, x0, T, dt, occlusion_scenarios=None):
        x = np.asarray(x0, float).reshape(self._n_state, 1)
        Phi = np.eye(self._n_state)
        N = int(np.floor(T / dt)) + 1
        t_grid = dt * np.arange(N)

        backup_traj = np.zeros((N, self._n_state))
        stm_traj = np.zeros((N, self._n_state, self._n_state))
        backup_traj[0] = x.ravel()
        stm_traj[0] = Phi
        fcl_traj = np.zeros((N, self._n_state))

        I = np.eye(self._n_state)

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

    def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None, eps_A=1e-4):
        """
        RK4 rollout for x with a stable STM update.
        Uses a midpoint linearization F_cl and a trapezoidal (Cayley) step for Phi.
        """
        del eps_A
        N = int(np.floor(T / dt)) + 1
        if (
            (not self._jax_enabled)
            or (not self._jax_warmed)
            or N != self._jax_warm_n_steps
            or (
                self._occ_vref_scenario_weight_mode() == "barrier_unexpand"
                and self.model == "DynamicUnicycle2D"
            )
        ):
            return self._simulate_backup_trajectory_numpy(
                x0, T, dt, occlusion_scenarios=occlusion_scenarios
            )

        try:
            (
                A_pad,
                b0_pad,
                v_expand_pad,
                v_adv_vec,
                valid_mask,
                obs_center_pad,
                obs_center_valid,
            ) = self._pack_scenarios_for_jax(
                occlusion_scenarios, x_ref=x0
            )
            (
                score_A_pad,
                score_b0_pad,
                score_valid_mask,
                score_present,
            ) = self._pack_score_scenarios_for_jax(occlusion_scenarios)
            front_mode_los = np.bool_(self._occ_vref_front_mode() == "los")
            barrier_kappa = np.float32(self._occ_barrier_smoothing_kappa())
            if self.model == "DoubleIntegrator2D":
                x0_vec = np.asarray(x0, dtype=np.float32).reshape(4,)
                scenario_kappa = np.float32(self._occ_vref_scenario_softmax_kappa())
                kd_occ = np.float32(self.pid_occ_gains.get("Kd", 1.0))
                traj, stm, t_grid, fcl = _jax_rollout_kernel_di(
                    jnp.asarray(x0_vec),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    jnp.asarray(obs_center_pad),
                    jnp.asarray(obs_center_valid),
                    front_mode_los,
                    np.float32(dt),
                    int(N),
                    np.float32(self.pid_occ_gains.get("Kp", 1.0)),
                    kd_occ,
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    barrier_kappa,
                    scenario_kappa,
                    np.float32(self._jax_eps_fd),
                    jnp.asarray(score_A_pad),
                    jnp.asarray(score_b0_pad),
                    jnp.asarray(score_valid_mask),
                    jnp.asarray(score_present),
                    np.int32(self._occ_vref_weight_mode_code()),
                )
            elif self.model == "Unicycle2D":
                x0_vec = np.asarray(x0, dtype=np.float32).reshape(3,)
                ug = self._uni_occ_gains()
                v_max = float(self.robot_spec.get("v_max", 1.0))
                v_min = float(self.robot_spec.get("v_min", -v_max))
                scenario_kappa = np.float32(self._occ_vref_scenario_softmax_kappa())
                traj, stm, t_grid, fcl = _jax_rollout_kernel_uni(
                    jnp.asarray(x0_vec),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    jnp.asarray(obs_center_pad),
                    jnp.asarray(obs_center_valid),
                    front_mode_los,
                    np.float32(dt),
                    int(N),
                    np.float32(v_min),
                    np.float32(v_max),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(ug["k_theta_p"]),
                    np.float32(ug["k_theta_d"]),
                    np.float32(ug["k_v_p"]),
                    np.float32(ug["k_v_d"]),
                    np.float32(ug["k_turn_boost"]),
                    np.float32(ug["turn_boost_angle"]),
                    np.float32(ug["v_min_cmd"]),
                    np.float32(ug.get("v_min_cmd_rev", ug["v_min_cmd"])),
                    np.float32(ug.get("turn_crawl_speed", 0.0)),
                    np.float32(ug.get("turn_crawl_angle", np.pi / 2.0)),
                    np.float32(ug.get("reverse_bias", 0.05)),
                    np.float32(ug.get("reverse_gate_angle", 0.45)),
                    np.float32(ug.get("reverse_gate_power", 2.0)),
                    np.int32(ug.get("tracking_mode_code", 0)),
                    np.float32(self._uni_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    np.bool_(self._use_strict_occ_vref()),
                    np.bool_(self._use_occ_vref_normalization()),
                    barrier_kappa,
                    scenario_kappa,
                    jnp.asarray(score_A_pad),
                    jnp.asarray(score_b0_pad),
                    jnp.asarray(score_valid_mask),
                    jnp.asarray(score_present),
                    np.int32(self._occ_vref_weight_mode_code()),
                    np.float32(getattr(self, "last_speed", 0.0)),
                )
            else:
                x0_vec = np.asarray(x0, dtype=np.float32).reshape(4,)
                dg = self._du_occ_gains()
                v_min, v_max = self._du_speed_bounds()
                scenario_kappa = np.float32(self._occ_vref_scenario_softmax_kappa())
                traj, stm, t_grid, fcl = _jax_rollout_kernel_du(
                    jnp.asarray(x0_vec),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    jnp.asarray(obs_center_pad),
                    jnp.asarray(obs_center_valid),
                    front_mode_los,
                    np.float32(dt),
                    int(N),
                    np.float32(v_min),
                    np.float32(v_max),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(dg["k_theta_emu_p"]),
                    np.float32(dg["k_theta_emu_d"]),
                    np.float32(dg["k_turn_boost_emu"]),
                    np.float32(dg["turn_boost_angle_emu"]),
                    np.float32(dg["k_a_track_p"]),
                    np.float32(dg["k_a_track_d"]),
                    np.float32(dg["k_brake"]),
                    np.float32(dg["k_vemu_p"]),
                    np.float32(dg["k_vemu_d"]),
                    np.float32(dg["v_min_cmd_emu"]),
                    np.float32(1.0 if dg.get("du_nonnegative_speed", True) else 0.0),
                    np.float32(self._du_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    np.bool_(self._use_strict_occ_vref()),
                    np.bool_(self._use_occ_vref_normalization()),
                    barrier_kappa,
                    scenario_kappa,
                )
            return (
                np.asarray(traj, dtype=float),
                np.asarray(stm, dtype=float),
                np.asarray(t_grid, dtype=float),
                np.asarray(fcl, dtype=float),
            )
        except Exception as e:
            print(f"[OcclusionController][WARN] JAX rollout failed, fallback to NumPy rollout: {e}")
            return self._simulate_backup_trajectory_numpy(
                x0, T, dt, occlusion_scenarios=occlusion_scenarios
            )
