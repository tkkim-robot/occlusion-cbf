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
from functools import partial

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
    def _jax_occ_vref_from_pos(
        p,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        tau,
        radius,
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
        normal = jnp.einsum("sk,skd->sd", lam, A_pad)

        norm = jnp.linalg.norm(normal, axis=1)
        direction = normal / jnp.maximum(norm[:, None], 1e-9)

        scenario_present = jnp.any(valid_bool, axis=1)
        use_target = jnp.logical_and(scenario_present, norm > 1e-9)
        targets = jnp.where(use_target[:, None], direction * v_adv_vec[:, None], 0.0)

        count = jnp.sum(use_target.astype(A_pad.dtype))
        v_sum = jnp.sum(targets, axis=0)
        v_ref = v_sum / jnp.maximum(count, 1.0)
        return jnp.where(count > 0.0, v_ref, jnp.zeros((2,), dtype=A_pad.dtype))


    _jax_occ_vref_pos_jac = jax.jacfwd(_jax_occ_vref_from_pos, argnums=0)


    def _jax_di_f_cl(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        tau,
        Kp,
        k_d,
        a_lim,
        v_max,
        radius,
    ):
        v = x[2:4]
        v_ref = _jax_occ_vref_from_pos(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, tau, radius
        )
        e = v - v_ref
        u_unsat = -Kp * e - k_d * v
        u = jnp.clip(u_unsat, -a_lim, a_lim)

        eps = 1e-6
        u = jnp.where(jnp.logical_and(v >= (v_max - eps), u > 0.0), 0.0, u)
        u = jnp.where(jnp.logical_and(v <= (-v_max + eps), u < 0.0), 0.0, u)

        return jnp.array([v[0], v[1], u[0], u[1]], dtype=x.dtype)


    def _jax_di_F_cl_jac(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        tau,
        Kp,
        k_d,
        a_lim,
        v_max,
        radius,
        eps_fd,
    ):
        del eps_fd
        Jp = _jax_occ_vref_pos_jac(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, tau, radius
        )

        Bv = -(Kp + k_d) * jnp.eye(2, dtype=x.dtype)
        lower_left = Kp * Jp
        top = jnp.concatenate([jnp.zeros((2, 2), dtype=x.dtype), jnp.eye(2, dtype=x.dtype)], axis=1)
        bottom = jnp.concatenate([lower_left, Bv], axis=1)
        return jnp.concatenate([top, bottom], axis=0)


    @partial(jax.jit, static_argnums=(7,))
    def _jax_rollout_kernel_di(
        x0,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        dt,
        n_steps,
        Kp,
        k_d,
        a_lim,
        v_max,
        radius,
        eps_fd,
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
                t, Kp, k_d, a_lim, v_max, radius
            )
            k2 = _jax_di_f_cl(
                x + 0.5 * dt * k1, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, Kp, k_d, a_lim, v_max, radius
            )
            k3 = _jax_di_f_cl(
                x + 0.5 * dt * k2, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, Kp, k_d, a_lim, v_max, radius
            )
            k4 = _jax_di_f_cl(
                x + dt * k3, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + dt, Kp, k_d, a_lim, v_max, radius
            )
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_mid = x + 0.5 * dt * k2
            A_mid = _jax_di_F_cl_jac(
                x_mid, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, Kp, k_d, a_lim, v_max, radius, eps_fd
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
            t_last, Kp, k_d, a_lim, v_max, radius
        )

        traj = jnp.concatenate([x0[None, :], x_seq], axis=0)
        stm = jnp.concatenate([I[None, :, :], Phi_seq], axis=0)
        fcl = jnp.concatenate([k1_seq, f_last[None, :]], axis=0)
        t_grid = dt * jnp.arange(n_steps, dtype=x0.dtype)
        return traj, stm, t_grid, fcl


    def _jax_angle_normalize(x):
        return ((x + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi


    def _jax_uni_backup_input_occ(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        tau,
        dt,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        uni_vref_speed,
        radius,
    ):
        theta = x[2]
        v_ref = _jax_occ_vref_from_pos(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, tau, radius
        )
        v_ref_norm = jnp.linalg.norm(v_ref)
        tau_prev = jnp.maximum(tau - dt, 0.0)
        v_ref_prev = _jax_occ_vref_from_pos(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, tau_prev, radius
        )
        v_ref_prev_norm = jnp.linalg.norm(v_ref_prev)
        use_norm = uni_vref_speed > 0.0
        scale = jnp.where(use_norm, uni_vref_speed / jnp.maximum(v_ref_norm, 1e-9), 1.0)
        scale_prev = jnp.where(use_norm, uni_vref_speed / jnp.maximum(v_ref_prev_norm, 1e-9), 1.0)
        v_ref = jnp.where(v_ref_norm > 1e-9, v_ref * scale, v_ref)
        v_ref_prev = jnp.where(v_ref_prev_norm > 1e-9, v_ref_prev * scale_prev, v_ref_prev)
        v_ref_norm = jnp.linalg.norm(v_ref)

        theta_ref = jnp.arctan2(v_ref[1], v_ref[0])
        theta_ref_prev = jnp.arctan2(v_ref_prev[1], v_ref_prev[0])
        e_theta = _jax_angle_normalize(theta_ref - theta)
        e_theta_prev = _jax_angle_normalize(theta_ref_prev - theta)

        dt_safe = jnp.maximum(dt, 1e-6)
        theta_ref_dot = _jax_angle_normalize(theta_ref - theta_ref_prev) / dt_safe
        e_theta_dot = theta_ref_dot

        boost = 1.0 + k_turn_boost * jnp.minimum(
            1.0, jnp.abs(e_theta) / jnp.maximum(turn_boost_angle, 1e-3)
        )
        omega_unsat = boost * (k_theta_p * e_theta + k_theta_d * e_theta_dot)
        omega = jnp.clip(omega_unsat, -w_max, w_max)

        gate = jnp.maximum(0.0, jnp.cos(e_theta))
        gate_prev = jnp.maximum(0.0, jnp.cos(e_theta_prev))
        v_des = jnp.clip(v_ref_norm * gate, 0.0, v_max)
        v_des_prev = jnp.clip(jnp.linalg.norm(v_ref_prev) * gate_prev, 0.0, v_max)
        v_des_dot = (v_des - v_des_prev) / dt_safe

        v_cmd_unsat = k_v_p * v_des + k_v_d * v_des_dot
        v_cmd = jnp.clip(v_cmd_unsat, 0.0, v_max)
        v_cmd = jnp.where(v_ref_norm > 1e-9, jnp.maximum(v_cmd, v_min_cmd), 0.0)

        # No active facet direction -> stop command.
        omega = jnp.where(v_ref_norm < 1e-9, 0.0, omega)
        return jnp.array([v_cmd, omega], dtype=x.dtype)


    def _jax_du_backup_input_occ(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        tau,
        dt,
        v_max,
        w_max,
        a_max,
        k_theta_p,
        k_theta_d,
        k_theta_ff,
        k_a_p,
        k_a_d,
        k_turn_boost,
        turn_boost_angle,
        turn_hard_angle,
        turn_lock_angle,
        turn_speed_ratio,
        k_turn_brake,
        k_brake,
        theta_ref_dot_max,
        theta_deadzone,
        pi_switch_eps,
        theta_speed_gate_ratio,
        du_vref_speed,
        radius,
    ):
        theta = x[2]
        v_cur = x[3]

        v_ref = _jax_occ_vref_from_pos(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, tau, radius
        )
        v_ref_norm = jnp.linalg.norm(v_ref)
        tau_prev = jnp.maximum(tau - dt, 0.0)
        v_ref_prev = _jax_occ_vref_from_pos(
            x[:2], A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask, tau_prev, radius
        )
        v_ref_prev_norm = jnp.linalg.norm(v_ref_prev)

        use_norm = du_vref_speed > 0.0
        scale = jnp.where(use_norm, du_vref_speed / jnp.maximum(v_ref_norm, 1e-9), 1.0)
        scale_prev = jnp.where(use_norm, du_vref_speed / jnp.maximum(v_ref_prev_norm, 1e-9), 1.0)
        v_ref = jnp.where(v_ref_norm > 1e-9, v_ref * scale, v_ref)
        v_ref_prev = jnp.where(v_ref_prev_norm > 1e-9, v_ref_prev * scale_prev, v_ref_prev)
        v_ref_norm = jnp.linalg.norm(v_ref)

        dt_safe = jnp.maximum(dt, 1e-6)
        a_stop = jnp.clip(-k_brake * v_cur, -a_max, a_max)
        no_ref = v_ref_norm < 1e-9

        theta_ref = jnp.arctan2(v_ref[1], v_ref[0])
        theta_ref_prev = jnp.arctan2(v_ref_prev[1], v_ref_prev[0])
        # Shortest-turn heading error in [-pi, pi].
        e_theta = _jax_angle_normalize(theta_ref - theta)
        e_theta_prev = _jax_angle_normalize(theta_ref_prev - theta)

        theta_ref_dot = _jax_angle_normalize(theta_ref - theta_ref_prev) / dt_safe
        theta_ref_dot = jnp.clip(theta_ref_dot, -theta_ref_dot_max, theta_ref_dot_max)
        e_theta = jnp.where(jnp.abs(e_theta) < theta_deadzone, 0.0, e_theta)
        e_theta_dot = (e_theta - e_theta_prev) / dt_safe

        boost = 1.0 + k_turn_boost * jnp.minimum(
            1.0, jnp.abs(e_theta) / jnp.maximum(turn_boost_angle, 1e-3)
        )
        omega_pd = k_theta_p * e_theta + k_theta_d * e_theta_dot + k_theta_ff * theta_ref_dot
        omega = jnp.clip(boost * omega_pd, -w_max, w_max)

        # Speed policy:
        # - If heading error is outside gate_angle, stop first (v_des=0).
        # - Otherwise, track |v_ref| with gate-angle-aware alignment scaling.
        #   This makes theta_speed_gate_ratio meaningful even when gate_angle > pi/2.
        gate_angle = jnp.clip(theta_speed_gate_ratio, 0.1, 1.0) * jnp.pi
        cos_gate = jnp.cos(gate_angle)
        den_gate = jnp.maximum(1.0 - cos_gate, 1e-6)
        abs_e = jnp.abs(e_theta)
        abs_e_prev = jnp.abs(e_theta_prev)
        gate_scale = jnp.clip((jnp.cos(abs_e) - cos_gate) / den_gate, 0.0, 1.0)
        gate_scale_prev = jnp.clip((jnp.cos(abs_e_prev) - cos_gate) / den_gate, 0.0, 1.0)
        v_des = jnp.where(
            abs_e > gate_angle,
            0.0,
            jnp.clip(v_ref_norm * gate_scale, 0.0, v_max),
        )
        v_des_prev = jnp.where(
            abs_e_prev > gate_angle,
            0.0,
            jnp.clip(jnp.linalg.norm(v_ref_prev) * gate_scale_prev, 0.0, v_max),
        )
        v_des_dot = (v_des - v_des_prev) / dt_safe

        a_cmd_track = k_a_p * (v_des - v_cur) + k_a_d * v_des_dot
        a_cmd_stop = -k_brake * v_cur
        a_cmd_unsat = jnp.where(abs_e > gate_angle, a_cmd_stop, a_cmd_track)
        a_cmd = jnp.clip(a_cmd_unsat, -a_max, a_max)

        eps = 1e-6
        a_cmd = jnp.where(jnp.logical_and(v_cur >= (v_max - eps), a_cmd > 0.0), 0.0, a_cmd)
        a_cmd = jnp.where(jnp.logical_and(v_cur <= eps, a_cmd < 0.0), 0.0, a_cmd)

        omega = jnp.where(no_ref, 0.0, omega)
        a_cmd = jnp.where(no_ref, a_stop, a_cmd)
        return jnp.array([a_cmd, omega], dtype=x.dtype)


    def _jax_uni_f_cl(
        x,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        tau,
        dt,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        uni_vref_speed,
        radius,
    ):
        u = _jax_uni_backup_input_occ(
            x,
            A_pad,
            b0_pad,
            v_expand_pad,
            v_adv_vec,
            valid_mask,
            tau,
            dt,
            v_max,
            w_max,
            k_theta_p,
            k_theta_d,
            k_v_p,
            k_v_d,
            k_turn_boost,
            turn_boost_angle,
            v_min_cmd,
            uni_vref_speed,
            radius,
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
        tau,
        dt,
        v_max,
        w_max,
        a_max,
        k_theta_p,
        k_theta_d,
        k_theta_ff,
        k_a_p,
        k_a_d,
        k_turn_boost,
        turn_boost_angle,
        turn_hard_angle,
        turn_lock_angle,
        turn_speed_ratio,
        k_turn_brake,
        k_brake,
        theta_ref_dot_max,
        theta_deadzone,
        pi_switch_eps,
        theta_speed_gate_ratio,
        du_vref_speed,
        radius,
    ):
        u = _jax_du_backup_input_occ(
            x,
            A_pad,
            b0_pad,
            v_expand_pad,
            v_adv_vec,
            valid_mask,
            tau,
            dt,
            v_max,
            w_max,
            a_max,
            k_theta_p,
            k_theta_d,
            k_theta_ff,
            k_a_p,
            k_a_d,
            k_turn_boost,
            turn_boost_angle,
            turn_hard_angle,
            turn_lock_angle,
            turn_speed_ratio,
            k_turn_brake,
            k_brake,
            theta_ref_dot_max,
            theta_deadzone,
            pi_switch_eps,
            theta_speed_gate_ratio,
            du_vref_speed,
            radius,
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


    @partial(jax.jit, static_argnums=(7,))
    def _jax_rollout_kernel_uni(
        x0,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        dt,
        n_steps,
        v_max,
        w_max,
        k_theta_p,
        k_theta_d,
        k_v_p,
        k_v_d,
        k_turn_boost,
        turn_boost_angle,
        v_min_cmd,
        uni_vref_speed,
        radius,
    ):
        I = jnp.eye(3, dtype=x0.dtype)

        def step_fn(carry, idx):
            x, Phi = carry
            t = dt * idx.astype(x0.dtype)

            k1 = _jax_uni_f_cl(
                x, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t, dt, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, uni_vref_speed, radius
            )
            k2 = _jax_uni_f_cl(
                x + 0.5 * dt * k1, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, dt, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, uni_vref_speed, radius
            )
            k3 = _jax_uni_f_cl(
                x + 0.5 * dt * k2, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, dt, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, uni_vref_speed, radius
            )
            k4 = _jax_uni_f_cl(
                x + dt * k3, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + dt, dt, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, uni_vref_speed, radius
            )
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_mid = x + 0.5 * dt * k2
            A_mid = _jax_uni_F_cl_jac(
                x_mid, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, dt, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
                k_turn_boost, turn_boost_angle, v_min_cmd, uni_vref_speed, radius
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
            t_last, dt, v_max, w_max, k_theta_p, k_theta_d, k_v_p, k_v_d,
            k_turn_boost, turn_boost_angle, v_min_cmd, uni_vref_speed, radius
        )

        traj = jnp.concatenate([x0[None, :], x_seq], axis=0)
        stm = jnp.concatenate([I[None, :, :], Phi_seq], axis=0)
        fcl = jnp.concatenate([k1_seq, f_last[None, :]], axis=0)
        t_grid = dt * jnp.arange(n_steps, dtype=x0.dtype)
        return traj, stm, t_grid, fcl


    @partial(jax.jit, static_argnums=(7,))
    def _jax_rollout_kernel_du(
        x0,
        A_pad,
        b0_pad,
        v_expand_pad,
        v_adv_vec,
        valid_mask,
        dt,
        n_steps,
        v_max,
        w_max,
        a_max,
        k_theta_p,
        k_theta_d,
        k_theta_ff,
        k_a_p,
        k_a_d,
        k_turn_boost,
        turn_boost_angle,
        turn_hard_angle,
        turn_lock_angle,
        turn_speed_ratio,
        k_turn_brake,
        k_brake,
        theta_ref_dot_max,
        theta_deadzone,
        pi_switch_eps,
        theta_speed_gate_ratio,
        du_vref_speed,
        radius,
    ):
        I = jnp.eye(4, dtype=x0.dtype)

        def step_fn(carry, idx):
            x, Phi = carry
            t = dt * idx.astype(x0.dtype)

            k1 = _jax_du_f_cl(
                x, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t, dt, v_max, w_max, a_max, k_theta_p, k_theta_d, k_theta_ff, k_a_p, k_a_d,
                k_turn_boost, turn_boost_angle, turn_hard_angle, turn_lock_angle, turn_speed_ratio, k_turn_brake,
                k_brake, theta_ref_dot_max, theta_deadzone, pi_switch_eps, theta_speed_gate_ratio, du_vref_speed, radius
            )
            k2 = _jax_du_f_cl(
                x + 0.5 * dt * k1, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, dt, v_max, w_max, a_max, k_theta_p, k_theta_d, k_theta_ff, k_a_p, k_a_d,
                k_turn_boost, turn_boost_angle, turn_hard_angle, turn_lock_angle, turn_speed_ratio, k_turn_brake,
                k_brake, theta_ref_dot_max, theta_deadzone, pi_switch_eps, theta_speed_gate_ratio, du_vref_speed, radius
            )
            k3 = _jax_du_f_cl(
                x + 0.5 * dt * k2, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, dt, v_max, w_max, a_max, k_theta_p, k_theta_d, k_theta_ff, k_a_p, k_a_d,
                k_turn_boost, turn_boost_angle, turn_hard_angle, turn_lock_angle, turn_speed_ratio, k_turn_brake,
                k_brake, theta_ref_dot_max, theta_deadzone, pi_switch_eps, theta_speed_gate_ratio, du_vref_speed, radius
            )
            k4 = _jax_du_f_cl(
                x + dt * k3, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + dt, dt, v_max, w_max, a_max, k_theta_p, k_theta_d, k_theta_ff, k_a_p, k_a_d,
                k_turn_boost, turn_boost_angle, turn_hard_angle, turn_lock_angle, turn_speed_ratio, k_turn_brake,
                k_brake, theta_ref_dot_max, theta_deadzone, pi_switch_eps, theta_speed_gate_ratio, du_vref_speed, radius
            )
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_mid = x + 0.5 * dt * k2
            A_mid = _jax_du_F_cl_jac(
                x_mid, A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask,
                t + 0.5 * dt, dt, v_max, w_max, a_max, k_theta_p, k_theta_d, k_theta_ff, k_a_p, k_a_d,
                k_turn_boost, turn_boost_angle, turn_hard_angle, turn_lock_angle, turn_speed_ratio, k_turn_brake,
                k_brake, theta_ref_dot_max, theta_deadzone, pi_switch_eps, theta_speed_gate_ratio, du_vref_speed, radius
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
            t_last, dt, v_max, w_max, a_max, k_theta_p, k_theta_d, k_theta_ff, k_a_p, k_a_d,
            k_turn_boost, turn_boost_angle, turn_hard_angle, turn_lock_angle, turn_speed_ratio, k_turn_brake,
            k_brake, theta_ref_dot_max, theta_deadzone, pi_switch_eps, theta_speed_gate_ratio, du_vref_speed, radius
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
            "Kd": 0.1,
            "aw_limit": 1.0
        }
        cfg = self.robot_spec.setdefault("backup_cbf", {})
        self.T_horizon = float(cfg.get("T_horizon", 2.0))
        cfg["T_horizon"] = self.T_horizon

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
        k_theta_p = float(cfg.get("k_theta_occ_uni_p", cfg.get("k_theta_occ_uni", 2.5)))
        k_theta_d = float(cfg.get("k_theta_occ_uni_d", 0.4))
        k_v_p = float(cfg.get("k_v_occ_uni_p", 1.0))
        k_v_d = float(cfg.get("k_v_occ_uni_d", 0.2))
        k_turn_boost = float(cfg.get("k_turn_boost_occ_uni", 1.0))
        turn_boost_angle = float(cfg.get("turn_boost_angle_occ_uni", np.pi / 6.0))
        v_min_cmd = float(cfg.get("v_min_occ_uni", 0.05))
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
        return {
            "k_theta_p": k_theta_p,
            "k_theta_d": k_theta_d,
            "k_v_p": k_v_p,
            "k_v_d": k_v_d,
            "k_turn_boost": k_turn_boost,
            "turn_boost_angle": turn_boost_angle,
            "v_min_cmd": v_min_cmd,
        }

    def _uni_occ_vref_speed(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        s = cfg.get(
            "v_ref_norm_occ_uni",
            self.robot_spec.get("v_obs_max", self.robot_spec.get("v_adv_max_occ", 0.0)),
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
        k_theta_p = float(cfg.get("k_theta_occ_du_p", cfg.get("k_theta_occ_du", 3.9)))
        k_theta_d = float(cfg.get("k_theta_occ_du_d", 1.37))
        k_theta_ff = float(cfg.get("k_theta_occ_du_ff", 0.55))
        k_a_p = float(cfg.get("k_a_occ_du_p", 1.71))
        k_a_d = float(cfg.get("k_a_occ_du_d", 0.69))
        k_turn_boost = float(cfg.get("k_turn_boost_occ_du", 2.96))
        turn_boost_angle = float(cfg.get("turn_boost_angle_occ_du", 0.95))
        turn_hard_angle = float(cfg.get("turn_hard_angle_occ_du", np.pi / 3.0))
        turn_lock_angle = float(cfg.get("turn_lock_angle_occ_du", 2.7))
        turn_unlock_angle = float(cfg.get("turn_unlock_angle_occ_du", 2.2))
        turn_speed_ratio = float(cfg.get("turn_speed_ratio_occ_du", 0.75))
        k_turn_brake = float(cfg.get("k_turn_brake_occ_du", 0.4))
        k_brake = float(cfg.get("k_brake_occ_du", 0.34))
        theta_ref_dot_max = float(cfg.get("theta_ref_dot_max_occ_du", 20.0))
        theta_deadzone = float(cfg.get("theta_deadzone_occ_du", 0.0))
        pi_switch_eps = float(cfg.get("pi_switch_eps_occ_du", 0.15))
        theta_speed_gate_ratio = float(cfg.get("theta_speed_gate_ratio_occ_du", 0.5))
        if not np.isfinite(k_theta_p):
            k_theta_p = 3.9
        if not np.isfinite(k_theta_d):
            k_theta_d = 1.37
        if not np.isfinite(k_theta_ff):
            k_theta_ff = 0.55
        if not np.isfinite(k_a_p):
            k_a_p = 1.71
        if not np.isfinite(k_a_d):
            k_a_d = 0.69
        if not np.isfinite(k_turn_boost):
            k_turn_boost = 2.96
        if (not np.isfinite(turn_boost_angle)) or (turn_boost_angle <= 1e-3):
            turn_boost_angle = 0.95
        if (not np.isfinite(turn_hard_angle)) or (turn_hard_angle <= 1e-3):
            turn_hard_angle = np.pi / 3.0
        if (not np.isfinite(turn_lock_angle)) or (turn_lock_angle <= 1e-3):
            turn_lock_angle = 2.7
        if (not np.isfinite(turn_unlock_angle)) or (turn_unlock_angle <= 1e-3):
            turn_unlock_angle = 2.2
        if turn_unlock_angle > turn_lock_angle:
            turn_unlock_angle = 0.8 * turn_lock_angle
        if (not np.isfinite(turn_speed_ratio)) or (turn_speed_ratio < 0.0):
            turn_speed_ratio = 0.75
        turn_speed_ratio = min(max(turn_speed_ratio, 0.0), 1.0)
        if (not np.isfinite(k_turn_brake)) or (k_turn_brake < 0.0):
            k_turn_brake = 0.4
        if (not np.isfinite(k_brake)) or (k_brake < 0.0):
            k_brake = 0.34
        if (not np.isfinite(theta_ref_dot_max)) or (theta_ref_dot_max <= 0.0):
            theta_ref_dot_max = 20.0
        if (not np.isfinite(theta_deadzone)) or (theta_deadzone < 0.0):
            theta_deadzone = 0.0
        if (not np.isfinite(pi_switch_eps)) or (pi_switch_eps < 0.0):
            pi_switch_eps = 0.15
        if (not np.isfinite(theta_speed_gate_ratio)) or (theta_speed_gate_ratio <= 0.0):
            theta_speed_gate_ratio = 0.5
        theta_speed_gate_ratio = min(max(theta_speed_gate_ratio, 0.1), 1.0)
        return {
            "k_theta_p": k_theta_p,
            "k_theta_d": k_theta_d,
            "k_theta_ff": k_theta_ff,
            "k_a_p": k_a_p,
            "k_a_d": k_a_d,
            "k_turn_boost": k_turn_boost,
            "turn_boost_angle": turn_boost_angle,
            "turn_hard_angle": turn_hard_angle,
            "turn_lock_angle": turn_lock_angle,
            "turn_unlock_angle": turn_unlock_angle,
            "turn_speed_ratio": turn_speed_ratio,
            "k_turn_brake": k_turn_brake,
            "k_brake": k_brake,
            "theta_ref_dot_max": theta_ref_dot_max,
            "theta_deadzone": theta_deadzone,
            "pi_switch_eps": pi_switch_eps,
            "theta_speed_gate_ratio": theta_speed_gate_ratio,
        }

    def _du_occ_vref_speed(self):
        cfg = self.robot_spec.get("backup_cbf", {})
        # Dynamic unicycle uses |v_ref| as speed target by default.
        # Set backup_cbf.v_ref_norm_occ_du > 0 only when fixed-speed normalization is desired.
        s = cfg.get("v_ref_norm_occ_du", 0.0)
        try:
            s = float(s)
        except Exception:
            return 0.0
        if (not np.isfinite(s)) or s <= 1e-9:
            return 0.0
        return s

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
        R     = self.robot_spec['radius']
        kappa_vref = float(
            self.robot_spec.get("backup_cbf", {}).get("vref_softmax_kappa", 8.0)
        )
        if (not np.isfinite(kappa_vref)) or (kappa_vref <= 1e-6):
            kappa_vref = 8.0

        v_entries = []
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

            if h_vec.size == 0:
                continue
            max_h = float(np.max(h_vec))
            z = np.exp(kappa_vref * (h_vec - max_h))
            z_sum = float(np.sum(z))
            if (not np.isfinite(z_sum)) or z_sum <= 1e-12:
                continue
            lam = z / z_sum
            avg_normal = (lam[:, None] * A).sum(axis=0)
            norm = float(np.linalg.norm(avg_normal))
            if norm > 1e-9:
                direction = avg_normal / norm
                v_target = direction * vadv
                risk = float(np.max(h_vec))
                v_entries.append((v_target, risk))

        if len(v_entries) == 0:
            return np.zeros(2, dtype=float)

        A_all = np.vstack([item[0] for item in v_entries])

        v_avg = A_all.mean(axis=0)
        if self.model == "Unicycle2D":
            s = self._uni_occ_vref_speed()
            n = float(np.linalg.norm(v_avg))
            if s > 0.0 and n > 1e-9:
                v_avg = (v_avg / n) * s
        # elif self.model == "DynamicUnicycle2D":
        #     s = self._du_occ_vref_speed()
        #     n = float(np.linalg.norm(v_avg))
        #     if s > 0.0 and n > 1e-9:
        #         v_avg = (v_avg / n) * s
        
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
            k_d = 1.0
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
            if gp.shape == (1,2):
                # Expand to full backup-state dimension: [grad_pos, 0, ..., 0]
                pad = np.zeros((1, max(0, self._n_state - 2)), dtype=float)
                return np.hstack([gp, pad])
            return gp

    def grad_occ_terminal(self, X):
        return self.grad_h_b_stop(X)

    def backup_input_occlusion(self, X, occlusion_scenarios, t=None,
                            k_d=1.0, k_occ=1.0):
        # print(f"DEBUG: backup_input_occlusion CALLED with t={t}")

        X = np.asarray(X, dtype=float).reshape(self._n_state, 1)
        v_ref = self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, t)

        if self.model == "DoubleIntegrator2D":
            a_lim = float(self.robot_spec.get("a_max", 1.0))
            v_max = float(self.robot_spec.get("v_max", 1.0))
            Kp = float(self.pid_occ_gains.get("Kp", 1.0))

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
            v_ref_norm = float(np.linalg.norm(v_ref))
            if v_ref_norm < 1e-9:
                return np.zeros((2, 1), dtype=float)

            tau_now = self.clamp_tau(float(t) if t is not None else 0.0)
            tau_prev = self.clamp_tau(max(0.0, float(tau_now) - float(self.dt)))
            v_ref_prev = self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, tau_prev)

            theta_ref = float(np.arctan2(v_ref[1], v_ref[0]))
            theta_ref_prev = float(np.arctan2(v_ref_prev[1], v_ref_prev[0]))
            w_max = float(self.robot_spec.get("w_max", 0.8))
            v_max = float(self.robot_spec.get("v_max", 1.0))
            gains = self._uni_occ_gains()
            e_theta = angle_normalize(theta_ref - theta)
            e_theta_prev = angle_normalize(theta_ref_prev - theta)
            dt_safe = max(float(self.dt), 1e-6)
            theta_ref_dot = angle_normalize(theta_ref - theta_ref_prev) / dt_safe
            e_theta_dot = theta_ref_dot

            boost = 1.0 + gains["k_turn_boost"] * min(
                1.0, abs(e_theta) / max(gains["turn_boost_angle"], 1e-3)
            )
            omega_unsat = boost * (gains["k_theta_p"] * e_theta + gains["k_theta_d"] * e_theta_dot)
            omega = float(np.clip(omega_unsat, -w_max, w_max))

            gate = max(0.0, np.cos(e_theta))
            gate_prev = max(0.0, np.cos(e_theta_prev))
            v_des = float(np.clip(v_ref_norm * gate, 0.0, v_max))
            v_des_prev = float(np.clip(np.linalg.norm(v_ref_prev) * gate_prev, 0.0, v_max))
            v_des_dot = (v_des - v_des_prev) / dt_safe
            v_cmd_unsat = gains["k_v_p"] * v_des + gains["k_v_d"] * v_des_dot
            v_cmd = float(np.clip(v_cmd_unsat, 0.0, v_max))
            v_cmd = max(v_cmd, min(gains["v_min_cmd"], v_max))
            return np.array([[v_cmd], [omega]], dtype=float)

        if self.model == "DynamicUnicycle2D":
            theta = float(X[2, 0])
            v_cur = float(X[3, 0])
            v_ref_norm = float(np.linalg.norm(v_ref))
            tau_now = self.clamp_tau(float(t) if t is not None else 0.0)
            tau_prev = self.clamp_tau(max(0.0, float(tau_now) - float(self.dt)))
            v_ref_prev = self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, tau_prev)

            a_max = float(self.robot_spec.get("a_max", 1.0))
            w_max = float(self.robot_spec.get("w_max", 0.8))
            v_max = float(self.robot_spec.get("v_max", 1.0))
            gains = self._du_occ_gains()
            dt_safe = max(float(self.dt), 1e-6)

            if v_ref_norm < 1e-9:
                a_stop = float(np.clip(-gains["k_brake"] * v_cur, -a_max, a_max))
                return np.array([[a_stop], [0.0]], dtype=float)

            theta_ref = float(np.arctan2(v_ref[1], v_ref[0]))
            theta_ref_prev = float(np.arctan2(v_ref_prev[1], v_ref_prev[0]))
            # Shortest-turn heading error in [-pi, pi].
            e_theta = angle_normalize(theta_ref - theta)
            e_theta_prev = angle_normalize(theta_ref_prev - theta)

            theta_ref_dot = angle_normalize(theta_ref - theta_ref_prev) / dt_safe
            theta_ref_dot = float(
                np.clip(
                    theta_ref_dot,
                    -gains["theta_ref_dot_max"],
                    gains["theta_ref_dot_max"],
                )
            )
            if abs(e_theta) < gains["theta_deadzone"]:
                e_theta = 0.0
            e_theta_dot = (e_theta - e_theta_prev) / dt_safe

            boost = 1.0 + gains["k_turn_boost"] * min(
                1.0, abs(e_theta) / max(gains["turn_boost_angle"], 1e-3)
            )
            omega_pd = (
                gains["k_theta_p"] * e_theta
                + gains["k_theta_d"] * e_theta_dot
                + gains["k_theta_ff"] * theta_ref_dot
            )
            omega = float(np.clip(boost * omega_pd, -w_max, w_max))

            # Speed policy:
            # - If heading error is outside gate_angle, stop first (v_des=0).
            # - Otherwise, track |v_ref| with gate-angle-aware alignment scaling.
            #   This makes theta_speed_gate_ratio meaningful even when gate_angle > pi/2.
            gate_angle = float(gains["theta_speed_gate_ratio"]) * np.pi
            cos_gate = float(np.cos(gate_angle))
            den_gate = max(1.0 - cos_gate, 1e-6)
            abs_e = abs(e_theta)
            abs_e_prev = abs(e_theta_prev)
            gate_scale = float(np.clip((np.cos(abs_e) - cos_gate) / den_gate, 0.0, 1.0))
            gate_scale_prev = float(np.clip((np.cos(abs_e_prev) - cos_gate) / den_gate, 0.0, 1.0))

            if abs_e > gate_angle:
                v_des = 0.0
            else:
                v_des = float(np.clip(v_ref_norm * gate_scale, 0.0, v_max))
            if abs_e_prev > gate_angle:
                v_des_prev = 0.0
            else:
                v_des_prev = float(np.clip(np.linalg.norm(v_ref_prev) * gate_scale_prev, 0.0, v_max))

            v_des_dot = (v_des - v_des_prev) / dt_safe

            if abs_e > gate_angle:
                a_cmd_unsat = -gains["k_brake"] * v_cur
            else:
                a_cmd_unsat = gains["k_a_p"] * (v_des - v_cur) + gains["k_a_d"] * v_des_dot
            a_cmd = float(np.clip(a_cmd_unsat, -a_max, a_max))
            eps = 1e-6
            if v_cur >= (v_max - eps) and a_cmd > 0.0:
                a_cmd = 0.0
            if v_cur <= eps and a_cmd < 0.0:
                a_cmd = 0.0
            return np.array([[a_cmd], [omega]], dtype=float)

        raise NotImplementedError(f"Unsupported model: {self.model}")

    def _pack_scenarios_for_jax(self, occlusion_scenarios, x_ref=None):
        S = int(self._jax_max_scenarios)
        K = int(self._jax_max_halfspaces)

        A_pad = np.zeros((S, K, 2), dtype=np.float32)
        b0_pad = np.zeros((S, K), dtype=np.float32)
        v_expand_pad = np.zeros((S, K), dtype=np.float32)
        v_adv_vec = np.zeros((S,), dtype=np.float32)
        valid_mask = np.zeros((S, K), dtype=np.float32)

        if not occlusion_scenarios:
            return A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask

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

        return A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask

    def _warm_start_jax_rollout(self, n_steps):
        if not self._jax_enabled:
            return
        try:
            A_pad = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces, 2), dtype=np.float32)
            b0_pad = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces), dtype=np.float32)
            v_expand_pad = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces), dtype=np.float32)
            v_adv_vec = np.zeros((self._jax_max_scenarios,), dtype=np.float32)
            valid_mask = np.zeros((self._jax_max_scenarios, self._jax_max_halfspaces), dtype=np.float32)

            # Compile ahead of control loop; no JIT should happen inside step loop.
            if self.model == "DoubleIntegrator2D":
                x0 = np.zeros((4,), dtype=np.float32)
                traj, _, _, _ = _jax_rollout_kernel_di(
                    jnp.asarray(x0),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    np.float32(self.dt),
                    int(n_steps),
                    np.float32(self.pid_occ_gains.get("Kp", 1.0)),
                    np.float32(1.0),
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    np.float32(self._jax_eps_fd),
                )
            elif self.model == "Unicycle2D":
                x0 = np.zeros((3,), dtype=np.float32)
                ug = self._uni_occ_gains()
                traj, _, _, _ = _jax_rollout_kernel_uni(
                    jnp.asarray(x0),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    np.float32(self.dt),
                    int(n_steps),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(ug["k_theta_p"]),
                    np.float32(ug["k_theta_d"]),
                    np.float32(ug["k_v_p"]),
                    np.float32(ug["k_v_d"]),
                    np.float32(ug["k_turn_boost"]),
                    np.float32(ug["turn_boost_angle"]),
                    np.float32(ug["v_min_cmd"]),
                    np.float32(self._uni_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                )
            else:
                x0 = np.zeros((4,), dtype=np.float32)
                dg = self._du_occ_gains()
                traj, _, _, _ = _jax_rollout_kernel_du(
                    jnp.asarray(x0),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    np.float32(self.dt),
                    int(n_steps),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(dg["k_theta_p"]),
                    np.float32(dg["k_theta_d"]),
                    np.float32(dg["k_theta_ff"]),
                    np.float32(dg["k_a_p"]),
                    np.float32(dg["k_a_d"]),
                    np.float32(dg["k_turn_boost"]),
                    np.float32(dg["turn_boost_angle"]),
                    np.float32(dg["turn_hard_angle"]),
                    np.float32(dg["turn_lock_angle"]),
                    np.float32(dg["turn_speed_ratio"]),
                    np.float32(dg["k_turn_brake"]),
                    np.float32(dg["k_brake"]),
                    np.float32(dg["theta_ref_dot_max"]),
                    np.float32(dg["theta_deadzone"]),
                    np.float32(dg["pi_switch_eps"]),
                    np.float32(dg["theta_speed_gate_ratio"]),
                    np.float32(self._du_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
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
        if (not self._jax_enabled) or (not self._jax_warmed) or N != self._jax_warm_n_steps:
            return self._simulate_backup_trajectory_numpy(
                x0, T, dt, occlusion_scenarios=occlusion_scenarios
            )

        try:
            A_pad, b0_pad, v_expand_pad, v_adv_vec, valid_mask = self._pack_scenarios_for_jax(
                occlusion_scenarios, x_ref=x0
            )
            if self.model == "DoubleIntegrator2D":
                x0_vec = np.asarray(x0, dtype=np.float32).reshape(4,)
                traj, stm, t_grid, fcl = _jax_rollout_kernel_di(
                    jnp.asarray(x0_vec),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    np.float32(dt),
                    int(N),
                    np.float32(self.pid_occ_gains.get("Kp", 1.0)),
                    np.float32(1.0),
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                    np.float32(self._jax_eps_fd),
                )
            elif self.model == "Unicycle2D":
                x0_vec = np.asarray(x0, dtype=np.float32).reshape(3,)
                ug = self._uni_occ_gains()
                traj, stm, t_grid, fcl = _jax_rollout_kernel_uni(
                    jnp.asarray(x0_vec),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    np.float32(dt),
                    int(N),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(ug["k_theta_p"]),
                    np.float32(ug["k_theta_d"]),
                    np.float32(ug["k_v_p"]),
                    np.float32(ug["k_v_d"]),
                    np.float32(ug["k_turn_boost"]),
                    np.float32(ug["turn_boost_angle"]),
                    np.float32(ug["v_min_cmd"]),
                    np.float32(self._uni_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
                )
            else:
                x0_vec = np.asarray(x0, dtype=np.float32).reshape(4,)
                dg = self._du_occ_gains()
                traj, stm, t_grid, fcl = _jax_rollout_kernel_du(
                    jnp.asarray(x0_vec),
                    jnp.asarray(A_pad),
                    jnp.asarray(b0_pad),
                    jnp.asarray(v_expand_pad),
                    jnp.asarray(v_adv_vec),
                    jnp.asarray(valid_mask),
                    np.float32(dt),
                    int(N),
                    np.float32(self.robot_spec.get("v_max", 1.0)),
                    np.float32(self.robot_spec.get("w_max", 0.8)),
                    np.float32(self.robot_spec.get("a_max", 1.0)),
                    np.float32(dg["k_theta_p"]),
                    np.float32(dg["k_theta_d"]),
                    np.float32(dg["k_theta_ff"]),
                    np.float32(dg["k_a_p"]),
                    np.float32(dg["k_a_d"]),
                    np.float32(dg["k_turn_boost"]),
                    np.float32(dg["turn_boost_angle"]),
                    np.float32(dg["turn_hard_angle"]),
                    np.float32(dg["turn_lock_angle"]),
                    np.float32(dg["turn_speed_ratio"]),
                    np.float32(dg["k_turn_brake"]),
                    np.float32(dg["k_brake"]),
                    np.float32(dg["theta_ref_dot_max"]),
                    np.float32(dg["theta_deadzone"]),
                    np.float32(dg["pi_switch_eps"]),
                    np.float32(dg["theta_speed_gate_ratio"]),
                    np.float32(self._du_occ_vref_speed()),
                    np.float32(self.robot_spec.get("radius", 0.25)),
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
