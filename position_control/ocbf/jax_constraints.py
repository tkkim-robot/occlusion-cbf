"""JAX kernels for stacked Occlusion-CBF constraint rows.

These functions are deliberately stateless.  They take padded occlusion
scenario arrays and backup rollout data, then return the linearized CBF rows
used by :class:`position_control.occlusion_cbf_qp.OcclusionCBFQP`.
"""

from functools import partial

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


if JAX_AVAILABLE:
    @partial(jax.jit, static_argnums=(14,))
    def jax_occ_constraints_kernel_di(
        phi_b,           # (N,4)
        Phi_b,           # (N,4,4)
        fcl_traj,        # (N,4)
        tau_points,      # (N,)
        A_pad,           # (S,K,2)
        b0_pad,          # (S,K)
        v_expand_pad,    # (S,K)
        valid_mask,      # (S,K)
        sc_present,      # (S,)
        f_x_vec,         # (4,)
        g_x_mat,         # (4,2)
        kappa,
        alpha,
        radius,
        n_tau,           # static: avoid re-trace on Python loop
    ):
        """Build DI occlusion CBF rows in vectorized form."""
        phi = phi_b[:n_tau, :]
        Phi = Phi_b[:n_tau, :, :]
        fcl = fcl_traj[:n_tau, :]
        tau = tau_points[:n_tau]

        pos = phi[:, :2]
        h = (
            jnp.einsum("skd,nd->snk", A_pad, pos)
            - b0_pad[:, None, :]
            - (radius + v_expand_pad[:, None, :] * tau[None, :, None])
        )

        valid_bool = valid_mask[:, None, :] > 0.5
        h_valid = jnp.where(valid_bool, h, jnp.full_like(h, -jnp.inf))

        max_h = jnp.max(h_valid, axis=2)
        finite_row = jnp.isfinite(max_h)
        max_h_safe = jnp.where(finite_row, max_h, 0.0)

        z = jnp.where(valid_bool, jnp.exp(kappa * (h - max_h_safe[:, :, None])), 0.0)
        Z = jnp.sum(z, axis=2)
        Z_safe = jnp.maximum(Z, 1e-12)
        lam = z / Z_safe[:, :, None]

        K_count = jnp.sum(valid_mask, axis=1)
        K_safe = jnp.maximum(K_count, 1.0)

        h_tilde = max_h_safe + (jnp.log(Z_safe) - jnp.log(K_safe)[:, None]) / kappa
        grad_pos = jnp.einsum("snk,skd->snd", lam, A_pad)

        # Under pure facet propagation, dh/dt and dh/ds cancel in Eq. (35b).
        dh_ds = jnp.einsum("snk,sk->sn", lam, -v_expand_pad)
        dh_dt = jnp.einsum("snk,sk->sn", lam, -v_expand_pad)
        time_residual = dh_dt - dh_ds

        zeros2 = jnp.zeros((grad_pos.shape[0], grad_pos.shape[1], 2), dtype=grad_pos.dtype)
        grad_h_phi = jnp.concatenate([grad_pos, zeros2], axis=2)

        Phi_f = jnp.einsum("nij,j->ni", Phi, f_x_vec)
        delta_f = Phi_f - fcl
        c_occ = jnp.einsum("snd,nd->sn", grad_h_phi, delta_f) + time_residual

        Phi_g = jnp.einsum("nij,jm->nim", Phi, g_x_mat)
        A_occ = jnp.einsum("snd,ndm->snm", grad_h_phi, Phi_g)

        rhs = c_occ + alpha * h_tilde
        norm_A_occ = jnp.linalg.norm(A_occ, axis=2)

        finite_mask = finite_row & jnp.isfinite(rhs) & jnp.all(jnp.isfinite(A_occ), axis=2)
        active_mask = sc_present[:, None] & finite_mask
        degenerate = (norm_A_occ < 1e-9) & (rhs < 0.0)
        keep = active_mask & (~degenerate)

        return -A_occ, rhs, keep

    @partial(jax.jit, static_argnums=(14,))
    def jax_occ_constraints_kernel_uni(
        phi_b,           # (N,3)
        Phi_b,           # (N,3,3)
        fcl_traj,        # (N,3)
        tau_points,      # (N,)
        A_pad,           # (S,K,2)
        b0_pad,          # (S,K)
        v_expand_pad,    # (S,K)
        valid_mask,      # (S,K)
        sc_present,      # (S,)
        f_x_vec,         # (3,)
        g_x_mat,         # (3,2)
        kappa,
        alpha,
        radius,
        n_tau,           # static
    ):
        """Build unicycle occlusion CBF rows in vectorized form."""
        phi = phi_b[:n_tau, :]
        Phi = Phi_b[:n_tau, :, :]
        fcl = fcl_traj[:n_tau, :]
        tau = tau_points[:n_tau]

        pos = phi[:, :2]
        h = (
            jnp.einsum("skd,nd->snk", A_pad, pos)
            - b0_pad[:, None, :]
            - (radius + v_expand_pad[:, None, :] * tau[None, :, None])
        )

        valid_bool = valid_mask[:, None, :] > 0.5
        h_valid = jnp.where(valid_bool, h, jnp.full_like(h, -jnp.inf))

        max_h = jnp.max(h_valid, axis=2)
        finite_row = jnp.isfinite(max_h)
        max_h_safe = jnp.where(finite_row, max_h, 0.0)

        z = jnp.where(valid_bool, jnp.exp(kappa * (h - max_h_safe[:, :, None])), 0.0)
        Z = jnp.sum(z, axis=2)
        Z_safe = jnp.maximum(Z, 1e-12)
        lam = z / Z_safe[:, :, None]

        K_count = jnp.sum(valid_mask, axis=1)
        K_safe = jnp.maximum(K_count, 1.0)

        h_tilde = max_h_safe + (jnp.log(Z_safe) - jnp.log(K_safe)[:, None]) / kappa
        grad_pos = jnp.einsum("snk,skd->snd", lam, A_pad)

        dh_ds = jnp.einsum("snk,sk->sn", lam, -v_expand_pad)
        dh_dt = jnp.einsum("snk,sk->sn", lam, -v_expand_pad)
        time_residual = dh_dt - dh_ds

        zeros1 = jnp.zeros((grad_pos.shape[0], grad_pos.shape[1], 1), dtype=grad_pos.dtype)
        grad_h_phi = jnp.concatenate([grad_pos, zeros1], axis=2)

        Phi_f = jnp.einsum("nij,j->ni", Phi, f_x_vec)
        delta_f = Phi_f - fcl
        c_occ = jnp.einsum("snd,nd->sn", grad_h_phi, delta_f) + time_residual

        Phi_g = jnp.einsum("nij,jm->nim", Phi, g_x_mat)
        A_occ = jnp.einsum("snd,ndm->snm", grad_h_phi, Phi_g)

        rhs = c_occ + alpha * h_tilde
        norm_A_occ = jnp.linalg.norm(A_occ, axis=2)

        finite_mask = finite_row & jnp.isfinite(rhs) & jnp.all(jnp.isfinite(A_occ), axis=2)
        active_mask = sc_present[:, None] & finite_mask
        degenerate = (norm_A_occ < 1e-9) & (rhs < 0.0)
        keep = active_mask & (~degenerate)

        return -A_occ, rhs, keep
else:
    jax_occ_constraints_kernel_di = None
    jax_occ_constraints_kernel_uni = None
