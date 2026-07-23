# Occlusion-CBF internals

This directory holds shared defaults and JAX kernels for
`position_control/occlusion_cbf_qp.py`. Backup rollout and sensitivity
propagation live in `position_control/backup_controller.py`.

## Constraint flow

For each control step, the controller:

1. Constructs convex hidden-region polygons from visible occluders.
2. Selects the active occlusion scenarios.
3. Rolls out the configured backup policy and its state-transition
   sensitivities.
4. Evaluates smoothed facet barriers along the rollout.
5. Builds rollout and terminal OCBF-QP rows, optionally alongside visible
   obstacle CBF/HOCBF rows.
6. Solves for the safety-filtered control.

The canonical `crowd` benchmark is the route-focused layout formerly called
`crowd2`. New configuration and documentation use `crowd`; old names remain
only as compatibility aliases.

## Temporal derivative convention

This project always uses pure facet propagation:

```text
a_dot = 0
nu_dot = 0
b_dot = nu
```

For the softmax facet weights `lambda_l`, this gives

```text
dh_ds = -sum(lambda_l * nu_l)
dh_dt = -sum(lambda_l * nu_l)
```

The implemented temporal term is therefore:

```text
c_occ = grad_h @ (Phi @ f(x) - f_backup(y)) + dh_dt - dh_ds
```

The explicit residual is zero under the stated model. This temporal/gradient
correction has been reviewed and is considered correct for that assumption in
both the NumPy and JAX constraint paths.

## Configuration

Shared choices and defaults live in `defaults.py`. Scenario runners expose the
main experimental parameters, including:

- backup horizon and integration step;
- terminal margin and terminal-row selection;
- active-occlusion count and selection mode;
- facet softmax parameter;
- backup-reference mode and scenario blending;
- optional visible-obstacle HOCBF rows.

Inspect the canonical runner for the exact current flags:

```bash
uv run python -m examples.run_scenario --scenario crowd -- --help
```

Automated Optuna search is planned but not included yet. Until that follow-up,
configuration changes should be explicit and recorded with experiment
results.

## Deferred QP review

This refactor does not change or certify the treatment of zero-authority
(degenerate) constraint rows or the semantics of QP solver fallback modes.
Those controller-level questions are intentionally deferred for a separate
discussion.
