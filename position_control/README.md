# Position controllers

This package contains the project-specific safety controllers and comparison
baselines.

## Layout

- `occlusion_cbf_qp.py`: OCBF-QP assembly and solve path.
- `backup_controller.py`: backup-policy rollout, sensitivities, and terminal
  backup-set logic.
- `ocbf/`: shared OCBF defaults and JAX constraint kernels.
- `single_risk_mpc.py`, `control_tree_mpc.py`, `oacp_mpc.py`, and
  `oa_mpc.py`: comparison planners.
- `_mpc_common.py`: shared MPC utilities.

The generic robot, tracking, environment, and baseline CBF-QP pieces needed by
these controllers are vendored as the minimal `base_control` package. It is an
in-tree runtime dependency, not a submodule. See
[`../base_control/UPSTREAM.md`](../base_control/UPSTREAM.md) for provenance.

## Controller selection

Scenario scripts accept either their lower-level controller flag
(`--algo` or `--controller`) or the clearer `--baseline` aliases. For example:

```bash
uv run python -m examples.run_scenario \
  --scenario crowd -- \
  --model di \
  --baseline occlusion_cbf \
  --disable-plot
```

Use scenario help for the supported model/controller combinations:

```bash
uv run python -m examples.run_scenario --scenario crowd -- --help
```

## Current review boundary

The OCBF temporal/gradient expression is accepted for the pure
facet-propagation model used by this code. The NumPy and JAX paths both include
`dh_dt - dh_ds`; under this model the two derivatives are equal and their
residual is zero.

The behavior of degenerate QP rows and the safety interpretation of solver
fallbacks have not been changed in this refactor. Those questions remain a
separate follow-up review.
