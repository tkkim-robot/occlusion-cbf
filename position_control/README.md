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

Shared robot models, tracking, environments, and the CBF-QP baseline are
provided by `base_control/`.

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

## Controller behavior

Occlusion-CBF uses the pure facet-propagation model documented in
[`ocbf/README.md`](ocbf/README.md). The NumPy and JAX paths both include
`dh_dt - dh_ds`; the two derivatives are equal under this model, so their
explicit residual is zero.

OA-MPC constructs occlusion boundaries from adjacent LiDAR discontinuities
and avoids first/last-beam wraparound for partial-field-of-view scans.
Dynamic reachable-set circles and hidden occlusion capsules use
complementarity, while static point-cloud circles remain hard constraints.
The Double Integrator crowd configuration uses the exact terminal stopping
constraint.

Zero-authority QP rows and solver fallback modes are implementation-level
behaviors and do not provide a separate formal safety certificate.
