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

The optimized controller profiles are committed as:

- `config/occlusion_cbf_di_params.yaml`
- `config/occlusion_cbf_unicycle_params.yaml`

Every maintained test and benchmark runner loads the matching profile when
Occlusion-CBF is selected. Built-in scenario values are replaced by the YAML
profile, while explicit CLI and programmatic overrides still take precedence.
Dynamic Unicycle keeps its existing defaults because it has not been tuned.
The shared tracking-controller constructor also fills missing values from
these files before creating the robot, which covers new scenario entry points.

Inspect the canonical runner for the exact current flags:

```bash
uv run python -m examples.run_scenario --scenario crowd -- --help
```

## Optuna tuning

`tools/tune_ocbf_optuna.py` tunes the Occlusion-CBF controller on the exact
canonical crowd cases used by the paper benchmark. Double Integrator and
Unicycle use separate studies and search spaces. Full-circle (`2*pi`) obstacle
selection, `barrier_unexpand`, barrier smoothing, visibility, and hard safety
rows remain fixed.

The objective is lexicographic: minimize collisions, then maximize successes,
then minimize mean controller compute time. Completed trials evaluate all
requested cases; median pruning only occurs after deterministic case batches.
The active-occlusion count is sampled from `all`, `3`, `5`, and `10` only
(`0` is the controller's internal representation of `all`).
Optuna state, every per-case row, best parameters, and W&B aggregates are
persisted in the selected output directory.

Run one study directly:

```bash
.venv-optuna/bin/python -m tools.tune_ocbf_optuna \
  --model di --n-rand 50 --workers 4 --trials 40 \
  --output-dir output/optuna_di_n50 --wandb
```

Launch both paper studies in detached, CPU-isolated sessions:

```bash
python -m tools.launch_ocbf_optuna
```

## Deferred QP review

This refactor does not change or certify the treatment of zero-authority
(degenerate) constraint rows or the semantics of QP solver fallback modes.
Those controller-level questions are intentionally deferred for a separate
discussion.
