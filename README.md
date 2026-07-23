# Occlusion CBF

Research code for evaluating Occlusion Control Barrier Functions (OCBFs) with
double-integrator, dynamic-unicycle, and unicycle robot models.

The repository is self-contained. The small portion of implementation formerly
supplied by `safe_control` now lives in `base_control/`; there is no runtime
dependency, Git submodule, or private repository. Its source revision and scope
are recorded in [`base_control/UPSTREAM.md`](base_control/UPSTREAM.md).

## Install

Install [uv](https://docs.astral.sh/uv/), clone this repository normally, and
create the environment:

```bash
git clone https://github.com/tkkim-robot/occlusion-cbf.git
cd occlusion-cbf
uv sync --frozen
```

Use `uv sync` instead only when intentionally creating or updating
`uv.lock`.

## Run a scenario

The maintained scenarios share one launcher:

```bash
uv run python -m examples.run_scenario \
  --scenario crowd -- \
  --model di \
  --baseline occlusion_cbf \
  --idx 1 \
  --disable-plot
```

Valid scenario names are `crowd`, `crowd_narrow`, `campus`, and `crosswalk`.
Pass `-- --help` after the selected scenario to see its own options:

```bash
uv run python -m examples.run_scenario --scenario crosswalk -- --help
```

`crowd` is the route-focused benchmark previously called `crowd2`.
`crowd_narrow` is the older, smaller layout previously called `crowd` or
`crowd1`. Thin `test_crowd2.py` and `test_crowd1.py` entry points remain for
older scripts. The obsolete hospital scenario has been removed.

See [`examples/README.md`](examples/README.md) for scenario usage and
[`position_control/README.md`](position_control/README.md) for the controller
layout.

## Implementation status

The temporal term in the occlusion constraint has been reviewed for the
project's pure facet-propagation model. Both explicit derivatives are
represented and cancel as required:

```text
c_occ = grad_h @ (Phi @ f(x) - f_backup(y)) + dh_dt - dh_ds
dh_dt = dh_ds = -sum(lambda_l * nu_l)
```

Questions about degenerate QP rows and solver-fallback semantics are
intentionally deferred; this refactor does not claim to resolve them.
Automated Optuna tuning is also future work and is not configured here.
