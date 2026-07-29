# Scenarios

Use the shared launcher for maintained experiments:

```bash
uv run python -m examples.run_scenario --scenario SCENARIO -- [scenario options]
```

| Scenario | Purpose |
| --- | --- |
| `crowd` | Route-focused forced-emergence benchmark. |
| `crowd_narrow` | Compact narrow-crowd layout. |
| `campus` | Long campus walkway with moving pedestrians. |
| `crosswalk` | Crosswalk scene with an optional bus occluder. |

## Examples

Run the canonical crowd benchmark without plotting:

```bash
uv run python -m examples.run_scenario \
  --scenario crowd -- \
  --model di \
  --baseline occlusion_cbf \
  --seed 42 \
  --idx 1 \
  --disable-plot
```

Run the campus layout:

```bash
uv run python -m examples.run_scenario \
  --scenario campus -- \
  --model uni \
  --baseline occlusion_cbf \
  --idx 1 \
  --disable-plot
```

Run the crosswalk with bus occlusion enabled:

```bash
uv run python -m examples.run_scenario \
  --scenario crosswalk -- \
  --model di \
  --baseline occlusion_cbf \
  --bus 1 \
  --idx 1 \
  --disable-plot
```

Each scenario owns its detailed flags:

```bash
uv run python -m examples.run_scenario --scenario crowd -- --help
uv run python -m examples.run_scenario --scenario campus -- --help
uv run python -m examples.run_scenario --scenario crosswalk -- --help
```

Common controller baselines include `occlusion_cbf`, `cbf_qp`, `oa_mpc`,
`single_risk_mpc`, `control_tree_mpc`, and `oacp_mpc`; the crosswalk runner
supports the subset shown by its `--help`.

When `occlusion_cbf` is selected, Double Integrator and Unicycle automatically
load their committed optimized YAML profiles from
`position_control/ocbf/config/`. Explicit scenario flags still override those
values. Dynamic Unicycle uses its model defaults.

## Benchmark sweeps

The benchmark tool runs deterministic case ranges for one controller or the
predefined comparison suite:

```bash
uv run python -m tools.benchmark_crowd_trials \
  --scenario crowd \
  --baseline occlusion_cbf \
  --model uni \
  --n-rand 20 \
  --seed 42 \
  --idx-start 1 \
  --idx-end 100
```

Use `uv run python -m tools.benchmark_crowd_trials --help` for parallelism,
output, controller, and scenario-generation options.
