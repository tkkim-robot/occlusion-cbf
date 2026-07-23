# Scenarios

Use the shared launcher for maintained experiments:

```bash
uv run python -m examples.run_scenario --scenario SCENARIO -- [scenario options]
```

| Scenario | Purpose |
| --- | --- |
| `crowd` | Canonical route-focused crowd benchmark; formerly `crowd2`. |
| `crowd_narrow` | Legacy small crowd layout; formerly `crowd`/`crowd1`. |
| `campus` | Long campus walkway with moving pedestrians. |
| `crosswalk` | Crosswalk scene with an optional bus occluder. |

The former hospital experiment is not maintained and has been removed.

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

## Compatibility names

New code should import `examples.test_crowd` or
`examples.test_crowd_narrow`. For existing integrations:

- `examples.test_crowd2` forwards to `examples.test_crowd`.
- `examples.test_crowd1` forwards to `examples.test_crowd_narrow`.
- The benchmark tool accepts `crowd2` and `crowd1` as aliases for `crowd` and
  `crowd_narrow`, respectively.

For multi-view evaluation, `examples.test_multi_crowd2` similarly forwards to
`examples.test_multi_crowd`.
