# Occlusion-CBF

Research code for evaluating Occlusion Control Barrier Functions with
double-integrator, dynamic-unicycle, and unicycle robot models. The repository
contains the proposed Occlusion-CBF controller, five comparison controllers,
reproducible crowd scenarios, tuning tools, and the complete benchmark
artifacts.

Robot models, tracking utilities, environments, and the CBF-QP baseline live
in `base_control/`. Occlusion-CBF and the MPC comparison controllers live in
`position_control/`.

## Installation

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the
locked environment:

```bash
git clone https://github.com/tkkim-robot/occlusion-cbf.git
cd occlusion-cbf
uv sync --frozen
```

Use `uv sync` without `--frozen` only when intentionally updating `uv.lock`.

## Quick start

Run Occlusion-CBF on the first canonical crowd case:

```bash
uv run python -m examples.run_scenario \
  --scenario crowd -- \
  --model di \
  --baseline occlusion_cbf \
  --idx 1 \
  --disable-plot
```

Plotting is enabled unless `--disable-plot` is supplied. Add
`--save-animation` to export an MP4.

Each scenario exposes its complete command-line interface:

```bash
uv run python -m examples.run_scenario --scenario crowd -- --help
```

## Scenarios

| Name | Description |
|---|---|
| `crowd` | Route-focused forced-emergence benchmark used for the reported results. |
| `crowd_narrow` | Compact narrow-crowd experiment. |
| `campus` | Long campus walkway with moving pedestrians. |
| `crosswalk` | Crosswalk scene with an optional bus occluder. |

See [`examples/README.md`](examples/README.md) for scenario-specific examples.

## Controllers

Select a controller with `--baseline`:

| Name | Controller |
|---|---|
| `occlusion_cbf` | Occlusion-CBF safety filter |
| `cbf_qp` | Visible-obstacle CBF-QP |
| `single_risk_mpc` | Single-Risk MPC |
| `control_tree_mpc` | Control-Tree MPC |
| `oacp_mpc` | OACP-MPC |
| `oa_mpc` | OA-MPC |

The available controller/model combinations are listed by each scenario's
`--help` output. Controller implementation details are documented in
[`position_control/README.md`](position_control/README.md).

## Benchmarking

Run a deterministic crowd sweep with:

```bash
uv run python -m tools.benchmark_crowd_trials \
  --scenario crowd \
  --baseline occlusion_cbf \
  --model di \
  --n-rand 10 \
  --seed 42 \
  --idx-start 1 \
  --idx-end 100
```

The final comparison contains 100 cases per configuration. Each outcome cell
below is `Success / Collision / Infeasible`:

| Method | DI-10 | DI-30 | DI-50 | Unicycle-10 | Unicycle-20 | Unicycle-30 |
|---|---:|---:|---:|---:|---:|---:|
| CBF-QP | 86 / 13 / 1 | 45 / 38 / 17 | 33 / 39 / 28 | 55 / 4 / 41 | 33 / 5 / 62 | 18 / 5 / 77 |
| Single-Risk MPC | 23 / 0 / 77 | 1 / 0 / 99 | 0 / 0 / 100 | 15 / 0 / 85 | 3 / 0 / 97 | 0 / 0 / 100 |
| Control-Tree MPC | 53 / 0 / 47 | 10 / 0 / 90 | 1 / 0 / 99 | 21 / 0 / 79 | 6 / 0 / 94 | 3 / 0 / 97 |
| OACP-MPC | 46 / 2 / 52 | 17 / 6 / 77 | 5 / 6 / 89 | 56 / 43 / 1 | 30 / 70 / 0 | 14 / 86 / 0 |
| OA-MPC | 0 / 1 / 99 | 0 / 1 / 99 | 0 / 3 / 97 | 0 / 15 / 85 | 0 / 7 / 93 | 0 / 6 / 94 |
| **Occlusion-CBF** | **97 / 0 / 3** | **87 / 0 / 13** | **67 / 0 / 33** | **84 / 1 / 15** | **73 / 0 / 27** | **69 / 0 / 31** |

Comparable warmed controller timing was measured sequentially on an Apple M4.
Each entry is mean milliseconds per controller step:

| Method | DI-10 | DI-30 | DI-50 | Unicycle-10 | Unicycle-20 | Unicycle-30 |
|---|---:|---:|---:|---:|---:|---:|
| CBF-QP | 0.659 | 0.740 | 0.753 | 0.691 | 0.711 | 0.797 |
| Single-Risk MPC | 12.990 | 13.757 | 15.356 | 7.249 | 8.138 | 8.676 |
| Control-Tree MPC | 106.676 | 110.651 | 117.039 | 48.304 | 54.170 | 58.929 |
| OACP-MPC | 87.368 | 89.434 | 88.830 | 80.533 | 76.900 | 78.478 |
| OA-MPC | 93.822 | 151.444 | 186.946 | 69.225 | 101.502 | 117.458 |
| **Occlusion-CBF** | **1.596** | **2.346** | **3.881** | **4.116** | **4.498** | **4.507** |

The first 10 warm-up steps of each case are excluded. Controller-internal
timing boundaries differ, so these are controller-reported compute times
rather than uniform outer-loop latency.

The [complete result report](output/comparison_baselines_main_358cb84_20260728_184721/final_results_table.md)
contains the measurement protocol and artifact provenance. The
[case-reproduction guide](OCBF_BENCHMARK_REPRODUCTION.md) documents exact
seed/index semantics and expected outcomes.

## Tuned profiles

Double Integrator and Unicycle load their selected Occlusion-CBF profiles
automatically:

- `position_control/ocbf/config/occlusion_cbf_di_params.yaml`
- `position_control/ocbf/config/occlusion_cbf_unicycle_params.yaml`

Explicit CLI or programmatic parameters override the committed profiles.
Dynamic Unicycle uses its model defaults.

Run an Optuna study with:

```bash
uv run python -m tools.tune_ocbf_optuna \
  --model di \
  --n-rand 50 \
  --workers 4 \
  --trials 40 \
  --output-dir output/optuna_di_n50
```

## Tests

Run the full regression suite:

```bash
uv run python -m unittest discover -s tests -v
```

## Technical notes

Occlusion-CBF uses pure facet propagation. Under this model,
`dh_dt == dh_ds`, and the explicit residual cancels in both NumPy and JAX
constraint paths:

```text
c_occ = grad_h @ (Phi @ f(x) - f_backup(y)) + dh_dt - dh_ds
dh_dt = dh_ds = -sum(lambda_l * nu_l)
```

OA-MPC constructs occlusion boundaries from adjacent LiDAR discontinuities,
does not wrap partial-field-of-view scans, and applies complementarity only to
dynamic reachable-set and hidden-region rows. Static point-cloud rows remain
hard constraints.
