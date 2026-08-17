# Occlusion-CBF

Research code for Occlusion Control Barrier Functions with double-integrator,
dynamic-unicycle, and unicycle robot models. The repository includes the
Occlusion-CBF controller, comparison controllers, deterministic crowd
benchmarks, interactive scenarios, and Optuna tuning utilities.

## Installation

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the
locked environment:

```bash
git clone https://github.com/tkkim-robot/occlusion-cbf.git
cd occlusion-cbf
uv sync --frozen
```

Several controller paths use Gurobi and require a working installation and a
valid license. The Python package is installed by `uv`; license setup is
separate and should be completed before running those scenarios or benchmarks.

## Repository layout

- `base_control/`: robot models, tracking, and the visible-obstacle CBF-QP.
- `position_control/`: Occlusion-CBF and MPC comparison controllers.
- `dynamic_env/`: dynamic environments and robot simulation support.
- `examples/`: scenario definitions and the unified scenario launcher.
- `tools/`: benchmark, Optuna, and plotting commands.
- `tests/`: regression and scenario-contract tests.

Available scenarios are `crowd`, `crowd_narrow`, `campus`, and `crosswalk`.
Controller aliases include `occlusion_cbf`, `occlusion_cbf_terminal_relax`,
`cbf_qp`, `single_risk_mpc`, `control_tree_mpc`, `oacp_mpc`, and `oa_mpc`;
each scenario's `--help` output lists the combinations it supports.

## Single scenarios

Run a deterministic double-integrator crowd case with 50 moving obstacles:

```bash
uv run python -m examples.run_scenario --scenario crowd -- \
  --model di \
  --baseline occlusion_cbf \
  --n-rand 50 \
  --seed 42 \
  --idx 1 \
  --disable-plot
```

Run the corresponding unicycle setup with 30 moving obstacles:

```bash
uv run python -m examples.run_scenario --scenario crowd -- \
  --model uni \
  --baseline occlusion_cbf \
  --n-rand 30 \
  --seed 42 \
  --idx 1 \
  --disable-plot
```

Omit `--disable-plot` for the interactive animation. To inspect all
scenario-specific options, run:

```bash
uv run python -m examples.run_scenario --scenario crowd -- --help
```

The crosswalk scenario can include a bus occluder:

```bash
uv run python -m examples.run_scenario --scenario crosswalk -- \
  --model di \
  --baseline occlusion_cbf \
  --bus 1 \
  --idx 1 \
  --disable-plot
```

## Tuned Occlusion-CBF defaults

Crowd single runs and benchmark sweeps automatically load the committed
model-specific profiles:

- `position_control/ocbf/config/occlusion_cbf_di_params.yaml`
- `position_control/ocbf/config/occlusion_cbf_unicycle_params.yaml`

Explicit command-line or programmatic values take precedence over the tuned
profile. Dynamic Unicycle retains its model defaults because it has no tuned
profile.

## Benchmarking

Run the 100-case Occlusion-CBF crowd sweep for the double integrator:

```bash
uv run python -m tools.benchmark_crowd_trials \
  --scenario crowd \
  --baseline occlusion_cbf \
  --model di \
  --n-rand 50 \
  --seed 42 \
  --idx-start 1 \
  --idx-end 100 \
  --tf 500 \
  --out-dir output/crowd_di_50
```

The benchmark selects a conservative worker count automatically. Pass
`--workers N` to choose the number of case processes explicitly. To run the
five comparison controllers sequentially with the same case set, replace the
single baseline with the suite:

```bash
uv run python -m tools.benchmark_crowd_trials \
  --scenario crowd \
  --baseline-suite non_occlusion_5 \
  --model di \
  --n-rand 50 \
  --seed 42 \
  --idx-start 1 \
  --idx-end 100 \
  --tf 500 \
  --out-dir output/crowd_di_50_comparisons
```

Repeat with `--model uni` and the desired obstacle count for unicycle runs.
Generated CSV and JSON files should remain under the ignored `output/`
directory.

## Optuna tuning

The launcher runs the DI-50 and Unicycle-30 studies sequentially from an
immutable snapshot, supervised by a detached background process:

```bash
uv run python -m tools.launch_ocbf_optuna --dry-run
uv run python -m tools.launch_ocbf_optuna
```

It requires a clean Git worktree and defaults to 32 case workers. On a smaller
machine, pass matching `--workers` and `--batch-size` values that fit the
available CPUs. The lower-level tuner exposes single-study and
terminal-relaxation options:

```bash
uv run python -m tools.launch_ocbf_optuna --help
uv run python -m tools.tune_ocbf_optuna --help
```

Study databases, logs, and manifests are written below `output/` and are not
tracked.

## Plotting

Render the bundled benchmark outcome chart as SVG, with an optional PNG
preview:

```bash
uv run python tools/plot_crowd_benchmark_results.py \
  --output output/crowd_benchmark_results.svg \
  --preview-png output/crowd_benchmark_results.png
```

Both generated files remain in the ignored `output/` directory.

## Tests

Run the regression suite with:

```bash
uv run python -m unittest discover -s tests -v
```
