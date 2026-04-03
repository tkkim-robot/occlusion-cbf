# Occlusion-CBF

This document is a practical setup and run guide for collaborators who are receiving this
`occlusion-cbf` repository for the first time.

It focuses on:
- `uv` environment setup
- `JAX` installation/verification (important for the current compute pipeline)
- Running the main scenarios (`examples/test_crowd.py`, `examples/test_crosswalk.py`)
- Frequently used CLI options (`--model`, `--idx`, `--save-ani`, `--n-rand`, `--bus`, etc.)


## 1. Repository Setup (`uv`)

### 1.1 Clone (with submodules)

This repository uses `safe_control` as a submodule, so cloning with `--recursive` is the safest option.

```bash
git clone --recursive <your-occlusion-cbf-repo-url>
cd occlusion-cbf
```

If you already cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

### 1.2 Install dependencies (`uv`)

This repository uses `uv` for dependency management.

```bash
uv sync
```

Notes:
- `safe_control` is configured as a local editable source in `pyproject.toml`, so no separate install step is needed.
- `jax` is included in dependencies (`jax>=0.4.30`).


## 2. JAX Installation / Activation Check (Important)

This repository uses `JAX` to accelerate heavy computation (especially backup rollout and
occlusion-constraint construction). It is recommended to verify that `JAX` imports correctly after installation.

### 2.1 Check version and devices

```bash
uv run python - <<'PY'
import jax
print("jax version:", jax.__version__)
print("devices:", jax.devices())
PY
```

### 2.2 First-run note (JIT warm-up)

- The first run can be slower because of JIT compilation.
- Repeated runs with the same shapes/settings are typically much faster and more stable.


## 3. Scenario 1: Crowd (`examples/test_crowd.py`)

This scenario tests dense moving obstacles and supports comparing `occlusion_cbf_qp`, `cbf_qp`,
and `backup_cbf_qp`.

### 3.1 Show CLI help

```bash
uv run python examples/test_crowd.py --help
```

### 3.2 Common run examples

`DI` model test (default random obs=50):

```bash
uv run python examples/test_crowd.py --model di --idx 7 --save-ani false
```

`DU` + occlusion CBF, 30 random moving obstacles, deterministic case selection:

```bash
uv run python examples/test_crowd.py --model du --n-rand 30 --idx 32 --save-ani false
```

Headless run (recommended for measuring computation time):

```bash
uv run python examples/test_crowd.py --model du --n-rand 30 --idx 32 --disable-plot --save-ani false
```

Algorithm comparison (`cbf_qp`, `backup_cbf_qp`):

```bash
uv run python examples/test_crowd.py --model du --algo cbf_qp --n-rand 30 --idx 32 --save-ani false
uv run python examples/test_crowd.py --model du --algo backup_cbf_qp --n-rand 30 --idx 32 --save-ani false
```

### 3.3 Key options (`test_crowd.py`)

- `--model {di,du,uni}`
  - `di`: Double Integrator
  - `du`: Dynamic Unicycle
  - `uni`: Unicycle

- `--algo {occlusion_cbf_qp,cbf_qp,backup_cbf_qp}`
  - Position controller algorithm.

- `--idx` / `--case-idx` (1-based)
  - Deterministically selects a random scenario case under a fixed `--seed`.
  - Example: `--seed 42 --idx 32`

- `--n-rand`
  - Number of random moving obstacles (crowd scenario only).

- `--no-rand-obs`
  - Disable random moving obstacles.

- `--disable-plot`
  - Disable animation rendering (recommended for compute profiling).

- `--save-ani`
  - Enable/disable animation saving.
  - Accepts `true` / `false`.
  - Example: `--save-ani false`

- `--tf`
  - Maximum simulation time [s].

- `--seed`
  - Random seed for crowd generation.

### 3.4 Performance note

- If plotting is enabled, rendering overhead can dominate the perceived runtime.
- To measure controller computation only, use `--disable-plot`.
- At the end of the run, the terminal prints:
  - `[STATS] Avg computation time (preprocess+solver, no plotting): ...`


## 4. Scenario 2: Crosswalk (`examples/test_crosswalk.py`)

This scenario includes a bus occluder and opposite-lane vehicles. `bus_type` lets you compare
occlusion OFF vs ON behavior.

### 4.1 Show CLI help

```bash
uv run python examples/test_crosswalk.py --help
```

### 4.2 Common run examples

Occlusion ON (`bus=1`), deterministic case replay:

```bash
uv run python examples/test_crosswalk.py --model di --bus 1 --idx 66 --save-ani false
```

Occlusion OFF (`bus=0`) comparison:

```bash
uv run python examples/test_crosswalk.py --model di --bus 0 --idx 66 --save-ani false
```

Headless run:

```bash
uv run python examples/test_crosswalk.py --model di --bus 1 --idx 66 --disable-plot
```

Batch evaluation (case search mode):

```bash
uv run python examples/test_crosswalk.py --batch-eval --num-trials 100 --seed 42 --disable-plot
```

### 4.3 Key options (`test_crosswalk.py`)

- `--model`
  - Currently only `di` is supported in this scenario.

- `--controller`
  - Position controller type (typically `occlusion_cbf_qp`, `cbf_qp`, or `backup_cbf_qp`).

- `--bus` / `--bus-type {0,1}`
  - `0`: bus occlusion OFF
  - `1`: bus occlusion ON

- `--idx` / `--case-idx` (1-based)
  - Deterministically selects a scenario case under a fixed seed.

- `--seed`
  - Random seed.

- `--batch-eval`
  - Enable batch evaluation mode.

- `--num-trials`
  - Number of batch trials.

- `--disable-plot`
  - Disable animation rendering.

- `--save-ani` / `--save_ani` / `--save-animation`
  - Enable/disable animation saving (`true` / `false` accepted).


## 5. Baseline Runs (Crowd Benchmark)

For crowd benchmarking, use `--baseline` (recommended) instead of `--algo`.

Baseline names:
- `occlusion_cbf` (ours, maps to `occlusion_cbf_qp`)
- `oa_mpc` (two variants by input constraint)
- `single_risk_mpc`
- `control_tree_mpc`

### 5.1 One command template

```bash
uv run python examples/test_crowd.py --model uni --baseline <baseline_name> --idx <idx num> --n-rand <random obs num>
```

### 5.2 Ours (`occlusion_cbf`)

```bash
uv run python examples/test_crowd.py --model uni --baseline occlusion_cbf --idx 1 --n-rand 30
```

### 5.3 OA-MPC baselines (omega constraint variants)

OA-MPC for unicycle is now compared with two `omega`-bound variants:

- `OA-MPC (wmax=default)`:
  - `--wmax default`  -> `w_max = 0.8`
- `OA-MPC (wmax=pi)`:
  - `--wmax pi` -> `w_max = pi`

```bash
uv run python examples/test_crowd.py --model uni --baseline oa_mpc --wmax default --idx 1 --n-rand 30
uv run python examples/test_crowd.py --model uni --baseline oa_mpc --wmax pi --idx 1 --n-rand 30
```

### 5.4 Single-Risk MPC baseline

```bash
uv run python examples/test_crowd.py --model uni --baseline single_risk_mpc --idx 1 --n-rand 30
```

### 5.5 Control-Tree-inspired baseline

```bash
uv run python examples/test_crowd.py --model uni --baseline control_tree_mpc --idx 1 --n-rand 30
```

### 5.6 Batch Trials Helper (idx sweep)

Run a single baseline over an idx range (default 1..100) with plotting disabled, and print:
- success / collision / infeasible idx lists
- average solve time over idx
- average control intervention over idx

```bash
uv run python tools/benchmark_crowd_trials.py --baseline occlusion_cbf --model uni --seed 42 --idx-start 1 --idx-end 100 --n-rand 50 --tf 100
```

OA-MPC variants can be swept with:

```bash
uv run python tools/benchmark_crowd_trials.py --baseline oa_mpc --wmax default --model uni --seed 42 --idx-start 1 --idx-end 100 --n-rand 50 --tf 100
uv run python tools/benchmark_crowd_trials.py --baseline oa_mpc --wmax pi --model uni --seed 42 --idx-start 1 --idx-end 100 --n-rand 50 --tf 100
```

Run the 5 non-occlusion baselines sequentially in one command:

```bash
uv run python tools/benchmark_crowd_trials.py --baseline-suite non_occlusion_5 --model uni --seed 42 --idx-start 1 --idx-end 100 --n-rand 50 --tf 100
```


## 6. Recommended First Commands (for collaborators)

Use the following sequence to verify setup and run both main scenarios.

```bash
uv sync
uv run python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
uv run python examples/test_crowd.py --model du --n-rand 30 --idx 32 --disable-plot --save-ani false
uv run python examples/test_crosswalk.py --model di --bus 1 --idx 66 --disable-plot
```
