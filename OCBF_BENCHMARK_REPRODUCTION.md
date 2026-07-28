# Occlusion-CBF crowd benchmark case reproduction

This guide reproduces one saved case from the tuned Occlusion-CBF crowd
benchmark. It also records the complete success, collision, and infeasible
case-index partitions for the six benchmark configurations.

The catalog below comes from the 2026-07-28 sweep of cases 1–100 with base
seed `42`, generated from source commit
`80eb3dc05565690e192da1f6f4e9f76ade12f96b`. Every finalized CSV contained
100 unique indices and no exception rows.

> **Unicycle compatibility note:** the saved Unicycle catalog was generated
> with the former forward-only bound `v_min=0`. The current canonical profile
> is reverse-capable with `v_min=-1`. Add `--uni-forward-only true` to replay
> the old Unicycle rows exactly. The Double Integrator catalog is unaffected.
> Reverse-capable Unicycle tuning and result regeneration are intentionally
> deferred.

## Setup

Create the locked environment from the repository root:

```bash
uv sync --frozen
```

Occlusion-CBF is the default method. Its committed tuned profile loads
automatically for the selected dynamics:

- `position_control/ocbf/config/occlusion_cbf_di_params.yaml`
- `position_control/ocbf/config/occlusion_cbf_unicycle_params.yaml`

Running the launcher without any arguments therefore starts Double Integrator,
10-obstacle, case 1 with tuned Occlusion-CBF and a live plot:

```bash
uv run python -m examples.run_scenario
```

The launcher and crowd runner supply the paper configuration internally:
base seed `42`, final time `500`, forced-emergence `v2`, six emergence events,
the validated occluder geometry, visible-obstacle HOCBF, reverse-capable
Unicycle motion, and the benchmark CPU/JAX environment. Users do not need to
pass any of those options for a current run.

## Seed and case-index semantics

Within a fixed obstacle-count configuration, the random realization is
selected by the pair `(--seed, --idx)`, not by either value alone. The
benchmark used base seed `42` and indices 1 through 100. Seed `42` is the
runner default, so a cataloged scene normally needs only `--idx`.

`--seed` is the base seed. `--idx` is a one-based draw number from NumPy's
generator:

```python
rng = numpy.random.default_rng(base_seed)
derived_case_seed = the idx-th rng.integers(0, 2**31 - 1)
```

Here, `--idx` is the benchmark case (scene) index from 1 through 100. It is
not an Optuna optimization-trial number.

For example:

| Base seed | Case index | Derived case seed |
|---:|---:|---:|
| 42 | 1 | 191664963 |
| 42 | 2 | 1662057957 |
| 42 | 3 | 1405681631 |
| 42 | 28 | 1766867109 |

Replay a catalog entry with its case index, such as `--idx 28`. If selecting
the seed explicitly, use the original pair `--seed 42 --idx 28`; do not
replace `--seed` with the derived case seed. The single-case runner prints
both values before printing its result.

The launcher automatically applies the same one-thread CPU/JAX environment
used by the benchmark before NumPy and JAX are imported. This matters for
solver-boundary cases such as Unicycle, 10 obstacles, index 28. The expert
opt-out `--no-benchmark-cpu-env` should not be used for exact replay.

## Reproduce one Double Integrator case

Only the dynamics, obstacle count, and case index are needed:

```bash
uv run python -m examples.run_scenario \
  --model di \
  --n-rand 50 \
  --idx 1
```

Use obstacle count `10`, `30`, or `50`. Occlusion-CBF is already the default;
`--method occlusion_cbf` may be added when an explicit method label is useful.

## Reproduce one Unicycle case

The tuned Unicycle controller profile and the reverse-capable `v_min=-1`
robot bound load automatically:

```bash
uv run python -m examples.run_scenario \
  --model uni \
  --n-rand 10 \
  --idx 28
```

Use obstacle count `10`, `20`, or `30`.

To reproduce an entry in the legacy Unicycle outcome catalog below, add the
former actuation assumption explicitly:

```bash
uv run python -m examples.run_scenario \
  --model uni \
  --n-rand 10 \
  --idx 28 \
  --uni-forward-only true
```

`--n-rand` is the only additional benchmark selector that cannot be inferred
from `--model` and `--idx`: every dynamics has three obstacle-count
configurations, and the same index can have a different outcome in each.
When omitted, it defaults to `10`.

The final lines identify the exact generated scene and outcome:

```text
[CASE] base_seed=42 idx=28 derived_case_seed=1766867109
[RESULT] outcome=collision steps=2561 sim_time_s=128.05 ...
```

The cataloged `--tf 500` outcomes are `success`, `collision`, and
`infeasible`. A deliberately shortened run can instead report
`timeout/deadlock`.

## Plotting and animation

Plotting is enabled by default, so the two commands above show the case live.

To also save an MP4, add just:

```bash
--save-animation
```

The default animation settings save every fifth controller step at 150 DPI.
To save without displaying the live plot, use
`--save-animation --disable-plot`. Advanced users may override
`--animation-frame-stride` or `--animation-frame-dpi`; these affect only
rendering, not simulation dynamics.

Without an explicit output directory, each case uses a non-colliding path:

```text
output/animations/crowd/occlusion_cbf/<model>_n<N>_seed<SEED>_idx<IDX>/tracking.mp4
```

For example, Unicycle case 28 with 10 obstacles is saved to:

```text
output/animations/crowd/occlusion_cbf/uni_n10_seed42_idx28/tracking.mp4
```

Override the directory for a presentation asset with:

```bash
--animation-subdir presentation/uni_n10_collision_idx28
```

The custom path is relative to `output/animations/`. Video export requires
`ffmpeg` on `PATH`. Re-running the same case safely replaces its video. If
`ffmpeg` is missing or encoding fails, the PNG frames are retained instead of
being deleted.

## Verified single-case replays

These public-CLI checks were compared with the finalized benchmark rows:

| Dynamics | Obstacles | Index | Derived seed | Expected result | Steps | Replay |
|---|---:|---:|---:|---|---:|---|
| DI | 10 | 1 | 191664963 | success | 1944 | exact |
| DI | 50 | 1 | 191664963 | infeasible | 522 | exact |
| Unicycle | 10 | 28 | 1766867109 | collision | 2561 | exact with `--uni-forward-only true` |

## Benchmark summary

Each outcome entry below is a case count out of 100, not a percentage.

| Dynamics | Obstacles | Success | Collision | Infeasible |
|---|---:|---:|---:|---:|
| DI | 10 | 97 | 0 | 3 |
| DI | 30 | 87 | 0 | 13 |
| DI | 50 | 67 | 0 | 33 |
| Unicycle | 10 | 84 | 1 | 15 |
| Unicycle | 20 | 73 | 0 | 27 |
| Unicycle | 30 | 69 | 0 | 31 |

DI with 50 obstacles exactly reproduced the selected Optuna trial 95.
Unicycle with 30 obstacles exactly reproduced the selected Optuna trial 100.
The Unicycle summary and index partitions below describe the legacy
forward-only run, not the current reverse-capable default.

## Complete outcome index catalog

All indices below are one-based and use base seed `42`.

### Double Integrator, 10 obstacles

Success (97):

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
```

Collision (0): none.

Infeasible (3):

```text
43, 51, 88
```

### Double Integrator, 30 obstacles

Success (87):

```text
2, 3, 4, 5, 6, 7, 9, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100
```

Collision (0): none.

Infeasible (13):

```text
1, 8, 10, 11, 13, 15, 24, 58, 64, 80, 81, 83, 94
```

### Double Integrator, 50 obstacles

Success (67):

```text
2, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 20, 21, 23, 25, 27, 28, 30, 33, 34, 36, 37, 38, 40, 41, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 72, 74, 75, 76, 79, 81, 83, 85, 86, 87, 89, 90, 91, 92, 93, 94, 97, 98, 100
```

Collision (0): none.

Infeasible (33):

```text
1, 3, 4, 5, 11, 12, 15, 19, 22, 24, 26, 29, 31, 32, 35, 39, 42, 43, 50, 53, 58, 66, 67, 73, 77, 78, 80, 82, 84, 88, 95, 96, 99
```

### Unicycle, 10 obstacles

Success (84):

```text
1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100
```

Collision (1):

```text
28
```

Infeasible (15):

```text
3, 9, 20, 39, 48, 49, 55, 65, 66, 74, 75, 79, 80, 84, 95
```

### Unicycle, 20 obstacles

Success (73):

```text
2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 18, 20, 21, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 68, 69, 71, 72, 73, 74, 76, 77, 78, 79, 81, 82, 83, 84, 87, 88, 90, 91, 94, 96, 97, 98, 99, 100
```

Collision (0): none.

Infeasible (27):

```text
1, 8, 9, 11, 15, 16, 17, 19, 22, 25, 31, 37, 52, 53, 58, 65, 66, 67, 70, 75, 80, 85, 86, 89, 92, 93, 95
```

### Unicycle, 30 obstacles

Success (69):

```text
3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 23, 24, 25, 26, 27, 29, 31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 64, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 81, 82, 85, 86, 87, 88, 89, 91, 92, 93, 94, 97, 98
```

Collision (0): none.

Infeasible (31):

```text
1, 2, 5, 6, 18, 19, 22, 28, 30, 33, 34, 36, 44, 49, 57, 61, 62, 63, 65, 68, 70, 78, 79, 80, 83, 84, 90, 95, 96, 99, 100
```

## Source artifacts

The original full rows and summaries are under:

```text
output/ocbf_tuned_benchmark_20260728_145048/
```

`output/` is intentionally gitignored, so this tracked guide is the durable
record of the outcome partitions. See `examples/README.md` for general
scenario usage and `position_control/ocbf/README.md` for controller and tuned
profile details.
