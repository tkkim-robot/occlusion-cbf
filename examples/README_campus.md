# Campus Benchmark Reproduction Guide

This document is the collaborator-facing run guide for `examples/test_campus.py`.
It records the current narrow-and-long hallway benchmark setup and the selected
case indices that were useful for qualitative comparison videos.

Use this file when you want to:

- reproduce the exact campus benchmark cases from this repo
- run all baselines on the same deterministic obstacle layout
- save the animations for later import into MetaUrban or other visualization code


## 1. What This Scenario Is

`examples/test_campus.py` is a 2D benchmark scenario built on top of the
`examples/test_crowd.py` controller stack.

It is not a MetaUrban simulator. Instead:

- `occlusion-cbf` owns the robot dynamics, control, obstacle motion, and collision logic
- `test_campus.py` generates the deterministic 2D benchmark case
- external visualization or 3D wrappers can mirror the resulting trajectories


## 2. Main Benchmark Metric

The main benchmark metric is a 3-way outcome over the same trial set:

- `success`
- `collision`
- `infeasible`

For benchmarking and reporting in this repo, `infeasible` also includes
timeout/deadlock cases.

Secondary logged metrics are:

- average solve time
- average control intervention
- average simulation time


## 3. Shared Campus Environment

All cases below assume the current `test_campus.py` layout:

- workspace width: `15.0 m`
- workspace height: `40.0 m`
- goal threshold: `0.5 m`
- route start waypoint: `(7.5, 1.0)`
- route goal waypoint: `(7.5, 36.0)`
- route direction: bottom-to-top along the center corridor
- random obstacle count for benchmark cases: `30`
- simulation horizon: `tf = 200`
- case seed: `0`

Important:

- The obstacle layout is deterministic for the pair `(--seed, --idx)`.
- If you change `--seed`, the selected `idx` values in this document are no longer valid.


## 4. Shared Dynamic Obstacle Setting

These settings are shared by both the Double Integrator and Unicycle benchmark cases below.

### 4.1 Obstacle geometry and motion

- obstacle type: dynamic pedestrian (`type = 1`)
- pedestrian radius: `0.35 m`
- number of pedestrians: `30`
- random heading initialization: uniform in `[-pi, pi]`
- base pedestrian speed range: `0.3 ~ 0.5 m/s`
- spawn x-range: `[2.5, 12.5]`
- spawn y-range: `[6.0, 40.0]`
- start/goal clearance and pairwise overlap checks are enforced during sampling

### 4.2 Occlusion-related setting

- visible obstacle scale: `0.7`
  - an occluded obstacle becomes visible once about `30%` of the body is exposed
- hidden obstacle speed bound: `1.0 m/s`
  - CLI name: `--hidden-obs-velocity 1.0`
- in the campus scenario, FOV-occluded pedestrians are forced to move with the
  exact occluded speed bound while they remain occluded:
  - `occluded_speed_boost_enable = True`
  - `occluded_speed_boost_vmax = 1.0`
  - `occluded_speed_boost_fov_only = True`
  - `occluded_speed_boost_exact = True`
  - on/off hysteresis steps = `2 / 5`


## 5. Model Defaults Used By This Scenario

### 5.1 Double Integrator (`--model di`)

Campus default robot configuration:

- `v_max = 1.0`
- `a_max = 1.0`
- `radius = 0.25`
- `sensing_range = 10.0`
- `fov_angle = 360`
- `dynamic_obs_types = [1]`

Campus default occlusion-backup configuration:

- `T_horizon = 1.0`
- `vref_scenario_softmax_kappa = 0.0`
- `rho_T = auto`

For the selected DI case below, we use:

- `--vref los`
- `vref_mode_occ` is left at the controller default (`strict`)
- visible-obstacle HOCBF inside `occlusion_cbf` is left disabled


### 5.2 Unicycle (`--model uni`)

Campus default robot configuration:

- `v_max = 1.0`
- `w_max = 0.8`
- `radius = 0.25`
- `sensing_range = 10.0`
- `fov_angle = 360`
- `dynamic_obs_types = [1]`

Campus default occlusion-backup configuration:

- `T_horizon = 0.5`
- `vref_scenario_softmax_kappa = 0.0`

For the selected Unicycle cases below, we use the tuned occlusion-CBF parameters:

- `--vref los`
- `--uni-reverse-gate-angle 0.625`
- `--uni-reverse-gate-power 1.05`
- `vref_mode_occ` is left at the controller default (`strict`)
- visible-obstacle HOCBF inside `occlusion_cbf` is left disabled

These are the tuned `Unicycle2D` parameters that gave the best success rate in
the current campus setup for `T_horizon = 0.5`.


## 6. Supported Baselines For Campus

These are the supported campus baselines:

- `occlusion_cbf`
- `cbf_qp`
- `single_risk_mpc`
- `control_tree_mpc`
- `oacp_mpc`
- `oa_mpc`

Notes:

- `backup_cbf_qp` is intentionally not used here.
- For `oa_mpc`, the commands below disable the safe-stop solver fallback:
  - `--oa-allow-solver-fallback false`


## 7. Common Shell Helper

All one-shot animation commands below use the same helper.

```bash
run_save() {
  local out_name="$1"
  shift
  rm -f output/animations/tracking.mp4
  uv run python examples/test_campus.py "$@" "${COMMON_ARGS[@]}" &&
  mv output/animations/tracking.mp4 "${OUT_DIR}/${out_name}.mp4"
}
```


## 8. Double Integrator Reference Case

Selected case:

- model: `di`
- index: `76`

This is the current DI reference case for the campus benchmark.

### 8.1 Single occlusion-CBF command

```bash
uv run python examples/test_campus.py \
  --model di \
  --baseline occlusion_cbf \
  --seed 0 \
  --n-rand 30 \
  --tf 200 \
  --idx 76 \
  --goal-threshold 0.5 \
  --sensing-range 8.0 \
  --ped-speed-min 0.3 \
  --ped-speed-max 0.5 \
  --ped-radius 0.35 \
  --occ-visible-scale 0.7 \
  --hidden-obs-velocity 1.0 \
  --vref los
```

### 8.2 One-shot animation export for all baselines

```bash
mkdir -p output/animations/campus_di_idx76 && \
OUT_DIR=output/animations/campus_di_idx76 && \
COMMON_ARGS=(
  --model di
  --seed 0
  --n-rand 30
  --tf 200
  --idx 76
  --goal-threshold 0.5
  --sensing-range 8.0
  --ped-speed-min 0.3
  --ped-speed-max 0.5
  --ped-radius 0.35
  --occ-visible-scale 0.7
  --hidden-obs-velocity 1.0
  --save-animation true
) && \
run_save() {
  local out_name="$1"
  shift
  rm -f output/animations/tracking.mp4
  uv run python examples/test_campus.py "$@" "${COMMON_ARGS[@]}" &&
  mv output/animations/tracking.mp4 "${OUT_DIR}/${out_name}.mp4"
} && \
run_save occlusion_cbf_vref_los_idx76 \
  --baseline occlusion_cbf \
  --vref los && \
run_save cbf_qp_idx76 \
  --baseline cbf_qp && \
run_save single_risk_mpc_idx76 \
  --baseline single_risk_mpc && \
run_save control_tree_mpc_idx76 \
  --baseline control_tree_mpc && \
run_save oacp_mpc_idx76 \
  --baseline oacp_mpc && \
run_save oa_mpc_no_fallback_idx76 \
  --baseline oa_mpc \
  --oa-allow-solver-fallback false
```


## 9. Unicycle Reference Cases

Selected cases:

- model: `uni`
- indices: `17`, `25`

These are the current Unicycle reference cases for the campus benchmark.

### 9.1 Shared occlusion-CBF setting for both Unicycle cases

Use these extra flags for `occlusion_cbf`:

- `--vref los`
- `--uni-reverse-gate-angle 0.625`
- `--uni-reverse-gate-power 1.05`

### 9.2 Single occlusion-CBF command template

Replace `IDX` with `17` or `25`.

```bash
IDX=17
uv run python examples/test_campus.py \
  --model uni \
  --baseline occlusion_cbf \
  --seed 0 \
  --n-rand 30 \
  --tf 200 \
  --idx "${IDX}" \
  --goal-threshold 0.5 \
  --sensing-range 8.0 \
  --ped-speed-min 0.3 \
  --ped-speed-max 0.5 \
  --ped-radius 0.35 \
  --occ-visible-scale 0.7 \
  --hidden-obs-velocity 1.0 \
  --vref los \
  --uni-reverse-gate-angle 0.625 \
  --uni-reverse-gate-power 1.05
```

### 9.3 One-shot animation export for `idx 17`

```bash
mkdir -p output/animations/campus_uni_idx17 && \
OUT_DIR=output/animations/campus_uni_idx17 && \
COMMON_ARGS=(
  --model uni
  --seed 0
  --n-rand 30
  --tf 200
  --idx 17
  --goal-threshold 0.5
  --sensing-range 8.0
  --ped-speed-min 0.3
  --ped-speed-max 0.5
  --ped-radius 0.35
  --occ-visible-scale 0.7
  --hidden-obs-velocity 1.0
  --save-animation true
) && \
run_save() {
  local out_name="$1"
  shift
  rm -f output/animations/tracking.mp4
  uv run python examples/test_campus.py "$@" "${COMMON_ARGS[@]}" &&
  mv output/animations/tracking.mp4 "${OUT_DIR}/${out_name}.mp4"
} && \
run_save occlusion_cbf_vref_los_a0625_p105_idx17 \
  --baseline occlusion_cbf \
  --vref los \
  --uni-reverse-gate-angle 0.625 \
  --uni-reverse-gate-power 1.05 && \
run_save cbf_qp_idx17 \
  --baseline cbf_qp && \
run_save single_risk_mpc_idx17 \
  --baseline single_risk_mpc && \
run_save control_tree_mpc_idx17 \
  --baseline control_tree_mpc && \
run_save oacp_mpc_idx17 \
  --baseline oacp_mpc && \
run_save oa_mpc_no_fallback_idx17 \
  --baseline oa_mpc \
  --oa-allow-solver-fallback false
```

### 9.4 One-shot animation export for `idx 25`

```bash
mkdir -p output/animations/campus_uni_idx25 && \
OUT_DIR=output/animations/campus_uni_idx25 && \
COMMON_ARGS=(
  --model uni
  --seed 0
  --n-rand 30
  --tf 200
  --idx 25
  --goal-threshold 0.5
  --sensing-range 8.0
  --ped-speed-min 0.3
  --ped-speed-max 0.5
  --ped-radius 0.35
  --occ-visible-scale 0.7
  --hidden-obs-velocity 1.0
  --save-animation true
) && \
run_save() {
  local out_name="$1"
  shift
  rm -f output/animations/tracking.mp4
  uv run python examples/test_campus.py "$@" "${COMMON_ARGS[@]}" &&
  mv output/animations/tracking.mp4 "${OUT_DIR}/${out_name}.mp4"
} && \
run_save occlusion_cbf_vref_los_a0625_p105_idx25 \
  --baseline occlusion_cbf \
  --vref los \
  --uni-reverse-gate-angle 0.625 \
  --uni-reverse-gate-power 1.05 && \
run_save cbf_qp_idx25 \
  --baseline cbf_qp && \
run_save single_risk_mpc_idx25 \
  --baseline single_risk_mpc && \
run_save control_tree_mpc_idx25 \
  --baseline control_tree_mpc && \
run_save oacp_mpc_idx25 \
  --baseline oacp_mpc && \
run_save oa_mpc_no_fallback_idx25 \
  --baseline oa_mpc \
  --oa-allow-solver-fallback false
```


## 10. Optional OA-MPC Variant For Unicycle

If you also want the `OA-MPC` Unicycle `wmax=pi` variant, add this command
after the one-shot block above:

```bash
run_save oa_mpc_wmax_pi_no_fallback_idx25 \
  --baseline oa_mpc \
  --wmax pi \
  --oa-allow-solver-fallback false
```

Replace `idx25` with `idx17` if needed.


## 11. Practical Notes For Collaborators

- Keep `--seed 0` fixed when using the `idx` values in this file.
- Keep `--n-rand 30` and `--tf 200` fixed.
- Keep `--occ-visible-scale 0.7` and `--hidden-obs-velocity 1.0` fixed.
- Keep `--occ-enable-visible-hocbf` disabled.
- Keep the Unicycle occlusion-CBF gate parameters fixed:
  - `angle = 0.625`
  - `power = 1.05`
- For `cbf_qp`, the `--vref los` and Unicycle reverse-gate parameters are not used.
- For `oa_mpc`, the commands above intentionally disable the safe-stop solver fallback.

If you change any of the above, the selected `idx` cases may no longer produce
the same qualitative comparison.
