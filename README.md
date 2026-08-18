# OcclusionCBF: Safe Robot Navigation with Occluded Dynamic Obstacles

This repository implements **OcclusionCBF**, a safety filter for robot navigation with potentially hidden dynamic obstacles. OcclusionCBF propagates occluded regions into collision-inflated, time-indexed reachable occupancy and certifies a prescribed backup rollout against every occupancy component and a verified time-varying terminal set. Differentiating these prediction-indexed margins through the backup flow yields constraints affine in the current input for minimally invasive quadratic-program filtering. Under the conditions stated in the paper, the filter is recursively feasible on its certified recoverable set and avoids every hidden-obstacle motion covered by the predictor. Please see our [project page](https://www.taekyung.me/occlusion-cbf) for more details.

<div align="center">
  <img src="https://github.com/user-attachments/assets/083a1d86-ab8d-4e66-93a2-674b9eb9d568" alt="Occlusion-agnostic CBF-QP reacts after detection" height="300px" />
  <img src="https://github.com/user-attachments/assets/8537ae49-91c4-478e-aafa-c2fc742e17d9" alt="OcclusionCBF intervenes before detection" height="300px" />
</div>

<div align="center">

[[Project Page]](https://www.taekyung.me/occlusion-cbf)
[[Paper]](REPLACE_WITH_PAPER_URL)
[[Video]](REPLACE_WITH_VIDEO_URL)
[[Web Demo]](https://occlusion-cbf.taekyung.me/)
[[Research Group]](https://dasc-lab.github.io/)

</div>

## Features

- **Planner-agnostic safety filtering** that minimally modifies commands from a planner, tracker, or learned policy through the OCBF-QP.
- **Reachable-occupancy prediction** that propagates potentially hidden dynamic obstacles from current occluded regions over the backup horizon.
- **Backup-rollout certification** against every time-indexed occupancy component and a verified terminal set while accounting for robot dynamics and input constraints.
- **Multiple robot models**, including double-integrator, dynamic-unicycle, and unicycle systems.
- **Five comparison controllers**: CBF-QP, OA-MPC, Control-Tree MPC, Single-Risk MPC, and OACP.
- **Reproducible evaluation tools** for deterministic scenarios, paired randomized benchmarks, visualization, regression tests, and Optuna-based tuning.

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
- `position_control/`: OcclusionCBF and MPC comparison controllers.
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
```

Run the corresponding unicycle setup with 30 moving obstacles:

```bash
uv run python -m examples.run_scenario --scenario crowd -- \
  --model uni \
  --baseline occlusion_cbf \
  --n-rand 30 \
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
```


## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2026occlusioncbf,
  author  = {Kim, Taekyung and Park, Hun Kuk and Wada, Renya and Atanasov, Nikolay and Koga, Shumon and Panagou, Dimitra},
  title   = {OcclusionCBF: Safe Robot Navigation with Occluded Dynamic Obstacles},
  booktitle = {arXiv preprint },
  shorttitle = {OcclusionCBF},
  year    = {2026}
}
```
