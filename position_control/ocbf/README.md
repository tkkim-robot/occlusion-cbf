# Occlusion-CBF

This directory contains shared utilities, defaults, and JAX kernels used by the
Occlusion-CBF controller. The main controller entry points are still
`position_control/backup_controller.py` and
`position_control/occlusion_cbf_qp.py`.

## Algorithm Flow

At each control step, the crowd scenario runner builds visible-obstacle and
occlusion information from the current robot sensor geometry. Occlusion-CBF then
uses the following pipeline:

1. Build one convex occlusion polygon for each visible occluder. The polygon
   over-approximates the sensor-hidden region induced by the occluder and uses
   outward facet normals for worst-case expansion during the backup rollout.
2. Select active occlusion scenarios using `--occ-selection-mode` and
   `--occ-max-active-occlusions`.
3. Roll out the robot under the occlusion backup policy for horizon
   `--occ-t-horizon`.
4. During the rollout, generate a backup velocity reference `v_ref` from the
   active occlusion facets. With the default `los` mode, the front-facet
   avoidance direction is recomputed from the rollout robot position to the
   occluder center.
5. Blend multiple active scenario directions using
   `--occ-vref-scenario-weight-mode` and
   `--occ-vref-scenario-softmax-kappa`.
6. Build OCBF-QP rows from the rollout occlusion barriers and terminal backup
   set. Visible-obstacle HOCBF rows are enabled by default for crowd2.
7. Solve the QP. If the QP solver fails but the current state is still safe by
   the barrier values, `state_safe` fallback applies the backup policy for one
   control step and retries the QP at the next step.

## Crowd2 Defaults

For crowd2 benchmark runs, these settings are treated as fixed defaults and
normally should not be swept:

| Setting | Default | Meaning |
| --- | --- | --- |
| Occlusion geometry | fixed polygon construction | Over-approximates each occluder-induced hidden region with a convex polygon. |
| `--occ-kappa` | `10.0` | Fixed log-sum-exp smoothing kappa for polygon-facet occlusion barriers. Keep this fixed unless running a barrier-smoothing ablation. |
| `--vref` | `los` | Recompute front-facet escape direction from rollout state. |
| `--occ-selection-mode` | `h_tilde` | Select top-K occlusions by occlusion-barrier risk. |
| `--occ-qp-failure-fallback-mode` | `state_safe` | One-step backup fallback only when current barrier state is safe. |
| `--occ-enable-visible-hocbf` | `true` for crowd2 | Adds visible dynamic obstacle CBF/HOCBF rows. |
| `--occ-rollout-mode` | `common` | Use one blended backup rollout for selected scenarios. |
| `--occ-terminal-mode` | `all` | Add terminal rows for selected occlusion scenarios. |

The defaults live in `position_control/ocbf/defaults.py`. Explicit CLI values
still override these defaults.

## Main Hyperparameters To Tune

These are the parameters collaborators should sweep when searching for the best
OCBF result on crowd2:

| CLI | Typical values | Effect |
| --- | --- | --- |
| `--occ-t-horizon` | `0.5, 0.75, 1.0, ...` | Backup rollout horizon. Longer horizon can help Uni but may increase infeasibility in dense scenes. |
| `--occ-rho-T` | `auto`, `0.0` | Terminal margin. `auto` uses the paper-style stopping margin for DI; Uni currently resolves `auto` to `0.0`. |
| `--occ-max-active-occlusions` | `1, 2, 3, 5` | Top-K active occlusion scenarios used in the OCBF rollout/QP. |
| `--occ-vref-scenario-softmax-kappa` | `0, 10, 20, 40, 60` | Larger values concentrate the blended backup direction on the highest-risk selected scenario. |
| `--occ-vref-scenario-weight-mode` | `barrier_expand`, `barrier_unexpand` | `barrier_expand` scores expanded rollout margins; `barrier_unexpand` scores current-geometry margins along the rollout. |
| `--occ-terminal-slack-weight` | e.g. `1, 5, 10` | Enables terminal-only slack when used with `--occ-terminal-slack-max`. |
| `--occ-terminal-slack-max` | e.g. `0.5, 1, 2` | Maximum allowed terminal-only slack. Omit both slack flags for pure OCBF-QP. |

Avoid sweeping `--occ-obs-hocbf-slack-max` and `--occ-rollout-slack-max` unless
you explicitly want an ablation that relaxes non-terminal safety rows.

## Minimal Crowd2 Benchmark Command

The fixed defaults above do not need to be repeated. This command shows the
main knobs that should be tuned:

```bash
uv run python tools/benchmark_crowd_trials.py \
  --scenario crowd2 \
  --baseline occlusion_cbf \
  --model di \
  --seed 42 \
  --idx-start 1 \
  --idx-end 100 \
  --n-rand 50 \
  --tf 500 \
  --crowd-mode forced_emergence \
  --forced-events 6 \
  --forced-hidden-speed 1.0 \
  --forced-occluder-radius-min 0.8 \
  --forced-occluder-radius-max 1.0 \
  --forced-validate-occlusion true \
  --forced-require-corridor-conflict true \
  --workers 1 \
  --occ-t-horizon 0.5 \
  --occ-rho-T auto \
  --occ-max-active-occlusions 3 \
  --occ-vref-scenario-softmax-kappa 40.0 \
  --occ-vref-scenario-weight-mode barrier_unexpand
```

For unicycle runs, change `--model di` to `--model uni`. The benchmark tool
uses forward-only unicycle behavior by default because it was empirically more
stable in crowd2 and avoids reverse-motion deadlocks around slow occluders.
Reverse motion can still be tested with `--uni-allow-reverse true`.

## Output Metrics

`tools/benchmark_crowd_trials.py` writes JSON results under `debug_logs/` or the
specified `--out-dir`. The main metrics are:

- `success`: reached the goal without collision or infeasibility.
- `collision`: executed trajectory collided with an obstacle.
- `infeasible`: controller or planner failed before reaching the goal.
- `avg_compute_ms`: average per-step controller computation time.
- `intervention`: average amount of safety-filter intervention.

For paper tables, use `--workers 1` for final timing. Larger worker counts are
useful for parameter search but can distort timing-related metrics.
