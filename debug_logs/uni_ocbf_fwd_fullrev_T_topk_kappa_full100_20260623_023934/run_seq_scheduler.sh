#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

printf 'seq_scheduler_started %s\n' "$(date)" >> "$ROOT_DIR/scheduler_progress.log"
printf 'label,mode,T,topK,kappa,out_dir,status\n' > "$ROOT_DIR/seq_scheduler_manifest.csv"

WORKERS=12

for T in 0.5 0.75 1.0 1.25; do
  for MODE in forward fullrev; do
    for TOPK in 1 2 3; do
      if [ "$TOPK" = "1" ]; then
        KAPPAS=(20)
      else
        KAPPAS=(0 10 20 40 60)
      fi
      for KAPPA in "${KAPPAS[@]}"; do
        TLAB=${T//./p}
        KLAB=${KAPPA//./p}
        LABEL="T${TLAB}_${MODE}_top${TOPK}_k${KLAB}"
        OUT="$ROOT_DIR/$LABEL"
        mkdir -p "$OUT"

        if find "$OUT" -name 'crowd_trials_occlusion_cbf_42_1_100_*.json' | grep -q .; then
          printf '%s %s skip_existing\n' "$(date '+%F %T')" "$LABEL" >> "$ROOT_DIR/scheduler_progress.log"
          printf '%s,%s,%s,%s,%s,%s,%s\n' "$LABEL" "$MODE" "$T" "$TOPK" "$KAPPA" "$OUT" "skip_existing" >> "$ROOT_DIR/seq_scheduler_manifest.csv"
          continue
        fi

        printf '%s %s start\n' "$(date '+%F %T')" "$LABEL" >> "$ROOT_DIR/scheduler_progress.log"
        printf '%s,%s,%s,%s,%s,%s,%s\n' "$LABEL" "$MODE" "$T" "$TOPK" "$KAPPA" "$OUT" "started" >> "$ROOT_DIR/seq_scheduler_manifest.csv"

        COMMON_ARGS=(
          --scenario crowd2
          --baseline occlusion_cbf
          --model uni
          --seed 42
          --idx-start 1
          --idx-end 100
          --n-rand 50
          --tf 500
          --crowd-mode forced_emergence
          --forced-events 6
          --forced-hidden-speed 1.0
          --forced-occluder-radius-min 0.8
          --forced-occluder-radius-max 1.0
          --forced-validate-occlusion true
          --forced-require-corridor-conflict true
          --workers "$WORKERS"
          --occ-version v2
          --occ-enable-visible-hocbf false
          --occ-t-horizon "$T"
          --occ-rho-T auto
          --occ-terminal-slack-max 0.0
          --occ-obs-hocbf-slack-max 0.0
          --occ-rollout-slack-max 0.0
          --occ-max-active-occlusions "$TOPK"
          --occ-selection-mode h_tilde
          --occ-vref-scenario-softmax-kappa "$KAPPA"
          --occ-vref-scenario-weight-mode barrier_predicted_margin
          --occ-vref-scenario-prediction-dt 0.0
          --occ-qp-failure-fallback-mode state_safe
          --uni-vref-tracking-mode gated
          --vref los
          --out-dir "$OUT"
        )

        if [ "$MODE" = "forward" ]; then
          uv run python tools/benchmark_crowd_trials.py "${COMMON_ARGS[@]}" --uni-forward-only true > "$OUT/run.log" 2>&1
        else
          uv run python tools/benchmark_crowd_trials.py "${COMMON_ARGS[@]}" --uni-allow-reverse true --uni-v-min -1.0 > "$OUT/run.log" 2>&1
        fi

        RC=$?
        if [ "$RC" -eq 0 ]; then
          STATUS="done"
        else
          STATUS="failed_${RC}"
        fi
        printf '%s %s %s\n' "$(date '+%F %T')" "$LABEL" "$STATUS" >> "$ROOT_DIR/scheduler_progress.log"
        printf '%s,%s,%s,%s,%s,%s,%s\n' "$LABEL" "$MODE" "$T" "$TOPK" "$KAPPA" "$OUT" "$STATUS" >> "$ROOT_DIR/seq_scheduler_manifest.csv"
      done
    done
  done
done

printf 'seq_scheduler_completed %s\n' "$(date)" >> "$ROOT_DIR/scheduler_progress.log"
