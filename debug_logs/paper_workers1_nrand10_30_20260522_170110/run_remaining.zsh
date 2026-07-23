#!/usr/bin/env zsh
set -u
ROOT="$1"
MASTER="$ROOT/master.log"
mkdir -p "$ROOT"
echo "[MASTER] resume root=$ROOT" | tee -a "$MASTER"
echo "[MASTER] resume start $(date)" | tee -a "$MASTER"

COMMON_BASE=(
  --scenario crowd2
  --model di
  --seed 42
  --idx-start 1
  --idx-end 100
  --tf 500
  --crowd-mode forced_emergence
  --forced-events 6
  --forced-hidden-speed 1.0
  --forced-occluder-radius-min 0.8
  --forced-occluder-radius-max 1.0
  --forced-validate-occlusion true
  --forced-require-corridor-conflict true
  --workers 1
  --occ-version v2
)

run_case() {
  local nrand="$1"
  local label="$2"
  local baseline="$3"
  shift 3
  local out="$ROOT/nrand${nrand}_${label}"
  local log="$out/run.log"
  mkdir -p "$out"
  echo "[MASTER] START nrand=$nrand label=$label baseline=$baseline $(date)" | tee -a "$MASTER"
  uv run python tools/benchmark_crowd_trials.py "${COMMON_BASE[@]}" --n-rand "$nrand" --baseline "$baseline" "$@" --out-dir "$out" 2>&1 | tee "$log"
  local rc=${pipestatus[1]}
  echo "[MASTER] END nrand=$nrand label=$label baseline=$baseline rc=$rc $(date)" | tee -a "$MASTER"
}

run_ocbf_best() {
  local nrand="$1"
  local kappa="$2"
  run_case "$nrand" occlusion_cbf_best occlusion_cbf \
    --occ-t-horizon 0.5 \
    --occ-enable-visible-hocbf true \
    --occ-max-active-occlusions 2 \
    --occ-selection-mode h_tilde \
    --occ-rho-T auto \
    --occ-qp-failure-fallback-mode state_safe \
    --occ-vref-scenario-weight-mode barrier_predicted_margin \
    --occ-vref-scenario-prediction-dt 0.0 \
    --occ-vref-scenario-softmax-kappa "$kappa" \
    --occ-terminal-slack-weight 10.0 \
    --occ-terminal-slack-max 2.0 \
    --vref los
}

# nrand=10: cbf_qp, oa_mpc, and single_risk_mpc are already complete in this root.
run_case 10 control_tree_mpc control_tree_mpc
run_case 10 oacp_mpc oacp_mpc --oacp-allow-solver-fallback false --oacp-dynamic-occluders true --oacp-visible-reach-mode constant_velocity
run_ocbf_best 10 40.0

run_case 30 cbf_qp cbf_qp
run_case 30 oa_mpc oa_mpc --oa-allow-solver-fallback false --oa-dynamic-occluders true --oa-visible-reach-mode worst_case --oa-use-nominal-tracking-cost false
run_case 30 single_risk_mpc single_risk_mpc
run_case 30 control_tree_mpc control_tree_mpc
run_case 30 oacp_mpc oacp_mpc --oacp-allow-solver-fallback false --oacp-dynamic-occluders true --oacp-visible-reach-mode constant_velocity
run_ocbf_best 30 50.0

python3 - "$ROOT" <<'PY' | tee -a "$MASTER"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
rows=[]
for p in sorted(root.rglob('crowd_trials_*_42_1_100_*.json')):
    try:
        d=json.load(open(p))
    except Exception as e:
        print('skip', p, e)
        continue
    cfg=d.get('config',{}) or {}
    cnt=d.get('counts',{}) or {}
    avg=d.get('averages',{}) or {}
    if cnt.get('total') != 100:
        continue
    rows.append({
        'n_rand': cfg.get('n_rand'),
        'label': p.parent.name.split('_', 1)[1] if '_' in p.parent.name else p.parent.name,
        'baseline': cfg.get('baseline'),
        'success': cnt.get('success'),
        'collision': cnt.get('collision'),
        'infeasible': cnt.get('infeasible'),
        'total': cnt.get('total'),
        'avg_compute_ms': avg.get('avg_compute_time_ms_over_idx', avg.get('avg_solve_time_ms_over_idx')),
        'avg_sim_s': avg.get('avg_total_sim_time_s_over_idx'),
        'avg_intervention': avg.get('avg_control_intervention_l2_sq_over_idx'),
        'avg_terminal_slack_l1': avg.get('avg_terminal_slack_l1_over_idx'),
        'avg_terminal_slack_max': avg.get('avg_terminal_slack_max_over_idx'),
        'avg_terminal_slack_active_ratio': avg.get('avg_terminal_slack_active_ratio_over_idx'),
        'json': str(p),
    })
rows.sort(key=lambda r:(r['n_rand'], r['label']))
out=root/'summary_workers1_nrand10_30.json'
json.dump(rows, open(out,'w'), indent=2)
print('[MASTER] saved', out)
for r in rows:
    print(r)
PY

echo "[MASTER] resume all done $(date)" | tee -a "$MASTER"
