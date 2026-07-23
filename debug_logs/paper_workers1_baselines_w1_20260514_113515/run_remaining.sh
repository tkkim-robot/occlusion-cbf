#!/usr/bin/env zsh
set -u

ROOT="debug_logs/paper_workers1_baselines_w1_20260514_113515"
OCBF_PID=39537
MASTER="$ROOT/master.log"

mkdir -p "$ROOT"
echo "[MASTER] root=$ROOT" | tee -a "$MASTER"
echo "[MASTER] waiting for OCBF pid=$OCBF_PID before running remaining baselines" | tee -a "$MASTER"
while kill -0 "$OCBF_PID" 2>/dev/null; do
  date "+[MASTER] %Y-%m-%d %H:%M:%S OCBF still running" | tee -a "$MASTER"
  sleep 120
done

echo "[MASTER] OCBF done; starting workers=1 baseline sweeps" | tee -a "$MASTER"

COMMON=(
  --scenario crowd2
  --model di
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
  --workers 1
  --occ-version v2
)

run_baseline() {
  local name="$1"
  shift
  local log="$ROOT/di_${name}.log"
  echo "[MASTER] START $name $(date)" | tee -a "$MASTER"
  uv run python tools/benchmark_crowd_trials.py "${COMMON[@]}" --baseline "$name" "$@" --out-dir "$ROOT" 2>&1 | tee "$log"
  local rc=${pipestatus[1]}
  echo "[MASTER] END $name rc=$rc $(date)" | tee -a "$MASTER"
  return $rc
}

run_baseline cbf_qp
run_baseline oa_mpc --oa-allow-solver-fallback false --oa-dynamic-occluders true --oa-visible-reach-mode worst_case --oa-use-nominal-tracking-cost false
run_baseline single_risk_mpc
run_baseline control_tree_mpc
run_baseline oacp_mpc --oacp-allow-solver-fallback false --oacp-dynamic-occluders true --oacp-visible-reach-mode constant_velocity

python3 - <<'PY' >> "$MASTER" 2>&1
import json
import pathlib

root = pathlib.Path("debug_logs/paper_workers1_baselines_w1_20260514_113515")
rows = []
for p in sorted(root.glob("crowd_trials_*_42_1_100_*.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        print("skip", p, e)
        continue
    cfg = d.get("config", {})
    cnt = d.get("counts", {})
    avg = d.get("averages", {})
    rows.append(
        {
            "label": cfg.get("label"),
            "baseline": cfg.get("baseline"),
            "success": cnt.get("success"),
            "collision": cnt.get("collision"),
            "infeasible": cnt.get("infeasible"),
            "total": cnt.get("total"),
            "avg_compute_ms": avg.get("avg_compute_time_ms_over_idx", avg.get("avg_solve_time_ms_over_idx")),
            "avg_sim_s": avg.get("avg_total_sim_time_s_over_idx"),
            "avg_intervention": avg.get("avg_control_intervention_l2_sq_over_idx"),
            "json": str(p),
        }
    )

out = root / "summary_workers1_baselines.json"
json.dump(rows, open(out, "w"), indent=2)
print("[MASTER] saved", out)
for r in rows:
    print(r)
PY

echo "[MASTER] ALL DONE $(date)" | tee -a "$MASTER"
