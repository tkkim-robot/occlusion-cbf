#!/usr/bin/env bash
set -u
ROOT="$1"
run_case() {
  label="$1"; T="$2"; topk="$3"; kappa="$4"
  OUT="$ROOT/$label"
  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] RUN $label T=$T topK=$topk kappa=$kappa" | tee -a "$ROOT/master.log"
  uv run python tools/benchmark_crowd_trials.py \
    --scenario crowd2 \
    --baseline occlusion_cbf \
    --model uni \
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
    --workers 12 \
    --occ-version v2 \
    --occ-t-horizon "$T" \
    --occ-rho-T auto \
    --occ-max-active-occlusions "$topk" \
    --occ-selection-mode h_tilde \
    --occ-vref-scenario-softmax-kappa "$kappa" \
    --occ-vref-scenario-weight-mode barrier_predicted_margin \
    --occ-vref-scenario-prediction-dt 0.0 \
    --occ-qp-failure-fallback-mode state_safe \
    --vref los \
    --uni-vref-tracking-mode gated \
    --uni-forward-only true \
    --out-dir "$OUT" \
    2>&1 | tee "$OUT/$label.log"
  python - "$OUT" "$label" "$ROOT" <<'PY'
import json, pathlib, sys
out=pathlib.Path(sys.argv[1]); label=sys.argv[2]; root=pathlib.Path(sys.argv[3])
js=sorted(out.glob('crowd_trials_occlusion_cbf_42_1_100_*.json'))
if not js:
    line=f"RESULT {label}: FAILED_NO_JSON\n"
else:
    d=json.loads(js[-1].read_text())
    c=d.get('counts',{}); av=d.get('averages',{}); idx=d.get('idx_lists',{}).get('success',[])
    line=f"RESULT {label}: S/C/I={c.get('success')}/{c.get('collision')}/{c.get('infeasible')} compute={av.get('avg_compute_time_ms_over_idx')} success_idx={idx}\n"
print(line.strip())
with (root/'results.log').open('a') as f: f.write(line)
PY
}
run_case T0p75_fwd_top1_k10 0.75 1 10
run_case T0p75_fwd_top1_k40 0.75 1 40
run_case T0p9_fwd_top1_k10 0.9 1 10
run_case T0p9_fwd_top1_k20 0.9 1 20
run_case T0p9_fwd_top1_k40 0.9 1 40
run_case T1p0_fwd_top1_k10 1.0 1 10
run_case T1p0_fwd_top1_k40 1.0 1 40
run_case T1p0_fwd_top1_k60 1.0 1 60
run_case T1p0_fwd_top2_k20 1.0 2 20
run_case T1p0_fwd_top2_k40 1.0 2 40
run_case T1p1_fwd_top1_k20 1.1 1 20
python - "$ROOT" <<'PY'
import json, pathlib, csv, sys
root=pathlib.Path(sys.argv[1]); rows=[]
for p in root.glob('*/crowd_trials_occlusion_cbf_42_1_100_*.json'):
    d=json.loads(p.read_text()); c=d.get('counts',{}); av=d.get('averages',{}); bc=d.get('config',{}).get('backup_cbf_overrides',{})
    rows.append({
        'label':p.parent.name,
        'success':c.get('success'), 'collision':c.get('collision'), 'infeasible':c.get('infeasible'), 'total':c.get('total'),
        'T':bc.get('T_horizon'), 'topK':bc.get('max_active_occlusions'), 'kappa':bc.get('vref_scenario_softmax_kappa'),
        'avg_compute_ms':av.get('avg_compute_time_ms_over_idx'),
        'success_idx':d.get('idx_lists',{}).get('success',[]), 'json':str(p)
    })
if rows:
    rows.sort(key=lambda r:(-int(r['success'] or 0), int(r['collision'] or 999), float(r['T'] or 999)))
    out=root/'focus_summary.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('SUMMARY', out)
    for r in rows:
        print(f"{r['label']}: T={r['T']} topK={r['topK']} kappa={r['kappa']} S/C/I={r['success']}/{r['collision']}/{r['infeasible']} compute={r['avg_compute_ms']} idx={r['success_idx']}")
PY
