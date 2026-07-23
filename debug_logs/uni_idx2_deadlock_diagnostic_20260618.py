
import csv
import json
import math
from pathlib import Path
import numpy as np
from examples import test_crowd2 as crowd2
from examples import test_crowd as crowd1
from safe_control.utils import env

OUT=Path('debug_logs/uni_idx2_deadlock_diagnostic_20260618')
OUT.mkdir(parents=True, exist_ok=True)

def route_progress(pt, path):
    pt=np.asarray(pt,float).reshape(2)
    best_d=1e18; best_s=0.0; cum=0.0
    for i in range(len(path)-1):
        a=np.asarray(path[i],float); b=np.asarray(path[i+1],float); ab=b-a
        L=float(np.linalg.norm(ab))
        if L <= 1e-12: continue
        t=float(np.clip(np.dot(pt-a,ab)/(L*L),0,1))
        q=a+t*ab
        d=float(np.linalg.norm(pt-q))
        if d<best_d:
            best_d=d; best_s=cum+t*L
        cum += L
    return best_s, best_d

# Build exact crowd2 idx=2 scenario first.
case_seed = crowd1._compute_case_seed(42, 2)
known_obs, obs_meta, scenario_diag = crowd2._build_route_forced_emergence_scenario(
    case_seed=case_seed,
    n_rand=50,
    rand_obs=True,
    static_occluders=False,
    forced_events=6,
    forced_bg_rand=None,
    forced_hidden_speed=1.0,
    forced_occluder_radius_min=0.8,
    forced_occluder_radius_max=1.0,
    forced_validate_occlusion=True,
    forced_require_corridor_conflict=True,
    rand_obs_setting='v2',
)
backup_cbf_overrides={
    'T_horizon':0.5,
    'rho_T':'auto',
    'qp_failure_fallback_mode':'state_safe',
    'vref_scenario_softmax_kappa':40.0,
    'vref_scenario_weight_mode':'barrier_predicted_margin',
    'vref_scenario_prediction_dt':0.0,
    'max_active_occlusions':3,
    'occ_selection_mode':'h_tilde',
    'v_min_cmd_rev_occ_uni':0.2,
    'reverse_speed_gate_angle_occ_uni':1.2,
    'reverse_speed_gate_power_occ_uni':1.0,
}
runtime=crowd1._prepare_crowd_runtime(
    controller_type={'pos':'occlusion_cbf_qp'},
    model_key='uni', tf=500, seed=42, case_idx=2, rand_obs=True, n_rand=50,
    occ_version='v2', occ_enable_visible_hocbf=False,
    crowd_mode='forced_emergence', forced_events=6, forced_hidden_speed=1.0,
    forced_occluder_radius_min=0.8, forced_occluder_radius_max=1.0,
    forced_validate_occlusion=True, forced_require_corridor_conflict=True,
    rand_obs_setting='v2', static_occluders=False,
    backup_cbf_overrides=backup_cbf_overrides,
    waypoints_override=crowd2.ROUTE_WAYPOINTS,
    env_width_override=crowd2.ENV_WIDTH,
    env_height_override=crowd2.ENV_HEIGHT,
    known_obs_override=known_obs,
    obs_meta_override=obs_meta,
    scenario_diag_override=scenario_diag,
    scenario_name='Crowd2',
)
ctrl=crowd1.LocalTrackingControllerDyn_OCC(
    runtime['waypoints'][0], runtime['robot_spec'], controller_type=runtime['controller_type'],
    dt=runtime['dt'], show_animation=False, save_animation=False, show_mpc_traj=False,
    ax=None, fig=None, env=env.Env(), rand_seed=runtime['case_seed'],
)
ctrl.obs=runtime['known_obs'].astype(float)
ctrl.set_obs_meta(runtime['obs_meta'])
ctrl.set_waypoints(runtime['waypoints'])
path=np.asarray(runtime['waypoints'][:,:2],float)
robot_r=float(runtime['robot_spec'].get('radius',0.25))
rows=[]
progress_hist=[]
ret=0
for step in range(int(math.ceil(500/runtime['dt']))):
    t=step*runtime['dt']
    x_pre=np.asarray(ctrl.robot.X,dtype=float).reshape(-1)
    prog, path_d=route_progress(x_pre[:2], path)
    progress_hist.append((t,prog))
    old_t=t-5.0
    old_prog=progress_hist[0][1]
    for tt,pp in progress_hist:
        if tt>=old_t:
            old_prog=pp; break
    progress_5s=prog-old_prog
    obs=np.asarray(ctrl.obs,dtype=float)
    if obs.size:
        dcent=np.linalg.norm(obs[:,:2]-x_pre[:2], axis=1)
        clearance=dcent-obs[:,2]-robot_r
        near_i=int(np.argmin(clearance))
        near=obs[near_i]
        near_clear=float(clearance[near_i])
        near_dist=float(dcent[near_i])
        near_r=float(near[2]); near_speed=float(np.linalg.norm(near[3:5]))
    else:
        near_i=-1; near_clear=near_dist=near_r=near_speed=float('nan')
    ret=ctrl.control_step()
    pc=getattr(ctrl,'pos_controller',None)
    u=getattr(pc,'last_u',None)
    uref=getattr(pc,'last_u_ref',None)
    prof=getattr(pc,'last_profile',{}) if pc is not None else {}
    u=np.asarray(u,dtype=float).reshape(-1) if u is not None else np.array([np.nan,np.nan])
    uref=np.asarray(uref,dtype=float).reshape(-1) if uref is not None else np.array([np.nan,np.nan])
    x_post=np.asarray(ctrl.robot.X,dtype=float).reshape(-1)
    status=getattr(pc,'status',None)
    intervention=getattr(pc,'last_intervention',None)
    selected=prof.get('occ_selected_indices',[]) if isinstance(prof,dict) else []
    weights=[]
    if isinstance(prof,dict):
        sd=prof.get('occ_vref_scenario_debug',[]) or []
        try:
            weights=[float(s.get('softmax_weight',np.nan)) for s in sd]
        except Exception:
            weights=[]
    rows.append({
        'step':step,'t':t,
        'x':float(x_pre[0]),'y':float(x_pre[1]),'theta':float(x_pre[2]) if len(x_pre)>2 else float('nan'),
        'progress':prog,'path_dist':path_d,'progress_5s':progress_5s,
        'u_v':float(u[0]) if len(u)>0 else float('nan'),'u_w':float(u[1]) if len(u)>1 else float('nan'),
        'uref_v':float(uref[0]) if len(uref)>0 else float('nan'),'uref_w':float(uref[1]) if len(uref)>1 else float('nan'),
        'status':status,'intervention':intervention,
        'qp_raw':getattr(pc,'last_qp_status_raw',None),
        'fallback_allowed':getattr(pc,'last_fallback_allowed',None),
        'fallback_min_h':getattr(pc,'last_fallback_min_h',None),
        'fallback_cmd_feasible':getattr(pc,'last_fallback_cmd_feasible',None),
        'fallback_cmd_max_violation':getattr(pc,'last_fallback_cmd_max_violation',None),
        'nearest_i':near_i,'nearest_clearance':near_clear,'nearest_dist':near_dist,'nearest_r':near_r,'nearest_speed':near_speed,
        'occ_selected':json.dumps(selected),'occ_max_weight':prof.get('occ_vref_max_softmax_weight') if isinstance(prof,dict) else None,
        'occ_pred_margin':prof.get('occ_vref_avg_predicted_margin') if isinstance(prof,dict) else None,
        'num_constraints':getattr(pc,'last_num_constraints',None),
        'ret':ret,'terminal_event':getattr(ctrl,'last_terminal_event',None),
        'x_post':float(x_post[0]),'y_post':float(x_post[1]),
    })
    if ret in (-1,-2):
        break

csv_path=OUT/'idx2_step_log.csv'
with csv_path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)
summary={
    'ret':ret,
    'terminal_event':getattr(ctrl,'last_terminal_event',None),
    'steps':len(rows),
    'sim_time':len(rows)*runtime['dt'],
    'final_state':np.asarray(ctrl.robot.X,dtype=float).reshape(-1).tolist(),
    'final_progress':route_progress(np.asarray(ctrl.robot.X,dtype=float).reshape(-1)[:2],path)[0],
    'final_goal_dist':float(np.linalg.norm(np.asarray(ctrl.robot.X,dtype=float).reshape(-1)[:2]-path[-1])),
    'scenario_diag':runtime['scenario_diag'],
    'csv':str(csv_path),
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:summary[k] for k in ['ret','terminal_event','steps','sim_time','final_state','final_progress','final_goal_dist','csv']},indent=2))
