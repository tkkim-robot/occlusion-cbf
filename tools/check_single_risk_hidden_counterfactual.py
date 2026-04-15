#!/usr/bin/env python3
"""
Counterfactual check for single_risk_mpc hidden-obstacle leakage.

For each requested crowd idx, this script:
1. Rebuilds the forced-emergence scene.
2. Removes all hidden-agent rows while keeping visible occluders unchanged.
3. Compares single_risk_mpc at t=0 between the actual and counterfactual scenes.

It reports two comparisons:
- full_obs: pass the full obstacle array directly to single_risk_mpc
- selected_obs: mimic the actual control_step path via get_nearest_unpassed_obs()

If the outputs match exactly, hidden actual obstacle rows are not affecting the
first-step single_risk decision for that idx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from safe_control.utils import env
from examples.test_crowd import (
    _apply_crowd_dynamic_obstacle_defaults,
    _apply_single_risk_defaults,
    _build_forced_emergence_crowd_scenario,
    _compute_case_seed,
    LocalTrackingControllerDyn_OCC,
)


WAYPOINTS = np.array(
    [
        [4.0, 7.5, 0.0],
        [20.0, 7.5, 0.0],
    ],
    dtype=np.float64,
)


def _build_robot_spec(occ_visible_scale: float) -> dict:
    robot_spec = {
        "model": "DoubleIntegrator2D",
        "v_max": 1.0,
        "a_max": 1.0,
        "v_obs_max": 1.0,
        "radius": 0.25,
        "debug_backup_qp": False,
        "cbf_feas_tol": 5e-3,
        "sensing_range": 10.0,
        "fov_angle": 360.0,
        "backup_cbf": {
            "T_horizon": 0.5,
            "vref_scenario_softmax_kappa": 2.0,
            "rho_T": "auto",
        },
        "show_backup_rollout": True,
        "backup_rollout_every": 1,
        "use_occ": True,
        "dynamic_obs_types": [1],
    }
    _apply_crowd_dynamic_obstacle_defaults(robot_spec)
    robot_spec["v_adv_max_occ"] = 1.0
    dyn_cfg = robot_spec.setdefault("crowd_dyn_obs", {})
    dyn_cfg["occluded_speed_boost_enable"] = True
    dyn_cfg["occluded_speed_boost_vmax"] = float(robot_spec["v_adv_max_occ"])
    dyn_cfg["occluded_speed_boost_fov_only"] = True
    dyn_cfg["occluded_speed_boost_on_hysteresis_steps"] = 2
    dyn_cfg["occluded_speed_boost_off_hysteresis_steps"] = 5
    robot_spec["occ_visible_scale"] = float(occ_visible_scale)
    _apply_single_risk_defaults(robot_spec)
    return robot_spec


def _init_tracking_controller(known_obs, obs_meta, case_seed: int, occ_visible_scale: float):
    tracking_controller = LocalTrackingControllerDyn_OCC(
        WAYPOINTS[0],
        _build_robot_spec(occ_visible_scale),
        controller_type={"pos": "single_risk_mpc"},
        dt=0.05,
        show_animation=False,
        save_animation=False,
        show_mpc_traj=False,
        ax=None,
        fig=None,
        env=env.Env(),
        rand_seed=case_seed,
        tracking_view_ax=None,
        tracking_view_window_size=5.0,
    )
    tracking_controller.obs = np.asarray(known_obs, dtype=float)
    tracking_controller.set_obs_meta(list(obs_meta))
    tracking_controller.set_waypoints(WAYPOINTS)
    if tracking_controller.state_machine == "stop":
        if tracking_controller.robot.has_stopped():
            if tracking_controller.enable_rotation:
                tracking_controller.state_machine = "rotate"
            else:
                tracking_controller.state_machine = "track"
            tracking_controller.goal = tracking_controller.update_goal()
    else:
        tracking_controller.goal = tracking_controller.update_goal()
    return tracking_controller


def _flatten_hidden_indices(forced_event_meta) -> list[int]:
    hidden = []
    for meta in forced_event_meta:
        hidden.extend(int(i) for i in meta.get("hidden_indices", [meta["hidden_index"]]))
    return sorted(set(hidden))


def _preselected_obs(tracking_controller):
    detected_obs = tracking_controller.robot.detect_unknown_obs(tracking_controller.unknown_obs)
    return tracking_controller.get_nearest_unpassed_obs(
        detected_obs,
        obs_num=tracking_controller.num_constraints,
    )


def _risk_region_signature(risk_regions) -> list[tuple]:
    return [
        (
            tuple(np.round(np.asarray(center, dtype=float).reshape(2,), 9)),
            round(float(radius), 9),
        )
        for center, radius in risk_regions
    ]


def _analyze(tracking_controller, obs_list):
    pos_controller = tracking_controller.pos_controller
    robot_state = np.asarray(tracking_controller.robot.X, dtype=float).reshape(-1, 1)
    u_ref = tracking_controller.robot.nominal_input(tracking_controller.goal)
    control_ref = {
        "state_machine": tracking_controller.state_machine,
        "u_ref": u_ref,
        "goal": tracking_controller.goal,
    }

    x0 = pos_controller._state_from_robot_state(robot_state)
    u_ref_clip = pos_controller._clip_input(np.asarray(u_ref, dtype=float).reshape(-1, 1))
    goal_xy = pos_controller._goal_xy(x0, control_ref.get("goal", None))
    v_ref_nom = pos_controller._nominal_speed_reference(x0, u_ref_clip, float(pos_controller.v_ref_default))

    visible_obs_all, occ_scenarios_all, visible_indices = pos_controller._occ_utils._filter_visible_and_build_occ(
        robot_state,
        obs_list,
        return_indices=True,
    )
    visible_obs = pos_controller._nearest_visible_obs(visible_obs_all, x0)
    guidance_xy, guidance_meta = pos_controller._select_guidance_point(x0, control_ref, goal_xy, visible_obs)

    nominal_points = None
    if pos_controller.risk_time_model == "nominal_rollout":
        nominal_points = pos_controller._nominal_rollout_positions(x0, guidance_xy, v_ref_nom)

    occ_scenarios = pos_controller._nearest_occ_scenarios(occ_scenarios_all, x0)
    risk_regions = pos_controller._build_risk_regions(occ_scenarios, v_ref_nom, nominal_points)
    u = pos_controller.solve_control_problem(robot_state, control_ref, obs_list)

    return {
        "u": tuple(np.round(np.asarray(u, dtype=float).reshape(-1), 12)),
        "guidance_xy": tuple(np.round(np.asarray(guidance_xy, dtype=float).reshape(2,), 12)),
        "visible_count": int(len(visible_obs_all)),
        "visible_indices_local": [int(i) for i in np.asarray(visible_indices, dtype=int).reshape(-1).tolist()],
        "occ_count": int(len(occ_scenarios_all)),
        "risk_count": int(len(risk_regions)),
        "risk_regions": _risk_region_signature(risk_regions),
        "guidance_meta": {
            "guidance_source": guidance_meta.get("guidance_source", "unknown"),
            "selected_gap_angle": guidance_meta.get("selected_gap_angle", None),
            "selected_gap_width": guidance_meta.get("selected_gap_width", None),
        },
        "status": str(pos_controller.status),
        "last_profile": pos_controller.last_profile,
    }


def _lookup_global_indices(rows, full_obs):
    out = []
    for row in np.asarray(rows, dtype=float):
        global_idx = None
        for i, obs in enumerate(np.asarray(full_obs, dtype=float)):
            if np.allclose(row, obs, atol=1e-9, rtol=0.0):
                global_idx = int(i)
                break
        out.append(global_idx)
    return out


def _compare_dicts(a: dict, b: dict) -> dict:
    ua = np.asarray(a["u"], dtype=float)
    ub = np.asarray(b["u"], dtype=float)
    return {
        "u_same": bool(np.allclose(ua, ub, atol=1e-12, rtol=0.0)),
        "guidance_same": bool(np.allclose(np.asarray(a["guidance_xy"], dtype=float),
                                          np.asarray(b["guidance_xy"], dtype=float),
                                          atol=1e-12, rtol=0.0)),
        "visible_count_same": bool(a["visible_count"] == b["visible_count"]),
        "occ_count_same": bool(a["occ_count"] == b["occ_count"]),
        "risk_count_same": bool(a["risk_count"] == b["risk_count"]),
        "risk_regions_same": bool(a["risk_regions"] == b["risk_regions"]),
        "u_diff": (ua - ub).tolist(),
    }


def run_idx(idx: int, seed: int, occ_visible_scale: float) -> dict:
    case_seed = _compute_case_seed(seed, idx)
    known_obs, obs_meta, scenario_diag = _build_forced_emergence_crowd_scenario(
        case_seed=case_seed,
        n_rand=50,
        rand_obs=True,
        static_occluders=False,
        waypoints=WAYPOINTS,
        forced_events=6,
        forced_bg_rand=None,
        forced_hidden_speed=1.0,
        forced_occluder_radius_min=0.8,
        forced_occluder_radius_max=1.0,
        forced_validate_occlusion=True,
        forced_require_corridor_conflict=True,
    )

    hidden_indices = _flatten_hidden_indices(scenario_diag.get("forced_event_meta", []))
    mask = np.ones(len(known_obs), dtype=bool)
    mask[hidden_indices] = False
    known_obs_cf = np.asarray(known_obs, dtype=float)[mask]
    obs_meta_cf = [m for i, m in enumerate(obs_meta) if mask[i]]

    actual_tc = _init_tracking_controller(known_obs, obs_meta, case_seed, occ_visible_scale)
    cf_tc = _init_tracking_controller(known_obs_cf, obs_meta_cf, case_seed, occ_visible_scale)

    actual_selected = _preselected_obs(actual_tc)
    cf_selected = _preselected_obs(cf_tc)

    actual_full = _analyze(_init_tracking_controller(known_obs, obs_meta, case_seed, occ_visible_scale), known_obs)
    cf_full = _analyze(_init_tracking_controller(known_obs_cf, obs_meta_cf, case_seed, occ_visible_scale), known_obs_cf)
    actual_sel = _analyze(_init_tracking_controller(known_obs, obs_meta, case_seed, occ_visible_scale), actual_selected)
    cf_sel = _analyze(_init_tracking_controller(known_obs_cf, obs_meta_cf, case_seed, occ_visible_scale), cf_selected)

    # Diagnose initial visible hidden rows under the shared occlusion filter.
    diag_tc = _init_tracking_controller(known_obs, obs_meta, case_seed, occ_visible_scale)
    _, _, visible_idx_full = diag_tc.pos_controller._occ_utils._filter_visible_and_build_occ(
        diag_tc.robot.X,
        known_obs,
        return_indices=True,
    )
    visible_idx_full = [int(i) for i in np.asarray(visible_idx_full, dtype=int).reshape(-1).tolist()]
    hidden_visible = sorted(set(hidden_indices).intersection(set(visible_idx_full)))

    selected_only_in_actual = []
    if actual_selected is not None and cf_selected is not None:
        act_keys = {tuple(np.round(np.asarray(row, dtype=float), 8).tolist()) for row in np.asarray(actual_selected, dtype=float)}
        cf_keys = {tuple(np.round(np.asarray(row, dtype=float), 8).tolist()) for row in np.asarray(cf_selected, dtype=float)}
        only_actual_rows = [
            row for row in np.asarray(actual_selected, dtype=float)
            if tuple(np.round(np.asarray(row, dtype=float), 8).tolist()) not in cf_keys
        ]
        only_actual_globals = _lookup_global_indices(only_actual_rows, known_obs)
        selected_only_in_actual = [
            {"global_idx": gi, "is_hidden": bool(gi in hidden_indices) if gi is not None else None}
            for gi in only_actual_globals
        ]

    return {
        "idx": int(idx),
        "seed": int(seed),
        "case_seed": int(case_seed),
        "n_obs_actual": int(len(known_obs)),
        "n_obs_counterfactual": int(len(known_obs_cf)),
        "hidden_indices": hidden_indices,
        "n_hidden_removed": int(len(hidden_indices)),
        "hidden_visible_full_filter": hidden_visible,
        "n_hidden_visible_full_filter": int(len(hidden_visible)),
        "full_obs": {
            "compare": _compare_dicts(actual_full, cf_full),
            "actual": actual_full,
            "counterfactual": cf_full,
        },
        "selected_obs": {
            "actual_selected_count": int(0 if actual_selected is None else len(np.asarray(actual_selected))),
            "counterfactual_selected_count": int(0 if cf_selected is None else len(np.asarray(cf_selected))),
            "selected_rows_present_only_in_actual": selected_only_in_actual,
            "compare": _compare_dicts(actual_sel, cf_sel),
            "actual": actual_sel,
            "counterfactual": cf_sel,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--idx", type=int, nargs="+", required=True, help="One or more 1-based crowd case indices.")
    parser.add_argument("--occ-visible-scale", type=float, default=0.7)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = [run_idx(idx, args.seed, args.occ_visible_scale) for idx in args.idx]

    out_path = args.out
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
