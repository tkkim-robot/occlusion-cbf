"""
Crowd scenario test migrated from dynamic_env/main.py::single_agent_main.

Run:
    uv run python examples/test_crowd.py --model di
"""

import argparse
from collections import deque

import numpy as np

try:
    from examples._baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        CROWD_PLANNER_LABELS,
        resolve_baseline_alias,
    )
    from examples._runtime import ensure_repo_root, install_position_controller_shims, load_local_occ_controller
except ImportError:
    from _baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        CROWD_PLANNER_LABELS,
        resolve_baseline_alias,
    )
    from _runtime import ensure_repo_root, install_position_controller_shims, load_local_occ_controller

ensure_repo_root()
install_position_controller_shims()
LocalTrackingControllerDyn_OCC = load_local_occ_controller("crowd")
from safe_control.utils import env, plotting


def _str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _apply_single_risk_defaults(robot_spec):
    """Apply finalized single_risk_mpc baseline defaults."""
    robot_spec["occlusion_types"] = [1]
    sr_cfg = robot_spec.setdefault("single_risk_mpc", {})
    sr_cfg.setdefault("dt_plan", 0.25)
    sr_cfg.setdefault("Th", 6.0)
    sr_cfg.setdefault("N", 24)
    sr_cfg.setdefault("risk_regions_per_tangent", 2)
    sr_cfg.setdefault("wguide", 3.5)
    sr_cfg.setdefault("wgoal", 3.5)
    sr_cfg.setdefault("wvel", 5.0)
    sr_cfg.setdefault("wacc", 1.8)
    sr_cfg.setdefault("wtrack", 0.5)
    sr_cfg.setdefault("n_split", 8)
    sr_cfg.setdefault("lambda_w", 1.0)
    sr_cfg.setdefault("margin_obs", 0.05)
    sr_cfg.setdefault("margin_risk", 0.05)
    sr_cfg.setdefault("use_guidance_point", True)
    sr_cfg.setdefault("guidance_mode", "gap")
    sr_cfg.setdefault("guidance_lookahead", 2.5)
    sr_cfg.setdefault("guidance_side_clearance", 0.5)
    sr_cfg.setdefault("guidance_forward_fov_deg", 180.0)
    sr_cfg.setdefault("guidance_obs_max_dist", 4.5)
    sr_cfg.setdefault("tau_guidance", 0.75)
    sr_cfg.setdefault("risk_time_model", "distance_over_vref")


def _apply_control_tree_defaults(robot_spec):
    """Apply literature-like control_tree_mpc baseline defaults."""
    robot_spec["occlusion_types"] = [1]
    ct_cfg = robot_spec.setdefault("control_tree_mpc", {})
    ct_cfg.setdefault("dt_plan", 0.25)
    ct_cfg.setdefault("Th", 3.0)
    ct_cfg.setdefault("N", 12)
    ct_cfg.setdefault("gap_lookahead", 4.0)
    ct_cfg.setdefault("min_gap_width", 0.25)
    ct_cfg.setdefault("cluster_merge_distance", 0.8)
    ct_cfg.setdefault("forward_fov_deg_for_guidance", 180.0)
    ct_cfg.setdefault("forward_only", True)
    ct_cfg.setdefault("v_plan_min", 0.05)
    ct_cfg.setdefault("n_split", 2)
    ct_cfg.setdefault("n_occ_hypotheses", 2)
    ct_cfg.setdefault("risk_regions_per_tangent", 1)
    ct_cfg.setdefault("drisk", 0.7)
    ct_cfg.setdefault("risk_sigma", 1e-4)
    ct_cfg.setdefault("rrisk_max", 1.0)
    ct_cfg.setdefault("min_v_for_risk", 0.3)
    ct_cfg.setdefault("risk_time_model", "distance_over_vref")
    ct_cfg.setdefault("max_branch_risk_regions", 4)
    ct_cfg.setdefault("belief_prob_scale", 0.22)
    ct_cfg.setdefault("belief_prob_min", 0.02)
    ct_cfg.setdefault("belief_prob_max", 0.35)
    ct_cfg.setdefault("belief_dist_scale", 4.0)
    ct_cfg.setdefault("belief_path_scale", 1.0)
    ct_cfg.setdefault("belief_align_floor", 0.25)
    ct_cfg.setdefault("belief_align_power", 1.5)
    ct_cfg.setdefault("hypothesis_score_min", 0.10)
    ct_cfg.setdefault("wgoal", 4.0)
    ct_cfg.setdefault("wvel", 12.0)
    ct_cfg.setdefault("wacc", 1.2)
    ct_cfg.setdefault("wtrack_shared", 0.4)
    ct_cfg.setdefault("wtrack_tail", 0.12)
    ct_cfg.setdefault("branch_width_weight", 0.75)
    ct_cfg.setdefault("branch_clearance_weight", 1.3)
    ct_cfg.setdefault("lambda_w", 1.0)
    ct_cfg.setdefault("margin_obs", 0.05)
    ct_cfg.setdefault("margin_risk", 0.05)
    ct_cfg.setdefault("solver_backend", "opti")
    ct_cfg.setdefault("persistent_fallback_opti", False)
    ct_cfg.setdefault("warm_start_dual", False)
    ct_cfg.setdefault("max_iter", 150)
    ct_cfg.setdefault("solver_tol", 1e-4)
    ct_cfg.setdefault("solver_acceptable_tol", 1e-2)
    ct_cfg.setdefault("solver_acceptable_iter", 8)
    ct_cfg.setdefault("goal_handover_radius", 1.5)
    ct_cfg.setdefault("nominal_k_heading", 1.4)
    ct_cfg.setdefault("branch_zero_prob_reg", 1e-3)


def _apply_oacp_defaults(robot_spec):
    """Apply adapted occlusion-aware contingency planner defaults."""
    oacp_cfg = robot_spec.setdefault("oacp_mpc", {})
    oacp_cfg.setdefault("dt_plan", 0.20)
    oacp_cfg.setdefault("Th", 2.4)
    oacp_cfg.setdefault("N", 12)
    oacp_cfg.setdefault("n_shared", 3)
    oacp_cfg.setdefault("risk_explore_scale", 0.55)
    oacp_cfg.setdefault("risk_fallback_scale", 1.10)
    oacp_cfg.setdefault("explore_speed_scale", 1.00)
    oacp_cfg.setdefault("fallback_speed_scale", 0.65)
    oacp_cfg.setdefault("allow_solver_fallback", False)
    oacp_cfg.setdefault("dynamic_occluders", True)
    oacp_cfg.setdefault("visible_reach_mode", "constant_velocity")
    oacp_cfg.setdefault("forward_only", True)
    oacp_cfg.setdefault("du_nonnegative_speed", True)
    oacp_cfg.setdefault("max_visible_obs", 30)
    oacp_cfg.setdefault("max_occ_scenarios", 30)
    robot_spec["occlusion_types"] = [0, 1] if bool(oacp_cfg.get("dynamic_occluders", True)) else [0]


def _apply_crowd_dynamic_obstacle_defaults(robot_spec):
    """
    Crowd-specific dynamic-obstacle behavior model.
    Keeps base random motion, then adds:
    1) near-robot visible pedestrian sidestep bias
    2) occluded-agent emergence bias toward robot-forward corridor
    """
    dyn_cfg = robot_spec.setdefault("crowd_dyn_obs", {})

    # Base random walker noise.
    dyn_cfg.setdefault("heading_jitter_std", 0.01)
    dyn_cfg.setdefault("large_turn_prob", 0.05)
    dyn_cfg.setdefault("large_turn_std", 0.05)

    # Visible & near robot: avoid charging toward robot.
    dyn_cfg.setdefault("ped_aware_enable", True)
    dyn_cfg.setdefault("ped_aware_visible_only", True)

    dyn_cfg.setdefault("ped_aware_radius", 1.5)
    dyn_cfg.setdefault("ped_aware_approach_cos", 0.25)
    dyn_cfg.setdefault("ped_aware_turn_gain", 0.2)
    dyn_cfg.setdefault("ped_aware_side_weight", 0.6)
    dyn_cfg.setdefault("ped_aware_away_weight", 0.15)
    dyn_cfg.setdefault("ped_aware_noise_std", 0.03)
    dyn_cfg.setdefault("ped_aware_side_flip_prob", 0.02)

    # Occlusion-emergence: increase natural "hidden then appear" events.
    dyn_cfg.setdefault("occlusion_emergence_enable", True)
    dyn_cfg.setdefault("occlusion_emergence_occluded_only", True)
    dyn_cfg.setdefault("occlusion_emergence_zone_radius", 5.5)
    dyn_cfg.setdefault("occlusion_emergence_forward_only", True)
    dyn_cfg.setdefault("occlusion_emergence_forward_min_proj", -0.2)
    dyn_cfg.setdefault("occlusion_emergence_turn_prob", 0.08)
    dyn_cfg.setdefault("occlusion_emergence_turn_gain", 0.28)
    dyn_cfg.setdefault("occlusion_emergence_target_ahead", 2.8)
    dyn_cfg.setdefault("occlusion_emergence_lateral_jitter", 0.9)
    dyn_cfg.setdefault("occlusion_emergence_noise_std", 0.04)
    dyn_cfg.setdefault("visibility_refresh_steps", 1)

    # Optional speed-up for currently occluded agents.
    # When enabled, non-occluder dynamic obstacles inside the robot FOV/range
    # use this speed magnitude until they become visible again.
    dyn_cfg.setdefault("occluded_speed_boost_enable", False)
    dyn_cfg.setdefault("occluded_speed_boost_vmax", 1.0)
    dyn_cfg.setdefault("occluded_speed_boost_fov_only", True)
    dyn_cfg.setdefault("occluded_speed_boost_on_hysteresis_steps", 2)
    dyn_cfg.setdefault("occluded_speed_boost_off_hysteresis_steps", 5)


def _compute_case_seed(seed, case_idx):
    if case_idx is not None:
        if int(case_idx) < 1:
            raise ValueError("case_idx must be >= 1 (1-based).")
        rng_case = np.random.default_rng(int(seed))
        case_seed = int(seed)
        for _ in range(int(case_idx)):
            case_seed = int(rng_case.integers(0, 2**31 - 1))
        return case_seed
    return int(seed)


def _safe_normalize(vec):
    arr = np.asarray(vec, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 1e-9:
        return None
    return arr / nrm


def _point_segment_distance(pt, seg_a, seg_b):
    pt = np.asarray(pt, dtype=float).reshape(2,)
    seg_a = np.asarray(seg_a, dtype=float).reshape(2,)
    seg_b = np.asarray(seg_b, dtype=float).reshape(2,)
    ab = seg_b - seg_a
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return float(np.linalg.norm(pt - seg_a))
    t = float(np.dot(pt - seg_a, ab) / ab2)
    t = float(np.clip(t, 0.0, 1.0))
    proj = seg_a + t * ab
    return float(np.linalg.norm(pt - proj))


def _disc_occludes_target(observer_xy, target_xy, occ_center_xy, occ_radius, target_radius=0.0):
    observer_xy = np.asarray(observer_xy, dtype=float).reshape(2,)
    target_xy = np.asarray(target_xy, dtype=float).reshape(2,)
    occ_center_xy = np.asarray(occ_center_xy, dtype=float).reshape(2,)

    seg = target_xy - observer_xy
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= 1e-9:
        return False
    seg_dir = seg / seg_len
    proj = float(np.dot(occ_center_xy - observer_xy, seg_dir))
    if proj <= 0.0 or proj >= seg_len:
        return False
    closest = observer_xy + proj * seg_dir
    clearance = float(np.linalg.norm(occ_center_xy - closest))
    return clearance <= float(occ_radius) + 0.2 * float(target_radius)


def _nominal_ego_position(start_xy, goal_xy, speed, t):
    start_xy = np.asarray(start_xy, dtype=float).reshape(2,)
    goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
    path = goal_xy - start_xy
    path_len = float(np.linalg.norm(path))
    if path_len <= 1e-9:
        return start_xy.copy()
    d = float(np.clip(float(speed) * float(t), 0.0, path_len))
    return start_xy + (d / path_len) * path


def _waypoint_xy_array(waypoints):
    arr = np.asarray(waypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("waypoints must be an array of shape (N, >=2).")
    return arr[:, :2].astype(float, copy=False)


def _nominal_ego_position_on_polyline(waypoints_xy, speed, t):
    path_xy = _waypoint_xy_array(waypoints_xy)
    if path_xy.shape[0] == 1:
        return path_xy[0].copy()
    remaining = float(max(0.0, speed) * max(0.0, t))
    for idx in range(path_xy.shape[0] - 1):
        seg_a = path_xy[idx]
        seg_b = path_xy[idx + 1]
        seg = seg_b - seg_a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-12:
            continue
        if remaining <= seg_len:
            return seg_a + (remaining / seg_len) * seg
        remaining -= seg_len
    return path_xy[-1].copy()


def _point_polyline_distance(pt, waypoints_xy):
    path_xy = _waypoint_xy_array(waypoints_xy)
    if path_xy.shape[0] == 1:
        return float(np.linalg.norm(np.asarray(pt, dtype=float).reshape(2,) - path_xy[0]))
    return float(
        min(
            _point_segment_distance(pt, path_xy[idx], path_xy[idx + 1])
            for idx in range(path_xy.shape[0] - 1)
        )
    )


def _rows_overlap(candidate_xy, candidate_r, existing_rows, margin):
    cxy = np.asarray(candidate_xy, dtype=float).reshape(2,)
    cr = float(candidate_r)
    for row in existing_rows:
        exy = np.asarray(row[:2], dtype=float).reshape(2,)
        er = float(row[2])
        if np.linalg.norm(cxy - exy) < (cr + er + float(margin)):
            return True
    return False


def _simulate_forced_event_nominal(
    *,
    start_xy,
    goal_xy,
    occluder_xy,
    occluder_radius,
    hidden_xy,
    hidden_vel,
    hidden_radius,
    nominal_speed=0.8,
    dt=0.1,
    horizon_s=10.0,
):
    start_xy = np.asarray(start_xy, dtype=float).reshape(2,)
    goal_xy = np.asarray(goal_xy, dtype=float).reshape(2,)
    occluder_xy = np.asarray(occluder_xy, dtype=float).reshape(2,)
    hidden_xy = np.asarray(hidden_xy, dtype=float).reshape(2,)
    hidden_vel = np.asarray(hidden_vel, dtype=float).reshape(2,)

    reveal_step = None
    reveal_time_s = None
    reveal_distance_to_path = None
    reveal_ttc_nominal = None
    min_distance_to_path = float("inf")
    min_ttc_nominal = float("inf")
    corridor_conflict = False

    n_steps = int(max(1, np.ceil(float(horizon_s) / float(dt))))
    hidden_speed = float(np.linalg.norm(hidden_vel))
    for step in range(n_steps + 1):
        t = float(step) * float(dt)
        ego_xy = _nominal_ego_position(start_xy, goal_xy, nominal_speed, t)
        hxy = hidden_xy + t * hidden_vel
        d_path = _point_segment_distance(hxy, start_xy, goal_xy)
        dist_to_ego = float(np.linalg.norm(hxy - ego_xy))
        ttc_nominal = dist_to_ego / max(float(nominal_speed) + hidden_speed, 1e-6)

        min_distance_to_path = min(min_distance_to_path, d_path)
        min_ttc_nominal = min(min_ttc_nominal, ttc_nominal)

        occluded = _disc_occludes_target(
            ego_xy,
            hxy,
            occluder_xy,
            occluder_radius,
            target_radius=hidden_radius,
        )
        if reveal_step is None and not occluded:
            reveal_step = int(step)
            reveal_time_s = float(t)
            reveal_distance_to_path = float(d_path)
            reveal_ttc_nominal = float(ttc_nominal)

        if d_path <= max(0.9, float(hidden_radius) + 0.45) and dist_to_ego <= 4.0:
            corridor_conflict = True

    return {
        "predicted_reveal_step": reveal_step,
        "predicted_reveal_time_s": reveal_time_s,
        "predicted_reveal_distance_to_path": reveal_distance_to_path,
        "predicted_reveal_ttc_nominal_s": reveal_ttc_nominal,
        "min_predicted_distance_to_path": (
            None if not np.isfinite(min_distance_to_path) else float(min_distance_to_path)
        ),
        "min_predicted_ttc_nominal_s": (
            None if not np.isfinite(min_ttc_nominal) else float(min_ttc_nominal)
        ),
        "corridor_conflict": bool(corridor_conflict),
    }


def _build_random_crowd_scenario(*, case_seed, n_rand, rand_obs, static_occluders):
    known_obs = np.empty((0, 8), dtype=float)
    obs_meta = []

    rand_rows, rand_meta = LocalTrackingControllerDyn_OCC.make_random_obstacles7(
        n_rand=int(n_rand),
        v_obs_max=(0.0 if bool(static_occluders) else 0.5),
        x_range=(8.0, 30.0),
        y_spawn_range=(0.0, 15.0),
        r_range=(0.3, 0.4),
        y_bounds=(0.0, 15.0),
        seed=int(case_seed),
        rand_obs=bool(rand_obs),
    )
    if rand_rows.size > 0:
        if bool(static_occluders):
            rand_rows[:, 3] = 0.0
            rand_rows[:, 4] = 0.0
            obs_meta = [{"mode": 0, "v_max": 0.0, "theta": 0.0} for _ in range(rand_rows.shape[0])]
            type_column = np.zeros((rand_rows.shape[0], 1))
        else:
            obs_meta = list(rand_meta)
            type_column = np.ones((rand_rows.shape[0], 1))
        known_obs = np.hstack((rand_rows, type_column))

    scenario_diag = {
        "crowd_mode": "random",
        "n_forced_events": 0,
        "n_forced_hidden_total": 0,
        "n_forced_extra_hidden": 0,
        "n_background_rand": int(known_obs.shape[0]),
        "n_forced_initially_occluded": 0,
        "n_forced_revealed": 0,
        "n_forced_corridor_conflict": 0,
        "min_reveal_distance_to_ego_path": None,
        "min_reveal_ttc_to_nominal_ego": None,
        "reveal_steps": [],
        "forced_event_meta": [],
    }
    return known_obs, obs_meta, scenario_diag


def _build_forced_emergence_crowd_scenario(
    *,
    case_seed,
    n_rand,
    rand_obs,
    static_occluders,
    waypoints,
    forced_events,
    forced_bg_rand,
    forced_hidden_speed,
    forced_occluder_radius_min,
    forced_occluder_radius_max,
    forced_validate_occlusion,
    forced_require_corridor_conflict,
):
    rng = np.random.default_rng(int(case_seed))
    start_xy = np.asarray(waypoints[0][:2], dtype=float).reshape(2,)
    goal_xy = np.asarray(waypoints[-1][:2], dtype=float).reshape(2,)

    forced_rows = []
    forced_meta = []
    forced_event_meta = []
    forced_occluder_guard_rows = []

    n_events_target = max(0, int(forced_events))
    y_min, y_max = 0.0, 15.0
    nominal_speed = 0.8

    def _sample_hidden_for_occluder(
        *,
        occ_xy,
        occ_radius,
        x_event,
        y_cross,
        existing_rows,
        lateral_jitter,
        target_x_jitter,
        target_y_jitter,
        overlap_margin,
        require_corridor_validation,
        attempts=64,
    ):
        occ_dir = _safe_normalize(np.asarray(occ_xy, dtype=float).reshape(2,) - start_xy)
        if occ_dir is None:
            return None
        occ_lat = np.array([-occ_dir[1], occ_dir[0]], dtype=float)

        for _ in range(int(max(1, attempts))):
            hidden_radius = float(rng.uniform(0.28, 0.48))
            behind_gap = occ_radius + hidden_radius + float(rng.uniform(0.08, 0.30))
            hidden_xy = (
                np.asarray(occ_xy, dtype=float).reshape(2,)
                + behind_gap * occ_dir
                + occ_lat * float(rng.uniform(-lateral_jitter, lateral_jitter))
            )
            if not (4.5 <= hidden_xy[0] <= 22.0 and 0.8 <= hidden_xy[1] <= 14.2):
                continue

            target_xy = np.array(
                [
                    float(x_event) + float(rng.uniform(-target_x_jitter, target_x_jitter)),
                    float(np.clip(float(y_cross) + rng.uniform(-target_y_jitter, target_y_jitter), 0.5, 14.5)),
                ],
                dtype=float,
            )
            vel_dir = _safe_normalize(target_xy - hidden_xy)
            if vel_dir is None:
                continue

            hidden_speed = float(forced_hidden_speed) * float(rng.uniform(0.88, 1.0))
            hidden_vel = hidden_speed * vel_dir

            initially_occluded = _disc_occludes_target(
                start_xy,
                hidden_xy,
                occ_xy,
                occ_radius,
                target_radius=hidden_radius,
            )
            if bool(forced_validate_occlusion) and not initially_occluded:
                continue

            pred = _simulate_forced_event_nominal(
                start_xy=start_xy,
                goal_xy=goal_xy,
                occluder_xy=occ_xy,
                occluder_radius=occ_radius,
                hidden_xy=hidden_xy,
                hidden_vel=hidden_vel,
                hidden_radius=hidden_radius,
                nominal_speed=nominal_speed,
                dt=0.1,
                horizon_s=10.0,
            )
            if bool(require_corridor_validation) and bool(forced_require_corridor_conflict) and not bool(pred["corridor_conflict"]):
                continue
            if _rows_overlap(hidden_xy, hidden_radius, existing_rows, margin=float(overlap_margin)):
                continue

            return {
                "row": np.array(
                    [hidden_xy[0], hidden_xy[1], hidden_radius, hidden_vel[0], hidden_vel[1], y_min, y_max, 1.0],
                    dtype=float,
                ),
                "meta": {
                    "mode": 1,
                    "v_max": float(hidden_speed),
                    "theta": float(np.arctan2(hidden_vel[1], hidden_vel[0])),
                },
                "hidden_xy": hidden_xy,
                "hidden_vel": hidden_vel,
                "hidden_speed": float(hidden_speed),
                "hidden_radius": float(hidden_radius),
                "initially_occluded": bool(initially_occluded),
                "pred": pred,
            }
        return None

    for event_id in range(n_events_target):
        event_added = False
        for _ in range(96):
            x_event = float(rng.uniform(6.0, 25.0))
            y_cross = float(np.clip(7.5 + rng.uniform(-1.2, 1.2), 1.0, 14.0))
            occ_radius = float(
                rng.uniform(float(forced_occluder_radius_min), float(forced_occluder_radius_max))
            )
            occ_speed = 0.0 if bool(static_occluders) else float(rng.uniform(0.10, 0.2))
            occ_theta = float(rng.uniform(-np.pi, np.pi))
            roam_half_x = 0.0 if bool(static_occluders) else float(rng.uniform(0.55, 1.05))
            # Let moving occluders roam more freely away from the ego corridor so
            # they do not keep getting reflected back near y=7.5 and cause deadlock.
            roam_inward_y = 0.0 if bool(static_occluders) else float(rng.uniform(0.12, 0.30))
            roam_outward_y = 0.0 if bool(static_occluders) else float(rng.uniform(1.60, 2.60))
            side_sign = 1.0 if rng.random() < 0.5 else -1.0
            lateral = float(rng.uniform(0.7, 1.6))
            occ_xy = np.array(
                [
                    x_event + float(rng.uniform(-0.35, 0.35)),
                    float(np.clip(y_cross + side_sign * lateral, 1.0, 14.0)),
                ],
                dtype=float,
            )
            occ_guard_radius = occ_radius + max(roam_half_x, roam_outward_y)
            if _rows_overlap(occ_xy, occ_guard_radius, forced_occluder_guard_rows, margin=0.45):
                continue

            occ_dir = _safe_normalize(occ_xy - start_xy)
            if occ_dir is None:
                continue
            primary_hidden = _sample_hidden_for_occluder(
                occ_xy=occ_xy,
                occ_radius=occ_radius,
                x_event=x_event,
                y_cross=y_cross,
                existing_rows=forced_rows,
                lateral_jitter=0.14,
                target_x_jitter=0.6,
                target_y_jitter=0.3,
                overlap_margin=0.22,
                require_corridor_validation=True,
                attempts=72,
            )
            if primary_hidden is None:
                continue

            occ_vx = float(occ_speed * np.cos(occ_theta))
            occ_vy = float(occ_speed * np.sin(occ_theta))
            corridor_y = 7.5
            corridor_clearance = 0.35
            occ_side_sign = 1.0 if float(occ_xy[1]) >= corridor_y else -1.0
            if bool(static_occluders):
                occ_y_min = float(occ_xy[1])
                occ_y_max = float(occ_xy[1])
            elif occ_side_sign > 0.0:
                occ_y_min = max(corridor_y + corridor_clearance, float(occ_xy[1] - roam_inward_y))
                occ_y_max = min(14.5, float(occ_xy[1] + roam_outward_y))
            else:
                occ_y_min = max(0.5, float(occ_xy[1] - roam_outward_y))
                occ_y_max = min(corridor_y - corridor_clearance, float(occ_xy[1] + roam_inward_y))
            occ_x_min = max(4.5, float(occ_xy[0] - roam_half_x))
            occ_x_max = min(21.5, float(occ_xy[0] + roam_half_x))
            occ_row = np.array(
                [occ_xy[0], occ_xy[1], occ_radius, occ_vx, occ_vy, occ_y_min, occ_y_max, 1.0],
                dtype=float,
            )
            hid_row = primary_hidden["row"]

            if _rows_overlap(occ_xy, occ_radius, forced_rows, margin=0.45):
                continue

            occ_idx = len(forced_rows)
            hid_idx = occ_idx + 1
            forced_rows.extend([occ_row, hid_row])
            forced_meta.extend(
                [
                    {
                        "mode": 1,
                        "v_max": float(occ_speed),
                        "theta": float(occ_theta),
                        "forced_occluder": True,
                        "forced_occluder_sep_margin": 0.45,
                        "heading_jitter_std": 0.004,
                        "large_turn_prob": 0.01,
                        "large_turn_std": 0.035,
                        "x_min": float(occ_x_min),
                        "x_max": float(occ_x_max),
                        "home_x": float(occ_xy[0]),
                        "home_y": float(occ_xy[1]),
                    },
                    {
                        "mode": 1,
                        "v_max": float(primary_hidden["hidden_speed"]),
                        "theta": float(np.arctan2(primary_hidden["hidden_vel"][1], primary_hidden["hidden_vel"][0])),
                    },
                ]
            )
            forced_event_meta.append(
                {
                    "event_id": int(event_id),
                    "occluder_index": int(occ_idx),
                    "hidden_index": int(hid_idx),
                    "hidden_indices": [int(hid_idx)],
                    "occluder_center": [float(occ_xy[0]), float(occ_xy[1])],
                    "occluder_radius": float(occ_radius),
                    "occluder_velocity": [float(occ_vx), float(occ_vy)],
                    "occluder_speed": float(occ_speed),
                    "occluder_roam_x_bounds": [float(occ_x_min), float(occ_x_max)],
                    "occluder_roam_y_bounds": [float(occ_y_min), float(occ_y_max)],
                    "x_event": float(x_event),
                    "y_cross": float(y_cross),
                    "hidden_initial_position": [
                        float(primary_hidden["hidden_xy"][0]),
                        float(primary_hidden["hidden_xy"][1]),
                    ],
                    "hidden_velocity": [
                        float(primary_hidden["hidden_vel"][0]),
                        float(primary_hidden["hidden_vel"][1]),
                    ],
                    "hidden_speed": float(primary_hidden["hidden_speed"]),
                    "initially_occluded_geom": bool(primary_hidden["initially_occluded"]),
                    "corridor_conflict": bool(primary_hidden["pred"]["corridor_conflict"]),
                    "predicted_reveal_step": primary_hidden["pred"]["predicted_reveal_step"],
                    "predicted_reveal_time_s": primary_hidden["pred"]["predicted_reveal_time_s"],
                    "predicted_reveal_distance_to_path": primary_hidden["pred"]["predicted_reveal_distance_to_path"],
                    "predicted_reveal_ttc_nominal_s": primary_hidden["pred"]["predicted_reveal_ttc_nominal_s"],
                    "min_predicted_distance_to_path": primary_hidden["pred"]["min_predicted_distance_to_path"],
                    "min_predicted_ttc_nominal_s": primary_hidden["pred"]["min_predicted_ttc_nominal_s"],
                    "extra_hidden_count": 0,
                    "initially_occluded_actual": None,
                    "revealed_actual": False,
                    "reveal_step_actual": None,
                    "reveal_time_actual_s": None,
                    "reveal_distance_to_path_actual": None,
                    "reveal_ttc_nominal_actual_s": None,
                }
            )
            forced_occluder_guard_rows.append(np.array([occ_xy[0], occ_xy[1], occ_guard_radius], dtype=float))
            event_added = True
            break

        if not event_added:
            continue

    hidden_total_target = int(max(len(forced_event_meta), int(n_rand)))
    preferred_bg_target = 0
    extra_hidden_target = 0
    extra_hidden_budget = max(0, hidden_total_target - int(len(forced_event_meta)))
    if forced_bg_rand is None:
        preferred_bg_target = (
            min(max(0, extra_hidden_budget // 3), max(0, 2 * int(len(forced_event_meta))))
            if bool(rand_obs)
            else 0
        )
        extra_hidden_target = max(0, extra_hidden_budget - preferred_bg_target)
    else:
        preferred_bg_target = max(0, int(forced_bg_rand)) if bool(rand_obs) else 0
        extra_hidden_target = max(0, hidden_total_target - int(len(forced_event_meta)) - preferred_bg_target)

    if extra_hidden_target > 0 and len(forced_event_meta) > 0:
        per_event = extra_hidden_target // len(forced_event_meta)
        rem = extra_hidden_target % len(forced_event_meta)
        for event_idx, meta in enumerate(forced_event_meta):
            add_count = int(per_event + (1 if event_idx < rem else 0))
            if add_count <= 0:
                continue
            occ_idx = int(meta["occluder_index"])
            occ_xy = np.asarray(meta["occluder_center"], dtype=float).reshape(2,)
            occ_radius = float(meta["occluder_radius"])
            x_event = float(meta.get("x_event", occ_xy[0]))
            y_cross = float(meta.get("y_cross", 7.5))
            occ_dir = _safe_normalize(occ_xy - start_xy)
            occ_lat = None if occ_dir is None else np.array([-occ_dir[1], occ_dir[0]], dtype=float)
            for extra_i in range(add_count):
                existing_rows_wo_self_occ = [row for i, row in enumerate(forced_rows) if int(i) != occ_idx]
                extra_hidden = _sample_hidden_for_occluder(
                    occ_xy=occ_xy,
                    occ_radius=occ_radius,
                    x_event=x_event,
                    y_cross=y_cross,
                    existing_rows=existing_rows_wo_self_occ,
                    lateral_jitter=0.42,
                    target_x_jitter=0.85,
                    target_y_jitter=0.45,
                    overlap_margin=0.14,
                    require_corridor_validation=False,
                    attempts=80,
                )
                if extra_hidden is None and occ_dir is not None and occ_lat is not None:
                    for det_try in range(5):
                        hidden_radius = float(rng.uniform(0.28, 0.42))
                        side_slot = float((extra_i % 3) - 1)
                        layer = int(extra_i // 3) + int(det_try)
                        lateral_offset = side_slot * (0.26 + 0.06 * float(det_try))
                        behind_gap = occ_radius + hidden_radius + 0.18 + 0.22 * float(layer)
                        hidden_xy = occ_xy + behind_gap * occ_dir + lateral_offset * occ_lat
                        if not (4.5 <= hidden_xy[0] <= 22.0 and 0.8 <= hidden_xy[1] <= 14.2):
                            continue
                        if not _disc_occludes_target(start_xy, hidden_xy, occ_xy, occ_radius, target_radius=hidden_radius):
                            continue
                        if _rows_overlap(hidden_xy, hidden_radius, existing_rows_wo_self_occ, margin=0.10):
                            continue
                        target_xy = np.array(
                            [
                                x_event + float(rng.uniform(-0.6, 0.6)),
                                float(np.clip(y_cross + rng.uniform(-0.35, 0.35), 0.5, 14.5)),
                            ],
                            dtype=float,
                        )
                        vel_dir = _safe_normalize(target_xy - hidden_xy)
                        if vel_dir is None:
                            continue
                        hidden_speed = float(forced_hidden_speed) * float(rng.uniform(0.9, 1.0))
                        hidden_vel = hidden_speed * vel_dir
                        extra_hidden = {
                            "row": np.array(
                                [hidden_xy[0], hidden_xy[1], hidden_radius, hidden_vel[0], hidden_vel[1], y_min, y_max, 1.0],
                                dtype=float,
                            ),
                            "meta": {
                                "mode": 1,
                                "v_max": float(hidden_speed),
                                "theta": float(np.arctan2(hidden_vel[1], hidden_vel[0])),
                            },
                        }
                        break
                if extra_hidden is None:
                    continue
                hid_idx = len(forced_rows)
                forced_rows.append(extra_hidden["row"])
                forced_meta.append(extra_hidden["meta"])
                meta["hidden_indices"].append(int(hid_idx))
                meta["extra_hidden_count"] = int(meta.get("extra_hidden_count", 0) + 1)

    hidden_total_actual = int(sum(len(meta.get("hidden_indices", [meta["hidden_index"]])) for meta in forced_event_meta))
    bg_target = max(int(preferred_bg_target), hidden_total_target - hidden_total_actual)

    bg_rows_8 = np.empty((0, 8), dtype=float)
    bg_meta = []
    if bg_target > 0:
        keep_rows = []
        keep_meta = []
        batch_id = 0
        while len(keep_rows) < bg_target and batch_id < 16:
            sample_target = max((bg_target - len(keep_rows)) * 4, bg_target)
            extra_rows, extra_meta = LocalTrackingControllerDyn_OCC.make_random_obstacles7(
                n_rand=int(sample_target),
                v_obs_max=(0.0 if bool(static_occluders) else 0.5),
                x_range=(20.0, 30.0),
                y_spawn_range=(1.0, 14.0),
                r_range=(0.3, 0.4),
                y_bounds=(0.0, 15.0),
                seed=int(case_seed) + 7919 * int(batch_id),
                rand_obs=True,
            )
            for row, meta in zip(extra_rows, extra_meta):
                if len(keep_rows) >= bg_target:
                    break
                if float(np.linalg.norm(np.asarray(row[:2], dtype=float).reshape(2,) - start_xy)) < 2.4:
                    continue
                if _rows_overlap(row[:2], row[2], forced_rows + keep_rows, margin=0.2):
                    continue
                keep_rows.append(np.asarray(row, dtype=float))
                if bool(static_occluders):
                    keep_meta.append({"mode": 0, "v_max": 0.0, "theta": 0.0})
                else:
                    keep_meta.append(dict(meta))
            batch_id += 1

        if len(keep_rows) < bg_target:
            raise RuntimeError(
                f"Failed to backfill background obstacles to target count: "
                f"need {bg_target}, placed {len(keep_rows)}."
            )

        bg_rows = np.vstack(keep_rows)
        if bool(static_occluders):
            bg_rows[:, 3] = 0.0
            bg_rows[:, 4] = 0.0
            bg_type = np.zeros((bg_rows.shape[0], 1), dtype=float)
        else:
            bg_type = np.ones((bg_rows.shape[0], 1), dtype=float)
        bg_rows_8 = np.hstack((bg_rows, bg_type))
        bg_meta = keep_meta

    known_obs_parts = []
    if forced_rows:
        known_obs_parts.append(np.vstack(forced_rows))
    if bg_rows_8.size > 0:
        known_obs_parts.append(bg_rows_8)
    known_obs = (
        np.vstack(known_obs_parts)
        if known_obs_parts
        else np.empty((0, 8), dtype=float)
    )
    obs_meta = list(forced_meta) + list(bg_meta)

    min_reveal_dist = []
    min_reveal_ttc = []
    for meta in forced_event_meta:
        if meta["predicted_reveal_distance_to_path"] is not None:
            min_reveal_dist.append(float(meta["predicted_reveal_distance_to_path"]))
        if meta["predicted_reveal_ttc_nominal_s"] is not None:
            min_reveal_ttc.append(float(meta["predicted_reveal_ttc_nominal_s"]))

    scenario_diag = {
        "crowd_mode": "forced_emergence",
        "n_forced_events": int(len(forced_event_meta)),
        "n_forced_hidden_total": int(sum(len(meta.get("hidden_indices", [meta["hidden_index"]])) for meta in forced_event_meta)),
        "n_forced_extra_hidden": int(sum(int(meta.get("extra_hidden_count", 0) or 0) for meta in forced_event_meta)),
        "n_background_rand": int(bg_rows_8.shape[0]),
        "n_forced_initially_occluded": int(
            sum(1 for meta in forced_event_meta if bool(meta["initially_occluded_geom"]))
        ),
        "n_forced_revealed": 0,
        "n_forced_corridor_conflict": int(
            sum(1 for meta in forced_event_meta if bool(meta["corridor_conflict"]))
        ),
        "min_reveal_distance_to_ego_path": (
            None if len(min_reveal_dist) == 0 else float(np.min(min_reveal_dist))
        ),
        "min_reveal_ttc_to_nominal_ego": (
            None if len(min_reveal_ttc) == 0 else float(np.min(min_reveal_ttc))
        ),
        "reveal_steps": [],
        "forced_event_meta": forced_event_meta,
    }
    return known_obs, obs_meta, scenario_diag



def run_crowd_scenario(
    controller_type=None,
    model_key="di",
    show_animation=True,
    save_animation=False,
    tf=100.0,
    seed=42,
    case_idx=None,
    rand_obs=True,
    n_rand=50,
    du_min_speed_scale=None,
    du_k_turn_brake=None,
    du_k_a_p=None,
    du_k_a_d=None,
    du_reverse_enter_cos=None,
    du_reverse_exit_cos=None,
    du_reverse_min_scale=None,
    vref_mode_occ=None,
    vref_front_mode_occ=None,
    occ_version=None,
    occ_visible_scale=None,
    occ_enable_visible_hocbf=False,
    oa_dynamic_occluders=None,
    oa_allow_solver_fallback=None,
    oa_dsafe=None,
    oa_visible_reach_mode=None,
    oa_use_nominal_tracking_cost=None,
    oa_wmax="default",
    oa_dt=None,
    crowd_mode="random",
    forced_events=6,
    forced_bg_rand=None,
    forced_hidden_speed=1.0,
    forced_occluder_radius_min=0.8,
    forced_occluder_radius_max=1.0,
    forced_validate_occlusion=True,
    forced_require_corridor_conflict=True,
    static_occluders=False,
    backup_cbf_overrides=None,
    robot_spec_overrides=None,
    waypoints_override=None,
    env_width_override=None,
    env_height_override=None,
    known_obs_override=None,
    obs_meta_override=None,
    scenario_diag_override=None,
    return_metrics=False,
    max_steps=None,
    max_sim_time=None,
    deadlock_window_steps=120,
    deadlock_progress_eps=0.05,
    tracking_view_enable=False,
    tracking_view_window_size=5.0,
    scenario_name="Crowd1",
):
    if controller_type is None:
        controller_type = {"pos": "occlusion_cbf_qp"}

    is_oa_mpc = str(controller_type.get("pos", "")).strip().lower() == "oa_mpc"

    mk = str(model_key).strip().lower()
    if mk in {"di", "doubleintegrator2d"}:
        model = "DoubleIntegrator2D"
    elif mk in {"du", "dynamicunicycle2d"}:
        model = "DynamicUnicycle2D"
    elif mk in {"uni", "unicycle2d", "un"}:
        model = "Unicycle2D"
    else:
        raise ValueError(f"Unsupported model `{model_key}`. Use `di`, `du`, or `uni`.")

    if is_oa_mpc and oa_dt is not None:
        dt = float(oa_dt)
        if (not np.isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"Invalid --oa-dt: {oa_dt}. It must be a positive finite value.")
    else:
        # Keep default simulation dt fixed for fair preset on/off comparison.
        dt = 0.05

    if waypoints_override is None:
        waypoints = np.array(
            [
                [4.0, 7.5, 0.0],
                [20.0, 7.5, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        waypoints = np.asarray(waypoints_override, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 2:
            raise ValueError("waypoints_override must have shape (N>=2, >=2).")

    # Crowd scenario base obstacles are intentionally disabled so that this
    # scenario is composed only of random obstacles.
    # known_obs format: [x, y, r, vx, vy, y_min, y_max, type]
    case_seed = _compute_case_seed(seed, case_idx)
    crowd_mode = str(crowd_mode).strip().lower()
    if crowd_mode not in {"random", "forced_emergence"}:
        raise ValueError(f"Unsupported crowd_mode `{crowd_mode}`. Use `random` or `forced_emergence`.")

    if known_obs_override is not None:
        known_obs = np.asarray(known_obs_override, dtype=float)
        if known_obs.ndim != 2 or known_obs.shape[1] != 8:
            raise ValueError("known_obs_override must have shape (N, 8).")
        obs_meta = [] if obs_meta_override is None else list(obs_meta_override)
        scenario_diag = {} if scenario_diag_override is None else dict(scenario_diag_override)
        scenario_diag.setdefault("crowd_mode", crowd_mode)
    else:
        if crowd_mode == "forced_emergence":
            known_obs, obs_meta, scenario_diag = _build_forced_emergence_crowd_scenario(
                case_seed=case_seed,
                n_rand=n_rand,
                rand_obs=rand_obs,
                static_occluders=static_occluders,
                waypoints=waypoints,
                forced_events=forced_events,
                forced_bg_rand=forced_bg_rand,
                forced_hidden_speed=forced_hidden_speed,
                forced_occluder_radius_min=forced_occluder_radius_min,
                forced_occluder_radius_max=forced_occluder_radius_max,
                forced_validate_occlusion=forced_validate_occlusion,
                forced_require_corridor_conflict=forced_require_corridor_conflict,
            )
        else:
            known_obs, obs_meta, scenario_diag = _build_random_crowd_scenario(
                case_seed=case_seed,
                n_rand=n_rand,
                rand_obs=rand_obs,
                static_occluders=static_occluders,
            )

    env_width = float(24.0 if env_width_override is None else env_width_override)
    env_height = float(15.0 if env_height_override is None else env_height_override)

    if backup_cbf_overrides is None:
        backup_cbf_overrides = {}
    else:
        backup_cbf_overrides = dict(backup_cbf_overrides)
    if vref_front_mode_occ is not None:
        backup_cbf_overrides["vref_front_mode_occ"] = str(vref_front_mode_occ).strip().lower()
    if robot_spec_overrides is None:
        robot_spec_overrides = {}
    else:
        robot_spec_overrides = dict(robot_spec_overrides)

    if model == "DoubleIntegrator2D":
        # DI is sensitive to sharp multi-scenario occlusion velocity aggregation
        # and overly deep terminal stop margins in dense forced-emergence scenes.
        di_backup_cfg = {
            "T_horizon": 0.5,
            "vref_scenario_softmax_kappa": 0.0,
            "rho_T": "auto",
        }
        di_backup_cfg.update(backup_cbf_overrides)
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
            "backup_cbf": di_backup_cfg,
            "show_backup_rollout": True,
            "backup_rollout_every": 1,
            "use_occ": True,
            "dynamic_obs_types": [1],
        }
    elif model == "DynamicUnicycle2D":
        du_vmax = 1.0
        du_backup_cfg = {
            "T_horizon": 1.0,
            "vref_scenario_softmax_kappa": 0.0,
            "du_nonnegative_speed_occ_du": False,
        }
        if du_k_a_p is not None:
            du_backup_cfg["k_a_occ_du_p"] = float(du_k_a_p)
            du_backup_cfg["k_a_track_occ_du_p"] = float(du_k_a_p)
        if du_k_a_d is not None:
            du_backup_cfg["k_a_occ_du_d"] = float(du_k_a_d)
            du_backup_cfg["k_a_track_occ_du_d"] = float(du_k_a_d)
        if vref_mode_occ is not None:
            du_backup_cfg["vref_mode_occ_du"] = str(vref_mode_occ).strip().lower()
        if backup_cbf_overrides:
            du_backup_cfg.update(backup_cbf_overrides)
        robot_spec = {
            "model": "DynamicUnicycle2D",
            "v_max": du_vmax,
            "v_min": -du_vmax,
            "v_obs_max": 0.5,
            "a_max": 1.0,
            "w_max": 0.8,
            "radius": 0.25,
            "debug_backup_qp": False,
            "cbf_feas_tol": 5e-3,
            "sensing_range": 10.0,
            "fov_angle": 360.0,
            "backup_cbf": du_backup_cfg,
            "show_backup_rollout": True,
            "backup_rollout_every": 1,
            "use_occ": True,
            "dynamic_obs_types": [1 ],
        }
    elif model == "Unicycle2D":
        uni_backup_cfg = {
            "T_horizon": 0.5,
            "vref_scenario_softmax_kappa": 0.0,
        }
        if vref_mode_occ is not None:
            uni_backup_cfg["vref_mode_occ_uni"] = str(vref_mode_occ).strip().lower()
        if backup_cbf_overrides:
            uni_backup_cfg.update(backup_cbf_overrides)
        uni_vmax = 1.0
        wmax_mode = str(oa_wmax).strip().lower()
        if wmax_mode not in {"default", "pi"}:
            raise ValueError(f"Invalid --wmax: {oa_wmax}. Use one of: default, pi.")
        uni_wmax = float(np.pi) if (is_oa_mpc and wmax_mode == "pi") else 0.8
        # Keep robot radius fixed for fair preset/non-preset comparison.
        uni_radius = 0.25
        robot_spec = {
            "model": "Unicycle2D",
            "v_max": uni_vmax,
            "w_max": uni_wmax,
            "radius": uni_radius,
            "debug_backup_qp": False,
            "sensing_range": 10.0,
            "fov_angle": 360.0,
            "backup_cbf": uni_backup_cfg,
            "show_backup_rollout": True,
            "backup_rollout_every": 1,
            "use_occ": True,
            "dynamic_obs_types": [1],
        }

    if robot_spec_overrides:
        robot_spec.update(robot_spec_overrides)

    # Crowd scenario behavior model for moving obstacles.
    _apply_crowd_dynamic_obstacle_defaults(robot_spec)
    if crowd_mode == "forced_emergence":
        robot_spec["v_adv_max_occ"] = 1.0
        dyn_cfg = robot_spec.setdefault("crowd_dyn_obs", {})
        dyn_cfg["occluded_speed_boost_enable"] = True
        dyn_cfg["occluded_speed_boost_vmax"] = float(robot_spec["v_adv_max_occ"])
        dyn_cfg["occluded_speed_boost_fov_only"] = True
        dyn_cfg["occluded_speed_boost_on_hysteresis_steps"] = 2
        dyn_cfg["occluded_speed_boost_off_hysteresis_steps"] = 5

    if occ_visible_scale is not None:
        vis_scale = float(occ_visible_scale)
        if (not np.isfinite(vis_scale)) or vis_scale <= 0.0:
            raise ValueError(
                f"Invalid --occ-visible-scale: {occ_visible_scale}. It must be a positive finite value."
            )
        robot_spec["occ_visible_scale"] = float(vis_scale)
    if occ_version is not None:
        occ_version_str = str(occ_version).strip().lower()
        if occ_version_str not in {"v1", "v2"}:
            raise ValueError(f"Invalid --occ-version: {occ_version}. Use one of: v1, v2.")
        robot_spec["occ_version"] = occ_version_str
    if occ_enable_visible_hocbf is not None:
        robot_spec["enable_visible_hocbf_in_occ"] = bool(occ_enable_visible_hocbf)

    if str(controller_type.get("pos", "")).strip().lower() == "oa_mpc":
        oa_cfg = robot_spec.setdefault("oa_mpc", {})
        # Fix OA-MPC to paper-mode behavior in crowd benchmark.
        oa_cfg["paper_mode"] = True
        oa_cfg["N"] = 10
        oa_cfg["auto_scale_N_with_dt"] = True
        oa_cfg["paper_horizon_time"] = 1.0
        # oa_cfg.setdefault("dsafe", 0.5)
        # oa_cfg.setdefault("visible_reach_mode", "worst_case")
        # oa_cfg.setdefault("use_nominal_tracking_cost", False)
        # oa_cfg.setdefault("dynamic_occluders", False)
        # Keep default safety distance unless overridden
        if oa_dsafe is None:
            oa_cfg["dsafe"] = float(oa_cfg.get("dsafe", 0.5))
        else:
            oa_cfg["dsafe"] = float(oa_dsafe)

        # Crowd-adapted OA defaults
        oa_cfg["dynamic_occluders"] = True
        oa_cfg["visible_reach_mode"] = "constant_velocity"
        oa_cfg["use_nominal_tracking_cost"] = True

        # OA paper baseline: static occluders only.
        robot_spec["occlusion_types"] = [0, 1]
        # robot_spec.setdefault("occlusion_types", [0])
        if oa_dynamic_occluders is not None:
            oa_cfg["dynamic_occluders"] = bool(oa_dynamic_occluders)
            robot_spec["occlusion_types"] = [0, 1] if oa_cfg["dynamic_occluders"] else [0]
        if oa_allow_solver_fallback is not None:
            oa_cfg["allow_solver_fallback"] = bool(oa_allow_solver_fallback)
        if oa_dsafe is not None:
            oa_cfg["dsafe"] = float(oa_dsafe)
        if oa_visible_reach_mode is not None:
            oa_cfg["visible_reach_mode"] = str(oa_visible_reach_mode).strip().lower()
        if oa_use_nominal_tracking_cost is not None:
            oa_cfg["use_nominal_tracking_cost"] = bool(oa_use_nominal_tracking_cost)

    pos_name = str(controller_type.get("pos", "")).strip().lower()
    if pos_name == "single_risk_mpc":
        _apply_single_risk_defaults(robot_spec)
    elif pos_name == "control_tree_mpc":
        _apply_control_tree_defaults(robot_spec)
    elif pos_name == "oacp_mpc":
        _apply_oacp_defaults(robot_spec)

    planner_label_map = dict(CROWD_PLANNER_LABELS)
    planner_label_map["oa_mpc"] = f"OA-MPC (wmax={str(oa_wmax).strip().lower()})"
    planner_label = planner_label_map.get(pos_name, str(controller_type.get("pos", "")).strip())
    model_label_map = {
        "DoubleIntegrator2D": "DI",
        "DynamicUnicycle2D": "DU",
        "Unicycle2D": "Uni",
    }
    model_label = model_label_map.get(model, model)
    figure_title = f"{scenario_name} | {planner_label} | model={model_label}"

    x_init = waypoints[0]

    if show_animation:
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
        tracking_view_ax = None
        if bool(tracking_view_enable):
            (ax, tracking_view_ax, _), fig = plot_handler.plot_grid(
                "Global View",
                with_right_subplot=True,
            )
        else:
            ax, fig = plot_handler.plot_grid("Global View")
        fig.suptitle(figure_title, fontsize=13, y=0.975)
        if bool(tracking_view_enable):
            fig.subplots_adjust(top=0.86)
        else:
            fig.subplots_adjust(top=0.90)
    else:
        ax = None
        fig = None
        tracking_view_ax = None

    env_handler = env.Env()

    tracking_controller = LocalTrackingControllerDyn_OCC(
        x_init,
        robot_spec,
        controller_type=controller_type,
        dt=dt,
        show_animation=show_animation,
        save_animation=save_animation,
        show_mpc_traj=False,
        ax=ax,
        fig=fig,
        env=env_handler,
        rand_seed=case_seed,
        tracking_view_ax=tracking_view_ax,
        tracking_view_window_size=tracking_view_window_size,
    )

    tracking_controller.obs = known_obs.astype(float)
    tracking_controller.set_obs_meta(obs_meta)

    tracking_controller.set_waypoints(waypoints)
    if not bool(return_metrics):
        return tracking_controller.run_all_steps(tf=float(tf))

    tf_cap = float(max_sim_time) if max_sim_time is not None else float(tf)
    n_steps = int(np.ceil(tf_cap / dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    final_goal = np.asarray(waypoints[-1], dtype=float).reshape(-1)[:2]
    waypoint_xy = _waypoint_xy_array(waypoints)
    ret_last = 0
    terminal_event = None
    compute_ms = []
    feasible_frac = []
    min_risk_margin = []
    min_visible_margin = []
    solve_ms_trunk_vals = []
    solve_ms_branch_sum_vals = []
    guidance_activated_steps = 0
    effective_wtrack = []
    effective_n_split = []
    v_ref_floor_eff = []
    selected_branch_vals = []
    intervention_l2_sq = []
    intervention_active_steps = 0
    total_steps = 0
    nominal_speed = 0.8

    forced_event_meta = []
    reveal_steps = []
    min_reveal_distance_to_path_actual = []
    min_reveal_ttc_actual = []
    if crowd_mode == "forced_emergence":
        forced_event_meta = [dict(meta) for meta in scenario_diag.get("forced_event_meta", [])]

    def _visible_index_set():
        occ_filter_fn = getattr(tracking_controller, "_resolve_occ_filter_fn", None)
        if occ_filter_fn is None:
            return set(range(int(tracking_controller.obs.shape[0])))
        try:
            fn = occ_filter_fn()
            if fn is None:
                return set(range(int(tracking_controller.obs.shape[0])))
            _, _, vis_idx = fn(tracking_controller.robot.X, tracking_controller.obs, return_indices=True)
            vis_idx = np.asarray(vis_idx, dtype=int).reshape(-1)
            return {int(i) for i in vis_idx.tolist()}
        except Exception:
            return set(range(int(tracking_controller.obs.shape[0])))

    if forced_event_meta:
        initial_visible = _visible_index_set()
        for meta in forced_event_meta:
            hidden_indices = [int(i) for i in meta.get("hidden_indices", [meta["hidden_index"]])]
            initially_occluded_actual = bool(any(hidden_idx not in initial_visible for hidden_idx in hidden_indices))
            meta["initially_occluded_actual"] = initially_occluded_actual
            if not initially_occluded_actual:
                meta["revealed_actual"] = True
                meta["reveal_step_actual"] = 0
                meta["reveal_time_actual_s"] = 0.0
                ego_nom_xy = _nominal_ego_position_on_polyline(waypoint_xy, nominal_speed, 0.0)
                reveal_candidates = []
                for hidden_idx in hidden_indices:
                    if hidden_idx not in initial_visible:
                        continue
                    hidden_xy = np.asarray(tracking_controller.obs[hidden_idx, :2], dtype=float).reshape(2,)
                    d_path = _point_polyline_distance(hidden_xy, waypoint_xy)
                    hidden_speed = float(np.linalg.norm(tracking_controller.obs[hidden_idx, 3:5]))
                    d_nom = float(np.linalg.norm(hidden_xy - ego_nom_xy))
                    reveal_ttc = float(d_nom / max(float(nominal_speed) + hidden_speed, 1e-6))
                    reveal_candidates.append((float(reveal_ttc), float(d_path)))
                if reveal_candidates:
                    reveal_ttc, d_path = min(reveal_candidates, key=lambda x: (x[0], x[1]))
                    meta["reveal_distance_to_path_actual"] = float(d_path)
                    meta["reveal_ttc_nominal_actual_s"] = float(reveal_ttc)
                reveal_steps.append(0)
                if reveal_candidates:
                    min_reveal_distance_to_path_actual.append(float(d_path))
                    min_reveal_ttc_actual.append(float(meta["reveal_ttc_nominal_actual_s"]))

    deadlock_window = int(max(0, deadlock_window_steps if deadlock_window_steps is not None else 0))
    deadlock_eps = float(max(0.0, deadlock_progress_eps))
    goal_dist_window = deque(maxlen=max(1, deadlock_window)) if deadlock_window > 0 else None
    deadlock_detected = False

    for _ in range(n_steps):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        ret_last = ret
        total_steps += 1
        terminal_event = getattr(tracking_controller, "last_terminal_event", None)

        pos_controller = getattr(tracking_controller, "pos_controller", None)
        profile = getattr(pos_controller, "last_profile", None) if pos_controller is not None else None
        step_ms = None
        if isinstance(profile, dict):
            step_ms = profile.get("total_ms", profile.get("solve_ms", None))
            ff = profile.get("feasible_horizon_fraction", None)
            mr = profile.get("min_risk_margin", None)
            mv = profile.get("min_visible_margin", None)
            if ff is not None:
                feasible_frac.append(float(ff))
            if mr is not None:
                min_risk_margin.append(float(mr))
            if mv is not None:
                min_visible_margin.append(float(mv))
            if bool(profile.get("guidance_active", False)):
                guidance_activated_steps += 1
            ew = profile.get("effective_wtrack", None)
            en = profile.get("effective_n_split", None)
            vf = profile.get("v_ref_floor_eff", None)
            st = profile.get("solve_ms_trunk", None)
            sbs = profile.get("solve_ms_total_branch_sum", None)
            sb = profile.get("selected_branch", None)
            if ew is not None:
                effective_wtrack.append(float(ew))
            if en is not None:
                effective_n_split.append(float(en))
            if vf is not None:
                v_ref_floor_eff.append(float(vf))
            if st is not None:
                solve_ms_trunk_vals.append(float(st))
            if sbs is not None:
                solve_ms_branch_sum_vals.append(float(sbs))
            if sb is not None:
                selected_branch_vals.append(str(sb))
        if pos_controller is not None:
            u_cmd = getattr(pos_controller, "last_u", None)
            u_nom = getattr(pos_controller, "last_u_ref", None)
            if u_cmd is not None and u_nom is not None:
                try:
                    uc = np.asarray(u_cmd, dtype=float).reshape(-1)
                    un = np.asarray(u_nom, dtype=float).reshape(-1)
                    m = min(len(uc), len(un))
                    if m > 0:
                        val = float(np.sum((uc[:m] - un[:m]) ** 2))
                        if np.isfinite(val):
                            intervention_l2_sq.append(val)
                            tol = float(robot_spec.get("intervention_tol", 1e-3))
                            if val > (tol * tol):
                                intervention_active_steps += 1
                except Exception:
                    pass
        if step_ms is not None and np.isfinite(step_ms):
            compute_ms.append(float(step_ms))

        if forced_event_meta:
            visible_now = _visible_index_set()
            cur_step = int(total_steps)
            cur_time = float(cur_step * dt)
            ego_nom_xy = _nominal_ego_position_on_polyline(waypoint_xy, nominal_speed, cur_time)
            for meta in forced_event_meta:
                if bool(meta.get("revealed_actual", False)):
                    continue
                hidden_indices = [int(i) for i in meta.get("hidden_indices", [meta["hidden_index"]])]
                reveal_candidates = []
                for hidden_idx in hidden_indices:
                    if hidden_idx not in visible_now:
                        continue
                    hidden_xy = np.asarray(tracking_controller.obs[hidden_idx, :2], dtype=float).reshape(2,)
                    d_path = _point_polyline_distance(hidden_xy, waypoint_xy)
                    hidden_speed = float(np.linalg.norm(np.asarray(tracking_controller.obs[hidden_idx, 3:5], dtype=float)))
                    d_nom = float(np.linalg.norm(hidden_xy - ego_nom_xy))
                    reveal_ttc = float(d_nom / max(float(nominal_speed) + hidden_speed, 1e-6))
                    reveal_candidates.append((float(reveal_ttc), float(d_path)))
                if reveal_candidates:
                    reveal_ttc, d_path = min(reveal_candidates, key=lambda x: (x[0], x[1]))
                    meta["revealed_actual"] = True
                    meta["reveal_step_actual"] = cur_step
                    meta["reveal_time_actual_s"] = cur_time
                    meta["reveal_distance_to_path_actual"] = float(d_path)
                    meta["reveal_ttc_nominal_actual_s"] = float(reveal_ttc)
                    reveal_steps.append(cur_step)
                    min_reveal_distance_to_path_actual.append(float(d_path))
                    min_reveal_ttc_actual.append(float(reveal_ttc))

        if ret in (-1, -2):
            break

        # Simple deadlock detector for full-rollout benchmarking:
        # if goal distance does not improve over a sliding window, terminate as timeout/deadlock.
        if goal_dist_window is not None:
            cur_xy = np.asarray(tracking_controller.robot.X[:2, 0], dtype=float).reshape(2,)
            cur_goal_dist = float(np.linalg.norm(cur_xy - final_goal))
            goal_dist_window.append(cur_goal_dist)
            if len(goal_dist_window) >= goal_dist_window.maxlen:
                if (max(goal_dist_window) - min(goal_dist_window)) <= deadlock_eps:
                    deadlock_detected = True
                    break

    if terminal_event == "success" or ret_last == -1:
        outcome = "success"
    elif terminal_event == "collision":
        outcome = "collision"
    elif terminal_event == "infeasible":
        outcome = "infeasible"
    elif ret_last == -2:
        # Backward-compatible fallback if terminal_event was not set.
        outcome = "infeasible"
    else:
        outcome = "timeout/deadlock"

    if outcome in {"collision", "infeasible"}:
        status = "failure"
    elif outcome == "success":
        status = "success"
    else:
        status = "ongoing"

    end_xy = np.asarray(tracking_controller.robot.X[:2, 0], dtype=float).reshape(2,)
    end_goal_distance = float(np.linalg.norm(end_xy - final_goal))
    steps_executed = int(max(1, total_steps))
    total_sim_time = float(steps_executed * dt)
    guidance_active_ratio = float(guidance_activated_steps) / float(max(1, steps_executed))
    intervention_active_ratio = float(intervention_active_steps) / float(max(1, steps_executed))
    selected_branch_counts = {}
    for sb in selected_branch_vals:
        key = str(sb)
        selected_branch_counts[key] = int(selected_branch_counts.get(key, 0) + 1)

    if forced_event_meta:
        scenario_diag["forced_event_meta"] = forced_event_meta
        scenario_diag["n_forced_initially_occluded"] = int(
            sum(1 for meta in forced_event_meta if bool(meta.get("initially_occluded_actual", False)))
        )
        scenario_diag["n_forced_revealed"] = int(
            sum(1 for meta in forced_event_meta if bool(meta.get("revealed_actual", False)))
        )
        scenario_diag["n_forced_corridor_conflict"] = int(
            sum(1 for meta in forced_event_meta if bool(meta.get("corridor_conflict", False)))
        )
        scenario_diag["reveal_steps"] = [int(s) for s in reveal_steps]
        scenario_diag["min_reveal_distance_to_ego_path"] = (
            None
            if len(min_reveal_distance_to_path_actual) == 0
            else float(np.min(min_reveal_distance_to_path_actual))
        )
        scenario_diag["min_reveal_ttc_to_nominal_ego"] = (
            None if len(min_reveal_ttc_actual) == 0 else float(np.min(min_reveal_ttc_actual))
        )

    return {
        "status": status,
        "ret": int(ret_last),
        "outcome": outcome,
        "terminal_event": terminal_event,
        "steps_executed": steps_executed,
        "total_steps": steps_executed,
        "total_sim_time": total_sim_time,
        "end_goal_distance": end_goal_distance,
        "final_goal_distance": end_goal_distance,
        "avg_solve_time_ms": (None if len(compute_ms) == 0 else float(np.mean(compute_ms))),
        "avg_solve_ms_trunk": (None if len(solve_ms_trunk_vals) == 0 else float(np.mean(solve_ms_trunk_vals))),
        "avg_branch_sum_ms": (None if len(solve_ms_branch_sum_vals) == 0 else float(np.mean(solve_ms_branch_sum_vals))),
        "avg_feasible_horizon_fraction": (None if len(feasible_frac) == 0 else float(np.mean(feasible_frac))),
        "avg_min_risk_margin": (None if len(min_risk_margin) == 0 else float(np.mean(min_risk_margin))),
        "avg_min_visible_margin": (None if len(min_visible_margin) == 0 else float(np.mean(min_visible_margin))),
        "guidance_activated_steps": int(guidance_activated_steps),
        "guidance_active_ratio": guidance_active_ratio,
        "avg_effective_wtrack": (None if len(effective_wtrack) == 0 else float(np.mean(effective_wtrack))),
        "avg_effective_n_split": (None if len(effective_n_split) == 0 else float(np.mean(effective_n_split))),
        "avg_v_ref_floor_eff": (None if len(v_ref_floor_eff) == 0 else float(np.mean(v_ref_floor_eff))),
        "avg_control_intervention_l2_sq": (
            None if len(intervention_l2_sq) == 0 else float(np.mean(intervention_l2_sq))
        ),
        "intervention_active_steps": int(intervention_active_steps),
        "intervention_active_ratio": float(intervention_active_ratio),
        "selected_branch_counts": selected_branch_counts,
        "deadlock_detected": bool(deadlock_detected),
        "crowd_mode": str(scenario_diag.get("crowd_mode", crowd_mode)),
        "n_forced_events": int(scenario_diag.get("n_forced_events", 0) or 0),
        "n_forced_hidden_total": int(scenario_diag.get("n_forced_hidden_total", 0) or 0),
        "n_forced_extra_hidden": int(scenario_diag.get("n_forced_extra_hidden", 0) or 0),
        "n_background_rand": int(scenario_diag.get("n_background_rand", 0) or 0),
        "n_forced_initially_occluded": int(scenario_diag.get("n_forced_initially_occluded", 0) or 0),
        "n_forced_revealed": int(scenario_diag.get("n_forced_revealed", 0) or 0),
        "n_forced_corridor_conflict": int(scenario_diag.get("n_forced_corridor_conflict", 0) or 0),
        "min_reveal_distance_to_ego_path": scenario_diag.get("min_reveal_distance_to_ego_path", None),
        "min_reveal_ttc_to_nominal_ego": scenario_diag.get("min_reveal_ttc_to_nominal_ego", None),
        "reveal_steps": list(scenario_diag.get("reveal_steps", [])),
        "forced_event_meta": list(scenario_diag.get("forced_event_meta", [])),
    }


def main():
    parser = argparse.ArgumentParser(description="Run crowd scenario with many moving obstacles.")
    parser.add_argument(
        "--model",
        type=str,
        default="di",
        choices=["di", "du", "uni"],
        help="Robot model alias (`di`, `du`, or `uni`).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="occlusion_cbf_qp",
        choices=CROWD_ALGO_CHOICES,
        help="Position controller algorithm.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=CROWD_BASELINE_CHOICES,
        help="Baseline alias. If provided, overrides --algo.",
    )
    parser.add_argument("--tf", type=float, default=100.0, help="Simulation final time [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for crowd generation.")
    parser.add_argument(
        "--idx",
        "--case-idx",
        dest="case_idx",
        type=int,
        default=None,
        help="Case index (1-based) for deterministic random scenario selection with fixed seed.",
    )
    parser.add_argument(
        "--n-rand",
        type=int,
        default=50,
        help=(
            "Random mode: number of moving obstacles. Forced-emergence mode: target number of non-occluder movers "
            "(hidden-emergence agents plus optional background clutter). Large forced occluders are controlled by "
            "--forced-events and are not counted in n-rand."
        ),
    )
    parser.add_argument("--no-rand-obs", action="store_true", help="Disable random moving obstacles.")
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument(
        "--crowd-mode",
        type=str,
        default="random",
        choices=["random", "forced_emergence"],
        help="Crowd scenario generator mode.",
    )
    parser.add_argument(
        "--forced-events",
        type=int,
        default=3,
        help="Forced-emergence mode: number of occluder/hidden-agent event pairs to generate.",
    )
    parser.add_argument(
        "--forced-bg-rand",
        type=int,
        default=None,
        help=(
            "Forced-emergence mode: explicit number of background random obstacles. "
            "Default uses a small clutter share and allocates the rest of n-rand to hidden emergence agents."
        ),
    )
    parser.add_argument(
        "--forced-hidden-speed",
        type=float,
        default=0.5,
        help="Forced-emergence mode: nominal hidden-agent speed magnitude.",
    )
    parser.add_argument(
        "--forced-occluder-radius-min",
        type=float,
        default=0.8,
        help="Forced-emergence mode: minimum occluder radius.",
    )
    parser.add_argument(
        "--forced-occluder-radius-max",
        type=float,
        default=1.0,
        help="Forced-emergence mode: maximum occluder radius.",
    )
    parser.add_argument(
        "--forced-validate-occlusion",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Forced-emergence mode: require hidden agent to be initially occluded during generation.",
    )
    parser.add_argument(
        "--forced-require-corridor-conflict",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Forced-emergence mode: require predicted ego-corridor conflict during generation.",
    )
    parser.add_argument("--du-min-speed-scale", type=float, default=None, help="Override backup_cbf.min_speed_scale_occ_du.")
    parser.add_argument("--du-k-turn-brake", type=float, default=None, help="Override backup_cbf.k_turn_brake_occ_du.")
    parser.add_argument("--du-k-a-p", type=float, default=None, help="Override backup_cbf.k_a_occ_du_p.")
    parser.add_argument("--du-k-a-d", type=float, default=None, help="Override backup_cbf.k_a_occ_du_d.")
    parser.add_argument("--du-reverse-enter-cos", type=float, default=None, help="Override backup_cbf.reverse_enter_cos_occ_du.")
    parser.add_argument("--du-reverse-exit-cos", type=float, default=None, help="Override backup_cbf.reverse_exit_cos_occ_du.")
    parser.add_argument("--du-reverse-min-scale", type=float, default=None, help="Override backup_cbf.reverse_min_scale_occ_du.")
    parser.add_argument("--uni-reverse-bias", type=float, default=None, help="Override backup_cbf.reverse_bias_occ_uni.")
    parser.add_argument("--uni-reverse-gate-angle", type=float, default=None, help="Override backup_cbf.reverse_speed_gate_angle_occ_uni.")
    parser.add_argument("--uni-reverse-gate-power", type=float, default=None, help="Override backup_cbf.reverse_speed_gate_power_occ_uni.")
    parser.add_argument("--uni-v-min-cmd-rev", type=float, default=None, help="Override backup_cbf.v_min_cmd_rev_occ_uni.")
    parser.add_argument("--occ-dt-backup", type=float, default=None, help="Override backup_cbf.dt_backup for occlusion backup rollout.")
    parser.add_argument(
        "--vref-mode-occ",
        type=str,
        choices=["soft", "strict"],
        default=None,
        help="Facet aggregation mode for UNI/DU occlusion backup v_ref.",
    )
    parser.add_argument(
        "--vref",
        type=str,
        choices=["default", "los"],
        default=None,
        help="Front-facet direction mode used when building occlusion backup v_target.",
    )
    parser.add_argument(
        "--occ-visible-scale",
        type=float,
        default=None,
        help=(
            "Visibility fraction threshold for occlusion filtering. "
            "Example: 0.5 keeps an obstacle occluded until roughly half of its disc is exposed; "
            "1.0 matches the current slightest-visible-is-visible behavior."
        ),
    )
    parser.add_argument(
        "--occ-version",
        type=str,
        choices=["v1", "v2"],
        default=None,
        help=(
            "Occlusion polygon construction. "
            "`v1` uses the tangent-chord front facet; "
            "`v2` moves the front facet toward the robot so the visible obstacle body is included."
        ),
    )
    parser.add_argument(
        "--occ-enable-visible-hocbf",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Occlusion-CBF only: also add visible-obstacle CBF/HOCBF rows in the stacked QP.",
    )
    parser.add_argument(
        "--oa-dynamic-occluders",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="OA-MPC extension: allow dynamic visible obstacles as occluders.",
    )
    parser.add_argument(
        "--oa-allow-solver-fallback",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="OA-MPC: if solver fails, use stop fallback instead of infeasible termination.",
    )
    parser.add_argument(
        "--oa-dsafe",
        type=float,
        default=None,
        help="OA-MPC: minimum safety distance used in projection constraints.",
    )
    parser.add_argument(
        "--oa-visible-reach-mode",
        type=str,
        choices=["worst_case", "constant_velocity"],
        default=None,
        help="OA-MPC: visible-agent reachable set mode.",
    )
    parser.add_argument(
        "--oa-use-nominal-tracking-cost",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="OA-MPC extension: use ||u-u_ref|| cost term.",
    )
    parser.add_argument("--oacp-dt-plan", type=float, default=None, help="Override oacp_mpc.dt_plan.")
    parser.add_argument("--oacp-Th", type=float, default=None, help="Override oacp_mpc.Th.")
    parser.add_argument("--oacp-N", type=int, default=None, help="Override oacp_mpc.N.")
    parser.add_argument("--oacp-n-shared", type=int, default=None, help="Override oacp_mpc.n_shared.")
    parser.add_argument("--oacp-risk-explore-scale", type=float, default=None, help="Override oacp_mpc.risk_explore_scale.")
    parser.add_argument("--oacp-risk-fallback-scale", type=float, default=None, help="Override oacp_mpc.risk_fallback_scale.")
    parser.add_argument("--oacp-explore-speed-scale", type=float, default=None, help="Override oacp_mpc.explore_speed_scale.")
    parser.add_argument("--oacp-fallback-speed-scale", type=float, default=None, help="Override oacp_mpc.fallback_speed_scale.")
    parser.add_argument(
        "--oacp-use-nominal-tracking-cost",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override oacp_mpc.use_nominal_tracking_cost.",
    )
    parser.add_argument(
        "--oacp-allow-solver-fallback",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override oacp_mpc.allow_solver_fallback.",
    )
    parser.add_argument(
        "--oacp-dynamic-occluders",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override oacp_mpc.dynamic_occluders.",
    )
    parser.add_argument(
        "--oacp-visible-reach-mode",
        type=str,
        choices=["constant_velocity", "worst_case"],
        default=None,
        help="Override oacp_mpc.visible_reach_mode.",
    )
    parser.add_argument(
        "--wmax",
        type=str,
        choices=["default", "pi"],
        default="default",
        help=(
            "Unicycle yaw-rate bound mode for OA-MPC comparison: "
            "`default` -> w_max=0.8, `pi` -> w_max=pi. "
            "Only affects OA-MPC Unicycle runs."
        ),
    )
    parser.add_argument(
        "--oa-dt",
        type=float,
        default=None,
        help=(
            "Override crowd simulation dt for OA-MPC runs only. "
            "Useful for fair true/false preset comparison (e.g., --oa-dt 0.05)."
        ),
    )
    parser.add_argument(
        "--static-occluders",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Force all crowd obstacles to static type-0 occluders (vx=vy=0).",
    )
    parser.add_argument(
        "--save_ani",
        "--save-ani",
        "--save-anim",
        "--save-animation",
        dest="save_anim",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Save animation frames/video. Accepts true/false or can be passed as a flag.",
    )
    args = parser.parse_args()

    pos_algo = resolve_baseline_alias(args.baseline, args.algo, CROWD_BASELINE_MAP)
    controller_type = {"pos": pos_algo}
    backup_cbf_overrides = {}
    if args.uni_reverse_bias is not None:
        backup_cbf_overrides["reverse_bias_occ_uni"] = float(args.uni_reverse_bias)
    if args.uni_reverse_gate_angle is not None:
        backup_cbf_overrides["reverse_speed_gate_angle_occ_uni"] = float(args.uni_reverse_gate_angle)
    if args.uni_reverse_gate_power is not None:
        backup_cbf_overrides["reverse_speed_gate_power_occ_uni"] = float(args.uni_reverse_gate_power)
    if args.uni_v_min_cmd_rev is not None:
        backup_cbf_overrides["v_min_cmd_rev_occ_uni"] = float(args.uni_v_min_cmd_rev)
    if args.occ_dt_backup is not None:
        backup_cbf_overrides["dt_backup"] = float(args.occ_dt_backup)
    if args.vref is not None:
        backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    robot_spec_overrides = {}
    oacp_cfg = {}
    if args.oacp_dt_plan is not None:
        oacp_cfg["dt_plan"] = float(args.oacp_dt_plan)
    if args.oacp_Th is not None:
        oacp_cfg["Th"] = float(args.oacp_Th)
    if args.oacp_N is not None:
        oacp_cfg["N"] = int(args.oacp_N)
    if args.oacp_n_shared is not None:
        oacp_cfg["n_shared"] = int(args.oacp_n_shared)
    if args.oacp_risk_explore_scale is not None:
        oacp_cfg["risk_explore_scale"] = float(args.oacp_risk_explore_scale)
    if args.oacp_risk_fallback_scale is not None:
        oacp_cfg["risk_fallback_scale"] = float(args.oacp_risk_fallback_scale)
    if args.oacp_explore_speed_scale is not None:
        oacp_cfg["explore_speed_scale"] = float(args.oacp_explore_speed_scale)
    if args.oacp_fallback_speed_scale is not None:
        oacp_cfg["fallback_speed_scale"] = float(args.oacp_fallback_speed_scale)
    if args.oacp_use_nominal_tracking_cost is not None:
        oacp_cfg["use_nominal_tracking_cost"] = bool(args.oacp_use_nominal_tracking_cost)
    if args.oacp_allow_solver_fallback is not None:
        oacp_cfg["allow_solver_fallback"] = bool(args.oacp_allow_solver_fallback)
    if args.oacp_dynamic_occluders is not None:
        oacp_cfg["dynamic_occluders"] = bool(args.oacp_dynamic_occluders)
    if args.oacp_visible_reach_mode is not None:
        oacp_cfg["visible_reach_mode"] = str(args.oacp_visible_reach_mode).strip().lower()
    if oacp_cfg:
        robot_spec_overrides["oacp_mpc"] = oacp_cfg
    run_crowd_scenario(
        controller_type=controller_type,
        model_key=args.model,
        show_animation=not args.disable_plot,
        save_animation=args.save_anim,
        tf=args.tf,
        seed=args.seed,
        case_idx=args.case_idx,
        rand_obs=(not args.no_rand_obs),
        n_rand=args.n_rand,
        du_min_speed_scale=args.du_min_speed_scale,
        du_k_turn_brake=args.du_k_turn_brake,
        du_k_a_p=args.du_k_a_p,
        du_k_a_d=args.du_k_a_d,
        du_reverse_enter_cos=args.du_reverse_enter_cos,
        du_reverse_exit_cos=args.du_reverse_exit_cos,
        du_reverse_min_scale=args.du_reverse_min_scale,
        vref_mode_occ=args.vref_mode_occ,
        vref_front_mode_occ=args.vref,
        occ_version=args.occ_version,
        occ_visible_scale=args.occ_visible_scale,
        occ_enable_visible_hocbf=args.occ_enable_visible_hocbf,
        oa_dynamic_occluders=args.oa_dynamic_occluders,
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        oa_dsafe=args.oa_dsafe,
        oa_visible_reach_mode=args.oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=args.oa_use_nominal_tracking_cost,
        oa_wmax=args.wmax,
        oa_dt=args.oa_dt,
        crowd_mode=args.crowd_mode,
        forced_events=args.forced_events,
        forced_bg_rand=args.forced_bg_rand,
        forced_hidden_speed=args.forced_hidden_speed,
        forced_occluder_radius_min=args.forced_occluder_radius_min,
        forced_occluder_radius_max=args.forced_occluder_radius_max,
        forced_validate_occlusion=args.forced_validate_occlusion,
        forced_require_corridor_conflict=args.forced_require_corridor_conflict,
        static_occluders=args.static_occluders,
        backup_cbf_overrides=(backup_cbf_overrides or None),
        robot_spec_overrides=(robot_spec_overrides or None),
    )


if __name__ == "__main__":
    main()
