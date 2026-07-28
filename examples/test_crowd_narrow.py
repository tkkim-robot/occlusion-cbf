"""
Legacy narrow-crowd scenario migrated from dynamic_env/main.py::single_agent_main.

Run:
    uv run python examples/test_crowd_narrow.py --model di
"""

import argparse

import numpy as np

try:
    from examples._baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        CROWD_PLANNER_LABELS,
        resolve_baseline_alias,
    )
    from examples._runtime import ensure_repo_root, load_local_occ_controller
except ImportError:
    from _baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        CROWD_PLANNER_LABELS,
        resolve_baseline_alias,
    )
    from _runtime import ensure_repo_root, load_local_occ_controller

ensure_repo_root()
LocalTrackingControllerDyn_OCC = load_local_occ_controller("crowd_narrow")
from position_control.ocbf.defaults import (
    OCBF_QP_FAILURE_FALLBACK_MODES,
    OCBF_ROLLOUT_MODES,
    OCBF_SELECTION_MODES,
    OCBF_TERMINAL_MODES,
    OCBF_TERMINAL_RESIDUAL_MODES,
    OCBF_VREF_FRONT_MODES,
    OCBF_VREF_SCENARIO_WEIGHT_MODES,
    OCBF_VREF_TRACKING_MODES,
    merge_shared_robot_parameters,
    merge_ocbf_best_parameters,
)
from base_control.utils import env, plotting

SMALL_DYN_SPEED_MIN = 0.3
SMALL_DYN_SPEED_MAX = 1.0
LEGACY_SMALL_DYN_SPEED_MAX = 0.5
LEGACY_RAND_OBS_SETTING = "v1"
CURRENT_RAND_OBS_SETTING = "v2"
DEFAULT_RAND_OBS_SETTING = CURRENT_RAND_OBS_SETTING


def _sample_small_dyn_speed(rng, speed_max=SMALL_DYN_SPEED_MAX, speed_min=SMALL_DYN_SPEED_MIN):
    lo = float(max(0.0, speed_min))
    hi = float(max(lo, speed_max))
    if hi <= lo + 1e-12:
        return hi
    return float(rng.uniform(lo, hi))


def _normalize_rand_obs_setting(rand_obs_setting):
    setting = str(rand_obs_setting).strip().lower()
    if setting not in {LEGACY_RAND_OBS_SETTING, CURRENT_RAND_OBS_SETTING}:
        raise ValueError(
            f"Unsupported rand_obs_setting `{rand_obs_setting}`. "
            f"Use `{LEGACY_RAND_OBS_SETTING}` or `{CURRENT_RAND_OBS_SETTING}`."
        )
    return setting


def _rand_obs_speed_window(*, static_occluders, rand_obs_setting, legacy_speed_max):
    setting = _normalize_rand_obs_setting(rand_obs_setting)
    if bool(static_occluders):
        return 0.0, 0.0
    if setting == LEGACY_RAND_OBS_SETTING:
        vmax = float(max(0.0, legacy_speed_max))
        return vmax, vmax
    return float(SMALL_DYN_SPEED_MAX), float(SMALL_DYN_SPEED_MIN)


def _sample_hidden_speed_for_setting(
    rng,
    *,
    forced_hidden_speed,
    rand_obs_setting,
    legacy_low,
    legacy_high,
):
    setting = _normalize_rand_obs_setting(rand_obs_setting)
    speed_nominal = float(max(0.0, forced_hidden_speed))
    if setting == LEGACY_RAND_OBS_SETTING:
        return speed_nominal * float(rng.uniform(legacy_low, legacy_high))
    return _sample_small_dyn_speed(
        rng,
        speed_max=min(SMALL_DYN_SPEED_MAX, speed_nominal),
    )


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
    """Apply ICRA-style joint hidden-world single-hypothesis defaults."""
    robot_spec["occlusion_types"] = [1]
    sr_cfg = robot_spec.setdefault("single_risk_mpc", {})
    # Use a planning grid aligned with the runtime DI integration step.
    # Keep the horizon short enough for dense crowd scenes.
    sr_cfg.setdefault("dt_plan", 0.05)
    sr_cfg.setdefault("Th", 1.0)
    sr_cfg.setdefault("N", 20)
    sr_cfg.setdefault("forward_only", False)
    sr_cfg.setdefault("max_active_occlusions", 2)
    sr_cfg.setdefault("max_occ_regions", 2)
    sr_cfg.setdefault("hypothesis_model", "direct_hidden_obstacle")
    # Use the current small hidden-obstacle upper bound as a conservative
    # single-world hidden-agent model in crowd-style emergence scenes.
    sr_cfg.setdefault("hidden_agent_radius", 0.4)
    sr_cfg.setdefault("hidden_spawn_clearance", 0.12)
    sr_cfg.setdefault("hidden_speed_scale", 1.0)
    sr_cfg.setdefault("active_selection_delta", 1.0)
    sr_cfg.setdefault("hidden_selection_mode", "joint_min_clearance_earliest")
    sr_cfg.setdefault("wguide", 3.5)
    sr_cfg.setdefault("wgoal", 3.5)
    sr_cfg.setdefault("wvel", 5.0)
    sr_cfg.setdefault("wacc", 1.8)
    sr_cfg.setdefault("lambda_w", 1.0)
    sr_cfg.setdefault("margin_obs", 0.05)


def _apply_control_tree_defaults(robot_spec):
    """Apply direct-hidden control-tree defaults with K-active occlusion world hypotheses."""
    robot_spec["occlusion_types"] = [1]
    ct_cfg = robot_spec.setdefault("control_tree_mpc", {})
    ct_cfg.setdefault("dt_plan", 0.05)
    ct_cfg.setdefault("Th", 1.0)
    ct_cfg.setdefault("N", 20)
    ct_cfg.setdefault("forward_only", False)
    if robot_spec.get("model") == "Unicycle2D":
        ct_cfg.setdefault(
            "v_plan_min",
            float(robot_spec.get("v_min", 0.0)),
        )
    else:
        ct_cfg.setdefault("v_plan_min", 0.0)
    ct_cfg.setdefault("n_split", 3)
    ct_cfg.setdefault("max_active_occlusions", 2)
    ct_cfg.setdefault("n_occ_hypotheses", 2)
    ct_cfg.setdefault("hypothesis_model", "direct_hidden_obstacle")
    # Use the current small hidden-obstacle upper bound as a conservative
    # branch model for the direct hidden-world hypotheses.
    ct_cfg.setdefault("hidden_agent_radius", 0.4)
    ct_cfg.setdefault("hidden_spawn_clearance", 0.12)
    ct_cfg.setdefault("hidden_speed_scale", 1.0)
    ct_cfg.setdefault("active_selection_delta", 1.0)
    ct_cfg.setdefault("max_branch_hidden_obs", 2)
    ct_cfg.setdefault("wgoal", 4.0)
    ct_cfg.setdefault("wvel", 12.0)
    ct_cfg.setdefault("wacc", 1.2)
    ct_cfg.setdefault("wtrack_shared", 0.4)
    ct_cfg.setdefault("wtrack_tail", 0.12)
    ct_cfg.setdefault("wlat_vel_di", 4.0)
    ct_cfg.setdefault("lambda_w", 1.0)
    ct_cfg.setdefault("margin_obs", 0.05)
    # Use the persistent joint CasADi backend by default so control-tree is
    # benchmarked with the same class of solver engineering already used by
    # the single-hypothesis baseline. Keep opti as an automatic fallback if
    # persistent setup fails, and enable dual warm-starts on the persistent
    # path to reduce per-step solve time without changing planner semantics.
    ct_cfg.setdefault("solver_backend", "joint_persistent")
    ct_cfg.setdefault("persistent_fallback_opti", True)
    ct_cfg.setdefault("warm_start_dual", True)
    ct_cfg.setdefault("max_iter", 150)
    ct_cfg.setdefault("solver_tol", 1e-4)
    ct_cfg.setdefault("solver_acceptable_tol", 1e-2)
    ct_cfg.setdefault("solver_acceptable_iter", 8)
    ct_cfg.setdefault("branch_zero_prob_reg", 1e-3)


def _apply_oacp_defaults(robot_spec):
    """Apply adapted occlusion-aware contingency planner defaults."""
    oacp_cfg = robot_spec.setdefault("oacp_mpc", {})
    # Centralized, ADMM-free OACP adaptation for crowd-style occlusion events.
    # Keep the dual-branch shared-prefix structure, but use SRQ-inspired risk
    # quantification, dynamic velocity boundaries, explicit phantom barriers,
    # and smooth Bezier reference scaffolds.
    oacp_cfg.setdefault("backend", "coupled_nlp")
    oacp_cfg.setdefault("dt_plan", 0.05)
    oacp_cfg.setdefault("Th", 1.0)
    oacp_cfg.setdefault("N", 20)
    oacp_cfg.setdefault("n_shared", 3)
    oacp_cfg.setdefault("max_active_occlusions", 2)
    oacp_cfg.setdefault("hidden_agent_radius", 0.4)
    oacp_cfg.setdefault("hidden_spawn_clearance", 0.12)
    oacp_cfg.setdefault("hidden_speed_scale", 1.0)
    oacp_cfg.setdefault("active_selection_delta", 1.0)
    oacp_cfg.setdefault("srq_confidence_z", 1.645)
    oacp_cfg.setdefault("srq_lane_width_min", 0.8)
    oacp_cfg.setdefault("srq_lane_width_max", 2.5)
    oacp_cfg.setdefault("cth_min", 0.0)
    oacp_cfg.setdefault("cth_max_explore", 0.85)
    oacp_cfg.setdefault("cth_max_fallback", 0.55)
    oacp_cfg.setdefault("v_occ_min_scale", 0.15)
    oacp_cfg.setdefault("v_occ_min_abs", 0.0)
    oacp_cfg.setdefault("barrier_alpha_start", 0.4)
    oacp_cfg.setdefault("barrier_alpha_end", 1.0)
    oacp_cfg.setdefault("ellipse_scale_x", 1.0)
    oacp_cfg.setdefault("ellipse_scale_y", 1.0)
    oacp_cfg.setdefault("ellipse_buffer_x", 0.05)
    oacp_cfg.setdefault("ellipse_buffer_y", 0.05)
    oacp_cfg.setdefault("use_bezier_reference", True)
    oacp_cfg.setdefault("bezier_ref_order", 10)
    oacp_cfg.setdefault("branch_switch_margin", 0.05)
    oacp_cfg.setdefault("allow_solver_fallback", False)
    oacp_cfg.setdefault("dynamic_occluders", True)
    oacp_cfg.setdefault("visible_reach_mode", "constant_velocity")
    oacp_cfg.setdefault("forward_only", False)
    oacp_cfg.setdefault("du_nonnegative_speed", True)
    oacp_cfg.setdefault("max_visible_obs", 30)
    oacp_cfg.setdefault("max_occ_scenarios", 30)
    robot_spec["occlusion_types"] = [0, 1] if bool(oacp_cfg.get("dynamic_occluders", True)) else [0]


def _apply_crowd_dynamic_obstacle_defaults(robot_spec):
    """
    Crowd-specific dynamic-obstacle behavior model.
    Keeps base random motion by default, and exposes optional extras:
    1) near-robot visible pedestrian sidestep bias
    2) occluded-agent emergence bias toward robot-forward corridor
    """
    dyn_cfg = robot_spec.setdefault("crowd_dyn_obs", {})

    # Base random walker noise.
    dyn_cfg.setdefault("heading_jitter_std", 0.01)
    dyn_cfg.setdefault("large_turn_prob", 0.05)
    dyn_cfg.setdefault("large_turn_std", 0.05)

    # Optional ego-dependent sidestep. Default off so obstacle motion does not
    # depend on the robot state unless explicitly enabled.
    dyn_cfg.setdefault("ped_aware_enable", False)
    dyn_cfg.setdefault("ped_aware_visible_only", True)

    dyn_cfg.setdefault("ped_aware_radius", 1.5)
    dyn_cfg.setdefault("ped_aware_approach_cos", 0.25)
    dyn_cfg.setdefault("ped_aware_turn_gain", 0.2)
    dyn_cfg.setdefault("ped_aware_side_weight", 0.6)
    dyn_cfg.setdefault("ped_aware_away_weight", 0.15)
    dyn_cfg.setdefault("ped_aware_noise_std", 0.03)
    dyn_cfg.setdefault("ped_aware_side_flip_prob", 0.02)

    # Occlusion-emergence: increase natural "hidden then appear" events.
    dyn_cfg.setdefault("occlusion_emergence_enable", False)
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
    dyn_cfg.setdefault("occluded_speed_boost_exact", True)
    dyn_cfg.setdefault("occluded_speed_boost_hold_time_s", 0.5)
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


def _build_random_crowd_scenario(*, case_seed, n_rand, rand_obs, static_occluders, rand_obs_setting):
    rand_obs_setting = _normalize_rand_obs_setting(rand_obs_setting)
    known_obs = np.empty((0, 8), dtype=float)
    obs_meta = []
    v_obs_max, v_obs_min = _rand_obs_speed_window(
        static_occluders=static_occluders,
        rand_obs_setting=rand_obs_setting,
        legacy_speed_max=LEGACY_SMALL_DYN_SPEED_MAX,
    )

    rand_rows, rand_meta = LocalTrackingControllerDyn_OCC.make_random_obstacles7(
        n_rand=int(n_rand),
        v_obs_max=v_obs_max,
        v_obs_min=v_obs_min,
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
        "rand_obs_setting": str(rand_obs_setting),
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
    rand_obs_setting,
):
    rand_obs_setting = _normalize_rand_obs_setting(rand_obs_setting)
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
            hidden_radius = float(rng.uniform(0.3, 0.4))
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

            hidden_speed = _sample_hidden_speed_for_setting(
                rng,
                forced_hidden_speed=forced_hidden_speed,
                rand_obs_setting=rand_obs_setting,
                legacy_low=0.88,
                legacy_high=1.0,
            )
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
                        hidden_speed = _sample_hidden_speed_for_setting(
                            rng,
                            forced_hidden_speed=forced_hidden_speed,
                            rand_obs_setting=rand_obs_setting,
                            legacy_low=0.9,
                            legacy_high=1.0,
                        )
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
        bg_v_obs_max, bg_v_obs_min = _rand_obs_speed_window(
            static_occluders=static_occluders,
            rand_obs_setting=rand_obs_setting,
            legacy_speed_max=LEGACY_SMALL_DYN_SPEED_MAX,
        )
        keep_rows = []
        keep_meta = []
        batch_id = 0
        while len(keep_rows) < bg_target and batch_id < 16:
            sample_target = max((bg_target - len(keep_rows)) * 4, bg_target)
            extra_rows, extra_meta = LocalTrackingControllerDyn_OCC.make_random_obstacles7(
                n_rand=int(sample_target),
                v_obs_max=bg_v_obs_max,
                v_obs_min=bg_v_obs_min,
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
        "rand_obs_setting": str(rand_obs_setting),
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


def _prepare_crowd_runtime(
    controller_type=None,
    model_key="di",
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
    occ_visible_scale=None,
    occ_enable_visible_hocbf=None,
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
    rand_obs_setting=DEFAULT_RAND_OBS_SETTING,
    static_occluders=False,
    backup_cbf_overrides=None,
    robot_spec_overrides=None,
    waypoints_override=None,
    env_width_override=None,
    env_height_override=None,
    known_obs_override=None,
    obs_meta_override=None,
    scenario_diag_override=None,
    scenario_name="Crowd Narrow",
):
    if controller_type is None:
        controller_type = {"pos": "occlusion_cbf_qp"}

    requested_pos_name = str(controller_type.get("pos", "")).strip().lower()
    is_oa_mpc = requested_pos_name == "oa_mpc"
    is_ocbf = requested_pos_name in {"occlusion_cbf", "occlusion_cbf_qp"}

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

    case_seed = _compute_case_seed(seed, case_idx)
    crowd_mode = str(crowd_mode).strip().lower()
    if crowd_mode not in {"random", "forced_emergence"}:
        raise ValueError(f"Unsupported crowd_mode `{crowd_mode}`. Use `random` or `forced_emergence`.")
    rand_obs_setting = _normalize_rand_obs_setting(rand_obs_setting)

    if known_obs_override is not None:
        known_obs = np.asarray(known_obs_override, dtype=float)
        if known_obs.ndim != 2 or known_obs.shape[1] != 8:
            raise ValueError("known_obs_override must have shape (N, 8).")
        obs_meta = [] if obs_meta_override is None else list(obs_meta_override)
        scenario_diag = {} if scenario_diag_override is None else dict(scenario_diag_override)
        scenario_diag.setdefault("crowd_mode", crowd_mode)
        scenario_diag.setdefault("rand_obs_setting", rand_obs_setting)
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
                rand_obs_setting=rand_obs_setting,
            )
        else:
            known_obs, obs_meta, scenario_diag = _build_random_crowd_scenario(
                case_seed=case_seed,
                n_rand=n_rand,
                rand_obs=rand_obs,
                static_occluders=static_occluders,
                rand_obs_setting=rand_obs_setting,
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
    allow_reverse_uni_raw = robot_spec_overrides.pop("_uni_allow_reverse", None)
    forward_only_uni = bool(robot_spec_overrides.pop("_uni_forward_only", False))
    explicit_uni_v_min = "v_min" in robot_spec_overrides
    if model == "DoubleIntegrator2D":
        di_backup_cfg = {
            "T_horizon": 0.5,
            "vref_scenario_softmax_kappa": 10.0,
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
            "dynamic_obs_types": [1],
        }
    elif model == "Unicycle2D":
        uni_backup_cfg = {
            "T_horizon": 0.5,
            "vref_scenario_softmax_kappa": 1.0,
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
        robot_spec = {
            "model": "Unicycle2D",
            "v_max": uni_vmax,
            "v_min": 0.0,
            "w_max": uni_wmax,
            "radius": 0.25,
            "debug_backup_qp": False,
            "sensing_range": 10.0,
            "fov_angle": 360.0,
            "backup_cbf": uni_backup_cfg,
            "show_backup_rollout": True,
            "backup_rollout_every": 1,
            "use_occ": True,
            "dynamic_obs_types": [1],
        }
    else:
        raise ValueError(f"Unsupported resolved model `{model}`.")

    # Apply only the model-wide values explicitly marked as shared in the
    # committed profiles. OCBF-only backup and barrier parameters stay isolated.
    robot_spec = merge_shared_robot_parameters(
        model,
        robot_defaults=robot_spec,
    )

    if is_ocbf:
        tuned_backup, tuned_robot = merge_ocbf_best_parameters(
            model,
            backup_defaults=robot_spec["backup_cbf"],
            robot_defaults=robot_spec,
            backup_overrides=backup_cbf_overrides,
        )
        robot_spec = tuned_robot
        robot_spec["backup_cbf"] = tuned_backup

    if robot_spec_overrides:
        robot_spec.update(robot_spec_overrides)
    if model == "Unicycle2D" and not explicit_uni_v_min:
        if forward_only_uni or allow_reverse_uni_raw is False:
            robot_spec["v_min"] = 0.0
        elif allow_reverse_uni_raw is True:
            robot_spec["v_min"] = -abs(float(robot_spec.get("v_max", 1.0)))

    _apply_crowd_dynamic_obstacle_defaults(robot_spec)
    if crowd_mode == "forced_emergence":
        robot_spec["v_adv_max_occ"] = 1.0
        dyn_cfg = robot_spec.setdefault("crowd_dyn_obs", {})
        dyn_cfg["occluded_speed_boost_enable"] = True
        dyn_cfg["occluded_speed_boost_vmax"] = float(robot_spec.get("v_obs_max", robot_spec["v_adv_max_occ"]))
        dyn_cfg["occluded_speed_boost_exact"] = True
        dyn_cfg["occluded_speed_boost_fov_only"] = True
        dyn_cfg["occluded_speed_boost_on_hysteresis_steps"] = 1
        dyn_cfg["occluded_speed_boost_hold_time_s"] = 0.5
        dyn_cfg["occluded_speed_boost_off_hysteresis_steps"] = int(np.ceil(0.5 / max(float(dt), 1e-9))) + 1

    if occ_visible_scale is not None:
        vis_scale = float(occ_visible_scale)
        if (not np.isfinite(vis_scale)) or vis_scale <= 0.0:
            raise ValueError(
                f"Invalid --occ-visible-scale: {occ_visible_scale}. It must be a positive finite value."
            )
        robot_spec["occ_visible_scale"] = float(vis_scale)
    if occ_enable_visible_hocbf is not None:
        robot_spec["enable_visible_hocbf_in_occ"] = bool(occ_enable_visible_hocbf)

    if str(controller_type.get("pos", "")).strip().lower() == "oa_mpc":
        oa_cfg = robot_spec.setdefault("oa_mpc", {})
        oa_cfg["paper_mode"] = True
        oa_cfg.setdefault("N", 10)
        oa_cfg.setdefault("auto_scale_N_with_dt", True)
        oa_cfg.setdefault("paper_horizon_time", 1.0)
        if oa_dsafe is None:
            oa_cfg["dsafe"] = float(oa_cfg.get("dsafe", 0.5))
        else:
            oa_cfg["dsafe"] = float(oa_dsafe)
        oa_cfg["dynamic_occluders"] = True
        # Keep visible dynamic obstacles in the same no-prediction,
        # forward-reachable-set spirit used by the OA-MPC paper.
        oa_cfg["visible_reach_mode"] = "worst_case"
        oa_cfg.setdefault("di_terminal_stop_mode", "brake_reachable")
        robot_spec["occlusion_types"] = [0, 1]
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

    if model == "Unicycle2D" and (forward_only_uni or allow_reverse_uni_raw is False):
        for cfg_name in ("single_risk_mpc", "control_tree_mpc", "oacp_mpc"):
            if cfg_name in robot_spec:
                robot_spec[cfg_name]["forward_only"] = True
        if "control_tree_mpc" in robot_spec:
            robot_spec["control_tree_mpc"]["v_plan_min"] = max(
                0.0, float(robot_spec["control_tree_mpc"].get("v_plan_min", 0.0))
            )

    planner_label_map = dict(CROWD_PLANNER_LABELS)
    if model == "Unicycle2D":
        planner_label_map["oa_mpc"] = f"OA-MPC (wmax={str(oa_wmax).strip().lower()})"
    else:
        planner_label_map["oa_mpc"] = "OA-MPC"
    planner_label = planner_label_map.get(pos_name, str(controller_type.get("pos", "")).strip())
    model_label_map = {
        "DoubleIntegrator2D": "Double Integrator2D",
        "DynamicUnicycle2D": "Dynamic Unicycle2D",
        "Unicycle2D": "Unicycle2D",
    }
    model_label = model_label_map.get(model, model)
    idx_suffix = f" | idx {int(case_idx)}" if case_idx is not None else ""
    figure_title = f"{scenario_name} | {planner_label} | {model_label}{idx_suffix}"

    return {
        "controller_type": dict(controller_type),
        "model": model,
        "dt": float(dt),
        "tf": float(tf),
        "case_seed": int(case_seed),
        "waypoints": np.asarray(waypoints, dtype=np.float64),
        "known_obs": np.asarray(known_obs, dtype=float),
        "obs_meta": list(obs_meta),
        "scenario_diag": dict(scenario_diag),
        "env_width": float(env_width),
        "env_height": float(env_height),
        "robot_spec": robot_spec,
        "planner_label": str(planner_label),
        "model_label": str(model_label),
        "figure_title": str(figure_title),
        "scenario_name": str(scenario_name),
        "crowd_mode": str(crowd_mode),
        "pos_name": str(pos_name),
    }


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
    occ_visible_scale=None,
    occ_enable_visible_hocbf=None,
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
    rand_obs_setting=DEFAULT_RAND_OBS_SETTING,
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
    tracking_view_enable=False,
    tracking_view_window_size=5.0,
    scenario_name="Crowd Narrow",
    hide_env_boundary=False,
):
    runtime = _prepare_crowd_runtime(
        controller_type=controller_type,
        model_key=model_key,
        tf=tf,
        seed=seed,
        case_idx=case_idx,
        rand_obs=rand_obs,
        n_rand=n_rand,
        du_min_speed_scale=du_min_speed_scale,
        du_k_turn_brake=du_k_turn_brake,
        du_k_a_p=du_k_a_p,
        du_k_a_d=du_k_a_d,
        du_reverse_enter_cos=du_reverse_enter_cos,
        du_reverse_exit_cos=du_reverse_exit_cos,
        du_reverse_min_scale=du_reverse_min_scale,
        vref_mode_occ=vref_mode_occ,
        vref_front_mode_occ=vref_front_mode_occ,
        occ_visible_scale=occ_visible_scale,
        occ_enable_visible_hocbf=occ_enable_visible_hocbf,
        oa_dynamic_occluders=oa_dynamic_occluders,
        oa_allow_solver_fallback=oa_allow_solver_fallback,
        oa_dsafe=oa_dsafe,
        oa_visible_reach_mode=oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=oa_use_nominal_tracking_cost,
        oa_wmax=oa_wmax,
        oa_dt=oa_dt,
        crowd_mode=crowd_mode,
        forced_events=forced_events,
        forced_bg_rand=forced_bg_rand,
        forced_hidden_speed=forced_hidden_speed,
        forced_occluder_radius_min=forced_occluder_radius_min,
        forced_occluder_radius_max=forced_occluder_radius_max,
        forced_validate_occlusion=forced_validate_occlusion,
        forced_require_corridor_conflict=forced_require_corridor_conflict,
        rand_obs_setting=rand_obs_setting,
        static_occluders=static_occluders,
        backup_cbf_overrides=backup_cbf_overrides,
        robot_spec_overrides=robot_spec_overrides,
        waypoints_override=waypoints_override,
        env_width_override=env_width_override,
        env_height_override=env_height_override,
        known_obs_override=known_obs_override,
        obs_meta_override=obs_meta_override,
        scenario_diag_override=scenario_diag_override,
        scenario_name=scenario_name,
    )

    controller_type = runtime["controller_type"]
    model = runtime["model"]
    dt = runtime["dt"]
    case_seed = runtime["case_seed"]
    waypoints = runtime["waypoints"]
    known_obs = runtime["known_obs"]
    obs_meta = runtime["obs_meta"]
    scenario_diag = runtime["scenario_diag"]
    env_width = runtime["env_width"]
    env_height = runtime["env_height"]
    robot_spec = runtime["robot_spec"]
    figure_title = runtime["figure_title"]
    crowd_mode = runtime["crowd_mode"]

    x_init = waypoints[0]

    if show_animation or save_animation:
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
        if bool(hide_env_boundary):
            plot_handler.obs_bound = []
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
    intervention_v_abs = []
    intervention_v_sq = []
    intervention_w_abs = []
    intervention_w_sq = []
    intervention_active_steps = 0
    terminal_slack_l1_vals = []
    terminal_slack_max_vals = []
    terminal_slack_active_steps = 0
    occ_vref_unexpanded_margin_vals = []
    occ_vref_max_weight_vals = []
    total_steps = 0
    nominal_speed = 0.8
    final_controller_profile = None

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
            final_controller_profile = dict(profile)
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
            occ_pm = profile.get("occ_vref_avg_unexpanded_margin", None)
            occ_mw = profile.get("occ_vref_max_softmax_weight", None)
            if occ_pm is not None:
                occ_vref_unexpanded_margin_vals.append(float(occ_pm))
            if occ_mw is not None:
                occ_vref_max_weight_vals.append(float(occ_mw))
        if pos_controller is not None:
            u_cmd = getattr(pos_controller, "last_u", None)
            u_nom = getattr(pos_controller, "last_u_ref", None)
            term_slack_l1 = getattr(pos_controller, "last_terminal_slack_l1", None)
            term_slack_max = getattr(pos_controller, "last_terminal_slack_max", None)
            term_slack_active_count = getattr(pos_controller, "last_terminal_slack_active_count", None)
            if u_cmd is not None and u_nom is not None:
                try:
                    uc = np.asarray(u_cmd, dtype=float).reshape(-1)
                    un = np.asarray(u_nom, dtype=float).reshape(-1)
                    m = min(len(uc), len(un))
                    if m > 0:
                        du = uc[:m] - un[:m]
                        val = float(np.sum(du ** 2))
                        if np.isfinite(val):
                            intervention_l2_sq.append(val)
                            tol = float(robot_spec.get("intervention_tol", 1e-3))
                            if val > (tol * tol):
                                intervention_active_steps += 1
                        if m >= 1:
                            dv_abs = float(abs(du[0]))
                            dv_sq = float(du[0] ** 2)
                            if np.isfinite(dv_abs):
                                intervention_v_abs.append(dv_abs)
                            if np.isfinite(dv_sq):
                                intervention_v_sq.append(dv_sq)
                        if m >= 2:
                            dw_abs = float(abs(du[1]))
                            dw_sq = float(du[1] ** 2)
                            if np.isfinite(dw_abs):
                                intervention_w_abs.append(dw_abs)
                            if np.isfinite(dw_sq):
                                intervention_w_sq.append(dw_sq)
                except Exception:
                    pass
            if term_slack_l1 is not None and np.isfinite(term_slack_l1):
                terminal_slack_l1_vals.append(float(term_slack_l1))
            if term_slack_max is not None and np.isfinite(term_slack_max):
                terminal_slack_max_vals.append(float(term_slack_max))
            try:
                if int(term_slack_active_count or 0) > 0:
                    terminal_slack_active_steps += 1
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

    avg_compute_time_ms = (None if len(compute_ms) == 0 else float(np.mean(compute_ms)))

    tracking_controller.export_video()
    if show_animation or save_animation:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.close(fig)

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
        # Canonical benchmark timing metric: full controller compute time per
        # step, excluding plotting. Keep the old `avg_solve_time_ms` alias for
        # compatibility with existing analysis scripts.
        "avg_compute_time_ms": avg_compute_time_ms,
        "avg_solve_time_ms": avg_compute_time_ms,
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
        "avg_control_intervention_v_abs": (
            None if len(intervention_v_abs) == 0 else float(np.mean(intervention_v_abs))
        ),
        "avg_control_intervention_v_sq": (
            None if len(intervention_v_sq) == 0 else float(np.mean(intervention_v_sq))
        ),
        "avg_control_intervention_w_abs": (
            None if len(intervention_w_abs) == 0 else float(np.mean(intervention_w_abs))
        ),
        "avg_control_intervention_w_sq": (
            None if len(intervention_w_sq) == 0 else float(np.mean(intervention_w_sq))
        ),
        "avg_terminal_slack_l1": (
            None if len(terminal_slack_l1_vals) == 0 else float(np.mean(terminal_slack_l1_vals))
        ),
        "avg_terminal_slack_max": (
            None if len(terminal_slack_max_vals) == 0 else float(np.mean(terminal_slack_max_vals))
        ),
        "terminal_slack_active_steps": int(terminal_slack_active_steps),
        "terminal_slack_active_ratio": float(terminal_slack_active_steps) / float(max(1, steps_executed)),
        "avg_occ_vref_unexpanded_margin": (
            None if len(occ_vref_unexpanded_margin_vals) == 0 else float(np.mean(occ_vref_unexpanded_margin_vals))
        ),
        "avg_occ_vref_max_softmax_weight": (
            None if len(occ_vref_max_weight_vals) == 0 else float(np.mean(occ_vref_max_weight_vals))
        ),
        "intervention_active_steps": int(intervention_active_steps),
        "intervention_active_ratio": float(intervention_active_ratio),
        "selected_branch_counts": selected_branch_counts,
        "final_controller_profile": final_controller_profile,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the legacy narrow-crowd scenario.")
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
        "--rand-obs-setting",
        type=str,
        default=DEFAULT_RAND_OBS_SETTING,
        choices=[LEGACY_RAND_OBS_SETTING, CURRENT_RAND_OBS_SETTING],
        help=(
            "Random-obstacle generator preset. "
            "`v1` reproduces the last committed fixed-speed crowd generator; "
            "`v2` uses the current distributed-speed generator."
        ),
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
    parser.add_argument("--uni-allow-reverse", type=_str2bool, nargs="?", const=True, default=None, help="Explicitly allow Uni reverse by setting v_min=-v_max unless --uni-v-min is given. Forward-only is the default.")
    parser.add_argument("--uni-forward-only", type=_str2bool, nargs="?", const=True, default=False, help="Force Unicycle2D forward-only by setting v_min=0 unless --uni-v-min is given.")
    parser.add_argument("--uni-v-min", type=float, default=None, help="Override Unicycle2D input lower speed bound.")
    parser.add_argument("--uni-vref-tracking-mode", type=str, choices=["gated", "projected"], default=None)
    parser.add_argument("--uni-k-theta-p", type=float, default=None, help="Override backup_cbf.k_theta_occ_uni_p.")
    parser.add_argument("--uni-k-theta-d", type=float, default=None, help="Override backup_cbf.k_theta_occ_uni_d.")
    parser.add_argument("--uni-k-v-p", type=float, default=None, help="Override backup_cbf.k_v_occ_uni_p.")
    parser.add_argument("--uni-k-v-d", type=float, default=None, help="Override backup_cbf.k_v_occ_uni_d.")
    parser.add_argument("--uni-turn-boost", type=float, default=None, help="Override backup_cbf.k_turn_boost_occ_uni.")
    parser.add_argument("--uni-turn-boost-angle", type=float, default=None, help="Override backup_cbf.turn_boost_angle_occ_uni.")
    parser.add_argument("--uni-v-min-cmd", type=float, default=None, help="Override backup_cbf.v_min_occ_uni.")
    parser.add_argument("--uni-turn-crawl-speed", type=float, default=None, help="Minimum forward crawl speed while turning toward OCBF v_ref.")
    parser.add_argument("--uni-turn-crawl-angle", type=float, default=None, help="Max heading error [rad] where OCBF turn-crawl is allowed.")
    parser.add_argument("--occ-dt-backup", type=float, default=None, help="Override backup_cbf.dt_backup for occlusion backup rollout.")
    parser.add_argument("--occ-t-horizon", type=float, default=None, help="Override backup_cbf.T_horizon for occlusion backup rollout.")
    parser.add_argument("--occ-rho-T", type=str, default=None, help="Override backup_cbf.rho_T for terminal occlusion backup constraint. Accepts a float or 'auto'.")
    parser.add_argument("--occ-k-p", type=float, default=None, help="Override backup_cbf.k_p_occ_di for DI occlusion backup controller.")
    parser.add_argument("--occ-k-d", type=float, default=None, help="Override backup_cbf.k_d_occ_di for DI occlusion backup controller.")
    parser.add_argument(
        "--occ-kappa",
        type=float,
        default=None,
        help="Override the OCBF barrier smoothing kappa. Default is fixed at 10.0; use only for ablations.",
    )
    parser.add_argument(
        "--occ-vref-scenario-softmax-kappa",
        type=float,
        default=None,
        help="Override backup_cbf.vref_scenario_softmax_kappa for occlusion-CBF front-speed scenario weighting.",
    )
    parser.add_argument(
        "--occ-vref-scenario-weight-mode",
        type=str,
        choices=OCBF_VREF_SCENARIO_WEIGHT_MODES,
        default=None,
        help=(
            "Override backup_cbf.vref_scenario_weight_mode for OCBF scenario blending. "
            "barrier_expand scores scenarios with rollout-expanded margins; "
            "barrier_unexpand scores them with unexpanded current geometry margins."
        ),
    )
    parser.add_argument(
        "--occ-max-active-occlusions",
        type=int,
        default=None,
        help="Limit occlusion-CBF to the top-K active occlusion scenarios. 0 keeps all scenarios.",
    )
    parser.add_argument(
        "--occ-selection-mode",
        type=str,
        choices=OCBF_SELECTION_MODES,
        default=None,
        help="Occlusion-CBF active occlusion selection score.",
    )
    parser.add_argument(
        "--occ-rollout-mode",
        type=str,
        choices=OCBF_ROLLOUT_MODES,
        default=None,
        help="Override backup_cbf.occ_rollout_mode for occlusion backup rollout construction.",
    )
    parser.add_argument(
        "--occ-terminal-slack-weight",
        type=float,
        default=None,
        help="Override backup_cbf.terminal_slack_weight for terminal-set rows only.",
    )
    parser.add_argument(
        "--occ-terminal-slack-max",
        type=float,
        default=None,
        help="Override backup_cbf.terminal_slack_max for terminal-set rows only.",
    )
    parser.add_argument("--occ-obs-hocbf-slack-max", type=float, default=None)
    parser.add_argument("--occ-rollout-slack-max", type=float, default=None)
    parser.add_argument("--occ-terminal-mode", type=str, choices=OCBF_TERMINAL_MODES, default=None)
    parser.add_argument("--occ-terminal-active-count", type=int, default=None)
    parser.add_argument(
        "--occ-terminal-residual-mode",
        type=str,
        choices=OCBF_TERMINAL_RESIDUAL_MODES,
        default=None,
    )
    parser.add_argument("--occ-terminal-visibility-reaction-margin", type=float, default=None)
    parser.add_argument(
        "--occ-qp-failure-fallback-mode",
        type=str,
        choices=OCBF_QP_FAILURE_FALLBACK_MODES,
        default=None,
    )
    parser.add_argument(
        "--vref-mode-occ",
        type=str,
        choices=OCBF_VREF_TRACKING_MODES,
        default=None,
        help="Facet aggregation mode for UNI/DU occlusion backup v_ref.",
    )
    parser.add_argument(
        "--vref",
        type=str,
        choices=OCBF_VREF_FRONT_MODES,
        default=None,
        help=(
            "Front-facet direction mode for occlusion backup v_target. "
            "Internal default is `los`; `default` keeps the fixed polygon normal."
        ),
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
        "--occ-enable-visible-hocbf",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help=(
            "Occlusion-CBF only: override whether visible-obstacle CBF/HOCBF rows "
            "are added. The tuned YAML parameter is used when omitted."
        ),
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
    parser.add_argument("--oacp-max-visible-obs", type=int, default=None, help="Override oacp_mpc.max_visible_obs.")
    parser.add_argument("--oacp-max-occ-scenarios", type=int, default=None, help="Override oacp_mpc.max_occ_scenarios.")
    parser.add_argument("--oacp-max-active-occlusions", type=int, default=None, help="Override oacp_mpc.max_active_occlusions.")
    parser.add_argument("--oacp-margin-obs", type=float, default=None, help="Override oacp_mpc.margin_obs.")
    parser.add_argument("--oacp-v-ref-default", type=float, default=None, help="Override oacp_mpc.v_ref_default.")
    parser.add_argument("--oacp-hidden-agent-radius", type=float, default=None, help="Override oacp_mpc.hidden_agent_radius.")
    parser.add_argument("--oacp-hidden-spawn-clearance", type=float, default=None, help="Override oacp_mpc.hidden_spawn_clearance.")
    parser.add_argument("--oacp-hidden-speed", type=float, default=None, help="Override oacp_mpc.hidden_speed.")
    parser.add_argument("--oacp-hidden-speed-scale", type=float, default=None, help="Override oacp_mpc.hidden_speed_scale.")
    parser.add_argument("--oacp-active-selection-delta", type=float, default=None, help="Override oacp_mpc.active_selection_delta.")
    parser.add_argument("--oacp-srq-confidence-z", type=float, default=None, help="Override oacp_mpc.srq_confidence_z.")
    parser.add_argument("--oacp-srq-lane-width-min", type=float, default=None, help="Override oacp_mpc.srq_lane_width_min.")
    parser.add_argument("--oacp-srq-lane-width-max", type=float, default=None, help="Override oacp_mpc.srq_lane_width_max.")
    parser.add_argument("--oacp-cth-min", type=float, default=None, help="Override oacp_mpc.cth_min.")
    parser.add_argument("--oacp-cth-max-explore", type=float, default=None, help="Override oacp_mpc.cth_max_explore.")
    parser.add_argument("--oacp-cth-max-fallback", type=float, default=None, help="Override oacp_mpc.cth_max_fallback.")
    parser.add_argument("--oacp-v-occ-min-scale", type=float, default=None, help="Override oacp_mpc.v_occ_min_scale.")
    parser.add_argument("--oacp-v-occ-min-abs", type=float, default=None, help="Override oacp_mpc.v_occ_min_abs.")
    parser.add_argument("--oacp-barrier-alpha-start", type=float, default=None, help="Override oacp_mpc.barrier_alpha_start.")
    parser.add_argument("--oacp-barrier-alpha-end", type=float, default=None, help="Override oacp_mpc.barrier_alpha_end.")
    parser.add_argument("--oacp-ellipse-scale-x", type=float, default=None, help="Override oacp_mpc.ellipse_scale_x.")
    parser.add_argument("--oacp-ellipse-scale-y", type=float, default=None, help="Override oacp_mpc.ellipse_scale_y.")
    parser.add_argument("--oacp-ellipse-buffer-x", type=float, default=None, help="Override oacp_mpc.ellipse_buffer_x.")
    parser.add_argument("--oacp-ellipse-buffer-y", type=float, default=None, help="Override oacp_mpc.ellipse_buffer_y.")
    parser.add_argument(
        "--oacp-use-bezier-reference",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override oacp_mpc.use_bezier_reference.",
    )
    parser.add_argument("--oacp-bezier-ref-order", type=int, default=None, help="Override oacp_mpc.bezier_ref_order.")
    parser.add_argument("--oacp-branch-switch-margin", type=float, default=None, help="Override oacp_mpc.branch_switch_margin.")
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
    args = parser.parse_args(argv)

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
    if args.uni_vref_tracking_mode is not None:
        backup_cbf_overrides["vref_tracking_mode_occ_uni"] = str(args.uni_vref_tracking_mode).strip().lower()
    if args.uni_k_theta_p is not None:
        backup_cbf_overrides["k_theta_occ_uni_p"] = float(args.uni_k_theta_p)
    if args.uni_k_theta_d is not None:
        backup_cbf_overrides["k_theta_occ_uni_d"] = float(args.uni_k_theta_d)
    if args.uni_k_v_p is not None:
        backup_cbf_overrides["k_v_occ_uni_p"] = float(args.uni_k_v_p)
    if args.uni_k_v_d is not None:
        backup_cbf_overrides["k_v_occ_uni_d"] = float(args.uni_k_v_d)
    if args.uni_turn_boost is not None:
        backup_cbf_overrides["k_turn_boost_occ_uni"] = float(args.uni_turn_boost)
    if args.uni_turn_boost_angle is not None:
        backup_cbf_overrides["turn_boost_angle_occ_uni"] = float(args.uni_turn_boost_angle)
    if args.uni_v_min_cmd is not None:
        backup_cbf_overrides["v_min_occ_uni"] = float(args.uni_v_min_cmd)
    if args.uni_turn_crawl_speed is not None:
        backup_cbf_overrides["turn_crawl_speed_occ_uni"] = float(args.uni_turn_crawl_speed)
    if args.uni_turn_crawl_angle is not None:
        backup_cbf_overrides["turn_crawl_angle_occ_uni"] = float(args.uni_turn_crawl_angle)
    if args.occ_dt_backup is not None:
        backup_cbf_overrides["dt_backup"] = float(args.occ_dt_backup)
    if args.occ_t_horizon is not None:
        backup_cbf_overrides["T_horizon"] = float(args.occ_t_horizon)
    if args.occ_rho_T is not None:
        rho_raw = str(args.occ_rho_T).strip()
        backup_cbf_overrides["rho_T"] = "auto" if rho_raw.lower() == "auto" else float(rho_raw)
    if args.occ_k_p is not None:
        backup_cbf_overrides["k_p_occ_di"] = float(args.occ_k_p)
    if args.occ_k_d is not None:
        backup_cbf_overrides["k_d_occ_di"] = float(args.occ_k_d)
    if args.occ_vref_scenario_softmax_kappa is not None:
        backup_cbf_overrides["vref_scenario_softmax_kappa"] = float(args.occ_vref_scenario_softmax_kappa)
    if args.occ_vref_scenario_weight_mode is not None:
        backup_cbf_overrides["vref_scenario_weight_mode"] = str(args.occ_vref_scenario_weight_mode).strip().lower()
    if args.occ_max_active_occlusions is not None:
        backup_cbf_overrides["max_active_occlusions"] = int(args.occ_max_active_occlusions)
    if args.occ_selection_mode is not None:
        backup_cbf_overrides["occ_selection_mode"] = str(args.occ_selection_mode).strip().lower()
    if args.occ_rollout_mode is not None:
        backup_cbf_overrides["occ_rollout_mode"] = str(args.occ_rollout_mode).strip().lower()
    if args.occ_terminal_slack_weight is not None:
        backup_cbf_overrides["terminal_slack_weight"] = float(args.occ_terminal_slack_weight)
    if args.occ_terminal_slack_max is not None:
        backup_cbf_overrides["terminal_slack_max"] = float(args.occ_terminal_slack_max)
    if args.occ_obs_hocbf_slack_max is not None:
        backup_cbf_overrides["obs_hocbf_slack_max"] = float(args.occ_obs_hocbf_slack_max)
    if args.occ_rollout_slack_max is not None:
        backup_cbf_overrides["occ_rollout_slack_max"] = float(args.occ_rollout_slack_max)
    if args.occ_terminal_mode is not None:
        backup_cbf_overrides["terminal_mode"] = str(args.occ_terminal_mode).strip().lower()
    if args.occ_terminal_active_count is not None:
        backup_cbf_overrides["terminal_active_count"] = int(args.occ_terminal_active_count)
    if args.occ_terminal_residual_mode is not None:
        backup_cbf_overrides["terminal_residual_mode"] = str(args.occ_terminal_residual_mode).strip().lower()
    if args.occ_terminal_visibility_reaction_margin is not None:
        backup_cbf_overrides["terminal_visibility_reaction_margin"] = float(
            args.occ_terminal_visibility_reaction_margin
        )
    if args.occ_qp_failure_fallback_mode is not None:
        backup_cbf_overrides["qp_failure_fallback_mode"] = str(args.occ_qp_failure_fallback_mode).strip().lower()
    if args.vref is not None:
        backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    robot_spec_overrides = {}
    if args.uni_allow_reverse is not None:
        robot_spec_overrides["_uni_allow_reverse"] = bool(args.uni_allow_reverse)
    if args.uni_forward_only:
        robot_spec_overrides["_uni_forward_only"] = True
    if args.uni_v_min is not None:
        robot_spec_overrides["v_min"] = float(args.uni_v_min)
    if args.occ_kappa is not None:
        robot_spec_overrides["occ_kappa"] = float(args.occ_kappa)
    oacp_cfg = {}
    if args.oacp_dt_plan is not None:
        oacp_cfg["dt_plan"] = float(args.oacp_dt_plan)
    if args.oacp_Th is not None:
        oacp_cfg["Th"] = float(args.oacp_Th)
    if args.oacp_N is not None:
        oacp_cfg["N"] = int(args.oacp_N)
    if args.oacp_n_shared is not None:
        oacp_cfg["n_shared"] = int(args.oacp_n_shared)
    if args.oacp_max_visible_obs is not None:
        oacp_cfg["max_visible_obs"] = int(args.oacp_max_visible_obs)
    if args.oacp_max_occ_scenarios is not None:
        oacp_cfg["max_occ_scenarios"] = int(args.oacp_max_occ_scenarios)
    if args.oacp_max_active_occlusions is not None:
        oacp_cfg["max_active_occlusions"] = int(args.oacp_max_active_occlusions)
    if args.oacp_margin_obs is not None:
        oacp_cfg["margin_obs"] = float(args.oacp_margin_obs)
    if args.oacp_v_ref_default is not None:
        oacp_cfg["v_ref_default"] = float(args.oacp_v_ref_default)
    if args.oacp_hidden_agent_radius is not None:
        oacp_cfg["hidden_agent_radius"] = float(args.oacp_hidden_agent_radius)
    if args.oacp_hidden_spawn_clearance is not None:
        oacp_cfg["hidden_spawn_clearance"] = float(args.oacp_hidden_spawn_clearance)
    if args.oacp_hidden_speed is not None:
        oacp_cfg["hidden_speed"] = float(args.oacp_hidden_speed)
    if args.oacp_hidden_speed_scale is not None:
        oacp_cfg["hidden_speed_scale"] = float(args.oacp_hidden_speed_scale)
    if args.oacp_active_selection_delta is not None:
        oacp_cfg["active_selection_delta"] = float(args.oacp_active_selection_delta)
    if args.oacp_srq_confidence_z is not None:
        oacp_cfg["srq_confidence_z"] = float(args.oacp_srq_confidence_z)
    if args.oacp_srq_lane_width_min is not None:
        oacp_cfg["srq_lane_width_min"] = float(args.oacp_srq_lane_width_min)
    if args.oacp_srq_lane_width_max is not None:
        oacp_cfg["srq_lane_width_max"] = float(args.oacp_srq_lane_width_max)
    if args.oacp_cth_min is not None:
        oacp_cfg["cth_min"] = float(args.oacp_cth_min)
    if args.oacp_cth_max_explore is not None:
        oacp_cfg["cth_max_explore"] = float(args.oacp_cth_max_explore)
    if args.oacp_cth_max_fallback is not None:
        oacp_cfg["cth_max_fallback"] = float(args.oacp_cth_max_fallback)
    if args.oacp_v_occ_min_scale is not None:
        oacp_cfg["v_occ_min_scale"] = float(args.oacp_v_occ_min_scale)
    if args.oacp_v_occ_min_abs is not None:
        oacp_cfg["v_occ_min_abs"] = float(args.oacp_v_occ_min_abs)
    if args.oacp_barrier_alpha_start is not None:
        oacp_cfg["barrier_alpha_start"] = float(args.oacp_barrier_alpha_start)
    if args.oacp_barrier_alpha_end is not None:
        oacp_cfg["barrier_alpha_end"] = float(args.oacp_barrier_alpha_end)
    if args.oacp_ellipse_scale_x is not None:
        oacp_cfg["ellipse_scale_x"] = float(args.oacp_ellipse_scale_x)
    if args.oacp_ellipse_scale_y is not None:
        oacp_cfg["ellipse_scale_y"] = float(args.oacp_ellipse_scale_y)
    if args.oacp_ellipse_buffer_x is not None:
        oacp_cfg["ellipse_buffer_x"] = float(args.oacp_ellipse_buffer_x)
    if args.oacp_ellipse_buffer_y is not None:
        oacp_cfg["ellipse_buffer_y"] = float(args.oacp_ellipse_buffer_y)
    if args.oacp_use_bezier_reference is not None:
        oacp_cfg["use_bezier_reference"] = bool(args.oacp_use_bezier_reference)
    if args.oacp_bezier_ref_order is not None:
        oacp_cfg["bezier_ref_order"] = int(args.oacp_bezier_ref_order)
    if args.oacp_branch_switch_margin is not None:
        oacp_cfg["branch_switch_margin"] = float(args.oacp_branch_switch_margin)
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
        rand_obs_setting=args.rand_obs_setting,
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
