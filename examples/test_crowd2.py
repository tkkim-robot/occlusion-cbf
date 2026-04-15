"""
Route-focused occlusion benchmark.

This scenario keeps the same controller stack/FOV/visibility pipeline as
`examples/test_crowd.py`, but changes the mission and obstacle generator:
- workspace: 30 x 30
- route: (2,2) -> (2,28) -> (28,2) -> (28,28)
- occluder/hidden-agent events are distributed along the route segments
- background clutter is kept sparse and away from the route corridor

Run:
    uv run python examples/test_crowd2.py --baseline occlusion_cbf --model uni
"""

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    from examples._baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        resolve_baseline_alias,
    )
    from examples import test_crowd as crowd1
except ImportError:
    from _baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        resolve_baseline_alias,
    )

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))

    import test_crowd as crowd1

ENV_WIDTH = 30.0
ENV_HEIGHT = 30.0
TRACKING_VIEW_WINDOW_SIZE = 10.0
ROUTE_OUTER_BOUND_MARGIN = 2.0
ROUTE_WAYPOINTS = np.array(
    [
        [2.0, 2.0, 0.5 * np.pi],
        [2.0, 28.0, 0.0],
        [28.0, 2.0, 0.0],
        [28.0, 28.0, 0.0],
    ],
    dtype=np.float64,
)
ROUTE_NOMINAL_SPEED = 0.8
ROUTE_EVENT_VALIDATION_DT = 0.1
ROUTE_EVENT_VALIDATION_HORIZON_S = 12.0
ROUTE_EVENT_PLACE_ATTEMPTS = 48
ROUTE_HIDDEN_ATTEMPTS = 36
ROUTE_EXTRA_HIDDEN_ATTEMPTS = 24
ROUTE_BG_BATCH_LIMIT = 12
DEFAULT_FORCED_EVENTS = 6
DEFAULT_FORCED_HIDDEN_SPEED = 0.6
DEFAULT_FORCED_OCCLUDER_RADIUS_MIN = 0.8
DEFAULT_FORCED_OCCLUDER_RADIUS_MAX = 1.0


def _route_xy():
    return crowd1._waypoint_xy_array(ROUTE_WAYPOINTS)


def _polyline_cumulative_lengths(path_xy):
    path_xy = crowd1._waypoint_xy_array(path_xy)
    cum = [0.0]
    total = 0.0
    for idx in range(path_xy.shape[0] - 1):
        total += float(np.linalg.norm(path_xy[idx + 1] - path_xy[idx]))
        cum.append(total)
    return np.asarray(cum, dtype=float)


ROUTE_XY = _route_xy()
ROUTE_CUM = _polyline_cumulative_lengths(ROUTE_XY)
ROUTE_LENGTH = float(ROUTE_CUM[-1])


def _route_point_at_distance(dist):
    d = float(np.clip(dist, 0.0, ROUTE_LENGTH))
    for idx in range(ROUTE_XY.shape[0] - 1):
        seg_a = ROUTE_XY[idx]
        seg_b = ROUTE_XY[idx + 1]
        seg_len = float(np.linalg.norm(seg_b - seg_a))
        if seg_len <= 1e-12:
            continue
        if d <= ROUTE_CUM[idx + 1]:
            local = d - ROUTE_CUM[idx]
            return seg_a + (local / seg_len) * (seg_b - seg_a)
    return ROUTE_XY[-1].copy()


def _route_segments():
    segs = []
    for idx in range(ROUTE_XY.shape[0] - 1):
        a = ROUTE_XY[idx]
        b = ROUTE_XY[idx + 1]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-12:
            continue
        tangent = seg / seg_len
        lateral = np.array([-tangent[1], tangent[0]], dtype=float)
        segs.append(
            {
                "idx": idx,
                "a": a,
                "b": b,
                "length": seg_len,
                "tangent": tangent,
                "lateral": lateral,
                "s0": float(ROUTE_CUM[idx]),
                "s1": float(ROUTE_CUM[idx + 1]),
            }
        )
    return segs


ROUTE_SEGS = _route_segments()


def _clip_to_workspace(xy, margin=0.8):
    arr = np.asarray(xy, dtype=float).reshape(2,)
    return np.array(
        [
            float(np.clip(arr[0], margin, ENV_WIDTH - margin)),
            float(np.clip(arr[1], margin, ENV_HEIGHT - margin)),
        ],
        dtype=float,
    )


def _apply_outer_flow_bounds(row, meta, margin=ROUTE_OUTER_BOUND_MARGIN):
    row = np.asarray(row, dtype=float).copy()
    meta = dict(meta)
    margin = float(max(0.0, margin))
    x_min = float(-margin)
    x_max = float(ENV_WIDTH + margin)
    y_min = float(-margin)
    y_max = float(ENV_HEIGHT + margin)
    row[5] = y_min
    row[6] = y_max
    meta["x_min"] = x_min
    meta["x_max"] = x_max
    return row, meta


def _rows_overlap(candidate_xy, candidate_r, existing_rows, margin):
    return crowd1._rows_overlap(candidate_xy, candidate_r, existing_rows, margin)


def _simulate_route_forced_event_nominal(
    *,
    occluder_xy,
    occluder_radius,
    hidden_xy,
    hidden_vel,
    hidden_radius,
    nominal_speed=ROUTE_NOMINAL_SPEED,
    dt=ROUTE_EVENT_VALIDATION_DT,
    horizon_s=ROUTE_EVENT_VALIDATION_HORIZON_S,
):
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
        ego_xy = crowd1._nominal_ego_position_on_polyline(ROUTE_XY, nominal_speed, t)
        hxy = hidden_xy + t * hidden_vel
        d_path = crowd1._point_polyline_distance(hxy, ROUTE_XY)
        dist_to_ego = float(np.linalg.norm(hxy - ego_xy))
        ttc_nominal = dist_to_ego / max(float(nominal_speed) + hidden_speed, 1e-6)

        min_distance_to_path = min(min_distance_to_path, d_path)
        min_ttc_nominal = min(min_ttc_nominal, ttc_nominal)

        occluded = crowd1._disc_occludes_target(
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

        if d_path <= max(0.9, float(hidden_radius) + 0.45) and dist_to_ego <= 4.2:
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


def _background_candidate_valid(row, existing_rows, min_route_clearance, start_clearance):
    row = np.asarray(row, dtype=float)
    center = row[:2]
    radius = float(row[2])
    if crowd1._point_polyline_distance(center, ROUTE_XY) < float(min_route_clearance):
        return False
    if float(np.linalg.norm(center - ROUTE_XY[0])) < float(start_clearance):
        return False
    if _rows_overlap(center, radius, existing_rows, margin=0.2):
        return False
    return True


def _sample_hidden_for_event(
    *,
    rng,
    occ_xy,
    occ_radius,
    observer_xy,
    event_xy,
    tangent,
    lateral,
    existing_rows,
    forced_hidden_speed,
    forced_validate_occlusion,
    forced_require_corridor_conflict,
    attempts=ROUTE_HIDDEN_ATTEMPTS,
    lateral_jitter=0.16,
    target_along_jitter=0.6,
    target_cross_jitter=0.35,
    overlap_margin=0.18,
):
    occ_dir = crowd1._safe_normalize(np.asarray(occ_xy, dtype=float).reshape(2,) - np.asarray(observer_xy, dtype=float).reshape(2,))
    if occ_dir is None:
        return None
    max_attempts = int(max(1, attempts))
    for _ in range(max_attempts):
        hidden_radius = float(rng.uniform(0.28, 0.45))
        gap = occ_radius + hidden_radius + float(rng.uniform(0.12, 0.34))
        hidden_xy = np.asarray(occ_xy, dtype=float).reshape(2,) + gap * occ_dir + lateral * float(rng.uniform(-lateral_jitter, lateral_jitter))
        hidden_xy = _clip_to_workspace(hidden_xy, margin=0.8)
        if _rows_overlap(hidden_xy, hidden_radius, existing_rows, margin=float(overlap_margin)):
            continue

        target_xy = (
            np.asarray(event_xy, dtype=float).reshape(2,)
            + tangent * float(rng.uniform(-target_along_jitter, target_along_jitter))
            + lateral * float(rng.uniform(-target_cross_jitter, target_cross_jitter))
        )
        target_xy = _clip_to_workspace(target_xy, margin=0.5)
        vel_dir = crowd1._safe_normalize(target_xy - hidden_xy)
        if vel_dir is None:
            continue
        hidden_speed = float(forced_hidden_speed) * float(rng.uniform(0.9, 1.05))
        hidden_vel = hidden_speed * vel_dir
        initially_occluded = crowd1._disc_occludes_target(
            observer_xy,
            hidden_xy,
            occ_xy,
            occ_radius,
            target_radius=hidden_radius,
        )
        if bool(forced_validate_occlusion) and not initially_occluded:
            continue
        pred = _simulate_route_forced_event_nominal(
            occluder_xy=occ_xy,
            occluder_radius=occ_radius,
            hidden_xy=hidden_xy,
            hidden_vel=hidden_vel,
            hidden_radius=hidden_radius,
        )
        if pred["predicted_reveal_step"] is None:
            continue
        if bool(forced_require_corridor_conflict) and not bool(pred["corridor_conflict"]):
            continue
        hidden_row, hidden_meta = _apply_outer_flow_bounds(
            np.array(
                [hidden_xy[0], hidden_xy[1], hidden_radius, hidden_vel[0], hidden_vel[1], 0.0, ENV_HEIGHT, 1.0],
                dtype=float,
            ),
            {
                "mode": 1,
                "v_max": float(hidden_speed),
                "theta": float(np.arctan2(hidden_vel[1], hidden_vel[0])),
            },
        )
        return {
            "row": hidden_row,
            "meta": hidden_meta,
            "hidden_xy": hidden_xy,
            "hidden_vel": hidden_vel,
            "hidden_speed": float(hidden_speed),
            "hidden_radius": float(hidden_radius),
            "initially_occluded": bool(initially_occluded),
            "pred": pred,
        }
    return None


def _event_segment_quotas(n_events):
    n_events = int(max(0, n_events))
    n_seg = max(1, len(ROUTE_SEGS))
    quotas = [n_events // n_seg] * n_seg
    for idx in range(n_events % n_seg):
        quotas[idx] += 1
    if n_events > 0 and any(q == 0 for q in quotas):
        order = sorted(range(n_seg), key=lambda i: ROUTE_SEGS[i]["length"], reverse=True)
        for idx in range(n_seg):
            if quotas[idx] > 0:
                continue
            donor = next((j for j in order if quotas[j] > 1), None)
            if donor is None:
                break
            quotas[donor] -= 1
            quotas[idx] += 1
    return quotas


def _build_route_random_scenario(*, case_seed, n_rand, rand_obs, static_occluders):
    rng = np.random.default_rng(int(case_seed))
    existing_rows = []
    rows = []
    meta = []
    batch_seed = int(case_seed)
    while len(rows) < int(n_rand):
        extra_rows, extra_meta = crowd1.LocalTrackingControllerDyn_OCC.make_random_obstacles7(
            n_rand=max(8, 4 * max(1, int(n_rand) - len(rows))),
            v_obs_max=(0.0 if bool(static_occluders) else 0.45),
            x_range=(1.0, ENV_WIDTH - 1.0),
            y_spawn_range=(1.0, ENV_HEIGHT - 1.0),
            r_range=(0.28, 0.42),
            y_bounds=(0.0, ENV_HEIGHT),
            seed=batch_seed,
            rand_obs=bool(rand_obs),
        )
        batch_seed += 7919
        for row, m in zip(extra_rows, extra_meta):
            row = np.asarray(row, dtype=float)
            if not _background_candidate_valid(row, existing_rows, min_route_clearance=2.6, start_clearance=2.0):
                continue
            if bool(static_occluders):
                row[3] = 0.0
                row[4] = 0.0
                m = {"mode": 0, "v_max": 0.0, "theta": 0.0}
                row8 = np.hstack((row, [0.0]))
            else:
                m = dict(m)
                m["v_max"] = float(min(m.get("v_max", 0.45), 0.45))
                row, m = _apply_outer_flow_bounds(row, m)
                row8 = np.hstack((row, [1.0]))
            rows.append(row8)
            meta.append(m)
            existing_rows.append(row)
            if len(rows) >= int(n_rand):
                break
        if batch_seed - int(case_seed) > 16 * 7919 and len(rows) < int(n_rand):
            raise RuntimeError(f"Failed to place {n_rand} route-random obstacles; placed {len(rows)}.")

    scenario_diag = {
        "crowd_mode": "random",
        "n_forced_events": 0,
        "n_forced_hidden_total": 0,
        "n_forced_extra_hidden": 0,
        "n_background_rand": int(len(rows)),
        "n_forced_initially_occluded": 0,
        "n_forced_revealed": 0,
        "n_forced_corridor_conflict": 0,
        "min_reveal_distance_to_ego_path": None,
        "min_reveal_ttc_to_nominal_ego": None,
        "reveal_steps": [],
        "forced_event_meta": [],
    }
    return np.vstack(rows) if rows else np.empty((0, 8), dtype=float), meta, scenario_diag


def _build_route_forced_emergence_scenario(
    *,
    case_seed,
    n_rand,
    rand_obs,
    static_occluders,
    forced_events,
    forced_bg_rand,
    forced_hidden_speed,
    forced_occluder_radius_min,
    forced_occluder_radius_max,
    forced_validate_occlusion,
    forced_require_corridor_conflict,
):
    rng = np.random.default_rng(int(case_seed))
    forced_rows = []
    forced_meta = []
    forced_event_meta = []
    guard_rows = []

    quotas = _event_segment_quotas(int(forced_events))
    for seg_idx, quota in enumerate(quotas):
        seg = ROUTE_SEGS[seg_idx]
        for local_idx in range(int(quota)):
            added = False
            base_ratio = float(local_idx + 1) / float(quota + 1)
            ratio = float(np.clip(base_ratio + rng.uniform(-0.06, 0.06), 0.18, 0.82))
            event_xy = seg["a"] + ratio * (seg["b"] - seg["a"])
            side_order = [1.0, -1.0] if ((seg_idx + local_idx) % 2 == 0) else [-1.0, 1.0]
            lateral_candidates = [1.15, 1.45, 1.75]
            along_candidates = [0.0, -0.25, 0.25]
            lead_candidates = [2.0, 2.6, 3.0]
            radius_candidates = [
                float(forced_occluder_radius_min),
                float(0.5 * (forced_occluder_radius_min + forced_occluder_radius_max)),
                float(forced_occluder_radius_max),
            ]
            for side_sign in side_order:
                for lateral_mag in lateral_candidates:
                    for along_jitter in along_candidates:
                        for lead_distance in lead_candidates:
                            for occ_radius in radius_candidates:
                                occ_xy = event_xy + along_jitter * seg["tangent"] + side_sign * lateral_mag * seg["lateral"]
                                occ_xy = _clip_to_workspace(occ_xy, margin=1.0)
                                occ_guard_radius = float(occ_radius) + 1.0
                                if _rows_overlap(occ_xy, occ_guard_radius, guard_rows, margin=0.5):
                                    continue
                                observer_dist = float(
                                    np.clip(seg["s0"] + ratio * seg["length"] - lead_distance, seg["s0"], seg["s1"])
                                )
                                observer_xy = _route_point_at_distance(observer_dist)
                                primary_hidden = _sample_hidden_for_event(
                                    rng=rng,
                                    occ_xy=occ_xy,
                                    occ_radius=occ_radius,
                                    observer_xy=observer_xy,
                                    event_xy=event_xy,
                                    tangent=seg["tangent"],
                                    lateral=seg["lateral"],
                                    existing_rows=forced_rows,
                                    forced_hidden_speed=forced_hidden_speed,
                                    forced_validate_occlusion=forced_validate_occlusion,
                                    forced_require_corridor_conflict=forced_require_corridor_conflict,
                                    attempts=ROUTE_HIDDEN_ATTEMPTS,
                                )
                                if primary_hidden is None and bool(forced_require_corridor_conflict):
                                    primary_hidden = _sample_hidden_for_event(
                                        rng=rng,
                                        occ_xy=occ_xy,
                                        occ_radius=occ_radius,
                                        observer_xy=observer_xy,
                                        event_xy=event_xy,
                                        tangent=seg["tangent"],
                                        lateral=seg["lateral"],
                                        existing_rows=forced_rows,
                                        forced_hidden_speed=forced_hidden_speed,
                                        forced_validate_occlusion=forced_validate_occlusion,
                                        forced_require_corridor_conflict=False,
                                        attempts=max(8, ROUTE_HIDDEN_ATTEMPTS // 2),
                                    )
                                if primary_hidden is None:
                                    continue

                                away_dir = crowd1._safe_normalize(side_sign * seg["lateral"])
                                if away_dir is None:
                                    away_dir = np.array([1.0, 0.0], dtype=float)
                                occ_speed = 0.0 if bool(static_occluders) else float(rng.uniform(0.18, 0.32))
                                occ_theta = float(np.arctan2(away_dir[1], away_dir[0]) + rng.normal(0.0, 0.22))
                                occ_vx = float(occ_speed * np.cos(occ_theta))
                                occ_vy = float(occ_speed * np.sin(occ_theta))
                                roam_outward = 0.0 if bool(static_occluders) else float(rng.uniform(3.0, 5.0))
                                roam_inward = 0.0 if bool(static_occluders) else float(rng.uniform(0.35, 0.85))
                                roam_along = 0.0 if bool(static_occluders) else float(rng.uniform(0.45, 1.10))

                                away_x = float(abs(away_dir[0]))
                                away_y = float(abs(away_dir[1]))
                                tang_x = float(abs(seg["tangent"][0]))
                                tang_y = float(abs(seg["tangent"][1]))
                                outward_x = max(0.6, roam_outward * away_x + roam_along * tang_x)
                                outward_y = max(0.6, roam_outward * away_y + roam_along * tang_y)
                                inward_x = max(0.25, roam_inward * away_x + 0.45 * roam_along * tang_x)
                                inward_y = max(0.25, roam_inward * away_y + 0.45 * roam_along * tang_y)

                                if away_dir[0] >= 0.0:
                                    occ_x_min = occ_xy[0] - inward_x
                                    occ_x_max = occ_xy[0] + outward_x
                                else:
                                    occ_x_min = occ_xy[0] - outward_x
                                    occ_x_max = occ_xy[0] + inward_x

                                if away_dir[1] >= 0.0:
                                    occ_y_min = occ_xy[1] - inward_y
                                    occ_y_max = occ_xy[1] + outward_y
                                else:
                                    occ_y_min = occ_xy[1] - outward_y
                                    occ_y_max = occ_xy[1] + inward_y

                                occ_x_min = max(0.5, float(occ_x_min))
                                occ_x_max = min(ENV_WIDTH - 0.5, float(occ_x_max))
                                occ_y_min = max(0.5, float(occ_y_min))
                                occ_y_max = min(ENV_HEIGHT - 0.5, float(occ_y_max))


                                occ_row = np.array(
                                    [occ_xy[0], occ_xy[1], occ_radius, occ_vx, occ_vy, occ_y_min, occ_y_max, 1.0],
                                    dtype=float,
                                )
                                hid_row = primary_hidden["row"]
                                if _rows_overlap(occ_xy, occ_radius, forced_rows, margin=0.5):
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
                                            "forced_occluder_sep_margin": 0.5,
                                            "heading_jitter_std": 0.004,
                                            "large_turn_prob": 0.01,
                                            "large_turn_std": 0.03,
                                            "x_min": float(occ_x_min),
                                            "x_max": float(occ_x_max),
                                            "home_x": float(occ_xy[0]),
                                            "home_y": float(occ_xy[1]),
                                            "forced_drift_dir_x": float(away_dir[0]),
                                            "forced_drift_dir_y": float(away_dir[1]),
                                            "forced_drift_gain": 0.35,
                                        },
                                        primary_hidden["meta"],
                                    ]
                                )
                                forced_event_meta.append(
                                    {
                                        "event_id": int(len(forced_event_meta)),
                                        "segment_index": int(seg_idx),
                                        "occluder_index": int(occ_idx),
                                        "hidden_index": int(hid_idx),
                                        "hidden_indices": [int(hid_idx)],
                                        "occluder_center": [float(occ_xy[0]), float(occ_xy[1])],
                                        "occluder_radius": float(occ_radius),
                                        "occluder_velocity": [float(occ_vx), float(occ_vy)],
                                        "occluder_speed": float(occ_speed),
                                        "occluder_roam_x_bounds": [float(occ_x_min), float(occ_x_max)],
                                        "occluder_roam_y_bounds": [float(occ_y_min), float(occ_y_max)],
                                        "event_xy": [float(event_xy[0]), float(event_xy[1])],
                                        "observer_xy": [float(observer_xy[0]), float(observer_xy[1])],
                                        "hidden_initial_position": [float(primary_hidden["hidden_xy"][0]), float(primary_hidden["hidden_xy"][1])],
                                        "hidden_velocity": [float(primary_hidden["hidden_vel"][0]), float(primary_hidden["hidden_vel"][1])],
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
                                guard_rows.append(np.array([occ_xy[0], occ_xy[1], occ_guard_radius], dtype=float))
                                added = True
                                break
                            if added:
                                break
                        if added:
                            break
                    if added:
                        break
                if added:
                    break
            if not added:
                continue

    hidden_total_target = int(max(len(forced_event_meta), int(round(0.7 * max(0, int(n_rand))))))
    if forced_bg_rand is None:
        preferred_bg_target = max(0, int(n_rand) - hidden_total_target)
    else:
        preferred_bg_target = max(0, int(forced_bg_rand)) if bool(rand_obs) else 0
        hidden_total_target = max(hidden_total_target, int(n_rand) - preferred_bg_target)
    extra_hidden_target = max(0, hidden_total_target - len(forced_event_meta))

    if extra_hidden_target > 0 and forced_event_meta:
        per_event = extra_hidden_target // len(forced_event_meta)
        rem = extra_hidden_target % len(forced_event_meta)
        for event_idx, meta in enumerate(forced_event_meta):
            add_count = int(per_event + (1 if event_idx < rem else 0))
            if add_count <= 0:
                continue
            occ_xy = np.asarray(meta["occluder_center"], dtype=float)
            occ_radius = float(meta["occluder_radius"])
            observer_xy = np.asarray(meta["observer_xy"], dtype=float)
            event_xy = np.asarray(meta["event_xy"], dtype=float)
            seg = ROUTE_SEGS[int(meta["segment_index"])]
            for _ in range(add_count):
                extra_hidden = _sample_hidden_for_event(
                    rng=rng,
                    occ_xy=occ_xy,
                    occ_radius=occ_radius,
                    observer_xy=observer_xy,
                    event_xy=event_xy,
                    tangent=seg["tangent"],
                    lateral=seg["lateral"],
                    existing_rows=[row for i, row in enumerate(forced_rows) if i != int(meta["occluder_index"])],
                    forced_hidden_speed=forced_hidden_speed,
                    forced_validate_occlusion=forced_validate_occlusion,
                    forced_require_corridor_conflict=False,
                    attempts=ROUTE_EXTRA_HIDDEN_ATTEMPTS,
                    lateral_jitter=0.30,
                    target_along_jitter=0.8,
                    target_cross_jitter=0.45,
                    overlap_margin=0.12,
                )
                if extra_hidden is None:
                    continue
                hid_idx = len(forced_rows)
                forced_rows.append(extra_hidden["row"])
                forced_meta.append(extra_hidden["meta"])
                meta["hidden_indices"].append(int(hid_idx))
                meta["extra_hidden_count"] = int(meta.get("extra_hidden_count", 0) + 1)

    hidden_total_actual = int(sum(len(meta.get("hidden_indices", [meta["hidden_index"]])) for meta in forced_event_meta))
    bg_target = max(int(preferred_bg_target), max(0, int(n_rand) - hidden_total_actual))

    bg_rows_8 = np.empty((0, 8), dtype=float)
    bg_meta = []
    if bg_target > 0:
        keep_rows = []
        keep_meta = []
        batch_id = 0
        while len(keep_rows) < bg_target and batch_id < ROUTE_BG_BATCH_LIMIT:
            sample_target = max((bg_target - len(keep_rows)) * 3, bg_target)
            extra_rows, extra_meta = crowd1.LocalTrackingControllerDyn_OCC.make_random_obstacles7(
                n_rand=int(sample_target),
                v_obs_max=(0.0 if bool(static_occluders) else 0.45),
                x_range=(1.0, ENV_WIDTH - 1.0),
                y_spawn_range=(1.0, ENV_HEIGHT - 1.0),
                r_range=(0.28, 0.40),
                y_bounds=(0.0, ENV_HEIGHT),
                seed=int(case_seed) + 7919 * int(batch_id),
                rand_obs=bool(rand_obs),
            )
            for row, meta in zip(extra_rows, extra_meta):
                if len(keep_rows) >= bg_target:
                    break
                row = np.asarray(row, dtype=float)
                existing = forced_rows + keep_rows
                if not _background_candidate_valid(row, existing, min_route_clearance=2.6, start_clearance=2.2):
                    continue
                row, meta = _apply_outer_flow_bounds(row, meta)
                keep_rows.append(row)
                if bool(static_occluders):
                    keep_meta.append({"mode": 0, "v_max": 0.0, "theta": 0.0})
                else:
                    keep_meta.append(dict(meta))
            batch_id += 1
        if len(keep_rows) < bg_target:
            raise RuntimeError(
                f"Failed to backfill route background obstacles to target count: need {bg_target}, placed {len(keep_rows)}."
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
    known_obs = np.vstack(known_obs_parts) if known_obs_parts else np.empty((0, 8), dtype=float)
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
        "n_forced_initially_occluded": int(sum(1 for meta in forced_event_meta if bool(meta["initially_occluded_geom"]))),
        "n_forced_revealed": 0,
        "n_forced_corridor_conflict": int(sum(1 for meta in forced_event_meta if bool(meta["corridor_conflict"]))),
        "min_reveal_distance_to_ego_path": (None if len(min_reveal_dist) == 0 else float(np.min(min_reveal_dist))),
        "min_reveal_ttc_to_nominal_ego": (None if len(min_reveal_ttc) == 0 else float(np.min(min_reveal_ttc))),
        "reveal_steps": [],
        "forced_event_meta": forced_event_meta,
    }
    return known_obs, obs_meta, scenario_diag


def run_crowd_scenario(
    controller_type=None,
    model_key="di",
    show_animation=True,
    save_animation=False,
    tf=200.0,
    seed=42,
    case_idx=None,
    rand_obs=True,
    n_rand=20,
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
    crowd_mode="forced_emergence",
    forced_events=DEFAULT_FORCED_EVENTS,
    forced_bg_rand=None,
    forced_hidden_speed=DEFAULT_FORCED_HIDDEN_SPEED,
    forced_occluder_radius_min=DEFAULT_FORCED_OCCLUDER_RADIUS_MIN,
    forced_occluder_radius_max=DEFAULT_FORCED_OCCLUDER_RADIUS_MAX,
    forced_validate_occlusion=True,
    forced_require_corridor_conflict=True,
    static_occluders=False,
    backup_cbf_overrides=None,
    robot_spec_overrides=None,
    return_metrics=False,
    max_steps=None,
    max_sim_time=None,
):
    case_seed = crowd1._compute_case_seed(seed, case_idx)
    mode = str(crowd_mode).strip().lower()
    if mode not in {"random", "forced_emergence"}:
        raise ValueError(f"Unsupported crowd_mode `{crowd_mode}`. Use `random` or `forced_emergence`.")

    if mode == "forced_emergence":
        known_obs, obs_meta, scenario_diag = _build_route_forced_emergence_scenario(
            case_seed=case_seed,
            n_rand=n_rand,
            rand_obs=rand_obs,
            static_occluders=static_occluders,
            forced_events=forced_events,
            forced_bg_rand=forced_bg_rand,
            forced_hidden_speed=forced_hidden_speed,
            forced_occluder_radius_min=forced_occluder_radius_min,
            forced_occluder_radius_max=forced_occluder_radius_max,
            forced_validate_occlusion=forced_validate_occlusion,
            forced_require_corridor_conflict=forced_require_corridor_conflict,
        )
    else:
        known_obs, obs_meta, scenario_diag = _build_route_random_scenario(
            case_seed=case_seed,
            n_rand=n_rand,
            rand_obs=rand_obs,
            static_occluders=static_occluders,
        )

    return crowd1.run_crowd_scenario(
        controller_type=controller_type,
        model_key=model_key,
        show_animation=show_animation,
        save_animation=save_animation,
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
        occ_version=occ_version,
        occ_visible_scale=occ_visible_scale,
        occ_enable_visible_hocbf=occ_enable_visible_hocbf,
        oa_dynamic_occluders=oa_dynamic_occluders,
        oa_allow_solver_fallback=oa_allow_solver_fallback,
        oa_dsafe=oa_dsafe,
        oa_visible_reach_mode=oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=oa_use_nominal_tracking_cost,
        oa_wmax=oa_wmax,
        oa_dt=oa_dt,
        crowd_mode=mode,
        forced_events=forced_events,
        forced_bg_rand=forced_bg_rand,
        forced_hidden_speed=forced_hidden_speed,
        forced_occluder_radius_min=forced_occluder_radius_min,
        forced_occluder_radius_max=forced_occluder_radius_max,
        forced_validate_occlusion=forced_validate_occlusion,
        forced_require_corridor_conflict=forced_require_corridor_conflict,
        static_occluders=static_occluders,
        backup_cbf_overrides=backup_cbf_overrides,
        robot_spec_overrides=robot_spec_overrides,
        waypoints_override=ROUTE_WAYPOINTS,
        env_width_override=ENV_WIDTH,
        env_height_override=ENV_HEIGHT,
        known_obs_override=known_obs,
        obs_meta_override=obs_meta,
        scenario_diag_override=scenario_diag,
        return_metrics=return_metrics,
        max_steps=max_steps,
        max_sim_time=max_sim_time,
        tracking_view_enable=True,
        tracking_view_window_size=TRACKING_VIEW_WINDOW_SIZE,
        scenario_name="Crowd2",
        hide_env_boundary=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run route-focused occlusion benchmark scenario.")
    parser.add_argument("--model", type=str, default="di", choices=["di", "du", "uni"], help="Robot model alias (`di`, `du`, or `uni`).")
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
    parser.add_argument("--tf", type=float, default=200.0, help="Simulation final time [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scenario generation.")
    parser.add_argument("--idx", "--case-idx", dest="case_idx", type=int, default=None, help="Case index (1-based) for deterministic random scenario selection.")
    parser.add_argument(
        "--n-rand",
        type=int,
        default=20,
        help=(
            "Target number of non-occluder movers. The generator uses most of this budget for hidden-emergence "
            "agents and the rest for sparse background clutter."
        ),
    )
    parser.add_argument("--no-rand-obs", action="store_true", help="Disable random moving obstacles.")
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument("--crowd-mode", type=str, default="forced_emergence", choices=["random", "forced_emergence"], help="Scenario generator mode.")
    parser.add_argument("--forced-events", type=int, default=DEFAULT_FORCED_EVENTS, help="Number of occluder/hidden-agent events along the route.")
    parser.add_argument("--forced-bg-rand", type=int, default=None, help="Explicit background clutter count. Default leaves the remainder after hidden-event allocation.")
    parser.add_argument("--forced-hidden-speed", type=float, default=DEFAULT_FORCED_HIDDEN_SPEED, help="Nominal hidden-agent speed magnitude.")
    parser.add_argument("--forced-occluder-radius-min", type=float, default=DEFAULT_FORCED_OCCLUDER_RADIUS_MIN, help="Minimum occluder radius.")
    parser.add_argument("--forced-occluder-radius-max", type=float, default=DEFAULT_FORCED_OCCLUDER_RADIUS_MAX, help="Maximum occluder radius.")
    parser.add_argument("--forced-validate-occlusion", type=crowd1._str2bool, nargs="?", const=True, default=True, help="Require hidden agent to be initially occluded during generation.")
    parser.add_argument("--forced-require-corridor-conflict", type=crowd1._str2bool, nargs="?", const=True, default=True, help="Require predicted route-corridor conflict during generation.")
    parser.add_argument("--du-min-speed-scale", type=float, default=None)
    parser.add_argument("--du-k-turn-brake", type=float, default=None)
    parser.add_argument("--du-k-a-p", type=float, default=None)
    parser.add_argument("--du-k-a-d", type=float, default=None)
    parser.add_argument("--du-reverse-enter-cos", type=float, default=None)
    parser.add_argument("--du-reverse-exit-cos", type=float, default=None)
    parser.add_argument("--du-reverse-min-scale", type=float, default=None)
    parser.add_argument("--uni-reverse-bias", type=float, default=None)
    parser.add_argument("--uni-reverse-gate-angle", type=float, default=None)
    parser.add_argument("--uni-reverse-gate-power", type=float, default=None)
    parser.add_argument("--uni-v-min-cmd-rev", type=float, default=None)
    parser.add_argument("--occ-dt-backup", type=float, default=None)
    parser.add_argument("--occ-t-horizon", type=float, default=None)
    parser.add_argument("--occ-rho-T", type=float, default=None)
    parser.add_argument("--occ-k-p", type=float, default=None)
    parser.add_argument("--occ-k-d", type=float, default=None)
    parser.add_argument("--occ-kappa", type=float, default=None)
    parser.add_argument("--occ-rollout-mode", type=str, choices=["common", "per_scenario"], default=None)
    parser.add_argument("--occ-terminal-slack-weight", type=float, default=None)
    parser.add_argument("--occ-terminal-slack-max", type=float, default=None)
    parser.add_argument("--vref-mode-occ", type=str, choices=["soft", "strict"], default=None)
    parser.add_argument("--vref", type=str, choices=["default", "los"], default=None)
    parser.add_argument("--occ-visible-scale", type=float, default=None)
    parser.add_argument("--occ-version", type=str, choices=["v1", "v2"], default=None)
    parser.add_argument(
        "--occ-enable-visible-hocbf",
        type=crowd1._str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Occlusion-CBF only: also add visible-obstacle CBF/HOCBF rows in the stacked QP.",
    )
    parser.add_argument("--oa-dynamic-occluders", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-allow-solver-fallback", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-dsafe", type=float, default=None)
    parser.add_argument("--oa-visible-reach-mode", type=str, choices=["worst_case", "constant_velocity"], default=None)
    parser.add_argument("--oa-use-nominal-tracking-cost", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oacp-dt-plan", type=float, default=None)
    parser.add_argument("--oacp-Th", type=float, default=None)
    parser.add_argument("--oacp-N", type=int, default=None)
    parser.add_argument("--oacp-n-shared", type=int, default=None)
    parser.add_argument("--oacp-backend", type=str, choices=["coupled_nlp", "admm_lowdim"], default=None)
    parser.add_argument("--oacp-risk-explore-scale", type=float, default=None)
    parser.add_argument("--oacp-risk-fallback-scale", type=float, default=None)
    parser.add_argument("--oacp-explore-speed-scale", type=float, default=None)
    parser.add_argument("--oacp-fallback-speed-scale", type=float, default=None)
    parser.add_argument("--oacp-shared-speed-blend", type=float, default=None)
    parser.add_argument("--oacp-di-profile-occ-scale", type=float, default=None)
    parser.add_argument("--oacp-di-profile-visible-scale", type=float, default=None)
    parser.add_argument("--oacp-di-profile-speed-floor-scale", type=float, default=None)
    parser.add_argument("--oacp-admm-shared-ctrl-pts", type=int, default=None)
    parser.add_argument("--oacp-admm-tail-ctrl-pts", type=int, default=None)
    parser.add_argument("--oacp-admm-rho", type=float, default=None)
    parser.add_argument("--oacp-admm-max-iter", type=int, default=None)
    parser.add_argument("--oacp-admm-pri-tol", type=float, default=None)
    parser.add_argument("--oacp-admm-dual-tol", type=float, default=None)
    parser.add_argument("--oacp-use-nominal-tracking-cost", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oacp-allow-solver-fallback", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oacp-dynamic-occluders", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oacp-visible-reach-mode", type=str, choices=["constant_velocity", "worst_case"], default=None)
    parser.add_argument("--wmax", type=str, choices=["default", "pi"], default="default")
    parser.add_argument("--oa-dt", type=float, default=None)
    parser.add_argument("--static-occluders", type=crowd1._str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_ani", "--save-ani", "--save-anim", "--save-animation", dest="save_anim", type=crowd1._str2bool, nargs="?", const=True, default=False)
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
    if args.occ_t_horizon is not None:
        backup_cbf_overrides["T_horizon"] = float(args.occ_t_horizon)
    if args.occ_rho_T is not None:
        backup_cbf_overrides["rho_T"] = float(args.occ_rho_T)
    if args.occ_k_p is not None:
        backup_cbf_overrides["k_p_occ_di"] = float(args.occ_k_p)
    if args.occ_k_d is not None:
        backup_cbf_overrides["k_d_occ_di"] = float(args.occ_k_d)
    if args.occ_rollout_mode is not None:
        backup_cbf_overrides["occ_rollout_mode"] = str(args.occ_rollout_mode).strip().lower()
    if args.occ_terminal_slack_weight is not None:
        backup_cbf_overrides["terminal_slack_weight"] = float(args.occ_terminal_slack_weight)
    if args.occ_terminal_slack_max is not None:
        backup_cbf_overrides["terminal_slack_max"] = float(args.occ_terminal_slack_max)
    if args.vref is not None:
        backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    robot_spec_overrides = {}
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
    if args.oacp_backend is not None:
        oacp_cfg["backend"] = str(args.oacp_backend).strip().lower()
    if args.oacp_risk_explore_scale is not None:
        oacp_cfg["risk_explore_scale"] = float(args.oacp_risk_explore_scale)
    if args.oacp_risk_fallback_scale is not None:
        oacp_cfg["risk_fallback_scale"] = float(args.oacp_risk_fallback_scale)
    if args.oacp_explore_speed_scale is not None:
        oacp_cfg["explore_speed_scale"] = float(args.oacp_explore_speed_scale)
    if args.oacp_fallback_speed_scale is not None:
        oacp_cfg["fallback_speed_scale"] = float(args.oacp_fallback_speed_scale)
    if args.oacp_shared_speed_blend is not None:
        oacp_cfg["shared_speed_blend"] = float(args.oacp_shared_speed_blend)
    if args.oacp_di_profile_occ_scale is not None:
        oacp_cfg["di_profile_occ_scale"] = float(args.oacp_di_profile_occ_scale)
    if args.oacp_di_profile_visible_scale is not None:
        oacp_cfg["di_profile_visible_scale"] = float(args.oacp_di_profile_visible_scale)
    if args.oacp_di_profile_speed_floor_scale is not None:
        oacp_cfg["di_profile_speed_floor_scale"] = float(args.oacp_di_profile_speed_floor_scale)
    if args.oacp_admm_shared_ctrl_pts is not None:
        oacp_cfg["admm_shared_ctrl_pts"] = int(args.oacp_admm_shared_ctrl_pts)
    if args.oacp_admm_tail_ctrl_pts is not None:
        oacp_cfg["admm_tail_ctrl_pts"] = int(args.oacp_admm_tail_ctrl_pts)
    if args.oacp_admm_rho is not None:
        oacp_cfg["admm_rho"] = float(args.oacp_admm_rho)
    if args.oacp_admm_max_iter is not None:
        oacp_cfg["admm_max_iter"] = int(args.oacp_admm_max_iter)
    if args.oacp_admm_pri_tol is not None:
        oacp_cfg["admm_pri_tol"] = float(args.oacp_admm_pri_tol)
    if args.oacp_admm_dual_tol is not None:
        oacp_cfg["admm_dual_tol"] = float(args.oacp_admm_dual_tol)
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
