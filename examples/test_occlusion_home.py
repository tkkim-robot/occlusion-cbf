"""
Reveal-conditioned occlusion benchmark.

This benchmark keeps the same robot/obstacle geometry and controller stack as
`examples/test_crowd.py`, but replaces the dense mixed-crowd generator with a
reveal-conditioned benchmark family aimed at evaluating occlusion-aware hidden
emergence handling more directly.

Two benchmark families are supported:

- `single_event`: exactly one forced occluder/hidden-agent event plus sparse
  background dynamic clutter.
- `two_event`: two sequential forced occlusion events plus moderate sparse
  background dynamic clutter.

Each `case_idx` maps to a stratified 100-case profile over:

- route-bin along the mission polyline
- occluder-side / side-pattern
- target reveal-TTC bin
- sparse background clutter bin

This is not a "favorable cheat" for OCBF; it isolates the phenomenon OCBF is
designed to handle so that hidden-emergence performance is not drowned out by
unrelated visible crowd congestion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from examples._baseline_defs import (
        CROWD_ALGO_CHOICES,
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        resolve_baseline_alias,
    )
    from examples import test_crowd_narrow as crowd_narrow
    from examples import test_crowd as crowd
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

    import test_crowd_narrow as crowd_narrow
    import test_crowd as crowd

from position_control.ocbf.defaults import (
    OCBF_QP_FAILURE_FALLBACK_MODES,
    OCBF_SELECTION_MODES,
    OCBF_TERMINAL_MODES,
    OCBF_TERMINAL_RESIDUAL_MODES,
    OCBF_VREF_FRONT_MODES,
    OCBF_VREF_SCENARIO_WEIGHT_MODES,
)


HOME_FAMILY_CHOICES = ("single_event", "two_event")
HOME_CASES_PER_FAMILY = 100
HOME_TTC_BINS = (
    (0.65, 0.85),
    (0.85, 1.05),
    (1.05, 1.25),
    (1.25, 1.45),
    (1.45, 1.70),
)

# Background clutter remains intentionally sparse so that hidden reveal is the
# principal hazard. The dense mixed-crowd case already exists as crowd.
HOME_BG_COUNTS = {
    "single_event": (0, 2),
    "two_event": (1, 3),
}

# Distance from the route corridor used to keep nuisance visible clutter away
# from the reveal-critical part of the scene.
HOME_BG_ROUTE_CLEARANCE = {
    "single_event": 3.2,
    "two_event": 2.8,
}

HOME_BG_EVENT_CLEARANCE = {
    "single_event": 3.2,
    "two_event": 2.6,
}


def _route_seg_for_distance(dist: float) -> dict[str, Any]:
    d = float(np.clip(dist, 0.0, crowd.ROUTE_LENGTH))
    for seg in crowd.ROUTE_SEGS:
        if d <= float(seg["s1"]) + 1e-9:
            return seg
    return crowd.ROUTE_SEGS[-1]


def _case_profile(case_idx: int | None, family: str) -> dict[str, Any]:
    idx0 = 0 if case_idx is None else max(0, int(case_idx) - 1)
    idx0 = idx0 % HOME_CASES_PER_FAMILY
    route_bin = idx0 // 20
    side_bin = (idx0 // 10) % 2
    ttc_bin = (idx0 // 2) % 5
    clutter_bin = idx0 % 2
    return {
        "case_bin": int(idx0),
        "route_bin": int(route_bin),
        "side_bin": int(side_bin),
        "ttc_bin": int(ttc_bin),
        "clutter_bin": int(clutter_bin),
        "family": str(family),
    }


def _secondary_ttc_range(primary_range: tuple[float, float]) -> tuple[float, float]:
    mid = 0.5 * (float(primary_range[0]) + float(primary_range[1]))
    mid2 = float(np.clip(mid + 0.25, 0.95, 1.95))
    half = 0.12
    return (float(mid2 - half), float(mid2 + half))


def _build_event_specs(profile: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    family = str(profile["family"])
    route_bin = int(profile["route_bin"])
    side_bin = int(profile["side_bin"])
    ttc_bin = int(profile["ttc_bin"])

    ttc_range = HOME_TTC_BINS[ttc_bin]
    side_sign = 1.0 if side_bin == 0 else -1.0
    single_windows = [
        (0.38, 0.46),
        (0.48, 0.56),
        (0.58, 0.66),
        (0.68, 0.76),
        (0.80, 0.88),
    ]

    if family == "single_event":
        f0, f1 = single_windows[route_bin]
        s_event = float(crowd.ROUTE_LENGTH * rng.uniform(f0, f1))
        return [
            {
                "event_id": 0,
                "s_event": s_event,
                "side_sign": side_sign,
                "ttc_range": ttc_range,
                "max_reveal_path_distance": 1.10,
                "label": "primary",
            }
        ]

    # Two-event family: event 1 is placed in the earlier portion of the route;
    # event 2 follows later so the controller must resolve two reveal episodes
    # within a single navigation problem.
    f0, f1 = single_windows[min(route_bin, len(single_windows) - 3)]
    s_event_1 = float(crowd.ROUTE_LENGTH * rng.uniform(f0, f1))
    f2_0, f2_1 = single_windows[min(route_bin + 2, len(single_windows) - 1)]
    s_event_2 = float(crowd.ROUTE_LENGTH * rng.uniform(f2_0, f2_1))
    second_side = -side_sign if side_bin == 0 else side_sign
    ttc_range_2 = _secondary_ttc_range(ttc_range)
    return [
        {
            "event_id": 0,
            "s_event": s_event_1,
            "side_sign": side_sign,
            "ttc_range": ttc_range,
            "max_reveal_path_distance": 1.10,
            "label": "primary",
        },
        {
            "event_id": 1,
            "s_event": s_event_2,
            "side_sign": second_side,
            "ttc_range": ttc_range_2,
            "max_reveal_path_distance": 1.20,
            "label": "secondary",
        },
    ]


def _ttc_in_range(value: float | None, target: tuple[float, float], slack: float = 0.0) -> bool:
    if value is None:
        return False
    return float(target[0]) - float(slack) <= float(value) <= float(target[1]) + float(slack)


def _background_candidate_valid_home(
    row: np.ndarray,
    existing_rows: list[np.ndarray],
    event_points: list[np.ndarray],
    *,
    min_route_clearance: float,
    start_clearance: float,
    min_event_clearance: float,
) -> bool:
    row = np.asarray(row, dtype=float)
    center = np.asarray(row[:2], dtype=float).reshape(2,)
    radius = float(row[2])
    if crowd_narrow._point_polyline_distance(center, crowd.ROUTE_XY) < float(min_route_clearance):
        return False
    if float(np.linalg.norm(center - crowd.ROUTE_XY[0])) < float(start_clearance):
        return False
    if crowd._rows_overlap(center, radius, existing_rows, margin=0.2):
        return False
    for event_xy in event_points:
        if float(np.linalg.norm(center - np.asarray(event_xy, dtype=float).reshape(2,))) < float(min_event_clearance):
            return False
    return True


def _manual_home_event_fallback(
    *,
    s_event: float,
    side_sign: float,
    existing_rows: list[np.ndarray],
    guard_rows: list[np.ndarray],
    forced_hidden_speed: float,
    rand_obs_setting: str,
    forced_validate_occlusion: bool,
    forced_require_corridor_conflict: bool,
    static_occluders: bool,
    forced_occluder_radius_min: float,
    forced_occluder_radius_max: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    seg = _route_seg_for_distance(s_event)
    event_xy = crowd._route_point_at_distance(s_event)
    tangent = np.asarray(seg["tangent"], dtype=float).reshape(2,)
    lateral = np.asarray(seg["lateral"], dtype=float).reshape(2,)
    for side_try in (float(side_sign), -float(side_sign)):
        for occ_radius in (float(forced_occluder_radius_max), float(forced_occluder_radius_min)):
            occ_xy = crowd._clip_to_workspace(event_xy + side_try * 1.30 * lateral, margin=1.0)
            occ_guard_radius = float(occ_radius) + 1.0
            if crowd._rows_overlap(occ_xy, occ_guard_radius, guard_rows, margin=0.5):
                continue
            if crowd._rows_overlap(occ_xy, float(occ_radius), existing_rows, margin=0.5):
                continue
            observer_xy = crowd._route_point_at_distance(
                float(np.clip(s_event - 2.3, float(seg["s0"]), float(seg["s1"])))
            )
            occ_dir = crowd_narrow._safe_normalize(occ_xy - observer_xy)
            if occ_dir is None:
                continue
            hidden_radius = 0.34
            hidden_xy = occ_xy + (occ_radius + hidden_radius + 0.18) * occ_dir
            if crowd._rows_overlap(hidden_xy, hidden_radius, existing_rows, margin=0.16):
                continue
            for target_shift in (
                (-0.20, 0.00),
                (0.00, 0.00),
                (-0.10, -0.15 * side_try),
                (-0.10, 0.15 * side_try),
            ):
                target_xy = crowd._clip_to_workspace(
                    event_xy + float(target_shift[0]) * tangent + float(target_shift[1]) * lateral,
                    margin=0.5,
                )
                vel_dir = crowd_narrow._safe_normalize(target_xy - hidden_xy)
                if vel_dir is None:
                    continue
                hidden_speed = crowd_narrow._sample_hidden_speed_for_setting(
                    rng,
                    forced_hidden_speed=forced_hidden_speed,
                    rand_obs_setting=rand_obs_setting,
                    legacy_low=0.95,
                    legacy_high=1.05,
                )
                hidden_vel = hidden_speed * vel_dir
                initially_occluded = crowd_narrow._disc_occludes_target(
                    observer_xy,
                    hidden_xy,
                    occ_xy,
                    occ_radius,
                    target_radius=hidden_radius,
                )
                if bool(forced_validate_occlusion) and not initially_occluded:
                    continue
                pred = crowd._simulate_route_forced_event_nominal(
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
                hidden_row, hidden_meta = crowd._apply_outer_flow_bounds(
                    np.array(
                        [hidden_xy[0], hidden_xy[1], hidden_radius, hidden_vel[0], hidden_vel[1], 0.0, crowd.ENV_HEIGHT, 1.0],
                        dtype=float,
                    ),
                    {
                        "mode": 1,
                        "v_max": float(hidden_speed),
                        "theta": float(np.arctan2(hidden_vel[1], hidden_vel[0])),
                    },
                )
                occ_row, occ_meta = _make_occluder_row(
                    occ_xy=np.asarray(occ_xy, dtype=float).reshape(2,),
                    occ_radius=float(occ_radius),
                    seg_tangent=tangent,
                    side_sign=side_try,
                    static_occluders=static_occluders,
                    rng=rng,
                )
                return {
                    "occ_row": occ_row,
                    "occ_meta": occ_meta,
                    "occ_guard_radius": occ_guard_radius,
                    "occ_xy": np.asarray(occ_xy, dtype=float).reshape(2,),
                    "occ_radius": float(occ_radius),
                    "observer_xy": np.asarray(observer_xy, dtype=float).reshape(2,),
                    "event_xy": np.asarray(event_xy, dtype=float).reshape(2,),
                    "hidden": {
                        "row": hidden_row,
                        "meta": hidden_meta,
                        "hidden_xy": hidden_xy,
                        "hidden_vel": hidden_vel,
                        "hidden_speed": float(hidden_speed),
                        "hidden_radius": float(hidden_radius),
                        "initially_occluded": bool(initially_occluded),
                        "pred": pred,
                    },
                    "pred": pred,
                    "segment_index": int(seg["idx"]),
                    "ttc_range_target": None,
                }
    return None


def _make_hidden_candidate_home(
    *,
    rng: np.random.Generator,
    occ_xy: np.ndarray,
    occ_radius: float,
    observer_xy: np.ndarray,
    event_xy: np.ndarray,
    tangent: np.ndarray,
    lateral: np.ndarray,
    existing_rows: list[np.ndarray],
    forced_hidden_speed: float,
    rand_obs_setting: str,
    forced_validate_occlusion: bool,
    forced_require_corridor_conflict: bool,
    overlap_margin: float,
) -> dict[str, Any] | None:
    occ_xy = np.asarray(occ_xy, dtype=float).reshape(2,)
    observer_xy = np.asarray(observer_xy, dtype=float).reshape(2,)
    event_xy = np.asarray(event_xy, dtype=float).reshape(2,)
    tangent = np.asarray(tangent, dtype=float).reshape(2,)
    lateral = np.asarray(lateral, dtype=float).reshape(2,)
    occ_dir = crowd_narrow._safe_normalize(occ_xy - observer_xy)
    if occ_dir is None:
        return None

    hidden_radius_candidates = [0.32, 0.38]
    gap_pad_candidates = [0.14, 0.22]
    hidden_lat_candidates = [0.0]
    target_along_candidates = [0.0, -0.18]
    target_cross_candidates = [0.0, 0.12]

    for hidden_radius in hidden_radius_candidates:
        for gap_pad in gap_pad_candidates:
            gap = float(occ_radius + hidden_radius + gap_pad)
            for hidden_lat in hidden_lat_candidates:
                hidden_xy = occ_xy + gap * occ_dir + float(hidden_lat) * lateral
                hidden_xy = crowd._clip_to_workspace(hidden_xy, margin=0.8)
                if crowd._rows_overlap(hidden_xy, hidden_radius, existing_rows, margin=float(overlap_margin)):
                    continue
                for target_along in target_along_candidates:
                    for target_cross in target_cross_candidates:
                        target_xy = event_xy + float(target_along) * tangent + float(target_cross) * lateral
                        target_xy = crowd._clip_to_workspace(target_xy, margin=0.5)
                        vel_dir = crowd_narrow._safe_normalize(target_xy - hidden_xy)
                        if vel_dir is None:
                            continue
                        hidden_speed = crowd_narrow._sample_hidden_speed_for_setting(
                            rng,
                            forced_hidden_speed=forced_hidden_speed,
                            rand_obs_setting=rand_obs_setting,
                            legacy_low=0.95,
                            legacy_high=1.05,
                        )
                        hidden_vel = hidden_speed * vel_dir
                        initially_occluded = crowd_narrow._disc_occludes_target(
                            observer_xy,
                            hidden_xy,
                            occ_xy,
                            occ_radius,
                            target_radius=hidden_radius,
                        )
                        if bool(forced_validate_occlusion) and not initially_occluded:
                            continue
                        pred = crowd._simulate_route_forced_event_nominal(
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
                        hidden_row, hidden_meta = crowd._apply_outer_flow_bounds(
                            np.array(
                                [hidden_xy[0], hidden_xy[1], hidden_radius, hidden_vel[0], hidden_vel[1], 0.0, crowd.ENV_HEIGHT, 1.0],
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


def _make_occluder_row(
    *,
    occ_xy: np.ndarray,
    occ_radius: float,
    seg_tangent: np.ndarray,
    side_sign: float,
    static_occluders: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    away_dir = crowd_narrow._safe_normalize(side_sign * np.asarray([-seg_tangent[1], seg_tangent[0]], dtype=float))
    if away_dir is None:
        away_dir = np.array([1.0, 0.0], dtype=float)
    occ_speed = 0.0 if bool(static_occluders) else float(rng.uniform(0.12, 0.22))
    occ_theta = float(np.arctan2(away_dir[1], away_dir[0]) + rng.normal(0.0, 0.18))
    occ_vx = float(occ_speed * np.cos(occ_theta))
    occ_vy = float(occ_speed * np.sin(occ_theta))
    roam_outward = 0.0 if bool(static_occluders) else float(rng.uniform(2.4, 3.6))
    roam_inward = 0.0 if bool(static_occluders) else float(rng.uniform(0.30, 0.65))
    roam_along = 0.0 if bool(static_occluders) else float(rng.uniform(0.35, 0.90))

    lateral = np.array([-seg_tangent[1], seg_tangent[0]], dtype=float)
    away_x = float(abs(lateral[0]))
    away_y = float(abs(lateral[1]))
    tang_x = float(abs(seg_tangent[0]))
    tang_y = float(abs(seg_tangent[1]))
    outward_x = max(0.6, roam_outward * away_x + roam_along * tang_x)
    outward_y = max(0.6, roam_outward * away_y + roam_along * tang_y)
    inward_x = max(0.25, roam_inward * away_x + 0.45 * roam_along * tang_x)
    inward_y = max(0.25, roam_inward * away_y + 0.45 * roam_along * tang_y)

    if lateral[0] * side_sign >= 0.0:
        occ_x_min = occ_xy[0] - inward_x
        occ_x_max = occ_xy[0] + outward_x
    else:
        occ_x_min = occ_xy[0] - outward_x
        occ_x_max = occ_xy[0] + inward_x

    if lateral[1] * side_sign >= 0.0:
        occ_y_min = occ_xy[1] - inward_y
        occ_y_max = occ_xy[1] + outward_y
    else:
        occ_y_min = occ_xy[1] - outward_y
        occ_y_max = occ_xy[1] + inward_y

    occ_x_min = max(0.5, float(occ_x_min))
    occ_x_max = min(crowd.ENV_WIDTH - 0.5, float(occ_x_max))
    occ_y_min = max(0.5, float(occ_y_min))
    occ_y_max = min(crowd.ENV_HEIGHT - 0.5, float(occ_y_max))

    occ_row = np.array(
        [occ_xy[0], occ_xy[1], occ_radius, occ_vx, occ_vy, occ_y_min, occ_y_max, 1.0],
        dtype=float,
    )
    occ_meta = {
        "mode": 0 if bool(static_occluders) else 1,
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
        "forced_drift_gain": 0.25,
    }
    return occ_row, occ_meta


def _place_home_event(
    *,
    spec: dict[str, Any],
    rng: np.random.Generator,
    existing_rows: list[np.ndarray],
    guard_rows: list[np.ndarray],
    forced_hidden_speed: float,
    rand_obs_setting: str,
    forced_validate_occlusion: bool,
    forced_require_corridor_conflict: bool,
    static_occluders: bool,
    forced_occluder_radius_min: float,
    forced_occluder_radius_max: float,
) -> dict[str, Any] | None:
    target_lo, target_hi = tuple(spec["ttc_range"])
    target_mid = 0.5 * (target_lo + target_hi)
    preferred_side_sign = float(spec["side_sign"])
    max_reveal_path_distance = float(spec["max_reveal_path_distance"])

    if target_mid <= 0.95:
        lateral_candidates = [1.00, 1.22]
        lead_candidates = [1.7, 2.2]
    elif target_mid <= 1.20:
        lateral_candidates = [1.08, 1.32]
        lead_candidates = [2.0, 2.6]
    else:
        lateral_candidates = [1.18, 1.48]
        lead_candidates = [2.5, 3.2]
    along_candidates = [0.0, -0.18, 0.18]
    radius_candidates = [
        float(forced_occluder_radius_min),
        float(forced_occluder_radius_max),
    ]

    for side_sign in (preferred_side_sign, -preferred_side_sign):
        for s_offset in (0.0, -1.5, 1.5, -3.0, 3.0):
            s_event = float(np.clip(float(spec["s_event"]) + s_offset, 0.08 * crowd.ROUTE_LENGTH, 0.92 * crowd.ROUTE_LENGTH))
            seg = _route_seg_for_distance(s_event)
            event_xy = crowd._route_point_at_distance(s_event)
            for relax in (0.18,):
                for lateral_mag in lateral_candidates:
                    for along_jitter in along_candidates:
                        for lead_distance in lead_candidates:
                            for occ_radius in radius_candidates:
                                occ_xy = (
                                    np.asarray(event_xy, dtype=float).reshape(2,)
                                    + along_jitter * np.asarray(seg["tangent"], dtype=float).reshape(2,)
                                    + side_sign * lateral_mag * np.asarray(seg["lateral"], dtype=float).reshape(2,)
                                )
                                occ_xy = crowd._clip_to_workspace(occ_xy, margin=1.0)
                                occ_guard_radius = float(occ_radius) + 1.0
                                if crowd._rows_overlap(occ_xy, occ_guard_radius, guard_rows, margin=0.5):
                                    continue
                                if crowd._rows_overlap(occ_xy, float(occ_radius), existing_rows, margin=0.5):
                                    continue

                                observer_dist = float(
                                    np.clip(s_event - lead_distance, float(seg["s0"]), float(seg["s1"]))
                                )
                                observer_xy = crowd._route_point_at_distance(observer_dist)
                                hidden = _make_hidden_candidate_home(
                                    rng=rng,
                                    occ_xy=occ_xy,
                                    occ_radius=occ_radius,
                                    observer_xy=observer_xy,
                                    event_xy=event_xy,
                                    tangent=np.asarray(seg["tangent"], dtype=float).reshape(2,),
                                    lateral=np.asarray(seg["lateral"], dtype=float).reshape(2,),
                                    existing_rows=existing_rows,
                                    forced_hidden_speed=forced_hidden_speed,
                                    rand_obs_setting=rand_obs_setting,
                                    forced_validate_occlusion=forced_validate_occlusion,
                                    forced_require_corridor_conflict=forced_require_corridor_conflict,
                                    overlap_margin=0.16,
                                )
                                if hidden is None:
                                    continue
                                pred = hidden["pred"]
                                pred_ttc = pred.get("predicted_reveal_ttc_nominal_s", None)
                                pred_d_path = pred.get("predicted_reveal_distance_to_path", None)
                                if pred_d_path is None or float(pred_d_path) > (max_reveal_path_distance + relax):
                                    continue
                                if not _ttc_in_range(pred_ttc, (target_lo, target_hi), slack=relax):
                                    continue

                                occ_row, occ_meta = _make_occluder_row(
                                    occ_xy=np.asarray(occ_xy, dtype=float).reshape(2,),
                                    occ_radius=float(occ_radius),
                                    seg_tangent=np.asarray(seg["tangent"], dtype=float).reshape(2,),
                                    side_sign=side_sign,
                                    static_occluders=static_occluders,
                                    rng=rng,
                                )
                                return {
                                    "occ_row": occ_row,
                                    "occ_meta": occ_meta,
                                    "occ_guard_radius": occ_guard_radius,
                                    "occ_xy": np.asarray(occ_xy, dtype=float).reshape(2,),
                                    "occ_radius": float(occ_radius),
                                    "observer_xy": np.asarray(observer_xy, dtype=float).reshape(2,),
                                    "event_xy": np.asarray(event_xy, dtype=float).reshape(2,),
                                    "hidden": hidden,
                                    "pred": pred,
                                    "segment_index": int(seg["idx"]),
                                    "ttc_range_target": [float(target_lo), float(target_hi)],
                                }

    # Fallback: keep the reveal-conditioned hidden-emergence structure even if
    # the exact TTC bucket is hard to hit for this route-bin/side combination.
    for side_sign in (preferred_side_sign, -preferred_side_sign):
        for s_offset in (0.0, -2.5, 2.5):
            s_event = float(np.clip(float(spec["s_event"]) + s_offset, 0.08 * crowd.ROUTE_LENGTH, 0.92 * crowd.ROUTE_LENGTH))
            seg = _route_seg_for_distance(s_event)
            event_xy = crowd._route_point_at_distance(s_event)
            for lateral_mag in lateral_candidates:
                for lead_distance in lead_candidates:
                    for occ_radius in radius_candidates:
                        occ_xy = (
                            np.asarray(event_xy, dtype=float).reshape(2,)
                            + side_sign * lateral_mag * np.asarray(seg["lateral"], dtype=float).reshape(2,)
                        )
                        occ_xy = crowd._clip_to_workspace(occ_xy, margin=1.0)
                        occ_guard_radius = float(occ_radius) + 1.0
                        if crowd._rows_overlap(occ_xy, occ_guard_radius, guard_rows, margin=0.5):
                            continue
                        if crowd._rows_overlap(occ_xy, float(occ_radius), existing_rows, margin=0.5):
                            continue
                        observer_dist = float(
                            np.clip(s_event - lead_distance, float(seg["s0"]), float(seg["s1"]))
                        )
                        observer_xy = crowd._route_point_at_distance(observer_dist)
                        hidden = _make_hidden_candidate_home(
                            rng=rng,
                            occ_xy=occ_xy,
                            occ_radius=occ_radius,
                            observer_xy=observer_xy,
                            event_xy=event_xy,
                            tangent=np.asarray(seg["tangent"], dtype=float).reshape(2,),
                            lateral=np.asarray(seg["lateral"], dtype=float).reshape(2,),
                            existing_rows=existing_rows,
                            forced_hidden_speed=forced_hidden_speed,
                            rand_obs_setting=rand_obs_setting,
                            forced_validate_occlusion=forced_validate_occlusion,
                            forced_require_corridor_conflict=forced_require_corridor_conflict,
                            overlap_margin=0.16,
                        )
                        if hidden is None:
                            continue
                        occ_row, occ_meta = _make_occluder_row(
                            occ_xy=np.asarray(occ_xy, dtype=float).reshape(2,),
                            occ_radius=float(occ_radius),
                            seg_tangent=np.asarray(seg["tangent"], dtype=float).reshape(2,),
                            side_sign=side_sign,
                            static_occluders=static_occluders,
                            rng=rng,
                        )
                        return {
                            "occ_row": occ_row,
                            "occ_meta": occ_meta,
                            "occ_guard_radius": occ_guard_radius,
                            "occ_xy": np.asarray(occ_xy, dtype=float).reshape(2,),
                            "occ_radius": float(occ_radius),
                            "observer_xy": np.asarray(observer_xy, dtype=float).reshape(2,),
                            "event_xy": np.asarray(event_xy, dtype=float).reshape(2,),
                            "hidden": hidden,
                            "pred": hidden["pred"],
                            "segment_index": int(seg["idx"]),
                            "ttc_range_target": [float(target_lo), float(target_hi)],
                        }

    # Last-resort fallback: use the crowd hidden sampler with shallow attempts.
    for side_sign in (preferred_side_sign, -preferred_side_sign):
        for s_offset in (0.0, -2.5, 2.5):
            s_event = float(np.clip(float(spec["s_event"]) + s_offset, 0.08 * crowd.ROUTE_LENGTH, 0.92 * crowd.ROUTE_LENGTH))
            seg = _route_seg_for_distance(s_event)
            event_xy = crowd._route_point_at_distance(s_event)
            for lateral_mag in lateral_candidates:
                for lead_distance in lead_candidates:
                    for occ_radius in radius_candidates:
                        occ_xy = (
                            np.asarray(event_xy, dtype=float).reshape(2,)
                            + side_sign * lateral_mag * np.asarray(seg["lateral"], dtype=float).reshape(2,)
                        )
                        occ_xy = crowd._clip_to_workspace(occ_xy, margin=1.0)
                        occ_guard_radius = float(occ_radius) + 1.0
                        if crowd._rows_overlap(occ_xy, occ_guard_radius, guard_rows, margin=0.5):
                            continue
                        if crowd._rows_overlap(occ_xy, float(occ_radius), existing_rows, margin=0.5):
                            continue
                        observer_dist = float(
                            np.clip(s_event - lead_distance, float(seg["s0"]), float(seg["s1"]))
                        )
                        observer_xy = crowd._route_point_at_distance(observer_dist)
                        hidden = crowd._sample_hidden_for_event(
                            rng=rng,
                            occ_xy=occ_xy,
                            occ_radius=occ_radius,
                            observer_xy=observer_xy,
                            event_xy=event_xy,
                            tangent=np.asarray(seg["tangent"], dtype=float).reshape(2,),
                            lateral=np.asarray(seg["lateral"], dtype=float).reshape(2,),
                            existing_rows=existing_rows,
                            forced_hidden_speed=forced_hidden_speed,
                            rand_obs_setting=rand_obs_setting,
                            forced_validate_occlusion=forced_validate_occlusion,
                            forced_require_corridor_conflict=forced_require_corridor_conflict,
                            attempts=8,
                            lateral_jitter=0.14,
                            target_along_jitter=0.45,
                            target_cross_jitter=0.30,
                            overlap_margin=0.16,
                        )
                        if hidden is None:
                            continue
                        occ_row, occ_meta = _make_occluder_row(
                            occ_xy=np.asarray(occ_xy, dtype=float).reshape(2,),
                            occ_radius=float(occ_radius),
                            seg_tangent=np.asarray(seg["tangent"], dtype=float).reshape(2,),
                            side_sign=side_sign,
                            static_occluders=static_occluders,
                            rng=rng,
                        )
                        return {
                            "occ_row": occ_row,
                            "occ_meta": occ_meta,
                            "occ_guard_radius": occ_guard_radius,
                            "occ_xy": np.asarray(occ_xy, dtype=float).reshape(2,),
                            "occ_radius": float(occ_radius),
                            "observer_xy": np.asarray(observer_xy, dtype=float).reshape(2,),
                            "event_xy": np.asarray(event_xy, dtype=float).reshape(2,),
                            "hidden": hidden,
                            "pred": hidden["pred"],
                            "segment_index": int(seg["idx"]),
                            "ttc_range_target": [float(target_lo), float(target_hi)],
                        }
    return None


def _build_occlusion_home_scenario(
    *,
    case_seed: int,
    case_idx: int | None,
    n_rand: int,
    rand_obs: bool,
    static_occluders: bool,
    crowd_mode: str,
    forced_hidden_speed: float,
    forced_occluder_radius_min: float,
    forced_occluder_radius_max: float,
    forced_validate_occlusion: bool,
    forced_require_corridor_conflict: bool,
    rand_obs_setting: str,
):
    family = str(crowd_mode).strip().lower()
    if family not in HOME_FAMILY_CHOICES:
        raise ValueError(f"Unsupported occlusion-home family `{crowd_mode}`.")

    rand_obs_setting = crowd_narrow._normalize_rand_obs_setting(rand_obs_setting)
    rng = np.random.default_rng(int(case_seed))
    profile = _case_profile(case_idx, family)
    event_specs = _build_event_specs(profile, rng)

    forced_rows: list[np.ndarray] = []
    forced_meta: list[dict[str, Any]] = []
    forced_event_meta: list[dict[str, Any]] = []
    guard_rows: list[np.ndarray] = []

    for spec in event_specs:
        if family == "two_event":
            placed = _manual_home_event_fallback(
                s_event=float(spec["s_event"]),
                side_sign=float(spec.get("side_sign", 1.0)),
                existing_rows=forced_rows,
                guard_rows=guard_rows,
                forced_hidden_speed=forced_hidden_speed,
                rand_obs_setting=rand_obs_setting,
                forced_validate_occlusion=forced_validate_occlusion,
                forced_require_corridor_conflict=forced_require_corridor_conflict,
                static_occluders=static_occluders,
                forced_occluder_radius_min=forced_occluder_radius_min,
                forced_occluder_radius_max=forced_occluder_radius_max,
                rng=rng,
            )
            if placed is None:
                fallback_fracs = (0.48, 0.58) if int(spec.get("event_id", 0)) == 0 else (0.72, 0.82)
                for frac in fallback_fracs:
                    placed = _manual_home_event_fallback(
                        s_event=float(frac * crowd.ROUTE_LENGTH),
                        side_sign=float(spec.get("side_sign", 1.0)),
                        existing_rows=forced_rows,
                        guard_rows=guard_rows,
                        forced_hidden_speed=forced_hidden_speed,
                        rand_obs_setting=rand_obs_setting,
                        forced_validate_occlusion=forced_validate_occlusion,
                        forced_require_corridor_conflict=forced_require_corridor_conflict,
                        static_occluders=static_occluders,
                        forced_occluder_radius_min=forced_occluder_radius_min,
                        forced_occluder_radius_max=forced_occluder_radius_max,
                        rng=rng,
                    )
                    if placed is not None:
                        break
            if placed is None:
                fallback_fracs = (0.48, 0.58) if int(spec.get("event_id", 0)) == 0 else (0.72, 0.82)
                for frac in fallback_fracs:
                    placed = _manual_home_event_fallback(
                        s_event=float(frac * crowd.ROUTE_LENGTH),
                        side_sign=float(spec.get("side_sign", 1.0)),
                        existing_rows=forced_rows,
                        guard_rows=guard_rows,
                        forced_hidden_speed=forced_hidden_speed,
                        rand_obs_setting=rand_obs_setting,
                        forced_validate_occlusion=forced_validate_occlusion,
                        forced_require_corridor_conflict=False,
                        static_occluders=static_occluders,
                        forced_occluder_radius_min=forced_occluder_radius_min,
                        forced_occluder_radius_max=forced_occluder_radius_max,
                        rng=rng,
                    )
                    if placed is not None:
                        pred_d = placed["pred"].get("predicted_reveal_distance_to_path", None)
                        if pred_d is not None and float(pred_d) <= 1.0:
                            break
                        placed = None
        else:
            placed = _place_home_event(
                spec=spec,
                rng=rng,
                existing_rows=forced_rows,
                guard_rows=guard_rows,
                forced_hidden_speed=forced_hidden_speed,
                rand_obs_setting=rand_obs_setting,
                forced_validate_occlusion=forced_validate_occlusion,
                forced_require_corridor_conflict=forced_require_corridor_conflict,
                static_occluders=static_occluders,
                forced_occluder_radius_min=forced_occluder_radius_min,
                forced_occluder_radius_max=forced_occluder_radius_max,
            )
        if placed is None:
            raise RuntimeError(
                f"Failed to place {family} event {spec['event_id']} for case_idx={case_idx} seed={case_seed}."
            )

        occ_idx = len(forced_rows)
        hid_idx = occ_idx + 1
        forced_rows.extend([placed["occ_row"], placed["hidden"]["row"]])
        forced_meta.extend([placed["occ_meta"], placed["hidden"]["meta"]])
        forced_event_meta.append(
            {
                "event_id": int(spec["event_id"]),
                "event_label": str(spec["label"]),
                "benchmark_family": str(family),
                "segment_index": int(placed["segment_index"]),
                "occluder_index": int(occ_idx),
                "hidden_index": int(hid_idx),
                "hidden_indices": [int(hid_idx)],
                "occluder_center": [float(placed["occ_xy"][0]), float(placed["occ_xy"][1])],
                "occluder_radius": float(placed["occ_radius"]),
                "occluder_velocity": [float(placed["occ_row"][3]), float(placed["occ_row"][4])],
                "occluder_speed": float(np.linalg.norm(placed["occ_row"][3:5])),
                "event_xy": [float(placed["event_xy"][0]), float(placed["event_xy"][1])],
                "observer_xy": [float(placed["observer_xy"][0]), float(placed["observer_xy"][1])],
                "hidden_initial_position": [float(placed["hidden"]["hidden_xy"][0]), float(placed["hidden"]["hidden_xy"][1])],
                "hidden_velocity": [float(placed["hidden"]["hidden_vel"][0]), float(placed["hidden"]["hidden_vel"][1])],
                "hidden_speed": float(placed["hidden"]["hidden_speed"]),
                "hidden_radius": float(placed["hidden"]["hidden_radius"]),
                "initially_occluded_geom": bool(placed["hidden"]["initially_occluded"]),
                "corridor_conflict": bool(placed["pred"]["corridor_conflict"]),
                "predicted_reveal_step": placed["pred"]["predicted_reveal_step"],
                "predicted_reveal_time_s": placed["pred"]["predicted_reveal_time_s"],
                "predicted_reveal_distance_to_path": placed["pred"]["predicted_reveal_distance_to_path"],
                "predicted_reveal_ttc_nominal_s": placed["pred"]["predicted_reveal_ttc_nominal_s"],
                "min_predicted_distance_to_path": placed["pred"]["min_predicted_distance_to_path"],
                "min_predicted_ttc_nominal_s": placed["pred"]["min_predicted_ttc_nominal_s"],
                "target_reveal_ttc_range_s": (
                    None
                    if placed.get("ttc_range_target", None) is None
                    else list(placed["ttc_range_target"])
                ),
                "extra_hidden_count": 0,
                "initially_occluded_actual": None,
                "revealed_actual": False,
                "reveal_step_actual": None,
                "reveal_time_actual_s": None,
                "reveal_distance_to_path_actual": None,
                "reveal_ttc_nominal_actual_s": None,
            }
        )
        guard_rows.append(np.array([placed["occ_xy"][0], placed["occ_xy"][1], placed["occ_guard_radius"]], dtype=float))

    event_points = [
        np.asarray(meta["event_xy"], dtype=float).reshape(2,)
        for meta in forced_event_meta
    ]
    bg_target = int(min(
        max(0, int(n_rand)),
        HOME_BG_COUNTS[family][int(profile["clutter_bin"])],
    ))

    bg_rows_8 = np.empty((0, 8), dtype=float)
    bg_meta: list[dict[str, Any]] = []
    if bool(rand_obs) and bg_target > 0:
        bg_v_obs_max, bg_v_obs_min = crowd_narrow._rand_obs_speed_window(
            static_occluders=False,
            rand_obs_setting=rand_obs_setting,
            legacy_speed_max=crowd.LEGACY_ROUTE_DYN_SPEED_MAX,
        )
        keep_rows: list[np.ndarray] = []
        keep_meta: list[dict[str, Any]] = []
        batch_id = 0
        while len(keep_rows) < bg_target and batch_id < crowd.ROUTE_BG_BATCH_LIMIT:
            sample_target = max((bg_target - len(keep_rows)) * 4, bg_target)
            extra_rows, extra_meta = crowd_narrow.LocalTrackingControllerDyn_OCC.make_random_obstacles7(
                n_rand=int(sample_target),
                v_obs_max=bg_v_obs_max,
                v_obs_min=bg_v_obs_min,
                x_range=(1.0, crowd.ENV_WIDTH - 1.0),
                y_spawn_range=(1.0, crowd.ENV_HEIGHT - 1.0),
                r_range=(0.3, 0.4),
                y_bounds=(0.0, crowd.ENV_HEIGHT),
                seed=int(case_seed) + 7919 * int(batch_id),
                rand_obs=bool(rand_obs),
            )
            for row, meta in zip(extra_rows, extra_meta):
                if len(keep_rows) >= bg_target:
                    break
                row = np.asarray(row, dtype=float)
                existing = forced_rows + keep_rows
                if not _background_candidate_valid_home(
                    row,
                    existing,
                    event_points,
                    min_route_clearance=HOME_BG_ROUTE_CLEARANCE[family],
                    start_clearance=2.4,
                    min_event_clearance=HOME_BG_EVENT_CLEARANCE[family],
                ):
                    continue
                row, meta = crowd._apply_outer_flow_bounds(row, meta)
                keep_rows.append(row)
                keep_meta.append(dict(meta))
            batch_id += 1
        if len(keep_rows) < bg_target:
            raise RuntimeError(
                f"Failed to place occlusion-home background clutter to target count: need {bg_target}, placed {len(keep_rows)}."
            )
        bg_rows = np.vstack(keep_rows)
        bg_rows_8 = np.hstack((bg_rows, np.ones((bg_rows.shape[0], 1), dtype=float)))
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
        "crowd_mode": str(family),
        "benchmark_family": str(family),
        "benchmark_profile": profile,
        "rand_obs_setting": str(rand_obs_setting),
        "n_forced_events": int(len(forced_event_meta)),
        "n_forced_hidden_total": int(len(forced_event_meta)),
        "n_forced_extra_hidden": 0,
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
    occ_visible_scale=None,
    occ_enable_visible_hocbf=None,
    oa_dynamic_occluders=None,
    oa_allow_solver_fallback=None,
    oa_dsafe=None,
    oa_visible_reach_mode=None,
    oa_use_nominal_tracking_cost=None,
    oa_wmax="default",
    oa_dt=None,
    crowd_mode="single_event",
    forced_events=1,
    forced_bg_rand=None,
    forced_hidden_speed=None,
    forced_occluder_radius_min=crowd.DEFAULT_FORCED_OCCLUDER_RADIUS_MIN,
    forced_occluder_radius_max=crowd.DEFAULT_FORCED_OCCLUDER_RADIUS_MAX,
    forced_validate_occlusion=True,
    forced_require_corridor_conflict=True,
    rand_obs_setting=crowd_narrow.DEFAULT_RAND_OBS_SETTING,
    static_occluders=True,
    backup_cbf_overrides=None,
    robot_spec_overrides=None,
    return_metrics=False,
    max_steps=None,
    max_sim_time=None,
):
    case_seed = crowd_narrow._compute_case_seed(seed, case_idx)
    mode = str(crowd_mode).strip().lower()
    if mode not in HOME_FAMILY_CHOICES:
        raise ValueError(
            f"Unsupported crowd_mode `{crowd_mode}` for occlusion-home. Use one of {HOME_FAMILY_CHOICES}."
        )
    rand_obs_setting = crowd_narrow._normalize_rand_obs_setting(rand_obs_setting)
    if forced_hidden_speed is None:
        forced_hidden_speed = crowd._resolve_forced_hidden_speed(None, rand_obs_setting)
    else:
        forced_hidden_speed = float(forced_hidden_speed)
        if (
            rand_obs_setting == crowd_narrow.CURRENT_RAND_OBS_SETTING
            and abs(forced_hidden_speed - 0.5) <= 1e-9
        ):
            # The shared benchmark wrapper defaults to 0.5 for historical
            # crowd sweeps. Occlusion-home uses the current route-emergence
            # default hidden speed unless the caller explicitly chooses
            # something else.
            forced_hidden_speed = float(crowd.DEFAULT_FORCED_HIDDEN_SPEED)

    known_obs, obs_meta, scenario_diag = _build_occlusion_home_scenario(
        case_seed=case_seed,
        case_idx=case_idx,
        n_rand=n_rand,
        rand_obs=rand_obs,
        static_occluders=bool(static_occluders),
        crowd_mode=mode,
        forced_hidden_speed=forced_hidden_speed,
        forced_occluder_radius_min=forced_occluder_radius_min,
        forced_occluder_radius_max=forced_occluder_radius_max,
        forced_validate_occlusion=forced_validate_occlusion,
        forced_require_corridor_conflict=forced_require_corridor_conflict,
        rand_obs_setting=rand_obs_setting,
    )

    return crowd_narrow.run_crowd_scenario(
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
        occ_visible_scale=occ_visible_scale,
        occ_enable_visible_hocbf=occ_enable_visible_hocbf,
        oa_dynamic_occluders=oa_dynamic_occluders,
        oa_allow_solver_fallback=oa_allow_solver_fallback,
        oa_dsafe=oa_dsafe,
        oa_visible_reach_mode=oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=oa_use_nominal_tracking_cost,
        oa_wmax=oa_wmax,
        oa_dt=oa_dt,
        crowd_mode="forced_emergence",
        forced_events=int(scenario_diag.get("n_forced_events", forced_events) or forced_events),
        forced_bg_rand=forced_bg_rand,
        forced_hidden_speed=forced_hidden_speed,
        forced_occluder_radius_min=forced_occluder_radius_min,
        forced_occluder_radius_max=forced_occluder_radius_max,
        forced_validate_occlusion=forced_validate_occlusion,
        forced_require_corridor_conflict=forced_require_corridor_conflict,
        static_occluders=bool(static_occluders),
        backup_cbf_overrides=backup_cbf_overrides,
        robot_spec_overrides=robot_spec_overrides,
        waypoints_override=crowd.ROUTE_WAYPOINTS,
        env_width_override=crowd.ENV_WIDTH,
        env_height_override=crowd.ENV_HEIGHT,
        known_obs_override=known_obs,
        obs_meta_override=obs_meta,
        scenario_diag_override=scenario_diag,
        return_metrics=return_metrics,
        max_steps=max_steps,
        max_sim_time=max_sim_time,
        tracking_view_enable=True,
        tracking_view_window_size=crowd.TRACKING_VIEW_WINDOW_SIZE,
        scenario_name="OcclusionHome",
        hide_env_boundary=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run reveal-conditioned occlusion-home benchmark scenario."
    )
    parser.add_argument("--model", type=str, default="di", choices=["di", "du", "uni"])
    parser.add_argument("--algo", type=str, default="occlusion_cbf_qp", choices=CROWD_ALGO_CHOICES)
    parser.add_argument("--baseline", type=str, default=None, choices=CROWD_BASELINE_CHOICES)
    parser.add_argument("--tf", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--idx", "--case-idx", dest="case_idx", type=int, default=1)
    parser.add_argument("--n-rand", type=int, default=10)
    parser.add_argument("--no-rand-obs", action="store_true")
    parser.add_argument("--disable-plot", action="store_true")
    parser.add_argument(
        "--crowd-mode",
        type=str,
        default="single_event",
        choices=list(HOME_FAMILY_CHOICES),
        help="Occlusion-home benchmark family.",
    )
    parser.add_argument(
        "--rand-obs-setting",
        type=str,
        default=crowd_narrow.DEFAULT_RAND_OBS_SETTING,
        choices=[crowd_narrow.LEGACY_RAND_OBS_SETTING, crowd_narrow.CURRENT_RAND_OBS_SETTING],
    )
    parser.add_argument("--forced-hidden-speed", type=float, default=None)
    parser.add_argument("--forced-occluder-radius-min", type=float, default=crowd.DEFAULT_FORCED_OCCLUDER_RADIUS_MIN)
    parser.add_argument("--forced-occluder-radius-max", type=float, default=crowd.DEFAULT_FORCED_OCCLUDER_RADIUS_MAX)
    parser.add_argument("--forced-validate-occlusion", type=crowd_narrow._str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--forced-require-corridor-conflict", type=crowd_narrow._str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--static-occluders", type=crowd_narrow._str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--occ-visible-scale", type=float, default=None)
    parser.add_argument(
        "--occ-enable-visible-hocbf",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override the tuned Occlusion-CBF visible-obstacle HOCBF setting.",
    )
    parser.add_argument("--oa-dynamic-occluders", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-allow-solver-fallback", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-dsafe", type=float, default=None)
    parser.add_argument("--oa-visible-reach-mode", type=str, choices=["worst_case", "constant_velocity"], default=None)
    parser.add_argument("--oa-use-nominal-tracking-cost", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-dt", type=float, default=None)
    parser.add_argument("--wmax", type=str, choices=["default", "pi"], default="default")
    parser.add_argument(
        "--vref",
        type=str,
        choices=OCBF_VREF_FRONT_MODES,
        default=None,
        help="OCBF front-facet direction mode. Internal default is `los`; `default` keeps the fixed polygon normal.",
    )
    parser.add_argument("--occ-t-horizon", type=float, default=None)
    parser.add_argument("--occ-rho-T", type=str, default=None)
    parser.add_argument("--occ-dt-backup", type=float, default=None)
    parser.add_argument("--occ-terminal-slack-weight", type=float, default=None)
    parser.add_argument("--occ-terminal-slack-max", type=float, default=None)
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
    parser.add_argument("--occ-qp-failure-fallback-mode", type=str, choices=OCBF_QP_FAILURE_FALLBACK_MODES, default=None)
    parser.add_argument("--occ-vref-scenario-softmax-kappa", type=float, default=None)
    parser.add_argument(
        "--occ-vref-scenario-weight-mode",
        type=str,
        choices=OCBF_VREF_SCENARIO_WEIGHT_MODES,
        default=None,
        help=(
            "Override OCBF scenario blending score. barrier_expand uses rollout-expanded "
            "margins; barrier_unexpand uses unexpanded current-geometry margins."
        ),
    )
    parser.add_argument("--occ-max-active-occlusions", type=int, default=None)
    parser.add_argument("--occ-selection-mode", type=str, choices=OCBF_SELECTION_MODES, default=None)
    parser.add_argument("--occ-kappa", type=float, default=None)
    args = parser.parse_args()

    pos_algo = resolve_baseline_alias(args.baseline, args.algo, CROWD_BASELINE_MAP)
    controller_type = {"pos": pos_algo}

    backup_cbf_overrides: dict[str, Any] = {}
    if args.occ_t_horizon is not None:
        backup_cbf_overrides["T_horizon"] = float(args.occ_t_horizon)
    if args.occ_rho_T is not None:
        rho_raw = str(args.occ_rho_T).strip()
        rho_key = rho_raw.lower()
        if rho_key in {"auto", "auto_stop", "stop", "stopping_distance"}:
            backup_cbf_overrides["rho_T"] = rho_key
        else:
            backup_cbf_overrides["rho_T"] = float(rho_raw)
    if args.occ_dt_backup is not None:
        backup_cbf_overrides["dt_backup"] = float(args.occ_dt_backup)
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
    if args.occ_vref_scenario_softmax_kappa is not None:
        backup_cbf_overrides["vref_scenario_softmax_kappa"] = float(args.occ_vref_scenario_softmax_kappa)
    if args.occ_vref_scenario_weight_mode is not None:
        backup_cbf_overrides["vref_scenario_weight_mode"] = str(args.occ_vref_scenario_weight_mode).strip().lower()
    if args.occ_max_active_occlusions is not None:
        backup_cbf_overrides["max_active_occlusions"] = int(args.occ_max_active_occlusions)
    if args.occ_selection_mode is not None:
        backup_cbf_overrides["occ_selection_mode"] = str(args.occ_selection_mode).strip().lower()
    if args.vref is not None:
        backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    if not backup_cbf_overrides:
        backup_cbf_overrides = None

    robot_spec_overrides: dict[str, Any] = {}
    if args.occ_kappa is not None:
        robot_spec_overrides["occ_kappa"] = float(args.occ_kappa)
    if not robot_spec_overrides:
        robot_spec_overrides = None

    run_crowd_scenario(
        controller_type=controller_type,
        model_key=args.model,
        show_animation=not args.disable_plot,
        save_animation=False,
        tf=args.tf,
        seed=args.seed,
        case_idx=args.case_idx,
        rand_obs=(not args.no_rand_obs),
        n_rand=args.n_rand,
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
        forced_hidden_speed=args.forced_hidden_speed,
        forced_occluder_radius_min=args.forced_occluder_radius_min,
        forced_occluder_radius_max=args.forced_occluder_radius_max,
        forced_validate_occlusion=args.forced_validate_occlusion,
        forced_require_corridor_conflict=args.forced_require_corridor_conflict,
        rand_obs_setting=args.rand_obs_setting,
        static_occluders=args.static_occluders,
        backup_cbf_overrides=backup_cbf_overrides,
        robot_spec_overrides=robot_spec_overrides,
    )


if __name__ == "__main__":
    main()
