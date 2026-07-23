"""
Deterministic explanatory visualization for occlusionCBF backup rollout logic.

This script is intentionally not a benchmark runner. It builds one fixed robot
state and one or two hand-designed occluding obstacles, then uses the actual
repo occlusionCBF logic to visualize:

    occlusion geometry
    -> facetwise expansion
    -> scenario direction synthesis
    -> common executable backup reference
    -> closed-loop backup rollout
    -> sampled trajectory points used in OCBF constraints

Examples:
    uv run python examples/test_vis_ocbf.py --mode single_common --save-animation true
    uv run python examples/test_vis_ocbf.py --mode multi_common --save-animation true
    uv run python examples/test_vis_ocbf.py --mode single_common --save-svg true --snapshot-name single_fig
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib

if ("--show" not in sys.argv) and ("--show-animation" not in sys.argv) and ("--show_animation" not in sys.argv):
    matplotlib.use("Agg")

import matplotlib.patheffects as pe
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from examples._runtime import ensure_repo_root, load_local_occ_controller

ensure_repo_root()
LocalTrackingControllerDyn_OCC = load_local_occ_controller("vis_ocbf")

from base_control.utils import env


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "mathtext.default": "it",
    }
)


BG_COLOR = "#f5f6fb"
TEXT_COLOR = "#24324a"
ROBOT_FILL = "#b3cde3"
ROBOT_EDGE = "#f7fafc"
ROBOT_VEL = "#6b7f9c"
OBS_FILL = "#808080"
OBS_EDGE = "#000000"
POLY_BASE = "#5b5fef"
POLY_ALT = "#7c4dff"
ACTIVE_FACET = "#ff6b6b"
INACTIVE_FACET = "#a9b2c5"
REP_NORMAL_A = "#f2994a"
REP_NORMAL_B = "#f2c94c"
COMMON_VREF = "#ffd166"
ROLLOUT_COLOR = "tab:green"
CURRENT_POINT = "#fff4a3"
TERMINAL_COLOR = "#c96a28"
SAMPLE_DOT = "#f8fafc"
VECTOR_DRAW_SCALE = 1.0
ANNOTATION_TEXT_SCALE = 1.5


def _str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _phase_progress(u: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if u >= end else 0.0
    return _smoothstep((u - start) / (end - start))


def _safe_normalize(vec, eps: float = 1e-9):
    arr = np.asarray(vec, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= eps:
        return None
    return arr / nrm


def _stroke(width: float = 3.0, color: str = "white"):
    return [pe.Stroke(linewidth=width, foreground=color), pe.Normal()]


def _afs(size: float) -> float:
    return float(size) * ANNOTATION_TEXT_SCALE


def _text(ax, xy, text, **kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("color", TEXT_COLOR)
    kwargs.setdefault("fontsize", _afs(12.5))
    kwargs.setdefault("zorder", 40)
    return ax.text(xy[0], xy[1], text, **kwargs)


def _ordered_polygon_from_halfspaces(A, b, eps: float = 1e-8):
    """
    Recover vertices from cyclic half-space rows A z <= b.
    First tries adjacent-line intersections, then falls back to pairwise
    intersection filtering if needed.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m = int(A.shape[0])
    if m < 3:
        return None

    pts = []
    for i in range(m):
        j = (i + 1) % m
        M = np.vstack([A[i], A[j]])
        det = float(np.linalg.det(M))
        if abs(det) <= 1e-9:
            pts = []
            break
        x = np.linalg.solve(M, np.array([b[i], b[j]], dtype=float))
        if np.all(A @ x <= b + 1e-6):
            pts.append(x)
        else:
            pts = []
            break

    if len(pts) == m:
        return np.asarray(pts, dtype=float)

    pts = []
    for i in range(m):
        for j in range(i + 1, m):
            M = np.vstack([A[i], A[j]])
            det = float(np.linalg.det(M))
            if abs(det) <= 1e-9:
                continue
            x = np.linalg.solve(M, np.array([b[i], b[j]], dtype=float))
            if np.all(A @ x <= b + 1e-6):
                pts.append(x)

    if not pts:
        return None

    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < eps for q in uniq):
            uniq.append(p)
    pts = np.asarray(uniq, dtype=float)
    if pts.shape[0] < 3:
        return None
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]


def _expanded_polygon_at_tau(scenario, robot_radius: float, tau: float, extra_margin: float = 0.0):
    A = np.asarray(scenario["A"], dtype=float)
    b0 = np.asarray(scenario["b0"], dtype=float).reshape(-1)
    v_expand = np.asarray(
        scenario.get("v_expand_vec", np.full(len(b0), float(scenario["v_adv_max"]))),
        dtype=float,
    ).reshape(-1)
    b_tau = b0 + float(robot_radius) + float(extra_margin) + v_expand * float(tau)
    return _ordered_polygon_from_halfspaces(A, b_tau)


def _edge_midpoints(poly):
    poly = np.asarray(poly, dtype=float)
    mids = []
    for i in range(poly.shape[0]):
        p1 = poly[i]
        p2 = poly[(i + 1) % poly.shape[0]]
        mids.append(0.5 * (p1 + p2))
    return np.asarray(mids, dtype=float)


def _label_anchor(poly, mode: str = "top"):
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if poly.shape[0] == 0:
        return np.zeros(2, dtype=float)
    if mode == "top":
        idx = int(np.argmax(poly[:, 1]))
    elif mode == "bottom":
        idx = int(np.argmin(poly[:, 1]))
    elif mode == "right":
        idx = int(np.argmax(poly[:, 0]))
    elif mode == "left":
        idx = int(np.argmin(poly[:, 0]))
    else:
        idx = int(np.argmax(poly[:, 1]))
    return np.asarray(poly[idx], dtype=float)


def _inset_corner_label_point(poly, corner: str = "upper_right", frac: float = 0.18):
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if poly.shape[0] == 0:
        return np.zeros(2, dtype=float)
    centroid = np.mean(poly, axis=0)
    if corner == "upper_right":
        score = poly[:, 0] + poly[:, 1]
        idx = int(np.argmax(score))
    elif corner == "upper_left":
        score = -poly[:, 0] + poly[:, 1]
        idx = int(np.argmax(score))
    elif corner == "lower_right":
        score = poly[:, 0] - poly[:, 1]
        idx = int(np.argmax(score))
    else:
        score = -poly[:, 0] - poly[:, 1]
        idx = int(np.argmax(score))
    p = np.asarray(poly[idx], dtype=float)
    frac = float(np.clip(frac, 0.0, 0.95))
    return p + frac * (centroid - p)


def _lower_right_facet_index(anchors):
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
    if anchors.shape[0] == 0:
        return None
    order = np.lexsort((-anchors[:, 0], anchors[:, 1]))
    return int(order[0])


def _facet_anchor_points(A, b, poly, tol: float = 1e-6):
    """
    Recover one anchor point per half-space row A_k x <= b_k on the actual
    corresponding polygon facet.

    Important:
    The polygon vertex order obtained from half-space intersection is not
    guaranteed to align index-wise with the half-space row order. Using raw
    edge midpoints by index can therefore attach a_k to the wrong edge.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    poly = np.asarray(poly, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2 or poly.ndim != 2 or poly.shape[1] != 2:
        return None
    if poly.shape[0] < 3 or A.shape[0] == 0:
        return None

    anchors = []
    residual = A @ poly.T - b[:, None]

    for k in range(A.shape[0]):
        on_facet = np.where(np.abs(residual[k]) <= tol)[0]
        if on_facet.size >= 2:
            pts = poly[on_facet[:2]]
            anchors.append(np.mean(pts, axis=0))
            continue

        best_pair = None
        best_score = np.inf
        n_poly = poly.shape[0]
        for i in range(n_poly):
            j = (i + 1) % n_poly
            score = abs(residual[k, i]) + abs(residual[k, j])
            if score < best_score:
                best_score = score
                best_pair = (i, j)

        if best_pair is None:
            anchors.append(np.mean(poly, axis=0))
        else:
            i, j = best_pair
            anchors.append(0.5 * (poly[i] + poly[j]))

    return np.asarray(anchors, dtype=float)


def _facet_segments(A, b, poly, tol: float = 1e-6):
    """
    Recover one line segment per half-space row using the polygon vertices that
    lie on that facet. This is used only for clean visual highlighting.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    poly = np.asarray(poly, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2 or poly.ndim != 2 or poly.shape[1] != 2:
        return None
    if poly.shape[0] < 3 or A.shape[0] == 0:
        return None

    residual = A @ poly.T - b[:, None]
    segments = []
    n_poly = poly.shape[0]

    for k in range(A.shape[0]):
        on_facet = np.where(np.abs(residual[k]) <= tol)[0]
        if on_facet.size >= 2:
            pts = poly[on_facet[:2]]
            segments.append(np.asarray(pts, dtype=float))
            continue

        best_pair = None
        best_score = np.inf
        for i in range(n_poly):
            j = (i + 1) % n_poly
            score = abs(residual[k, i]) + abs(residual[k, j])
            if score < best_score:
                best_score = score
                best_pair = (i, j)

        if best_pair is None:
            c = np.mean(poly, axis=0)
            segments.append(np.vstack([c, c]))
        else:
            i, j = best_pair
            segments.append(np.vstack([poly[i], poly[j]]))

    return segments


def _selected_facet_mask(dbg):
    active_mask = np.asarray(dbg.get("active_mask", []), dtype=bool)
    if active_mask.size == 0:
        return active_mask
    if np.any(active_mask):
        return active_mask

    mask = np.zeros_like(active_mask)
    facet_weights = dbg.get("facet_weights", None)
    if facet_weights is not None:
        facet_weights = np.asarray(facet_weights, dtype=float).reshape(-1)
        if facet_weights.size == active_mask.size:
            mask[int(np.argmax(facet_weights))] = True
            return mask

    h_vec = np.asarray(dbg.get("h_vec", []), dtype=float).reshape(-1)
    if h_vec.size == active_mask.size and h_vec.size > 0:
        mask[int(np.argmax(h_vec))] = True
    return mask


def _ghost_tau_levels(current_tau: float, T: float, max_count: int = 2):
    if T <= 1e-9 or current_tau <= 1e-9 or max_count <= 0:
        return []
    offsets = [0.18, 0.08, 0.28]
    levels = []
    for frac in offsets[:max_count]:
        tau = float(current_tau) - frac * float(T)
        if tau > 0.04 * float(T):
            levels.append(tau)
    return levels


def _poly_closed(poly):
    poly = np.asarray(poly, dtype=float)
    return np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]]


def _interpolate_rollout_state(traj, t_grid, tau: float):
    traj = np.asarray(traj, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    if traj.shape[0] == 0 or t_grid.size == 0:
        return traj[:0], np.zeros(2, dtype=float), 0

    tau = float(np.clip(tau, float(t_grid[0]), float(t_grid[-1])))
    if tau <= float(t_grid[0]) + 1e-12:
        pt = np.asarray(traj[0], dtype=float)
        return traj[:1], pt, 0
    if tau >= float(t_grid[-1]) - 1e-12:
        pt = np.asarray(traj[-1], dtype=float)
        return traj, pt, traj.shape[0] - 1

    idx_hi = int(np.searchsorted(t_grid, tau, side="right"))
    idx_lo = max(0, idx_hi - 1)
    t0 = float(t_grid[idx_lo])
    t1 = float(t_grid[idx_hi])
    denom = max(t1 - t0, 1e-9)
    alpha = (tau - t0) / denom
    pt = (1.0 - alpha) * traj[idx_lo] + alpha * traj[idx_hi]
    seg = np.vstack([traj[: idx_lo + 1], pt])
    return seg, np.asarray(pt, dtype=float), idx_lo


def _interpolate_full_state(traj, t_grid, tau: float):
    traj = np.asarray(traj, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    if traj.shape[0] == 0 or t_grid.size == 0:
        return None
    tau = float(np.clip(tau, float(t_grid[0]), float(t_grid[-1])))
    if tau <= float(t_grid[0]) + 1e-12:
        return np.asarray(traj[0], dtype=float).reshape(-1)
    if tau >= float(t_grid[-1]) - 1e-12:
        return np.asarray(traj[-1], dtype=float).reshape(-1)

    idx_hi = int(np.searchsorted(t_grid, tau, side="right"))
    idx_lo = max(0, idx_hi - 1)
    t0 = float(t_grid[idx_lo])
    t1 = float(t_grid[idx_hi])
    alpha = (tau - t0) / max(t1 - t0, 1e-9)
    state = (1.0 - alpha) * traj[idx_lo] + alpha * traj[idx_hi]
    return np.asarray(state, dtype=float).reshape(-1)


def _phase_state(scene: SceneData, u: float):
    if scene.mode == "single_common":
        scene_alpha = _phase_progress(u, 0.00, 0.12)
        inflate_alpha = _phase_progress(u, 0.10, 0.30)
        direction_alpha = _phase_progress(u, 0.28, 0.48)
        rollout_prog = _phase_progress(u, 0.46, 0.84)
        terminal_alpha = _phase_progress(u, 0.82, 1.00)
    else:
        scene_alpha = _phase_progress(u, 0.00, 0.12)
        inflate_alpha = _phase_progress(u, 0.10, 0.30)
        direction_alpha = _phase_progress(u, 0.26, 0.46)
        rollout_prog = _phase_progress(u, 0.44, 0.86)
        terminal_alpha = _phase_progress(u, 0.84, 1.00)

    T = float(scene.controller.pos_controller.T_horizon)
    current_tau = T * rollout_prog
    if terminal_alpha > 0.0:
        current_tau = T

    normal_alpha = np.clip(
        (direction_alpha * (1.0 - 0.75 * rollout_prog) + 0.10 * rollout_prog) * (1.0 - 0.92 * terminal_alpha),
        0.0,
        1.0,
    )
    vref_alpha = np.clip(0.10 + 0.90 * max(direction_alpha, rollout_prog), 0.0, 1.0)
    return {
        "scene_alpha": float(scene_alpha),
        "inflate_alpha": float(inflate_alpha),
        "direction_alpha": float(direction_alpha),
        "rollout_alpha": float(rollout_prog),
        "terminal_alpha": float(terminal_alpha),
        "normal_alpha": float(normal_alpha),
        "vref_alpha": float(vref_alpha),
        "current_tau": float(current_tau),
    }


def _scenario_speed_mag(scene: SceneData, scenario: dict) -> float:
    try:
        s = float(scenario.get("v_adv_max", scene.robot_spec.get("v_adv_max_occ", scene.robot_spec.get("v_obs_max", 0.5))))
    except Exception:
        s = 0.5
    if not np.isfinite(s):
        s = 0.5
    return max(s, 0.0)


def _compute_scene_bounds(x0_vis, visible_obs, occlusion_scenarios, robot_radius: float, T: float, backup_traj, target_aspect: float = 1.42):
    focus_pts = [np.asarray(x0_vis[:2], dtype=float).reshape(1, 2), np.asarray(backup_traj[:, :2], dtype=float)]

    for obs in visible_obs:
        c = np.asarray(obs[:2], dtype=float).reshape(2,)
        r = float(obs[2])
        focus_pts.append(np.array([[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]], dtype=float))

    context_pts = []
    for sc in occlusion_scenarios:
        poly0 = np.asarray(sc.get("poly", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        if poly0.size > 0:
            context_pts.append(poly0)
        polyT = _expanded_polygon_at_tau(sc, robot_radius, T)
        if polyT is not None and np.asarray(polyT).size > 0:
            context_pts.append(np.asarray(polyT, dtype=float).reshape(-1, 2))

    focus_all = np.vstack(focus_pts)
    min_xy = np.min(focus_all, axis=0)
    max_xy = np.max(focus_all, axis=0)

    pad = np.array([0.9, 0.9], dtype=float)
    min_xy -= pad
    max_xy += pad

    focus_width = max(float(max_xy[0] - min_xy[0]), 1e-6)
    focus_height = max(float(max_xy[1] - min_xy[1]), 1e-6)

    if context_pts:
        ctx_all = np.vstack(context_pts)
        ctx_min = np.min(ctx_all, axis=0)
        ctx_max = np.max(ctx_all, axis=0)

        # Keep some occlusion context, but do not let far-right expanded wedges
        # dominate the framing. The scene should stay centered on the ego,
        # obstacle, and executable backup rollout.
        right_extra = 1.0 * focus_width
        y_extra = 1.0 * focus_height

        min_xy[0] = min(min_xy[0], float(ctx_min[0]) - 0.12)
        max_xy[0] = max(max_xy[0], min(float(ctx_max[0]) + 0.12, float(max_xy[0] + right_extra)))
        min_xy[1] = min(min_xy[1], max(float(ctx_min[1]) - 0.12, float(min_xy[1] - y_extra)))
        max_xy[1] = max(max_xy[1], min(float(ctx_max[1]) + 0.12, float(max_xy[1] + y_extra)))

    width = max(float(max_xy[0] - min_xy[0]), 1e-6)
    height = max(float(max_xy[1] - min_xy[1]), 1e-6)
    current_aspect = width / height

    if current_aspect < target_aspect:
        desired_width = target_aspect * height
        grow = 0.5 * (desired_width - width)
        min_xy[0] -= grow
        max_xy[0] += grow
    else:
        desired_height = width / target_aspect
        grow = 0.5 * (desired_height - height)
        min_xy[1] -= grow
        max_xy[1] += grow

    max_xy[1] += 0.045 * max(float(max_xy[1] - min_xy[1]), 1e-6)
    return (float(min_xy[0]), float(max_xy[0])), (float(min_xy[1]), float(max_xy[1]))


def _make_robot_spec(args, sensing_range: float):
    return {
        "model": "DoubleIntegrator2D",
        "v_max": 1.0,
        "a_max": 1.0,
        "v_obs_max": float(args.v_adv_max_occ),
        "v_adv_max_occ": float(args.v_adv_max_occ),
        "radius": 0.25,
        "debug_backup_qp": False,
        "cbf_feas_tol": 5e-3,
        "sensing_range": float(sensing_range),
        "fov_angle": 360.0,
        "backup_cbf": {
            "T_horizon": float(args.T_horizon),
            "dt_backup": float(args.dt_backup),
            "vref_scenario_softmax_kappa": float(args.vref_scenario_softmax_kappa),
            "rho_T": "auto",
            "vref_front_mode_occ": "los",
        },
        "show_backup_rollout": False,
        "backup_rollout_every": 1,
        "use_occ": True,
        "dynamic_obs_types": [1],
        "occlusion_types": [1],
        "occ_visible_scale": 1.0,
        "occ_kappa": float(args.occ_kappa),
    }


def _make_controller(x0_vis, args, sensing_range: float):
    robot_spec = _make_robot_spec(args, sensing_range=sensing_range)
    controller = LocalTrackingControllerDyn_OCC(
        np.asarray(x0_vis, dtype=float),
        robot_spec,
        controller_type={"pos": "occlusion_cbf_qp"},
        dt=float(args.dt_backup),
        show_animation=False,
        save_animation=False,
        env=env.Env(),
        rand_seed=0,
    )
    return controller


@dataclass
class SceneData:
    mode: str
    title: str
    controller: Any
    robot_spec: dict
    obs: np.ndarray
    obs_meta: list[dict]
    visible_obs: list[np.ndarray]
    occlusion_scenarios: list[dict]
    visible_indices: list[int]
    backup_traj: np.ndarray
    stm_traj: np.ndarray
    t_grid: np.ndarray
    fcl_traj: np.ndarray
    x0_vis: np.ndarray
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    scenario_colors: list[str]
    terminal_margins: list[float]
    terminal_rho: float
    terminal_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None


def _build_scene_common(mode: str, args):
    x0_vis = np.array([2.0, 1.0, 1.5, 1.5, 0.0], dtype=float)
    sensing_range = 10.0
    controller = _make_controller(x0_vis, args, sensing_range=sensing_range)

    if mode == "single_common":
        obs = np.array(
            [
                [7.0, 2.0, 0.70, 0.0, 0.0, -4.0, 4.0, 1.0],
            ],
            dtype=float,
        )
        title = "Occlusion Backup Rollout"
        expected_scenarios = 1
        x_bounds = None
        y_bounds = None
        scenario_colors = [POLY_BASE]
    elif mode == "multi_common":
        obs = np.array(
            [
                [2.30, 1.10, 0.65, 0.0, 0.0, -4.0, 4.0, 1.0],
                [2.40, -1.20, 0.65, 0.0, 0.0, -4.0, 4.0, 1.0],
            ],
            dtype=float,
        )
        title = "Occlusion Backup Rollout"
        expected_scenarios = 2
        x_bounds = None
        y_bounds = None
        scenario_colors = [POLY_BASE, POLY_ALT]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    obs_meta = []
    for row in obs:
        theta0 = float(np.arctan2(row[4], row[3])) if np.hypot(row[3], row[4]) > 1e-9 else 0.0
        obs_meta.append({"mode": 1, "v_max": 0.0, "theta": theta0})
    controller.obs = np.asarray(obs, dtype=float).copy()
    controller.set_obs_meta(obs_meta)

    visible_obs, occlusion_scenarios, visible_indices = controller.pos_controller._occ_utils._filter_visible_and_build_occ(
        controller.robot.X,
        controller.obs,
        return_indices=True,
    )

    if len(occlusion_scenarios) != expected_scenarios:
        raise RuntimeError(
            f"{mode}: expected {expected_scenarios} occlusion scenario(s), got {len(occlusion_scenarios)}. "
            f"Adjust scene geometry."
        )
    if len(visible_obs) != expected_scenarios:
        raise RuntimeError(
            f"{mode}: expected {expected_scenarios} visible occluding obstacle(s), got {len(visible_obs)}."
        )

    backup = controller.pos_controller.occlusion_backup
    T = float(controller.pos_controller.T_horizon)
    dt_backup = float(controller.pos_controller.dt_backup)
    backup_traj, stm_traj, t_grid, fcl_traj = backup.simulate_backup_trajectory(
        controller.robot.X,
        T,
        dt_backup,
        occlusion_scenarios=occlusion_scenarios,
    )

    terminal_rho = float(controller.pos_controller._resolve_terminal_rho())
    terminal_margins = []
    xT = backup_traj[-1].reshape(-1, 1)
    for sc in occlusion_scenarios:
        backup.set_terminal_backup_context(sc, T, kappa=controller.pos_controller.kappa, rho_T=terminal_rho)
        terminal_margins.append(float(backup._occ_terminal_set(xT)))

    terminal_field = _build_terminal_field(
        controller,
        occlusion_scenarios,
        backup_traj[-1],
        T,
        terminal_rho,
        args.grid_res,
        radius_box=2.0,
    )

    x_bounds, y_bounds = _compute_scene_bounds(
        controller.robot.X.reshape(-1),
        [np.asarray(v, dtype=float) for v in visible_obs],
        occlusion_scenarios,
        float(controller.robot_spec["radius"]),
        T,
        np.asarray(backup_traj, dtype=float),
        target_aspect=float(args.figure_width) / max(float(args.figure_height), 1e-6),
    )

    return SceneData(
        mode=mode,
        title=title,
        controller=controller,
        robot_spec=controller.robot_spec,
        obs=obs,
        obs_meta=obs_meta,
        visible_obs=[np.asarray(v, dtype=float) for v in visible_obs],
        occlusion_scenarios=occlusion_scenarios,
        visible_indices=list(visible_indices),
        backup_traj=np.asarray(backup_traj, dtype=float),
        stm_traj=np.asarray(stm_traj, dtype=float),
        t_grid=np.asarray(t_grid, dtype=float),
        fcl_traj=np.asarray(fcl_traj, dtype=float),
        x0_vis=np.asarray(controller.robot.X, dtype=float).reshape(-1),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        scenario_colors=scenario_colors,
        terminal_margins=terminal_margins,
        terminal_rho=terminal_rho,
        terminal_field=terminal_field,
    )


def build_single_common_scene(args):
    return _build_scene_common("single_common", args)


def build_multi_common_scene(args):
    return _build_scene_common("multi_common", args)


def _build_terminal_field(controller, scenarios, x_final, T: float, terminal_rho: float, grid_res: float, radius_box: float = 1.0):
    backup = controller.pos_controller.occlusion_backup
    x_final = np.asarray(x_final, dtype=float).reshape(-1)
    x0, y0 = float(x_final[0]), float(x_final[1])
    xs = np.arange(x0 - radius_box, x0 + radius_box + 0.5 * grid_res, grid_res)
    ys = np.arange(y0 - radius_box, y0 + radius_box + 0.5 * grid_res, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    H = np.full_like(XX, np.nan, dtype=float)

    template = np.asarray(x_final, dtype=float).reshape(-1, 1)
    for iy in range(YY.shape[0]):
        for ix in range(XX.shape[1]):
            X = template.copy()
            X[0, 0] = float(XX[iy, ix])
            X[1, 0] = float(YY[iy, ix])
            margins = []
            for sc in scenarios:
                backup.set_terminal_backup_context(sc, T, kappa=controller.pos_controller.kappa, rho_T=terminal_rho)
                h_val = backup._occ_terminal_set(X)
                if h_val is not None and np.isfinite(h_val):
                    margins.append(float(h_val))
            if margins:
                H[iy, ix] = float(min(margins))
    if not np.any(np.isfinite(H)):
        return None
    return XX, YY, H


def compute_scenario_direction_data(scene: SceneData, tau: float):
    backup = scene.controller.pos_controller.occlusion_backup
    x_query = _interpolate_full_state(scene.backup_traj, scene.t_grid, tau)
    if x_query is None:
        x_query = np.asarray(scene.backup_traj[0], dtype=float).reshape(-1)
    X_query = np.asarray(x_query, dtype=float).reshape(-1, 1)
    p = np.asarray(X_query[:2, 0], dtype=float).reshape(2,)
    robot_radius = float(scene.robot_spec["radius"])
    tau = float(np.clip(tau, 0.0, float(scene.controller.pos_controller.T_horizon)))
    use_strict = bool(backup._use_strict_occ_vref())
    scenario_kappa = float(backup._occ_vref_scenario_softmax_kappa())
    soft_facet_kappa = float(scene.robot_spec.get("backup_cbf", {}).get("vref_softmax_kappa", 8.0))
    if (not np.isfinite(soft_facet_kappa)) or soft_facet_kappa <= 1e-6:
        soft_facet_kappa = 8.0

    scenario_debug = []
    scenario_scores = []
    v_targets = []
    valid_mask = []

    for sc_idx, sc in enumerate(scene.occlusion_scenarios):
        A = np.asarray(sc["A"], dtype=float)
        b0 = np.asarray(sc["b0"], dtype=float).reshape(-1)
        vadv = float(sc["v_adv_max"])
        A_target = backup._occ_target_normals(sc, p)
        v_expand = np.asarray(sc.get("v_expand_vec", np.full(len(b0), vadv)), dtype=float).reshape(-1)

        delta = robot_radius + v_expand * tau
        h_vec = (A @ p) - b0 - delta
        barrier_margin = backup._occ_scenario_barrier_margin_from_hvec(h_vec)
        threat_score = backup._occ_scenario_threat_score_from_hvec(h_vec)

        if use_strict:
            active_mask = h_vec >= 0.0
            if np.any(active_mask):
                rep_normal = np.mean(A_target[active_mask], axis=0)
                facet_weights = None
                rep_source = "active_mean"
            else:
                best_idx = int(np.argmax(h_vec))
                rep_normal = np.asarray(A_target[best_idx], dtype=float)
                facet_weights = None
                rep_source = "argmax"
        else:
            max_h = float(np.max(h_vec))
            z = np.exp(soft_facet_kappa * (h_vec - max_h))
            z_sum = float(np.sum(z))
            if (not np.isfinite(z_sum)) or z_sum <= 1e-12:
                facet_weights = np.zeros_like(h_vec)
            else:
                facet_weights = z / z_sum
            rep_normal = np.sum(facet_weights[:, None] * A_target, axis=0)
            active_mask = facet_weights > (1.0 / max(10.0 * len(h_vec), 100.0))
            rep_source = "softmax"

        rep_dir = _safe_normalize(rep_normal)
        if rep_dir is None:
            rep_dir = np.zeros(2, dtype=float)
            v_target = np.zeros(2, dtype=float)
            valid = False
        else:
            v_target = rep_dir * vadv
            valid = threat_score is not None and np.isfinite(threat_score)

        scenario_debug.append(
            {
                "index": sc_idx,
                "scenario": sc,
                "A_target": A_target,
                "h_vec": np.asarray(h_vec, dtype=float),
                "active_mask": np.asarray(active_mask, dtype=bool),
                "facet_weights": None if facet_weights is None else np.asarray(facet_weights, dtype=float),
                "rep_normal": np.asarray(rep_normal, dtype=float),
                "rep_dir": np.asarray(rep_dir, dtype=float),
                "rep_source": rep_source,
                "v_target": np.asarray(v_target, dtype=float),
                "barrier_margin": None if barrier_margin is None else float(barrier_margin),
                "threat_score": None if threat_score is None else float(threat_score),
            }
        )

        scenario_scores.append(-np.inf if threat_score is None else float(threat_score))
        v_targets.append(np.asarray(v_target, dtype=float))
        valid_mask.append(bool(valid))

    weights = np.zeros(len(scene.occlusion_scenarios), dtype=float)
    if len(scenario_scores) > 0:
        scores = np.asarray(scenario_scores, dtype=float)
        valid = np.asarray(valid_mask, dtype=bool)
        if np.any(valid):
            max_score = float(np.max(scores[valid]))
            z = np.zeros_like(scores)
            z[valid] = np.exp(scenario_kappa * (scores[valid] - max_score))
            z_sum = float(np.sum(z))
            if z_sum > 1e-12 and np.isfinite(z_sum):
                weights = z / z_sum

    for dbg, w in zip(scenario_debug, weights):
        dbg["scenario_weight"] = float(w)

    if v_targets:
        common_vref_debug = (weights[:, None] * np.vstack(v_targets)).sum(axis=0)
    else:
        common_vref_debug = np.zeros(2, dtype=float)

    common_vref_actual = np.asarray(
        backup._occ_safe_velocity_reference_rollout(X_query, scene.occlusion_scenarios, tau),
        dtype=float,
    ).reshape(2,)
    if not np.allclose(common_vref_debug, common_vref_actual, atol=1e-7, rtol=1e-6):
        raise RuntimeError(
            "Debug v_ref reconstruction does not match backup controller output: "
            f"{common_vref_debug} vs {common_vref_actual}"
        )

    return {
        "tau": tau,
        "query_state": X_query,
        "query_position": p,
        "scenario_debug": scenario_debug,
        "weights": weights,
        "common_vref": common_vref_actual,
    }


def build_occ_debug_at_tau(scene: SceneData, tau: float):
    data = compute_scenario_direction_data(scene, tau)
    tau = float(data["tau"])
    raw_expanded_polys = []
    expanded_polys = []
    for dbg in data["scenario_debug"]:
        raw_poly_tau = _expanded_polygon_at_tau(dbg["scenario"], 0.0, tau)
        poly_tau = _expanded_polygon_at_tau(dbg["scenario"], scene.robot_spec["radius"], tau)
        raw_expanded_polys.append(raw_poly_tau)
        expanded_polys.append(poly_tau)
    data["raw_expanded_polys"] = raw_expanded_polys
    data["expanded_polys"] = expanded_polys
    return data


def compute_terminal_debug(scene: SceneData):
    margins = np.asarray(scene.terminal_margins, dtype=float)
    return {
        "per_scenario_margin": margins,
        "min_margin": float(np.min(margins)) if margins.size else np.nan,
        "terminal_point": np.asarray(scene.backup_traj[-1, :2], dtype=float),
    }


def _draw_arrow(ax, start, vec, color, lw=2.8, mutation_scale=16.0, zorder=15, alpha=1.0, linestyle="-"):
    start = np.asarray(start, dtype=float).reshape(2,)
    vec = np.asarray(vec, dtype=float).reshape(2,)
    arrow = mpatches.FancyArrowPatch(
        (float(start[0]), float(start[1])),
        (float(start[0] + vec[0]), float(start[1] + vec[1])),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    arrow.set_path_effects(_stroke(1.4, "white"))
    ax.add_patch(arrow)
    return arrow


def _draw_robot_stamp(ax, state, radius: float, alpha: float, zorder: float):
    state = np.asarray(state, dtype=float).reshape(-1)
    center = np.array([float(state[0]), float(state[1])], dtype=float)
    vel = np.asarray(state[2:4], dtype=float).reshape(2,)
    v_norm = float(np.linalg.norm(vel))
    yaw = float(np.arctan2(vel[1], vel[0])) if v_norm > 1e-9 else 0.0

    robot = mpatches.Circle(
        center,
        float(radius),
        facecolor=ROBOT_FILL,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.78 * alpha,
        zorder=zorder,
    )
    ax.add_patch(robot)

    orient_len = 0.5
    orient_end = center + orient_len * np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
    ax.plot(
        [center[0], orient_end[0]],
        [center[1], orient_end[1]],
        color="r",
        linewidth=1.7,
        alpha=0.90,
        zorder=zorder + 0.2,
        solid_capstyle="round",
        path_effects=_stroke(0.9, "#ffffff"),
    )


def draw_base_scene(
    ax,
    scene: SceneData,
    alpha_robot: float,
    alpha_scene: float,
    labels: bool = True,
    show_entity_labels: bool = True,
    show_base_robot: bool = True,
):
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(*scene.x_bounds)
    ax.set_ylim(*scene.y_bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Workspace frame
    frame = mpatches.FancyBboxPatch(
        (scene.x_bounds[0], scene.y_bounds[0]),
        scene.x_bounds[1] - scene.x_bounds[0],
        scene.y_bounds[1] - scene.y_bounds[0],
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=BG_COLOR,
        edgecolor="#d6dceb",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(frame)

    for obs in scene.visible_obs:
        circ = mpatches.Circle(
            (float(obs[0]), float(obs[1])),
            float(obs[2]),
            facecolor=OBS_FILL,
            edgecolor=OBS_EDGE,
            linewidth=1.8,
            alpha=0.90 * alpha_scene,
            zorder=10,
        )
        circ.set_path_effects(_stroke(1.0, "#f9fafb"))
        ax.add_patch(circ)

    if labels and show_entity_labels and alpha_scene > 0.18:
        n_vis = len(scene.visible_obs)
        for i, obs in enumerate(scene.visible_obs):
            center_obs = np.array([float(obs[0]), float(obs[1])], dtype=float)
            r_obs = float(obs[2])
            if n_vis == 1:
                obs_label = "detected\nobs"
            else:
                obs_label = f"detected\nobs {i+1}"
            _text(
                ax,
                center_obs + np.array([0.0, -0.2], dtype=float),
                obs_label,
                ha="center",
                fontsize=_afs(10),
                alpha=0.90 * alpha_scene,
            )

    if show_base_robot:
        x = scene.x0_vis
        center = np.array([float(x[0]), float(x[1])], dtype=float)
        robot = mpatches.Circle(
            center,
            float(scene.robot_spec["radius"]),
            facecolor=ROBOT_FILL,
            edgecolor=ROBOT_EDGE,
            linewidth=2.2,
            alpha=0.96 * alpha_robot,
            zorder=30,
        )
        robot.set_path_effects(_stroke(1.0, "#ffffff"))
        ax.add_patch(robot)

        v = np.asarray(x[2:4], dtype=float).reshape(2,)
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-9:
            yaw = float(np.arctan2(v[1], v[0]))
        else:
            yaw = float(scene.controller.robot.get_orientation())
        orient_len = float(getattr(scene.controller.robot, "vis_orient_len", 0.5))
        orient_end = center + orient_len * np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        ax.plot(
            [center[0], orient_end[0]],
            [center[1], orient_end[1]],
            color="r",
            linewidth=2.0,
            alpha=0.92,
            zorder=31,
            solid_capstyle="round",
            path_effects=_stroke(1.2, "#ffffff"),
        )

        if labels and show_entity_labels and alpha_robot > 0.18:
            _text(ax, center + np.array([-0.02, -0.6]), "ego robot", ha="center", fontsize=_afs(10), alpha=0.92 * alpha_robot)


def draw_occlusion_geometry(
    ax,
    scene: SceneData,
    occ_debug: dict,
    current_tau: float,
    show_expanded: bool = True,
    show_normals: bool = True,
    labels: bool = True,
    phase_alpha: float = 1.0,
    inflate_alpha: float = 0.0,
    direction_alpha: float = 0.0,
    rollout_alpha: float = 0.0,
    normal_alpha: float = 0.0,
    active_facet_alpha: float = 0.0,
):
    T = float(scene.controller.pos_controller.T_horizon)
    scenario_debug = occ_debug["scenario_debug"]
    raw_expanded_polys = occ_debug["raw_expanded_polys"]
    expanded_polys = occ_debug["expanded_polys"]

    for sc_idx, dbg in enumerate(scenario_debug):
        color = scene.scenario_colors[sc_idx % len(scene.scenario_colors)]
        sc = dbg["scenario"]
        raw_poly0 = np.asarray(sc["poly"], dtype=float).reshape(-1, 2)
        poly0 = _expanded_polygon_at_tau(sc, float(scene.robot_spec["radius"]), 0.0)
        if poly0 is None:
            poly0 = raw_poly0
        poly0 = np.asarray(poly0, dtype=float).reshape(-1, 2)

        xr, yr = _poly_closed(raw_poly0)
        patch_raw = mpatches.Polygon(
            raw_poly0,
            closed=True,
            facecolor="none",
            edgecolor=color,
            linewidth=1.55,
            linestyle=(0, (5.5, 3.5)),
            alpha=(0.58 if scene.mode == "single_common" else 0.48) * phase_alpha,
            zorder=2,
        )
        patch_raw.set_path_effects(_stroke(0.7, "#ffffff"))
        ax.add_patch(patch_raw)

        if labels:
            label_alpha = 0.88 * max(phase_alpha, rollout_alpha, direction_alpha, 0.55 * inflate_alpha)
            pos_O0 = _inset_corner_label_point(raw_poly0, corner="upper_right", frac=0.16)
            raw_label = rf"$O_0^{{({sc_idx+1})}}$" if len(scenario_debug) > 1 else r"$O_0$"
            _text(
                ax,
                pos_O0 + np.array([-0.05, -0.03]),
                raw_label,
                ha="right",
                va="top",
                fontsize=_afs(10),
                alpha=label_alpha,
            )

        x0, y0 = _poly_closed(poly0)
        patch0 = mpatches.Polygon(
            poly0,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.8,
            alpha=(0.085 if scene.mode == "single_common" else 0.06) * (0.95 - 0.30 * rollout_alpha) * phase_alpha * max(inflate_alpha, 0.12),
            zorder=3.2,
        )
        ax.add_patch(patch0)
        ax.plot(
            x0,
            y0,
            color=color,
            linewidth=1.75,
            alpha=(0.56 if scene.mode == "single_common" else 0.44) * phase_alpha * max(inflate_alpha, 0.12),
            zorder=4.2,
        )

        barrier_poly_now = poly0

        if scene.mode == "single_common" and inflate_alpha > 0.08 and raw_poly0.shape[0] >= 3 and poly0.shape[0] >= 3:
            A = np.asarray(sc["A"], dtype=float)
            b_raw = np.asarray(sc["b0"], dtype=float).reshape(-1)
            b_infl = b_raw + float(scene.robot_spec["radius"])
            anchors_raw = _facet_anchor_points(A, b_raw, raw_poly0)
            anchors_infl = _facet_anchor_points(A, b_infl, poly0)
            if anchors_raw is not None and anchors_infl is not None and len(anchors_raw) > 0 and len(anchors_infl) > 0:
                k = _lower_right_facet_index(anchors_infl)
                if k is None:
                    k = 0
                if k < len(anchors_raw) and k < len(anchors_infl):
                    p0 = np.asarray(anchors_raw[k], dtype=float)
                    p1 = np.asarray(anchors_infl[k], dtype=float)
                    n = np.asarray(A[k], dtype=float).reshape(2,)
                    n_norm = float(np.linalg.norm(n))
                    if n_norm > 1e-9:
                        n = n / n_norm
                        t_hat = np.array([-n[1], n[0]], dtype=float)
                        robot_pos = np.asarray(scene.x0_vis[:2], dtype=float).reshape(2,)
                        shift_mag = 0.18
                        cand_a = 0.5 * (p0 + p1) + shift_mag * t_hat
                        cand_b = 0.5 * (p0 + p1) - shift_mag * t_hat
                        shift_vec = shift_mag * t_hat if np.linalg.norm(cand_a - robot_pos) >= np.linalg.norm(cand_b - robot_pos) else -shift_mag * t_hat
                        q0 = p0 + shift_vec
                        q1 = p1 + shift_vec
                        arrow = mpatches.FancyArrowPatch(
                            (float(q0[0]), float(q0[1])),
                            (float(q1[0]), float(q1[1])),
                            arrowstyle="<->",
                            mutation_scale=10.0,
                            linewidth=1.8,
                            linestyle="-",
                            color="#111111",
                            alpha=0.90 * inflate_alpha * phase_alpha,
                            zorder=4.8,
                            shrinkA=0.0,
                            shrinkB=0.0,
                        )
                        arrow.set_path_effects(_stroke(1.1, "#ffffff"))
                        ax.add_patch(arrow)
                        if labels:
                            pm = 0.5 * (q0 + q1)
                            _text(
                                ax,
                                pm + np.array([0.12, 0.0], dtype=float),
                                r"$r_{rob}$",
                                ha="left",
                                va="center",
                                fontsize=_afs(9.2),
                                color="#111111",
                                alpha=0.92 * inflate_alpha * phase_alpha,
                            )

        if show_expanded and current_tau > 1e-9:
            ghost_levels = _ghost_tau_levels(current_tau, T, max_count=0 if scene.mode == "single_common" else 0)
            for ghost_idx, tau_level in enumerate(ghost_levels):
                poly_tau = _expanded_polygon_at_tau(sc, 0.0, tau_level)
                if poly_tau is None:
                    continue
                xg, yg = _poly_closed(poly_tau)
                ax.plot(
                    xg,
                    yg,
                    color=color,
                    linewidth=1.2 if ghost_idx == 0 else 0.9,
                    linestyle=(0, (4.5, 3.0)),
                    alpha=(0.16 if ghost_idx == 0 else 0.09) * max(rollout_alpha, 0.20) * phase_alpha,
                    zorder=2.5,
                )

            raw_poly_tau = raw_expanded_polys[sc_idx]
            if raw_poly_tau is not None:
                raw_poly_tau = np.asarray(raw_poly_tau, dtype=float).reshape(-1, 2)
                xr_tau, yr_tau = _poly_closed(raw_poly_tau)
                ax.plot(
                    xr_tau,
                    yr_tau,
                    color=color,
                    linewidth=1.55 if scene.mode == "single_common" else 1.35,
                    linestyle=(0, (5.5, 3.5)),
                    alpha=(0.80 if scene.mode == "single_common" else 0.68) * max(rollout_alpha, 0.18) * phase_alpha,
                    zorder=3.1,
                )
                if labels and scene.mode == "single_common" and rollout_alpha > 0.18 and current_tau < (T - 1e-3):
                    pos_Otau = _inset_corner_label_point(raw_poly_tau, corner="upper_right", frac=0.16)
                    _text(
                        ax,
                        pos_Otau + np.array([-0.05, -0.03]),
                        r"$O_\tau$",
                        ha="right",
                        va="top",
                        fontsize=_afs(10),
                        alpha=0.88 * rollout_alpha,
                    )

            barrier_poly_now = expanded_polys[sc_idx]
            if barrier_poly_now is not None:
                poly_tau = np.asarray(barrier_poly_now, dtype=float).reshape(-1, 2)
                xc, yc = _poly_closed(poly_tau)
                ax.fill(
                    xc,
                    yc,
                    facecolor=color,
                    edgecolor="none",
                    alpha=(0.11 if scene.mode == "single_common" else 0.08) * max(rollout_alpha, 0.22) * phase_alpha,
                    zorder=3.5,
                )
                ax.plot(
                    xc,
                    yc,
                    color=color,
                    linewidth=2.4 if scene.mode == "single_common" else 2.0,
                    alpha=(0.86 if scene.mode == "single_common" else 0.76) * max(rollout_alpha, 0.18) * phase_alpha,
                    zorder=4,
                )

        if labels and scene.mode == "single_common":
            barrier_alpha = max(0.92 * inflate_alpha * (1.0 - 0.55 * rollout_alpha), 0.86 * rollout_alpha)
            if barrier_poly_now is not None and barrier_alpha > 0.08:
                barrier_poly_now = np.asarray(barrier_poly_now, dtype=float).reshape(-1, 2)
                c_h = _label_anchor(barrier_poly_now, mode="bottom")
                if current_tau < (T - 1e-3):
                    h_label = r"$h^C$"
                else:
                    h_label = r"$h^C$"
                _text(
                    ax,
                    c_h + np.array([0.08, -0.10]),
                    h_label,
                    ha="left",
                    va="top",
                    fontsize=_afs(10),
                    alpha=min(0.90, barrier_alpha) * phase_alpha,
                )

    if (not show_normals) or normal_alpha <= 1e-3:
        return

    for sc_idx, dbg in enumerate(scenario_debug):
        sc = dbg["scenario"]
        A = np.asarray(sc["A"], dtype=float)
        rep_dir = np.asarray(dbg["rep_dir"], dtype=float)
        v_mag = _scenario_speed_mag(scene, sc)
        poly_tau = expanded_polys[sc_idx]
        if poly_tau is None:
            continue
        b_tau = dbg["scenario"]["b0"] + scene.robot_spec["radius"] + np.asarray(
            dbg["scenario"].get("v_expand_vec", np.full(A.shape[0], float(dbg["scenario"]["v_adv_max"]))),
            dtype=float,
        ) * float(occ_debug["tau"])
        facet_anchors = _facet_anchor_points(A, b_tau, poly_tau)
        if facet_anchors is None:
            facet_anchors = _edge_midpoints(poly_tau)
        facet_segments = _facet_segments(A, b_tau, poly_tau)
        selected_mask_raw = _selected_facet_mask(dbg)
        highlight_active = float(occ_debug["tau"]) < (float(scene.controller.pos_controller.T_horizon) - 1e-6)
        if highlight_active:
            selected_mask = selected_mask_raw
            selected_idx = np.where(selected_mask_raw)[0]
        else:
            selected_mask = np.zeros_like(selected_mask_raw, dtype=bool)
            selected_idx = np.array([], dtype=int)
        rep_anchor = np.mean(facet_anchors[np.where(selected_mask_raw)[0]], axis=0) if np.any(selected_mask_raw) else np.mean(facet_anchors, axis=0)
        rep_color = REP_NORMAL_A if sc_idx == 0 else REP_NORMAL_B
        n_show = min(len(facet_anchors), A.shape[0])

        if scene.mode == "single_common":
            for k in range(n_show):
                anchor = facet_anchors[k]
                is_active = bool(selected_mask[k]) if k < selected_mask.size else False
                color = ACTIVE_FACET if is_active else INACTIVE_FACET
                scale = VECTOR_DRAW_SCALE * v_mag
                alpha_k = (0.86 if is_active else 0.48) * normal_alpha
                lw_k = 2.0 if is_active else 1.55
                _draw_arrow(
                    ax,
                    anchor,
                    scale * A[k],
                    color,
                    lw=lw_k,
                    mutation_scale=10.5 if is_active else 9.2,
                    zorder=9 if is_active else 7,
                    alpha=alpha_k,
                )

        for k in selected_idx:
            if facet_segments is not None and k < len(facet_segments):
                seg = np.asarray(facet_segments[k], dtype=float)
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    color=ACTIVE_FACET,
                    linewidth=3.1,
                    alpha=0.92 * max(normal_alpha, active_facet_alpha),
                    zorder=8.5,
                    solid_capstyle="round",
                    path_effects=_stroke(2.0, "#ffffff"),
                )

        if scene.mode != "single_common":
            _draw_arrow(
                ax,
                rep_anchor,
                VECTOR_DRAW_SCALE * v_mag * rep_dir,
                rep_color,
                lw=2.5,
                mutation_scale=14.0,
                zorder=14,
                alpha=0.88 * normal_alpha,
            )

        if labels and scene.mode == "single_common" and direction_alpha > 0.35:
            pass


def draw_normals_and_targets(
    ax,
    scene: SceneData,
    occ_debug: dict,
    current_tau: float,
    labels: bool = True,
    phase_alpha: float = 1.0,
    rollout_alpha: float = 0.0,
):
    current_state = _interpolate_full_state(scene.backup_traj, scene.t_grid, float(current_tau))
    if current_state is None:
        robot_pos = np.asarray(scene.x0_vis[:2], dtype=float)
    else:
        robot_pos = np.asarray(current_state[:2], dtype=float).reshape(2,)
    scenario_debug = occ_debug["scenario_debug"]
    common_vref = np.asarray(occ_debug["common_vref"], dtype=float)

    if scene.mode == "multi_common":
        for sc_idx, dbg in enumerate(scenario_debug):
            v_target = np.asarray(dbg["v_target"], dtype=float)
            offset = np.array([0.0, -0.18 * (sc_idx - 0.5 * (len(scenario_debug) - 1))])
            origin = robot_pos + offset
            color = REP_NORMAL_A if sc_idx == 0 else REP_NORMAL_B
            _draw_arrow(
                ax,
                origin,
                VECTOR_DRAW_SCALE * v_target,
                color,
                lw=2.0,
                mutation_scale=13.0,
                zorder=19,
                alpha=0.70 * phase_alpha,
                linestyle=(0, (4.0, 3.0)),
            )

    _draw_arrow(
        ax,
        robot_pos,
        VECTOR_DRAW_SCALE * common_vref,
        COMMON_VREF,
        lw=2.0 if scene.mode == "single_common" else 2.8,
        mutation_scale=10.5 if scene.mode == "single_common" else 18.0,
        zorder=34,
        alpha=0.96 * phase_alpha,
    )
    if labels:
        vref_norm = float(np.linalg.norm(common_vref))
        if vref_norm > 1e-9:
            vref_dir = common_vref / vref_norm
            head = robot_pos + VECTOR_DRAW_SCALE * common_vref
            normal = np.array([-vref_dir[1], vref_dir[0]], dtype=float)
            label_pos = head + 0.15 * vref_dir + 0.2 * normal
        else:
            label_pos = robot_pos + np.array([0.10, 0.10], dtype=float)
        _text(ax, label_pos, r"$v_{ref}$", ha="right", va="bottom", fontsize=_afs(10.5), alpha=0.92 * phase_alpha)


def draw_rollout(
    ax,
    scene: SceneData,
    current_tau: float,
    labels: bool = True,
    phase_alpha: float = 1.0,
    terminal_alpha: float = 0.0,
):
    full_traj = np.asarray(scene.backup_traj, dtype=float)
    traj = np.asarray(full_traj[:, :2], dtype=float)
    t_grid = np.asarray(scene.t_grid, dtype=float).reshape(-1)
    T = float(t_grid[-1]) if t_grid.size else 0.0
    if T <= 0.0:
        return

    current_tau = float(np.clip(current_tau, 0.0, T))
    seg, current_pt, idx_lo = _interpolate_rollout_state(traj, t_grid, current_tau)
    current_state = _interpolate_full_state(full_traj, t_grid, current_tau)
    if current_state is None:
        current_state = np.asarray(full_traj[min(idx_lo, full_traj.shape[0] - 1)], dtype=float)
    robot_radius = float(scene.robot_spec["radius"])
    stamp_times = [0.0]
    if current_tau > 0.5:
        stamp_times.extend(list(np.arange(0.5, max(current_tau - 0.24, 0.5) + 1e-6, 0.5)))
    if stamp_times:
        denom = max(len(stamp_times) - 1, 1)
        for i, tau_s in enumerate(stamp_times):
            state_s = _interpolate_full_state(full_traj, t_grid, float(tau_s))
            if state_s is None:
                continue
            tau_frac = float(np.clip(tau_s / max(current_tau, 1e-6), 0.0, 1.0))
            alpha_s = phase_alpha * (0.28 + 0.34 * tau_frac**1.20)
            _draw_robot_stamp(ax, state_s, robot_radius, alpha=alpha_s, zorder=22.0 + 0.2 * (i / denom))

    ax.plot(
        seg[:, 0],
        seg[:, 1],
        color=ROLLOUT_COLOR,
        linewidth=2.35,
        alpha=0.98 * phase_alpha,
        zorder=25,
        solid_capstyle="round",
    )

    _draw_robot_stamp(ax, current_state, robot_radius, alpha=max(0.96 * phase_alpha, 0.35), zorder=32.0)

    if labels and phase_alpha > 0.42:
        offset = np.array([-1.16, 0.24]) if scene.mode == "single_common" else np.array([0.02, 0.28])
        _text(ax, current_pt + offset, r"$\phi_b(x,\tau)$", ha="left", va="bottom")

    if terminal_alpha > 1e-3:
        terminal_pt = traj[-1]


def draw_terminal_annotation(
    ax,
    scene: SceneData,
    labels: bool = True,
    phase_alpha: float = 1.0,
    show_smooth_terminal_contour: bool = False,
):
    term = compute_terminal_debug(scene)
    T = float(scene.controller.pos_controller.T_horizon)
    rho_T = float(scene.terminal_rho)
    occ_debug_T = build_occ_debug_at_tau(scene, T)

    # Draw the raw expanded geometry at tau = T as a geometric reference.
    for sc_idx, sc in enumerate(scene.occlusion_scenarios):
        color = scene.scenario_colors[sc_idx % len(scene.scenario_colors)]
        poly_T = _expanded_polygon_at_tau(sc, 0.0, T)
        if poly_T is None:
            continue
        poly_T = np.asarray(poly_T, dtype=float).reshape(-1, 2)
        if poly_T.shape[0] < 3:
            continue
        xTg, yTg = _poly_closed(poly_T)
        ax.fill(
            xTg,
            yTg,
            facecolor=color,
            edgecolor="none",
            alpha=0.08 * phase_alpha,
            zorder=10.5,
        )
        ax.plot(
            xTg,
            yTg,
            color=color,
            linewidth=2.0 if scene.mode == "single_common" else 1.7,
            linestyle=(0, (6.0, 3.0)),
            alpha=0.72 * phase_alpha,
            zorder=11.0,
        )
        if labels and scene.mode == "single_common":
            aT = _label_anchor(poly_T, mode="top")
            pos_OT = _inset_corner_label_point(poly_T, corner="upper_right", frac=0.16)
            _text(ax, pos_OT + np.array([-0.05, -0.03]), r"$O_T$", ha="right", va="top", fontsize=_afs(10), alpha=0.82 * phase_alpha)

        # Draw the terminal reference geometry associated with h^S(x)=h^C(x,T)-rho_T.
        poly_hc_T = _expanded_polygon_at_tau(sc, float(scene.robot_spec["radius"]), T)
        poly_term = _expanded_polygon_at_tau(sc, float(scene.robot_spec["radius"]), T, extra_margin=rho_T)
        if poly_hc_T is not None:
            poly_hc_T = np.asarray(poly_hc_T, dtype=float).reshape(-1, 2)
        if poly_term is not None:
            poly_term = np.asarray(poly_term, dtype=float).reshape(-1, 2)
            if poly_term.shape[0] >= 3:
                xtm, ytm = _poly_closed(poly_term)
                ax.fill(
                    xtm,
                    ytm,
                    facecolor=TERMINAL_COLOR,
                    edgecolor="none",
                    alpha=0.05 * phase_alpha,
                    zorder=11.4,
                )
                ax.plot(
                    xtm,
                    ytm,
                    color=TERMINAL_COLOR,
                    linewidth=2.6 if scene.mode == "single_common" else 2.1,
                    linestyle="-",
                    alpha=0.92 * phase_alpha,
                    zorder=12.2,
                    solid_capstyle="round",
                    path_effects=_stroke(1.6, "#ffffff"),
                )
                if labels and scene.mode == "single_common":
                    cterm = _label_anchor(poly_term, mode="bottom")
                    _text(
                        ax,
                        cterm + np.array([0.10, -0.12]),
                        r"$h^S = 0$",
                        ha="left",
                        va="top",
                        fontsize=_afs(9.6),
                        alpha=0.90 * phase_alpha,
                    )

                # Show rho_T on the lower side facet, keeping the label to the right of the arrow.
                if scene.mode == "single_common" and sc_idx < len(occ_debug_T["scenario_debug"]):
                    dbg_T = occ_debug_T["scenario_debug"][sc_idx]
                    A = np.asarray(sc["A"], dtype=float)
                    b0 = np.asarray(sc["b0"], dtype=float).reshape(-1)
                    v_expand = np.asarray(
                        sc.get("v_expand_vec", np.full(len(b0), float(sc["v_adv_max"]))),
                        dtype=float,
                    ).reshape(-1)
                    b_T = b0 + float(scene.robot_spec["radius"]) + v_expand * T
                    b_term = b_T + rho_T
                    anchors_T = _facet_anchor_points(A, b_T, poly_hc_T)
                    anchors_term = _facet_anchor_points(A, b_term, poly_term)

                    if anchors_T is not None and anchors_term is not None:
                        k = _lower_right_facet_index(anchors_T)
                        if k is None:
                            k = 0
                        if k < len(anchors_T) and k < len(anchors_term):
                            p0 = np.asarray(anchors_T[k], dtype=float)
                            p1 = np.asarray(anchors_term[k], dtype=float)
                            n = np.asarray(A[k], dtype=float).reshape(2,)
                            n_norm = float(np.linalg.norm(n))
                            if n_norm > 1e-9:
                                n = n / n_norm
                            t_hat = np.array([-n[1], n[0]], dtype=float)
                            robot_pos = np.asarray(scene.x0_vis[:2], dtype=float).reshape(2,)
                            shift_mag = 0.22
                            cand_a = 0.5 * (p0 + p1) + shift_mag * t_hat
                            cand_b = 0.5 * (p0 + p1) - shift_mag * t_hat
                            if np.linalg.norm(cand_a - robot_pos) >= np.linalg.norm(cand_b - robot_pos):
                                shift_vec = shift_mag * t_hat
                            else:
                                shift_vec = -shift_mag * t_hat

                            q0 = p0 + shift_vec
                            q1 = p1 + shift_vec
                            arrow = mpatches.FancyArrowPatch(
                                (float(q0[0]), float(q0[1])),
                                (float(q1[0]), float(q1[1])),
                                arrowstyle="<->",
                                mutation_scale=11.0,
                                linewidth=1.9,
                                linestyle="-",
                                color="#111111",
                                alpha=0.95 * phase_alpha,
                                zorder=13.3,
                                shrinkA=0.0,
                                shrinkB=0.0,
                            )
                            arrow.set_path_effects(_stroke(1.2, "#ffffff"))
                            ax.add_patch(arrow)
                            if labels:
                                pm = 0.5 * (q0 + q1)
                                _text(
                                    ax,
                                    pm + np.array([0.12, 0.0], dtype=float),
                                    r"$\rho_T$",
                                    ha="left",
                                    va="center",
                                    fontsize=_afs(9.4),
                                    color="#111111",
                                    alpha=0.94 * phase_alpha,
                                )

    if show_smooth_terminal_contour and scene.terminal_field is not None:
        XX, YY, H = scene.terminal_field
        try:
            ax.contour(
                XX,
                YY,
                H,
                levels=[0.0],
                colors=[TERMINAL_COLOR],
                linewidths=1.2,
                alpha=0.26 * phase_alpha,
                zorder=11.8,
            )
        except Exception:
            pass

    if not labels:
        return

    if len(scene.occlusion_scenarios) > 1:
        txt = rf"min terminal margin = {term['min_margin']:.3f}"
        x_text, y_text, fsize = 0.028, 0.06, _afs(11.2)
    else:
        txt = rf"terminal margin = {term['min_margin']:.3f}"
        x_text, y_text, fsize = 0.00, -5.03, _afs(10.8)
    artist = ax.text(
        x_text,
        y_text,
        txt,
        transform=ax.transAxes,
        fontsize=fsize,
        color=TEXT_COLOR,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#9fc68b", alpha=0.92),
        zorder=42,
        alpha=0.96 * phase_alpha,
    )


def _draw_weight_box(ax, scene: SceneData, occ_debug: dict, labels: bool = True, phase_alpha: float = 1.0):
    if (not labels) or scene.mode != "multi_common":
        return

    lines = [r"scenario blend"]
    for sc_idx, dbg in enumerate(occ_debug["scenario_debug"]):
        w = float(dbg["scenario_weight"])
        lines.append(rf"$w^{{({sc_idx+1})}} = {w:.2f}$")

    artist = ax.text(
        0.972,
        0.945,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=_afs(10.6),
        color=TEXT_COLOR,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d9e2f0", alpha=0.84),
        zorder=42,
        alpha=0.96 * phase_alpha,
    )


def _draw_scene_title(ax, scene: SceneData, labels: bool = True, phase_alpha: float = 1.0):
    if not labels:
        return
    title = scene.title
    artist = ax.text(
        0.5,
        0.988,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=17.0,
        fontweight="semibold",
        color=TEXT_COLOR,
        zorder=45,
        alpha=0.92 * phase_alpha,
    )


def render_scene_frame(
    ax,
    scene: SceneData,
    u: float,
    labels: bool = True,
    show_normals: bool = True,
    show_expanded: bool = True,
    show_smooth_terminal_contour: bool = False,
):
    phase = _phase_state(scene, u)
    alpha_scene = phase["scene_alpha"]
    alpha_targets = phase["direction_alpha"]
    alpha_rollout = phase["rollout_alpha"]
    alpha_terminal = phase["terminal_alpha"]
    current_tau = phase["current_tau"]

    ax.cla()
    base_robot_alpha = alpha_scene * (1.0 - 0.88 * alpha_rollout)
    draw_base_scene(
        ax,
        scene,
        alpha_robot=base_robot_alpha,
        alpha_scene=alpha_scene,
        labels=labels,
        show_entity_labels=(alpha_rollout <= 1e-6),
        show_base_robot=(alpha_rollout <= 1e-6),
    )
    occ_debug = build_occ_debug_at_tau(scene, current_tau)
    draw_occlusion_geometry(
        ax,
        scene,
        occ_debug,
        current_tau=current_tau,
        show_expanded=show_expanded,
        show_normals=show_normals,
        labels=labels,
        phase_alpha=max(alpha_scene, 0.15),
        inflate_alpha=phase["inflate_alpha"],
        direction_alpha=alpha_targets,
        rollout_alpha=alpha_rollout,
        normal_alpha=phase["normal_alpha"],
        active_facet_alpha=max(phase["normal_alpha"], 0.95 * alpha_terminal),
    )
    if alpha_targets > 0.0 and alpha_terminal <= 1e-6:
        draw_normals_and_targets(
            ax,
            scene,
            occ_debug,
            current_tau=current_tau,
            labels=labels,
            phase_alpha=phase["vref_alpha"],
            rollout_alpha=alpha_rollout,
        )
    if alpha_rollout > 0.0 or alpha_terminal > 0.0:
        draw_rollout(
            ax,
            scene,
            current_tau=current_tau,
            labels=labels,
            phase_alpha=max(alpha_rollout, 0.2),
            terminal_alpha=alpha_terminal,
        )
    if alpha_terminal > 0.0:
        draw_terminal_annotation(
            ax,
            scene,
            labels=labels,
            phase_alpha=alpha_terminal,
            show_smooth_terminal_contour=show_smooth_terminal_contour,
        )
    _draw_weight_box(ax, scene, occ_debug, labels=labels, phase_alpha=max(alpha_targets, alpha_rollout))
    _draw_scene_title(ax, scene, labels=labels, phase_alpha=max(alpha_scene, 0.25))

    if labels and (alpha_rollout > 0.08 or alpha_terminal > 0.08):
        tau_artist = ax.text(
            0.97,
            0.05,
            rf"$\tau = {current_tau:.2f}\; / \; {scene.controller.pos_controller.T_horizon:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=_afs(11.6),
            color=TEXT_COLOR,
            zorder=44,
            alpha=0.88 * max(alpha_rollout, alpha_terminal),
        )


def _save_frame(fig, path: Path, dpi: int):
    fig.savefig(path, dpi=dpi, facecolor=BG_COLOR, edgecolor="none")


def _export_video(frame_dir: Path, video_path: Path, fps: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-vf",
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2,fps={int(fps)}",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print(f"[WARN] ffmpeg not found. Frames were saved under: {frame_dir}")


def _prepare_output_dir(args):
    root = Path(args.output_dir) / args.mode
    root.mkdir(parents=True, exist_ok=True)
    frame_dir = root / "frames"
    if args.save_animation:
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
    snapshot_png = root / f"{args.snapshot_name}.png"
    snapshot_svg = root / f"{args.snapshot_name}.svg"
    video_path = root / f"{args.mode}.mp4"
    return root, frame_dir, snapshot_png, snapshot_svg, video_path


def render_single_common(args):
    scene = build_single_common_scene(args)
    return _render_scene_driver(scene, args)


def render_multi_common(args):
    scene = build_multi_common_scene(args)
    return _render_scene_driver(scene, args)


def _render_scene_driver(scene: SceneData, args):
    out_root, frame_dir, snapshot_png, snapshot_svg, video_path = _prepare_output_dir(args)

    fig, ax = plt.subplots(figsize=(float(args.figure_width), float(args.figure_height)))
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=1.00)
    if bool(args.show_animation):
        plt.show(block=False)

    n_frames = max(2, int(np.ceil(float(args.duration) * float(args.fps))))
    last_u = 1.0
    for frame_idx in range(n_frames):
        u = 1.0 if n_frames <= 1 else float(frame_idx / (n_frames - 1))
        last_u = u
        render_scene_frame(
            ax,
            scene,
            u=u,
            labels=not bool(args.no_labels),
            show_normals=not bool(args.no_normals),
            show_expanded=not bool(args.no_expanded_envelopes),
            show_smooth_terminal_contour=bool(args.show_smooth_terminal_contour),
        )
        if args.save_animation:
            _save_frame(fig, frame_dir / f"frame_{frame_idx:04d}.png", dpi=int(args.dpi))
        if bool(args.show_animation):
            fig.canvas.draw_idle()
            plt.pause(max(1e-3, 1.0 / max(float(args.fps), 1.0)))

    # Save final snapshot from the last rendered frame.
    render_scene_frame(
        ax,
        scene,
        u=last_u,
        labels=not bool(args.no_labels),
        show_normals=not bool(args.no_normals),
        show_expanded=not bool(args.no_expanded_envelopes),
        show_smooth_terminal_contour=bool(args.show_smooth_terminal_contour),
    )
    _save_frame(fig, snapshot_png, dpi=int(args.dpi))
    if bool(args.save_svg):
        fig.savefig(snapshot_svg, facecolor=BG_COLOR, edgecolor="none")

    if args.save_animation:
        _export_video(frame_dir, video_path, fps=int(args.fps))

    if args.show or bool(args.show_animation):
        plt.show()
    else:
        plt.close(fig)

    print(f"saved_root={out_root}")
    print(f"snapshot_png={snapshot_png}")
    if bool(args.save_svg):
        print(f"snapshot_svg={snapshot_svg}")
    if args.save_animation:
        print(f"frames_dir={frame_dir}")
        print(f"video_path={video_path}")
    return {
        "root": str(out_root),
        "snapshot_png": str(snapshot_png),
        "snapshot_svg": str(snapshot_svg) if bool(args.save_svg) else None,
        "frames_dir": str(frame_dir) if args.save_animation else None,
        "video_path": str(video_path) if args.save_animation else None,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Deterministic explanatory visualization for occlusionCBF.")
    parser.add_argument("--mode", type=str, default="single_common", choices=["single_common", "multi_common"])
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    parser.add_argument(
        "--show-animation",
        "--show_animation",
        dest="show_animation",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Play the explanatory sequence interactively without requiring video export.",
    )
    parser.add_argument("--save-animation", "--save-anim", "--save_ani", dest="save_animation", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output-dir", type=str, default="output/animations/vis_ocbf")
    parser.add_argument("--save-svg", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--snapshot-name", type=str, default="final_snapshot")
    parser.add_argument("--duration", type=float, default=8.0, help="Total video duration in seconds.")
    parser.add_argument("--tau-samples", type=int, default=9, help="Number of rollout sample dots / envelope ladder levels.")
    parser.add_argument("--grid-res", type=float, default=0.04, help="Grid resolution for terminal contour sampling.")
    parser.add_argument(
        "--show-smooth-terminal-contour",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Show the original smoothed terminal-set contour in addition to the raw geometric terminal polygons.",
    )
    parser.add_argument("--T-horizon", dest="T_horizon", type=float, default=3.0)
    parser.add_argument("--dt-backup", type=float, default=0.05)
    parser.add_argument("--v-adv-max-occ", type=float, default=0.5)
    parser.add_argument("--occ-kappa", type=float, default=10.0)
    parser.add_argument("--vref-scenario-softmax-kappa", type=float, default=6.0)
    parser.add_argument("--figure-width", type=float, default=11.2)
    parser.add_argument("--figure-height", type=float, default=7.8)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--no-expanded-envelopes", action="store_true")
    return parser


def main():
    global args_global
    parser = build_arg_parser()
    args = parser.parse_args()
    args_global = args

    if args.mode == "single_common":
        render_single_common(args)
    else:
        render_multi_common(args)


if __name__ == "__main__":
    args_global = None
    main()
