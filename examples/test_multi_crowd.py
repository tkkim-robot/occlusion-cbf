"""
Multi-baseline replay for the canonical crowd benchmark.

This script runs multiple crowd baselines on the same generated scenario and
renders:
- one shared global view on the left
- one per-baseline tracking view on the right

The simulation keeps running until every baseline reaches a terminal metric
(success / infeasible / collision) or the global horizon expires. Once a
baseline terminates, its last frame is frozen while the others continue.

Run:
    uv run python examples/test_multi_crowd.py --model di --idx 13
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

try:
    from examples._baseline_defs import (
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        CROWD_BENCHMARK_DEFAULTS,
        OACP_BENCHMARK_DEFAULTS,
        resolve_baseline_alias,
    )
    from examples import test_crowd_narrow as crowd_narrow
    from examples import test_crowd as crowd
except ImportError:
    from _baseline_defs import (
        CROWD_BASELINE_CHOICES,
        CROWD_BASELINE_MAP,
        CROWD_BENCHMARK_DEFAULTS,
        OACP_BENCHMARK_DEFAULTS,
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
    OCBF_VREF_SCENARIO_WEIGHT_MODES,
)

from base_control.utils import env, plotting


DEFAULT_BASELINES = [
    "occlusion_cbf",
    "cbf_qp",
    "oa_mpc",
    "single_risk_mpc",
    "control_tree_mpc",
    "oacp_mpc",
]

TRAJ_CMAPS = [
    "Reds",
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
    "Greys",
]

TRAJ_CMAPS_BY_BASELINE = {
    "occlusion_cbf_qp": "trajectory_red_pink",
    "occlusion_cbf_terminal_relax": "trajectory_red_pink",
    "cbf_qp": "trajectory_yellow",
    "oa_mpc": "Greens",
    "single_risk_mpc": "Blues",
    "control_tree_mpc": "Purples",
    "oacp_mpc": "Greys",
}

STATUS_COLORS = {
    "running": "tab:blue",
    "success": "tab:green",
    "collision": "tab:red",
    "infeasible": "tab:red",
    "timeout": "tab:orange",
}

TRACKING_LABELS = {
    "occlusion_cbf_qp": "Occlusion CBF-QP",
    "occlusion_cbf_terminal_relax": "OCBF (relaxed terminal)",
    "cbf_qp": "CBF-QP",
    "oa_mpc": "OA-MPC",
    "single_risk_mpc": "Single-Hypothesis MPC",
    "control_tree_mpc": "Control Tree MPC",
    "oacp_mpc": "OACP",
}

_MPC_BASELINES = {"oa_mpc", "single_risk_mpc", "control_tree_mpc", "oacp_mpc"}
_QP_BASELINES = {"occlusion_cbf_qp", "occlusion_cbf_terminal_relax", "cbf_qp"}


def _copy_obs_meta(obs_meta):
    return [dict(m) for m in list(obs_meta)]


def _trajectory_legend_color(cmap_name):
    cmap_key = str(cmap_name).strip()
    if cmap_key in {"trajectory_red_pink", "red_pink", "pink_red"}:
        return "#ff3b6b"
    if cmap_key in {"trajectory_teal_cyan", "teal_cyan", "cyan_teal"}:
        return "#18a8bb"
    if cmap_key in {"trajectory_yellow", "yellow_trajectory", "pure_yellow"}:
        return "#ffd400"
    try:
        cmap = plt.colormaps.get_cmap(cmap_key)
        return cmap(0.78)
    except Exception:
        return "tab:blue"


def _build_crowd_scenario(args):
    case_seed = crowd_narrow._compute_case_seed(args.seed, args.idx)
    mode = str(args.crowd_mode).strip().lower()
    if mode == "forced_emergence":
        known_obs, obs_meta, scenario_diag = crowd._build_route_forced_emergence_scenario(
            case_seed=case_seed,
            n_rand=args.n_rand,
            rand_obs=True,
            static_occluders=False,
            forced_events=args.forced_events,
            forced_bg_rand=args.forced_bg_rand,
            forced_hidden_speed=args.forced_hidden_speed,
            forced_occluder_radius_min=args.forced_occluder_radius_min,
            forced_occluder_radius_max=args.forced_occluder_radius_max,
            forced_validate_occlusion=args.forced_validate_occlusion,
            forced_require_corridor_conflict=args.forced_require_corridor_conflict,
            rand_obs_setting=args.rand_obs_setting,
        )
    else:
        known_obs, obs_meta, scenario_diag = crowd._build_route_random_scenario(
            case_seed=case_seed,
            n_rand=args.n_rand,
            rand_obs=True,
            static_occluders=False,
            rand_obs_setting=args.rand_obs_setting,
        )
    return case_seed, known_obs, obs_meta, scenario_diag


def _make_figure_layout(n_baselines, *, width, height, known_obs, hide_env_boundary=True):
    plot_handler = plotting.Plotting(width=width, height=height, known_obs=known_obs)
    if bool(hide_env_boundary):
        plot_handler.obs_bound = []

    n_cols_right = 2
    n_rows_right = int(np.ceil(float(max(1, n_baselines)) / float(n_cols_right)))
    fig = plt.figure(figsize=(18.5, max(8.0, 2.8 * n_rows_right)))
    gs = fig.add_gridspec(
        n_rows_right,
        4,
        width_ratios=[2.25, 2.25, 1.45, 1.45],
        wspace=0.16,
        hspace=0.28,
    )

    global_ax = fig.add_subplot(gs[:, :2])
    plot_handler._draw_environment(global_ax, include_static_circles=True)
    plot_handler._configure_axis(global_ax, title="Global View")

    track_axes = []
    for ridx in range(n_rows_right):
        for cidx in range(n_cols_right):
            ax = fig.add_subplot(gs[ridx, 2 + cidx])
            plot_handler._draw_environment(ax, include_static_circles=False)
            plot_handler._configure_axis(ax, title=None)
            track_axes.append(ax)

    return fig, global_ax, track_axes[:n_baselines]


def _hide_artist(artist):
    if artist is None:
        return
    try:
        artist.set_visible(False)
    except Exception:
        pass


def _configure_multi_visuals(ctrl, planner_idx):
    robot = ctrl.robot
    body = getattr(robot, "body", None)
    if body is not None:
        try:
            body.set_zorder(12.0)
        except Exception:
            pass

    axis = getattr(robot, "axis", None)
    if axis is not None and body is not None:
        try:
            axis.set_linewidth(2.2)
            axis.set_zorder(13.0)
        except Exception:
            pass

    for attr in [
        "fov",
        "fov_fill",
        "sensing_footprints_fill",
        "safety_area_fill",
        "detected_obs_patch",
        "detected_points_scatter",
        "unsafe_points_handle",
    ]:
        _hide_artist(getattr(robot, attr, None))

    ctrl.plot_dyn_obs = False
    ctrl.plot_occ_polygons = False
    ctrl.show_backup_rollout = False
    ctrl.robot_spec["infeasible_pause"] = 0.0

    def _multi_draw_infeasible():
        ctrl._infeasible_active = True
        ctrl._infeasible_seen = True

    ctrl.draw_infeasible = _multi_draw_infeasible

    body_color = None
    if body is not None:
        try:
            body_color = body.get_facecolor()
        except Exception:
            body_color = None

    if ctrl.tracking_view_enabled:
        _install_multi_qp_stats_overlay(ctrl)
    return body_color


def _install_multi_qp_stats_overlay(ctrl):
    def _multi_update_qp_stats_text():
        if not getattr(ctrl, "show_animation", False):
            return

        pos_controller = getattr(ctrl, "pos_controller", None)
        if pos_controller is None:
            return

        total_ms = getattr(pos_controller, "last_total_compute_time_ms", None)
        qp_ms = getattr(pos_controller, "last_qp_solve_time_ms", None)
        if total_ms is None:
            prof = getattr(pos_controller, "last_profile", None)
            if isinstance(prof, dict):
                total_ms = prof.get("total_ms", None)
        if total_ms is None:
            total_ms = qp_ms

        intervention = getattr(pos_controller, "last_intervention", None)
        if intervention is None:
            int_flag = getattr(pos_controller, "_last_intervention", None)
            if isinstance(int_flag, (bool, np.bool_)):
                intervention = "backup_qp" if bool(int_flag) else "u_ref"

        controller_type = str(getattr(ctrl, "pos_controller_type", "")).strip().lower()
        if intervention == "u_ref":
            mode_text = "Nominal"
        elif intervention == "infeasible":
            mode_text = "Infeasible"
        elif intervention == "backup_fallback":
            mode_text = "Fallback"
        elif controller_type in _QP_BASELINES:
            mode_text = "CBF-QP"
        elif controller_type in _MPC_BASELINES:
            mode_text = "MPC"
        else:
            mode_text = "Control"

        time_text = f"{float(total_ms):.3f} ms" if total_ms is not None else "n/a"
        text = f"Computation: {time_text}\nMode: {mode_text}"
        text_ax = ctrl.tracking_view_ax if getattr(ctrl, "tracking_view_enabled", False) else ctrl.ax

        if ctrl.qp_stats_text is not None and getattr(ctrl.qp_stats_text, "axes", None) is not text_ax:
            try:
                ctrl.qp_stats_text.remove()
            except Exception:
                pass
            ctrl.qp_stats_text = None

        if ctrl.qp_stats_text is None:
            ctrl.qp_stats_text = text_ax.text(
                0.02,
                0.02,
                text,
                transform=text_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                family="monospace",
                zorder=40,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.8),
            )
        else:
            ctrl.qp_stats_text.set_text(text)
            ctrl.qp_stats_text.set_position((0.02, 0.02))
            ctrl.qp_stats_text.set_ha("left")
            ctrl.qp_stats_text.set_va("bottom")

    ctrl._update_qp_stats_text = _multi_update_qp_stats_text


def _init_global_dyn_obs(ax, obs_arr):
    artists = {"patches": [], "arrows": []}
    obs_np = np.asarray(obs_arr, dtype=float)
    if obs_np.ndim != 2 or obs_np.shape[0] == 0:
        return artists
    for row in obs_np:
        patch = ax.add_patch(
            plt.Circle(
                (float(row[0]), float(row[1])),
                float(row[2]),
                edgecolor="black",
                facecolor="gray",
                alpha=0.92,
                linewidth=1.0,
                zorder=4.0,
            )
        )
        arrow = mpatches.FancyArrowPatch(
            (0, 0),
            (0, 0),
            arrowstyle='-|>',
            mutation_scale=6.0,
            color='deepskyblue',
            linewidth=1.0,
            zorder=5.0,
        )
        arrow.set_visible(False)
        ax.add_patch(arrow)
        artists["patches"].append(patch)
        artists["arrows"].append(arrow)
    return artists


def _update_global_dyn_obs(global_dyn_obs_artists, obs_arr):
    if not global_dyn_obs_artists:
        return
    obs_np = np.asarray(obs_arr, dtype=float)
    if obs_np.ndim != 2:
        obs_np = np.empty((0, 3), dtype=float)

    patch_artists = global_dyn_obs_artists.get("patches", [])
    arrow_artists = global_dyn_obs_artists.get("arrows", [])
    n_show = min(len(patch_artists), int(obs_np.shape[0]))

    speeds = np.hypot(obs_np[:, 3], obs_np[:, 4]) if (obs_np.size and obs_np.shape[1] >= 5) else np.array([])
    v_ref = float(np.max(speeds)) if speeds.size else 1.0
    if v_ref < 1e-9:
        v_ref = 1.0

    vis_color = "gray"
    arrow_color = "deepskyblue"
    arrow_len_min = 0.3
    arrow_len_max = 0.8
    arrow_head = 6.0

    for i in range(n_show):
        row = obs_np[i]
        patch = patch_artists[i]
        patch.center = (float(row[0]), float(row[1]))
        patch.set_radius(float(row[2]))
        patch.set_facecolor(vis_color)
        patch.set_visible(True)

        if i < len(arrow_artists):
            arrow = arrow_artists[i]
            arrow.set_color(arrow_color)
            arrow.set_mutation_scale(arrow_head)
            if obs_np.shape[1] < 5:
                arrow.set_visible(False)
            else:
                vx = float(row[3])
                vy = float(row[4])
                speed = float(np.hypot(vx, vy))
                if speed < 1e-9:
                    arrow.set_visible(False)
                else:
                    ux = vx / speed
                    uy = vy / speed
                    t = min(1.0, speed / v_ref)
                    length = arrow_len_min + (arrow_len_max - arrow_len_min) * np.sqrt(t)
                    arrow.set_positions((float(row[0]), float(row[1])), (float(row[0] + ux * length), float(row[1] + uy * length)))
                    arrow.set_visible(True)

    for i in range(n_show, len(patch_artists)):
        patch_artists[i].set_visible(False)
    for i in range(n_show, len(arrow_artists)):
        arrow_artists[i].set_visible(False)


def _shared_obs_source(runners):
    for runner in runners:
        if runner.get("active", False):
            ctrl = runner.get("controller", None)
            if ctrl is not None and isinstance(getattr(ctrl, "obs", None), np.ndarray):
                return ctrl.obs
    if runners:
        ctrl = runners[0].get("controller", None)
        if ctrl is not None and isinstance(getattr(ctrl, "obs", None), np.ndarray):
            return ctrl.obs
    return None


def _refresh_controller_artists(ctrl):
    if not getattr(ctrl, "show_animation", False):
        return
    try:
        if getattr(ctrl, "plot_dyn_obs_occlusion", False):
            ctrl._cached_occluded_mask = ctrl._get_occluded_obs_mask()
    except Exception:
        pass
    try:
        ctrl._update_trajectory_artist(ctrl.ax, "robot_trajectory_collection")
    except Exception:
        pass
    try:
        ctrl._update_infeasible_marker()
    except Exception:
        pass
    try:
        ctrl._update_qp_stats_text()
    except Exception:
        pass
    try:
        ctrl._update_tracking_view()
    except Exception:
        pass


def _runner_status(runner, dt, tf):
    if runner["active"]:
        return "running"
    event = runner.get("terminal_event", None)
    if event in {"success", "collision", "infeasible"}:
        return event
    if runner["steps"] * dt >= tf - 1e-9:
        return "timeout"
    return "timeout"


def _status_label(runner, dt, tf):
    status = _runner_status(runner, dt, tf)
    t = runner["steps"] * dt
    if status == "running":
        return f"RUN | t={t:.2f}s"
    if status == "timeout":
        return f"TIMEOUT | t={t:.2f}s"
    return f"{status.upper()} | t={t:.2f}s"


def _status_badge_label(runner, dt, tf):
    status = _runner_status(runner, dt, tf)
    t = runner["steps"] * dt
    short = {
        "running": "RUN",
        "success": "SUCCESS",
        "collision": "COLLISION",
        "infeasible": "INFEASIBLE",
        "timeout": "TIMEOUT",
    }.get(status, status.upper())
    return f"{short}  {t:.2f}s"


def _update_status_text(global_status_text, runners, dt, tf):
    if global_status_text is None:
        return
    lines = []
    for runner in runners:
        lines.append(f"{runner['label']:<18} {_status_label(runner, dt, tf)}")
    global_status_text.set_text("\n".join(lines))


def _set_tracking_titles(runners, dt, tf):
    for runner in runners:
        ax = runner["tracking_ax"]
        if ax is None:
            continue
        status = _runner_status(runner, dt, tf)
        color = STATUS_COLORS.get(status, "black")
        ax.set_title(runner["compact_label"], fontsize=10, color="black", pad=3)
        status_text = runner.get("tracking_status_text", None)
        if status_text is not None:
            status_text.set_text(_status_badge_label(runner, dt, tf))
            status_text.set_color(color)
            status_text.set_bbox(
                dict(
                    boxstyle="round,pad=0.22",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.92,
                )
            )


def _save_frame(fig, frame_dir: Path, frame_idx: int):
    frame_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        frame_dir / f"t_step_{frame_idx:04d}.png",
        dpi=220,
        facecolor=fig.get_facecolor(),
    )


def _export_video(frame_dir: Path, video_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(frame_dir / "t_step_%04d.png"),
        "-vf",
        "scale=1920:-2,fps=60",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.call(cmd)
    for png in frame_dir.glob("t_step_*.png"):
        png.unlink()


def _build_runtime_for_baseline(args, baseline_alias, known_obs, obs_meta, scenario_diag):
    algo = resolve_baseline_alias(baseline_alias, baseline_alias, CROWD_BASELINE_MAP)
    controller_type = {"pos": algo}
    backup_cbf_overrides = {}
    if args.occ_dt_backup is not None:
        backup_cbf_overrides["dt_backup"] = float(args.occ_dt_backup)
    if args.occ_t_horizon is not None:
        backup_cbf_overrides["T_horizon"] = float(args.occ_t_horizon)
    if args.occ_rho_T is not None:
        backup_cbf_overrides["rho_T"] = float(args.occ_rho_T)
    if args.occ_vref_scenario_softmax_kappa is not None:
        backup_cbf_overrides["vref_scenario_softmax_kappa"] = float(args.occ_vref_scenario_softmax_kappa)
    if args.occ_vref_scenario_weight_mode is not None:
        backup_cbf_overrides["vref_scenario_weight_mode"] = str(args.occ_vref_scenario_weight_mode).strip().lower()
    if args.occ_max_active_occlusions is not None:
        backup_cbf_overrides["max_active_occlusions"] = int(args.occ_max_active_occlusions)
    if args.occ_selection_mode is not None:
        backup_cbf_overrides["occ_selection_mode"] = str(args.occ_selection_mode).strip().lower()
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
    if args.occ_k_p is not None:
        backup_cbf_overrides["k_p_occ_di"] = float(args.occ_k_p)
    if args.occ_k_d is not None:
        backup_cbf_overrides["k_d_occ_di"] = float(args.occ_k_d)
    robot_spec_overrides = {}
    if args.occ_kappa is not None:
        robot_spec_overrides["occ_kappa"] = float(args.occ_kappa)
    if args.occ_enable_visible_hocbf is not None:
        robot_spec_overrides["enable_visible_hocbf_in_occ"] = bool(args.occ_enable_visible_hocbf)
    if algo == "oacp_mpc":
        oacp_cfg = {
            "allow_solver_fallback": bool(args.oacp_allow_solver_fallback),
            "dynamic_occluders": bool(args.oacp_dynamic_occluders),
            "visible_reach_mode": str(args.oacp_visible_reach_mode).strip().lower(),
            "branch_safety_gate": bool(args.oacp_branch_safety_gate),
        }
        if args.oacp_backend is not None:
            oacp_cfg["backend"] = str(args.oacp_backend).strip().lower()
        robot_spec_overrides["oacp_mpc"] = oacp_cfg

    return crowd_narrow._prepare_crowd_runtime(
        controller_type=controller_type,
        model_key=args.model,
        tf=args.tf,
        seed=args.seed,
        case_idx=args.idx,
        rand_obs=True,
        n_rand=args.n_rand,
        occ_visible_scale=args.occ_visible_scale,
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
        static_occluders=False,
        vref_front_mode_occ=args.vref,
        occ_enable_visible_hocbf=args.occ_enable_visible_hocbf,
        backup_cbf_overrides=(backup_cbf_overrides or None),
        robot_spec_overrides=robot_spec_overrides,
        waypoints_override=crowd.ROUTE_WAYPOINTS,
        env_width_override=crowd.ENV_WIDTH,
        env_height_override=crowd.ENV_HEIGHT,
        known_obs_override=np.asarray(known_obs, dtype=float).copy(),
        obs_meta_override=_copy_obs_meta(obs_meta),
        scenario_diag_override=deepcopy(scenario_diag),
        scenario_name="Crowd",
    )


def run_multi_crowd(args):
    baselines = list(args.baselines)
    if len(baselines) == 0:
        raise ValueError("At least one baseline must be specified.")

    case_seed, known_obs, obs_meta, scenario_diag = _build_crowd_scenario(args)

    show_animation = (not args.disable_plot) or bool(args.save_anim)
    if show_animation and not bool(args.disable_plot):
        plt.ion()
    else:
        plt.ioff()

    fig = None
    global_ax = None
    tracking_axes = []
    frame_dir = None
    video_path = None
    if show_animation:
        fig, global_ax, tracking_axes = _make_figure_layout(
            len(baselines),
            width=crowd.ENV_WIDTH,
            height=crowd.ENV_HEIGHT,
            known_obs=known_obs,
            hide_env_boundary=True,
        )
        fig.subplots_adjust(top=0.97, left=0.05, right=0.985, bottom=0.06)
    global_status_text = None

    runners = []
    legend_handles = []
    for idx, baseline_alias in enumerate(baselines):
        runtime = _build_runtime_for_baseline(args, baseline_alias, known_obs, obs_meta, scenario_diag)
        # Keep robot body styling identical to the single-baseline replay.
        runtime["robot_spec"]["robot_id"] = 0
        runtime["robot_spec"]["plot_robot_trajectory_cmap"] = TRAJ_CMAPS_BY_BASELINE.get(
            runtime["pos_name"],
            TRAJ_CMAPS[idx % len(TRAJ_CMAPS)],
        )
        runtime["robot_spec"]["plot_robot_trajectory_linewidth"] = 2.5
        runtime["robot_spec"]["plot_robot_trajectory_alpha"] = 0.96
        runtime["robot_spec"]["plot_robot_trajectory_zorder"] = 3.0

        tracking_ax = tracking_axes[idx] if show_animation else None
        ctrl = crowd_narrow.LocalTrackingControllerDyn_OCC(
            runtime["waypoints"][0],
            runtime["robot_spec"],
            controller_type=runtime["controller_type"],
            dt=runtime["dt"],
            show_animation=show_animation,
            save_animation=False,
            show_mpc_traj=False,
            ax=global_ax,
            fig=fig,
            env=env.Env(width=runtime["env_width"], height=runtime["env_height"], known_obs=known_obs),
            rand_seed=case_seed,
            tracking_view_ax=tracking_ax,
            tracking_view_window_size=crowd.TRACKING_VIEW_WINDOW_SIZE,
        )
        ctrl.obs = np.asarray(runtime["known_obs"], dtype=float).copy()
        ctrl.set_obs_meta(_copy_obs_meta(runtime["obs_meta"]))
        ctrl.set_waypoints(np.asarray(runtime["waypoints"], dtype=float))

        body_color = _configure_multi_visuals(ctrl, idx)
        label = runtime["planner_label"]
        compact_label = TRACKING_LABELS.get(runtime["pos_name"], label)
        tracking_status_text = None
        if tracking_ax is not None:
            tracking_ax.set_title(compact_label, fontsize=10, pad=3)
            tracking_status_text = tracking_ax.text(
                0.98,
                0.98,
                "",
                transform=tracking_ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                zorder=40,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_trajectory_legend_color(runtime["robot_spec"]["plot_robot_trajectory_cmap"]),
                linewidth=4,
                label=label,
            )
        )
        runners.append(
            {
                "alias": baseline_alias,
                "algo": runtime["pos_name"],
                "label": label,
                "compact_label": compact_label,
                "controller": ctrl,
                "tracking_ax": tracking_ax,
                "tracking_status_text": tracking_status_text,
                "active": True,
                "ret_last": 0,
                "terminal_event": None,
                "steps": 0,
                "frozen_dirty": True,
            }
        )

    if show_animation and legend_handles:
        global_ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=9,
            ncol=2,
            framealpha=0.88,
        )

    global_dyn_obs_artists = []
    if show_animation and global_ax is not None and runners:
        global_dyn_obs_artists = _init_global_dyn_obs(global_ax, runners[0]["controller"].obs)

    if bool(args.save_anim):
        out_root = Path(args.save_root) if args.save_root is not None else Path("output/animations/multi_crowd")
        idx_dir = out_root / (f"idx{int(args.idx):03d}" if args.idx is not None else "idx000")
        frame_dir = idx_dir / "frames"
        video_path = idx_dir / "tracking.mp4"
        if video_path.exists():
            video_path.unlink()
        for png in frame_dir.glob("t_step_*.png"):
            png.unlink()

    n_steps = int(np.ceil(float(args.tf) / float(runners[0]["controller"].dt)))
    frame_idx = 0
    for _ in range(n_steps):
        any_active = False
        for runner in runners:
            if not runner["active"]:
                continue
            any_active = True
            ret = runner["controller"].control_step()
            runner["ret_last"] = int(ret)
            runner["steps"] += 1
            evt = getattr(runner["controller"], "last_terminal_event", None)
            if ret == -1:
                runner["active"] = False
                runner["terminal_event"] = "success"
                runner["frozen_dirty"] = True
            elif ret == -2:
                runner["active"] = False
                runner["terminal_event"] = evt or "infeasible"
                runner["frozen_dirty"] = True

        for runner in runners:
            if runner["active"] or runner["frozen_dirty"]:
                _refresh_controller_artists(runner["controller"])
                runner["frozen_dirty"] = False

        if show_animation and global_dyn_obs_artists:
            shared_obs = _shared_obs_source(runners)
            if shared_obs is not None:
                _update_global_dyn_obs(global_dyn_obs_artists, shared_obs)

        if show_animation:
            _set_tracking_titles(runners, runners[0]["controller"].dt, float(args.tf))
            _update_status_text(global_status_text, runners, runners[0]["controller"].dt, float(args.tf))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            if not bool(args.disable_plot):
                plt.pause(float(args.plot_pause))
            if frame_dir is not None:
                frame_idx += 1
                _save_frame(fig, frame_dir, frame_idx)

        if not any_active:
            break

    for runner in runners:
        if runner["active"]:
            runner["active"] = False
            if runner["terminal_event"] is None:
                runner["terminal_event"] = "timeout"

    if show_animation:
        if global_dyn_obs_artists:
            shared_obs = _shared_obs_source(runners)
            if shared_obs is not None:
                _update_global_dyn_obs(global_dyn_obs_artists, shared_obs)
        _set_tracking_titles(runners, runners[0]["controller"].dt, float(args.tf))
        _update_status_text(global_status_text, runners, runners[0]["controller"].dt, float(args.tf))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if frame_dir is not None:
            frame_idx += 1
            _save_frame(fig, frame_dir, frame_idx)

    if video_path is not None and frame_dir is not None:
        _export_video(frame_dir, video_path)

    print("=== Multi Crowd Summary ===")
    for runner in runners:
        ctrl = runner["controller"]
        status = _runner_status(runner, ctrl.dt, float(args.tf))
        print(
            f"{runner['label']}: status={status}, steps={runner['steps']}, "
            f"t={runner['steps'] * ctrl.dt:.2f}s, event={runner['terminal_event']}"
        )
    if video_path is not None:
        print(f"saved_video={video_path}")

    if show_animation and bool(args.disable_plot):
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a multi-baseline crowd replay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CROWD_BENCHMARK_DEFAULTS["model"],
        choices=["di", "du", "uni"],
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=DEFAULT_BASELINES,
        choices=CROWD_BASELINE_CHOICES,
        help="Baseline aliases to replay together.",
    )
    parser.add_argument("--tf", type=float, default=CROWD_BENCHMARK_DEFAULTS["tf"])
    parser.add_argument("--seed", type=int, default=CROWD_BENCHMARK_DEFAULTS["seed"])
    parser.add_argument("--idx", type=int, default=CROWD_BENCHMARK_DEFAULTS["idx_start"])
    parser.add_argument("--n-rand", type=int, default=CROWD_BENCHMARK_DEFAULTS["n_rand"])
    parser.add_argument(
        "--rand-obs-setting",
        type=str,
        default=crowd_narrow.DEFAULT_RAND_OBS_SETTING,
        choices=[
            crowd_narrow.FIXED_SPEED_RAND_OBS_SETTING,
            crowd_narrow.DISTRIBUTED_SPEED_RAND_OBS_SETTING,
        ],
    )
    parser.add_argument("--disable-plot", action="store_true", help="Disable interactive plotting.")
    parser.add_argument(
        "--save_ani",
        "--save-ani",
        "--save-animation",
        dest="save_anim",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--plot-pause", type=float, default=0.001)

    parser.add_argument(
        "--crowd-mode",
        type=str,
        default=CROWD_BENCHMARK_DEFAULTS["crowd_mode"],
        choices=["random", "forced_emergence"],
    )
    parser.add_argument(
        "--forced-events",
        type=int,
        default=CROWD_BENCHMARK_DEFAULTS["forced_events"],
    )
    parser.add_argument("--forced-bg-rand", type=int, default=None)
    parser.add_argument(
        "--forced-hidden-speed",
        type=float,
        default=CROWD_BENCHMARK_DEFAULTS["forced_hidden_speed"],
    )
    parser.add_argument(
        "--forced-occluder-radius-min",
        type=float,
        default=CROWD_BENCHMARK_DEFAULTS["forced_occluder_radius_min"],
    )
    parser.add_argument(
        "--forced-occluder-radius-max",
        type=float,
        default=CROWD_BENCHMARK_DEFAULTS["forced_occluder_radius_max"],
    )
    parser.add_argument(
        "--forced-validate-occlusion",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=CROWD_BENCHMARK_DEFAULTS["forced_validate_occlusion"],
    )
    parser.add_argument(
        "--forced-require-corridor-conflict",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=CROWD_BENCHMARK_DEFAULTS["forced_require_corridor_conflict"],
    )

    parser.add_argument("--occ-visible-scale", type=float, default=0.7)
    parser.add_argument("--occ-dt-backup", type=float, default=None)
    parser.add_argument("--occ-t-horizon", type=float, default=None)
    parser.add_argument("--occ-rho-T", type=float, default=None)
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
    parser.add_argument("--occ-enable-visible-hocbf", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--occ-k-p", type=float, default=None)
    parser.add_argument("--occ-k-d", type=float, default=None)
    parser.add_argument("--occ-kappa", type=float, default=None)
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument(
        "--vref",
        type=str,
        choices=["default", "los"],
        default=None,
        help="OCBF front-facet direction mode. Internal default is `los`; `default` keeps the fixed polygon normal.",
    )
    parser.add_argument("--oa-dynamic-occluders", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-allow-solver-fallback", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-dsafe", type=float, default=None)
    parser.add_argument("--oa-visible-reach-mode", type=str, choices=["worst_case", "constant_velocity"], default=None)
    parser.add_argument("--oa-use-nominal-tracking-cost", type=crowd_narrow._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-dt", type=float, default=None)
    parser.add_argument("--wmax", type=str, choices=["default", "pi"], default="default")
    parser.add_argument("--oacp-backend", type=str, choices=["coupled_nlp", "admm_lowdim"], default=None)
    parser.add_argument(
        "--oacp-allow-solver-fallback",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=OACP_BENCHMARK_DEFAULTS["allow_solver_fallback"],
    )
    parser.add_argument(
        "--oacp-dynamic-occluders",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=OACP_BENCHMARK_DEFAULTS["dynamic_occluders"],
    )
    parser.add_argument(
        "--oacp-visible-reach-mode",
        type=str,
        choices=["constant_velocity", "worst_case"],
        default=OACP_BENCHMARK_DEFAULTS["visible_reach_mode"],
    )
    parser.add_argument(
        "--oacp-branch-safety-gate",
        type=crowd_narrow._str2bool,
        nargs="?",
        const=True,
        default=OACP_BENCHMARK_DEFAULTS["branch_safety_gate"],
    )

    args = parser.parse_args(argv)
    run_multi_crowd(args)

if __name__ == "__main__":
    main()
