"""
Crosswalk scenario test for the occlusion-aware CBF framework.

This script ports the `crosswalk_scenario_v3` setup into this repo and keeps
the scenario self-contained in this file.
"""

import argparse
import types
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import numpy as np

try:
    from examples._baseline_defs import (
        CROSSWALK_BASELINE_CHOICES,
        CROSSWALK_BASELINE_MAP,
        resolve_baseline_alias,
    )
    from examples._runtime import ensure_repo_root, load_local_occ_controller
except ImportError:
    from _baseline_defs import CROSSWALK_BASELINE_CHOICES, CROSSWALK_BASELINE_MAP, resolve_baseline_alias
    from _runtime import ensure_repo_root, load_local_occ_controller

REPO_ROOT = ensure_repo_root()
LocalTrackingControllerDyn_OCC = load_local_occ_controller("crosswalk")
from base_control.utils import env, plotting
from position_control.ocbf.defaults import merge_ocbf_best_parameters

BUS_TYPES = [0, 1]  # 0: bus occlusion off, 1: bus occlusion on


def _str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _crosswalk_plot_title(controller_type):
    pos_name = str((controller_type or {}).get("pos", "")).strip().lower()
    if pos_name == "occlusion_cbf_qp":
        return "Occlusion-Aware CBF"
    if pos_name == "cbf_qp":
        return "CBF-QP (Occlusion-agnostic)"
    if pos_name == "oa_mpc":
        return "OA-MPC"
    if not pos_name:
        return "Crosswalk Scenario"
    return pos_name.replace("_", " ").title()


def _default_crosswalk_svg_path(controller_type, model_key, bus_type, case_idx):
    pos_name = str((controller_type or {}).get("pos", "")).strip().lower() or "controller"
    model_name = str(model_key).strip().lower() or "model"
    case_name = f"idx{int(case_idx)}" if case_idx is not None else "single"
    out_dir = REPO_ROOT / "output" / "figures" / "crosswalk_svg"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{pos_name}_{model_name}_bus{int(bus_type)}_{case_name}.svg"


def _style_crosswalk_axis(ax, fig, title_text):
    fig.set_size_inches(12.8, 8.6, forward=True)
    # Keep the scene neutral so the robot and traffic behavior stand out.
    fig.set_facecolor("#f7f7f4")
    ax.set_facecolor("#f2f2ee")
    fig.subplots_adjust(left=0.02, right=0.985, top=0.965, bottom=0.03)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    header = ax.text(
        0.03,
        0.97,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#102a43",
        zorder=30,
    )
    header.set_path_effects(
        [
            path_effects.Stroke(linewidth=3.0, foreground="white", alpha=0.85),
            path_effects.Normal(),
        ]
    )


def _draw_crosswalk_scene(
    ax,
    env_width,
    env_height,
    bus_bounds,
    lane_y,
    lane_half,
    crosswalk_x_min,
    crosswalk_x_max,
):
    _, _, bus_min_y, bus_max_y = bus_bounds
    road_y_min = bus_min_y - 0.08
    road_y_max = 18.1
    lane_divider_y = bus_max_y + 0.22
    road_color = "#47515a"
    shoulder_color = "#cbb79a"
    curb_color = "#f7f1e3"
    centerline_color = "#f4d35e"
    crosswalk_color = "#fffaf0"

    ax.add_patch(
        patches.Rectangle(
            (0.0, road_y_min),
            env_width,
            road_y_max - road_y_min,
            facecolor=road_color,
            edgecolor="none",
            zorder=0.2,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (0.0, road_y_min - 0.55),
            env_width,
            0.55,
            facecolor=shoulder_color,
            edgecolor="none",
            zorder=0.15,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (0.0, road_y_max),
            env_width,
            0.55,
            facecolor=shoulder_color,
            edgecolor="none",
            zorder=0.15,
        )
    )
    ax.plot([0.0, env_width], [road_y_min, road_y_min], color=curb_color, linewidth=2.0, zorder=0.35)
    ax.plot([0.0, env_width], [road_y_max, road_y_max], color=curb_color, linewidth=2.0, zorder=0.35)

    dash_x = np.arange(1.5, env_width - 1.0, 3.2)
    for x0 in dash_x:
        ax.plot(
            [x0, min(x0 + 1.7, env_width - 0.5)],
            [lane_divider_y, lane_divider_y],
            color=centerline_color,
            linewidth=2.0,
            solid_capstyle="round",
            alpha=0.95,
            zorder=0.5,
        )

    crosswalk_center_x = 0.5 * (crosswalk_x_min + crosswalk_x_max)
    stripe_x = crosswalk_x_min
    stripe_w = crosswalk_x_max - crosswalk_x_min
    stripe_h = 0.62
    stripe_gap = 0.42
    # Start the zebra stripes so the lower lane shows four bars and the
    # bus body sits naturally inside the narrowed lower lane.
    y = max(road_y_min + 0.55, bus_min_y + 0.62)
    while y + stripe_h <= road_y_max - 0.35:
        ax.add_patch(
            patches.Rectangle(
                (stripe_x, y),
                stripe_w,
                stripe_h,
                facecolor=crosswalk_color,
                edgecolor="none",
                alpha=0.97,
                zorder=0.75,
            )
        )
        y += stripe_h + stripe_gap

    return {
        "road_y_min": road_y_min,
        "road_y_max": road_y_max,
        "lane_divider_y": lane_divider_y,
    }


def crosswalk_scenario_v3(
    controller_type=None,
    model_key="di",
    enable_plot=True,
    bus_type=0,
    batch_eval=False,
    num_trials=100,
    seed=42,
    case_idx=None,
    save_animation=False,
    save_frame_ext="png",
    animation_subdir=None,
    save_svg=False,
    svg_path=None,
    oa_paper_mode=None,
    oa_dynamic_occluders=None,
    oa_allow_solver_fallback=None,
    oa_dsafe=None,
    oa_visible_reach_mode=None,
    vref_scenario_softmax_kappa=None,
    v_adv_max_occ=None,
    occ_T_horizon=None,
):
    """
    [Scenario V3]
    1. Bus blocks lower-lane visibility (occlusion source)
    2. Opposite-lane cars move in a single lane without overlap
    3. Car speeds are randomized with a bounded maximum
    """
    if controller_type is None:
        controller_type = {"pos": "occlusion_cbf_qp"}

    dt = 0.05

    # User config for opposite-lane cars
    max_car_speed = 5.0
    lane_y = 15.5
    lane_half = 0.6

    # Map + ego route
    env_width = 40.0
    env_height = 30.0
    waypoints = np.array(
        [
            [20.0, 4.5, 0.0],
            [20.0, 23.0, 0.0],
        ],
        dtype=np.float64,
    )
    model_key = str(model_key).strip().lower()
    if model_key in {"di", "doubleintegrator2d"}:
        x_init = np.array([waypoints[0][0], waypoints[0][1], 0.0, 1.0, np.pi / 2.0], dtype=float)
    elif model_key in {"uni", "unicycle2d"}:
        x_init = np.array([waypoints[0][0], waypoints[0][1], np.pi / 2.0], dtype=float)
    elif model_key in {"du", "dynamicunicycle2d"}:
        x_init = np.array([waypoints[0][0], waypoints[0][1], np.pi / 2.0, 1.0], dtype=float)
    else:
        raise ValueError(f"Unsupported model `{model_key}`. Use one of di/uni/du.")

    # 1) Bus obstacle block (2 x 6 layout)
    bus_r = 1.1
    bus_rows = [10.3, 12.5]
    bus_dx = (17.0 - 12.0) / 3.0
    bus_left = 12.0 - 2.0 * bus_dx
    bus_cols = np.linspace(17.0, bus_left, 6)

    x_start = 40.0
    x_limit = -200.0

    def make_car_specs(rng):
        """
        Build periodic 3-car sets on one lane.

        Each set:
        - 3 cars arrive together (close intra-set spacing, randomized)
        - Set-to-set gap is randomized and larger than intra-set spacing
        - Rear car speed is sampled <= front car speed to avoid rear-end catch-up
        - Globally, every newly spawned rear car has speed <= nearest front car
          (prevents inter-set catch-up/overlap as well)
        """
        specs = []
        current_x = x_start

        set_size = 2
        set_gap_min, set_gap_max = 20.0, 30.1   # between sets
        intra_gap_min, intra_gap_max = 10.0, 10.01  # within one 3-car set
        # Randomize the very first visible set anchor so initial cars are
        # distributed differently inside the current plotting view (x in [0, 40]).
        # first_set_gap_min, first_set_gap_max = 15.0, 15.1
        first_set_gap_min, first_set_gap_max = 18.0, 18.1
        speed_drop_max = 1.2
        min_speed = 0.8 * max_car_speed
        prev_global_speed = None

        while current_x > x_limit:
            # Front car of the next set
            if prev_global_speed is None:
                set_gap = float(rng.uniform(first_set_gap_min, first_set_gap_max))
            else:
                set_gap = float(rng.uniform(set_gap_min, set_gap_max))
            current_x -= set_gap
            if current_x <= x_limit:
                break
            front_high = max_car_speed if prev_global_speed is None else min(max_car_speed, prev_global_speed)
            front_low = max(min_speed, front_high - speed_drop_max)
            if front_low > front_high:
                front_low = front_high
            front_speed = float(rng.uniform(front_low, front_high))
            front_speed = float(np.clip(front_speed, 0.0, max_car_speed))
            specs.append((set_gap, front_speed))
            prev_global_speed = front_speed

            # Remaining cars in the same set (rear cars), with non-increasing speed
            prev_speed = front_speed
            for _ in range(set_size - 1):
                intra_gap = float(rng.uniform(intra_gap_min, intra_gap_max))
                current_x -= intra_gap
                if current_x <= x_limit:
                    return specs

                v_high = prev_speed
                v_low = max(min_speed, prev_speed - speed_drop_max)
                if v_low > v_high:
                    v_low = v_high
                # Also enforce global non-increasing speed along the lane.
                if prev_global_speed is not None:
                    v_high = min(v_high, prev_global_speed)
                    v_low = min(v_low, v_high)
                c_speed = float(rng.uniform(v_low, v_high))
                c_speed = float(np.clip(c_speed, 0.0, max_car_speed))
                specs.append((intra_gap, c_speed))
                prev_speed = c_speed
                prev_global_speed = c_speed
        return specs

    def build_known_obs(bus_type, car_specs):
        obs_list = []
        for by in bus_rows:
            for bx in bus_cols:
                obs_list.append([bx, by, bus_r, 0.0, 0.0, 0.0, 16.0, bus_type])

        current_x = x_start
        for gap, c_speed in car_specs:
            current_x -= gap
            obs_list.append(
                [
                    current_x,
                    lane_y,
                    1.0,
                    c_speed,
                    0.0,
                    lane_y - lane_half,
                    lane_y + lane_half,
                    2,
                ]
            )
        return np.array(obs_list, dtype=float)

    def run_trial(
        bus_type,
        car_specs,
        trial_enable_plot,
        trial_save_animation,
        trial_save_frame_ext="png",
        trial_animation_subdir=None,
        trial_save_svg=False,
        trial_svg_path=None,
        return_infeasible=False,
    ):
        known_obs = build_known_obs(bus_type, car_specs)

        # 2) Robot spec
        robot_spec = {
            "radius": 0.25,
            "sensing_range": 25.0,
            "fov_angle": 360,
            "occ_visible_scale": 0.5,
            "debug_backup_qp": False,
            "show_qp_stats_text": False,
            "car_vis_length_scale": 1.18,
            "car_vis_width_scale": 1.12,
            "v_adv_max_occ": float(v_adv_max_occ) if v_adv_max_occ is not None else max_car_speed,
            "backup_cbf": {
                "T_horizon": float(occ_T_horizon) if occ_T_horizon is not None else 1.0,
                "dt_backup": 0.05,
                "alpha": 1.5,
                "rho_T": "stopping_distance",
                "vref_front_mode_occ": "los",
            },
            "occlusion_types": [int(bus_type)],
            "dynamic_obs_types": [2],
            "show_backup_rollout": False,
            "plot_occ_polygons": False,
            "dynamic_obs_rect_collision": True,
            "disable_occ_constraints": (bus_type == 0),
            "continue_on_infeasible": True,
            # Show infeasible marker and keep previous input when QP fails.
            "mark_qp_fail_infeasible": True,
            "use_occ": True,
            "enable_visible_hocbf_in_occ": True,
        }
        crosswalk_base_width = 4.9
        crosswalk_right_extra = 0.6
        crosswalk_center_x_base = float(waypoints[0][0]) + 1.0
        crosswalk_x_min = crosswalk_center_x_base - 0.5 * crosswalk_base_width
        crosswalk_x_max = crosswalk_center_x_base + 0.5 * crosswalk_base_width + crosswalk_right_extra
        corridor_buffer = float(robot_spec["radius"]) + 0.05
        robot_spec["position_corridor"] = {
            "enabled": True,
            "x_min": crosswalk_x_min,
            "x_max": crosswalk_x_max,
            "buffer": corridor_buffer,
            "alpha": 1.5,
            "alpha1": 1.5,
            "alpha2": 1.5,
        }
        if trial_save_animation:
            pos_name = str((controller_type or {}).get("pos", "")).strip().lower() or "controller"
            model_name = str(model_key).strip().lower() or "model"
            case_name = f"idx{int(case_idx)}" if case_idx is not None else "single"
            default_subdir = f"crosswalk_{pos_name}_{model_name}_bus{int(bus_type)}_{case_name}"
            robot_spec["animation_frame_ext"] = str(trial_save_frame_ext).strip().lower()
            robot_spec["animation_subdir"] = (
                str(trial_animation_subdir).strip() if trial_animation_subdir else default_subdir
            )
            robot_spec["animation_export_video"] = (
                str(robot_spec["animation_frame_ext"]).strip().lower() == "png"
            )
        if model_key in {"di", "doubleintegrator2d"}:
            robot_spec.update(
                {
                    "model": "DoubleIntegrator2D",
                    "v_max": 1.5,
                    "a_max": 2.0,
                }
            )
        elif model_key in {"uni", "unicycle2d"}:
            robot_spec.update(
                {
                    "model": "Unicycle2D",
                    "v_max": 2.0,
                    "w_max": 1.2,
                }
            )
        elif model_key in {"du", "dynamicunicycle2d"}:
            robot_spec.update(
                {
                    "model": "DynamicUnicycle2D",
                    "v_max": 2.0,
                    "a_max": 2.0,
                    "w_max": 1.2,
                }
            )
        pos_name = str(controller_type.get("pos", "")).strip().lower()
        if pos_name in {"occlusion_cbf", "occlusion_cbf_qp"}:
            tuned_backup, tuned_robot = merge_ocbf_best_parameters(
                robot_spec["model"],
                backup_defaults=robot_spec["backup_cbf"],
                robot_defaults=robot_spec,
            )
            robot_spec = tuned_robot
            robot_spec["backup_cbf"] = tuned_backup
        if occ_T_horizon is not None:
            robot_spec["backup_cbf"]["T_horizon"] = float(occ_T_horizon)
        if vref_scenario_softmax_kappa is not None:
            robot_spec["backup_cbf"]["vref_scenario_softmax_kappa"] = float(
                vref_scenario_softmax_kappa
            )

        if pos_name == "oa_mpc":
            oa_cfg = robot_spec.setdefault("oa_mpc", {})
            oa_cfg.setdefault("paper_mode", True)
            oa_cfg.setdefault("N", 10)
            oa_cfg.setdefault("visible_reach_mode", "worst_case")
            oa_cfg.setdefault("use_nominal_tracking_cost", False)
            oa_cfg.setdefault("dynamic_occluders", False)
            if oa_paper_mode is not None:
                oa_cfg["paper_mode"] = bool(oa_paper_mode)
            if oa_dynamic_occluders is not None:
                oa_cfg["dynamic_occluders"] = bool(oa_dynamic_occluders)
            if oa_allow_solver_fallback is not None:
                oa_cfg["allow_solver_fallback"] = bool(oa_allow_solver_fallback)
            if oa_dsafe is not None:
                oa_cfg["dsafe"] = float(oa_dsafe)
            if oa_visible_reach_mode is not None:
                oa_cfg["visible_reach_mode"] = str(oa_visible_reach_mode).strip().lower()

        # 3) Plot setup (optional)
        ax = None
        fig = None
        bus_indices = []
        bus_shadow = None
        bus_patch = None
        bus_label = None
        bus_window_patches = []
        bus_door_patch = None
        bus_occ_patch = None
        car_indices = []
        car_bodies = []
        car_cabins = []
        car_light_bars = []
        car_windshields = []
        car_rear_windows = []
        car_front_wheels = []
        car_rear_wheels = []
        car_last_occ = None
        bus_min_x = bus_max_x = bus_min_y = bus_max_y = 0.0

        if trial_enable_plot:
            plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
            plot_title = _crosswalk_plot_title(controller_type)
            ax, fig = plot_handler.plot_grid("")
            _style_crosswalk_axis(ax, fig, plot_title)

            bus_count = len(bus_rows) * len(bus_cols)
            bus_indices = list(range(bus_count))
            bus_min_x = float(np.min(bus_cols) - bus_r)
            bus_max_x = float(np.max(bus_cols) + bus_r)
            bus_min_y = float(np.min(bus_rows) - bus_r)
            bus_max_y = float(np.max(bus_rows) + bus_r)

            length_scale = float(robot_spec.get("bus_vis_length_scale", 1.0))
            width_scale = float(robot_spec.get("bus_vis_width_scale", 1.0))
            if length_scale != 1.0 or width_scale != 1.0:
                cx = 0.5 * (bus_min_x + bus_max_x)
                cy = 0.5 * (bus_min_y + bus_max_y)
                half_len = 0.5 * (bus_max_x - bus_min_x) * length_scale
                half_w = 0.5 * (bus_max_y - bus_min_y) * width_scale
                bus_min_x = cx - half_len
                bus_max_x = cx + half_len
                bus_min_y = cy - half_w
                bus_max_y = cy + half_w

            _draw_crosswalk_scene(
                ax,
                env_width=env_width,
                env_height=env_height,
                bus_bounds=(bus_min_x, bus_max_x, bus_min_y, bus_max_y),
                lane_y=lane_y,
                lane_half=lane_half,
                crosswalk_x_min=crosswalk_x_min,
                crosswalk_x_max=crosswalk_x_max,
            )

            bus_w = bus_max_x - bus_min_x
            bus_h = bus_max_y - bus_min_y
            bus_shadow = patches.FancyBboxPatch(
                (bus_min_x + 0.18, bus_min_y - 0.18),
                bus_w,
                bus_h,
                boxstyle="round,pad=0.02,rounding_size=0.22",
                edgecolor="none",
                facecolor="#102a43",
                alpha=0.16,
                zorder=6.4,
            )
            ax.add_patch(bus_shadow)
            bus_patch = patches.FancyBboxPatch(
                (bus_min_x, bus_min_y),
                bus_w,
                bus_h,
                boxstyle="round,pad=0.02,rounding_size=0.22",
                edgecolor="#102a43",
                linewidth=1.6,
                facecolor="#4b82bc",
                fill=True,
                zorder=7.0,
            )
            ax.add_patch(bus_patch)

            n_bus_windows = 5
            window_margin_x = 0.36
            window_gap = 0.18
            window_h = 0.34 * bus_h
            window_y = bus_min_y + 0.50 * bus_h - 0.5 * window_h
            win_total_w = bus_w - 2.0 * window_margin_x - (n_bus_windows - 1) * window_gap
            win_w = max(win_total_w / float(n_bus_windows), 0.18)
            for i in range(n_bus_windows):
                wx = bus_min_x + window_margin_x + i * (win_w + window_gap)
                win = patches.Rectangle(
                    (wx, window_y),
                    win_w,
                    window_h,
                    edgecolor="none",
                    facecolor="#d9e8f5",
                    alpha=0.92,
                    zorder=7.5,
                )
                ax.add_patch(win)
                bus_window_patches.append(win)

            bus_door_patch = patches.Rectangle(
                (bus_max_x - 0.85, bus_min_y + 0.18),
                0.42,
                bus_h - 0.36,
                edgecolor="#0b2033",
                linewidth=0.8,
                facecolor="#6697cb",
                alpha=0.95,
                zorder=7.6,
            )
            ax.add_patch(bus_door_patch)
            bus_label = ax.text(
                bus_min_x + 0.5 * bus_w,
                bus_min_y + 0.52 * bus_h,
                "BUS",
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                zorder=8,
            )
            bus_label.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.5, foreground="#17406d", alpha=0.95),
                    path_effects.Normal(),
                ]
            )

            car_indices = [
                idx for idx, row in enumerate(known_obs) if len(row) >= 8 and int(row[7]) == 2
            ]
            car_length_scale = float(robot_spec.get("car_vis_length_scale", 1.0))
            car_width_scale = float(robot_spec.get("car_vis_width_scale", 1.0))
            for idx in car_indices:
                r = float(known_obs[idx][2])
                car_width = 2.0 * r * car_width_scale
                car_length = 4.0 * r * car_length_scale
                body = patches.FancyBboxPatch(
                    (0.0, 0.0),
                    car_length,
                    car_width,
                    boxstyle="round,pad=0.01,rounding_size=0.24",
                    edgecolor="#1f2933",
                    linewidth=1.2,
                    facecolor="#6f7f8f",
                    fill=True,
                    zorder=5.6,
                )
                body.set_visible(False)
                ax.add_patch(body)
                cabin = patches.FancyBboxPatch(
                    (0.0, 0.0),
                    0.50 * car_length,
                    0.62 * car_width,
                    boxstyle="round,pad=0.01,rounding_size=0.18",
                    edgecolor="none",
                    facecolor="#8ea2b4",
                    alpha=0.98,
                    zorder=5.85,
                )
                cabin.set_visible(False)
                ax.add_patch(cabin)
                windshield = patches.FancyBboxPatch(
                    (0.0, 0.0),
                    0.18 * car_length,
                    0.46 * car_width,
                    boxstyle="round,pad=0.01,rounding_size=0.12",
                    edgecolor="none",
                    facecolor="#dfeaf5",
                    alpha=0.96,
                    zorder=5.95,
                )
                windshield.set_visible(False)
                ax.add_patch(windshield)
                rear_window = patches.FancyBboxPatch(
                    (0.0, 0.0),
                    0.14 * car_length,
                    0.40 * car_width,
                    boxstyle="round,pad=0.01,rounding_size=0.10",
                    edgecolor="none",
                    facecolor="#cad9e8",
                    alpha=0.92,
                    zorder=5.92,
                )
                rear_window.set_visible(False)
                ax.add_patch(rear_window)
                light_bar = patches.Rectangle(
                    (0.0, 0.0),
                    0.12 * car_length,
                    0.14 * car_width,
                    edgecolor="none",
                    facecolor="#ffe8a3",
                    alpha=0.9,
                    zorder=6.0,
                )
                light_bar.set_visible(False)
                ax.add_patch(light_bar)
                front_wheel = patches.Ellipse(
                    (0.0, 0.0),
                    0.16 * car_length,
                    0.22 * car_width,
                    edgecolor="none",
                    facecolor="#1f2933",
                    alpha=0.98,
                    zorder=5.55,
                )
                front_wheel.set_visible(False)
                ax.add_patch(front_wheel)
                rear_wheel = patches.Ellipse(
                    (0.0, 0.0),
                    0.16 * car_length,
                    0.22 * car_width,
                    edgecolor="none",
                    facecolor="#1f2933",
                    alpha=0.98,
                    zorder=5.55,
                )
                rear_wheel.set_visible(False)
                ax.add_patch(rear_wheel)
                car_bodies.append(body)
                car_cabins.append(cabin)
                car_light_bars.append(light_bar)
                car_windshields.append(windshield)
                car_rear_windows.append(rear_window)
                car_front_wheels.append(front_wheel)
                car_rear_wheels.append(rear_wheel)
            car_last_occ = np.full(len(car_bodies), -1, dtype=np.int8)

            if bus_type != 0:
                bus_occ_patch = patches.Polygon(
                    np.zeros((4, 2)),
                    closed=True,
                    fill=True,
                    facecolor="#7aa6c2",
                    edgecolor="none",
                    alpha=0.22,
                    zorder=1.2,
                )
                bus_occ_patch.set_visible(False)
                ax.add_patch(bus_occ_patch)

        env_handler = env.Env()

        tracking_controller = LocalTrackingControllerDyn_OCC(
            x_init,
            robot_spec,
            controller_type=controller_type,
            dt=dt,
            show_animation=trial_enable_plot,
            save_animation=trial_save_animation,
            ax=ax,
            fig=fig,
            env=env_handler,
        )

        # Keep enough obstacle constraints for dense traffic.
        tracking_controller.num_constraints = 30
        if hasattr(tracking_controller, "pos_controller") and hasattr(tracking_controller.pos_controller, "num_obs"):
            tracking_controller.pos_controller.num_obs = 30

        tracking_controller.obs = known_obs
        obs_meta = []
        for row in known_obs:
            v_mag = np.hypot(row[3], row[4])
            obs_meta.append({"mode": 0, "v_max": v_mag})

        tracking_controller.set_obs_meta(obs_meta)
        tracking_controller.set_waypoints(waypoints)

        if trial_enable_plot:
            if getattr(tracking_controller, "waypoints_scatter", None) is not None:
                tracking_controller.waypoints_scatter.set_visible(False)
            start_xy = np.asarray(waypoints[0][:2], dtype=float)
            goal_xy = np.asarray(waypoints[-1][:2], dtype=float)
            start_marker = ax.scatter(
                [start_xy[0]],
                [start_xy[1]],
                s=90,
                facecolors="#f8fafc",
                edgecolors="#0f172a",
                linewidths=1.4,
                zorder=9.2,
            )
            goal_marker = ax.scatter(
                [goal_xy[0]],
                [goal_xy[1]],
                s=430,
                marker="*",
                facecolors="#f6bd60",
                edgecolors="none",
                linewidths=0.0,
                zorder=8.4,
            )
            goal_text = ax.text(
                goal_xy[0],
                goal_xy[1] + 1.15,
                "GOAL",
                color="#102a43",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                zorder=9.5,
            )
            goal_text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.5, foreground="white", alpha=0.9),
                    path_effects.Normal(),
                ]
            )
            start_marker.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.2, foreground="white", alpha=0.9),
                    path_effects.Normal(),
                ]
            )
            if hasattr(tracking_controller.robot, "body"):
                try:
                    tracking_controller.robot.body.set_zorder(9.0)
                except Exception:
                    pass
            if getattr(tracking_controller.robot, "axis", None) is not None:
                tracking_controller.robot.axis.set_linewidth(2.4)
                tracking_controller.robot.axis.set_zorder(9.1)

            orig_render_dyn_obs = tracking_controller.render_dyn_obs
            sensing_range = float(robot_spec.get("sensing_range", 20.0))

            def compute_bus_occ_poly(px, py):
                if bus_min_x <= px <= bus_max_x and bus_min_y <= py <= bus_max_y:
                    return None
                corners = np.array(
                    [
                        [bus_min_x, bus_min_y],
                        [bus_min_x, bus_max_y],
                        [bus_max_x, bus_min_y],
                        [bus_max_x, bus_max_y],
                    ],
                    dtype=float,
                )
                vecs = corners - np.array([px, py])
                norms = np.linalg.norm(vecs, axis=1)
                if np.any(norms < 1e-9):
                    return None
                angles = np.arctan2(vecs[:, 1], vecs[:, 0])
                order = np.argsort(angles)
                ang_sorted = angles[order]
                gaps = np.diff(np.r_[ang_sorted, ang_sorted[0] + 2.0 * np.pi])
                max_gap_idx = int(np.argmax(gaps))
                start_idx = (max_gap_idx + 1) % len(angles)
                end_idx = max_gap_idx
                idx_start = order[start_idx]
                idx_end = order[end_idx]
                t1 = corners[idx_start]
                t2 = corners[idx_end]
                d1 = t1 - np.array([px, py])
                d2 = t2 - np.array([px, py])
                d1 /= np.linalg.norm(d1)
                d2 /= np.linalg.norm(d2)
                far1 = np.array([px, py]) + d1 * sensing_range
                far2 = np.array([px, py]) + d2 * sensing_range
                return np.array([t1, t2, far2, far1])

            def render_dyn_obs_with_bus(self):
                orig_render_dyn_obs()
                if self.dyn_obs_patch is not None:
                    for idx in bus_indices:
                        if idx < len(self.dyn_obs_patch):
                            self.dyn_obs_patch[idx].set_visible(False)
                        if idx < len(self.obs_vel_arrows):
                            self.obs_vel_arrows[idx].set_visible(False)
                    for idx in car_indices:
                        if idx < len(self.dyn_obs_patch):
                            self.dyn_obs_patch[idx].set_visible(False)

                occluded_mask = self._cached_occluded_mask
                for rect_i, (body, cabin, light_bar, windshield, rear_window, front_wheel, rear_wheel, idx) in enumerate(
                    zip(
                        car_bodies,
                        car_cabins,
                        car_light_bars,
                        car_windshields,
                        car_rear_windows,
                        car_front_wheels,
                        car_rear_wheels,
                        car_indices,
                    )
                ):
                    if idx >= len(self.obs) or not self.plot_dyn_obs:
                        body.set_visible(False)
                        cabin.set_visible(False)
                        light_bar.set_visible(False)
                        windshield.set_visible(False)
                        rear_window.set_visible(False)
                        front_wheel.set_visible(False)
                        rear_wheel.set_visible(False)
                        continue
                    obs_info = self.obs[idx]
                    ox, oy, r = obs_info[:3]
                    car_width = 2.0 * r * car_width_scale
                    car_length = 4.0 * r * car_length_scale
                    body_xy = (ox - 0.5 * car_length, oy - 0.5 * car_width)
                    body.set_bounds(body_xy[0], body_xy[1], car_length, car_width)
                    cabin_w = 0.50 * car_length
                    cabin_h = 0.62 * car_width
                    cabin_xy = (
                        ox - 0.5 * cabin_w + 0.02 * car_length,
                        oy - 0.5 * cabin_h,
                    )
                    cabin.set_bounds(cabin_xy[0], cabin_xy[1], cabin_w, cabin_h)
                    windshield_w = 0.18 * car_length
                    windshield_h = 0.46 * car_width
                    windshield.set_bounds(
                        ox + 0.12 * car_length,
                        oy - 0.5 * windshield_h,
                        windshield_w,
                        windshield_h,
                    )
                    rear_window_w = 0.14 * car_length
                    rear_window_h = 0.40 * car_width
                    rear_window.set_bounds(
                        ox - 0.34 * car_length,
                        oy - 0.5 * rear_window_h,
                        rear_window_w,
                        rear_window_h,
                    )
                    light_bar.set_width(0.12 * car_length)
                    light_bar.set_height(0.14 * car_width)
                    light_bar.set_xy(
                        (
                            ox + 0.5 * car_length - 0.15 * car_length,
                            oy - 0.07 * car_width,
                        )
                    )
                    front_wheel.width = 0.16 * car_length
                    front_wheel.height = 0.22 * car_width
                    front_wheel.center = (
                        ox + 0.18 * car_length,
                        oy - 0.34 * car_width,
                    )
                    rear_wheel.width = 0.16 * car_length
                    rear_wheel.height = 0.22 * car_width
                    rear_wheel.center = (
                        ox - 0.18 * car_length,
                        oy - 0.34 * car_width,
                    )
                    is_occ = int(occluded_mask is not None and bool(occluded_mask[idx]))
                    if car_last_occ is None or car_last_occ[rect_i] != is_occ:
                        body.set_facecolor("#ef8354" if is_occ else "#6f7f8f")
                        cabin.set_facecolor("#f4b08f" if is_occ else "#8ea2b4")
                        windshield.set_facecolor("#fde5d8" if is_occ else "#dfeaf5")
                        rear_window.set_facecolor("#f6d4c0" if is_occ else "#cad9e8")
                        light_bar.set_facecolor("#ffd166" if is_occ else "#fff2bf")
                        if car_last_occ is not None:
                            car_last_occ[rect_i] = is_occ
                    body.set_visible(True)
                    cabin.set_visible(True)
                    light_bar.set_visible(True)
                    windshield.set_visible(True)
                    rear_window.set_visible(True)
                    front_wheel.set_visible(True)
                    rear_wheel.set_visible(True)

                if bus_occ_patch is not None:
                    poly = compute_bus_occ_poly(float(self.robot.X[0, 0]), float(self.robot.X[1, 0]))
                    if poly is None:
                        bus_occ_patch.set_visible(False)
                    else:
                        bus_occ_patch.set_xy(poly)
                        bus_occ_patch.set_visible(True)

                if bus_shadow is not None:
                    bus_shadow.set_visible(True)
                if bus_patch is not None:
                    bus_patch.set_visible(True)
                if bus_door_patch is not None:
                    bus_door_patch.set_visible(True)
                for win in bus_window_patches:
                    win.set_visible(True)
                if bus_label is not None:
                    bus_label.set_visible(True)

            tracking_controller.render_dyn_obs = types.MethodType(render_dyn_obs_with_bus, tracking_controller)

        result = tracking_controller.run_all_steps(tf=400)
        if trial_save_svg and fig is not None:
            svg_out = Path(trial_svg_path).expanduser() if trial_svg_path else _default_crosswalk_svg_path(
                controller_type=controller_type,
                model_key=model_key,
                bus_type=bus_type,
                case_idx=case_idx,
            )
            svg_out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(svg_out, format="svg", dpi=300, bbox_inches="tight")
            print(f"Saved SVG: {svg_out}")
        if return_infeasible:
            infeasible_seen = bool(getattr(tracking_controller, "_infeasible_seen", False))
            return result, infeasible_seen
        return result

    if case_idx is not None:
        if case_idx < 1:
            raise ValueError("case_idx must be >= 1 (matches printed Case numbers).")
        rng = np.random.default_rng(seed)
        car_specs = None
        for _ in range(case_idx):
            car_specs = make_car_specs(rng)
        return run_trial(
            bus_type,
            car_specs,
            enable_plot,
            save_animation,
            trial_save_frame_ext=save_frame_ext,
            trial_animation_subdir=animation_subdir,
            trial_save_svg=save_svg,
            trial_svg_path=svg_path,
        )

    if batch_eval:
        import contextlib
        import io

        rng = np.random.default_rng(seed)
        case_indices = []
        for idx in range(num_trials):
            car_specs = make_car_specs(rng)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result0, _ = run_trial(
                    0,
                    car_specs,
                    trial_enable_plot=False,
                    trial_save_animation=False,
                    trial_save_frame_ext=save_frame_ext,
                    trial_animation_subdir=animation_subdir,
                    return_infeasible=True,
                )
                result1, infeasible1 = run_trial(
                    1,
                    car_specs,
                    trial_enable_plot=False,
                    trial_save_animation=False,
                    trial_save_frame_ext=save_frame_ext,
                    trial_animation_subdir=animation_subdir,
                    return_infeasible=True,
                )
            if result1 == -1 and (not infeasible1) and result0 == -2:
                case_indices.append(idx + 1)
                print(f"[Case {idx + 1}] type1 goal, type0 collision")
        print(f"Matched cases (goal vs collision): {case_indices}")
        return case_indices

    rng = np.random.default_rng(seed)
    car_specs = make_car_specs(rng)
    return run_trial(
        bus_type,
        car_specs,
        enable_plot,
        save_animation,
        trial_save_frame_ext=save_frame_ext,
        trial_animation_subdir=animation_subdir,
        trial_save_svg=save_svg,
        trial_svg_path=svg_path,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run crosswalk_scenario_v3 in occlusion-cbf framework.")
    parser.add_argument("--model", default="di", help="Model alias: di | uni | du")
    parser.add_argument("--controller", default="occlusion_cbf_qp", help="Position controller type.")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=CROSSWALK_BASELINE_CHOICES,
        help="Baseline alias. If provided, overrides --controller.",
    )
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument(
        "--bus",
        "--bus-type",
        dest="bus_type",
        type=int,
        default=0,
        choices=BUS_TYPES,
        help="Bus occlusion mode: 0 (off), 1 (on).",
    )
    parser.add_argument("--batch-eval", action="store_true", help="Run batch search mode.")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of batch trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--idx",
        "--case-idx",
        dest="case_idx",
        type=int,
        default=None,
        help="Run only the N-th generated case (1-based).",
    )
    parser.add_argument(
        "--save_ani",
        "--save-ani",
        "--save-animation",
        dest="save_animation",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Save animation frames. Accepts true/false or can be passed as a flag.",
    )
    parser.add_argument(
        "--save-frame-ext",
        type=str,
        choices=["png", "svg"],
        default="png",
        help="Animation frame format when --save-animation is enabled.",
    )
    parser.add_argument(
        "--animation-subdir",
        type=str,
        default=None,
        help="Optional subdirectory under output/animations for saved frames.",
    )
    parser.add_argument(
        "--save-svg",
        dest="save_svg",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Save the final rendered frame as an SVG file.",
    )
    parser.add_argument(
        "--svg-path",
        type=str,
        default=None,
        help="Optional explicit output path for the saved SVG.",
    )
    parser.add_argument(
        "--oa-paper-mode",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="OA-MPC: use paper-faithful defaults (True/False).",
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
        "--vref-scenario-softmax-kappa",
        type=float,
        default=None,
        help="Occlusion backup v_ref: scenario-level softmax kappa. Applies in both strict and soft v_ref modes.",
    )
    parser.add_argument(
        "--v-adv-max-occ",
        type=float,
        default=None,
        help="Occlusion backup: assumed hidden-agent speed bound used to expand the occlusion reachable set.",
    )
    parser.add_argument(
        "--occ-T-horizon",
        type=float,
        default=None,
        help="Occlusion backup: horizon length used by the backup rollout and occlusion v_ref construction.",
    )
    args = parser.parse_args(argv)

    model_key = str(args.model).strip().lower()
    if model_key not in {"di", "doubleintegrator2d", "uni", "unicycle2d", "du", "dynamicunicycle2d"}:
        raise ValueError(f"Unsupported model `{args.model}`. Use one of di/uni/du.")

    pos_algo = resolve_baseline_alias(args.baseline, args.controller, CROSSWALK_BASELINE_MAP)
    controller_type = {"pos": pos_algo}
    save_frame_ext = str(args.save_frame_ext).strip().lower()
    # In crosswalk runs, users often expect `--save-svg true --save-ani true`
    # to save every animation frame as SVG rather than only the last snapshot.
    if args.save_animation and args.save_svg and save_frame_ext == "png":
        save_frame_ext = "svg"
    crosswalk_scenario_v3(
        controller_type=controller_type,
        model_key=model_key,
        enable_plot=not args.disable_plot,
        bus_type=args.bus_type,
        batch_eval=args.batch_eval,
        num_trials=args.num_trials,
        seed=args.seed,
        case_idx=args.case_idx,
        save_animation=args.save_animation,
        save_frame_ext=save_frame_ext,
        animation_subdir=args.animation_subdir,
        save_svg=args.save_svg,
        svg_path=args.svg_path,
        oa_paper_mode=args.oa_paper_mode,
        oa_dynamic_occluders=args.oa_dynamic_occluders,
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        oa_dsafe=args.oa_dsafe,
        oa_visible_reach_mode=args.oa_visible_reach_mode,
        vref_scenario_softmax_kappa=args.vref_scenario_softmax_kappa,
        v_adv_max_occ=args.v_adv_max_occ,
        occ_T_horizon=args.occ_T_horizon,
    )


if __name__ == "__main__":
    main()
