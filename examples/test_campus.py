"""
Campus sidewalk scenario built on top of the crowd benchmark controller stack.

This is intentionally a thin wrapper around `examples/test_crowd.py`:
- same 2D controller/plotting pipeline
- different workspace size and route
- campus-specific random dynamic pedestrian placement for trial sweeps

Run:
    uv run python examples/test_campus.py --model di --baseline occlusion_cbf
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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


ENV_WIDTH = 15.0
ENV_HEIGHT = 40.0
GOAL_THRESHOLD = 0.5
PEDESTRIAN_SPEED = 0.3
PEDESTRIAN_SPEED_MAX = 0.5
PEDESTRIAN_RADIUS = 0.35
DEFAULT_N_RAND = 12
DEFAULT_OCC_VISIBLE_SCALE = 0.7
DEFAULT_HIDDEN_OBS_VELOCITY = 1.0
WAYPOINTS = np.array(
    [
        [7.5, 1.0, 0.5 * np.pi],
        [7.5, 36.0, 0.0],
    ],
    dtype=np.float64,
)
PEDESTRIAN_X_BOUNDS = (2.5, 12.5)
PEDESTRIAN_Y_RANGE = (6.0, 40.0)
PLOT_FIGSIZE = (5.4, 12.4)


def _resolve_model_name(model_key):
    mk = str(model_key).strip().lower()
    if mk in {"di", "doubleintegrator2d"}:
        return "DoubleIntegrator2D"
    if mk in {"du", "dynamicunicycle2d"}:
        return "DynamicUnicycle2D"
    if mk in {"uni", "unicycle2d", "un"}:
        return "Unicycle2D"
    raise ValueError(f"Unsupported model `{model_key}`. Use `di`, `du`, or `uni`.")


def _build_campus_model_defaults(model_key, controller_type=None, vref_mode_occ=None, oa_wmax="default"):
    model = _resolve_model_name(model_key)
    is_oa_mpc = str((controller_type or {}).get("pos", "")).strip().lower() == "oa_mpc"

    if model == "DoubleIntegrator2D":
        di_backup_cfg = {
            "T_horizon": 1.0,
            "vref_scenario_softmax_kappa": 0.0,
            "rho_T": "auto",
        }
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
        du_backup_cfg = {
            "T_horizon": 1.0,
            "vref_scenario_softmax_kappa": 0.0,
            "du_nonnegative_speed_occ_du": False,
        }
        if vref_mode_occ is not None:
            du_backup_cfg["vref_mode_occ_du"] = str(vref_mode_occ).strip().lower()
        robot_spec = {
            "model": "DynamicUnicycle2D",
            "v_max": 1.0,
            "v_min": -1.0,
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
    else:
        uni_backup_cfg = {
            "T_horizon": 0.5,
            "vref_scenario_softmax_kappa": 0.0,
        }
        if vref_mode_occ is not None:
            uni_backup_cfg["vref_mode_occ_uni"] = str(vref_mode_occ).strip().lower()
        wmax_mode = str(oa_wmax).strip().lower()
        if wmax_mode not in {"default", "pi"}:
            raise ValueError(f"Invalid --wmax: {oa_wmax}. Use one of: default, pi.")
        uni_wmax = float(np.pi) if (is_oa_mpc and wmax_mode == "pi") else 0.8
        robot_spec = {
            "model": "Unicycle2D",
            "v_max": 1.0,
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

    backup_cfg = dict(robot_spec.pop("backup_cbf", {}))
    return model, backup_cfg, dict(robot_spec)


def _build_backup_cbf_overrides(args):
    backup_cbf_overrides = {}
    if args.du_min_speed_scale is not None:
        backup_cbf_overrides["min_speed_scale_occ_du"] = float(args.du_min_speed_scale)
    if args.du_k_turn_brake is not None:
        backup_cbf_overrides["k_turn_brake_occ_du"] = float(args.du_k_turn_brake)
    if args.du_k_a_p is not None:
        backup_cbf_overrides["k_a_occ_du_p"] = float(args.du_k_a_p)
        backup_cbf_overrides["k_a_track_occ_du_p"] = float(args.du_k_a_p)
    if args.du_k_a_d is not None:
        backup_cbf_overrides["k_a_occ_du_d"] = float(args.du_k_a_d)
        backup_cbf_overrides["k_a_track_occ_du_d"] = float(args.du_k_a_d)
    if args.du_reverse_enter_cos is not None:
        backup_cbf_overrides["reverse_enter_cos_occ_du"] = float(args.du_reverse_enter_cos)
    if args.du_reverse_exit_cos is not None:
        backup_cbf_overrides["reverse_exit_cos_occ_du"] = float(args.du_reverse_exit_cos)
    if args.du_reverse_min_scale is not None:
        backup_cbf_overrides["reverse_min_scale_occ_du"] = float(args.du_reverse_min_scale)
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
    if getattr(args, "occ_terminal_slack_weight", None) is not None:
        backup_cbf_overrides["terminal_slack_weight"] = float(args.occ_terminal_slack_weight)
    if getattr(args, "occ_terminal_slack_max", None) is not None:
        backup_cbf_overrides["terminal_slack_max"] = float(args.occ_terminal_slack_max)
    if args.vref is not None:
        backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    return backup_cbf_overrides or None


def _build_robot_spec_overrides(args):
    robot_spec_overrides = {}
    if args.robot_v_max is not None:
        robot_spec_overrides["v_max"] = float(args.robot_v_max)
    if args.robot_v_min is not None:
        robot_spec_overrides["v_min"] = float(args.robot_v_min)
    if args.robot_a_max is not None:
        robot_spec_overrides["a_max"] = float(args.robot_a_max)
    if args.robot_w_max is not None:
        robot_spec_overrides["w_max"] = float(args.robot_w_max)
    if args.robot_radius is not None:
        robot_spec_overrides["radius"] = float(args.robot_radius)
    if args.di_qp_weight_ax is not None:
        robot_spec_overrides["qp_weight_ax"] = float(args.di_qp_weight_ax)
    if args.di_qp_weight_ay is not None:
        robot_spec_overrides["qp_weight_ay"] = float(args.di_qp_weight_ay)
    oacp_cfg = _build_oacp_overrides(args)
    if oacp_cfg is not None:
        robot_spec_overrides["oacp_mpc"] = oacp_cfg
    return robot_spec_overrides or None


def _build_oacp_overrides(args):
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
    return oacp_cfg or None


def _sample_campus_spawn_x(rng, x_center):
    x_min, x_max = map(float, PEDESTRIAN_X_BOUNDS)
    if float(rng.random()) < 0.7:
        return float(np.clip(rng.normal(loc=float(x_center), scale=1.6), x_min, x_max))
    return float(rng.uniform(x_min, x_max))


def _build_random_pedestrians(
    seed,
    case_idx=None,
    n_rand=DEFAULT_N_RAND,
    ped_speed_min=PEDESTRIAN_SPEED,
    ped_speed_max=PEDESTRIAN_SPEED_MAX,
    ped_radius=PEDESTRIAN_RADIUS,
):
    case_seed = crowd1._compute_case_seed(seed, case_idx)
    rng = np.random.default_rng(case_seed)
    y_min, y_max = (float(PEDESTRIAN_Y_RANGE[0]), float(PEDESTRIAN_Y_RANGE[1]))
    x_min, x_max = (float(PEDESTRIAN_X_BOUNDS[0]), float(PEDESTRIAN_X_BOUNDS[1]))
    ped_speed_min = float(ped_speed_min)
    ped_speed_max = float(ped_speed_max)
    ped_radius = float(ped_radius)
    n_rand = int(max(0, n_rand))
    if (not np.isfinite(ped_speed_min)) or (not np.isfinite(ped_speed_max)):
        raise ValueError("pedestrian speed bounds must be finite.")
    if ped_speed_min <= 0.0 or ped_speed_max <= 0.0:
        raise ValueError("pedestrian speed bounds must be positive.")
    if ped_speed_min > ped_speed_max:
        raise ValueError("pedestrian speed bounds must satisfy min <= max.")
    waypoints_xy = np.asarray(WAYPOINTS[:, :2], dtype=float)
    start_xy = waypoints_xy[0]
    goal_xy = waypoints_xy[-1]
    x_center = float(np.mean(waypoints_xy[:, 0]))
    route_y_min = float(np.min(waypoints_xy[:, 1]))
    route_y_max = float(np.max(waypoints_xy[:, 1]))
    spawn_y_min = y_min + max(1.2, 2.5 * ped_radius)
    spawn_y_max = y_max - max(1.2, 2.5 * ped_radius)
    start_goal_clearance = max(1.5, 3.0 * ped_radius)
    pair_margin = max(0.20, 0.5 * ped_radius)
    endpoint_band = max(2.5, 4.0 * ped_radius)
    centerline_guard = max(0.45, 1.2 * ped_radius)

    rows = []
    obs_meta = []
    for _ in range(n_rand):
        placed = False
        for _ in range(256):
            px = _sample_campus_spawn_x(rng, x_center)
            py = float(rng.uniform(spawn_y_min, spawn_y_max))

            if np.linalg.norm(np.array([px, py], dtype=float) - start_xy) < start_goal_clearance:
                continue
            if np.linalg.norm(np.array([px, py], dtype=float) - goal_xy) < start_goal_clearance:
                continue
            if crowd1._rows_overlap((px, py), ped_radius, rows, margin=pair_margin):
                continue

            dist_to_path = crowd1._point_polyline_distance((px, py), waypoints_xy)
            near_start_end = (py <= route_y_min + endpoint_band) or (py >= route_y_max - endpoint_band)
            if near_start_end and dist_to_path < centerline_guard:
                continue

            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"Could not place {n_rand} campus pedestrians without overlap in the current workspace "
                f"(seed={case_seed}, placed={len(rows)})."
            )

        theta0 = float(rng.uniform(-np.pi, np.pi))
        ped_speed_i = float(rng.uniform(ped_speed_min, ped_speed_max))
        vx0 = float(ped_speed_i * np.cos(theta0))
        vy0 = float(ped_speed_i * np.sin(theta0))
        rows.append([float(px), float(py), ped_radius, vx0, vy0, y_min, y_max, 1.0])
        obs_meta.append(
            {
                "mode": 1,
                "v_max": ped_speed_i,
                "theta": theta0,
                "x_min": x_min,
                "x_max": x_max,
            }
        )

    scenario_diag = {
        "crowd_mode": "random",
        "layout_name": "campus_walk",
        "n_background_rand": int(len(rows)),
        "pedestrian_speed_range": [float(ped_speed_min), float(ped_speed_max)],
        "pedestrian_radius": ped_radius,
        "workspace": {"width": float(ENV_WIDTH), "height": float(ENV_HEIGHT)},
        "x_bounds": [x_min, x_max],
        "y_range": [y_min, y_max],
        "start_goal_clearance": float(start_goal_clearance),
        "spawn_y_range": [float(spawn_y_min), float(spawn_y_max)],
    }
    return np.asarray(rows, dtype=float), obs_meta, scenario_diag


def run_campus_scenario(
    controller_type=None,
    model_key="di",
    show_animation=True,
    save_animation=False,
    tf=200.0,
    seed=0,
    case_idx=None,
    n_rand=DEFAULT_N_RAND,
    ped_speed=None,
    ped_speed_min=PEDESTRIAN_SPEED,
    ped_speed_max=PEDESTRIAN_SPEED_MAX,
    ped_radius=PEDESTRIAN_RADIUS,
    sensing_range=8.0,
    goal_threshold=GOAL_THRESHOLD,
    vref_mode_occ=None,
    vref_front_mode_occ=None,
    occ_visible_scale=DEFAULT_OCC_VISIBLE_SCALE,
    hidden_obs_velocity=DEFAULT_HIDDEN_OBS_VELOCITY,
    occ_version=None,
    occ_enable_visible_hocbf=None,
    oa_dynamic_occluders=None,
    oa_allow_solver_fallback=None,
    oa_dsafe=None,
    oa_visible_reach_mode=None,
    oa_use_nominal_tracking_cost=None,
    oa_wmax="default",
    oa_dt=None,
    backup_cbf_overrides=None,
    robot_spec_overrides=None,
):
    known_obs, obs_meta, scenario_diag = _build_random_pedestrians(
        seed=seed,
        case_idx=case_idx,
        n_rand=n_rand,
        ped_speed_min=(ped_speed if ped_speed is not None else ped_speed_min),
        ped_speed_max=(ped_speed if ped_speed is not None else ped_speed_max),
        ped_radius=ped_radius,
    )

    _, campus_backup_defaults, campus_robot_defaults = _build_campus_model_defaults(
        model_key=model_key,
        controller_type=controller_type,
        vref_mode_occ=vref_mode_occ,
        oa_wmax=oa_wmax,
    )

    merged_backup_overrides = dict(campus_backup_defaults)
    if backup_cbf_overrides:
        merged_backup_overrides.update(dict(backup_cbf_overrides))

    merged_robot_overrides = dict(campus_robot_defaults)
    if robot_spec_overrides:
        merged_robot_overrides.update(dict(robot_spec_overrides))
    merged_robot_overrides["reached_threshold"] = float(goal_threshold)
    merged_robot_overrides["sensing_range"] = float(sensing_range)
    merged_robot_overrides.setdefault("v_adv_max_occ", float(hidden_obs_velocity))
    crowd_dyn_cfg = dict(merged_robot_overrides.get("crowd_dyn_obs", {}))
    crowd_dyn_cfg.setdefault("occluded_speed_boost_enable", True)
    crowd_dyn_cfg.setdefault("occluded_speed_boost_vmax", float(hidden_obs_velocity))
    crowd_dyn_cfg.setdefault("occluded_speed_boost_fov_only", True)
    crowd_dyn_cfg.setdefault("occluded_speed_boost_on_hysteresis_steps", 2)
    crowd_dyn_cfg.setdefault("occluded_speed_boost_off_hysteresis_steps", 5)
    crowd_dyn_cfg.setdefault("occluded_speed_boost_exact", True)
    merged_robot_overrides["crowd_dyn_obs"] = crowd_dyn_cfg

    with plt.rc_context({"figure.figsize": PLOT_FIGSIZE}):
        return crowd1.run_crowd_scenario(
            controller_type=controller_type,
            model_key=model_key,
            show_animation=show_animation,
            save_animation=save_animation,
            tf=tf,
            seed=seed,
            case_idx=case_idx,
            rand_obs=False,
            n_rand=int(n_rand),
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
            crowd_mode="random",
            backup_cbf_overrides=merged_backup_overrides,
            robot_spec_overrides=merged_robot_overrides,
            waypoints_override=WAYPOINTS,
            env_width_override=ENV_WIDTH,
            env_height_override=ENV_HEIGHT,
            known_obs_override=known_obs,
            obs_meta_override=obs_meta,
            scenario_diag_override=scenario_diag,
            scenario_name="Campus",
        )


def main():
    parser = argparse.ArgumentParser(description="Run campus sidewalk scenario with randomly placed pedestrians.")
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
    parser.add_argument("--tf", type=float, default=120.0, help="Simulation final time [s].")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for pedestrian heading initialization.")
    parser.add_argument("--idx", "--case-idx", dest="case_idx", type=int, default=None, help="Case index (1-based). Changes the random obstacle layout.")
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument("--save_ani", "--save-ani", "--save-anim", "--save-animation", dest="save_anim", type=crowd1._str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--n-rand", type=int, default=DEFAULT_N_RAND, help="Number of random campus pedestrians.")
    parser.add_argument(
        "--ped-speed",
        type=float,
        default=None,
        help="Fixed pedestrian speed override. If set, uses one speed for all pedestrians.",
    )
    parser.add_argument("--ped-speed-min", type=float, default=PEDESTRIAN_SPEED, help="Minimum random pedestrian speed.")
    parser.add_argument("--ped-speed-max", type=float, default=PEDESTRIAN_SPEED_MAX, help="Maximum random pedestrian speed.")
    parser.add_argument("--ped-radius", type=float, default=PEDESTRIAN_RADIUS, help="Random pedestrian radius.")
    parser.add_argument("--sensing-range", type=float, default=8.0, help="Robot sensing range override for this campus scenario.")
    parser.add_argument("--goal-threshold", type=float, default=GOAL_THRESHOLD, help="Goal reach threshold.")
    parser.add_argument("--robot-v-max", type=float, default=None, help="Override robot_spec.v_max.")
    parser.add_argument("--robot-v-min", type=float, default=None, help="Override robot_spec.v_min.")
    parser.add_argument("--robot-a-max", type=float, default=None, help="Override robot_spec.a_max.")
    parser.add_argument("--robot-w-max", type=float, default=None, help="Override robot_spec.w_max.")
    parser.add_argument("--robot-radius", type=float, default=None, help="Override robot_spec.radius.")
    parser.add_argument("--di-qp-weight-ax", type=float, default=None, help="Override robot_spec.qp_weight_ax.")
    parser.add_argument("--di-qp-weight-ay", type=float, default=None, help="Override robot_spec.qp_weight_ay.")
    parser.add_argument("--du-min-speed-scale", type=float, default=None, help="Override backup_cbf.min_speed_scale_occ_du.")
    parser.add_argument("--du-k-turn-brake", type=float, default=None, help="Override backup_cbf.k_turn_brake_occ_du.")
    parser.add_argument("--du-k-a-p", type=float, default=None, help="Override backup_cbf.k_a_occ_du_p and k_a_track_occ_du_p.")
    parser.add_argument("--du-k-a-d", type=float, default=None, help="Override backup_cbf.k_a_occ_du_d and k_a_track_occ_du_d.")
    parser.add_argument("--du-reverse-enter-cos", type=float, default=None, help="Override backup_cbf.reverse_enter_cos_occ_du.")
    parser.add_argument("--du-reverse-exit-cos", type=float, default=None, help="Override backup_cbf.reverse_exit_cos_occ_du.")
    parser.add_argument("--du-reverse-min-scale", type=float, default=None, help="Override backup_cbf.reverse_min_scale_occ_du.")
    parser.add_argument("--uni-reverse-bias", type=float, default=None, help="Override backup_cbf.reverse_bias_occ_uni.")
    parser.add_argument("--uni-reverse-gate-angle", type=float, default=None, help="Override backup_cbf.reverse_speed_gate_angle_occ_uni.")
    parser.add_argument("--uni-reverse-gate-power", type=float, default=None, help="Override backup_cbf.reverse_speed_gate_power_occ_uni.")
    parser.add_argument("--uni-v-min-cmd-rev", type=float, default=None, help="Override backup_cbf.v_min_cmd_rev_occ_uni.")
    parser.add_argument("--occ-dt-backup", type=float, default=None, help="Override backup_cbf.dt_backup.")
    parser.add_argument("--occ-terminal-slack-weight", type=float, default=None, help="Override backup_cbf.terminal_slack_weight for terminal-set rows only.")
    parser.add_argument("--occ-terminal-slack-max", type=float, default=None, help="Override backup_cbf.terminal_slack_max for terminal-set rows only.")
    parser.add_argument("--vref-mode-occ", type=str, choices=["soft", "strict"], default=None)
    parser.add_argument("--vref", type=str, choices=["default", "los"], default=None)
    parser.add_argument("--occ-visible-scale", type=float, default=DEFAULT_OCC_VISIBLE_SCALE)
    parser.add_argument(
        "--v-adv-max-occ",
        "--hidden-obs-velocity",
        dest="hidden_obs_velocity",
        type=float,
        default=DEFAULT_HIDDEN_OBS_VELOCITY,
        help="Hidden-obstacle speed bound used by occlusion reasoning.",
    )
    parser.add_argument("--occ-version", type=str, choices=["v1", "v2"], default=None)
    parser.add_argument(
        "--occ-enable-visible-hocbf",
        type=crowd1._str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Occlusion-CBF only: also add visible-obstacle CBF/HOCBF rows in the stacked QP.",
    )
    parser.add_argument("--oa-dynamic-occluders", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-allow-solver-fallback", type=crowd1._str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oa-dsafe", type=float, default=None)
    parser.add_argument("--oa-visible-reach-mode", type=str, choices=["worst_case", "constant_velocity"], default=None)
    parser.add_argument("--oa-use-nominal-tracking-cost", type=crowd1._str2bool, nargs="?", const=True, default=None)
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
        type=crowd1._str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override oacp_mpc.use_nominal_tracking_cost.",
    )
    parser.add_argument(
        "--oacp-allow-solver-fallback",
        type=crowd1._str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override oacp_mpc.allow_solver_fallback.",
    )
    parser.add_argument(
        "--oacp-dynamic-occluders",
        type=crowd1._str2bool,
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
    parser.add_argument("--wmax", type=str, choices=["default", "pi"], default="default")
    parser.add_argument("--oa-dt", type=float, default=None)
    args = parser.parse_args()

    pos_algo = resolve_baseline_alias(args.baseline, args.algo, CROWD_BASELINE_MAP)
    controller_type = {"pos": pos_algo}
    backup_cbf_overrides = _build_backup_cbf_overrides(args)
    robot_spec_overrides = _build_robot_spec_overrides(args)

    run_campus_scenario(
        controller_type=controller_type,
        model_key=args.model,
        show_animation=not args.disable_plot,
        save_animation=args.save_anim,
        tf=args.tf,
        seed=args.seed,
        case_idx=args.case_idx,
        n_rand=args.n_rand,
        ped_speed=args.ped_speed,
        ped_speed_min=args.ped_speed_min,
        ped_speed_max=args.ped_speed_max,
        ped_radius=args.ped_radius,
        sensing_range=args.sensing_range,
        goal_threshold=args.goal_threshold,
        vref_mode_occ=args.vref_mode_occ,
        vref_front_mode_occ=args.vref,
        occ_visible_scale=args.occ_visible_scale,
        hidden_obs_velocity=args.hidden_obs_velocity,
        occ_version=args.occ_version,
        occ_enable_visible_hocbf=args.occ_enable_visible_hocbf,
        oa_dynamic_occluders=args.oa_dynamic_occluders,
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        oa_dsafe=args.oa_dsafe,
        oa_visible_reach_mode=args.oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=args.oa_use_nominal_tracking_cost,
        oa_wmax=args.wmax,
        oa_dt=args.oa_dt,
        backup_cbf_overrides=backup_cbf_overrides,
        robot_spec_overrides=robot_spec_overrides,
    )


if __name__ == "__main__":
    main()
