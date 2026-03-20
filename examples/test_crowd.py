"""
Crowd scenario test migrated from dynamic_env/main.py::single_agent_main.

Run:
    uv run python examples/test_crowd.py --model di
"""

import argparse
from collections import deque
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from safe_control.utils import env, plotting

# Ensure this repository root is imported first.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


def _install_position_controller_shims():
    """
    Allow LocalTrackingControllerDyn to resolve controller modules from this
    repo layout where `position_control/` may not include all baseline files.
    """
    shim_map = {
        "position_control.cbf_qp": "safe_control.position_control.cbf_qp",
        "position_control.backup_cbf_qp": "safe_control.position_control.backup_cbf_qp",
    }
    for dst_name, src_name in shim_map.items():
        if dst_name in sys.modules:
            continue
        try:
            sys.modules[dst_name] = importlib.import_module(src_name)
        except Exception:
            pass


_install_position_controller_shims()


def _load_local_occ_controller():
    """
    Load LocalTrackingControllerDyn_OCC from this repo's dynamic_env/main.py.
    Fallback to direct file import when namespace collisions exist.
    """
    try:
        mod = importlib.import_module("dynamic_env.main")
        cls = getattr(mod, "LocalTrackingControllerDyn_OCC", None)
        mod_file = Path(getattr(mod, "__file__", "")).resolve()
        if cls is not None and str(mod_file).startswith(REPO_ROOT_STR):
            return cls
    except Exception:
        pass

    local_main = REPO_ROOT / "dynamic_env" / "main.py"
    spec = importlib.util.spec_from_file_location("dynamic_env_main_local_crowd", local_main)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local dynamic_env.main at {local_main}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "LocalTrackingControllerDyn_OCC")


LocalTrackingControllerDyn_OCC = _load_local_occ_controller()


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
    """Apply finalized control_tree_mpc baseline defaults."""
    robot_spec["occlusion_types"] = [1]
    ct_cfg = robot_spec.setdefault("control_tree_mpc", {})
    ct_cfg.setdefault("dt_plan", 0.25)
    ct_cfg.setdefault("Th", 6.0)
    ct_cfg.setdefault("N", 24)
    ct_cfg.setdefault("n_branches", 3)
    ct_cfg.setdefault("gap_lookahead", 2.5)
    ct_cfg.setdefault("min_gap_width", 0.25)
    ct_cfg.setdefault("cluster_merge_distance", 0.8)
    ct_cfg.setdefault("forward_fov_deg_for_branching", 180.0)
    ct_cfg.setdefault("n_split", 12)
    ct_cfg.setdefault("wgoal", 3.5)
    ct_cfg.setdefault("wvel", 5.0)
    ct_cfg.setdefault("wacc", 1.8)
    ct_cfg.setdefault("wtrack", 2.0)
    ct_cfg.setdefault("lambda_w", 1.0)
    ct_cfg.setdefault("margin_obs", 0.05)
    ct_cfg.setdefault("backend", "sequential")
    ct_cfg.setdefault("solver_backend", "persistent_casadi")
    ct_cfg.setdefault("goal_handover_radius", 1.5)
    ct_cfg.setdefault("direct_goal_clearance_margin", 0.08)
    ct_cfg.setdefault("goal_handover_hysteresis", 0.2)
    ct_cfg.setdefault("near_goal_min_speed", 0.2)
    ct_cfg.setdefault("near_goal_speed_scale_radius", 1.5)
    ct_cfg.setdefault("near_goal_progress_window_steps", 30)
    ct_cfg.setdefault("near_goal_progress_min_drop", 0.03)
    ct_cfg.setdefault("near_goal_force_radius", 0.6)
    ct_cfg.setdefault("near_goal_mode_strategy", "strong_only")


def run_crowd_scenario(
    controller_type=None,
    model_key="di",
    show_animation=True,
    save_animation=False,
    tf=300.0,
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
    oa_paper_mode=None,
    oa_dynamic_occluders=None,
    oa_allow_solver_fallback=None,
    oa_dsafe=None,
    oa_visible_reach_mode=None,
    oa_use_nominal_tracking_cost=None,
    oa_paper_uni_preset=False,
    static_occluders=False,
    return_metrics=False,
    max_steps=None,
    max_sim_time=None,
    deadlock_window_steps=120,
    deadlock_progress_eps=0.05,
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

    use_oa_uni_preset = bool(oa_paper_uni_preset) and is_oa_mpc and model == "Unicycle2D"
    dt = 0.1 if use_oa_uni_preset else 0.05

    waypoints = np.array(
        [
            [1.0, 7.5, 0.0],
            [20.0, 7.5, 0.0],
        ],
        dtype=np.float64,
    )

    # Crowd scenario base obstacles: [x, y, r, type]
    base_obs_type = 0 if bool(static_occluders) else 1
    known_obs = np.array(
        [
            [8.0, 7.0, 0.3, base_obs_type],
        ],
        dtype=float,
    )

    # Convert to dynamic obstacle format:
    # [x, y, r, vx, vy, y_min, y_max, type]
    dynamic_obs = []
    for i, obs_info in enumerate(known_obs):
        ox, oy, r = float(obs_info[0]), float(obs_info[1]), float(obs_info[2])
        obs_type = int(obs_info[3]) if len(obs_info) >= 4 else 0
        if bool(static_occluders):
            vx, vy = 0.0, 0.0
        elif i % 2 == 1:
            vx, vy = -0.15, -0.15
        else:
            vx, vy = -0.15, 0.15
        y_min, y_max = 1.0, 14.0
        dynamic_obs.append([ox, oy, r, vx, vy, y_min, y_max, obs_type])
    known_obs = np.array(dynamic_obs, dtype=float)

    # Case-indexed reproducibility (1-based), similar to crosswalk scenario usage.
    # With fixed `seed`, changing `case_idx` deterministically selects a different
    # random crowd configuration.
    if case_idx is not None:
        if int(case_idx) < 1:
            raise ValueError("case_idx must be >= 1 (1-based).")
        rng_case = np.random.default_rng(int(seed))
        case_seed = int(seed)
        for _ in range(int(case_idx)):
            case_seed = int(rng_case.integers(0, 2**31 - 1))
    else:
        case_seed = int(seed)

    # Random moving obstacles
    rand_rows, rand_meta = LocalTrackingControllerDyn_OCC.make_random_obstacles7(
        n_rand=int(n_rand),
        v_obs_max=(0.0 if bool(static_occluders) else 0.5),
        x_range=(8.0, 30.0),
        y_spawn_range=(0.0, 15.0),
        r_range=(0.3, 0.4),
        y_bounds=(0.0, 15.0),
        seed=case_seed,
        rand_obs=bool(rand_obs),
    )
    if rand_rows.size > 0:
        if bool(static_occluders):
            rand_rows[:, 3] = 0.0
            rand_rows[:, 4] = 0.0
            rand_meta = [{"mode": 0, "v_max": 0.0, "theta": 0.0} for _ in range(rand_rows.shape[0])]
            type_column = np.zeros((rand_rows.shape[0], 1))
        else:
            type_column = np.ones((rand_rows.shape[0], 1))
        rand_rows_8col = np.hstack((rand_rows, type_column))
        known_obs = np.vstack([known_obs, rand_rows_8col])

    env_width = 24.0
    env_height = 15.0

    if model == "DoubleIntegrator2D":
        robot_spec = {
            "model": "DoubleIntegrator2D",
            "v_max": 1.0,
            "a_max": 1.0,
            "radius": 0.25,
            "debug_backup_qp": False,
            "sensing_range": 10.0,
            "backup_cbf": {"T_horizon": 1.5},
            "show_backup_rollout": True,
            "backup_rollout_every": 1,
            "use_occ": True,
            "dynamic_obs_types": [1],
        }
    elif model == "DynamicUnicycle2D":
        du_vmax = 1.0
        du_backup_cfg = {
            "T_horizon": 1.5,
        }
        if du_k_a_p is not None:
            du_backup_cfg["k_a_occ_du_p"] = float(du_k_a_p)
            du_backup_cfg["k_a_track_occ_du_p"] = float(du_k_a_p)
        if du_k_a_d is not None:
            du_backup_cfg["k_a_occ_du_d"] = float(du_k_a_d)
            du_backup_cfg["k_a_track_occ_du_d"] = float(du_k_a_d)
        if vref_mode_occ is not None:
            du_backup_cfg["vref_mode_occ_du"] = str(vref_mode_occ).strip().lower()
        robot_spec = {
            "model": "DynamicUnicycle2D",
            "v_max": du_vmax,
            "v_min": -du_vmax,
            "v_obs_max": 0.5,
            "a_max": 1.0,
            "w_max": 0.8,
            "radius": 0.25,
            "debug_backup_qp": False,
            "sensing_range": 10.0,
            "fov_angle": 360.0,
            "backup_cbf": du_backup_cfg,
            "show_backup_rollout": True,
            "backup_rollout_every": 1,
            "use_occ": True,
            "dynamic_obs_types": [1 ],
        }
    elif model == "Unicycle2D":
        uni_backup_cfg = {"T_horizon": 1.5}
        if vref_mode_occ is not None:
            uni_backup_cfg["vref_mode_occ_uni"] = str(vref_mode_occ).strip().lower()
        uni_vmax = 1.0 if use_oa_uni_preset else 1.0
        uni_wmax = float(np.pi) if use_oa_uni_preset else 0.8
        uni_radius = 0.2 if use_oa_uni_preset else 0.25
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

    if str(controller_type.get("pos", "")).strip().lower() == "oa_mpc":
        oa_cfg = robot_spec.setdefault("oa_mpc", {})
        oa_cfg.setdefault("paper_mode", True)
        oa_cfg.setdefault("N", 10)
        oa_cfg.setdefault("visible_reach_mode", "worst_case")
        oa_cfg.setdefault("use_nominal_tracking_cost", False)
        oa_cfg.setdefault("dynamic_occluders", False)
        if use_oa_uni_preset:
            # Paper-like OA-MPC preset for unicycle baseline reproduction.
            oa_cfg["paper_mode"] = True
            oa_cfg["N"] = 10
            oa_cfg["auto_scale_N_with_dt"] = False
            oa_cfg["paper_horizon_time"] = 1.0
            oa_cfg.setdefault("dsafe", 0.5)
        # OA paper baseline: static occluders only.
        robot_spec.setdefault("occlusion_types", [0])
        if oa_paper_mode is not None:
            oa_cfg["paper_mode"] = bool(oa_paper_mode)
        if oa_dynamic_occluders is not None:
            oa_cfg["dynamic_occluders"] = bool(oa_dynamic_occluders)
            if bool(oa_dynamic_occluders):
                robot_spec["occlusion_types"] = [0, 1]
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

    x_init = waypoints[0]

    if show_animation:
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
        ax, fig = plot_handler.plot_grid("Crowd Scenario")
    else:
        ax = None
        fig = None

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
    )

    tracking_controller.obs = known_obs.astype(float)
    n_const = known_obs.shape[0] - rand_rows.shape[0]
    const_meta = []
    for row in known_obs[:n_const]:
        vx, vy = float(row[3]), float(row[4])
        vmag = float(np.hypot(vx, vy))
        theta0 = float(np.arctan2(vy, vx)) if vmag > 1e-9 else 0.0
        const_meta.append({"mode": 0, "v_max": vmag, "theta": theta0})
    meta = const_meta + rand_meta
    tracking_controller.set_obs_meta(meta)

    tracking_controller.set_waypoints(waypoints)
    if not bool(return_metrics):
        return tracking_controller.run_all_steps(tf=float(tf))

    tf_cap = float(max_sim_time) if max_sim_time is not None else float(tf)
    n_steps = int(np.ceil(tf_cap / dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    final_goal = np.asarray(waypoints[-1], dtype=float).reshape(-1)[:2]
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
    total_steps = 0

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
                try:
                    selected_branch_vals.append(int(sb))
                except Exception:
                    pass
        if step_ms is not None and np.isfinite(step_ms):
            compute_ms.append(float(step_ms))

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
    selected_branch_counts = {}
    for sb in selected_branch_vals:
        selected_branch_counts[str(int(sb))] = int(selected_branch_counts.get(str(int(sb)), 0) + 1)
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
        "selected_branch_counts": selected_branch_counts,
        "deadlock_detected": bool(deadlock_detected),
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
        choices=[
            "occlusion_cbf_qp",
            "cbf_qp",
            "backup_cbf_qp",
            "oa_mpc",
            "single_risk_mpc",
            "control_tree_mpc",
        ],
        help="Position controller algorithm.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=[
            "occlusion_cbf",
            "cbf_qp",
            "backup_cbf_qp",
            "oa_mpc",
            "single_risk_mpc",
            "control_tree_mpc",
        ],
        help="Baseline alias. If provided, overrides --algo.",
    )
    parser.add_argument("--tf", type=float, default=300.0, help="Simulation final time [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for crowd generation.")
    parser.add_argument(
        "--idx",
        "--case-idx",
        dest="case_idx",
        type=int,
        default=None,
        help="Case index (1-based) for deterministic random scenario selection with fixed seed.",
    )
    parser.add_argument("--n-rand", type=int, default=50, help="Number of random moving obstacles.")
    parser.add_argument("--no-rand-obs", action="store_true", help="Disable random moving obstacles.")
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument("--du-min-speed-scale", type=float, default=None, help="Override backup_cbf.min_speed_scale_occ_du.")
    parser.add_argument("--du-k-turn-brake", type=float, default=None, help="Override backup_cbf.k_turn_brake_occ_du.")
    parser.add_argument("--du-k-a-p", type=float, default=None, help="Override backup_cbf.k_a_occ_du_p.")
    parser.add_argument("--du-k-a-d", type=float, default=None, help="Override backup_cbf.k_a_occ_du_d.")
    parser.add_argument("--du-reverse-enter-cos", type=float, default=None, help="Override backup_cbf.reverse_enter_cos_occ_du.")
    parser.add_argument("--du-reverse-exit-cos", type=float, default=None, help="Override backup_cbf.reverse_exit_cos_occ_du.")
    parser.add_argument("--du-reverse-min-scale", type=float, default=None, help="Override backup_cbf.reverse_min_scale_occ_du.")
    parser.add_argument(
        "--vref-mode-occ",
        type=str,
        choices=["soft", "strict"],
        default=None,
        help="Facet aggregation mode for UNI/DU occlusion backup v_ref.",
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
        "--oa-use-nominal-tracking-cost",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="OA-MPC extension: use ||u-u_ref|| cost term.",
    )
    parser.add_argument(
        "--oa-paper-uni-preset",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "OA-MPC unicycle paper-like preset (one-line reproduction): "
            "dt=0.1, v_max=2.0, w_max=pi, radius=0.2, N=10, dsafe=0.5."
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

    baseline_map = {
        "occlusion_cbf": "occlusion_cbf_qp",
        "cbf_qp": "cbf_qp",
        "backup_cbf_qp": "backup_cbf_qp",
        "oa_mpc": "oa_mpc",
        "single_risk_mpc": "single_risk_mpc",
        "control_tree_mpc": "control_tree_mpc",
    }
    pos_algo = baseline_map.get(args.baseline, args.algo)
    controller_type = {"pos": pos_algo}
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
        oa_paper_mode=args.oa_paper_mode,
        oa_dynamic_occluders=args.oa_dynamic_occluders,
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        oa_dsafe=args.oa_dsafe,
        oa_visible_reach_mode=args.oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=args.oa_use_nominal_tracking_cost,
        oa_paper_uni_preset=args.oa_paper_uni_preset,
        static_occluders=args.static_occluders,
    )


if __name__ == "__main__":
    main()
