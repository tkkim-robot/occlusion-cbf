"""
Crosswalk scenario test for the occlusion-aware CBF framework.

This script ports the `crosswalk_scenario_v3` setup into this repo and keeps
the scenario self-contained in this file.
"""

import argparse
import importlib
import importlib.util
import types
import sys
from pathlib import Path

import matplotlib.patches as patches
import numpy as np

# Ensure this repository root is imported first when running
# `python examples/test_crosswalk.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)


def _install_position_controller_shims():
    """
    Allow LocalTrackingControllerDyn_OCC to resolve controller modules from this
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
    Falls back to direct file import when namespace collisions exist.
    """
    try:
        mod = importlib.import_module("dynamic_env.main")
        cls = getattr(mod, "LocalTrackingControllerDyn_OCC", None)
        mod_file = Path(getattr(mod, "__file__", "")).resolve()
        if cls is not None and str(mod_file).startswith(repo_root_str):
            return cls
    except Exception:
        pass

    local_main = REPO_ROOT / "dynamic_env" / "main.py"
    spec = importlib.util.spec_from_file_location("dynamic_env_main_local_crosswalk", local_main)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local dynamic_env.main at {local_main}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "LocalTrackingControllerDyn_OCC")


LocalTrackingControllerDyn_OCC = _load_local_occ_controller()
from safe_control.utils import env, plotting

# =============================================================================
# Bus Types
# =============================================================================

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
    oa_paper_mode=None,
    oa_dynamic_occluders=None,
    oa_allow_solver_fallback=None,
    oa_dsafe=None,
    oa_visible_reach_mode=None,
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
    max_car_speed = 6.0
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

        set_size = 3
        set_gap_min, set_gap_max = 25.0, 25.1   # between sets
        intra_gap_min, intra_gap_max = 6.0, 6.01  # within one 3-car set
        # Randomize the very first visible set anchor so initial cars are
        # distributed differently inside the current plotting view (x in [0, 40]).
        # first_set_gap_min, first_set_gap_max = 15.0, 15.1
        first_set_gap_min, first_set_gap_max = 18.0, 18.1
        speed_drop_max = 1.2
        min_speed = 0.6 * max_car_speed
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

    def run_trial(bus_type, car_specs, trial_enable_plot, trial_save_animation, return_infeasible=False):
        known_obs = build_known_obs(bus_type, car_specs)

        # 2) Robot spec
        robot_spec = {
            "radius": 0.3,
            "sensing_range": 25.0,
            "fov_angle": 360,
            "occ_visible_scale": 0.5,
            "debug_backup_qp": False,
            "v_adv_max_occ": max_car_speed,
            "backup_cbf": {
                "T_horizon": 1.0,
                "dt_backup": 0.05,
                "alpha": 100.0,
                "rho_T": "stopping_distance",
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
        }
        if model_key in {"di", "doubleintegrator2d"}:
            robot_spec.update(
                {
                    "model": "DoubleIntegrator2D",
                    "v_max": 2.0,
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
        if str(controller_type.get("pos", "")).strip().lower() == "oa_mpc":
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
        bus_patch = None
        bus_label = None
        bus_occ_patch = None
        car_indices = []
        car_rects = []
        car_last_occ = None
        bus_min_x = bus_max_x = bus_min_y = bus_max_y = 0.0

        if trial_enable_plot:
            plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
            ax, fig = plot_handler.plot_grid("Single-Lane Crosswalk Scenario")

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

            bus_patch = patches.Rectangle(
                (bus_min_x, bus_min_y),
                bus_max_x - bus_min_x,
                bus_max_y - bus_min_y,
                edgecolor="black",
                facecolor="blue",
                fill=True,
                zorder=6,
            )
            ax.add_patch(bus_patch)
            bus_label = ax.text(
                bus_min_x + 0.5 * (bus_max_x - bus_min_x),
                bus_min_y + 0.5 * (bus_max_y - bus_min_y),
                "BUS",
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=7,
            )

            ax.plot([-200, 100], [14, 14], "k--", linewidth=1.5)

            car_indices = [
                idx for idx, row in enumerate(known_obs) if len(row) >= 8 and int(row[7]) == 2
            ]
            for idx in car_indices:
                r = float(known_obs[idx][2])
                car_width = 2.0 * r
                car_length = 4.0 * r
                rect = patches.Rectangle(
                    (0.0, 0.0),
                    car_length,
                    car_width,
                    edgecolor="black",
                    facecolor="gray",
                    fill=True,
                    zorder=5,
                )
                rect.set_visible(False)
                ax.add_patch(rect)
                car_rects.append(rect)
            car_last_occ = np.full(len(car_rects), -1, dtype=np.int8)

            if bus_type != 0:
                bus_occ_patch = patches.Polygon(
                    np.zeros((4, 2)),
                    closed=True,
                    fill=True,
                    facecolor="gray",
                    edgecolor="none",
                    alpha=0.25,
                    zorder=1,
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
                for rect_i, (rect, idx) in enumerate(zip(car_rects, car_indices)):
                    if idx >= len(self.obs) or not self.plot_dyn_obs:
                        rect.set_visible(False)
                        continue
                    obs_info = self.obs[idx]
                    ox, oy, r = obs_info[:3]
                    car_width = 2.0 * r
                    car_length = 4.0 * r
                    if abs(rect.get_width() - car_length) > 1e-9:
                        rect.set_width(car_length)
                    if abs(rect.get_height() - car_width) > 1e-9:
                        rect.set_height(car_width)
                    rect.set_xy((ox - 0.5 * car_length, oy - 0.5 * car_width))
                    is_occ = int(occluded_mask is not None and bool(occluded_mask[idx]))
                    if car_last_occ is None or car_last_occ[rect_i] != is_occ:
                        rect.set_facecolor("orange" if is_occ else "gray")
                        if car_last_occ is not None:
                            car_last_occ[rect_i] = is_occ
                    rect.set_visible(True)

                if bus_occ_patch is not None:
                    poly = compute_bus_occ_poly(float(self.robot.X[0, 0]), float(self.robot.X[1, 0]))
                    if poly is None:
                        bus_occ_patch.set_visible(False)
                    else:
                        bus_occ_patch.set_xy(poly)
                        bus_occ_patch.set_visible(True)

                if bus_patch is not None:
                    bus_patch.set_visible(True)
                if bus_label is not None:
                    bus_label.set_visible(True)

            tracking_controller.render_dyn_obs = types.MethodType(render_dyn_obs_with_bus, tracking_controller)

        result = tracking_controller.run_all_steps(tf=400)
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
        return run_trial(bus_type, car_specs, enable_plot, save_animation)

    if batch_eval:
        import contextlib
        import io

        rng = np.random.default_rng(seed)
        case_indices = []
        for idx in range(num_trials):
            car_specs = make_car_specs(rng)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result0, _ = run_trial(0, car_specs, trial_enable_plot=False, trial_save_animation=False, return_infeasible=True)
                result1, infeasible1 = run_trial(
                    1, car_specs, trial_enable_plot=False, trial_save_animation=False, return_infeasible=True
                )
            if result1 == -1 and (not infeasible1) and result0 == -2:
                case_indices.append(idx + 1)
                print(f"[Case {idx + 1}] type1 goal, type0 collision")
        print(f"Matched cases (goal vs collision): {case_indices}")
        return case_indices

    rng = np.random.default_rng(seed)
    car_specs = make_car_specs(rng)
    return run_trial(bus_type, car_specs, enable_plot, save_animation)


def main():
    parser = argparse.ArgumentParser(description="Run crosswalk_scenario_v3 in occlusion-cbf framework.")
    parser.add_argument("--model", default="di", help="Model alias: di | uni | du")
    parser.add_argument("--controller", default="occlusion_cbf_qp", help="Position controller type.")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["occlusion_cbf", "cbf_qp", "backup_cbf_qp", "oa_mpc"],
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
    args = parser.parse_args()

    model_key = str(args.model).strip().lower()
    if model_key not in {"di", "doubleintegrator2d", "uni", "unicycle2d", "du", "dynamicunicycle2d"}:
        raise ValueError(f"Unsupported model `{args.model}`. Use one of di/uni/du.")

    baseline_map = {
        "occlusion_cbf": "occlusion_cbf_qp",
        "cbf_qp": "cbf_qp",
        "backup_cbf_qp": "backup_cbf_qp",
        "oa_mpc": "oa_mpc",
    }
    pos_algo = baseline_map.get(args.baseline, args.controller)
    controller_type = {"pos": pos_algo}
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
        oa_paper_mode=args.oa_paper_mode,
        oa_dynamic_occluders=args.oa_dynamic_occluders,
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        oa_dsafe=args.oa_dsafe,
        oa_visible_reach_mode=args.oa_visible_reach_mode,
    )


if __name__ == "__main__":
    main()
