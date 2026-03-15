"""
Crowd scenario test migrated from dynamic_env/main.py::single_agent_main.

Run:
    uv run python examples/test_crowd.py --model di
"""

import argparse
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
):
    if controller_type is None:
        controller_type = {"pos": "occlusion_cbf_qp"}

    dt = 0.05
    mk = str(model_key).strip().lower()
    if mk in {"di", "doubleintegrator2d"}:
        model = "DoubleIntegrator2D"
    elif mk in {"du", "dynamicunicycle2d"}:
        model = "DynamicUnicycle2D"
    elif mk in {"uni", "unicycle2d", "un"}:
        model = "Unicycle2D"
    else:
        raise ValueError(f"Unsupported model `{model_key}`. Use `di`, `du`, or `uni`.")

    waypoints = np.array(
        [
            [1.0, 7.5, 0.0],
            [20.0, 7.5, 0.0],
        ],
        dtype=np.float64,
    )

    # Crowd scenario base obstacles: [x, y, r, type]
    known_obs = np.array(
        [
            [8.0, 7.5, 0.3, 1],  # dynamic type
        ],
        dtype=float,
    )

    # Convert to dynamic obstacle format:
    # [x, y, r, vx, vy, y_min, y_max, type]
    dynamic_obs = []
    for i, obs_info in enumerate(known_obs):
        ox, oy, r = float(obs_info[0]), float(obs_info[1]), float(obs_info[2])
        obs_type = int(obs_info[3]) if len(obs_info) >= 4 else 0
        if i % 2 == 1:
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
        v_obs_max=0.5,
        x_range=(8.0, 30.0),
        y_spawn_range=(0.0, 15.0),
        r_range=(0.3, 0.4),
        y_bounds=(0.0, 15.0),
        seed=case_seed,
        rand_obs=bool(rand_obs),
    )
    if rand_rows.size > 0:
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
    else:
        uni_backup_cfg = {"T_horizon": 1.5}
        if vref_mode_occ is not None:
            uni_backup_cfg["vref_mode_occ_uni"] = str(vref_mode_occ).strip().lower()
        robot_spec = {
            "model": "Unicycle2D",
            "v_max": 1.0,
            "w_max": 0.8,
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
    return tracking_controller.run_all_steps(tf=float(tf))


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
        choices=["occlusion_cbf_qp", "cbf_qp", "backup_cbf_qp"],
        help="Position controller algorithm.",
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

    controller_type = {"pos": args.algo}
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
    )


if __name__ == "__main__":
    main()
