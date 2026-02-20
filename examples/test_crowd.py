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


def run_crowd_scenario(
    controller_type=None,
    show_animation=True,
    save_animation=False,
    tf=300.0,
    seed=42,
    rand_obs=True,
    n_rand=50,
):
    if controller_type is None:
        controller_type = {"pos": "occlusion_cbf_qp"}

    dt = 0.05
    model = "DoubleIntegrator2D"

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

    # Random moving obstacles
    rand_rows, rand_meta = LocalTrackingControllerDyn_OCC.make_random_obstacles7(
        n_rand=int(n_rand),
        v_obs_max=0.5,
        x_range=(8.0, 30.0),
        y_spawn_range=(0.0, 15.0),
        r_range=(0.3, 0.4),
        y_bounds=(0.0, 15.0),
        seed=int(seed),
        rand_obs=bool(rand_obs),
    )
    if rand_rows.size > 0:
        type_column = np.ones((rand_rows.shape[0], 1))
        rand_rows_8col = np.hstack((rand_rows, type_column))
        known_obs = np.vstack([known_obs, rand_rows_8col])

    env_width = 24.0
    env_height = 15.0

    if model != "DoubleIntegrator2D":
        raise ValueError("Crowd scenario currently supports only DoubleIntegrator2D.")

    robot_spec = {
        "model": "DoubleIntegrator2D",
        "v_max": 1.0,
        "a_max": 1.0,
        "radius": 0.25,
        "debug_backup_qp": False,
        "sensing_range": 10.0,
        "backup_cbf": {"T_horizon": 2.0},
        "show_backup_rollout": True,
        "backup_rollout_every": 1,
        "use_occ": True,
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
        rand_seed=int(seed),
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
        choices=["di"],
        help="Robot model alias. Only `di` is supported for this scenario.",
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
    parser.add_argument("--n-rand", type=int, default=50, help="Number of random moving obstacles.")
    parser.add_argument("--no-rand-obs", action="store_true", help="Disable random moving obstacles.")
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument("--save-anim", action="store_true", help="Save animation frames/video (controller setting).")
    args = parser.parse_args()

    controller_type = {"pos": args.algo}
    run_crowd_scenario(
        controller_type=controller_type,
        show_animation=not args.disable_plot,
        save_animation=args.save_anim,
        tf=args.tf,
        seed=args.seed,
        rand_obs=(not args.no_rand_obs),
        n_rand=args.n_rand,
    )


if __name__ == "__main__":
    main()

