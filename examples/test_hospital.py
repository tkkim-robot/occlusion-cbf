"""
Hospital-like maze scenario for the occlusion-cbf framework.

Run:
    uv run python examples/test_hospital.py --model di
"""

import argparse

import numpy as np

from _baseline_defs import HOSPITAL_ALGO_CHOICES
from _runtime import ensure_repo_root, load_local_occ_controller

ensure_repo_root()
LocalTrackingControllerDyn_OCC = load_local_occ_controller("hospital")
from safe_control.utils import env, plotting


def _wall_segment_vertical(x, y_min, y_max, gap_ranges, r, spacing, env_h, obs_type=0):
    rows = []
    ys = np.arange(y_min, y_max + 1e-9, spacing)
    for y in ys:
        blocked = False
        for g0, g1 in gap_ranges:
            if g0 <= y <= g1:
                blocked = True
                break
        if blocked:
            continue
        rows.append([x, float(y), r, 0.0, 0.0, 0.0, env_h, float(obs_type)])
    return rows


def _wall_segment_horizontal(y, x_min, x_max, gap_ranges, r, spacing, env_h, obs_type=0):
    rows = []
    xs = np.arange(x_min, x_max + 1e-9, spacing)
    for x in xs:
        blocked = False
        for g0, g1 in gap_ranges:
            if g0 <= x <= g1:
                blocked = True
                break
        if blocked:
            continue
        rows.append([float(x), y, r, 0.0, 0.0, 0.0, env_h, float(obs_type)])
    return rows


def _build_hospital_layout(env_h):
    """
    Returns:
        static_walls: type=0 rows for physical collision constraints.
        occ_markers: sparse type=2 rows used only as occlusion sources (optional).
    """
    wall_r = 0.30
    spacing = 0.60

    static_walls = []
    static_walls += _wall_segment_vertical(
        x=9.0,
        y_min=1.0,
        y_max=19.0,
        gap_ranges=[(4.8, 6.2)],
        r=wall_r,
        spacing=spacing,
        env_h=env_h,
        obs_type=1,
    )
    static_walls += _wall_segment_vertical(
        x=17.0,
        y_min=1.0,
        y_max=19.0,
        gap_ranges=[(9.2, 10.8)],
        r=wall_r,
        spacing=spacing,
        env_h=env_h,
        obs_type=1,
    )
    static_walls += _wall_segment_vertical(
        x=25.0,
        y_min=1.0,
        y_max=19.0,
        gap_ranges=[(13.8, 15.2)],
        r=wall_r,
        spacing=spacing,
        env_h=env_h,
        obs_type=1,
    )
    static_walls += _wall_segment_horizontal(
        y=11.2,
        x_min=10.0,
        x_max=18.0,
        gap_ranges=[(14.0, 15.6)],
        r=wall_r,
        spacing=spacing,
        env_h=env_h,
        obs_type=1,
    )
    static_walls += _wall_segment_horizontal(
        y=6.8,
        x_min=18.0,
        x_max=26.0,
        gap_ranges=[(21.0, 22.4)],
        r=wall_r,
        spacing=spacing,
        env_h=env_h,
        obs_type=1,
    )

    static_walls_arr = np.array(static_walls, dtype=float)

    # Sparse occlusion marker points along walls (keeps physical walls on type=0).
    # type=2 points are used as occlusion sources when enabled.
    occ_markers = static_walls_arr[::4].copy()
    if occ_markers.size > 0:
        occ_markers[:, 2] = 0.26  # slightly smaller than physical wall circles
        occ_markers[:, 7] = 2.0
    return static_walls_arr, occ_markers


def _build_pedestrians(n_people, seed):
    rng = np.random.default_rng(seed)
    lanes = [
        (2.0, 8.2, 1.5, 8.5),
        (10.0, 16.2, 4.0, 12.0),
        (18.0, 24.0, 7.0, 15.0),
        (26.0, 31.5, 10.0, 18.5),
    ]

    rows = []
    meta = []
    for i in range(int(n_people)):
        lane = lanes[i % len(lanes)]
        x = float(rng.uniform(lane[0], lane[1]))
        y = float(rng.uniform(lane[2], lane[3]))
        r = float(rng.uniform(0.28, 0.36))
        speed = float(rng.uniform(0.35, 0.70))
        vy = speed if (i % 2 == 0) else -speed
        y_min = float(lane[2])
        y_max = float(lane[3])
        rows.append([x, y, r, 0.0, vy, y_min, y_max, 1.0])
        theta0 = np.pi / 2.0 if vy >= 0.0 else -np.pi / 2.0
        meta.append({"mode": 0, "v_max": speed, "theta": theta0})
    return np.array(rows, dtype=float), meta


def run_hospital_scenario(
    controller_type=None,
    show_animation=True,
    save_animation=False,
    tf=260.0,
    seed=42,
    n_people=12,
    wall_occlusion=True,
):
    if controller_type is None:
        controller_type = {"pos": "occlusion_cbf_qp"}

    dt = 0.05
    model = "DoubleIntegrator2D"
    env_width = 34.0
    env_height = 21.0

    # Zig-zag hospital hall route
    waypoints = np.array(
        [
            [2.0, 5.2, 0.0],
            [8.4, 5.6, 0.0],
            [13.4, 8.0, 0.0],
            [16.2, 10.0, 0.0],
            [20.5, 12.4, 0.0],
            [24.2, 14.6, 0.0],
            [30.5, 17.0, 0.0],
        ],
        dtype=np.float64,
    )

    if model != "DoubleIntegrator2D":
        raise ValueError("Hospital scenario currently supports only DoubleIntegrator2D.")

    static_walls, occ_markers = _build_hospital_layout(env_height)
    people_rows, people_meta = _build_pedestrians(n_people=n_people, seed=seed)

    if bool(wall_occlusion):
        known_obs = np.vstack((static_walls, occ_markers, people_rows))
    else:
        known_obs = np.vstack((static_walls, people_rows))

    robot_spec = {
        "model": "DoubleIntegrator2D",
        "v_max": 1.2,
        "a_max": 1.5,
        "radius": 0.25,
        "debug_backup_qp": False,
        "sensing_range": 12.0,
        "backup_cbf": {"T_horizon": 1.5},
        "show_backup_rollout": False,
        "backup_rollout_every": 1,
        "use_occ": True,
        "dynamic_obs_types": [1],
        "occlusion_types": [2] if bool(wall_occlusion) else [],
    }

    x_init = waypoints[0]

    if show_animation:
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
        ax, fig = plot_handler.plot_grid("Hospital Maze Scenario")
    else:
        ax = None
        fig = None

    env_handler = env.Env(width=env_width, height=env_height)

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
    n_constraints = int(min(80, max(20, known_obs.shape[0])))
    tracking_controller.num_constraints = n_constraints
    if hasattr(tracking_controller, "pos_controller") and hasattr(tracking_controller.pos_controller, "num_obs"):
        tracking_controller.pos_controller.num_obs = n_constraints

    static_count = static_walls.shape[0] + (occ_markers.shape[0] if bool(wall_occlusion) else 0)
    obs_meta = []
    for row in known_obs[:static_count]:
        obs_meta.append({"mode": 0, "v_max": 0.0, "theta": 0.0})
    obs_meta.extend(people_meta)
    tracking_controller.set_obs_meta(obs_meta)

    tracking_controller.set_waypoints(waypoints)
    return tracking_controller.run_all_steps(tf=float(tf))


def main():
    parser = argparse.ArgumentParser(description="Run hospital maze scenario with moving people.")
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
        choices=HOSPITAL_ALGO_CHOICES,
        help="Position controller algorithm.",
    )
    parser.add_argument("--tf", type=float, default=260.0, help="Simulation final time [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pedestrians.")
    parser.add_argument("--n-people", type=int, default=12, help="Number of moving pedestrians.")
    parser.add_argument(
        "--wall-occ",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: enable wall occlusion markers, 0: disable wall occlusion markers.",
    )
    parser.add_argument("--disable-plot", action="store_true", help="Disable animation plotting.")
    parser.add_argument("--save-anim", action="store_true", help="Save animation frames/video (controller setting).")
    args = parser.parse_args()

    controller_type = {"pos": args.algo}
    run_hospital_scenario(
        controller_type=controller_type,
        show_animation=not args.disable_plot,
        save_animation=args.save_anim,
        tf=args.tf,
        seed=args.seed,
        n_people=args.n_people,
        wall_occlusion=bool(args.wall_occ),
    )


if __name__ == "__main__":
    main()
