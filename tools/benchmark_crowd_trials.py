#!/usr/bin/env python3
"""
Run crowd random-trial sweeps and print compact 3-way benchmark metrics.

Single-baseline mode:
    - one baseline over idx range

Suite mode:
    - sequentially run the 5 non-occlusion baselines:
      OA-MPC(wmax=default), OA-MPC(wmax=pi), control_tree_mpc,
      single_risk_mpc, cbf_qp

All runs force:
    - show_animation = False
    - save_animation = False
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from examples._baseline_defs import CROWD_BASELINE_MAP
BASELINE_MAP = dict(CROWD_BASELINE_MAP)

SUITE_NON_OCC_5 = [
    {"label": "oa_mpc_wmax_default", "baseline": "oa_mpc", "wmax": "default"},
    {"label": "oa_mpc_wmax_pi", "baseline": "oa_mpc", "wmax": "pi"},
    {"label": "control_tree_mpc", "baseline": "control_tree_mpc", "wmax": "default"},
    {"label": "single_risk_mpc", "baseline": "single_risk_mpc", "wmax": "default"},
    {"label": "cbf_qp", "baseline": "cbf_qp", "wmax": "default"},
]


def _load_run_crowd_scenario(scenario_name: str):
    scenario_key = str(scenario_name).strip().lower()
    if scenario_key in {"crowd1", "crowd", "test_crowd"}:
        from examples.test_crowd import run_crowd_scenario as runner

        return runner, "crowd1"
    if scenario_key in {"crowd2", "test_crowd2"}:
        from examples.test_crowd2 import run_crowd_scenario as runner

        return runner, "crowd2"
    raise ValueError(f"Unsupported scenario: {scenario_name}")


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _parse_json_arg(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    candidate = Path(s)
    if candidate.exists():
        with candidate.open("r") as f:
            data = json.load(f)
    else:
        data = json.loads(s)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("--backup-cbf-json must decode to a JSON object")
    return data


def _classify_3way(raw_outcome: str) -> str:
    o = str(raw_outcome).strip().lower()
    if o == "success":
        return "success"
    if o == "collision":
        return "collision"
    return "infeasible"


def _mean(vals: list[float | None]) -> float | None:
    vv = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vv:
        return None
    return float(np.mean(vv))


def _run_baseline_sweep(
    *,
    scenario_name: str,
    baseline_alias: str,
    run_label: str,
    model: str,
    seed: int,
    idx_start: int,
    idx_end: int,
    n_rand: int,
    tf: float,
    wmax: str,
    oa_allow_solver_fallback: bool | None,
    occ_version: str | None,
    occ_visible_scale: float | None,
    occ_enable_visible_hocbf: bool | None,
    crowd_mode: str,
    forced_events: int,
    forced_bg_rand: int | None,
    forced_hidden_speed: float,
    forced_occluder_radius_min: float,
    forced_occluder_radius_max: float,
    forced_validate_occlusion: bool,
    forced_require_corridor_conflict: bool,
    backup_cbf_overrides: dict[str, Any] | None,
    robot_spec_overrides: dict[str, Any] | None,
    workers: int,
    out_dir: Path,
    ts: str,
) -> dict[str, Any]:
    _, scenario_label = _load_run_crowd_scenario(scenario_name)
    controller_pos = BASELINE_MAP[str(baseline_alias)]
    stem = f"crowd_trials_{run_label}_{seed}_{idx_start}_{idx_end}_{ts}"
    rows_path = out_dir / f"{stem}.csv"
    summary_path = out_dir / f"{stem}.json"

    rows: list[dict[str, Any]] = []
    idx_success: list[int] = []
    idx_collision: list[int] = []
    idx_infeasible: list[int] = []

    print(
        f"[RUN] scenario={scenario_label} label={run_label} baseline={baseline_alias} pos={controller_pos} model={model} "
        f"seed={seed} idx={idx_start}..{idx_end} n_rand={n_rand} tf={tf} wmax={wmax} "
        f"oa_allow_solver_fallback={oa_allow_solver_fallback} "
        f"occ_version={occ_version} occ_visible_scale={occ_visible_scale} "
        f"occ_enable_visible_hocbf={occ_enable_visible_hocbf} "
        f"crowd_mode={crowd_mode}",
        flush=True,
    )
    print("[RUN] show_animation=False, save_animation=False", flush=True)

    idx_list = list(range(int(idx_start), int(idx_end) + 1))
    total = len(idx_list)

    if int(workers) <= 1:
        for k, idx in enumerate(idx_list, start=1):
            row = _run_one_idx_job(
                baseline_alias=str(baseline_alias),
                scenario_name=str(scenario_label),
                controller_pos=str(controller_pos),
                model=str(model),
                seed=int(seed),
                idx=int(idx),
                n_rand=int(n_rand),
                tf=float(tf),
                wmax=str(wmax),
                oa_allow_solver_fallback=oa_allow_solver_fallback,
                occ_version=occ_version,
                occ_visible_scale=occ_visible_scale,
                occ_enable_visible_hocbf=occ_enable_visible_hocbf,
                crowd_mode=str(crowd_mode),
                forced_events=int(forced_events),
                forced_bg_rand=forced_bg_rand,
                forced_hidden_speed=float(forced_hidden_speed),
                forced_occluder_radius_min=float(forced_occluder_radius_min),
                forced_occluder_radius_max=float(forced_occluder_radius_max),
                forced_validate_occlusion=bool(forced_validate_occlusion),
                forced_require_corridor_conflict=bool(forced_require_corridor_conflict),
                backup_cbf_overrides=backup_cbf_overrides,
                robot_spec_overrides=robot_spec_overrides,
                run_label=str(run_label),
            )
            rows.append(row)
            print(
                f"[{k:03d}/{total:03d}] idx={idx:3d} -> {row['class_3way']:10s} "
                f"(raw={row['raw_outcome']}, solve_ms={row['avg_solve_time_ms']}, "
                f"intv={row['avg_control_intervention_l2_sq']})",
                flush=True,
            )
    else:
        tasks = [
            (
                str(baseline_alias),
                str(scenario_label),
                str(controller_pos),
                str(model),
                int(seed),
                int(idx),
                int(n_rand),
                float(tf),
                str(wmax),
                oa_allow_solver_fallback,
                occ_version,
                occ_visible_scale,
                occ_enable_visible_hocbf,
                str(crowd_mode),
                int(forced_events),
                forced_bg_rand,
                float(forced_hidden_speed),
                float(forced_occluder_radius_min),
                float(forced_occluder_radius_max),
                bool(forced_validate_occlusion),
                bool(forced_require_corridor_conflict),
                backup_cbf_overrides,
                robot_spec_overrides,
                str(run_label),
            )
            for idx in idx_list
        ]
        with cf.ProcessPoolExecutor(max_workers=int(workers)) as ex:
            fut_map = {ex.submit(_run_one_idx_job_star, task): task[5] for task in tasks}
            for k, fut in enumerate(cf.as_completed(fut_map), start=1):
                idx = int(fut_map[fut])
                try:
                    row = fut.result()
                except Exception as ex_err:
                    row = {
                        "label": str(run_label),
                        "scenario": str(scenario_label),
                        "baseline": str(baseline_alias),
                        "controller_pos": str(controller_pos),
                        "wmax": str(wmax),
                        "idx": int(idx),
                        "occ_version": occ_version,
                        "occ_visible_scale": occ_visible_scale,
                        "occ_enable_visible_hocbf": occ_enable_visible_hocbf,
                        "raw_outcome": "exception",
                        "class_3way": "infeasible",
                        "crowd_mode": str(crowd_mode),
                        "n_forced_events": 0,
                        "n_background_rand": 0,
                        "n_forced_initially_occluded": 0,
                        "n_forced_revealed": 0,
                        "n_forced_corridor_conflict": 0,
                        "min_reveal_distance_to_ego_path": None,
                        "min_reveal_ttc_to_nominal_ego": None,
                        "reveal_steps": "[]",
                        "forced_event_meta": "[]",
                        "avg_solve_time_ms": None,
                        "avg_control_intervention_l2_sq": None,
                        "total_sim_time": None,
                        "case_wall_time_s": None,
                        "total_steps": 0,
                        "final_goal_distance": None,
                        "status": "failure",
                        "ret": -99,
                        "exception": str(ex_err),
                    }
                rows.append(row)
                print(
                    f"[{k:03d}/{total:03d}] idx={idx:3d} -> {row['class_3way']:10s} "
                    f"(raw={row['raw_outcome']}, solve_ms={row['avg_solve_time_ms']}, "
                    f"intv={row['avg_control_intervention_l2_sq']})",
                    flush=True,
                )

    rows.sort(key=lambda r: int(r["idx"]))
    for row in rows:
        cls = str(row["class_3way"])
        idx = int(row["idx"])
        if cls == "success":
            idx_success.append(idx)
        elif cls == "collision":
            idx_collision.append(idx)
        else:
            idx_infeasible.append(idx)

    avg_ms_all = _mean([_safe_float(r.get("avg_solve_time_ms")) for r in rows])
    avg_intv_all = _mean([_safe_float(r.get("avg_control_intervention_l2_sq")) for r in rows])
    avg_sim_time_all = _mean([_safe_float(r.get("total_sim_time")) for r in rows])
    avg_wall_time_all = _mean([_safe_float(r.get("case_wall_time_s")) for r in rows])
    avg_forced_init_occ = _mean([_safe_float(r.get("n_forced_initially_occluded")) for r in rows])
    avg_forced_revealed = _mean([_safe_float(r.get("n_forced_revealed")) for r in rows])
    avg_forced_conflict = _mean([_safe_float(r.get("n_forced_corridor_conflict")) for r in rows])
    reveal_positive_rate = _mean(
        [1.0 if int(r.get("n_forced_revealed", 0) or 0) > 0 else 0.0 for r in rows]
    )

    fields = [
        "label",
        "scenario",
        "baseline",
        "controller_pos",
        "wmax",
        "oa_allow_solver_fallback",
        "occ_version",
        "occ_visible_scale",
        "occ_enable_visible_hocbf",
        "crowd_mode",
        "idx",
        "raw_outcome",
        "class_3way",
        "n_forced_events",
        "n_background_rand",
        "n_forced_initially_occluded",
        "n_forced_revealed",
        "n_forced_corridor_conflict",
        "min_reveal_distance_to_ego_path",
        "min_reveal_ttc_to_nominal_ego",
        "reveal_steps",
        "forced_event_meta",
        "avg_solve_time_ms",
        "avg_control_intervention_l2_sq",
        "total_sim_time",
        "case_wall_time_s",
        "total_steps",
        "final_goal_distance",
        "status",
        "ret",
        "exception",
    ]
    with rows_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "config": {
            "scenario": str(scenario_label),
            "label": str(run_label),
            "baseline": str(baseline_alias),
            "controller_pos": controller_pos,
            "model": str(model),
            "seed": int(seed),
            "idx_start": int(idx_start),
            "idx_end": int(idx_end),
            "n_rand": int(n_rand),
            "tf": float(tf),
            "oa_wmax": str(wmax),
            "oa_allow_solver_fallback": oa_allow_solver_fallback,
            "occ_version": occ_version,
            "occ_visible_scale": occ_visible_scale,
            "occ_enable_visible_hocbf": occ_enable_visible_hocbf,
            "crowd_mode": str(crowd_mode),
            "forced_events": int(forced_events),
            "forced_bg_rand": forced_bg_rand,
            "forced_hidden_speed": float(forced_hidden_speed),
            "forced_occluder_radius_min": float(forced_occluder_radius_min),
            "forced_occluder_radius_max": float(forced_occluder_radius_max),
            "forced_validate_occlusion": bool(forced_validate_occlusion),
            "forced_require_corridor_conflict": bool(forced_require_corridor_conflict),
            "backup_cbf_overrides": backup_cbf_overrides,
            "robot_spec_overrides": robot_spec_overrides,
            "classification": {
                "success": "raw_outcome == success",
                "collision": "raw_outcome == collision",
                "infeasible": "all remaining raw outcomes (infeasible/timeout/deadlock/exception/unknown)",
            },
        },
        "counts": {
            "success": int(len(idx_success)),
            "collision": int(len(idx_collision)),
            "infeasible": int(len(idx_infeasible)),
            "total": int(len(rows)),
        },
        "idx_lists": {
            "success": idx_success,
            "collision": idx_collision,
            "infeasible": idx_infeasible,
        },
        "averages": {
            "avg_solve_time_ms_over_idx": avg_ms_all,
            "avg_control_intervention_l2_sq_over_idx": avg_intv_all,
            "avg_total_sim_time_s_over_idx": avg_sim_time_all,
            "avg_case_wall_time_s_over_idx": avg_wall_time_all,
            "avg_n_forced_initially_occluded_over_idx": avg_forced_init_occ,
            "avg_n_forced_revealed_over_idx": avg_forced_revealed,
            "avg_n_forced_corridor_conflict_over_idx": avg_forced_conflict,
            "forced_reveal_positive_trial_rate": reveal_positive_rate,
        },
        "artifacts": {
            "rows_csv": str(rows_path),
            "summary_json": str(summary_path),
        },
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Crowd Trial Summary (3-way) ===")
    print(
        f"success={len(idx_success)} | collision={len(idx_collision)} | "
        f"infeasible={len(idx_infeasible)} | total={len(rows)}"
    )
    print(f"success idx   : {idx_success}")
    print(f"collision idx : {idx_collision}")
    print(f"infeasible idx: {idx_infeasible}")
    print(f"avg solve time over idx [ms]: {avg_ms_all}")
    print(f"avg control intervention over idx: {avg_intv_all}")
    print(f"avg sim time over idx [s]: {avg_sim_time_all}")
    print(f"avg wall time per idx [s]: {avg_wall_time_all}")
    print(f"avg forced initially occluded over idx: {avg_forced_init_occ}")
    print(f"avg forced revealed over idx: {avg_forced_revealed}")
    print(f"avg forced corridor conflict over idx: {avg_forced_conflict}")
    print(f"forced reveal-positive trial rate: {reveal_positive_rate}")
    print(f"saved rows    : {rows_path}")
    print(f"saved summary : {summary_path}")
    print()

    return summary


def _run_non_occ_5_suite(args: argparse.Namespace, out_dir: Path, ts: str) -> int:
    print("=== Running baseline suite: non_occlusion_5 ===")
    print(
        "Methods: oa_mpc(wmax=default), oa_mpc(wmax=pi), "
        "control_tree_mpc, single_risk_mpc, cbf_qp"
    )
    print()

    results: list[dict[str, Any]] = []
    n_methods = len(SUITE_NON_OCC_5)
    for i, spec in enumerate(SUITE_NON_OCC_5, start=1):
        print(f"{'#' * 20} [{i}/{n_methods}] {spec['label']} {'#' * 20}")
        res = _run_baseline_sweep(
            scenario_name=str(args.scenario),
            baseline_alias=str(spec["baseline"]),
            run_label=str(spec["label"]),
            model=str(args.model),
            seed=int(args.seed),
            idx_start=int(args.idx_start),
            idx_end=int(args.idx_end),
            n_rand=int(args.n_rand),
            tf=float(args.tf),
            wmax=str(spec["wmax"]),
            oa_allow_solver_fallback=args.oa_allow_solver_fallback,
            occ_version=args.occ_version,
            occ_visible_scale=args.occ_visible_scale,
            occ_enable_visible_hocbf=args.occ_enable_visible_hocbf,
            crowd_mode=str(args.crowd_mode),
            forced_events=int(args.forced_events),
            forced_bg_rand=args.forced_bg_rand,
            forced_hidden_speed=float(args.forced_hidden_speed),
            forced_occluder_radius_min=float(args.forced_occluder_radius_min),
            forced_occluder_radius_max=float(args.forced_occluder_radius_max),
            forced_validate_occlusion=bool(args.forced_validate_occlusion),
            forced_require_corridor_conflict=bool(args.forced_require_corridor_conflict),
            backup_cbf_overrides=args.backup_cbf_overrides,
            robot_spec_overrides=args.robot_spec_overrides,
            workers=int(args.workers),
            out_dir=out_dir,
            ts=ts,
        )
        results.append(res)

    agg_rows_path = out_dir / f"crowd_trials_non_occlusion_5_summary_{ts}.csv"
    agg_json_path = out_dir / f"crowd_trials_non_occlusion_5_summary_{ts}.json"

    agg_fields = [
        "scenario",
        "label",
        "baseline",
        "wmax",
        "success",
        "collision",
        "infeasible",
        "total",
        "avg_solve_time_ms_over_idx",
        "avg_control_intervention_l2_sq_over_idx",
        "avg_total_sim_time_s_over_idx",
        "avg_case_wall_time_s_over_idx",
        "summary_json",
        "rows_csv",
    ]
    agg_rows: list[dict[str, Any]] = []
    for r in results:
        cfg = r.get("config", {})
        cnt = r.get("counts", {})
        avg = r.get("averages", {})
        art = r.get("artifacts", {})
        agg_rows.append(
            {
                "scenario": cfg.get("scenario"),
                "label": cfg.get("label"),
                "baseline": cfg.get("baseline"),
                "wmax": cfg.get("oa_wmax"),
                "success": cnt.get("success"),
                "collision": cnt.get("collision"),
                "infeasible": cnt.get("infeasible"),
                "total": cnt.get("total"),
                "avg_solve_time_ms_over_idx": avg.get("avg_solve_time_ms_over_idx"),
                "avg_control_intervention_l2_sq_over_idx": avg.get("avg_control_intervention_l2_sq_over_idx"),
                "avg_total_sim_time_s_over_idx": avg.get("avg_total_sim_time_s_over_idx"),
                "avg_case_wall_time_s_over_idx": avg.get("avg_case_wall_time_s_over_idx"),
                "summary_json": art.get("summary_json"),
                "rows_csv": art.get("rows_csv"),
            }
        )

    with agg_rows_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_rows)

    suite_report = {
        "config": {
            "suite": "non_occlusion_5",
            "scenario": str(args.scenario),
            "model": str(args.model),
            "seed": int(args.seed),
            "idx_start": int(args.idx_start),
            "idx_end": int(args.idx_end),
            "n_rand": int(args.n_rand),
            "tf": float(args.tf),
            "oa_allow_solver_fallback": args.oa_allow_solver_fallback,
            "occ_version": args.occ_version,
            "occ_visible_scale": args.occ_visible_scale,
            "occ_enable_visible_hocbf": args.occ_enable_visible_hocbf,
            "crowd_mode": str(args.crowd_mode),
            "forced_events": int(args.forced_events),
            "forced_bg_rand": args.forced_bg_rand,
            "forced_hidden_speed": float(args.forced_hidden_speed),
            "forced_occluder_radius_min": float(args.forced_occluder_radius_min),
            "forced_occluder_radius_max": float(args.forced_occluder_radius_max),
            "forced_validate_occlusion": bool(args.forced_validate_occlusion),
            "forced_require_corridor_conflict": bool(args.forced_require_corridor_conflict),
            "backup_cbf_overrides": args.backup_cbf_overrides,
            "robot_spec_overrides": args.robot_spec_overrides,
        },
        "rows": agg_rows,
    }
    with agg_json_path.open("w") as f:
        json.dump(suite_report, f, indent=2)

    print("=== Suite Summary (non_occlusion_5) ===")
    for row in agg_rows:
        print(
            f"{row['label']:24s} | success={row['success']} collision={row['collision']} "
            f"infeasible={row['infeasible']} total={row['total']} "
            f"| avg_ms={row['avg_solve_time_ms_over_idx']} "
            f"| avg_intv={row['avg_control_intervention_l2_sq_over_idx']} "
            f"| avg_sim_s={row['avg_total_sim_time_s_over_idx']} "
            f"| avg_wall_s={row['avg_case_wall_time_s_over_idx']}"
        )
    print(f"saved suite csv : {agg_rows_path}")
    print(f"saved suite json: {agg_json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run crowd idx sweep for one baseline or a predefined baseline suite."
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="crowd1",
        choices=["crowd1", "crowd2"],
        help="Select which benchmark scenario runner to use: examples/test_crowd.py or examples/test_crowd2.py.",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=list(BASELINE_MAP.keys()),
        help="Single-baseline mode only. Same aliases as examples/test_crowd.py",
    )
    p.add_argument(
        "--baseline-suite",
        type=str,
        default="none",
        choices=["none", "non_occlusion_5"],
        help="Run predefined baseline suite sequentially.",
    )
    p.add_argument("--model", type=str, default="uni", choices=["di", "du", "uni"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--idx-start", type=int, default=1)
    p.add_argument("--idx-end", type=int, default=100)
    p.add_argument(
        "--n-rand",
        type=int,
        default=50,
        help=(
            "Random mode: number of moving obstacles. Forced-emergence mode: target number of non-occluder movers "
            "(hidden-emergence agents plus optional background clutter). Large forced occluders are controlled by "
            "--forced-events and are not counted in n-rand."
        ),
    )
    p.add_argument("--tf", type=float, default=100.0)
    p.add_argument(
        "--crowd-mode",
        type=str,
        default="random",
        choices=["random", "forced_emergence"],
        help="Forwarded to examples/test_crowd.py crowd generator.",
    )
    p.add_argument("--forced-events", type=int, default=3)
    p.add_argument(
        "--forced-bg-rand",
        type=int,
        default=None,
        help=(
            "Forced-emergence mode: explicit number of background random obstacles. "
            "Default uses a small clutter share and allocates the rest of n-rand to hidden emergence agents."
        ),
    )
    p.add_argument("--forced-hidden-speed", type=float, default=0.5)
    p.add_argument("--forced-occluder-radius-min", type=float, default=0.8)
    p.add_argument("--forced-occluder-radius-max", type=float, default=1.0)
    p.add_argument(
        "--forced-validate-occlusion",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    p.add_argument(
        "--forced-require-corridor-conflict",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes per baseline (1 = sequential).",
    )
    p.add_argument(
        "--wmax",
        type=str,
        default="default",
        choices=["default", "pi"],
        help="Single-baseline mode only; used when --baseline oa_mpc.",
    )
    p.add_argument(
        "--oa-allow-solver-fallback",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Forwarded to test_crowd OA-MPC config. If false, OA solver failures are treated as infeasible.",
    )
    p.add_argument(
        "--occ-version",
        type=str,
        default=None,
        choices=["v1", "v2"],
        help="Forwarded to examples/test_crowd.py occlusion polygon variant.",
    )
    p.add_argument(
        "--occ-visible-scale",
        type=float,
        default=None,
        help="Forwarded to examples/test_crowd.py visibility fraction threshold for visible-obs filtering.",
    )
    p.add_argument(
        "--occ-enable-visible-hocbf",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Forwarded to examples/test_crowd.py: enable visible-obstacle CBF/HOCBF rows inside occlusion-CBF.",
    )
    p.add_argument("--oacp-dt-plan", type=float, default=None)
    p.add_argument("--oacp-Th", type=float, default=None)
    p.add_argument("--oacp-N", type=int, default=None)
    p.add_argument("--oacp-n-shared", type=int, default=None)
    p.add_argument("--oacp-risk-explore-scale", type=float, default=None)
    p.add_argument("--oacp-risk-fallback-scale", type=float, default=None)
    p.add_argument("--oacp-explore-speed-scale", type=float, default=None)
    p.add_argument("--oacp-fallback-speed-scale", type=float, default=None)
    p.add_argument(
        "--oacp-use-nominal-tracking-cost",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
    )
    p.add_argument(
        "--oacp-allow-solver-fallback",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
    )
    p.add_argument(
        "--oacp-dynamic-occluders",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
    )
    p.add_argument(
        "--oacp-visible-reach-mode",
        type=str,
        choices=["constant_velocity", "worst_case"],
        default=None,
    )
    p.add_argument(
        "--uni-reverse-bias",
        type=float,
        default=None,
        help="Override backup_cbf.reverse_bias_occ_uni.",
    )
    p.add_argument(
        "--uni-reverse-gate-angle",
        type=float,
        default=None,
        help="Override backup_cbf.reverse_speed_gate_angle_occ_uni.",
    )
    p.add_argument(
        "--uni-reverse-gate-power",
        type=float,
        default=None,
        help="Override backup_cbf.reverse_speed_gate_power_occ_uni.",
    )
    p.add_argument(
        "--uni-v-min-cmd-rev",
        type=float,
        default=None,
        help="Override backup_cbf.v_min_cmd_rev_occ_uni.",
    )
    p.add_argument(
        "--occ-dt-backup",
        type=float,
        default=None,
        help="Override backup_cbf.dt_backup for occlusion backup rollout.",
    )
    p.add_argument(
        "--vref",
        type=str,
        default=None,
        choices=["default", "los"],
        help="Override backup_cbf.vref_front_mode_occ for occlusion backup v_target construction.",
    )
    p.add_argument("--out-dir", type=str, default="debug_logs")
    p.add_argument(
        "--backup-cbf-json",
        type=str,
        default=None,
        help="JSON object or path to JSON file with backup_cbf overrides forwarded to examples/test_crowd.py.",
    )
    p.add_argument(
        "--robot-spec-json",
        type=str,
        default=None,
        help="JSON object or path to JSON file with robot_spec overrides forwarded to examples/test_crowd.py.",
    )
    args = p.parse_args()
    args.backup_cbf_overrides = _parse_json_arg(args.backup_cbf_json) or {}
    args.robot_spec_overrides = _parse_json_arg(args.robot_spec_json)
    if args.uni_reverse_bias is not None:
        args.backup_cbf_overrides["reverse_bias_occ_uni"] = float(args.uni_reverse_bias)
    if args.uni_reverse_gate_angle is not None:
        args.backup_cbf_overrides["reverse_speed_gate_angle_occ_uni"] = float(args.uni_reverse_gate_angle)
    if args.uni_reverse_gate_power is not None:
        args.backup_cbf_overrides["reverse_speed_gate_power_occ_uni"] = float(args.uni_reverse_gate_power)
    if args.uni_v_min_cmd_rev is not None:
        args.backup_cbf_overrides["v_min_cmd_rev_occ_uni"] = float(args.uni_v_min_cmd_rev)
    if args.occ_dt_backup is not None:
        args.backup_cbf_overrides["dt_backup"] = float(args.occ_dt_backup)
    if args.vref is not None:
        args.backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    if not args.backup_cbf_overrides:
        args.backup_cbf_overrides = None
    if args.robot_spec_overrides is None:
        args.robot_spec_overrides = {}
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
    if oacp_cfg:
        args.robot_spec_overrides.setdefault("oacp_mpc", {}).update(oacp_cfg)
    if not args.robot_spec_overrides:
        args.robot_spec_overrides = None

    if int(args.idx_start) < 1 or int(args.idx_end) < int(args.idx_start):
        raise ValueError("Require 1 <= idx-start <= idx-end")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if str(args.baseline_suite) == "non_occlusion_5":
        return _run_non_occ_5_suite(args, out_dir, ts)

    if args.baseline is None:
        raise ValueError("Single-baseline mode requires --baseline (or use --baseline-suite non_occlusion_5).")

    _run_baseline_sweep(
        scenario_name=str(args.scenario),
        baseline_alias=str(args.baseline),
        run_label=str(args.baseline),
        model=str(args.model),
        seed=int(args.seed),
        idx_start=int(args.idx_start),
        idx_end=int(args.idx_end),
        n_rand=int(args.n_rand),
        tf=float(args.tf),
        wmax=str(args.wmax),
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        occ_version=args.occ_version,
        occ_visible_scale=args.occ_visible_scale,
        occ_enable_visible_hocbf=args.occ_enable_visible_hocbf,
        crowd_mode=str(args.crowd_mode),
        forced_events=int(args.forced_events),
        forced_bg_rand=args.forced_bg_rand,
        forced_hidden_speed=float(args.forced_hidden_speed),
        forced_occluder_radius_min=float(args.forced_occluder_radius_min),
        forced_occluder_radius_max=float(args.forced_occluder_radius_max),
        forced_validate_occlusion=bool(args.forced_validate_occlusion),
        forced_require_corridor_conflict=bool(args.forced_require_corridor_conflict),
        backup_cbf_overrides=args.backup_cbf_overrides,
        robot_spec_overrides=args.robot_spec_overrides,
        workers=int(args.workers),
        out_dir=out_dir,
        ts=ts,
    )
    return 0


def _run_one_idx_job_star(
    task: tuple[
        str, str, str, str, int, int, float, str, bool | None,
        str | None, float | None, bool | None, str, int, int | None, float, float, float,
        bool, bool, dict[str, Any] | None, dict[str, Any] | None, str
    ]
) -> dict[str, Any]:
    return _run_one_idx_job(
        baseline_alias=task[0],
        scenario_name=task[1],
        controller_pos=task[2],
        model=task[3],
        seed=task[4],
        idx=task[5],
        n_rand=task[6],
        tf=task[7],
        wmax=task[8],
        oa_allow_solver_fallback=task[9],
        occ_version=task[10],
        occ_visible_scale=task[11],
        occ_enable_visible_hocbf=task[12],
        crowd_mode=task[13],
        forced_events=task[14],
        forced_bg_rand=task[15],
        forced_hidden_speed=task[16],
        forced_occluder_radius_min=task[17],
        forced_occluder_radius_max=task[18],
        forced_validate_occlusion=task[19],
        forced_require_corridor_conflict=task[20],
        backup_cbf_overrides=task[21],
        robot_spec_overrides=task[22],
        run_label=task[23],
    )


def _run_one_idx_job(
    *,
    baseline_alias: str,
    scenario_name: str,
    controller_pos: str,
    model: str,
    seed: int,
    idx: int,
    n_rand: int,
    tf: float,
    wmax: str,
    oa_allow_solver_fallback: bool | None,
    occ_version: str | None,
    occ_visible_scale: float | None,
    occ_enable_visible_hocbf: bool | None,
    crowd_mode: str,
    forced_events: int,
    forced_bg_rand: int | None,
    forced_hidden_speed: float,
    forced_occluder_radius_min: float,
    forced_occluder_radius_max: float,
    forced_validate_occlusion: bool,
    forced_require_corridor_conflict: bool,
    backup_cbf_overrides: dict[str, Any] | None,
    robot_spec_overrides: dict[str, Any] | None,
    run_label: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        run_crowd_scenario, scenario_label = _load_run_crowd_scenario(scenario_name)
        metrics = run_crowd_scenario(
            controller_type={"pos": str(controller_pos)},
            model_key=str(model),
            show_animation=False,
            save_animation=False,
            tf=float(tf),
            seed=int(seed),
            case_idx=int(idx),
            rand_obs=True,
            n_rand=int(n_rand),
            occ_version=occ_version,
            occ_visible_scale=occ_visible_scale,
            occ_enable_visible_hocbf=occ_enable_visible_hocbf,
            oa_wmax=str(wmax),
            oa_allow_solver_fallback=oa_allow_solver_fallback,
            crowd_mode=str(crowd_mode),
            forced_events=int(forced_events),
            forced_bg_rand=forced_bg_rand,
            forced_hidden_speed=float(forced_hidden_speed),
            forced_occluder_radius_min=float(forced_occluder_radius_min),
            forced_occluder_radius_max=float(forced_occluder_radius_max),
            forced_validate_occlusion=bool(forced_validate_occlusion),
            forced_require_corridor_conflict=bool(forced_require_corridor_conflict),
            backup_cbf_overrides=backup_cbf_overrides,
            robot_spec_overrides=robot_spec_overrides,
            return_metrics=True,
        )
        raw_outcome = str(metrics.get("outcome", "unknown"))
        return {
            "label": str(run_label),
            "scenario": str(scenario_label),
            "baseline": str(baseline_alias),
            "controller_pos": str(controller_pos),
            "wmax": str(wmax),
            "oa_allow_solver_fallback": oa_allow_solver_fallback,
            "occ_version": occ_version,
            "occ_visible_scale": occ_visible_scale,
            "occ_enable_visible_hocbf": occ_enable_visible_hocbf,
            "crowd_mode": str(metrics.get("crowd_mode", crowd_mode)),
            "idx": int(idx),
            "raw_outcome": raw_outcome,
            "class_3way": _classify_3way(raw_outcome),
            "n_forced_events": int(metrics.get("n_forced_events", 0) or 0),
            "n_background_rand": int(metrics.get("n_background_rand", 0) or 0),
            "n_forced_initially_occluded": int(metrics.get("n_forced_initially_occluded", 0) or 0),
            "n_forced_revealed": int(metrics.get("n_forced_revealed", 0) or 0),
            "n_forced_corridor_conflict": int(metrics.get("n_forced_corridor_conflict", 0) or 0),
            "min_reveal_distance_to_ego_path": _safe_float(
                metrics.get("min_reveal_distance_to_ego_path", None)
            ),
            "min_reveal_ttc_to_nominal_ego": _safe_float(
                metrics.get("min_reveal_ttc_to_nominal_ego", None)
            ),
            "reveal_steps": json.dumps(metrics.get("reveal_steps", []), separators=(",", ":")),
            "forced_event_meta": json.dumps(metrics.get("forced_event_meta", []), separators=(",", ":")),
            "avg_solve_time_ms": _safe_float(metrics.get("avg_solve_time_ms", None)),
            "avg_control_intervention_l2_sq": _safe_float(
                metrics.get("avg_control_intervention_l2_sq", None)
            ),
            "total_sim_time": _safe_float(metrics.get("total_sim_time", None)),
            "case_wall_time_s": float(time.perf_counter() - t0),
            "total_steps": int(metrics.get("total_steps", 0) or 0),
            "final_goal_distance": _safe_float(metrics.get("final_goal_distance", None)),
            "status": str(metrics.get("status", "unknown")),
            "ret": int(metrics.get("ret", 0) or 0),
            "exception": None,
        }
    except Exception as ex:
        _, scenario_label = _load_run_crowd_scenario(scenario_name)
        return {
            "label": str(run_label),
            "scenario": str(scenario_label),
            "baseline": str(baseline_alias),
            "controller_pos": str(controller_pos),
            "wmax": str(wmax),
            "oa_allow_solver_fallback": oa_allow_solver_fallback,
            "occ_version": occ_version,
            "occ_visible_scale": occ_visible_scale,
            "occ_enable_visible_hocbf": occ_enable_visible_hocbf,
            "crowd_mode": str(crowd_mode),
            "idx": int(idx),
            "raw_outcome": "exception",
            "class_3way": "infeasible",
            "n_forced_events": 0,
            "n_background_rand": 0,
            "n_forced_initially_occluded": 0,
            "n_forced_revealed": 0,
            "n_forced_corridor_conflict": 0,
            "min_reveal_distance_to_ego_path": None,
            "min_reveal_ttc_to_nominal_ego": None,
            "reveal_steps": "[]",
            "forced_event_meta": "[]",
            "avg_solve_time_ms": None,
            "avg_control_intervention_l2_sq": None,
            "total_sim_time": None,
            "case_wall_time_s": float(time.perf_counter() - t0),
            "total_steps": 0,
            "final_goal_distance": None,
            "status": "failure",
            "ret": -99,
            "exception": str(ex),
        }


if __name__ == "__main__":
    raise SystemExit(main())
