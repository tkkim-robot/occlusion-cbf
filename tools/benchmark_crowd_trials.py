#!/usr/bin/env python3
"""
Run crowd random-trial sweeps and print compact 3-way benchmark metrics.

Single-baseline mode:
    - one baseline over idx range

Suite mode:
    - sequentially run the 5 non-OCBF baselines:
      OA-MPC(wmax=default), OACP-MPC, control_tree_mpc,
      single_risk_mpc, cbf_qp

All runs force:
    - show_animation = False
    - save_animation = False

For Unicycle2D paper benchmark sweeps, all planners default to forward-only
actuation. Pass --uni-allow-reverse true or --uni-v-min to override.

With no arguments, this runs the canonical OACP crowd profile. Shared
scenario defaults also apply unchanged when another baseline or suite is chosen.
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

from examples._baseline_defs import (
    CROWD_BASELINE_MAP,
    CROWD_BENCHMARK_DEFAULTS,
    OACP_BENCHMARK_DEFAULTS,
    default_benchmark_workers,
)
from position_control.ocbf.defaults import (
    OCBF_QP_FAILURE_FALLBACK_MODES,
    OCBF_ROLLOUT_MODES,
    OCBF_SELECTION_MODES,
    OCBF_TERMINAL_MODES,
    OCBF_TERMINAL_RESIDUAL_MODES,
    OCBF_VREF_FRONT_MODES,
    OCBF_VREF_SCENARIO_WEIGHT_MODES,
    apply_crowd_ocbf_defaults,
    default_visible_hocbf_for_scenario,
)

BASELINE_MAP = dict(CROWD_BASELINE_MAP)

SUITE_NON_OCC_5 = [
    {"label": "oa_mpc", "baseline": "oa_mpc", "wmax": "default"},
    {"label": "oacp_mpc", "baseline": "oacp_mpc", "wmax": "default"},
    {"label": "control_tree_mpc", "baseline": "control_tree_mpc", "wmax": "default"},
    {"label": "single_risk_mpc", "baseline": "single_risk_mpc", "wmax": "default"},
    {"label": "cbf_qp", "baseline": "cbf_qp", "wmax": "default"},
]


def _is_ocbf_baseline(baseline_alias: str) -> bool:
    return str(BASELINE_MAP.get(str(baseline_alias), baseline_alias)).strip().lower() == "occlusion_cbf_qp"


def _default_uni_forward_only(
    *,
    model: str,
    baseline_alias: str,
    robot_spec_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Use forward-only Uni actuation by default for paper benchmark sweeps.

    The scenario runners keep their historical Uni default for direct
    visualization. This benchmark tool defaults all Uni benchmark runs to
    forward-only unless the caller explicitly sets a Uni speed-bound option.
    """
    if str(model).strip().lower() != "uni":
        return robot_spec_overrides

    out = dict(robot_spec_overrides or {})
    has_explicit_uni_bound = any(k in out for k in ("_uni_forward_only", "_uni_allow_reverse", "v_min"))
    if not has_explicit_uni_bound:
        out["_uni_forward_only"] = True
    return out


def _load_run_crowd_scenario(scenario_name: str):
    scenario_key = str(scenario_name).strip().lower()
    if scenario_key in {"crowd", "crowd2", "test_crowd", "test_crowd2"}:
        from examples.test_crowd import run_crowd_scenario as runner

        return runner, "crowd"
    if scenario_key in {"crowd_narrow", "crowd1", "test_crowd_narrow", "test_crowd1"}:
        from examples.test_crowd_narrow import run_crowd_scenario as runner

        return runner, "crowd_narrow"
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
    try:
        candidate_exists = candidate.exists()
    except OSError:
        candidate_exists = False
    if candidate_exists:
        with candidate.open("r") as f:
            data = json.load(f)
    else:
        data = json.loads(s)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("JSON override must decode to a JSON object")
    return data


def _parse_int_list_arg(value: str | None) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    out: list[int] = []
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


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
    exclude_idx: list[int] | None,
    n_rand: int,
    tf: float,
    wmax: str,
    oa_allow_solver_fallback: bool | None,
    oa_dynamic_occluders: bool | None,
    oa_dsafe: float | None,
    oa_visible_reach_mode: str | None,
    oa_use_nominal_tracking_cost: bool | None,
    oa_dt: float | None,
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
    if occ_enable_visible_hocbf is None:
        occ_enable_visible_hocbf = default_visible_hocbf_for_scenario(scenario_label)
    if str(scenario_label).strip().lower() == "crowd" and _is_ocbf_baseline(str(baseline_alias)):
        backup_cbf_overrides = apply_crowd_ocbf_defaults(backup_cbf_overrides)
    controller_pos = BASELINE_MAP[str(baseline_alias)]
    robot_spec_overrides = _default_uni_forward_only(
        model=str(model),
        baseline_alias=str(baseline_alias),
        robot_spec_overrides=robot_spec_overrides,
    )
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
        f"oa_dynamic_occluders={oa_dynamic_occluders} "
        f"oa_dsafe={oa_dsafe} oa_visible_reach_mode={oa_visible_reach_mode} "
        f"oa_use_nominal_tracking_cost={oa_use_nominal_tracking_cost} oa_dt={oa_dt} "
        f"occ_visible_scale={occ_visible_scale} "
        f"occ_enable_visible_hocbf={occ_enable_visible_hocbf} "
        f"crowd_mode={crowd_mode}",
        flush=True,
    )
    print("[RUN] show_animation=False, save_animation=False", flush=True)

    exclude_set = {int(i) for i in (exclude_idx or [])}
    idx_list = [idx for idx in range(int(idx_start), int(idx_end) + 1) if idx not in exclude_set]
    total = len(idx_list)
    if exclude_set:
        print(f"[RUN] excluding idx: {sorted(exclude_set)}", flush=True)
    if total <= 0:
        raise ValueError("No indices remain after applying --exclude-idx.")

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
                oa_dynamic_occluders=oa_dynamic_occluders,
                oa_dsafe=oa_dsafe,
                oa_visible_reach_mode=oa_visible_reach_mode,
                oa_use_nominal_tracking_cost=oa_use_nominal_tracking_cost,
                oa_dt=oa_dt,
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
                f"(raw={row['raw_outcome']}, compute_ms={row['avg_compute_time_ms']}, "
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
                oa_dynamic_occluders,
                oa_dsafe,
                oa_visible_reach_mode,
                oa_use_nominal_tracking_cost,
                oa_dt,
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
                        "avg_compute_time_ms": None,
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
                    f"(raw={row['raw_outcome']}, compute_ms={row['avg_compute_time_ms']}, "
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

    avg_ms_all = _mean(
        [_safe_float(r.get("avg_compute_time_ms", r.get("avg_solve_time_ms"))) for r in rows]
    )
    avg_intv_all = _mean([_safe_float(r.get("avg_control_intervention_l2_sq")) for r in rows])
    avg_term_slack_l1_all = _mean([_safe_float(r.get("avg_terminal_slack_l1")) for r in rows])
    avg_term_slack_max_all = _mean([_safe_float(r.get("avg_terminal_slack_max")) for r in rows])
    avg_term_slack_active_ratio_all = _mean([_safe_float(r.get("terminal_slack_active_ratio")) for r in rows])
    avg_occ_vref_unexpanded_margin_all = _mean([_safe_float(r.get("avg_occ_vref_unexpanded_margin")) for r in rows])
    avg_occ_vref_max_weight_all = _mean([_safe_float(r.get("avg_occ_vref_max_softmax_weight")) for r in rows])
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
        "oa_dynamic_occluders",
        "oa_dsafe",
        "oa_visible_reach_mode",
        "oa_use_nominal_tracking_cost",
        "oa_dt",
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
        "avg_compute_time_ms",
        "avg_solve_time_ms",
        "avg_control_intervention_l2_sq",
        "avg_terminal_slack_l1",
        "avg_terminal_slack_max",
        "terminal_slack_active_steps",
        "terminal_slack_active_ratio",
        "avg_occ_vref_unexpanded_margin",
        "avg_occ_vref_max_softmax_weight",
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
            "exclude_idx": sorted(exclude_set),
            "n_rand": int(n_rand),
            "tf": float(tf),
            "oa_wmax": str(wmax),
            "oa_allow_solver_fallback": oa_allow_solver_fallback,
            "oa_dynamic_occluders": oa_dynamic_occluders,
            "oa_dsafe": oa_dsafe,
            "oa_visible_reach_mode": oa_visible_reach_mode,
            "oa_use_nominal_tracking_cost": oa_use_nominal_tracking_cost,
            "oa_dt": oa_dt,
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
            "avg_compute_time_ms_over_idx": avg_ms_all,
            "avg_solve_time_ms_over_idx": avg_ms_all,
            "avg_control_intervention_l2_sq_over_idx": avg_intv_all,
            "avg_terminal_slack_l1_over_idx": avg_term_slack_l1_all,
            "avg_terminal_slack_max_over_idx": avg_term_slack_max_all,
            "avg_terminal_slack_active_ratio_over_idx": avg_term_slack_active_ratio_all,
            "avg_occ_vref_unexpanded_margin_over_idx": avg_occ_vref_unexpanded_margin_all,
            "avg_occ_vref_max_softmax_weight_over_idx": avg_occ_vref_max_weight_all,
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
    print(f"avg controller compute time over idx [ms]: {avg_ms_all}")
    print(f"avg control intervention over idx: {avg_intv_all}")
    print(f"avg terminal slack L1 over idx: {avg_term_slack_l1_all}")
    print(f"avg terminal slack max over idx: {avg_term_slack_max_all}")
    print(f"avg terminal slack active ratio over idx: {avg_term_slack_active_ratio_all}")
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
        "Methods: oa_mpc(wmax=default), oacp_mpc, "
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
            exclude_idx=list(args.exclude_idx),
            n_rand=int(args.n_rand),
            tf=float(args.tf),
            wmax=str(spec["wmax"]),
            oa_allow_solver_fallback=args.oa_allow_solver_fallback,
            oa_dynamic_occluders=args.oa_dynamic_occluders,
            oa_dsafe=args.oa_dsafe,
            oa_visible_reach_mode=args.oa_visible_reach_mode,
            oa_use_nominal_tracking_cost=args.oa_use_nominal_tracking_cost,
            oa_dt=args.oa_dt,
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
        "avg_compute_time_ms_over_idx",
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
                "avg_compute_time_ms_over_idx": avg.get(
                    "avg_compute_time_ms_over_idx",
                    avg.get("avg_solve_time_ms_over_idx"),
                ),
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
            "exclude_idx": list(args.exclude_idx),
            "n_rand": int(args.n_rand),
            "tf": float(args.tf),
            "oa_allow_solver_fallback": args.oa_allow_solver_fallback,
            "oa_dynamic_occluders": args.oa_dynamic_occluders,
            "oa_dsafe": args.oa_dsafe,
            "oa_visible_reach_mode": args.oa_visible_reach_mode,
            "oa_use_nominal_tracking_cost": args.oa_use_nominal_tracking_cost,
            "oa_dt": args.oa_dt,
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
            f"| avg_compute_ms={row['avg_compute_time_ms_over_idx']} "
            f"| avg_intv={row['avg_control_intervention_l2_sq_over_idx']} "
            f"| avg_sim_s={row['avg_total_sim_time_s_over_idx']} "
            f"| avg_wall_s={row['avg_case_wall_time_s_over_idx']}"
        )
    print(f"saved suite csv : {agg_rows_path}")
    print(f"saved suite json: {agg_json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run crowd idx sweep for one baseline or a predefined baseline suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=CROWD_BENCHMARK_DEFAULTS["scenario"],
        choices=["crowd", "crowd_narrow", "crowd2", "crowd1"],
        help=(
            "Select the canonical crowd benchmark or the legacy narrow layout. "
            "The former crowd2/crowd1 values remain accepted as aliases."
        ),
    )
    p.add_argument(
        "--baseline",
        type=str,
        default=CROWD_BENCHMARK_DEFAULTS["baseline"],
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
    p.add_argument(
        "--model",
        type=str,
        default=CROWD_BENCHMARK_DEFAULTS["model"],
        choices=["di", "du", "uni"],
    )
    p.add_argument("--seed", type=int, default=CROWD_BENCHMARK_DEFAULTS["seed"])
    p.add_argument("--idx-start", type=int, default=CROWD_BENCHMARK_DEFAULTS["idx_start"])
    p.add_argument("--idx-end", type=int, default=CROWD_BENCHMARK_DEFAULTS["idx_end"])
    p.add_argument(
        "--exclude-idx",
        type=str,
        default=None,
        help="Comma-separated idx list to skip from the inclusive idx range.",
    )
    p.add_argument(
        "--n-rand",
        type=int,
        default=CROWD_BENCHMARK_DEFAULTS["n_rand"],
        help=(
            "Random mode: number of moving obstacles. Forced-emergence mode: target number of non-occluder movers "
            "(hidden-emergence agents plus optional background clutter). Large forced occluders are controlled by "
            "--forced-events and are not counted in n-rand."
        ),
    )
    p.add_argument("--tf", type=float, default=CROWD_BENCHMARK_DEFAULTS["tf"])
    p.add_argument(
        "--crowd-mode",
        type=str,
        default=CROWD_BENCHMARK_DEFAULTS["crowd_mode"],
        choices=["random", "forced_emergence"],
        help="Forwarded to the selected scenario generator.",
    )
    p.add_argument(
        "--forced-events",
        type=int,
        default=CROWD_BENCHMARK_DEFAULTS["forced_events"],
    )
    p.add_argument(
        "--forced-bg-rand",
        type=int,
        default=None,
        help=(
            "Forced-emergence mode: explicit number of background random obstacles. "
            "Default uses a small clutter share and allocates the rest of n-rand to hidden emergence agents."
        ),
    )
    p.add_argument(
        "--forced-hidden-speed",
        type=float,
        default=CROWD_BENCHMARK_DEFAULTS["forced_hidden_speed"],
    )
    p.add_argument(
        "--forced-occluder-radius-min",
        type=float,
        default=CROWD_BENCHMARK_DEFAULTS["forced_occluder_radius_min"],
    )
    p.add_argument(
        "--forced-occluder-radius-max",
        type=float,
        default=CROWD_BENCHMARK_DEFAULTS["forced_occluder_radius_max"],
    )
    p.add_argument(
        "--forced-validate-occlusion",
        type=_str2bool,
        nargs="?",
        const=True,
        default=CROWD_BENCHMARK_DEFAULTS["forced_validate_occlusion"],
    )
    p.add_argument(
        "--forced-require-corridor-conflict",
        type=_str2bool,
        nargs="?",
        const=True,
        default=CROWD_BENCHMARK_DEFAULTS["forced_require_corridor_conflict"],
    )
    p.add_argument(
        "--workers",
        type=int,
        default=default_benchmark_workers(),
        help="Parallel worker processes per baseline; auto-capped at 8 and reserves two CPUs.",
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
        "--oa-dynamic-occluders",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Forwarded to test_crowd OA-MPC config.",
    )
    p.add_argument(
        "--oa-dsafe",
        type=float,
        default=None,
        help="Forwarded to test_crowd OA-MPC config.",
    )
    p.add_argument(
        "--oa-visible-reach-mode",
        type=str,
        choices=["constant_velocity", "worst_case"],
        default=None,
        help="Forwarded to test_crowd OA-MPC config.",
    )
    p.add_argument(
        "--oa-use-nominal-tracking-cost",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Forwarded to test_crowd OA-MPC config.",
    )
    p.add_argument(
        "--oa-dt",
        type=float,
        default=None,
        help="Forwarded to test_crowd OA-MPC config.",
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
        default=None,
        help=(
            "Forwarded to scenario runner: enable visible-obstacle CBF/HOCBF rows "
            "inside occlusion-CBF. Default is true for crowd and false otherwise."
        ),
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
        default=OACP_BENCHMARK_DEFAULTS["allow_solver_fallback"],
    )
    p.add_argument(
        "--oacp-dynamic-occluders",
        type=_str2bool,
        nargs="?",
        const=True,
        default=OACP_BENCHMARK_DEFAULTS["dynamic_occluders"],
    )
    p.add_argument(
        "--oacp-visible-reach-mode",
        type=str,
        choices=["constant_velocity", "worst_case"],
        default=OACP_BENCHMARK_DEFAULTS["visible_reach_mode"],
    )
    p.add_argument(
        "--oacp-branch-safety-gate",
        type=_str2bool,
        nargs="?",
        const=True,
        default=OACP_BENCHMARK_DEFAULTS["branch_safety_gate"],
        help="OACP only: gate the selected contingency branch and switch to the other safe branch when possible.",
    )
    p.add_argument(
        "--oacp-branch-gate-reject-all",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="OACP only: mark infeasible when both contingency branches fail the branch safety gate.",
    )
    p.add_argument(
        "--oacp-branch-slack-gate-tol",
        type=float,
        default=None,
        help="OACP only: maximum branch risk slack allowed by branch safety gate.",
    )
    p.add_argument(
        "--oacp-branch-clearance-gate-tol",
        type=float,
        default=None,
        help="OACP only: allowed negative planned clearance tolerance for branch safety gate.",
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
        "--uni-allow-reverse",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help=(
            "Allow Uni reverse by setting v_min=-v_max unless --uni-v-min is given. "
            "This overrides the benchmark-tool default forward-only mode for Uni paper sweeps."
        ),
    )
    p.add_argument(
        "--uni-forward-only",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Force Unicycle2D forward-only by setting v_min=0 unless --uni-v-min is given. "
            "This is already the default for Uni paper sweeps in this benchmark tool."
        ),
    )
    p.add_argument(
        "--uni-v-min",
        type=float,
        default=None,
        help="Override Unicycle2D input lower speed bound.",
    )
    p.add_argument(
        "--uni-vref-tracking-mode",
        type=str,
        choices=["gated", "projected"],
        default=None,
        help="Unicycle OCBF mapping from inertial v_ref to (v, omega).",
    )
    p.add_argument("--uni-k-theta-p", type=float, default=None, help="Override backup_cbf.k_theta_occ_uni_p.")
    p.add_argument("--uni-k-theta-d", type=float, default=None, help="Override backup_cbf.k_theta_occ_uni_d.")
    p.add_argument("--uni-k-v-p", type=float, default=None, help="Override backup_cbf.k_v_occ_uni_p.")
    p.add_argument("--uni-k-v-d", type=float, default=None, help="Override backup_cbf.k_v_occ_uni_d.")
    p.add_argument("--uni-turn-boost", type=float, default=None, help="Override backup_cbf.k_turn_boost_occ_uni.")
    p.add_argument("--uni-turn-boost-angle", type=float, default=None, help="Override backup_cbf.turn_boost_angle_occ_uni.")
    p.add_argument("--uni-v-min-cmd", type=float, default=None, help="Override backup_cbf.v_min_occ_uni.")
    p.add_argument("--uni-turn-crawl-speed", type=float, default=None, help="Minimum forward crawl speed while turning toward OCBF v_ref.")
    p.add_argument("--uni-turn-crawl-angle", type=float, default=None, help="Max heading error [rad] where OCBF turn-crawl is allowed.")
    p.add_argument(
        "--occ-t-horizon",
        type=float,
        default=None,
        help="Override backup_cbf.T_horizon for occlusion backup rollout.",
    )
    p.add_argument(
        "--occ-rho-T",
        type=str,
        default=None,
        help="Override backup_cbf.rho_T for terminal occlusion backup constraint. Accepts a float or 'auto'.",
    )
    p.add_argument(
        "--occ-dt-backup",
        type=float,
        default=None,
        help="Override backup_cbf.dt_backup for occlusion backup rollout.",
    )
    p.add_argument(
        "--occ-rollout-mode",
        type=str,
        default=None,
        choices=OCBF_ROLLOUT_MODES,
        help="Override backup_cbf.occ_rollout_mode for occlusion backup rollout construction.",
    )
    p.add_argument(
        "--occ-terminal-slack-weight",
        type=float,
        default=None,
        help="Override backup_cbf.terminal_slack_weight for terminal-set rows only.",
    )
    p.add_argument(
        "--occ-terminal-slack-max",
        type=float,
        default=None,
        help="Override backup_cbf.terminal_slack_max for terminal-set rows only.",
    )
    p.add_argument(
        "--occ-obs-hocbf-slack-max",
        type=float,
        default=None,
        help="Allow bounded slack on visible-obstacle HOCBF rows in OCBF-QP.",
    )
    p.add_argument(
        "--occ-rollout-slack-max",
        type=float,
        default=None,
        help="Allow bounded slack on occlusion rollout rows in OCBF-QP.",
    )
    p.add_argument(
        "--occ-terminal-mode",
        type=str,
        choices=OCBF_TERMINAL_MODES,
        default=None,
        help="Select which OCBF occlusion scenarios receive terminal backup-set rows.",
    )
    p.add_argument(
        "--occ-terminal-active-count",
        type=int,
        default=None,
        help="Top-M terminal rows to keep when --occ-terminal-mode=topm.",
    )
    p.add_argument(
        "--occ-terminal-residual-mode",
        type=str,
        choices=OCBF_TERMINAL_RESIDUAL_MODES,
        default=None,
        help="Optionally replace terminal rows with predicted terminal visibility residual sets.",
    )
    p.add_argument(
        "--occ-terminal-visibility-reaction-margin",
        type=float,
        default=None,
        help="Extra terminal buffer applied to predicted visibility residual halfspaces.",
    )
    p.add_argument(
        "--occ-qp-failure-fallback-mode",
        type=str,
        choices=OCBF_QP_FAILURE_FALLBACK_MODES,
        default=None,
        help="How OCBF-QP classifies backup-policy fallback when the QP solve itself fails.",
    )
    p.add_argument(
        "--occ-vref-scenario-softmax-kappa",
        type=float,
        default=None,
        help="Override backup_cbf.vref_scenario_softmax_kappa for occlusion-CBF front-speed scenario weighting.",
    )
    p.add_argument(
        "--occ-vref-scenario-weight-mode",
        type=str,
        choices=OCBF_VREF_SCENARIO_WEIGHT_MODES,
        default=None,
        help=(
            "Override backup_cbf.vref_scenario_weight_mode for OCBF scenario blending. "
            "barrier_expand scores rollout-expanded margins; barrier_unexpand scores "
            "unexpanded current-geometry margins."
        ),
    )
    p.add_argument(
        "--occ-max-active-occlusions",
        type=int,
        default=None,
        help="Limit occlusion-CBF to the top-K active occlusion scenarios. 0 keeps all scenarios.",
    )
    p.add_argument(
        "--occ-selection-mode",
        type=str,
        choices=OCBF_SELECTION_MODES,
        default=None,
        help="Occlusion-CBF active occlusion selection score.",
    )
    p.add_argument(
        "--occ-kappa",
        type=float,
        default=None,
        help="Override the OCBF barrier smoothing kappa. Default is fixed at 10.0; use only for ablations.",
    )
    p.add_argument(
        "--vref",
        type=str,
        default=None,
        choices=OCBF_VREF_FRONT_MODES,
        help=(
            "Override backup_cbf.vref_front_mode_occ for occlusion backup v_target construction. "
            "Internal default is `los`; `default` keeps the fixed polygon normal."
        ),
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
    args.exclude_idx = _parse_int_list_arg(args.exclude_idx)
    args.backup_cbf_overrides = _parse_json_arg(args.backup_cbf_json) or {}
    args.robot_spec_overrides = _parse_json_arg(args.robot_spec_json) or {}
    if args.uni_allow_reverse is not None:
        args.robot_spec_overrides["_uni_allow_reverse"] = bool(args.uni_allow_reverse)
    if args.uni_forward_only:
        args.robot_spec_overrides["_uni_forward_only"] = True
    if args.uni_v_min is not None:
        args.robot_spec_overrides["v_min"] = float(args.uni_v_min)
    if args.uni_reverse_bias is not None:
        args.backup_cbf_overrides["reverse_bias_occ_uni"] = float(args.uni_reverse_bias)
    if args.uni_reverse_gate_angle is not None:
        args.backup_cbf_overrides["reverse_speed_gate_angle_occ_uni"] = float(args.uni_reverse_gate_angle)
    if args.uni_reverse_gate_power is not None:
        args.backup_cbf_overrides["reverse_speed_gate_power_occ_uni"] = float(args.uni_reverse_gate_power)
    if args.uni_v_min_cmd_rev is not None:
        args.backup_cbf_overrides["v_min_cmd_rev_occ_uni"] = float(args.uni_v_min_cmd_rev)
    if args.uni_vref_tracking_mode is not None:
        args.backup_cbf_overrides["vref_tracking_mode_occ_uni"] = str(args.uni_vref_tracking_mode).strip().lower()
    if args.uni_k_theta_p is not None:
        args.backup_cbf_overrides["k_theta_occ_uni_p"] = float(args.uni_k_theta_p)
    if args.uni_k_theta_d is not None:
        args.backup_cbf_overrides["k_theta_occ_uni_d"] = float(args.uni_k_theta_d)
    if args.uni_k_v_p is not None:
        args.backup_cbf_overrides["k_v_occ_uni_p"] = float(args.uni_k_v_p)
    if args.uni_k_v_d is not None:
        args.backup_cbf_overrides["k_v_occ_uni_d"] = float(args.uni_k_v_d)
    if args.uni_turn_boost is not None:
        args.backup_cbf_overrides["k_turn_boost_occ_uni"] = float(args.uni_turn_boost)
    if args.uni_turn_boost_angle is not None:
        args.backup_cbf_overrides["turn_boost_angle_occ_uni"] = float(args.uni_turn_boost_angle)
    if args.uni_v_min_cmd is not None:
        args.backup_cbf_overrides["v_min_occ_uni"] = float(args.uni_v_min_cmd)
    if args.uni_turn_crawl_speed is not None:
        args.backup_cbf_overrides["turn_crawl_speed_occ_uni"] = float(args.uni_turn_crawl_speed)
    if args.uni_turn_crawl_angle is not None:
        args.backup_cbf_overrides["turn_crawl_angle_occ_uni"] = float(args.uni_turn_crawl_angle)
    if args.occ_t_horizon is not None:
        args.backup_cbf_overrides["T_horizon"] = float(args.occ_t_horizon)
    if args.occ_rho_T is not None:
        rho_raw = str(args.occ_rho_T).strip()
        rho_key = rho_raw.lower()
        if rho_key in {"auto", "auto_stop", "stop", "stopping_distance"}:
            args.backup_cbf_overrides["rho_T"] = rho_key
        else:
            args.backup_cbf_overrides["rho_T"] = float(rho_raw)
    if args.occ_dt_backup is not None:
        args.backup_cbf_overrides["dt_backup"] = float(args.occ_dt_backup)
    if args.occ_rollout_mode is not None:
        args.backup_cbf_overrides["occ_rollout_mode"] = str(args.occ_rollout_mode).strip().lower()
    if args.occ_terminal_slack_weight is not None:
        args.backup_cbf_overrides["terminal_slack_weight"] = float(args.occ_terminal_slack_weight)
    if args.occ_terminal_slack_max is not None:
        args.backup_cbf_overrides["terminal_slack_max"] = float(args.occ_terminal_slack_max)
    if args.occ_obs_hocbf_slack_max is not None:
        args.backup_cbf_overrides["obs_hocbf_slack_max"] = float(args.occ_obs_hocbf_slack_max)
    if args.occ_rollout_slack_max is not None:
        args.backup_cbf_overrides["occ_rollout_slack_max"] = float(args.occ_rollout_slack_max)
    if args.occ_terminal_mode is not None:
        args.backup_cbf_overrides["terminal_mode"] = str(args.occ_terminal_mode).strip().lower()
    if args.occ_terminal_active_count is not None:
        args.backup_cbf_overrides["terminal_active_count"] = int(args.occ_terminal_active_count)
    if args.occ_terminal_residual_mode is not None:
        args.backup_cbf_overrides["terminal_residual_mode"] = str(args.occ_terminal_residual_mode).strip().lower()
    if args.occ_terminal_visibility_reaction_margin is not None:
        args.backup_cbf_overrides["terminal_visibility_reaction_margin"] = float(
            args.occ_terminal_visibility_reaction_margin
        )
    if args.occ_qp_failure_fallback_mode is not None:
        args.backup_cbf_overrides["qp_failure_fallback_mode"] = str(args.occ_qp_failure_fallback_mode).strip().lower()
    if args.occ_vref_scenario_softmax_kappa is not None:
        args.backup_cbf_overrides["vref_scenario_softmax_kappa"] = float(args.occ_vref_scenario_softmax_kappa)
    if args.occ_vref_scenario_weight_mode is not None:
        args.backup_cbf_overrides["vref_scenario_weight_mode"] = str(args.occ_vref_scenario_weight_mode).strip().lower()
    if args.occ_max_active_occlusions is not None:
        args.backup_cbf_overrides["max_active_occlusions"] = int(args.occ_max_active_occlusions)
    if args.occ_selection_mode is not None:
        args.backup_cbf_overrides["occ_selection_mode"] = str(args.occ_selection_mode).strip().lower()
    if args.occ_kappa is not None:
        args.robot_spec_overrides["occ_kappa"] = float(args.occ_kappa)
    if args.vref is not None:
        args.backup_cbf_overrides["vref_front_mode_occ"] = str(args.vref).strip().lower()
    if not args.backup_cbf_overrides:
        args.backup_cbf_overrides = None
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
    if args.oacp_branch_safety_gate is not None:
        oacp_cfg["branch_safety_gate"] = bool(args.oacp_branch_safety_gate)
    if args.oacp_branch_gate_reject_all is not None:
        oacp_cfg["branch_gate_reject_all"] = bool(args.oacp_branch_gate_reject_all)
    if args.oacp_branch_slack_gate_tol is not None:
        oacp_cfg["branch_slack_gate_tol"] = float(args.oacp_branch_slack_gate_tol)
    if args.oacp_branch_clearance_gate_tol is not None:
        oacp_cfg["branch_clearance_gate_tol"] = float(args.oacp_branch_clearance_gate_tol)
    if oacp_cfg:
        args.robot_spec_overrides.setdefault("oacp_mpc", {}).update(oacp_cfg)
    if not args.robot_spec_overrides:
        args.robot_spec_overrides = None

    if int(args.idx_start) < 1 or int(args.idx_end) < int(args.idx_start):
        raise ValueError("Require 1 <= idx-start <= idx-end")
    if any(int(idx) < 1 for idx in args.exclude_idx):
        raise ValueError("--exclude-idx entries must be >= 1")

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
        exclude_idx=list(args.exclude_idx),
        n_rand=int(args.n_rand),
        tf=float(args.tf),
        wmax=str(args.wmax),
        oa_allow_solver_fallback=args.oa_allow_solver_fallback,
        oa_dynamic_occluders=args.oa_dynamic_occluders,
        oa_dsafe=args.oa_dsafe,
        oa_visible_reach_mode=args.oa_visible_reach_mode,
        oa_use_nominal_tracking_cost=args.oa_use_nominal_tracking_cost,
        oa_dt=args.oa_dt,
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
        bool | None, float | None, str | None, bool | None, float | None,
        float | None, bool | None, str, int, int | None, float, float, float,
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
        oa_dynamic_occluders=task[10],
        oa_dsafe=task[11],
        oa_visible_reach_mode=task[12],
        oa_use_nominal_tracking_cost=task[13],
        oa_dt=task[14],
        occ_visible_scale=task[15],
        occ_enable_visible_hocbf=task[16],
        crowd_mode=task[17],
        forced_events=task[18],
        forced_bg_rand=task[19],
        forced_hidden_speed=task[20],
        forced_occluder_radius_min=task[21],
        forced_occluder_radius_max=task[22],
        forced_validate_occlusion=task[23],
        forced_require_corridor_conflict=task[24],
        backup_cbf_overrides=task[25],
        robot_spec_overrides=task[26],
        run_label=task[27],
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
    oa_dynamic_occluders: bool | None,
    oa_dsafe: float | None,
    oa_visible_reach_mode: str | None,
    oa_use_nominal_tracking_cost: bool | None,
    oa_dt: float | None,
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
            occ_visible_scale=occ_visible_scale,
            occ_enable_visible_hocbf=occ_enable_visible_hocbf,
            oa_wmax=str(wmax),
            oa_allow_solver_fallback=oa_allow_solver_fallback,
            oa_dynamic_occluders=oa_dynamic_occluders,
            oa_dsafe=oa_dsafe,
            oa_visible_reach_mode=oa_visible_reach_mode,
            oa_use_nominal_tracking_cost=oa_use_nominal_tracking_cost,
            oa_dt=oa_dt,
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
        avg_compute_time_ms = _safe_float(
            metrics.get("avg_compute_time_ms", metrics.get("avg_solve_time_ms", None))
        )
        return {
            "label": str(run_label),
            "scenario": str(scenario_label),
            "baseline": str(baseline_alias),
            "controller_pos": str(controller_pos),
            "wmax": str(wmax),
            "oa_allow_solver_fallback": oa_allow_solver_fallback,
            "oa_dynamic_occluders": oa_dynamic_occluders,
            "oa_dsafe": oa_dsafe,
            "oa_visible_reach_mode": oa_visible_reach_mode,
            "oa_use_nominal_tracking_cost": oa_use_nominal_tracking_cost,
            "oa_dt": oa_dt,
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
            "avg_compute_time_ms": avg_compute_time_ms,
            "avg_solve_time_ms": avg_compute_time_ms,
            "avg_control_intervention_l2_sq": _safe_float(
                metrics.get("avg_control_intervention_l2_sq", None)
            ),
            "avg_terminal_slack_l1": _safe_float(metrics.get("avg_terminal_slack_l1", None)),
            "avg_terminal_slack_max": _safe_float(metrics.get("avg_terminal_slack_max", None)),
            "terminal_slack_active_steps": int(metrics.get("terminal_slack_active_steps", 0) or 0),
            "terminal_slack_active_ratio": _safe_float(metrics.get("terminal_slack_active_ratio", None)),
            "avg_occ_vref_unexpanded_margin": _safe_float(metrics.get("avg_occ_vref_unexpanded_margin", None)),
            "avg_occ_vref_max_softmax_weight": _safe_float(metrics.get("avg_occ_vref_max_softmax_weight", None)),
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
            "oa_dynamic_occluders": oa_dynamic_occluders,
            "oa_dsafe": oa_dsafe,
            "oa_visible_reach_mode": oa_visible_reach_mode,
            "oa_use_nominal_tracking_cost": oa_use_nominal_tracking_cost,
            "oa_dt": oa_dt,
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
                        "avg_compute_time_ms": None,
                        "avg_solve_time_ms": None,
                        "avg_control_intervention_l2_sq": None,
                        "avg_terminal_slack_l1": None,
                        "avg_terminal_slack_max": None,
                        "terminal_slack_active_steps": 0,
                        "terminal_slack_active_ratio": None,
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
