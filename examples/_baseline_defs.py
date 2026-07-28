from __future__ import annotations

import os


CROWD_BENCHMARK_DEFAULTS = {
    "scenario": "crowd",
    "baseline": "oacp_mpc",
    "model": "di",
    "seed": 42,
    "idx_start": 1,
    "idx_end": 100,
    "n_rand": 10,
    "tf": 500.0,
    "crowd_mode": "forced_emergence",
    "forced_events": 6,
    "forced_hidden_speed": 1.0,
    "forced_occluder_radius_min": 0.8,
    "forced_occluder_radius_max": 1.0,
    "forced_validate_occlusion": True,
    "forced_require_corridor_conflict": True,
}

OACP_BENCHMARK_DEFAULTS = {
    "allow_solver_fallback": False,
    "dynamic_occluders": True,
    "visible_reach_mode": "constant_velocity",
    "branch_safety_gate": False,
}


def default_benchmark_workers(cpu_count: int | None = None) -> int:
    """Choose useful parallelism while leaving capacity for other local work."""
    available = int(os.cpu_count() or 1) if cpu_count is None else int(cpu_count)
    available = max(1, available)
    usable = available - 2 if available > 2 else available
    return max(1, min(8, usable))


CROWD_BASELINE_MAP = {
    "occlusion_cbf": "occlusion_cbf_qp",
    "cbf_qp": "cbf_qp",
    "oa_mpc": "oa_mpc",
    "single_risk_mpc": "single_risk_mpc",
    "control_tree_mpc": "control_tree_mpc",
    "oacp_mpc": "oacp_mpc",
}

CROWD_ALGO_CHOICES = tuple(
    [
        "occlusion_cbf_qp",
        "cbf_qp",
        "oa_mpc",
        "single_risk_mpc",
        "control_tree_mpc",
        "oacp_mpc",
    ]
)
CROWD_BASELINE_CHOICES = tuple(CROWD_BASELINE_MAP.keys())

CROWD_PLANNER_LABELS = {
    "occlusion_cbf_qp": "Occlusion CBF-QP",
    "cbf_qp": "CBF-QP",
    "oa_mpc": "OA-MPC",
    "single_risk_mpc": "Single-Hypothesis MPC",
    "control_tree_mpc": "Control-Tree MPC",
    "oacp_mpc": "OACP",
}

CROSSWALK_BASELINE_MAP = {
    "occlusion_cbf": "occlusion_cbf_qp",
    "cbf_qp": "cbf_qp",
    "oa_mpc": "oa_mpc",
}
CROSSWALK_BASELINE_CHOICES = tuple(CROSSWALK_BASELINE_MAP.keys())

def resolve_baseline_alias(baseline: str | None, algo: str, baseline_map: dict[str, str]) -> str:
    return baseline_map.get(baseline, algo)
