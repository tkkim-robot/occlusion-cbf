from __future__ import annotations


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
    "occlusion_cbf_qp": "OCBF-QP",
    "cbf_qp": "CBF-QP",
    "oa_mpc": "OA-MPC",
    "single_risk_mpc": "Single-Risk MPC",
    "control_tree_mpc": "Control-Tree MPC",
    "oacp_mpc": "OACP-MPC",
}

CROSSWALK_BASELINE_MAP = {
    "occlusion_cbf": "occlusion_cbf_qp",
    "cbf_qp": "cbf_qp",
    "oa_mpc": "oa_mpc",
}
CROSSWALK_BASELINE_CHOICES = tuple(CROSSWALK_BASELINE_MAP.keys())

HOSPITAL_ALGO_CHOICES = ("occlusion_cbf_qp", "cbf_qp")


def resolve_baseline_alias(baseline: str | None, algo: str, baseline_map: dict[str, str]) -> str:
    return baseline_map.get(baseline, algo)
