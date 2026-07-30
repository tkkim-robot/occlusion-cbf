#!/usr/bin/env python3
"""Optuna tuning for the canonical Occlusion-CBF crowd benchmark.

Each completed trial runs the exact inclusive crowd case range requested by the
CLI. Cases are evaluated in deterministic batches so the median pruner compares
the same prefixes across trials. The scalar objective is strictly ordered:

1. minimize collision count;
2. maximize success count;
3. minimize mean controller compute time.

The paper-fixed settings (full-circle obstacle selection, unexpanded backup
scenario weighting, hard safety rows, and canonical crowd geometry) are not
sampled.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Pin numerical libraries before NumPy/JAX are imported in this process or any
# spawned case workers.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import numpy as np
import optuna
from optuna.pruners import BasePruner, MedianPruner, NopPruner
from optuna.samplers import GridSampler, TPESampler

from examples._baseline_defs import CROWD_BENCHMARK_DEFAULTS, CROWD_BASELINE_MAP
from position_control.ocbf.defaults import (
    OCBF_CROWD_QP_FAILURE_FALLBACK_MODE,
    OCBF_CROWD_VREF_SCENARIO_WEIGHT_MODE,
    apply_crowd_ocbf_defaults,
    load_shared_robot_parameters,
)
from tools.benchmark_crowd_trials import _run_one_idx_job


WANDB_PROJECT_DEFAULT = "occlusion-cbf-tuning"
COLLISION_WEIGHT = 1_000_000.0
FAILURE_WEIGHT = 1_000.0
COMPUTE_TIME_CAP_MS = 999.999
ACTIVE_OCCLUSION_CHOICES = [0, 3, 5, 10]  # 0 means all occlusions.
COMPATIBILITY_CONFIG_KEYS = (
    "scenario",
    "baseline",
    "model",
    "model_name",
    "seed",
    "indices",
    "n_rand",
    "tf",
    "workers",
    "batch_size",
    "study_name",
    "objective_priority",
    "objective_weights",
    "benchmark_protocol",
    "search_space",
    "fixed_ocbf",
    "sampler",
    "pruner",
    "source_commit",
    "source_fingerprint",
    "tuning_mode",
)
TUNING_MODES = ("controller", "terminal_relax")
TERMINAL_RELAX_GRID = {
    "terminal_slack_weight": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "terminal_slack_max": [0.25, 0.5, 1.0, 2.0],
}

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "di": {
        "display_name": "DoubleIntegrator2D",
        "n_rand": 50,
        "incumbents": [
            {
                "T_horizon": 0.5,
                "max_active_occlusions": 0,
                "vref_scenario_softmax_kappa": 10,
                "k_p_occ_di": 1.0,
                "k_d_occ_di": 1.0,
            },
            {
                "T_horizon": 0.5,
                "max_active_occlusions": 3,
                "vref_scenario_softmax_kappa": 40,
                "k_p_occ_di": 1.0,
                "k_d_occ_di": 1.0,
            },
        ],
    },
    "uni": {
        "display_name": "Unicycle2D",
        "n_rand": 30,
        "incumbents": [
            {
                "T_horizon": 0.5,
                "max_active_occlusions": 0,
                "vref_scenario_softmax_kappa": 1,
                "k_theta_occ_uni_p": 2.5,
                "k_theta_occ_uni_d": 0.4,
                "k_v_occ_uni_p": 1.0,
                "k_v_occ_uni_d": 0.2,
                "k_turn_boost_occ_uni": 1.0,
                "turn_boost_angle_occ_uni": float(np.pi / 6.0),
            },
            {
                "T_horizon": 1.5,
                "max_active_occlusions": 3,
                "vref_scenario_softmax_kappa": 40,
                "k_theta_occ_uni_p": 2.5,
                "k_theta_occ_uni_d": 0.4,
                "k_v_occ_uni_p": 1.0,
                "k_v_occ_uni_d": 0.2,
                "k_turn_boost_occ_uni": 1.0,
                "turn_boost_angle_occ_uni": float(np.pi / 6.0),
            },
        ],
    },
}

SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "di": {
        "T_horizon": {"type": "categorical", "choices": [0.25, 0.5, 0.75, 1.0]},
        "max_active_occlusions": {
            "type": "categorical",
            "choices": ACTIVE_OCCLUSION_CHOICES,
        },
        "vref_scenario_softmax_kappa": {
            "type": "categorical",
            "choices": [0, 5, 10, 20, 40, 60],
        },
        "k_p_occ_di": {"type": "float", "low": 0.5, "high": 3.0, "log": True},
        "k_d_occ_di": {"type": "float", "low": 0.1, "high": 3.0, "log": True},
    },
    "uni": {
        "T_horizon": {
            "type": "categorical",
            "choices": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        },
        "max_active_occlusions": {
            "type": "categorical",
            "choices": ACTIVE_OCCLUSION_CHOICES,
        },
        "vref_scenario_softmax_kappa": {
            "type": "categorical",
            "choices": [0, 1, 5, 10, 20, 40, 60],
        },
        "k_theta_occ_uni_p": {
            "type": "float",
            "low": 0.75,
            "high": 5.0,
            "log": True,
        },
        "k_theta_occ_uni_d": {
            "type": "float",
            "low": 0.0,
            "high": 1.0,
            "step": 0.05,
        },
        "k_v_occ_uni_p": {
            "type": "float",
            "low": 0.5,
            "high": 1.5,
            "step": 0.05,
        },
        "k_v_occ_uni_d": {
            "type": "float",
            "low": 0.0,
            "high": 0.4,
            "step": 0.02,
        },
        "k_turn_boost_occ_uni": {
            "type": "float",
            "low": 0.0,
            "high": 3.0,
            "step": 0.1,
        },
        "turn_boost_angle_occ_uni": {
            "type": "float",
            "low": float(np.pi / 12.0),
            "high": float(np.pi / 2.0),
        },
    },
}

FIXED_OCBF_CONFIG: dict[str, Any] = {
    "obstacle_selection_angle_rad": float(2.0 * np.pi),
    "vref_scenario_weight_mode": OCBF_CROWD_VREF_SCENARIO_WEIGHT_MODE,
    "qp_failure_fallback_mode": OCBF_CROWD_QP_FAILURE_FALLBACK_MODE,
    "barrier_kappa": 10.0,
    "dt_backup": 0.05,
    "vref_front_mode_occ": "los",
    "vref_mode_occ": "strict",
    "occ_selection_mode": "h_tilde",
    "occ_rollout_mode": "common",
    "terminal_mode": "all",
    "terminal_residual_mode": "off",
    "terminal_slack_weight": 0.0,
    "enable_visible_hocbf_in_occ": True,
    "unicycle_v_min": float(load_shared_robot_parameters("uni")["v_min"]),
    "rho_T_di": "auto",
    "vref_tracking_mode_occ_uni": "gated",
}

_FAULT_MARKERS = (
    "Traceback",
    "[ERROR]",
    "[WARN]",
    "exception",
    "Exception",
    "nan",
    "NaN",
)


class _FaultOnlyStream:
    """Suppress routine per-step output while retaining diagnostic lines."""

    def __init__(self, real_stream):
        self._real = real_stream
        self._pending_newline = False

    def write(self, text):
        if any(marker in text for marker in _FAULT_MARKERS):
            self._pending_newline = True
            return self._real.write(text)
        if not text.strip() and self._pending_newline:
            self._pending_newline = False
            return self._real.write(text)
        return len(text)

    def flush(self):
        return self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


class TrialCaseError(RuntimeError):
    """A canonical benchmark case failed before producing valid metrics."""


class LatestStepMedianPruner(BasePruner):
    """Compare the current prefix against completed trials at the same prefix.

    Optuna's built-in median pruner compares a trial's best intermediate
    value. That is unsuitable for this cumulative count objective because a
    shorter prefix nearly always has a numerically smaller score. This pruner
    compares only equal-sized case prefixes.
    """

    def __init__(
        self,
        *,
        n_startup_trials: int,
        n_warmup_steps: int,
        interval_steps: int,
        n_min_trials: int,
    ):
        self.n_startup_trials = int(n_startup_trials)
        self.n_warmup_steps = int(n_warmup_steps)
        self.interval_steps = max(1, int(interval_steps))
        self.n_min_trials = int(n_min_trials)

    def prune(self, study, trial) -> bool:
        completed = study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        )
        if len(completed) < self.n_startup_trials:
            return False

        step = trial.last_step
        if step is None or step < self.n_warmup_steps:
            return False
        if (step - self.n_warmup_steps) % self.interval_steps != 0:
            return False

        current = trial.intermediate_values.get(step)
        if current is None:
            return False
        if not math.isfinite(float(current)):
            return True

        references = [
            float(other.intermediate_values[step])
            for other in completed
            if step in other.intermediate_values
            and math.isfinite(float(other.intermediate_values[step]))
        ]
        if len(references) < self.n_min_trials:
            return False

        median = float(np.median(references))
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            return float(current) > median
        return float(current) < median


def compatibility_fingerprint(config: dict[str, Any]) -> str:
    """Hash every setting that can affect trial values or artifact identity."""
    payload = {
        key: config.get(key)
        for key in COMPATIBILITY_CONFIG_KEYS
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_case_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Reject exception rows so broken configurations cannot rank as safe."""
    failures = [
        (int(row.get("idx", -1)), str(row.get("exception")))
        for row in rows
        if row.get("exception")
    ]
    if failures:
        detail = "; ".join(
            f"case {idx}: {message}" for idx, message in failures[:3]
        )
        if len(failures) > 3:
            detail += f"; and {len(failures) - 3} more"
        raise TrialCaseError(
            f"{len(failures)} canonical case(s) raised exceptions: {detail}"
        )


def sample_hyperparameters(
    trial: optuna.Trial,
    model: str,
    tuning_mode: str = "controller",
) -> dict[str, Any]:
    """Sample the model-specific, paper-safe search space."""
    if tuning_mode == "terminal_relax":
        if model != "di":
            raise ValueError("Terminal-relax tuning is defined only for DI.")
        return {
            name: trial.suggest_categorical(name, choices)
            for name, choices in TERMINAL_RELAX_GRID.items()
        }
    if model == "di":
        max_active = trial.suggest_categorical(
            "max_active_occlusions", ACTIVE_OCCLUSION_CHOICES
        )
        sampled = {
            "T_horizon": trial.suggest_categorical(
                "T_horizon", [0.25, 0.5, 0.75, 1.0]
            ),
            "max_active_occlusions": max_active,
            "k_p_occ_di": trial.suggest_float(
                "k_p_occ_di", 0.5, 3.0, log=True
            ),
            "k_d_occ_di": trial.suggest_float(
                "k_d_occ_di", 0.1, 3.0, log=True
            ),
            "vref_scenario_softmax_kappa": trial.suggest_categorical(
                "vref_scenario_softmax_kappa", [0, 5, 10, 20, 40, 60]
            ),
        }
        return sampled
    if model == "uni":
        max_active = trial.suggest_categorical(
            "max_active_occlusions", ACTIVE_OCCLUSION_CHOICES
        )
        sampled = {
            "T_horizon": trial.suggest_categorical(
                "T_horizon", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            ),
            "max_active_occlusions": max_active,
            "k_theta_occ_uni_p": trial.suggest_float(
                "k_theta_occ_uni_p", 0.75, 5.0, log=True
            ),
            "k_theta_occ_uni_d": trial.suggest_float(
                "k_theta_occ_uni_d", 0.0, 1.0, step=0.05
            ),
            "k_v_occ_uni_p": trial.suggest_float(
                "k_v_occ_uni_p", 0.5, 1.5, step=0.05
            ),
            "k_v_occ_uni_d": trial.suggest_float(
                "k_v_occ_uni_d", 0.0, 0.4, step=0.02
            ),
            "k_turn_boost_occ_uni": trial.suggest_float(
                "k_turn_boost_occ_uni", 0.0, 3.0, step=0.1
            ),
            "turn_boost_angle_occ_uni": trial.suggest_float(
                "turn_boost_angle_occ_uni",
                float(np.pi / 12.0),
                float(np.pi / 2.0),
            ),
            "vref_scenario_softmax_kappa": trial.suggest_categorical(
                "vref_scenario_softmax_kappa", [0, 1, 5, 10, 20, 40, 60]
            ),
        }
        return sampled
    raise ValueError(f"Unsupported model: {model}")


def build_controller_overrides(
    model: str,
    sampled: dict[str, Any],
    tuning_mode: str = "controller",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert sampled parameters into canonical benchmark overrides."""
    if tuning_mode == "terminal_relax":
        if model != "di":
            raise ValueError("Terminal-relax tuning is defined only for DI.")
        backup = apply_crowd_ocbf_defaults(
            {
                "terminal_slack_weight": float(
                    sampled["terminal_slack_weight"]
                ),
                "terminal_slack_max": float(sampled["terminal_slack_max"]),
                "obs_hocbf_slack_max": 0.0,
                "occ_rollout_slack_max": 0.0,
            }
        )
        return backup, {}

    backup = {
        "T_horizon": float(sampled["T_horizon"]),
        "dt_backup": FIXED_OCBF_CONFIG["dt_backup"],
        "vref_scenario_softmax_kappa": float(
            sampled["vref_scenario_softmax_kappa"]
        ),
        "max_active_occlusions": int(sampled["max_active_occlusions"]),
        "vref_scenario_weight_mode": FIXED_OCBF_CONFIG[
            "vref_scenario_weight_mode"
        ],
        "qp_failure_fallback_mode": FIXED_OCBF_CONFIG[
            "qp_failure_fallback_mode"
        ],
        "vref_front_mode_occ": FIXED_OCBF_CONFIG["vref_front_mode_occ"],
        "vref_mode_occ": FIXED_OCBF_CONFIG["vref_mode_occ"],
        "occ_selection_mode": FIXED_OCBF_CONFIG["occ_selection_mode"],
        "occ_rollout_mode": FIXED_OCBF_CONFIG["occ_rollout_mode"],
        "terminal_mode": FIXED_OCBF_CONFIG["terminal_mode"],
        "terminal_residual_mode": FIXED_OCBF_CONFIG["terminal_residual_mode"],
        "terminal_slack_weight": FIXED_OCBF_CONFIG["terminal_slack_weight"],
    }

    if model == "di":
        backup.update(
            {
                "rho_T": FIXED_OCBF_CONFIG["rho_T_di"],
                "k_p_occ_di": float(sampled["k_p_occ_di"]),
                "k_d_occ_di": float(sampled["k_d_occ_di"]),
            }
        )
    elif model == "uni":
        backup.update(
            {
                "rho_T": 0.0,
                "vref_tracking_mode_occ_uni": FIXED_OCBF_CONFIG[
                    "vref_tracking_mode_occ_uni"
                ],
                "k_theta_occ_uni_p": float(sampled["k_theta_occ_uni_p"]),
                "k_theta_occ_uni_d": float(sampled["k_theta_occ_uni_d"]),
                "k_v_occ_uni_p": float(sampled["k_v_occ_uni_p"]),
                "k_v_occ_uni_d": float(sampled["k_v_occ_uni_d"]),
                "k_turn_boost_occ_uni": float(
                    sampled["k_turn_boost_occ_uni"]
                ),
                "turn_boost_angle_occ_uni": float(
                    sampled["turn_boost_angle_occ_uni"]
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    backup = apply_crowd_ocbf_defaults(backup)
    robot = {"occ_kappa": FIXED_OCBF_CONFIG["barrier_kappa"]}
    if model == "uni":
        robot["v_min"] = FIXED_OCBF_CONFIG["unicycle_v_min"]
    return backup, robot


def lexicographic_score(
    *,
    collision_count: int,
    success_count: int,
    evaluated_count: int,
    avg_compute_time_ms: float | None,
) -> float:
    """Return collision-first, success-second, compute-time-third score."""
    if evaluated_count <= 0:
        raise ValueError("evaluated_count must be positive")
    if collision_count < 0 or success_count < 0:
        raise ValueError("counts must be nonnegative")
    if collision_count + success_count > evaluated_count:
        raise ValueError("outcome counts exceed evaluated_count")

    failure_count = evaluated_count - success_count
    if avg_compute_time_ms is None or not math.isfinite(avg_compute_time_ms):
        time_term = COMPUTE_TIME_CAP_MS
    else:
        time_term = min(max(float(avg_compute_time_ms), 0.0), COMPUTE_TIME_CAP_MS)
    return (
        float(collision_count) * COLLISION_WEIGHT
        + float(failure_count) * FAILURE_WEIGHT
        + time_term
    )


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows using the benchmark's three-way classification."""
    row_list = list(rows)
    counts = {"success": 0, "collision": 0, "infeasible": 0}
    compute_times: list[float] = []
    wall_times: list[float] = []
    sim_times: list[float] = []
    exception_count = 0

    for row in row_list:
        outcome = str(row.get("class_3way", "infeasible"))
        if outcome not in counts:
            outcome = "infeasible"
        counts[outcome] += 1
        if row.get("exception"):
            exception_count += 1
        for key, target in (
            ("avg_compute_time_ms", compute_times),
            ("case_wall_time_s", wall_times),
            ("total_sim_time", sim_times),
        ):
            value = row.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                target.append(number)

    total = len(row_list)
    avg_compute = float(np.mean(compute_times)) if compute_times else None
    summary = {
        "counts": {**counts, "total": total},
        "rates": {
            key: (float(value) / total if total else 0.0)
            for key, value in counts.items()
        },
        "averages": {
            "avg_compute_time_ms": avg_compute,
            "avg_case_wall_time_s": (
                float(np.mean(wall_times)) if wall_times else None
            ),
            "avg_total_sim_time_s": (
                float(np.mean(sim_times)) if sim_times else None
            ),
        },
        "exception_count": exception_count,
    }
    if total:
        summary["objective"] = lexicographic_score(
            collision_count=counts["collision"],
            success_count=counts["success"],
            evaluated_count=total,
            avg_compute_time_ms=avg_compute,
        )
    else:
        summary["objective"] = None
    return summary


def _run_case(case_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Spawn-worker entrypoint for one canonical benchmark case."""
    with contextlib.redirect_stdout(_FaultOnlyStream(sys.stdout)):
        return _run_one_idx_job(**case_kwargs)


def _case_kwargs(
    *,
    model: str,
    idx: int,
    n_rand: int,
    tf: float,
    seed: int,
    backup_cbf_overrides: dict[str, Any],
    robot_spec_overrides: dict[str, Any],
    run_label: str,
    baseline_alias: str = "occlusion_cbf",
) -> dict[str, Any]:
    return {
        "baseline_alias": baseline_alias,
        "scenario_name": "crowd",
        "controller_pos": CROWD_BASELINE_MAP[baseline_alias],
        "model": model,
        "seed": int(seed),
        "idx": int(idx),
        "n_rand": int(n_rand),
        "tf": float(tf),
        "wmax": "default",
        "oa_allow_solver_fallback": None,
        "oa_dynamic_occluders": None,
        "oa_dsafe": None,
        "oa_visible_reach_mode": None,
        "oa_use_nominal_tracking_cost": None,
        "oa_dt": None,
        "occ_visible_scale": None,
        "occ_enable_visible_hocbf": True,
        "crowd_mode": CROWD_BENCHMARK_DEFAULTS["crowd_mode"],
        "forced_events": CROWD_BENCHMARK_DEFAULTS["forced_events"],
        "forced_bg_rand": None,
        "forced_hidden_speed": CROWD_BENCHMARK_DEFAULTS[
            "forced_hidden_speed"
        ],
        "forced_occluder_radius_min": CROWD_BENCHMARK_DEFAULTS[
            "forced_occluder_radius_min"
        ],
        "forced_occluder_radius_max": CROWD_BENCHMARK_DEFAULTS[
            "forced_occluder_radius_max"
        ],
        "forced_validate_occlusion": CROWD_BENCHMARK_DEFAULTS[
            "forced_validate_occlusion"
        ],
        "forced_require_corridor_conflict": CROWD_BENCHMARK_DEFAULTS[
            "forced_require_corridor_conflict"
        ],
        "backup_cbf_overrides": backup_cbf_overrides,
        "robot_spec_overrides": robot_spec_overrides,
        "run_label": run_label,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["idx"])))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _trial_artifacts(
    *,
    output_dir: Path,
    trial: optuna.Trial,
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    state: str,
    fixed_run_config: dict[str, Any],
) -> tuple[Path, Path]:
    rows_path = output_dir / "trials" / f"trial_{trial.number:04d}_rows.csv"
    summary_path = (
        output_dir / "trials" / f"trial_{trial.number:04d}_summary.json"
    )
    _write_rows(rows_path, rows)
    _write_json(
        summary_path,
        {
            "trial": trial.number,
            "state": state,
            "parameters": params,
            "fixed_config": fixed_run_config,
            "metrics": summary,
            "evaluated_indices": sorted(int(row["idx"]) for row in rows),
            "artifacts": {
                "rows_csv": str(rows_path),
                "summary_json": str(summary_path),
            },
        },
    )
    return rows_path, summary_path


def _set_trial_attrs(
    trial: optuna.Trial,
    summary: dict[str, Any],
    *,
    rows_path: Path,
    summary_path: Path,
    state: str,
) -> None:
    trial.set_user_attr("state", state)
    trial.set_user_attr("cases_evaluated", summary["counts"]["total"])
    for key in ("success", "collision", "infeasible"):
        trial.set_user_attr(f"{key}_count", summary["counts"][key])
        trial.set_user_attr(f"{key}_rate", summary["rates"][key])
    for key, value in summary["averages"].items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("exception_count", summary["exception_count"])
    trial.set_user_attr("rows_csv", str(rows_path))
    trial.set_user_attr("summary_json", str(summary_path))


def _wandb_trial_log(
    wandb_run,
    *,
    trial_number: int,
    params: dict[str, Any],
    summary: dict[str, Any],
    state: str,
    final: bool,
) -> None:
    if wandb_run is None:
        return
    payload = {
        "trial/number": int(trial_number),
        "trial/state": state,
        "trial/final": int(bool(final)),
        "trial/cases_evaluated": summary["counts"]["total"],
        "objective/lexicographic": summary["objective"],
        "metrics/success_count": summary["counts"]["success"],
        "metrics/collision_count": summary["counts"]["collision"],
        "metrics/infeasible_count": summary["counts"]["infeasible"],
        "metrics/success_rate": summary["rates"]["success"],
        "metrics/collision_rate": summary["rates"]["collision"],
        "metrics/infeasible_rate": summary["rates"]["infeasible"],
        "metrics/avg_compute_time_ms": summary["averages"][
            "avg_compute_time_ms"
        ],
        "metrics/avg_case_wall_time_s": summary["averages"][
            "avg_case_wall_time_s"
        ],
        "metrics/avg_total_sim_time_s": summary["averages"][
            "avg_total_sim_time_s"
        ],
        "metrics/exception_count": summary["exception_count"],
        **{f"param/{key}": value for key, value in params.items()},
    }
    wandb_run.log(payload)


def make_objective(
    *,
    model: str,
    indices: list[int],
    n_rand: int,
    tf: float,
    seed: int,
    batch_size: int,
    pool: ProcessPoolExecutor,
    output_dir: Path,
    wandb_run,
    fixed_run_config: dict[str, Any],
    tuning_mode: str = "controller",
):
    """Build the sequential-trial objective around a shared case process pool."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, model, tuning_mode)
        trial.set_user_attr("effective_parameters", params)
        backup_overrides, robot_overrides = build_controller_overrides(
            model, params, tuning_mode
        )
        rows: list[dict[str, Any]] = []
        run_label = f"optuna_trial_{trial.number:04d}"
        baseline_alias = (
            "occlusion_cbf_terminal_relax"
            if tuning_mode == "terminal_relax"
            else "occlusion_cbf"
        )

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            futures = {
                pool.submit(
                    _run_case,
                    _case_kwargs(
                        model=model,
                        idx=idx,
                        n_rand=n_rand,
                        tf=tf,
                        seed=seed,
                        backup_cbf_overrides=backup_overrides,
                        robot_spec_overrides=robot_overrides,
                        run_label=run_label,
                        baseline_alias=baseline_alias,
                    ),
                ): idx
                for idx in batch_indices
            }
            batch_rows = [future.result() for future in as_completed(futures)]
            batch_rows.sort(key=lambda row: int(row["idx"]))
            rows.extend(batch_rows)

            partial = summarize_rows(rows)
            try:
                validate_case_rows(batch_rows)
            except TrialCaseError:
                rows_path, summary_path = _trial_artifacts(
                    output_dir=output_dir,
                    trial=trial,
                    params=params,
                    rows=rows,
                    summary=partial,
                    state="failed",
                    fixed_run_config=fixed_run_config,
                )
                _set_trial_attrs(
                    trial,
                    partial,
                    rows_path=rows_path,
                    summary_path=summary_path,
                    state="failed",
                )
                _wandb_trial_log(
                    wandb_run,
                    trial_number=trial.number,
                    params=params,
                    summary=partial,
                    state="failed",
                    final=True,
                )
                print(
                    f"[trial {trial.number:03d}] rejected: "
                    f"{partial['exception_count']} case exception(s)",
                    flush=True,
                )
                raise

            trial.report(float(partial["objective"]), step=len(rows))
            _wandb_trial_log(
                wandb_run,
                trial_number=trial.number,
                params=params,
                summary=partial,
                state="running",
                final=False,
            )
            print(
                f"[trial {trial.number:03d}] cases={len(rows):03d}/{len(indices):03d} "
                f"collision={partial['counts']['collision']} "
                f"success={partial['counts']['success']} "
                f"infeasible={partial['counts']['infeasible']} "
                f"compute_ms={partial['averages']['avg_compute_time_ms']} "
                f"score={partial['objective']:.3f}",
                flush=True,
            )

            if len(rows) < len(indices) and trial.should_prune():
                rows_path, summary_path = _trial_artifacts(
                    output_dir=output_dir,
                    trial=trial,
                    params=params,
                    rows=rows,
                    summary=partial,
                    state="pruned",
                    fixed_run_config=fixed_run_config,
                )
                _set_trial_attrs(
                    trial,
                    partial,
                    rows_path=rows_path,
                    summary_path=summary_path,
                    state="pruned",
                )
                _wandb_trial_log(
                    wandb_run,
                    trial_number=trial.number,
                    params=params,
                    summary=partial,
                    state="pruned",
                    final=True,
                )
                raise optuna.TrialPruned(
                    f"Median-pruned after {len(rows)} canonical cases"
                )

        final_summary = summarize_rows(rows)
        rows_path, summary_path = _trial_artifacts(
            output_dir=output_dir,
            trial=trial,
            params=params,
            rows=rows,
            summary=final_summary,
            state="complete",
            fixed_run_config=fixed_run_config,
        )
        _set_trial_attrs(
            trial,
            final_summary,
            rows_path=rows_path,
            summary_path=summary_path,
            state="complete",
        )
        _wandb_trial_log(
            wandb_run,
            trial_number=trial.number,
            params=params,
            summary=final_summary,
            state="complete",
            final=True,
        )
        return float(final_summary["objective"])

    return objective


def _serialize_study(study: optuna.Study) -> dict[str, Any]:
    trials = []
    for trial in study.trials:
        trials.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "datetime_start": (
                    trial.datetime_start.isoformat()
                    if trial.datetime_start
                    else None
                ),
                "datetime_complete": (
                    trial.datetime_complete.isoformat()
                    if trial.datetime_complete
                    else None
                ),
            }
        )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best = None
    if completed:
        best_trial = study.best_trial
        best = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "effective_params": best_trial.user_attrs.get(
                "effective_parameters",
                best_trial.params,
            ),
            "user_attrs": best_trial.user_attrs,
        }
    return {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_trial": best,
        "trials": trials,
    }


def _save_study_snapshot(
    study: optuna.Study,
    output_dir: Path,
    wandb_run,
) -> None:
    serialized = _serialize_study(study)
    _write_json(output_dir / "study_snapshot.json", serialized)
    if serialized["best_trial"] is not None:
        _write_json(output_dir / "best.json", serialized["best_trial"])
        if wandb_run is not None:
            best = serialized["best_trial"]
            wandb_run.summary.update(
                {
                    "best/trial": best["number"],
                    "best/objective": best["value"],
                    "best/success_rate": best["user_attrs"].get(
                        "success_rate"
                    ),
                    "best/collision_rate": best["user_attrs"].get(
                        "collision_rate"
                    ),
                    "best/infeasible_rate": best["user_attrs"].get(
                        "infeasible_rate"
                    ),
                    "best/avg_compute_time_ms": best["user_attrs"].get(
                        "avg_compute_time_ms"
                    ),
                    **{
                        f"best/param/{key}": value
                        for key, value in best["effective_params"].items()
                    },
                }
            )


def _log_results_artifact(
    study: optuna.Study,
    output_dir: Path,
    wandb_run,
) -> None:
    """Upload the exact local result tables at normal study shutdown."""
    if wandb_run is None:
        return
    try:
        import wandb

        artifact = wandb.Artifact(
            name=f"{study.study_name}-results",
            type="ocbf-optuna-results",
            metadata={
                "study_name": study.study_name,
                "trial_count": len(study.trials),
            },
        )
        for filename in ("run_config.json", "study_snapshot.json", "best.json"):
            path = output_dir / filename
            if path.exists():
                artifact.add_file(str(path), name=filename)
        trials_dir = output_dir / "trials"
        if trials_dir.exists():
            artifact.add_dir(str(trials_dir), name="trials")
        wandb_run.log_artifact(artifact)
    except Exception as exc:
        print(f"[wandb] result artifact upload failed: {exc}", flush=True)


def _git_value(args: list[str], default: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return default
    return result.stdout.strip() or default


def _wandb_init(
    *,
    enabled: bool,
    output_dir: Path,
    project: str,
    group: str,
    name: str,
    config: dict[str, Any],
):
    if not enabled:
        return None
    import wandb

    run_id_path = output_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n")
    return wandb.init(
        project=project,
        group=group,
        name=name,
        id=run_id,
        resume="allow",
        job_type="ocbf_tuning",
        config=config,
        tags=["occlusion_cbf", "crowd", str(config["model"])],
        dir=str(output_dir),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune Occlusion-CBF on the canonical crowd benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=["di", "uni"])
    parser.add_argument(
        "--tuning-mode",
        choices=TUNING_MODES,
        default="controller",
        help=(
            "Tune the full controller profile or only the DI terminal-relax "
            "weight and cap while loading the committed hard-OCBF profile."
        ),
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument(
        "--n-rand",
        type=int,
        default=None,
        help="Defaults to 50 for DI and 30 for Uni.",
    )
    parser.add_argument(
        "--seed", type=int, default=CROWD_BENCHMARK_DEFAULTS["seed"]
    )
    parser.add_argument(
        "--idx-start",
        type=int,
        default=CROWD_BENCHMARK_DEFAULTS["idx_start"],
    )
    parser.add_argument(
        "--idx-end",
        type=int,
        default=CROWD_BENCHMARK_DEFAULTS["idx_end"],
    )
    parser.add_argument(
        "--tf", type=float, default=CROWD_BENCHMARK_DEFAULTS["tf"]
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--pruner-startup-trials", type=int, default=8)
    parser.add_argument("--pruner-warmup-cases", type=int, default=20)
    parser.add_argument("--pruner-interval-cases", type=int, default=10)
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Use only for smoke tests or diagnostics.",
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", default=WANDB_PROJECT_DEFAULT
    )
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.idx_end < args.idx_start:
        raise ValueError("--idx-end must be >= --idx-start")
    if args.trials <= 0 or args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("trials, workers, and batch-size must be positive")
    if args.tuning_mode == "terminal_relax" and args.model != "di":
        raise ValueError("--tuning-mode terminal_relax requires --model di")

    model_cfg = MODEL_DEFAULTS[args.model]
    n_rand = (
        int(model_cfg["n_rand"])
        if args.n_rand is None
        else int(args.n_rand)
    )
    indices = list(range(int(args.idx_start), int(args.idx_end) + 1))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_prefix = (
        "ocbf_terminal_relax"
        if args.tuning_mode == "terminal_relax"
        else "ocbf_crowd"
    )
    study_name = args.study_name or (
        f"{study_prefix}_{args.model}_n{n_rand}_{timestamp}"
    )
    wandb_group = args.wandb_group or f"ocbf_crowd_{timestamp}"
    wandb_name = args.wandb_name or f"{args.model}_n{n_rand}"
    source_commit = os.environ.get(
        "OCBF_SOURCE_COMMIT",
        _git_value(["rev-parse", "HEAD"], "unknown"),
    )
    source_fingerprint = os.environ.get(
        "OCBF_SOURCE_FINGERPRINT",
        "unavailable",
    )

    fixed_run_config = {
        "scenario": "crowd",
        "baseline": (
            "occlusion_cbf_terminal_relax"
            if args.tuning_mode == "terminal_relax"
            else "occlusion_cbf"
        ),
        "model": args.model,
        "model_name": model_cfg["display_name"],
        "study_name": study_name,
        "seed": int(args.seed),
        "indices": indices,
        "n_rand": n_rand,
        "tf": float(args.tf),
        "workers": int(args.workers),
        "batch_size": int(args.batch_size),
        "trials": int(args.trials),
        "timeout_s": args.timeout,
        "objective_priority": [
            "collision_count:minimize",
            "success_count:maximize",
            "avg_compute_time_ms:minimize",
        ],
        "objective_weights": {
            "collision": COLLISION_WEIGHT,
            "non_success": FAILURE_WEIGHT,
            "compute_time_cap_ms": COMPUTE_TIME_CAP_MS,
        },
        "benchmark_protocol": {
            "crowd_mode": CROWD_BENCHMARK_DEFAULTS["crowd_mode"],
            "forced_events": CROWD_BENCHMARK_DEFAULTS["forced_events"],
            "forced_hidden_speed": CROWD_BENCHMARK_DEFAULTS[
                "forced_hidden_speed"
            ],
            "forced_occluder_radius_min": CROWD_BENCHMARK_DEFAULTS[
                "forced_occluder_radius_min"
            ],
            "forced_occluder_radius_max": CROWD_BENCHMARK_DEFAULTS[
                "forced_occluder_radius_max"
            ],
            "forced_validate_occlusion": CROWD_BENCHMARK_DEFAULTS[
                "forced_validate_occlusion"
            ],
            "forced_require_corridor_conflict": CROWD_BENCHMARK_DEFAULTS[
                "forced_require_corridor_conflict"
            ],
            "enable_visible_hocbf_in_occ": True,
        },
        "search_space": (
            TERMINAL_RELAX_GRID
            if args.tuning_mode == "terminal_relax"
            else SEARCH_SPACE[args.model]
        ),
        "fixed_ocbf": FIXED_OCBF_CONFIG,
        "sampler": {
            "name": (
                "GridSampler"
                if args.tuning_mode == "terminal_relax"
                else "TPESampler"
            ),
            "seed": int(args.sampler_seed),
        },
        "pruner": {
            "name": (
                "NopPruner"
                if args.disable_pruning
                else (
                    "LatestStepMedianPruner"
                    if args.tuning_mode == "terminal_relax"
                    else "MedianPruner"
                )
            ),
            "startup_trials": int(args.pruner_startup_trials),
            "warmup_cases": int(args.pruner_warmup_cases),
            "interval_cases": int(args.pruner_interval_cases),
            "min_trials": 4,
        },
        "source_commit": source_commit,
        "source_fingerprint": source_fingerprint,
        "tuning_mode": args.tuning_mode,
        "source_dir": os.environ.get("OCBF_SOURCE_DIR", str(Path.cwd())),
        "command": sys.argv,
    }
    config_fingerprint = compatibility_fingerprint(fixed_run_config)
    fixed_run_config["compatibility_fingerprint"] = config_fingerprint
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text())
        existing_fingerprint = existing_config.get(
            "compatibility_fingerprint"
        )
        if existing_fingerprint != config_fingerprint:
            raise RuntimeError(
                "Refusing to mix incompatible tuning runs in "
                f"{output_dir}: existing fingerprint "
                f"{existing_fingerprint!r}, requested {config_fingerprint!r}"
            )

    if args.tuning_mode == "terminal_relax":
        sampler = GridSampler(
            TERMINAL_RELAX_GRID,
            seed=int(args.sampler_seed),
        )
    else:
        sampler = TPESampler(seed=int(args.sampler_seed))
    if args.disable_pruning:
        pruner = NopPruner()
    elif args.tuning_mode == "terminal_relax":
        pruner = LatestStepMedianPruner(
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=int(args.pruner_warmup_cases),
            interval_steps=int(args.pruner_interval_cases),
            n_min_trials=4,
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=int(args.pruner_warmup_cases),
            interval_steps=int(args.pruner_interval_cases),
            n_min_trials=4,
        )

    storage_path = output_dir / "study.db"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60,
        grace_period=300,
        heartbeat_stale_trial_callback=(
            optuna.storages.RetryHeartbeatStaleTrialCallback(
                max_retry=1,
                inherit_intermediate_values=True,
            )
        ),
    )
    existing_studies = {
        summary.study_name
        for summary in optuna.study.get_all_study_summaries(storage=storage)
    }
    if existing_studies and study_name not in existing_studies:
        raise RuntimeError(
            "Refusing to create a second study in an existing output "
            f"directory. Found {sorted(existing_studies)!r}; "
            f"requested {study_name!r}."
        )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    stored_fingerprint = study.user_attrs.get("compatibility_fingerprint")
    if stored_fingerprint is None and study.trials:
        raise RuntimeError(
            "Existing study has trials but no compatibility fingerprint; "
            "refusing an unsafe resume."
        )
    if (
        stored_fingerprint is not None
        and stored_fingerprint != config_fingerprint
    ):
        raise RuntimeError(
            "Existing Optuna study has a different compatibility "
            f"fingerprint: {stored_fingerprint!r} != "
            f"{config_fingerprint!r}"
        )
    study.set_user_attr("compatibility_fingerprint", config_fingerprint)
    study.set_user_attr("source_fingerprint", source_fingerprint)
    _write_json(run_config_path, fixed_run_config)

    wandb_run = _wandb_init(
        enabled=bool(args.wandb),
        output_dir=output_dir,
        project=str(args.wandb_project),
        group=str(wandb_group),
        name=str(wandb_name),
        config=fixed_run_config,
    )
    if len(study.trials) == 0 and args.tuning_mode == "controller":
        for incumbent in model_cfg["incumbents"]:
            study.enqueue_trial(dict(incumbent))

    print(
        f"[study] name={study_name} model={args.model} n_rand={n_rand} "
        f"cases={args.idx_start}..{args.idx_end} workers={args.workers} "
        f"trials={args.trials} output={output_dir}",
        flush=True,
    )
    print(
        f"[study] fixed angle=2pi weight_mode="
        f"{OCBF_CROWD_VREF_SCENARIO_WEIGHT_MODE}",
        flush=True,
    )

    def callback(active_study: optuna.Study, _trial) -> None:
        _save_study_snapshot(active_study, output_dir, wandb_run)

    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    finished_count = sum(
        trial.state in finished_states for trial in study.trials
    )
    remaining_trials = max(0, int(args.trials) - finished_count)

    mp_context = mp.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=int(args.workers),
        mp_context=mp_context,
    )
    objective = make_objective(
        model=args.model,
        indices=indices,
        n_rand=n_rand,
        tf=float(args.tf),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        pool=pool,
        output_dir=output_dir,
        wandb_run=wandb_run,
        fixed_run_config=fixed_run_config,
        tuning_mode=args.tuning_mode,
    )

    try:
        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=args.timeout,
                n_jobs=1,
                callbacks=[callback],
                show_progress_bar=False,
                catch=(TrialCaseError,),
            )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        _save_study_snapshot(study, output_dir, wandb_run)
        _log_results_artifact(study, output_dir, wandb_run)
        if wandb_run is not None:
            wandb_run.finish()

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    print(
        f"[study] finished completed={len(completed)} total={len(study.trials)}",
        flush=True,
    )
    if completed:
        print(
            f"[study] best trial={study.best_trial.number} "
            f"score={study.best_value:.3f} params={study.best_params}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
