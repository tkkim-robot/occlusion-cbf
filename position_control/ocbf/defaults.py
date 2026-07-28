"""Shared defaults and CLI choices for Occlusion-CBF experiments.

The controller classes intentionally keep model-agnostic defaults conservative.
Scenario-specific wrappers such as ``examples/test_crowd.py`` can apply the
paper benchmark defaults through the helpers in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml


OCBF_VREF_FRONT_MODES = ("default", "los")
OCBF_DEFAULT_VREF_FRONT_MODE = "los"

OCBF_VREF_TRACKING_MODES = ("soft", "strict")
OCBF_DEFAULT_VREF_TRACKING_MODE = "strict"

OCBF_DEFAULT_BARRIER_KAPPA = 10.0

OCBF_VREF_SCENARIO_WEIGHT_MODES = ("barrier_expand", "barrier_unexpand")
OCBF_DEFAULT_VREF_SCENARIO_WEIGHT_MODE = "barrier_expand"

OCBF_SELECTION_MODES = ("h_tilde", "distance")
OCBF_DEFAULT_SELECTION_MODE = "h_tilde"

OCBF_QP_FAILURE_FALLBACK_MODES = ("strict", "state_safe", "always")
OCBF_DEFAULT_QP_FAILURE_FALLBACK_MODE = "state_safe"
OCBF_CROWD_QP_FAILURE_FALLBACK_MODE = "state_safe"
OCBF_CROWD_VREF_SCENARIO_WEIGHT_MODE = "barrier_unexpand"

OCBF_TERMINAL_MODES = ("all", "topm", "dominant", "none")
OCBF_DEFAULT_TERMINAL_MODE = "all"

OCBF_TERMINAL_RESIDUAL_MODES = ("off", "visibility_intersection", "visibility_only")
OCBF_DEFAULT_TERMINAL_RESIDUAL_MODE = "off"

OCBF_ROLLOUT_MODES = ("common", "per_scenario")
OCBF_DEFAULT_ROLLOUT_MODE = "common"

CROWD_ENABLE_VISIBLE_HOCBF_DEFAULT = True

OCBF_PARAMETER_DIR = Path(__file__).resolve().parent / "config"
OCBF_BEST_PARAMETER_FILES = {
    "DoubleIntegrator2D": OCBF_PARAMETER_DIR / "occlusion_cbf_di_params.yaml",
    "Unicycle2D": OCBF_PARAMETER_DIR / "occlusion_cbf_unicycle_params.yaml",
}
SHARED_ROBOT_PARAMETER_KEYS = frozenset({"fov_angle", "v_min"})

_OCBF_MODEL_ALIASES = {
    "di": "DoubleIntegrator2D",
    "doubleintegrator": "DoubleIntegrator2D",
    "doubleintegrator2d": "DoubleIntegrator2D",
    "uni": "Unicycle2D",
    "un": "Unicycle2D",
    "unicycle": "Unicycle2D",
    "unicycle2d": "Unicycle2D",
}


def normalize_choice(value, choices, default):
    """Return a normalized string choice, falling back to ``default``."""
    mode = str(value).strip().lower()
    return mode if mode in choices else default


def default_visible_hocbf_for_scenario(scenario_label):
    """Visible-obstacle HOCBF is part of the canonical crowd benchmark default."""
    scenario = str(scenario_label).strip().lower()
    return CROWD_ENABLE_VISIBLE_HOCBF_DEFAULT if scenario in {"crowd", "crowd2", "test_crowd", "test_crowd2"} else False


def apply_crowd_ocbf_defaults(backup_cbf_overrides):
    """Apply crowd-specific OCBF defaults without overwriting explicit CLI values."""
    cfg = dict(backup_cbf_overrides or {})
    cfg.setdefault("qp_failure_fallback_mode", OCBF_CROWD_QP_FAILURE_FALLBACK_MODE)
    cfg.setdefault(
        "vref_scenario_weight_mode",
        OCBF_CROWD_VREF_SCENARIO_WEIGHT_MODE,
    )
    return cfg


def _canonical_tuned_model(model_key):
    normalized = (
        str(model_key)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    return _OCBF_MODEL_ALIASES.get(normalized)


@lru_cache(maxsize=None)
def _load_ocbf_best_parameters_cached(model_name):
    path = OCBF_BEST_PARAMETER_FILES[model_name]
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing Occlusion-CBF parameter file for {model_name}: {path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in Occlusion-CBF parameter file {path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Occlusion-CBF parameter file must contain a mapping: {path}")
    if payload.get("model") != model_name:
        raise ValueError(
            f"Occlusion-CBF parameter file {path} declares model "
            f"{payload.get('model')!r}, expected {model_name!r}"
        )

    normalized = {"model": model_name}
    for section in ("shared_robot_spec", "backup_cbf", "robot_spec"):
        values = payload.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(
                f"Occlusion-CBF parameter file section `{section}` must be a mapping: {path}"
            )
        normalized[section] = dict(values)
    unsupported_shared_keys = (
        set(normalized["shared_robot_spec"])
        - SHARED_ROBOT_PARAMETER_KEYS
    )
    if unsupported_shared_keys:
        raise ValueError(
            "Unsupported shared robot parameter(s) in "
            f"{path}: {sorted(unsupported_shared_keys)}"
        )
    duplicate_robot_keys = (
        set(normalized["shared_robot_spec"])
        & set(normalized["robot_spec"])
    )
    if duplicate_robot_keys:
        raise ValueError(
            "Robot parameter(s) must be declared in only one profile section "
            f"in {path}: {sorted(duplicate_robot_keys)}"
        )
    return normalized


def load_ocbf_best_parameters(model_key):
    """Load a copy of the tuned DI or Unicycle parameters.

    Dynamic Unicycle has no tuned study yet, so unsupported dynamics return
    ``None`` and retain their existing defaults.
    """
    model_name = _canonical_tuned_model(model_key)
    if model_name is None:
        return None
    return deepcopy(_load_ocbf_best_parameters_cached(model_name))


def load_shared_robot_parameters(model_key):
    """Load canonical model-wide values shared by all comparison methods."""
    parameters = load_ocbf_best_parameters(model_key)
    if parameters is None:
        return None
    return deepcopy(parameters["shared_robot_spec"])


def merge_shared_robot_parameters(
    model_key,
    *,
    robot_defaults=None,
    robot_overrides=None,
):
    """Merge shared model values without leaking OCBF-only configuration.

    Precedence is scenario defaults, explicitly shared YAML values, then
    explicit command-line or programmatic overrides.
    """
    robot_cfg = dict(robot_defaults or {})
    shared_parameters = load_shared_robot_parameters(model_key)
    if shared_parameters:
        robot_cfg.update(shared_parameters)
    if robot_overrides:
        robot_cfg.update(dict(robot_overrides))
    return robot_cfg


def merge_ocbf_best_parameters(
    model_key,
    *,
    backup_defaults=None,
    robot_defaults=None,
    backup_overrides=None,
    robot_overrides=None,
):
    """Merge OCBF configuration with explicit values taking precedence.

    Precedence is scenario defaults, tuned YAML parameters, then explicit
    command-line or programmatic overrides.
    """
    backup_cfg = dict(backup_defaults or {})
    robot_cfg = dict(robot_defaults or {})
    parameters = load_ocbf_best_parameters(model_key)
    if parameters is not None:
        backup_cfg.update(parameters["backup_cbf"])
        robot_cfg.update(parameters["shared_robot_spec"])
        robot_cfg.update(parameters["robot_spec"])
    if backup_overrides:
        backup_cfg.update(dict(backup_overrides))
    if robot_overrides:
        robot_cfg.update(dict(robot_overrides))
    return backup_cfg, robot_cfg


def apply_ocbf_best_parameters(robot_spec, *, overwrite=False):
    """Apply tuned parameters to a robot specification before construction.

    The shared tracking controller uses this as a safety net for future entry
    points. Maintained scenarios use :func:`merge_ocbf_best_parameters` so
    their built-in defaults are replaced while explicit user overrides remain
    authoritative.
    """
    parameters = load_ocbf_best_parameters(robot_spec.get("model"))
    if parameters is None:
        return robot_spec

    for section in ("shared_robot_spec", "robot_spec"):
        for key, value in parameters[section].items():
            if overwrite or key not in robot_spec:
                robot_spec[key] = deepcopy(value)

    backup_cfg = robot_spec.setdefault("backup_cbf", {})
    if not isinstance(backup_cfg, dict):
        raise TypeError("robot_spec['backup_cbf'] must be a dictionary")
    for key, value in parameters["backup_cbf"].items():
        if overwrite or key not in backup_cfg:
            backup_cfg[key] = deepcopy(value)
    return robot_spec


# Compatibility aliases retained for external scripts using the former Crowd2 names.
OCBF_CROWD2_QP_FAILURE_FALLBACK_MODE = OCBF_CROWD_QP_FAILURE_FALLBACK_MODE
CROWD2_ENABLE_VISIBLE_HOCBF_DEFAULT = CROWD_ENABLE_VISIBLE_HOCBF_DEFAULT
apply_crowd2_ocbf_defaults = apply_crowd_ocbf_defaults
