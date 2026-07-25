"""Shared defaults and CLI choices for Occlusion-CBF experiments.

The controller classes intentionally keep model-agnostic defaults conservative.
Scenario-specific wrappers such as ``examples/test_crowd.py`` can apply the
paper benchmark defaults through the helpers in this module.
"""

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


# Compatibility aliases retained for external scripts using the former Crowd2 names.
OCBF_CROWD2_QP_FAILURE_FALLBACK_MODE = OCBF_CROWD_QP_FAILURE_FALLBACK_MODE
CROWD2_ENABLE_VISIBLE_HOCBF_DEFAULT = CROWD_ENABLE_VISIBLE_HOCBF_DEFAULT
apply_crowd2_ocbf_defaults = apply_crowd_ocbf_defaults
