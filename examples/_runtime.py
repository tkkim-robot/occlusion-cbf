from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)


def ensure_repo_root() -> Path:
    if REPO_ROOT_STR in sys.path:
        sys.path.remove(REPO_ROOT_STR)
    sys.path.insert(0, REPO_ROOT_STR)
    return REPO_ROOT


def load_local_occ_controller(module_tag: str):
    try:
        mod = importlib.import_module("dynamic_env.main")
        cls = getattr(mod, "LocalTrackingControllerDyn_OCC", None)
        mod_file = Path(getattr(mod, "__file__", "")).resolve()
        if cls is not None and str(mod_file).startswith(REPO_ROOT_STR):
            return cls
    except Exception:
        pass

    local_main = REPO_ROOT / "dynamic_env" / "main.py"
    spec = importlib.util.spec_from_file_location(f"dynamic_env_main_local_{module_tag}", local_main)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local dynamic_env.main at {local_main}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "LocalTrackingControllerDyn_OCC")
