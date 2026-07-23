"""Unified command-line entry point for supported experiment scenarios."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


SCENARIO_MODULES = {
    "crowd": "examples.test_crowd",
    "crowd_narrow": "examples.test_crowd_narrow",
    "campus": "examples.test_campus",
    "crosswalk": "examples.test_crosswalk",
}
SCENARIO_ALIASES = {
    "crowd2": "crowd",
    "crowd1": "crowd_narrow",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one of the maintained occlusion-CBF scenarios.",
        epilog=(
            "Arguments not recognized by this launcher are forwarded to the selected "
            "scenario. Use `--scenario NAME -- --help` for scenario-specific options."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=tuple((*SCENARIO_MODULES, *SCENARIO_ALIASES)),
    )
    args, scenario_argv = parser.parse_known_args(argv)
    if scenario_argv[:1] == ["--"]:
        scenario_argv = scenario_argv[1:]

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    scenario_name = SCENARIO_ALIASES.get(args.scenario, args.scenario)
    scenario = importlib.import_module(SCENARIO_MODULES[scenario_name])
    result = scenario.main(scenario_argv)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
