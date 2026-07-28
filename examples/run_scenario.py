"""Unified command-line entry point for supported experiment scenarios."""

from __future__ import annotations

import argparse
import importlib
import os
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

BENCHMARK_CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JAX_PLATFORM_NAME": "cpu",
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": (
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1"
    ),
}


def _apply_benchmark_cpu_environment():
    """Apply the CPU/thread settings used by the paper benchmark launchers."""
    os.environ.update(BENCHMARK_CPU_ENVIRONMENT)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one of the maintained occlusion-CBF scenarios.",
        epilog=(
            "Arguments not recognized by this launcher are forwarded to the selected "
            "scenario. Crowd is the default, so `--model`, `--idx`, and other crowd "
            "options can be passed directly. Use `--scenario NAME -- --help` for "
            "scenario-specific help."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="crowd",
        choices=tuple((*SCENARIO_MODULES, *SCENARIO_ALIASES)),
        help="Scenario to run; the canonical crowd benchmark is the default.",
    )
    benchmark_environment = parser.add_mutually_exclusive_group()
    benchmark_environment.add_argument(
        "--benchmark-cpu-env",
        dest="benchmark_cpu_env",
        action="store_true",
        default=None,
        help=(
            "Apply the benchmark's one-thread CPU/JAX environment before "
            "importing the scenario. This is already the default for crowd."
        ),
    )
    benchmark_environment.add_argument(
        "--no-benchmark-cpu-env",
        dest="benchmark_cpu_env",
        action="store_false",
        help=(
            "Do not apply the canonical crowd benchmark CPU/JAX environment. "
            "This may change solver-boundary outcomes."
        ),
    )
    args, scenario_argv = parser.parse_known_args(argv)
    if scenario_argv[:1] == ["--"]:
        scenario_argv = scenario_argv[1:]

    scenario_name = SCENARIO_ALIASES.get(args.scenario, args.scenario)
    use_benchmark_environment = args.benchmark_cpu_env
    if use_benchmark_environment is None:
        use_benchmark_environment = scenario_name == "crowd"
    if use_benchmark_environment:
        _apply_benchmark_cpu_environment()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    scenario = importlib.import_module(SCENARIO_MODULES[scenario_name])
    result = scenario.main(scenario_argv)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
