#!/usr/bin/env python3
"""Launch isolated overnight DI and Unicycle OCBF Optuna studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WANDB_PROJECT = "occlusion-cbf-tuning"


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return default
    return result.stdout.strip() or default


def _snapshot_ignore(_path: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        "output",
        "tmp",
        "__pycache__",
        ".pytest_cache",
    }
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _source_manifest(source_root: Path) -> dict[str, Any]:
    """Return an exact, deterministic content fingerprint for a snapshot."""
    entries: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(
        item for item in source_root.rglob("*") if item.is_file()
    ):
        relative = path.relative_to(source_root).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hash = digest.hexdigest()
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_hash,
            }
        )
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(file_hash))
        combined.update(b"\n")
    return {
        "algorithm": "sha256(path NUL sha256 LF)",
        "fingerprint": combined.hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def _verify_wandb_auth(python: Path) -> None:
    """Fail before detaching if the existing W&B login cannot be verified."""
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import wandb; "
                "assert wandb.login(verify=True), "
                "'Weights & Biases authentication failed'"
            ),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "WANDB_SILENT": "true"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"W&B authentication preflight failed: {detail}")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def _study_command(
    *,
    python: Path,
    model: str,
    n_rand: int,
    workers: int,
    trials: int,
    timeout: float,
    output_dir: Path,
    study_name: str,
    wandb_project: str,
    wandb_group: str,
    wandb_enabled: bool,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "tools.tune_ocbf_optuna",
        "--model",
        model,
        "--n-rand",
        str(n_rand),
        "--workers",
        str(workers),
        "--trials",
        str(trials),
        "--timeout",
        str(timeout),
        "--batch-size",
        "10",
        "--idx-start",
        "1",
        "--idx-end",
        "100",
        "--tf",
        "500",
        "--study-name",
        study_name,
        "--output-dir",
        str(output_dir),
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        wandb_group,
        "--wandb-name",
        f"{model}_n{n_rand}",
    ]
    if wandb_enabled:
        command.append("--wandb")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch both OCBF tuning studies in detached sessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=14 * 60 * 60)
    parser.add_argument("--di-workers", type=int, default=6)
    parser.add_argument("--uni-workers", type=int, default=3)
    parser.add_argument("--di-cpus", default="0,1,2,3,4,5")
    parser.add_argument("--uni-cpus", default="6,7,8")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--startup-grace",
        type=float,
        default=15.0,
        help="One-time wait before detached process health validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Keep the environment path instead of resolving its Python symlink, which
    # would lose the active environment's package context.
    python = Path(os.path.abspath(args.python.expanduser()))
    if not python.exists():
        raise FileNotFoundError(f"Tuning Python not found: {python}")
    if (
        args.trials <= 0
        or args.timeout <= 0
        or args.di_workers <= 0
        or args.uni_workers <= 0
        or args.startup_grace < 0
    ):
        raise ValueError("trials, timeout, workers, and startup grace are invalid")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (REPO_ROOT / "output" / f"optuna_ocbf_{timestamp}").resolve()
    )
    wandb_group = args.wandb_group or f"ocbf_crowd_{timestamp}"
    source_commit = _git_value(["rev-parse", "HEAD"])
    source_snapshot = output_root / "source_snapshot"

    specs = [
        {
            "model": "di",
            "n_rand": 50,
            "workers": int(args.di_workers),
            "cpus": str(args.di_cpus),
        },
        {
            "model": "uni",
            "n_rand": 30,
            "workers": int(args.uni_workers),
            "cpus": str(args.uni_cpus),
        },
    ]

    commands: list[dict[str, Any]] = []
    for spec in specs:
        model = spec["model"]
        run_dir = output_root / f"{model}_n{spec['n_rand']}"
        study_name = (
            f"ocbf_crowd_{model}_n{spec['n_rand']}_{timestamp}"
        )
        command = _study_command(
            python=python,
            model=model,
            n_rand=spec["n_rand"],
            workers=spec["workers"],
            trials=int(args.trials),
            timeout=float(args.timeout),
            output_dir=run_dir,
            study_name=study_name,
            wandb_project=str(args.wandb_project),
            wandb_group=str(wandb_group),
            wandb_enabled=not bool(args.no_wandb),
        )
        commands.append(
            {
                **spec,
                "study_name": study_name,
                "output_dir": str(run_dir),
                "log": str(output_root / "logs" / f"{model}_n{spec['n_rand']}.log"),
                "command": ["taskset", "-c", spec["cpus"], *command],
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "wandb_group": wandb_group,
                    "source_commit": source_commit,
                    "commands": commands,
                },
                indent=2,
            )
        )
        return 0

    if not args.no_wandb:
        _verify_wandb_auth(python)

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "logs").mkdir()
    shutil.copytree(
        REPO_ROOT,
        source_snapshot,
        ignore=_snapshot_ignore,
    )
    source_manifest = _source_manifest(source_snapshot)
    source_fingerprint = str(source_manifest["fingerprint"])
    (output_root / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )

    git_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output_root / "source_worktree.patch").write_text(git_diff)

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
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
            "WANDB_SILENT": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OCBF_SOURCE_COMMIT": source_commit,
            "OCBF_SOURCE_FINGERPRINT": source_fingerprint,
            "OCBF_SOURCE_DIR": str(source_snapshot),
            "PYTHONPATH": str(source_snapshot),
        }
    )

    processes: dict[str, subprocess.Popen] = {}
    for entry in commands:
        log_path = Path(entry["log"])
        log_handle = log_path.open("a")
        process = subprocess.Popen(
            entry["command"],
            cwd=source_snapshot,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        entry["pid"] = process.pid
        entry["process_group"] = process.pid
        entry["startup_status"] = "starting"
        processes[str(entry["model"])] = process

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "starting",
        "output_root": str(output_root),
        "source_snapshot": str(source_snapshot),
        "source_commit": source_commit,
        "source_fingerprint": source_fingerprint,
        "source_manifest": str(output_root / "source_manifest.json"),
        "wandb_project": str(args.wandb_project),
        "wandb_group": wandb_group,
        "wandb_enabled": not bool(args.no_wandb),
        "trials": int(args.trials),
        "timeout_s": float(args.timeout),
        "runs": commands,
    }
    manifest_path = output_root / "launcher_manifest.json"
    _write_manifest(manifest_path, manifest)

    time.sleep(float(args.startup_grace))
    failed: list[dict[str, Any]] = []
    for entry in commands:
        process = processes[str(entry["model"])]
        return_code = process.poll()
        if return_code is None:
            entry["startup_status"] = "running"
        else:
            entry["startup_status"] = "exited"
            entry["startup_return_code"] = int(return_code)
            failed.append(entry)

    if failed:
        for entry in commands:
            process = processes[str(entry["model"])]
            if process.poll() is None:
                os.killpg(int(entry["process_group"]), signal.SIGTERM)
                entry["startup_status"] = "terminated_after_peer_failure"
        manifest["status"] = "startup_failed"
        manifest["startup_checked_at"] = datetime.now().astimezone().isoformat()
        _write_manifest(manifest_path, manifest)
        failed_labels = ", ".join(
            f"{entry['model']} (rc={entry['startup_return_code']})"
            for entry in failed
        )
        raise RuntimeError(
            f"Tuning startup validation failed for {failed_labels}; "
            f"see {output_root / 'logs'}"
        )

    manifest["status"] = "running"
    manifest["startup_checked_at"] = datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
