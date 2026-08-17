#!/usr/bin/env python3
"""Launch isolated overnight DI and Unicycle OCBF Optuna studies.

The two canonical studies run sequentially from one immutable source snapshot.
Each study can therefore use every requested case worker without competing
with the other study for CPUs.  A small detached supervisor owns the sequence
and records terminal status in ``launcher_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WANDB_PROJECT = "occlusion-cbf-tuning"
CANONICAL_CASE_COUNT = 100
DEFAULT_TRIALS = 100
DEFAULT_WORKERS = 32
DEFAULT_BATCH_SIZE = 32
DEFAULT_PRUNER_STARTUP_TRIALS = 12
DEFAULT_PRUNER_WARMUP_CASES = 64
DEFAULT_PRUNER_INTERVAL_CASES = 32
SOURCE_FINGERPRINT_ALGORITHM = "sha256(path NUL sha256 LF)"


def _available_worker_count() -> int:
    """Return the CPUs available to this process, respecting affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _require_clean_worktree() -> None:
    """Refuse a snapshot whose bytes are not exactly the current commit."""
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot verify a clean Git worktree") from exc
    if result.stdout.strip():
        raise RuntimeError(
            "Refusing to launch from a dirty worktree. Commit or remove every "
            "tracked and untracked change before creating the source snapshot."
        )


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


def _copy_tracked_source_snapshot(destination: Path) -> None:
    """Copy only Git-tracked files after the clean-worktree preflight."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot enumerate tracked source files") from exc
    relative_paths = [
        Path(raw.decode("utf-8", errors="surrogateescape"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]
    if not relative_paths:
        raise RuntimeError("Git reported no tracked source files")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in relative_paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Git reported an unsafe tracked path: {relative}")
        source = REPO_ROOT.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        elif source.is_file():
            shutil.copy2(source, target)
        else:
            raise RuntimeError(f"Unsupported tracked source entry: {relative}")


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
        "algorithm": SOURCE_FINGERPRINT_ALGORITHM,
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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _supervise_manifest(manifest_path: Path) -> int:
    """Run manifest commands in order and persist every state transition."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Launcher manifest must contain a JSON object.")

    output_root = Path(manifest["output_root"]).expanduser().resolve()
    source_snapshot = Path(manifest["source_snapshot"]).expanduser().resolve()
    runs = manifest.get("runs")
    if manifest.get("status") != "queued":
        raise RuntimeError(
            "Refusing to supervise a manifest that is not in queued state."
        )
    if manifest_path.parent != output_root:
        raise ValueError("Launcher manifest must live directly in output_root.")
    if not source_snapshot.is_dir() or not _path_within(source_snapshot, output_root):
        raise ValueError("source_snapshot must be a directory inside output_root.")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Launcher manifest must contain at least one run.")

    for entry in runs:
        if not isinstance(entry, dict):
            raise ValueError("Every launcher run must be a JSON object.")
        if entry.get("status") != "queued":
            raise RuntimeError("Every run must be queued before supervision.")
        command = entry.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
        ):
            raise ValueError("Every launcher run needs a nonempty argv list.")
        log_path = Path(entry["log"]).expanduser().resolve()
        if not _path_within(log_path, output_root / "logs"):
            raise ValueError("Every run log must be inside output_root/logs.")

    manifest["status"] = "running"
    manifest["started_at"] = _now_iso()
    manifest["supervisor"] = {
        "pid": int(os.getpid()),
        "process_group": int(os.getpgrp()),
        "status": "running",
        "log": manifest.get("supervisor_log"),
    }
    _write_manifest(manifest_path, manifest)

    for run_index, entry in enumerate(runs):
        entry["status"] = "running"
        entry["started_at"] = _now_iso()
        _write_manifest(manifest_path, manifest)

        log_path = Path(entry["log"]).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a") as log_handle:
                completed = subprocess.run(
                    entry["command"],
                    cwd=source_snapshot,
                    env=dict(os.environ),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            return_code = int(completed.returncode)
            entry["return_code"] = return_code
        except Exception as exc:
            return_code = 1
            entry["return_code"] = None
            entry["launch_error"] = str(exc)

        entry["completed_at"] = _now_iso()
        if return_code != 0:
            entry["status"] = "failed"
            for pending in runs[run_index + 1 :]:
                pending["status"] = "not_started_after_failure"
            manifest["status"] = "failed"
            manifest["failed_run"] = str(entry.get("model", run_index))
            manifest["completed_at"] = _now_iso()
            manifest["supervisor"]["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            return return_code if 0 < return_code <= 255 else 1

        entry["status"] = "complete"
        _write_manifest(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["completed_at"] = _now_iso()
    manifest["supervisor"]["status"] = "complete"
    _write_manifest(manifest_path, manifest)
    return 0


def _study_command(
    *,
    python: Path,
    model: str,
    n_rand: int,
    workers: int,
    trials: int,
    timeout: float | None,
    batch_size: int,
    pruner_startup_trials: int,
    pruner_warmup_cases: int,
    pruner_interval_cases: int,
    disable_pruning: bool,
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
        "--batch-size",
        str(batch_size),
        "--pruner-startup-trials",
        str(pruner_startup_trials),
        "--pruner-warmup-cases",
        str(pruner_warmup_cases),
        "--pruner-interval-cases",
        str(pruner_interval_cases),
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
    ]
    if timeout is not None:
        command.extend(["--timeout", str(timeout)])
    if disable_pruning:
        command.append("--disable-pruning")
    if wandb_enabled:
        command.extend(
            [
                "--wandb",
                "--wandb-project",
                wandb_project,
                "--wandb-group",
                wandb_group,
                "--wandb-name",
                f"{model}_n{n_rand}",
            ]
        )
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the DI-50 and Unicycle-30 OCBF studies sequentially "
            "under one detached supervisor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--supervise-manifest",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional per-study Optuna timeout in seconds; omitted by default.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Case worker processes per study (default: 32).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=DEFAULT_PRUNER_STARTUP_TRIALS,
    )
    parser.add_argument(
        "--pruner-warmup-cases",
        type=int,
        default=DEFAULT_PRUNER_WARMUP_CASES,
    )
    parser.add_argument(
        "--pruner-interval-cases",
        type=int,
        default=DEFAULT_PRUNER_INTERVAL_CASES,
    )
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=None)
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable W&B logging after an authentication preflight.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--startup-grace",
        type=float,
        default=15.0,
        help="One-time wait before detached process health validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_launcher_args(args: argparse.Namespace) -> None:
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive when supplied")
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("--workers and --batch-size must be positive")
    available_workers = _available_worker_count()
    if args.workers > available_workers:
        raise ValueError(
            f"--workers={args.workers} exceeds the {available_workers} CPUs "
            "available to this process"
        )
    if args.workers != args.batch_size:
        raise ValueError("--workers and --batch-size must match")
    if args.batch_size > CANONICAL_CASE_COUNT:
        raise ValueError("--batch-size cannot exceed the 100 canonical cases")
    if args.pruner_startup_trials < 0:
        raise ValueError("--pruner-startup-trials cannot be negative")
    if args.pruner_warmup_cases < 0 or args.pruner_interval_cases <= 0:
        raise ValueError("Pruner warmup must be nonnegative and interval positive")
    if args.startup_grace < 0:
        raise ValueError("--startup-grace cannot be negative")
    if args.disable_pruning:
        return
    if args.batch_size >= CANONICAL_CASE_COUNT:
        raise ValueError("Pruning requires at least one intermediate case batch")
    if not 0 < args.pruner_warmup_cases < CANONICAL_CASE_COUNT:
        raise ValueError("Pruner warmup must be between 1 and 99 cases")
    if args.pruner_warmup_cases % args.batch_size != 0:
        raise ValueError("Pruner warmup must align with a completed case batch")
    if args.pruner_interval_cases % args.batch_size != 0:
        raise ValueError("Pruner interval must be a multiple of --batch-size")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.supervise_manifest is not None:
        return _supervise_manifest(args.supervise_manifest)
    _validate_launcher_args(args)
    # Keep the environment path instead of resolving its Python symlink, which
    # would lose the active environment's package context.
    python = Path(os.path.abspath(args.python.expanduser()))
    if not python.exists():
        raise FileNotFoundError(f"Tuning Python not found: {python}")
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
            "workers": int(args.workers),
        },
        {
            "model": "uni",
            "n_rand": 30,
            "workers": int(args.workers),
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
            timeout=(None if args.timeout is None else float(args.timeout)),
            batch_size=int(args.batch_size),
            pruner_startup_trials=int(args.pruner_startup_trials),
            pruner_warmup_cases=int(args.pruner_warmup_cases),
            pruner_interval_cases=int(args.pruner_interval_cases),
            disable_pruning=bool(args.disable_pruning),
            output_dir=run_dir,
            study_name=study_name,
            wandb_project=str(args.wandb_project),
            wandb_group=str(wandb_group),
            wandb_enabled=bool(args.wandb),
        )
        commands.append(
            {
                **spec,
                "study_name": study_name,
                "output_dir": str(run_dir),
                "log": str(output_root / "logs" / f"{model}_n{spec['n_rand']}.log"),
                "command": command,
                "status": "queued",
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "wandb_group": wandb_group,
                    "source_commit": source_commit,
                    "execution_mode": "sequential",
                    "workers": int(args.workers),
                    "batch_size": int(args.batch_size),
                    "pruner": {
                        "disabled": bool(args.disable_pruning),
                        "startup_trials": int(args.pruner_startup_trials),
                        "warmup_cases": int(args.pruner_warmup_cases),
                        "interval_cases": int(args.pruner_interval_cases),
                    },
                    "timeout_s": args.timeout,
                    "wandb_enabled": bool(args.wandb),
                    "commands": commands,
                },
                indent=2,
            )
        )
        return 0

    _require_clean_worktree()
    if args.wandb:
        _verify_wandb_auth(python)

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "logs").mkdir()
    _copy_tracked_source_snapshot(source_snapshot)
    _require_clean_worktree()
    if _git_value(["rev-parse", "HEAD"]) != source_commit:
        raise RuntimeError("Git HEAD changed while the source snapshot was created")
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

    supervisor_log = output_root / "logs" / "supervisor.log"
    manifest = {
        "created_at": _now_iso(),
        "status": "queued",
        "execution_mode": "sequential",
        "output_root": str(output_root),
        "source_snapshot": str(source_snapshot),
        "source_commit": source_commit,
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_algorithm": SOURCE_FINGERPRINT_ALGORITHM,
        "source_manifest": str(output_root / "source_manifest.json"),
        "snapshot_scope": "git_tracked_files",
        "worktree_clean": True,
        "wandb_project": str(args.wandb_project),
        "wandb_group": wandb_group,
        "wandb_enabled": bool(args.wandb),
        "trials": int(args.trials),
        "workers": int(args.workers),
        "batch_size": int(args.batch_size),
        "pruner": {
            "disabled": bool(args.disable_pruning),
            "startup_trials": int(args.pruner_startup_trials),
            "warmup_cases": int(args.pruner_warmup_cases),
            "interval_cases": int(args.pruner_interval_cases),
        },
        "timeout_s": args.timeout,
        "supervisor_log": str(supervisor_log),
        "runs": commands,
    }
    manifest_path = output_root / "launcher_manifest.json"
    _write_manifest(manifest_path, manifest)

    supervisor_command = [
        str(python),
        "-m",
        "tools.launch_ocbf_optuna",
        "--supervise-manifest",
        str(manifest_path),
    ]
    try:
        supervisor_handle = supervisor_log.open("a")
        try:
            supervisor = subprocess.Popen(
                supervisor_command,
                cwd=source_snapshot,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=supervisor_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            supervisor_handle.close()
    except OSError as exc:
        manifest["status"] = "startup_failed"
        manifest["startup_error"] = str(exc)
        manifest["completed_at"] = _now_iso()
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(
            f"Failed to start tuning supervisor; see {supervisor_log}"
        ) from exc

    time.sleep(float(args.startup_grace))
    return_code = supervisor.poll()
    latest_manifest = json.loads(manifest_path.read_text())
    startup_failed = latest_manifest.get("status") in {
        "failed",
        "startup_failed",
    }
    if startup_failed or (return_code is not None and int(return_code) != 0):
        return_code_label = "running" if return_code is None else str(return_code)
        raise RuntimeError(
            "Tuning supervisor failed during startup validation "
            f"(rc={return_code_label}, status={latest_manifest.get('status')}); "
            f"see {supervisor_log}"
        )

    print(json.dumps(latest_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
