#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
WORKERS = 4
MAX_JOBS = 3


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(line: str) -> None:
    with (ROOT_DIR / "scheduler_progress.log").open("a") as f:
        f.write(f"{stamp()} {line}\n")


def has_summary(out_dir: Path) -> bool:
    return bool(glob.glob(str(out_dir / "crowd_trials_occlusion_cbf_42_1_100_*.json")))


def command_for(T: float, mode: str, topk: int, kappa: int, out_dir: Path) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "tools/benchmark_crowd_trials.py",
        "--scenario",
        "crowd2",
        "--baseline",
        "occlusion_cbf",
        "--model",
        "uni",
        "--seed",
        "42",
        "--idx-start",
        "1",
        "--idx-end",
        "100",
        "--n-rand",
        "50",
        "--tf",
        "500",
        "--crowd-mode",
        "forced_emergence",
        "--forced-events",
        "6",
        "--forced-hidden-speed",
        "1.0",
        "--forced-occluder-radius-min",
        "0.8",
        "--forced-occluder-radius-max",
        "1.0",
        "--forced-validate-occlusion",
        "true",
        "--forced-require-corridor-conflict",
        "true",
        "--workers",
        str(WORKERS),
        "--occ-version",
        "v2",
        "--occ-enable-visible-hocbf",
        "false",
        "--occ-t-horizon",
        str(T),
        "--occ-rho-T",
        "auto",
        "--occ-terminal-slack-max",
        "0.0",
        "--occ-obs-hocbf-slack-max",
        "0.0",
        "--occ-rollout-slack-max",
        "0.0",
        "--occ-max-active-occlusions",
        str(topk),
        "--occ-selection-mode",
        "h_tilde",
        "--occ-vref-scenario-softmax-kappa",
        str(kappa),
        "--occ-vref-scenario-weight-mode",
        "barrier_predicted_margin",
        "--occ-vref-scenario-prediction-dt",
        "0.0",
        "--occ-qp-failure-fallback-mode",
        "state_safe",
        "--uni-vref-tracking-mode",
        "gated",
        "--vref",
        "los",
        "--out-dir",
        str(out_dir),
    ]
    if mode == "forward":
        cmd.extend(["--uni-forward-only", "true"])
    else:
        cmd.extend(["--uni-allow-reverse", "true", "--uni-v-min", "-1.0"])
    return cmd


def main() -> int:
    manifest = ROOT_DIR / "python_scheduler_manifest.csv"
    manifest.write_text("label,mode,T,topK,kappa,out_dir,status\n")
    log("python_scheduler_started")

    configs: list[tuple[float, str, int, int]] = []
    for T in (0.5, 0.75, 1.0, 1.25):
        for mode in ("forward", "fullrev"):
            for topk in (1, 2, 3):
                kappas = (20,) if topk == 1 else (0, 10, 20, 40, 60)
                for kappa in kappas:
                    configs.append((T, mode, topk, kappa))

    running: list[tuple[subprocess.Popen, str, Path]] = []
    idx = 0

    def record(label: str, mode: str, T: float, topk: int, kappa: int, out_dir: Path, status: str) -> None:
        with manifest.open("a") as f:
            f.write(f"{label},{mode},{T},{topk},{kappa},{out_dir},{status}\n")

    while idx < len(configs) or running:
        while idx < len(configs) and len(running) < MAX_JOBS:
            T, mode, topk, kappa = configs[idx]
            idx += 1
            label = f"T{str(T).replace('.', 'p')}_{mode}_top{topk}_k{kappa}"
            out_dir = ROOT_DIR / label
            out_dir.mkdir(parents=True, exist_ok=True)

            if has_summary(out_dir):
                log(f"{label} skip_existing")
                record(label, mode, T, topk, kappa, out_dir, "skip_existing")
                continue

            log(f"{label} start")
            record(label, mode, T, topk, kappa, out_dir, "started")
            log_file = (out_dir / "run.log").open("w")
            proc = subprocess.Popen(
                command_for(T, mode, topk, kappa, out_dir),
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, label, out_dir))

        next_running: list[tuple[subprocess.Popen, str, Path]] = []
        for proc, label, out_dir in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((proc, label, out_dir))
                continue
            status = "done" if rc == 0 else f"failed_{rc}"
            log(f"{label} {status}")
            parts = label.split("_")
            record(label, parts[1], float(parts[0][1:].replace("p", ".")), int(parts[2][3:]), int(parts[3][1:]), out_dir, status)
        running = next_running
        time.sleep(5)

    log("python_scheduler_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
