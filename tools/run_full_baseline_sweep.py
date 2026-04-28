#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


FREEZE_BASELINES = [
    ("di", "control_tree_mpc"),
    ("di", "oacp_mpc"),
    ("uni", "control_tree_mpc"),
    ("uni", "oacp_mpc"),
    ("di", "single_risk_mpc"),
    ("uni", "single_risk_mpc"),
    ("di", "oa_mpc"),
    ("uni", "oa_mpc"),
    ("di", "cbf_qp"),
    ("uni", "cbf_qp"),
]


def _build_cmd(args: argparse.Namespace, model: str, baseline: str) -> list[str]:
    cmd = [
        str(sys.executable),
        "tools/benchmark_crowd_trials.py",
        "--scenario",
        str(args.scenario),
        "--baseline",
        str(baseline),
        "--model",
        str(model),
        "--seed",
        str(int(args.seed)),
        "--idx-start",
        str(int(args.idx_start)),
        "--idx-end",
        str(int(args.idx_end)),
        "--n-rand",
        str(int(args.n_rand)),
        "--tf",
        str(float(args.tf)),
        "--crowd-mode",
        str(args.crowd_mode),
        "--forced-events",
        str(int(args.forced_events)),
        "--forced-hidden-speed",
        str(float(args.forced_hidden_speed)),
        "--workers",
        str(int(args.workers)),
    ]
    if baseline == "oa_mpc":
        cmd += [
            "--wmax",
            str(args.oa_wmax),
            "--oa-dynamic-occluders",
            "true" if bool(args.oa_dynamic_occluders) else "false",
            "--oa-visible-reach-mode",
            str(args.oa_visible_reach_mode),
            "--oa-use-nominal-tracking-cost",
            "true" if bool(args.oa_use_nominal_tracking_cost) else "false",
            "--oa-allow-solver-fallback",
            "true" if bool(args.oa_allow_solver_fallback) else "false",
        ]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description="Run the full crowd2 baseline freeze sweep.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--scenario", type=str, default="crowd2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--idx-start", type=int, default=1)
    p.add_argument("--idx-end", type=int, default=100)
    p.add_argument("--n-rand", type=int, default=50)
    p.add_argument("--tf", type=float, default=500.0)
    p.add_argument("--crowd-mode", type=str, default="forced_emergence")
    p.add_argument("--forced-events", type=int, default=6)
    p.add_argument("--forced-hidden-speed", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--oa-wmax", type=str, default="default")
    p.add_argument("--oa-dynamic-occluders", action="store_true", default=True)
    p.add_argument("--oa-visible-reach-mode", type=str, default="worst_case")
    p.add_argument("--oa-use-nominal-tracking-cost", action="store_true", default=False)
    p.add_argument("--oa-allow-solver-fallback", action="store_true", default=False)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for model, baseline in FREEZE_BASELINES:
        log = out_dir / f"{model}_{baseline}.log"
        items.append(
            {
                "model": model,
                "baseline": baseline,
                "cmd": _build_cmd(args, model, baseline),
                "log": str(log),
            }
        )

    (out_dir / "run_manifest.json").write_text(json.dumps(items, indent=2))
    (out_dir / "run_status.json").write_text("[]\n")

    pending = list(items)
    running: list[dict] = []
    results: list[dict] = []

    def launch(item: dict) -> dict:
        log_path = Path(item["log"])
        fh = log_path.open("w")
        fh.write("[CMD] " + " ".join(item["cmd"]) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            item["cmd"],
            cwd=root,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launched = dict(item)
        launched["proc"] = proc
        launched["fh"] = fh
        launched["start"] = time.perf_counter()
        return launched

    while pending or running:
        while pending and len(running) < int(args.max_parallel):
            item = launch(pending.pop(0))
            running.append(item)
            print(f"[LAUNCH] {item['model']} {item['baseline']} pid={item['proc'].pid}", flush=True)

        time.sleep(5.0)
        still_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            dt = time.perf_counter() - float(item["start"])
            item["fh"].flush()
            item["fh"].close()
            rec = {
                "model": item["model"],
                "baseline": item["baseline"],
                "returncode": int(rc),
                "wall_s": float(dt),
                "log": item["log"],
            }
            results.append(rec)
            (out_dir / "run_status.json").write_text(json.dumps(results, indent=2))
            print(
                f"[DONE] {item['model']} {item['baseline']} rc={int(rc)} wall_s={float(dt):.1f}",
                flush=True,
            )
        running = still_running

    (out_dir / "run_status.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
