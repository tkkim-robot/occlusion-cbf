#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def _parse_override(spec: str) -> tuple[str, str, int, int]:
    parts = str(spec).split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --override-idx spec '{spec}'. Expected model:baseline:idx_start:idx_end."
        )
    model, baseline, idx_start, idx_end = parts
    return str(model), str(baseline), int(idx_start), int(idx_end)


def _apply_idx_override(cmd: list[str], idx_start: int, idx_end: int) -> list[str]:
    out = list(cmd)
    for flag, value in (("--idx-start", str(int(idx_start))), ("--idx-end", str(int(idx_end)))):
        if flag in out:
            k = out.index(flag)
            out[k + 1] = value
        else:
            out += [flag, value]
    return out


def _build_log_path(original_log: str, idx_start: int | None, idx_end: int | None) -> Path:
    p = Path(original_log)
    if idx_start is None or idx_end is None:
        return p
    return p.with_name(f"{p.stem}_{idx_start}_{idx_end}{p.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Resume an interrupted full baseline sweep.")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument(
        "--override-idx",
        action="append",
        default=[],
        help="Override one job range as model:baseline:idx_start:idx_end",
    )
    ap.add_argument("--max-parallel", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "run_manifest.json"
    status_path = out_dir / "run_status.json"

    manifest = _load_json(manifest_path, [])
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"No run manifest found at {manifest_path}")

    status = _load_json(status_path, [])
    if not isinstance(status, list):
        status = []

    completed_ok = {
        (str(item.get("model")), str(item.get("baseline")))
        for item in status
        if int(item.get("returncode", 1)) == 0
    }

    overrides = {}
    for spec in args.override_idx:
        model, baseline, idx_start, idx_end = _parse_override(spec)
        overrides[(model, baseline)] = (idx_start, idx_end)

    pending = []
    for item in manifest:
        model = str(item["model"])
        baseline = str(item["baseline"])
        if (model, baseline) in completed_ok:
            continue
        job = dict(item)
        if (model, baseline) in overrides:
            idx_start, idx_end = overrides[(model, baseline)]
            job["cmd"] = _apply_idx_override(list(job["cmd"]), idx_start, idx_end)
            job["idx_start"] = int(idx_start)
            job["idx_end"] = int(idx_end)
            job["log"] = str(_build_log_path(job["log"], idx_start, idx_end))
        pending.append(job)

    if not pending:
        print("[RESUME] nothing to run; all manifest jobs are already completed", flush=True)
        return 0

    running: list[dict] = []

    def launch(job: dict) -> dict:
        log_path = Path(job["log"])
        fh = log_path.open("w")
        fh.write("[CMD] " + " ".join(job["cmd"]) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            job["cmd"],
            cwd=root,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launched = dict(job)
        launched["proc"] = proc
        launched["fh"] = fh
        launched["start"] = time.perf_counter()
        return launched

    while pending or running:
        while pending and len(running) < int(args.max_parallel):
            job = launch(pending.pop(0))
            running.append(job)
            idx_note = ""
            if "idx_start" in job and "idx_end" in job:
                idx_note = f" idx={job['idx_start']}..{job['idx_end']}"
            print(f"[LAUNCH] {job['model']} {job['baseline']}{idx_note} pid={job['proc'].pid}", flush=True)

        time.sleep(5.0)
        still_running = []
        for job in running:
            rc = job["proc"].poll()
            if rc is None:
                still_running.append(job)
                continue
            dt = time.perf_counter() - float(job["start"])
            job["fh"].flush()
            job["fh"].close()
            rec = {
                "model": job["model"],
                "baseline": job["baseline"],
                "returncode": int(rc),
                "wall_s": float(dt),
                "log": job["log"],
            }
            if "idx_start" in job and "idx_end" in job:
                rec["idx_start"] = int(job["idx_start"])
                rec["idx_end"] = int(job["idx_end"])
                rec["resumed_partial"] = True
            status.append(rec)
            status_path.write_text(json.dumps(status, indent=2))
            idx_note = ""
            if "idx_start" in job and "idx_end" in job:
                idx_note = f" idx={job['idx_start']}..{job['idx_end']}"
            print(
                f"[DONE] {job['model']} {job['baseline']}{idx_note} rc={int(rc)} wall_s={float(dt):.1f}",
                flush=True,
            )
        running = still_running

    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
