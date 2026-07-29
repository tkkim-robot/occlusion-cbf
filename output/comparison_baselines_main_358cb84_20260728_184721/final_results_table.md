# Crowd Benchmark — Final Results

Updated: 2026-07-29

Each configuration contains 100 deterministic trials (seed 42, trial indices
1–100) in the `crowd` forced-emergence scenario. Therefore, every integer count
below is also the corresponding percentage.

## Outcomes

Each cell is `Success / Collision / Infeasible`.

| Method | DI-10 | DI-30 | DI-50 | Unicycle-10 | Unicycle-20 | Unicycle-30 |
|---|---:|---:|---:|---:|---:|---:|
| CBF-QP | 86 / 13 / 1 | 45 / 38 / 17 | 33 / 39 / 28 | 55 / 4 / 41 | 33 / 5 / 62 | 18 / 5 / 77 |
| Single-Risk MPC | 23 / 0 / 77 | 1 / 0 / 99 | 0 / 0 / 100 | 15 / 0 / 85 | 3 / 0 / 97 | 0 / 0 / 100 |
| Control-Tree MPC | 53 / 0 / 47 | 10 / 0 / 90 | 1 / 0 / 99 | 21 / 0 / 79 | 6 / 0 / 94 | 3 / 0 / 97 |
| OACP-MPC | 46 / 2 / 52 | 17 / 6 / 77 | 5 / 6 / 89 | 56 / 43 / 1 | 30 / 70 / 0 | 14 / 86 / 0 |
| OA-MPC | 0 / 1 / 99 | 0 / 1 / 99 | 0 / 3 / 97 | 0 / 15 / 85 | 0 / 7 / 93 | 0 / 6 / 94 |
| **Occlusion-CBF (ours)** | **97 / 0 / 3** | **87 / 0 / 13** | **67 / 0 / 33** | **84 / 1 / 15** | **73 / 0 / 27** | **69 / 0 / 31** |

## Controller compute time

Average controller compute time in milliseconds per control step, as recorded
by each full benchmark JSON:

| Method | DI-10 | DI-30 | DI-50 | Unicycle-10 | Unicycle-20 | Unicycle-30 |
|---|---:|---:|---:|---:|---:|---:|
| CBF-QP | 2.455 | 2.657 | 2.779 | 2.482 | 2.610 | 2.778 |
| Single-Risk MPC | 32.063 | 39.756 | 46.829 | 23.858 | 27.966 | 33.444 |
| Control-Tree MPC | 293.222 | 336.628 | 365.840 | 210.504 | 234.539 | 258.058 |
| OACP-MPC | 467.320 | 475.335 | 492.642 | 382.776 | 401.473 | 415.196 |
| OA-MPC | 297.126 | 549.251 | 709.609 | 258.884 | 407.638 | 482.560 |
| Occlusion-CBF (ours) | 9.467 | 15.645 | 22.254 | 21.861 | 28.049 | 33.594 |

These timings are diagnostic measurements from 10-worker full sweeps pinned to
physical CPUs 0–9. They average all controller steps, include cold/JIT steps,
and were collected while workers competed for compute resources. They should
not be presented as isolated single-controller latency measurements.

## Warmed sequential controller compute time

The comparable warmed timing pass below was measured on a MacBook Air with an
Apple M4 CPU (10 cores, 4 performance + 6 efficiency), 24 GB RAM, and native
arm64 Python 3.11.13. JAX 0.7.1 ran on its CPU backend.

| Method | DI-10 | DI-30 | DI-50 | Unicycle-10 | Unicycle-20 | Unicycle-30 |
|---|---:|---:|---:|---:|---:|---:|
| CBF-QP | 0.659 | 0.740 | 0.753 | 0.691 | 0.711 | 0.797 |
| Single-Risk MPC | 12.990 | 13.757 | 15.356 | 7.249 | 8.138 | 8.676 |
| Control-Tree MPC | 106.676 | 110.651 | 117.039 | 48.304 | 54.170 | 58.929 |
| OACP-MPC | 87.368 | 89.434 | 88.830 | 80.533 | 76.900 | 78.478 |
| OA-MPC | 93.822 | 151.444 | 186.946 | 69.225 | 101.502 | 117.458 |
| **Occlusion-CBF (ours)** | **1.596** | **2.346** | **3.881** | **4.116** | **4.498** | **4.507** |

Each cell is milliseconds per controller step, averaged equally over seed-42
cases 1 and 2. Each case ran for 60 steps; the first 10 were excluded and the
remaining 50 were timed, giving 100 warmed samples per cell and 3,600 samples
overall. The 72 short cases ran strictly sequentially in one process with
plotting disabled and the numerical/JAX runtime pinned to one thread. The host
was on AC power with Low Power Mode off, no competing benchmark process, and no
thermal or performance warning.

These are intentionally short-run warmed estimates, not full-trajectory
averages. The metric is each controller's reported
`last_total_compute_time_ms` (with profile timing as a fallback), excluding
plotting. Controller-internal timing boundaries differ: notably, CBF-QP
performs visibility filtering outside its internal timer. The values should
therefore be described as controller-reported compute time rather than uniform
outer-loop latency.

Raw warmed timing artifacts:

- [`compute_times_20260729_162447.csv`](../sequential_compute_timing/compute_times_20260729_162447.csv)
- [`compute_times_20260729_162447.json`](../sequential_compute_timing/compute_times_20260729_162447.json)
- [`compute_times_20260729_162447.md`](../sequential_compute_timing/compute_times_20260729_162447.md)

## Provenance and validation

- CBF-QP, Single-Risk MPC, Control-Tree MPC, and OACP-MPC results are under
  `results/<case>/` and were produced from commit
  `358cb8464e16feaade74c840c533cdf2c981344c`.
- Occlusion-CBF results are under
  `../../ocbf_tuned_benchmark_20260728_145048/<case>/` and were produced from
  commit `80eb3dc05565690e192da1f6f4e9f76ade12f96b`.
- OA-MPC results are under `results/<case>/` and were rerun from the exact
  committed source snapshot `8481eb86fd9185a551c199c4a44a20fe712785af`.
- All 36 method/configuration JSON summaries contain 100 trials. Their paired
  CSV files contain exactly the unique indices 1–100, summary counts agree with
  the row classifications, and no row contains an exception.

Fresh OA-MPC result timestamps:

| Configuration | Result timestamp |
|---|---|
| DI-10 | `20260729_133901` |
| DI-30 | `20260729_134715` |
| DI-50 | `20260729_140130` |
| Unicycle-10 | `20260729_141852` |
| Unicycle-20 | `20260729_145540` |
| Unicycle-30 | `20260729_151640` |
