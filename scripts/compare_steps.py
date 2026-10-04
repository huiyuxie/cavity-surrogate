#!/usr/bin/env python3
"""Compare final cavity velocity fields on the SAME grid at two time steps.

This script compares velocity fields obtained using two different time-step sizes
(base_dt vs. half_dt) on the same spatial grid (default: N=64) to evaluate temporal
discretization sensitivity.

Run from the project root: 
    python3 scripts/compare_steps.py \
    --root cfd/step_study_Re100 --out results/step_study_Re100
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from compare_grids import read_snapshots, velocity_change, centerlines
import matplotlib.pyplot as plt


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare two time steps on the same cavity grid.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing deltaT and deltaT_half.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Directory to save comparison results.")
    parser.add_argument("--times", type=float, nargs=3, default=[9.0, 9.5, 10.0])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--length", type=float, default=0.1)
    parser.add_argument("--u-lid", type=float, default=1.0)
    parser.add_argument("--steady-rms-tol", type=float, default=1e-4)
    parser.add_argument("--steady-max-tol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.n < 2 or args.n % 2:
        parser.error("--n must be an even integer >= 2 for centerline extraction.")
    if not np.isfinite([args.length, args.u_lid, args.steady_rms_tol,
                        args.steady_max_tol, *args.times]).all():
        parser.error("Length, lid speed, thresholds, and times must be finite.")
    if min(args.length, args.u_lid, args.steady_rms_tol,
           args.steady_max_tol) <= 0 or np.any(np.diff(args.times) <= 0):
        parser.error("Length, lid speed, and thresholds must be positive; times must increase.")
    if not np.allclose(np.diff(args.times), 0.5, rtol=0, atol=1e-8):
        parser.error("Use 0.5 s between snapshots for this steady-state screening.")
    args.root = args.root.expanduser().resolve()
    cases = (("base_dt", args.root / "deltaT"),
             ("half_dt", args.root / "deltaT_half"))

    final = {}
    all_steady = True
    print("Preliminary steady-state screening within each case:")
    print(f"  limits: E_RMS < {args.steady_rms_tol:g}, "
          f"E_max < {args.steady_max_tol:g} (normalized by U_lid)")
    for label, case in cases:
        print(f"\nReading {label}: {case}", flush=True)
        fields = read_snapshots(case, args.n, args.times, args.length)
        for i in range(2):
            rms, maximum = velocity_change(fields[i], fields[i + 1], args.u_lid)
            passed = rms < args.steady_rms_tol and maximum < args.steady_max_tol
            all_steady = all_steady and passed
            print(f"  t={args.times[i]:g} -> {args.times[i+1]:g} s: "
                  f"E_RMS={rms:.8e}, E_max={maximum:.8e}, pass={passed}")
        final[label] = fields[-1]
    if not all_steady:
        raise SystemExit("Steady-state screening failed in at least one case. "
                         "Inspect the time trend and solver logs before comparing time steps. "
                         "Existing output files, if any, are from earlier runs.")

    base, half = final["base_dt"], final["half_dt"]
    rms, maximum = velocity_change(base, half, args.u_lid)
    denominator = np.linalg.norm(half.ravel())
    if denominator == 0:
        raise ValueError("The half-step reference field is zero; relative L2 is undefined.")
    samples = np.linspace(0, args.length, 201)
    lines = {label: centerlines(field, args.length, args.u_lid, samples)
             for label, field in final.items()}
    metrics = dict(
        N=args.n, comparison_time=args.times[-1],
        relative_L2=float(np.linalg.norm((base - half).ravel()) / denominator),
        E_RMS_Ulid=rms, E_max_Ulid=maximum,
        centerline_u_max_Ulid=float(np.max(np.abs(lines["base_dt"][0] - lines["half_dt"][0]))),
        centerline_v_max_Ulid=float(np.max(np.abs(lines["base_dt"][1] - lines["half_dt"][1]))),
    )
    print(f"\nTime-step comparison at t={args.times[-1]:g} s only:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.8e}")
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "time_step_metrics.csv", [metrics])
    rows = []
    for i, position in enumerate(samples / args.length):
        row = {"s_over_L": float(position)}
        for label in final:
            row[f"u_vertical_{label}"] = float(lines[label][0][i])
            row[f"v_horizontal_{label}"] = float(lines[label][1][i])
        rows.append(row)
    write_rows(args.out / "time_step_centerlines.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for label, style in (("base_dt", "-"), ("half_dt", "--")):
        axes[0].plot(lines[label][0], samples / args.length, style, label=label)
        axes[1].plot(samples / args.length, lines[label][1], style, label=label)
    axes[0].set(xlabel="u / U_lid", ylabel="y / L", title="Vertical centerline: x/L = 0.5")
    axes[1].set(xlabel="x / L", ylabel="v / U_lid", title="Horizontal centerline: y/L = 0.5")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"N{args.n}: time-step comparison at t = {args.times[-1]:g} s")
    fig.savefig(args.out / "time_step_centerlines.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved results to {args.out}")
    print("Passing this screening alone does not prove steady-state convergence.")
    print("This checks final steady-field sensitivity, not transient-history accuracy.")
    print("Case names do not verify the actual time steps; check controlDict and run logs.")


if __name__ == "__main__":
    main()
