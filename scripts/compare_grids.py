#!/usr/bin/env python3
"""Lid-driven cavity grid sensitivity and centerline profile comparison.

This script processes OpenFOAM velocity fields across three nested uniform grids 
(N32, N64, N128) to assess grid sensitivity for a lid-driven cavity flow.

Run from the project root:
    python3 scripts/compare_cavity_grids.py \
    --root cfd/grid_study_Re100 --out results/grid_study_Re100
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNG files in headless environments, including SSH sessions.
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def ordered_velocity(mesh, n, length):
    """Order by physical position as velocity[y_index, x_index, component]."""
    xyz = np.asarray(mesh.cell_centers().points, dtype=float)
    velocity = np.asarray(mesh.cell_data["U"], dtype=float)
    if xyz.shape != (n * n, 3) or velocity.shape != (n * n, 3):
        raise ValueError(f"N{n}: Unexpected cell count or U shape: {xyz.shape}, {velocity.shape}")
    if not (np.isfinite(xyz).all() and np.isfinite(velocity).all()):
        raise ValueError(f"N{n}: Data contain NaN or Inf.")

    h = length / n
    ij = np.rint(xyz[:, :2] / h - 0.5).astype(int)
    if np.any(ij < 0) or np.any(ij >= n):
        raise ValueError(f"N{n}: Cell centers lie outside the expected domain.")
    expected = (ij + 0.5) * h
    if not np.allclose(xyz[:, :2], expected, rtol=0, atol=length * 1e-6):
        raise ValueError(f"N{n}: This script requires uniform grids with the specified geometry.")
    if np.ptp(xyz[:, 2]) > length * 1e-6:
        raise ValueError(f"N{n}: Expected a single layer of cells in z.")
    bounds = np.asarray(mesh.bounds, dtype=float)
    if not np.allclose(bounds[:4], [0, length, 0, length], rtol=0, atol=length * 1e-6):
        raise ValueError(f"N{n}: Mesh bounds do not match [0,L] x [0,L].")

    cell_index = ij[:, 1] * n + ij[:, 0]
    if np.unique(cell_index).size != n * n:
        raise ValueError(f"N{n}: Duplicate or missing cell indices.")
    order = np.argsort(cell_index)
    return velocity[order].reshape(n, n, 3).copy()


def read_snapshots(case, n, times, length):
    """Read original internal-cell U values using OpenFOAMReader."""
    marker = case / "case.foam"
    if not marker.is_file():
        raise FileNotFoundError(f"Missing {marker}; run touch case.foam in the case directory.")
    reader = pv.OpenFOAMReader(str(marker))
    reader.cell_to_point_creation = False
    reader.disable_all_patch_arrays()
    reader.enable_patch_array("internalMesh")
    reader.disable_all_cell_arrays()
    reader.enable_cell_array("U")
    available = np.asarray(reader.time_values, dtype=float)
    snapshots = []
    for time in times:
        match = np.flatnonzero(np.isclose(available, time, rtol=0, atol=1e-8))
        if match.size != 1:
            raise ValueError(f"{case}: Cannot find a unique t={time}; available times: {available}")
        reader.set_active_time_value(float(available[match[0]]))
        mesh = reader.read()["internalMesh"]
        snapshots.append(ordered_velocity(mesh, n, length))
    return snapshots


def velocity_change(a, b, u_lid):
    magnitude = np.linalg.norm(b - a, axis=-1)
    return float(np.sqrt(np.mean(magnitude**2)) / u_lid), float(magnitude.max() / u_lid)


def restrict_to_coarse(fine, n_coarse):
    """For 2x refinement, average each 2x2 fine-cell block into one coarse cell."""
    n_fine = fine.shape[0]
    if fine.shape != (n_fine, n_fine, 3) or n_fine % n_coarse:
        raise ValueError("Grids must be uniform, nested, and cover the same domain with an integer refinement ratio.")
    ratio = n_fine // n_coarse
    return fine.reshape(n_coarse, ratio, n_coarse, ratio, 3).mean(axis=(1, 3))


def centerlines(velocity, length, u_lid, samples):
    """Interpolate to x=L/2 or y=L/2, then to common sample coordinates."""
    n = velocity.shape[0]
    coordinates = (np.arange(n) + 0.5) * length / n
    mid = n // 2
    u_vertical = velocity[:, mid - 1:mid + 1, 0].mean(axis=1)
    v_horizontal = velocity[mid - 1:mid + 1, :, 1].mean(axis=0)
    # Endpoints use known wall boundary values; they are not independent validation evidence.
    support = np.r_[0.0, coordinates, length]
    u = np.interp(samples, support, np.r_[0.0, u_vertical, u_lid]) / u_lid
    v = np.interp(samples, support, np.r_[0.0, v_horizontal, 0.0]) / u_lid
    return u, v


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing N32, N64, and N128 case folders.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for comparison results.")
    parser.add_argument("--times", nargs=3, type=float, default=[9.0, 9.5, 10.0],
                        help="Three times for steady-state screening (default: 9 9.5 10 s); "
                             "only the last time is used for grid comparison and plots.")
    parser.add_argument("--length", type=float, default=0.1)
    parser.add_argument("--u-lid", type=float, default=1.0)
    args = parser.parse_args()
    gaps = np.diff(args.times)
    if args.length <= 0 or args.u_lid <= 0 or np.any(gaps <= 0):
        parser.error("Length and lid speed must be positive; the three times must be strictly increasing.")
    if not np.allclose(gaps, 0.5, rtol=0, atol=1e-8):
        parser.error("The project-specific steady-state thresholds use 0.5 s windows; keep consecutive times 0.5 s apart.")
    args.root = args.root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    sizes = (32, 64, 128)
    final_fields = {}
    all_steady = True
    print("Steady-state screening within each grid (terminal output only):")
    for n in sizes:
        case = args.root / f"N{n}"
        print(f"Reading {case}", flush=True)
        fields = read_snapshots(case, n, args.times, args.length)
        for i in range(2):
            rms, maximum = velocity_change(fields[i], fields[i + 1], args.u_lid)
            passed = rms < 1e-4 and maximum < 1e-3
            all_steady = all_steady and passed
            print(f"  t={args.times[i]:g}->{args.times[i+1]:g}: "
                  f"E_RMS={rms:.6e}, E_max={maximum:.6e}, pass={passed}")
        # Keep ONLY the last snapshot for all comparisons between grids.
        final_fields[n] = fields[-1]
    if not all_steady:
        raise SystemExit("At least one case failed the prescribed steady-state screening. Extend the simulation and check later times before comparing grids.")

    print(f"\nGrid comparison and centerline plots at t = {args.times[-1]:g} s only:")
    samples = np.linspace(0.0, args.length, 201)
    lines = {n: centerlines(final_fields[n], args.length, args.u_lid, samples) for n in sizes}
    metrics = []
    for coarse_n, fine_n in ((32, 64), (64, 128)):
        coarse = final_fields[coarse_n]
        reference = restrict_to_coarse(final_fields[fine_n], coarse_n)
        rms, maximum = velocity_change(coarse, reference, args.u_lid)
        denominator = np.linalg.norm(reference.ravel())
        if denominator == 0:
            raise ValueError("The reference velocity field is zero; the relative L2 difference is undefined.")
        relative_l2 = float(np.linalg.norm((coarse - reference).ravel()) / denominator)
        u_diff = lines[coarse_n][0] - lines[fine_n][0]
        v_diff = lines[coarse_n][1] - lines[fine_n][1]
        row = dict(coarse_N=coarse_n, fine_N=fine_n,
                   relative_L2=relative_l2, E_RMS_Ulid=rms, E_max_Ulid=maximum,
                   centerline_u_max_Ulid=float(np.max(np.abs(u_diff))),
                   centerline_v_max_Ulid=float(np.max(np.abs(v_diff))))
        metrics.append(row)
        print(f"N{coarse_n} vs N{fine_n}: relative_L2={relative_l2:.6e} "
              f"({100*relative_l2:.4f}%), E_RMS/U={rms:.6e}, E_max/U={maximum:.6e}")
        print(f"  centerline max differences / U: "
              f"u={row['centerline_u_max_Ulid']:.6e}, v={row['centerline_v_max_Ulid']:.6e}")
    write_rows(args.out / "grid_metrics.csv", metrics)

    line_rows = []
    for i, coordinate in enumerate(samples / args.length):
        row = {"s_over_L": float(coordinate)}
        for n in sizes:
            row[f"u_vertical_N{n}"] = float(lines[n][0][i])
            row[f"v_horizontal_N{n}"] = float(lines[n][1][i])
        line_rows.append(row)
    write_rows(args.out / "centerlines.csv", line_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for n, style in zip(sizes, ("--", "-.", "-")):
        axes[0].plot(lines[n][0], samples / args.length, style, label=f"{n} x {n}")
        axes[1].plot(samples / args.length, lines[n][1], style, label=f"{n} x {n}")
    axes[0].set(xlabel="u / U_lid", ylabel="y / L", title="Vertical centerline: x/L = 0.5")
    axes[1].set(xlabel="x / L", ylabel="v / U_lid", title="Horizontal centerline: y/L = 0.5")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"Cavity grid comparison at t = {args.times[-1]:g} s")
    fig.savefig(args.out / "centerlines.png", dpi=180)
    plt.close(fig)

    print(f"Saved results to {args.out}")
    print("Production-grid selection also requires time-step and benchmark checks.")


if __name__ == "__main__":
    main()
