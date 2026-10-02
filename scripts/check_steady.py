#!/usr/bin/env python3
"""Compare saved OpenFOAM velocity fields on a fixed, equal-volume mesh.

Supports ASCII U/U.gz with uniform or nonuniform List<vector> internal fields.
Requires matching cell numbering and serial or reconstructed case results.
"""

from pathlib import Path
import gzip
import re
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cfd/cavity_Re100_N64_T5e-3"
TIMES = ["9", "9.5", "10"]
N_CELLS = 64 * 64 * 1
U_LID = 1.0  # m/s


def read_velocity(path, n_cells):
    """Return internal velocities as an (n_cells, 3) array in cell order."""
    path = Path(path)
    if not path.is_file():
        path = path.with_name(path.name + ".gz")
    if not path.is_file():
        raise FileNotFoundError(f"Neither U nor U.gz found in {path.parent}")

    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    text = raw.decode("latin-1")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    if not re.search(r"\bformat\s+ascii\s*;", text):
        raise ValueError(f"{path}: only format ascii is supported; binary is not supported.")
    if not re.search(r"\bclass\s+volVectorField\s*;", text):
        raise ValueError(f"{path}: expected a volVectorField velocity file.")

    match = re.search(r"\binternalField\s+(.+?)\s*;", text, flags=re.S)
    if match is None:
        raise ValueError(f"{path}: no complete internalField found.")
    field = match.group(1).strip()

    uniform = re.fullmatch(r"uniform\s*\(([^()]*)\)", field, flags=re.S)
    if uniform:
        vector = np.array([float(x) for x in uniform.group(1).split()])
        if vector.shape != (3,):
            raise ValueError(f"{path}: uniform velocity must have three components.")
        velocity = np.tile(vector, (n_cells, 1))
    else:
        match = re.fullmatch(
            r"nonuniform\s+List<vector>\s+(\d+)\s*\((.*)\)", field, flags=re.S
        )
        if match is None:
            raise ValueError(f"{path}: unsupported internalField format.")
        declared_count = int(match.group(1))
        body = match.group(2)
        vectors = re.findall(r"\(([^()]*)\)", body)
        if re.sub(r"\([^()]*\)", "", body).strip():
            raise ValueError(f"{path}: unrecognized content in the vector list.")
        velocity = np.array([[float(x) for x in row.split()] for row in vectors])
        if declared_count != n_cells or velocity.shape != (n_cells, 3):
            raise ValueError(
                f"{path}: expected {n_cells} vectors; declared {declared_count}; "
                f"actual array shape is {velocity.shape}."
            )
    if not np.isfinite(velocity).all():
        raise ValueError(f"{path}: velocity contains NaN or inf; comparison aborted.")
    return velocity


def main():
    times = np.array([float(t) for t in TIMES])
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("TIMES must contain at least three finite, strictly increasing times.")
    if N_CELLS <= 0 or not np.isfinite(U_LID) or U_LID <= 0:
        raise ValueError("N_CELLS and U_LID must be positive; U_LID must be finite.")

    fields = [read_velocity(CASE / t / "U", N_CELLS) for t in TIMES]

    for k in range(len(TIMES) - 1):
        delta = fields[k + 1] - fields[k]
        change = np.linalg.norm(delta, axis=1)
        e_max = float(np.max(change)) / U_LID
        e_rms = float(np.sqrt(np.mean(change**2))) / U_LID
        print(f"\n{TIMES[k]} -> {TIMES[k + 1]} s")
        print(f"  E_max = {e_max:.8e}")
        print(f"  E_rms = {e_rms:.8e}")


if __name__ == "__main__":
    main()
