import argparse
import csv
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np


SPLITS = {
    "train": list(range(50, 201, 10)),
    "val": [65, 105, 145, 185],
    "test": [55, 95, 135, 175],
}
CONTROLS = {
    "solver": "incompressibleFluid",
    "startFrom": "startTime",
    "startTime": "0",
    "stopAt": "endTime",
    "endTime": "10",
    "deltaT": "0.0005",
    "writeControl": "timeStep",
    "writeInterval": "200",
    "purgeWrite": "11",
    "writeFormat": "ascii",
    "writePrecision": "6",
    "writeCompression": "off",
    "timeFormat": "general",
    "timePrecision": "6",
    "runTimeModifiable": "true",
    "adjustTimeStep": "no",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def require_command(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is not on PATH. Source /opt/openfoam13/etc/bashrc first.")


def dictionary(path, entry, value=None):
    cmd = ["foamDictionary", str(path), "-entry", entry, "-writePrecision", "17"]
    cmd += ["-value"] if value is None else ["-set", str(value)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{path}, entry {entry}:\n{result.stderr or result.stdout}")
    return result.stdout.strip()


def scalar(path, entry):
    tokens = re.findall(NUMBER, dictionary(path, entry))
    if not tokens or not math.isfinite(float(tokens[-1])):
        raise ValueError(f"Invalid scalar {entry} in {path}")
    return float(tokens[-1])


def checked_viscosity(case, expected):
    actual = scalar(case / "constant/physicalProperties", "nu")
    if not all(math.isfinite(value) and value > 0 for value in (actual, expected)):
        raise ValueError("Case and manifest viscosities must be positive and finite.")
    if math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0):
        return actual
    legacy = float(format(float(format(expected, ".12g")), ".6g"))
    relative_difference = abs(actual - expected) / expected
    if math.isclose(actual, legacy, rel_tol=1e-12, abs_tol=0):
        print(
            f"NOTE {case.name}: accepted legacy six-significant-digit nu rounding; "
            f"manifest={expected:.17g}, case={actual:.17g}, "
            f"relative difference={relative_difference:.3e}. CFD files are unchanged.",
            flush=True,
        )
        return actual
    raise ValueError(
        f"Viscosity no longer matches the manifest: "
        f"case={actual:.17g}, manifest={expected:.17g}, "
        f"relative difference={relative_difference:.3e}. "
        "This is not the expected legacy six-significant-digit rounding."
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def field_at(case, target):
    matches = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if math.isclose(value, target, rel_tol=0, abs_tol=1e-8):
            if (path / "U").is_file() or (path / "U.gz").is_file():
                matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{case}: expected one saved U field at t={target:g} s.")
    return matches[0]


def check_log(case, end):
    path = case / "log.foamRun"
    if not path.is_file():
        raise ValueError(f"Missing {path}")
    text = path.read_text(errors="replace")
    failure = re.compile(
        r"FOAM FATAL|\bFloating point exception\b(?![ \t]+trapping\b)|\bSegmentation fault\b",
        re.I,
    )
    for line_number, line in enumerate(text.splitlines(), 1):
        if failure.search(line):
            raise ValueError(f"Solver failure found in {path}:{line_number}\n  {line.strip()}")
    lines = [line.strip() for line in text.splitlines()]
    times = []
    last_time_line = -1
    last_end_line = -1
    for line_number, line in enumerate(lines):
        if re.match(r"Time[ \t]*=", line):
            match = re.fullmatch(r"Time[ \t]*=[ \t]*(" + NUMBER + r")[ \t]*(?:s)?", line)
            if match is None:
                raise ValueError(
                    f"{path}:{line_number + 1}: unsupported simulation time format: {line!r}. "
                    "Expected a number optionally followed by 's'."
                )
            times.append(float(match.group(1)))
            last_time_line = line_number
        if line == "End":
            last_end_line = line_number
    tail = "\n".join(f"  {line}" for line in lines[-12:])
    if not times:
        raise ValueError(f"No simulation 'Time = ...' records found in {path}.\nLast log lines:\n{tail}")
    if last_end_line < last_time_line:
        raise ValueError(
            f"No 'End' marker after the last time record in {path}. "
            f"Last recorded time: {times[-1]:g} s.\nLast log lines:\n{tail}"
        )
    if not math.isclose(times[-1], end, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{path}: last time is {times[-1]:g} s, expected {end:g} s.")
    field_at(case, end)
    co = [float(x) for x in re.findall(r"Courant Number[^\n]*max:\s*(" + NUMBER + r")", text)]
    local = re.findall(r"sum local\s*=\s*(" + NUMBER + r")", text)
    return {
        "max_Co_in_log": max(co) if co else "",
        "last_continuity_sum_local": float(local[-1]) if local else "",
    }


def records(root, selected):
    with (root / "cases.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    known = {int(row["Re"]) for row in rows}
    if selected and not set(selected).issubset(known):
        raise ValueError(f"Unknown Re values: {sorted(set(selected) - known)}")
    return [row for row in rows if not selected or int(row["Re"]) in selected]


def prepare(args):
    template = args.template.expanduser().resolve()
    for part in ("0", "constant", "system", "constant/polyMesh"):
        if not (template / part).is_dir():
            raise FileNotFoundError(f"Missing template directory: {template / part}")
    if dictionary(template / "constant/momentumTransport", "simulationType") != "laminar":
        raise ValueError("The template must already use simulationType laminar.")
    scalar(template / "constant/physicalProperties", "nu")
    wall = dictionary(template / "0/U", "boundaryField/movingWall/value")
    vector = re.fullmatch(r"uniform\s*\(\s*(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s*\)", wall)
    if vector is None or not np.allclose([float(v) for v in vector.groups()],
                                         [args.u_lid, 0, 0], rtol=0, atol=1e-12):
        raise ValueError("The template movingWall velocity must match (U_lid, 0, 0).")
    if args.root.exists():
        raise FileExistsError(f"Choose a new batch directory; {args.root} already exists.")
    args.root.mkdir(parents=True)
    rows = []
    for split, values in SPLITS.items():
        for reynolds in values:
            nu = args.u_lid * args.length / reynolds
            rows.append(dict(case=f"Re{reynolds}", Re=reynolds, split=split, nu=nu,
                             N=args.n, L=args.length, U_lid=args.u_lid,
                             deltaT=float(CONTROLS["deltaT"]),
                             endTime=float(CONTROLS["endTime"])))
    rows.sort(key=lambda row: row["Re"])
    for row in rows:
        case = args.root / row["case"]
        case.mkdir()
        for part in ("0", "constant", "system"):
            shutil.copytree(template / part, case / part)
        dictionary(case / "constant/physicalProperties", "nu",
                   f"[0 2 -1 0 0 0 0] {row['nu']:.17g}")
        for entry, value in CONTROLS.items():
            dictionary(case / "system/controlDict", entry, value)
        (case / "case.foam").touch()
        print(f"Prepared {row['case']}: split={row['split']}, nu={row['nu']:.12g}", flush=True)
    write_csv(args.root / "cases.csv", rows)
    print(f"Prepared {len(rows)} cases. Manifest: {args.root / 'cases.csv'}")


def run(args):
    require_command("foamRun")
    failed = []
    for row in records(args.root, args.re):
        case = args.root / row["case"]
        try:
            end = scalar(case / "system/controlDict", "endTime")
            try:
                check_log(case, end)
            except ValueError as error:
                completion_error = str(error)
            else:
                print(f"SKIP {row['case']}: completed run found.", flush=True)
                continue
            for path in case.iterdir():
                if path.is_dir():
                    try:
                        positive = float(path.name) > 0
                    except ValueError:
                        positive = False
                    if positive:
                        raise ValueError("Existing time directories found without a complete run. "
                                         "Inspect the log and resume this case manually.\n"
                                         f"Completion check: {completion_error}")
            print(f"RUN {row['case']} ...", flush=True)
            started = time.perf_counter()
            with (case / "log.foamRun").open("w") as log:
                result = subprocess.run(["foamRun", "-case", str(case)],
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise ValueError(f"foamRun exit code {result.returncode}; inspect {case / 'log.foamRun'}")
            info = check_log(case, end)
            print(f"DONE {row['case']}: {time.perf_counter()-started:.1f} s wall time, "
                  f"max Co={info['max_Co_in_log']}", flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            failed.append(row["case"])
            print(f"FAILED {row['case']}: {error}", flush=True)
    if failed:
        raise SystemExit(f"Failed/incomplete cases: {', '.join(failed)}")


def check(args):
    failed = []
    for row in records(args.root, args.re):
        case = args.root / row["case"]
        try:
            end = scalar(case / "system/controlDict", "endTime")
            check_log(case, end)
            print(f"PASS {row['case']}: log reached {end:g} s, End found, final U exists.", flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            failed.append(row["case"])
            print(f"FAILED {row['case']}: {error}", flush=True)
    if failed:
        raise SystemExit(f"Failed/incomplete cases: {', '.join(failed)}")


def export(args):
    from compare_grids import read_snapshots, velocity_change
    if args.out.exists():
        raise FileExistsError(f"Choose a new export directory; {args.out} already exists.")
    pending = []
    failed = []
    for row in records(args.root, args.re):
        case = args.root / row["case"]
        try:
            n, length, lid = int(row["N"]), float(row["L"]), float(row["U_lid"])
            end = scalar(case / "system/controlDict", "endTime")
            dt = scalar(case / "system/controlDict", "deltaT")
            if not math.isclose(dt, float(row["deltaT"]), rel_tol=1e-10, abs_tol=0):
                raise ValueError("Time step no longer matches the manifest.")
            adjustment = dictionary(case / "system/controlDict", "adjustTimeStep")
            if adjustment.lower() not in ("no", "false", "off", "0"):
                raise ValueError("Expected a fixed time step.")
            info = check_log(case, end)
            if end < 1:
                raise ValueError("Need at least 1 s of saved history.")
            nu = checked_viscosity(case, float(row["nu"]))
            if dictionary(case / "constant/momentumTransport", "simulationType") != "laminar":
                raise ValueError("Expected laminar momentum transport.")
            fields = read_snapshots(case, n, [end-1, end-0.5, end], length)
            changes = [velocity_change(fields[i], fields[i+1], lid) for i in range(2)]
            rms = max(pair[0] for pair in changes)
            maximum = max(pair[1] for pair in changes)
            print(f"{row['case']}: temporal RMS/U={rms:.3e}, max/U={maximum:.3e}", flush=True)
            if rms >= args.steady_rms_tol or maximum >= args.steady_max_tol:
                raise ValueError("Failed preliminary steady-state screening; inspect trends and logs.")
            velocity = fields[-1]
            if np.max(np.abs(velocity[..., 2])) / lid > 1e-8:
                raise ValueError("Non-negligible out-of-plane velocity for this 2D dataset.")
            coordinate = (np.arange(n) + 0.5) / n
            x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
            inputs = np.column_stack((x.ravel(), y.ravel(), np.full(n*n, int(row["Re"]))))
            targets = velocity[..., :2].reshape(n*n, 2) / lid
            record = dict(row)
            record.update(nu_actual=nu, Re_actual=lid*length/nu,
                          endTime=end, deltaT=dt, snapshot_time=end,
                          file=f"{row['case']}.npz", points=n*n,
                          steady_rms_max=rms, steady_max_max=maximum,
                          steady_rms_tol=args.steady_rms_tol,
                          steady_max_tol=args.steady_max_tol, **info)
            pending.append((record, inputs, targets))
        except (ValueError, RuntimeError, OSError) as error:
            failed.append(row["case"])
            print(f"REJECT {row['case']}: {error}", flush=True)
    if failed:
        raise SystemExit(f"No dataset written. Resolve these cases first: {', '.join(failed)}")
    args.out.mkdir(parents=True)
    for record, inputs, targets in pending:
        np.savez_compressed(args.out / record["file"], inputs=inputs, targets=targets,
                            input_names=np.array(["x_over_L", "y_over_L", "Re"]),
                            target_names=np.array(["u_over_Ulid", "v_over_Ulid"]),
                            Re=int(record["Re"]), nu=float(record["nu"]),
                            nu_actual=record["nu_actual"], Re_actual=record["Re_actual"],
                            L=float(record["L"]), U_lid=float(record["U_lid"]),
                            N=int(record["N"]), time=record["snapshot_time"], split=record["split"])
    write_csv(args.out / "dataset_index.csv", [record for record, _, _ in pending])
    print(f"Exported {len(pending)} cases to {args.out}")
    print("Numerical reference checks and Re200 grid/time-step checks are separate requirements.")
    print("A passed temporal screening is not proof of complete CFD validation.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--length", type=float, default=0.1)
    p.add_argument("--u-lid", type=float, default=1.0)
    p = sub.add_parser("run")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--re", nargs="+", type=int)
    p = sub.add_parser("check", help="Check existing logs and final U files without running the solver.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--re", nargs="+", type=int)
    p = sub.add_parser("export")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--re", nargs="+", type=int)
    p.add_argument("--steady-rms-tol", type=float, default=1e-4)
    p.add_argument("--steady-max-tol", type=float, default=1e-3)
    args = parser.parse_args()
    require_command("foamDictionary")
    args.root = args.root.expanduser().resolve()
    if args.action == "prepare":
        values = [args.length, args.u_lid]
        if args.n < 2 or not all(math.isfinite(x) and x > 0 for x in values):
            parser.error("Use positive finite length and U_lid, and N >= 2.")
        prepare(args)
    elif args.action == "run":
        run(args)
    elif args.action == "check":
        check(args)
    else:
        if not all(math.isfinite(x) and x > 0 for x in [args.steady_rms_tol, args.steady_max_tol]):
            parser.error("Steady-state screening thresholds must be positive and finite.")
        args.out = args.out.expanduser().resolve()
        export(args)


if __name__ == "__main__":
    main()
