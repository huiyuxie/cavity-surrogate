import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SPLITS = {
    "train": list(range(50, 201, 10)),
    "val": [65, 105, 145, 185],
    "test": [55, 95, 135, 175],
}


def read_index(root):
    root = Path(root).expanduser().resolve()
    with (root / "dataset_index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The dataset index is empty.")
    required = {"case", "Re", "split", "file", "points"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing index columns: {sorted(required - set(rows[0]))}")
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate files in dataset_index.csv.")
    if len({float(row["Re"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate Re values in dataset_index.csv.")
    if any(row["split"] not in EXPECTED_SPLITS for row in rows):
        raise ValueError("Expected split names: train, val, test.")
    for split, expected in EXPECTED_SPLITS.items():
        actual = sorted(float(row["Re"]) for row in rows if row["split"] == split)
        if actual != expected:
            raise ValueError(f"{split}: expected Re={expected}, found Re={actual}.")
    for row in rows:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Missing or invalid dataset file: {row['file']}")
    return sorted(rows, key=lambda row: float(row["Re"]))


def read_case(root, row):
    path = Path(root) / row["file"]
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["inputs"], dtype=np.float64)
        y = np.asarray(data["targets"], dtype=np.float64)
        n = int(data["N"].item())
        if float(data["Re"].item()) != float(row["Re"]):
            raise ValueError(f"{path}: Re does not match the index.")
        if str(data["split"].item()) != row["split"]:
            raise ValueError(f"{path}: split does not match the index.")
        if data["input_names"].tolist() != ["x_over_L", "y_over_L", "Re"]:
            raise ValueError(f"{path}: unexpected input names.")
        if data["target_names"].tolist() != ["u_over_Ulid", "v_over_Ulid"]:
            raise ValueError(f"{path}: unexpected target names.")
    points = int(row["points"])
    if n < 2 or points != n * n or x.shape != (points, 3) or y.shape != (points, 2):
        raise ValueError(f"{path}: inconsistent N, point count or array shape.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{path}: NaN or inf found.")
    if not np.allclose(x[:, 2], float(row["Re"]), rtol=0, atol=1e-12):
        raise ValueError(f"{path}: input Re does not match the index.")
    c = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(c, c, indexing="xy")
    expected_xy = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.allclose(x[:, :2], expected_xy, rtol=0, atol=1e-12):
        raise ValueError(f"{path}: expected the exporter's uniform cell-center coordinates and order.")
    return {"row": row, "x": x, "y": y, "n": n}


def normalize_inputs(x, normalization):
    x = np.array(x, dtype=np.float64, copy=True)
    lower = float(normalization["re_min"])
    upper = float(normalization["re_max"])
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("Invalid Re normalization limits.")
    x[:, :2] = 2.0 * x[:, :2] - 1.0
    x[:, 2] = 2.0 * (x[:, 2] - lower) / (upper - lower) - 1.0
    return x.astype(np.float32)


def load_split(data_root, split, normalization_path):
    if split not in EXPECTED_SPLITS:
        raise ValueError("Use train, val or test.")
    data_root = Path(data_root).expanduser().resolve()
    rows = [row for row in read_index(data_root) if row["split"] == split]
    with Path(normalization_path).open() as stream:
        normalization = json.load(stream)
    cases = [read_case(data_root, row) for row in rows]
    x = np.concatenate([normalize_inputs(case["x"], normalization) for case in cases])
    y = np.concatenate([case["y"].astype(np.float32) for case in cases])
    re_per_point = np.concatenate([
        np.full(len(case["x"]), float(case["row"]["Re"])) for case in cases
    ])
    return x, y, re_per_point


def interpolate_case(train_cases, query):
    reynolds = float(query["row"]["Re"])
    train_re = np.array([float(case["row"]["Re"]) for case in train_cases])
    right = int(np.searchsorted(train_re, reynolds))
    if right == 0 or right == len(train_re):
        raise ValueError("This baseline only supports interpolation between training Re values.")
    lower, upper = train_cases[right - 1], train_cases[right]
    for case in (lower, upper):
        if case["x"].shape != query["x"].shape or not np.allclose(
            case["x"][:, :2], query["x"][:, :2], rtol=0, atol=1e-12
        ):
            raise ValueError("Baseline cases must have the same coordinates in the same order.")
    re_lower, re_upper = train_re[right - 1], train_re[right]
    weight = (reynolds - re_lower) / (re_upper - re_lower)
    prediction = (1.0 - weight) * lower["y"] + weight * upper["y"]
    return prediction, re_lower, re_upper, weight


def velocity_errors(prediction, reference):
    delta = prediction - reference
    point_error = np.linalg.norm(delta, axis=1)
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        raise ValueError("Relative L2 is undefined for an all-zero reference field.")
    return {
        "relative_L2": float(np.linalg.norm(delta) / denominator),
        "E_RMS_Ulid": float(np.sqrt(np.mean(point_error ** 2))),
        "E_max_Ulid": float(point_error.max()),
    }


def plot_example(query, prediction, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = query["n"]
    reference = query["y"][:, 0].reshape(n, n)
    predicted = prediction[:, 0].reshape(n, n)
    error = np.abs(predicted - reference)
    vmin, vmax = min(reference.min(), predicted.min()), max(reference.max(), predicted.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, field, title in zip(axes, [reference, predicted, error],
                                ["CFD u/U_lid", "Linear baseline u/U_lid", "Absolute error in u/U_lid"]):
        limits = {} if ax is axes[2] else {"vmin": vmin, "vmax": vmax}
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                       cmap="viridis" if ax is axes[2] else "coolwarm", **limits)
        ax.set(title=title, xlabel="x/L", ylabel="y/L")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Test Re={float(query['row']['Re']):g}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.data.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out == root or root.is_relative_to(out) or out.is_relative_to(root):
        raise ValueError("Use a separate results directory outside the source data directory.")
    rows = read_index(root)
    cases = {split: [read_case(root, row) for row in rows if row["split"] == split]
             for split in ("train", "test")}
    train_re = [float(case["row"]["Re"]) for case in cases["train"]]
    normalization = {
        "input_names": ["x_over_L", "y_over_L", "Re"],
        "xy_transform": "2 * xy_over_L - 1",
        "re_transform": "2 * (Re - re_min) / (re_max - re_min) - 1",
        "re_min": min(train_re), "re_max": max(train_re),
        "target_names": ["u_over_Ulid", "v_over_Ulid"],
        "target_transform": "identity; already divided by U_lid during CFD export",
    }
    metrics = []
    examples = []
    for query in cases["test"]:
        prediction, lower, upper, weight = interpolate_case(cases["train"], query)
        metric = {"case": query["row"]["case"], "Re": float(query["row"]["Re"]),
                  "split": "test", "Re_lower_train": lower, "Re_upper_train": upper,
                  "weight_upper": weight, **velocity_errors(prediction, query["y"])}
        metrics.append(metric)
        examples.append((query, prediction))
        print(f"{metric['case']}: train Re={lower:g},{upper:g}; "
              f"relative_L2={100 * metric['relative_L2']:.6f}%; "
              f"RMS/U_lid={metric['E_RMS_Ulid']:.6e}; max/U_lid={metric['E_max_Ulid']:.6e}")
    out.mkdir(parents=True, exist_ok=True)
    splits = {split: [{"case": row["case"], "Re": float(row["Re"]), "file": row["file"]}
                       for row in rows if row["split"] == split] for split in EXPECTED_SPLITS}
    for name, content in [("splits.json", splits), ("normalization.json", normalization)]:
        (out / name).write_text(json.dumps(content, indent=2) + "\n")
    with (out / "baseline_test_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for query, prediction in examples:
        plot_example(query, prediction, out / f"baseline_test_{query['row']['case']}.png")
    for split in EXPECTED_SPLITS:
        points = sum(int(row["points"]) for row in rows if row["split"] == split)
        print(f"{split}: {len(splits[split])} cases, {points} points, "
              f"Re={[int(row['Re']) for row in splits[split]]}")
    print(f"Outputs: {out}")
    print("Only train and test fields were loaded. Validation fields were not evaluated.")


if __name__ == "__main__":
    main()
