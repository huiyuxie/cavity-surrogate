import argparse
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from prepare_ml_data import load_split, normalize_inputs, read_case, read_index, velocity_errors


def build_model(model_kwargs):
    from physicsnemo.models.mlp import FullyConnected

    return FullyConnected(**model_kwargs)


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or check your PyTorch installation.")
    return torch.device(name)


def create_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_normalization(path):
    with Path(path).expanduser().open() as stream:
        normalization = json.load(stream)
    if normalization["input_names"] != ["x_over_L", "y_over_L", "Re"]:
        raise ValueError("Unexpected normalization input names.")
    if normalization["target_names"] != ["u_over_Ulid", "v_over_Ulid"]:
        raise ValueError("Unexpected normalization target names.")
    if float(normalization["re_min"]) != 50 or float(normalization["re_max"]) != 200:
        raise ValueError("Expected the existing Re50-200 normalization, including for single-case fitting.")
    return normalization


@torch.inference_mode()
def predict_array(model, x, device, batch_size=4096):
    model.eval()
    predictions = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(device)
        predictions.append(model(batch).cpu().numpy())
    prediction = np.concatenate(predictions)
    if prediction.shape != (len(x), 2) or not np.isfinite(prediction).all():
        raise ValueError("Invalid model predictions.")
    return prediction


def evaluate(model, x, y, device):
    prediction = predict_array(model, x, device)
    delta = prediction.astype(np.float64) - y.astype(np.float64)
    metrics = {"mse": float(np.mean(delta ** 2)), **velocity_errors(prediction, y)}
    return metrics, prediction


def plot_history(history, out, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].semilogy(epochs, [max(row["train_mse"], 1e-16) for row in history], label="Train")
    axes[1].plot(epochs, [100 * row["train_relative_L2"] for row in history], label="Train")
    if mode == "train":
        axes[0].semilogy(epochs, [max(row["val_mse"], 1e-16) for row in history], label="Validation")
        axes[1].plot(epochs, [100 * row["val_relative_L2"] for row in history], label="Validation")
    axes[0].set(ylabel="MSE", title="End-of-epoch loss")
    axes[1].set(ylabel="Relative L2 (%)", title="Velocity prediction error")
    axes[2].semilogy(epochs, [row["lr"] for row in history])
    axes[2].set(ylabel="Learning rate", title="Adam learning rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_field(inputs, prediction, reference, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(round(math.sqrt(len(inputs))))
    if n * n != len(inputs):
        raise ValueError("Field plotting expects a uniform square grid.")
    columns = 3 if reference is not None else 1
    fig, axes = plt.subplots(2, columns, figsize=(4.2 * columns, 7.5),
                             squeeze=False, constrained_layout=True)
    for component, name in enumerate(["u/U_lid", "v/U_lid"]):
        predicted = prediction[:, component].reshape(n, n)
        if reference is None:
            fields, labels = [predicted], [f"NN {name}"]
            limits = [(predicted.min(), predicted.max())]
        else:
            actual = reference[:, component].reshape(n, n)
            error = np.abs(predicted - actual)
            vmin, vmax = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            fields, labels = [actual, predicted, error], [f"CFD {name}", f"NN {name}", f"Absolute error in {name}"]
            limits = [(vmin, vmax), (vmin, vmax), (0, max(float(error.max()), 1e-15))]
        for column, (field, label, bounds) in enumerate(zip(fields, labels, limits)):
            ax = axes[component, column]
            im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                           cmap="viridis" if column == 2 else "coolwarm",
                           vmin=bounds[0], vmax=bounds[1])
            ax.set(title=label, xlabel="x/L", ylabel="y/L")
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_checkpoint(path, model, model_kwargs, normalization, config, epoch, metric):
    torch.save({
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "model_kwargs": model_kwargs,
        "normalization": normalization,
        "config": config,
        "epoch": epoch,
        "selection_metric": "train_mse" if config["mode"] == "single" else "val_mse",
        "selection_value": float(metric),
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train(args):
    if min(args.epochs, args.batch_size, args.width, args.layers, args.threads, args.log_every) < 1:
        raise ValueError("Epochs, batch size, width, layers, threads and log interval must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    data_root = args.data.expanduser().resolve()
    normalization = read_normalization(args.normalization)
    if args.mode == "single":
        rows = [row for row in read_index(data_root)
                if row["split"] == "train" and float(row["Re"]) == args.re]
        if len(rows) != 1:
            raise ValueError("Single-case fitting must use an existing training Re, e.g. --re 100.")
        case = read_case(data_root, rows[0])
        x_train = normalize_inputs(case["x"], normalization)
        y_train = case["y"].astype(np.float32)
        re_train = np.full(len(x_train), args.re)
        x_val = y_val = re_val = None
    else:
        x_train, y_train, re_train = load_split(data_root, "train", args.normalization)
        x_val, y_val, re_val = load_split(data_root, "val", args.normalization)
    model_kwargs = dict(in_features=3, out_features=2, num_layers=args.layers,
                        layer_size=args.width, activation_fn="tanh", weight_norm=False,
                        skip_connections=False, adaptive_activations=False)
    model = build_model(model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True, num_workers=0,
                        drop_last=False, generator=generator)
    config = {"mode": args.mode, "data": str(data_root), "epochs": args.epochs,
              "batch_size": args.batch_size, "initial_lr": args.lr, "seed": args.seed,
              "device": str(device), "threads": args.threads, "model_kwargs": model_kwargs,
              "train_Re": np.unique(re_train).tolist(),
              "val_Re": [] if re_val is None else np.unique(re_val).tolist(),
              "torch_version": str(torch.__version__), "physicsnemo_version": version("nvidia-physicsnemo")}
    out = create_output(args.out)
    write_json(out / "config.json", config)
    print(f"Mode={args.mode}, device={device}, parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    print(f"Train points={len(x_train)}, batches/epoch={len(loader)}, train Re={config['train_Re']}", flush=True)
    if x_val is not None:
        print(f"Validation points={len(x_val)}, validation Re={config['val_Re']}", flush=True)
    print("Test velocity fields are not loaded by training.", flush=True)
    history = []
    best_metric = float("inf")
    best_epoch = 0
    started = time.perf_counter()

    def record(epoch, used_lr, epoch_seconds):
        nonlocal best_metric, best_epoch
        train_metrics, _ = evaluate(model, x_train, y_train, device)
        val_metrics = evaluate(model, x_val, y_val, device)[0] if x_val is not None else None
        row = {"epoch": epoch, "lr": used_lr,
               "train_mse": train_metrics["mse"],
               "train_relative_L2": train_metrics["relative_L2"],
               "train_E_RMS_Ulid": train_metrics["E_RMS_Ulid"],
               "val_mse": "" if val_metrics is None else val_metrics["mse"],
               "val_relative_L2": "" if val_metrics is None else val_metrics["relative_L2"],
               "val_E_RMS_Ulid": "" if val_metrics is None else val_metrics["E_RMS_Ulid"],
               "epoch_seconds": epoch_seconds, "elapsed_seconds": time.perf_counter() - started}
        metric = train_metrics["mse"] if val_metrics is None else val_metrics["mse"]
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            save_checkpoint(out / "best.pt", model, model_kwargs, normalization, config, epoch, metric)
        history.append(row)
        writer.writerow(row)
        stream.flush()
        if epoch == 0 or epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            message = f"Epoch {epoch}/{args.epochs}: train MSE={row['train_mse']:.6e}, train L2={100*row['train_relative_L2']:.3f}%"
            if val_metrics is not None:
                message += f", val MSE={row['val_mse']:.6e}, val L2={100*row['val_relative_L2']:.3f}%"
            print(message + f", lr={used_lr:.2e}, elapsed={row['elapsed_seconds']:.1f}s, best epoch={best_epoch}", flush=True)
        return metric

    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "lr", "train_mse", "train_relative_L2",
            "train_E_RMS_Ulid", "val_mse", "val_relative_L2", "val_E_RMS_Ulid", "epoch_seconds", "elapsed_seconds"])
        writer.writeheader()
        metric = record(0, args.lr, 0.0)
        epoch = 0
        interrupted = False
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()
                model.train()
                used_lr = optimizer.param_groups[0]["lr"]
                for x_batch, y_batch in loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x_batch)
                    loss = loss_fn(prediction, y_batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss; inspect the data and learning rate.")
                    loss.backward()
                    optimizer.step()
                metric = record(epoch, used_lr, time.perf_counter() - epoch_start)
                scheduler.step(metric)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; keeping the best model from completed evaluations.", flush=True)
        completed_epoch = int(history[-1]["epoch"])
        if not interrupted:
            save_checkpoint(out / "last.pt", model, model_kwargs, normalization, config, completed_epoch,
                            evaluate(model, x_train if x_val is None else x_val,
                                     y_train if y_val is None else y_val, device)[0]["mse"])
    plot_history(history, out / "learning_curves.png", args.mode)
    best_model, checkpoint = load_checkpoint(out / "best.pt", device)
    if args.mode == "single":
        metrics, prediction = evaluate(best_model, x_train, y_train, device)
        write_csv(out / "single_fit_errors.csv", [{"Re": args.re, **metrics}])
        plot_field(case["x"], prediction, y_train, out / f"single_fit_Re{args.re:g}.png",
                   f"Single training case Re={args.re:g}; fitting diagnostic")
    else:
        prediction = predict_array(best_model, x_val, device)
        rows = []
        for reynolds in np.unique(re_val):
            mask = re_val == reynolds
            rows.append({"Re": float(reynolds), **velocity_errors(prediction[mask], y_val[mask])})
        write_csv(out / "best_val_errors.csv", rows)
    write_json(out / "summary.json", {"best_epoch": checkpoint["epoch"],
        "selection_metric": checkpoint["selection_metric"], "selection_value": checkpoint["selection_value"],
        "completed_epochs": completed_epoch, "interrupted": interrupted,
        "total_wall_seconds": time.perf_counter() - started})
    print(f"Best model: {out / 'best.pt'}", flush=True)
    print(f"Curves: {out / 'learning_curves.png'}", flush=True)


def predict(args):
    if args.n < 2 or args.threads < 1 or not math.isfinite(args.re):
        raise ValueError("Use N >= 2, positive threads and a finite Re.")
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    normalization = checkpoint["normalization"]
    if not float(normalization["re_min"]) <= args.re <= float(normalization["re_max"]):
        raise ValueError("This project predicts within Re50-200; extrapolation is not supported here.")
    c = (np.arange(args.n) + 0.5) / args.n
    xx, yy = np.meshgrid(c, c, indexing="xy")
    inputs = np.column_stack((xx.ravel(), yy.ravel(), np.full(args.n * args.n, args.re)))
    prediction = predict_array(model, normalize_inputs(inputs, normalization), device)
    reference = None
    metrics = None
    if args.data is not None:
        root = args.data.expanduser().resolve()
        matches = [row for row in read_index(root) if float(row["Re"]) == args.re]
        if matches:
            case = read_case(root, matches[0])
            if case["x"].shape != inputs.shape or not np.allclose(case["x"], inputs, rtol=0, atol=1e-12):
                raise ValueError("Use --n matching the CFD reference grid for a direct comparison.")
            reference = case["y"]
            metrics = {"Re": args.re, "split": matches[0]["split"], **velocity_errors(prediction, reference)}
    out = create_output(args.out)
    np.savez_compressed(out / "prediction.npz", inputs=inputs, prediction=prediction,
                        Re=args.re, N=args.n, input_names=np.array(["x_over_L", "y_over_L", "Re"]),
                        target_names=np.array(["u_over_Ulid", "v_over_Ulid"]))
    plot_field(inputs, prediction, reference, out / "prediction.png", f"Neural network prediction Re={args.re:g}")
    if metrics is not None:
        write_csv(out / "prediction_errors.csv", [metrics])
        print(f"CFD comparison: relative L2={100*metrics['relative_L2']:.4f}%, "
              f"RMS/U_lid={metrics['E_RMS_Ulid']:.6e}, max/U_lid={metrics['E_max_Ulid']:.6e}")
    print(f"Prediction outputs: {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("single", "train"):
        p = sub.add_parser(mode)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--normalization", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--epochs", type=int, default=500 if mode == "single" else 1000)
        p.add_argument("--batch-size", type=int, default=1024)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--layers", type=int, default=4)
        p.add_argument("--width", type=int, default=64)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--log-every", type=int, default=20)
        if mode == "single":
            p.add_argument("--re", type=float, default=100)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "predict":
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
