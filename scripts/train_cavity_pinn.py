"""Train a steady, parameterized cavity MLP with velocity data + PDE loss.

Inputs are (x/L, y/L, nominal Re); outputs are (u/U_lid, v/U_lid,
p/(rho*U_lid**2)). Only velocity has supervised labels. No BC or IC loss is
used. See docs/pinn.md for installation, training, and prediction commands.
"""

import argparse
import csv
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from prepare_ml_data import read_case, read_index, velocity_errors
from train_cavity import (
    build_model, create_output, evaluate, get_device, plot_field, plot_history,
    predict_array, read_normalization, save_checkpoint, write_csv, write_json,
)


PDE_NAMES = ("continuity", "momentum_x", "momentum_y")


class CavityPINN(torch.nn.Module):
    """Normalize inside the graph so derivatives use the original unit square."""

    def __init__(self, model_kwargs, normalization):
        super().__init__()
        self.network = build_model(model_kwargs)
        self.re_min = float(normalization["re_min"])
        self.re_max = float(normalization["re_max"])

    def forward(self, inputs):
        xy = 2.0 * inputs[:, :2] - 1.0
        re = 2.0 * (inputs[:, 2:3] - self.re_min) / (self.re_max - self.re_min) - 1.0
        return self.network(torch.cat((xy, re), dim=1))


class VelocityView(torch.nn.Module):
    """Reuse the baseline's two-component evaluation and plotting helpers."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)[:, :2]


def make_physics_informer(device):
    try:
        from sympy import Function, Symbol
        from physicsnemo.sym.eq.pde import PDE
        from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    except ImportError as exc:
        raise RuntimeError(
            "PDE training requires PhysicsNeMo Sym. Install docs/requirements-pinn.txt "
            "in your existing environment; see docs/pinn.md."
        ) from exc

    class SteadyNavierStokes(PDE):
        def __init__(self):
            self.dim = 2
            x, y, re = Symbol("x"), Symbol("y"), Symbol("Re")
            u, v, p = (Function(name)(x, y) for name in ("u", "v", "p"))
            # Re is an independent parameter, spatially constant within each case.
            self.equations = {
                "continuity": u.diff(x) + v.diff(y),
                "momentum_x": u * u.diff(x) + v * u.diff(y) + p.diff(x)
                - (u.diff(x, 2) + u.diff(y, 2)) / re,
                "momentum_y": u * v.diff(x) + v * v.diff(y) + p.diff(y)
                - (v.diff(x, 2) + v.diff(y, 2)) / re,
            }

    return PhysicsInformer(
        required_outputs=list(PDE_NAMES), equations=SteadyNavierStokes(),
        grad_method="autodiff", device=device,
    )


def pde_residuals(model, informer, coordinates, reynolds, coefficient_re=None):
    """Differentiate in x/L,y/L only, holding the nominal Re input fixed."""
    xy = coordinates.detach().requires_grad_(True)
    fields = model(torch.cat((xy, reynolds), dim=1))
    effective_re = reynolds if coefficient_re is None else coefficient_re
    if informer is None:
        return autograd_residuals(fields, xy, effective_re)
    return informer.forward({
        "coordinates": xy,
        "u": fields[:, 0:1], "v": fields[:, 1:2], "p": fields[:, 2:3],
        "Re": effective_re,
    })


def spatial_gradient(field, coordinates):
    """Keep the graph for PDE backpropagation, including zero second derivatives."""
    if not field.requires_grad:
        return coordinates * 0
    gradient = torch.autograd.grad(field, coordinates, torch.ones_like(field),
                                   create_graph=True, allow_unused=True)[0]
    return coordinates * 0 if gradient is None else gradient


def autograd_residuals(fields, coordinates, reynolds):
    """The same dimensionless equations without the optional Sym dependency."""
    u, v, p = fields.split(1, dim=1)
    du, dv, dp = (spatial_gradient(field, coordinates) for field in (u, v, p))
    lap_u = (spatial_gradient(du[:, 0:1], coordinates)[:, 0:1]
             + spatial_gradient(du[:, 1:2], coordinates)[:, 1:2])
    lap_v = (spatial_gradient(dv[:, 0:1], coordinates)[:, 0:1]
             + spatial_gradient(dv[:, 1:2], coordinates)[:, 1:2])
    return {
        "continuity": du[:, 0:1] + dv[:, 1:2],
        "momentum_x": u*du[:, 0:1] + v*du[:, 1:2] + dp[:, 0:1] - lap_u/reynolds,
        "momentum_y": u*dv[:, 0:1] + v*dv[:, 1:2] + dp[:, 1:2] - lap_v/reynolds,
    }


def load_raw_split(root, rows, split):
    cases = [read_case(root, row) for row in rows if row["split"] == split]
    inputs = np.concatenate([case["x"] for case in cases]).astype(np.float32)
    targets = np.concatenate([case["y"] for case in cases]).astype(np.float32)
    return inputs, targets


def training_reynolds(rows, device):
    training = [row for row in rows if row["split"] == "train"]
    nominal = [float(row["Re"]) for row in training]
    # Respect the legacy viscosity rounding recorded by the CFD exporter.
    actual = [float(row.get("Re_actual") or row["Re"]) for row in training]
    if not all(math.isfinite(re) and re > 0 for re in nominal + actual):
        raise ValueError("Training Reynolds numbers must be finite and positive.")
    return (torch.tensor(nominal, dtype=torch.float32, device=device)[:, None],
            torch.tensor(actual, dtype=torch.float32, device=device)[:, None])


def sample_interior(count, nominal_re, actual_re, generator):
    device = nominal_re.device
    # Strictly interior points; this sampling does not create a boundary loss.
    xy = 1e-6 + (1.0 - 2e-6) * torch.rand(count, 2, device=device, generator=generator)
    indices = torch.randint(len(nominal_re), (count,), device=device, generator=generator)
    return xy, nominal_re[indices], actual_re[indices]


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def load_pinn_checkpoint(path, device):
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if checkpoint.get("config", {}).get("objective") != "velocity_data+pde":
        raise ValueError("Use a checkpoint produced by train_cavity_pinn.py.")
    model = CavityPINN(checkpoint["model_kwargs"], checkpoint["normalization"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train(args):
    if min(args.epochs, args.batch_size, args.pde_points, args.layers, args.width,
           args.threads, args.log_every) < 1:
        raise ValueError("Epochs, batch sizes, model sizes, threads and log interval must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if not math.isfinite(args.pde_weight) or args.pde_weight < 0:
        raise ValueError("PDE weight must be finite and nonnegative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    root = args.data.expanduser().resolve()
    normalization = read_normalization(args.normalization)
    rows = read_index(root)
    x_train, y_train = load_raw_split(root, rows, "train")
    x_val, y_val = load_raw_split(root, rows, "val")
    nominal_re, actual_re = training_reynolds(rows, device)
    model_kwargs = dict(in_features=3, out_features=3, num_layers=args.layers,
                        layer_size=args.width, activation_fn="tanh", weight_norm=False,
                        skip_connections=False, adaptive_activations=False)
    model = CavityPINN(model_kwargs, normalization).to(device)
    velocity_model = VelocityView(model)
    informer = (make_physics_informer(device)
                if args.pde_weight and args.pde_backend == "physicsinformer" else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pde_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    config = {
        "mode": "train", "objective": "velocity_data+pde", "data": str(root),
        "epochs": args.epochs, "batch_size": args.batch_size, "initial_lr": args.lr,
        "pde_weight": args.pde_weight, "pde_points_per_batch": args.pde_points,
        "pde_backend": args.pde_backend,
        "seed": args.seed, "device": str(device), "model_kwargs": model_kwargs,
        "input_space": "x/L,y/L,nominal_Re; normalization inside the model",
        "output_names": ["u_over_Ulid", "v_over_Ulid", "p_over_rho_Ulid_squared"],
        "train_Re": nominal_re[:, 0].tolist(), "pde_coefficient_Re": actual_re[:, 0].tolist(),
        "val_Re": sorted(set(x_val[:, 2].tolist())),
        "torch_version": str(torch.__version__),
        "physicsnemo_version": package_version("nvidia-physicsnemo"),
        "physicsnemo_sym_version": package_version("nvidia-physicsnemo.sym"),
    }
    out = create_output(args.out)
    write_json(out / "config.json", config)
    print(f"Device={device}; backend={args.pde_backend}; loss=data+{args.pde_weight:g}*PDE; "
          f"PDE points/batch={args.pde_points}", flush=True)
    print("PDE residuals: continuity, momentum_x, momentum_y. No BC or IC loss.", flush=True)
    print("CFD labels and PDE points use training Re only; test fields are not loaded.", flush=True)
    history, best_metric, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    loss_columns = ["batch_data_mse", "batch_pde_mse", "batch_total_loss"] + [
        f"batch_{name}_mse" for name in PDE_NAMES]
    columns = ["epoch", "lr", "train_mse", "train_relative_L2", "train_E_RMS_Ulid",
               "val_mse", "val_relative_L2", "val_E_RMS_Ulid"] + loss_columns + [
               "epoch_seconds", "elapsed_seconds"]

    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        def record(epoch, used_lr, epoch_seconds, batch_losses=None):
            nonlocal best_metric, best_epoch
            training, _ = evaluate(velocity_model, x_train, y_train, device)
            validation, _ = evaluate(velocity_model, x_val, y_val, device)
            row = {"epoch": epoch, "lr": used_lr,
                   "train_mse": training["mse"], "train_relative_L2": training["relative_L2"],
                   "train_E_RMS_Ulid": training["E_RMS_Ulid"],
                   "val_mse": validation["mse"], "val_relative_L2": validation["relative_L2"],
                   "val_E_RMS_Ulid": validation["E_RMS_Ulid"],
                   **({name: "" for name in loss_columns} if batch_losses is None else batch_losses),
                   "epoch_seconds": epoch_seconds, "elapsed_seconds": time.perf_counter() - started}
            if validation["mse"] < best_metric:
                best_metric, best_epoch = validation["mse"], epoch
                save_checkpoint(out / "best.pt", model, model_kwargs, normalization, config, epoch, best_metric)
            history.append(row)
            writer.writerow(row)
            stream.flush()
            if epoch <= 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                message = f"Epoch {epoch}/{args.epochs}: val L2={100*validation['relative_L2']:.4f}%"
                if batch_losses is not None:
                    message += f", data={row['batch_data_mse']:.3e}, PDE={row['batch_pde_mse']:.3e}"
                print(message + f", best epoch={best_epoch}, elapsed={row['elapsed_seconds']:.1f}s", flush=True)
            return validation["mse"]

        record(0, args.lr, 0.0)
        interrupted = False
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()
                model.train()
                used_lr = optimizer.param_groups[0]["lr"]
                totals = {name: 0.0 for name in loss_columns}
                count = 0
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss_data = torch.nn.functional.mse_loss(model(inputs)[:, :2], targets)
                    terms = {name: loss_data.new_zeros(()) for name in PDE_NAMES}
                    if args.pde_weight:
                        xy, re, effective_re = sample_interior(args.pde_points, nominal_re, actual_re, pde_generator)
                        residuals = pde_residuals(model, informer, xy, re, effective_re)
                        terms = {name: residuals[name].square().mean() for name in PDE_NAMES}
                    loss_pde = sum(terms.values())
                    loss = loss_data + args.pde_weight * loss_pde
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss; inspect the learning rate and PDE weight.")
                    loss.backward()
                    optimizer.step()
                    values = [loss_data, loss_pde, loss] + [terms[name] for name in PDE_NAMES]
                    for name, value in zip(loss_columns, values):
                        totals[name] += float(value.detach())
                    count += 1
                metric = record(epoch, used_lr, time.perf_counter() - epoch_start,
                                {name: value / count for name, value in totals.items()})
                scheduler.step(metric)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; keeping best.pt from completed evaluations.", flush=True)
        completed_epoch = history[-1]["epoch"]
        if not interrupted:
            save_checkpoint(out / "last.pt", model, model_kwargs, normalization, config,
                            completed_epoch, history[-1]["val_mse"])

    plot_history(history, out / "learning_curves.png", "train")
    best_model, checkpoint = load_pinn_checkpoint(out / "best.pt", device)
    predictions = predict_array(VelocityView(best_model), x_val, device)
    errors = []
    for re in np.unique(x_val[:, 2]):
        mask = x_val[:, 2] == re
        errors.append({"Re": float(re), **velocity_errors(predictions[mask], y_val[mask])})
    write_csv(out / "best_val_errors.csv", errors)
    write_json(out / "summary.json", {
        "best_epoch": checkpoint["epoch"], "selection_metric": "val_mse",
        "selection_value": checkpoint["selection_value"], "completed_epochs": completed_epoch,
        "interrupted": interrupted, "total_wall_seconds": time.perf_counter() - started,
    })
    print(f"Best model: {out / 'best.pt'}", flush=True)


@torch.inference_mode()
def predict(args):
    if args.n < 2 or args.threads < 1 or not math.isfinite(args.re):
        raise ValueError("Use N >= 2, positive threads and finite Re.")
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    model, checkpoint = load_pinn_checkpoint(args.checkpoint, device)
    norm = checkpoint["normalization"]
    if not float(norm["re_min"]) <= args.re <= float(norm["re_max"]):
        raise ValueError("Use Re within the training range Re50-200.")
    c = (np.arange(args.n) + 0.5) / args.n
    xx, yy = np.meshgrid(c, c, indexing="xy")
    inputs = np.column_stack((xx.ravel(), yy.ravel(), np.full(args.n * args.n, args.re)))
    fields = []
    for start in range(0, len(inputs), 4096):
        batch = torch.as_tensor(inputs[start:start + 4096], dtype=torch.float32, device=device)
        fields.append(model(batch).cpu().numpy())
    fields = np.concatenate(fields)
    reference_input = torch.tensor([[0.5, 0.5, args.re]], dtype=torch.float32, device=device)
    # Choose a gauge for exported pressure only; this is not a training loss.
    fields[:, 2] -= float(model(reference_input)[0, 2])
    if not np.isfinite(fields).all():
        raise FloatingPointError("Non-finite predictions.")
    reference, metrics = None, None
    if args.data is not None:
        matches = [row for row in read_index(args.data) if float(row["Re"]) == args.re]
        if matches:
            case = read_case(args.data, matches[0])
            if case["x"].shape != inputs.shape or not np.allclose(case["x"], inputs, rtol=0, atol=1e-12):
                raise ValueError("Use --n matching the CFD reference grid.")
            reference = case["y"]
            metrics = {"Re": args.re, "split": matches[0]["split"],
                       **velocity_errors(fields[:, :2], reference)}
    out = create_output(args.out)
    np.savez_compressed(out / "prediction.npz", inputs=inputs, prediction=fields[:, :2],
                        pressure=fields[:, 2:3], pressure_reference=np.array([0.5, 0.5]),
                        pressure_name="p_over_rho_Ulid_squared", Re=args.re, N=args.n,
                        input_names=np.array(["x_over_L", "y_over_L", "Re"]),
                        target_names=np.array(["u_over_Ulid", "v_over_Ulid"]))
    plot_field(inputs, fields[:, :2], reference, out / "prediction.png", f"Data + PDE prediction Re={args.re:g}")
    if metrics is not None:
        write_csv(out / "prediction_errors.csv", [metrics])
        print(f"CFD comparison: relative L2={100*metrics['relative_L2']:.4f}%, "
              f"max/U_lid={metrics['E_max_Ulid']:.6e}")
    print(f"Prediction outputs: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("train", help="Train with velocity data + steady PDE residuals")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--pde-points", type=int, default=256)
    p.add_argument("--pde-weight", type=float, default=0.01)
    p.add_argument("--pde-backend", choices=["autograd", "physicsinformer"], default="autograd")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p = sub.add_parser("predict", help="Evaluate a data + PDE checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    for command in sub.choices.values():
        command.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
