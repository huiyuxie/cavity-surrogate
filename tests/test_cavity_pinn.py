"""Numerical checks for the data + PDE trainer (CPU, no CFD solver needed)."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_cavity_pinn as pinn
from prepare_ml_data import EXPECTED_SPLITS


NORMALIZATION = {
    "input_names": ["x_over_L", "y_over_L", "Re"],
    "target_names": ["u_over_Ulid", "v_over_Ulid"],
    "re_min": 50.0, "re_max": 200.0,
}
MODEL_KWARGS = dict(in_features=3, out_features=3, num_layers=2, layer_size=8,
                    activation_fn="tanh", weight_norm=False,
                    skip_connections=False, adaptive_activations=False)


@pytest.fixture(autouse=True)
def cpu_only_profiler(monkeypatch):
    # Sym 2.3 profiles its graph with NVTX even on CPU. Disable only profiling
    # for CPU-only test wheels; its equations and gradient code are unchanged.
    if torch.version.cuda is None:
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


def backend_informer(backend):
    if backend == "autograd":
        return None
    pytest.importorskip("physicsnemo.sym.eq.phy_informer")
    return pinn.make_physics_informer("cpu")


class AnalyticNetwork(torch.nn.Module):
    """An independently specified field behind the trainer's normalization."""

    def __init__(self, poiseuille=False):
        super().__init__()
        self.poiseuille = poiseuille

    def forward(self, normalized):
        x, y = (normalized[:, :2] + 1.0).split(1, dim=1)
        x, y = x / 2, y / 2
        re = 50 + (normalized[:, 2:3] + 1) * 75
        if self.poiseuille:
            # Keep exact zero components connected for second-order autodiff.
            zero = 0 * (x**3 + y**3)
            return torch.cat((1 - y**2 + zero, zero, -2 * x / re), dim=1)
        return torch.cat((re / 100 * x**2 + y**3, x**3 - 2*x*y, x*y), dim=1)


@pytest.mark.parametrize("backend", ["autograd", "physicsinformer"])
def test_polynomial_residuals_use_physical_coordinates_and_per_point_re(monkeypatch, backend):
    monkeypatch.setattr(pinn, "build_model", lambda _: AnalyticNetwork())
    model = pinn.CavityPINN(MODEL_KWARGS, NORMALIZATION)
    xy = torch.tensor([[0.17, 0.23], [0.51, 0.67], [0.83, 0.41]], dtype=torch.float64)
    re = torch.tensor([[50.0], [100.0], [200.0]], dtype=torch.float64)
    coefficient_re = re * 1.01
    residuals = pinn.pde_residuals(model, backend_informer(backend), xy, re, coefficient_re)
    x, y = xy[:, 0:1], xy[:, 1:2]
    a = re / 100
    u, v = a*x**2 + y**3, x**3 - 2*x*y
    expected = {
        "continuity": 2*(a - 1)*x,
        "momentum_x": u*2*a*x + v*3*y**2 + y - (2*a + 6*y)/coefficient_re,
        "momentum_y": u*(3*x**2 - 2*y) - 2*x*v + x - 6*x/coefficient_re,
    }
    for name in pinn.PDE_NAMES:
        torch.testing.assert_close(residuals[name], expected[name], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("backend", ["autograd", "physicsinformer"])
def test_exact_poiseuille_solution_has_zero_residual_at_multiple_re(monkeypatch, backend):
    monkeypatch.setattr(pinn, "build_model", lambda _: AnalyticNetwork(poiseuille=True))
    model = pinn.CavityPINN(MODEL_KWARGS, NORMALIZATION)
    xy = torch.tensor([[0.11, 0.21], [0.41, 0.57], [0.79, 0.81]], dtype=torch.float64)
    re = torch.tensor([[50.0], [100.0], [200.0]], dtype=torch.float64)
    residuals = pinn.pde_residuals(model, backend_informer(backend), xy, re)
    for residual in residuals.values():
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-10)


@pytest.mark.parametrize("backend", ["autograd", "physicsinformer"])
def test_pde_loss_backpropagates_to_velocity_and_unlabelled_pressure(backend):
    torch.manual_seed(3)
    model = pinn.CavityPINN(MODEL_KWARGS, NORMALIZATION)
    xy = torch.rand(12, 2)
    re = torch.full((12, 1), 100.0)
    residuals = pinn.pde_residuals(model, backend_informer(backend), xy, re)
    sum(value.square().mean() for value in residuals.values()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    output_weights = [p for p in model.parameters() if p.ndim == 2 and p.shape[0] == 3]
    assert len(output_weights) == 1
    assert output_weights[0].grad[:2].abs().sum() > 0
    assert output_weights[0].grad[2].abs().sum() > 0


def test_collocation_sampling_uses_training_re_and_recorded_viscosity():
    rows = [{"split": "train", "Re": "50", "Re_actual": "50.001"},
            {"split": "train", "Re": "100", "Re_actual": "99.999"},
            {"split": "val", "Re": "65"}, {"split": "test", "Re": "55"}]
    nominal, actual = pinn.training_reynolds(rows, "cpu")
    xy, re, coefficient_re = pinn.sample_interior(256, nominal, actual, torch.Generator().manual_seed(5))
    assert ((xy > 0) & (xy < 1)).all()
    assert set(re[:, 0].tolist()) == {50, 100}
    for index in range(len(nominal)):
        torch.testing.assert_close(coefficient_re[re == nominal[index]],
                                   actual[index].expand_as(coefficient_re[re == nominal[index]]))


def make_small_dataset(root):
    root.mkdir()
    n = 4
    c = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(c, c, indexing="xy")
    rows = []
    for split, reynolds in EXPECTED_SPLITS.items():
        for re in reynolds:
            name = f"Re{re}.npz"
            # A test-file read would fail: training must only load train/val fields.
            if split == "test":
                (root / name).write_bytes(b"not loaded by training")
            else:
                inputs = np.column_stack((x.ravel(), y.ravel(), np.full(n*n, re)))
                u = (re/200) * x.ravel() * (1-x.ravel()) * y.ravel() * (1-y.ravel())
                np.savez(root / name, inputs=inputs, targets=np.column_stack((u, 0*u)),
                         Re=re, N=n, split=split,
                         input_names=np.array(NORMALIZATION["input_names"]),
                         target_names=np.array(NORMALIZATION["target_names"]))
            rows.append(dict(case=f"Re{re}", Re=re, split=split, file=name,
                             points=n*n, Re_actual=re))
    with (root / "dataset_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("backend", ["autograd", "physicsinformer"])
def test_short_training_checkpoint_and_prediction_round_trip(tmp_path, backend):
    backend_informer(backend)
    data, output = tmp_path / "data", tmp_path / "training"
    make_small_dataset(data)
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps(NORMALIZATION))
    args = argparse.Namespace(data=data, normalization=normalization, out=output,
                              epochs=1, batch_size=64, pde_points=8, pde_weight=0.01,
                              pde_backend=backend,
                              lr=0.001, layers=2, width=8, seed=7, log_every=1,
                              threads=1, device="cpu")
    pinn.train(args)
    model, checkpoint = pinn.load_pinn_checkpoint(output / "best.pt", "cpu")
    assert checkpoint["selection_metric"] == "val_mse"
    assert model(torch.tensor([[0.5, 0.5, 100.0]])).shape == (1, 3)
    with (output / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 2
    assert float(history[-1]["batch_pde_mse"]) > 0
    assert float(history[-1]["batch_total_loss"]) == pytest.approx(
        float(history[-1]["batch_data_mse"]) + 0.01*float(history[-1]["batch_pde_mse"]), rel=1e-6)
    prediction_out = tmp_path / "prediction"
    pinn.predict(argparse.Namespace(checkpoint=output / "best.pt", re=65, n=4,
                                    data=data, out=prediction_out, threads=1, device="cpu"))
    with np.load(prediction_out / "prediction.npz", allow_pickle=False) as result:
        assert result["prediction"].shape == (16, 2)
        assert result["pressure"].shape == (16, 1)
        assert np.isfinite(result["prediction"]).all()
    assert (prediction_out / "prediction_errors.csv").is_file()
    assert (prediction_out / "prediction.png").is_file()
