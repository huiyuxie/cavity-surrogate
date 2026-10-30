# Velocity data + steady Navier–Stokes loss

`scripts/train_cavity_pinn.py` trains the existing type of tanh MLP with
`loss = loss_data + pde_weight * loss_pde`. It predicts dimensionless `(u, v, p)`
from `(x/L, y/L, Re)`. Data MSE uses only the two velocity outputs.

The PDE loss includes continuity and the two steady momentum residuals:

$$
r_c=u_x+v_y,\qquad
r_x=uu_x+vu_y+p_x-Re^{-1}(u_{xx}+u_{yy}),\qquad
r_y=uv_x+vv_y+p_y-Re^{-1}(v_{xx}+v_{yy}).
$$

`loss_pde = mean(r_c**2) + mean(r_x**2) + mean(r_y**2)`.
There is no boundary-condition, initial-condition, pressure-label, or pressure-gauge
loss. Pressure is learned through its spatial gradients. Exported pressure is
shifted to zero at the cavity center for each requested Re; this changes only
the prediction output, not training.

Normalization stays inside the PyTorch graph, so spatial derivatives are with
respect to `x/L,y/L` on the original unit square. Re is a parameter, not a third
spatial coordinate. The viscous coefficient uses `Re_actual` from the dataset
index where available, preserving the exporter's viscosity rounding.

## Environment

The default `--pde-backend autograd` uses the existing README environment.
No additional packages or CFD simulations are needed. It evaluates the PDE
directly with `torch.autograd.grad`, including second spatial derivatives.

For the optional `--pde-backend physicsinformer`, install the separate Sym
dependency in your CUDA environment, following its
[installation instructions](https://github.com/NVIDIA/physicsnemo-sym/tree/v2.3.0#installation):

```bash
python3 -m pip install Cython
python3 -m pip install --no-build-isolation -r docs/requirements-pinn.txt
```

Sym 2.3.0 is pinned to keep PhysicsNeMo 1.3.0. Its source installation requires
a compatible CUDA toolkit/compiler. Its NVTX profiling also expects a
CUDA-enabled PyTorch build, including when running with `--device cpu`.
The default backend avoids these optional installation requirements. Both
backends implement the same equations and differentiable normalization.

## Train

Run from the repository root. The committed normalization metadata can be reused:

```bash
python3 scripts/train_cavity_pinn.py train \
  --data data/cavity_Re50_200 \
  --normalization results/ml_preparation/normalization.json \
  --out results/train_Re50_200_pinn \
  --epochs 5000 --batch-size 1024 \
  --pde-points 256 --pde-weight 0.01 \
  --device cuda
```

Use `--device cpu` for CPU execution. `--pde-points` is the number of interior
collocation points resampled per optimizer step. Both supervised labels and
PDE points use the 16 training Re values only; validation/test Re receive no
PDE training points. Validation velocity MSE selects `best.pt`. Test velocity
fields are not loaded during training. No new CFD simulations are needed.

The PDE weight is a starting experiment setting, not an accuracy guarantee.
Set `--pde-weight 0` for a control with the same three-output architecture.
Use a new or empty output directory for every run.

Outputs include `best.pt`, `last.pt` after normal completion, `config.json`,
`summary.json`, `history.csv`, `learning_curves.png`, and `best_val_errors.csv`.
The `batch_*` history columns average losses encountered during each epoch;
train/validation metrics evaluate the model at the end of the epoch. Checkpoints
contain model weights and metadata, not optimizer state for exact resume.

## Evaluate

Use this script's `predict` command for its three-output checkpoints:

```bash
python3 scripts/train_cavity_pinn.py predict \
  --checkpoint results/train_Re50_200_pinn/best.pt \
  --re 55 --n 64 --device cpu \
  --data data/cavity_Re50_200 \
  --out results/prediction_Re55_pinn
```

Repeat with Re95, Re135 and Re175 and distinct output directories. The velocity
plot and `prediction_errors.csv` use the baseline's metrics. `prediction.npz`
also includes dimensionless pressure; pressure accuracy is not validated here.
Compare held-out velocity errors with the original MLP and interpolation results.
Without an explicit BC loss, wall velocities are not directly enforced.
