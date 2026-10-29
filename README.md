# Cavity Surrogate

A reproducible CFD-to-machine-learning project for **2D steady lid-driven cavity flow**. OpenFOAM generates reference fields, and a PhysicsNeMo/PyTorch MLP learns

$$
(x/L,\; y/L,\; Re) \longrightarrow (u/U_{\mathrm{lid}},\; v/U_{\mathrm{lid}}).
$$

On four held-out Reynolds numbers, the MLP achieves **0.41%–0.52% relative L2 error**. Piecewise linear interpolation is more accurate on this densely sampled problem, at **0.019%–0.043%**.

## Data and model

The flow is incompressible and laminar in a square cavity with `L = 0.1 m` and `U_lid = 1 m/s`. The top wall moves; the side and bottom walls are stationary and no-slip. OpenFOAM Foundation 13 uses `foamRun` with `incompressibleFluid`, empty front/back patches, and `deltaT = 0.0005 s`.

The dataset contains **24 CFD cases** on a **64 × 64 × 1 grid**, using snapshots at `t = 10 s` after temporal screening. Reynolds number is varied through viscosity, with splits assigned by complete case:

| Split | Cases | Reynolds numbers |
|---|---:|---|
| Train | 16 | 50, 60, 70, …, 200 |
| Validation | 4 | 65, 105, 145, 185 |
| Test | 4 | 55, 95, 135, 175 |

Each `.npz` stores `inputs (4096, 3)`, `targets (4096, 2)`, and case metadata. Coordinates and velocities are dimensionless; training maps coordinates and Re to `[-1, 1]`. See the [dataset index](data/cavity_Re50_200/dataset_index.csv).

The MLP uses **4 hidden layers × 64 neurons**, `tanh`, and supervised velocity MSE. Training uses Adam (initial learning rate 0.001, batch size 1,024) with `ReduceLROnPlateau`. One shared checkpoint is selected by aggregate validation MSE; test fields are excluded from training.

## Setup

The recorded stack is Python 3.12.3, PyTorch 2.7.1 with CUDA 12.6, and PhysicsNeMo 1.3.0. See the [environment snapshot](docs/environment.md) and [dependency lock](docs/requirements-lock.txt).

```bash
git clone https://github.com/huiyuxie/cavity-surrogate.git
cd cavity-surrogate
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r docs/requirements-lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

The lock records the CUDA environment; scripts also support CPU execution. OpenFOAM is only needed to regenerate CFD data or repeat CFD checks. Exported data and trained checkpoints are included.

## Quick start

Predict Re55 with the saved model, running inference on CPU:

```bash
python3 scripts/train_cavity.py predict \
  --checkpoint results/train_Re50_200_gpu/best.pt \
  --re 55 --n 64 --device cpu \
  --data data/cavity_Re50_200 \
  --out results/reproduction/prediction_Re55
```

This writes `prediction.npz`, `prediction.png`, and `prediction_errors.csv`. Evaluate the other test cases with `--re 95`, `--re 135`, or `--re 175`, changing `--out` each time. **Prediction and training require a new or empty output directory.**

![CFD reference, MLP prediction, and absolute errors at Re55](results/prediction_Re55/prediction.png)

## Reproduce baseline and training

Run from the repository root:

```bash
# Prepare normalization and the linear interpolation baseline.
python3 scripts/prepare_ml_data.py \
  --data data/cavity_Re50_200 \
  --out results/reproduction/ml_preparation

# Train the MLP using the recorded experiment settings.
python3 scripts/train_cavity.py train \
  --data data/cavity_Re50_200 \
  --normalization results/reproduction/ml_preparation/normalization.json \
  --out results/reproduction/train_mlp \
  --epochs 5000 --batch-size 1024 --lr 0.001 \
  --layers 4 --width 64 --seed 42 --device cuda
```

The baseline interpolates between the two neighboring training Re fields at matching coordinates; test data is used only to measure errors. Preparation saves split/normalization metadata, baseline errors, and plots.

For CPU training, use `--device cpu`. Training saves `best.pt`, `last.pt`, learning curves, error metrics, and run metadata. To evaluate the new model, use the prediction command with `--checkpoint results/reproduction/train_mlp/best.pt`.

![Training and validation curves for the recorded GPU run](results/train_Re50_200_gpu/learning_curves.png)

## Test results

The recorded GPU run selected epoch **4,968 of 5,000**. Both metrics use the two-component velocity vector; all values below are percentages.

| Test Re | Interpolation relative L2 | MLP relative L2 | Interpolation max/U_lid | MLP max/U_lid |
|---|---:|---:|---:|---:|
| 55 | 0.0428% | 0.5202% | 0.0283% | 2.1242% |
| 95 | 0.0384% | 0.4214% | 0.0269% | 1.5980% |
| 135 | 0.0263% | 0.4056% | 0.0230% | 1.6841% |
| 175 | 0.0192% | 0.4073% | 0.0177% | 1.7520% |

Sources: [interpolation errors](results/ml_preparation/baseline_test_errors.csv) and MLP errors for [Re55](results/prediction_Re55/prediction_errors.csv), [Re95](results/prediction_Re95/prediction_errors.csv), [Re135](results/prediction_Re135/prediction_errors.csv), and [Re175](results/prediction_Re175/prediction_errors.csv).

Interpolation benefits from closely spaced training cases and nearly linear local variation with Re. The MLP must also fit spatial variation; the Re55 plot shows larger local errors near the upper corners, where the moving lid meets stationary walls.

## Numerical checks and scope

At Re100, relative L2 differences are **3.3233%** for N32→N64, **1.5366%** for N64→N128, and **0.0883%** for the N64 time-step comparison at 10 s. See [grid metrics](results/grid_study_Re100/grid_metrics.csv) and [time-step metrics](results/step_study_Re100/time_step_metrics.csv).

- These numerical-solution comparisons support a practical N64 dataset choice; they do not establish grid independence across all Re or complete benchmark validation.
- Model errors measure agreement with finite-resolution CFD data. Tests cover Re50–200 on N64 cell centers; other grids and coordinates are unvalidated, and the CLI rejects Re outside the training range.
- The model predicts steady velocities using supervised learning, without PDE-residual losses or hard boundary constraints.
- CFD, interpolation, and inference timings have not been compared.

## References

- [OpenFOAM Foundation documentation](https://doc.cfd.direct/openfoam/user-guide-v13)
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
- [PyTorch](https://pytorch.org/)
- [Botella and Peyret, *Benchmark spectral results on the lid-driven cavity flow* (1998)](https://www.sciencedirect.com/science/article/pii/S0045793098000024)
