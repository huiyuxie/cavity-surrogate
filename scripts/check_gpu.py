import torch
from importlib.metadata import version
from physicsnemo.models.mlp.fully_connected import FullyConnected


assert torch.cuda.is_available(), "PyTorch could not detect an available CUDA GPU"

torch.manual_seed(42)
device = torch.device("cuda:0")

model = FullyConnected(
    in_features=3,
    out_features=2,
    num_layers=4,
    layer_size=64,
    activation_fn="tanh",
).to(device)

x = torch.rand(1024, 3, device=device)
target = torch.stack(
    (x[:, 0] + x[:, 2], x[:, 1] - x[:, 2]),
    dim=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with torch.no_grad():
    initial_loss = ((model(x) - target) ** 2).mean().item()

for _ in range(100):
    optimizer.zero_grad(set_to_none=True)

    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()

    assert torch.isfinite(loss).item(), "Loss contains NaN or Inf"

    loss.backward()

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all().item(), (
                "Gradients contain NaN or Inf"
            )

    optimizer.step()

with torch.no_grad():
    final_loss = ((model(x) - target) ** 2).mean().item()

torch.cuda.synchronize()

print("PyTorch:", torch.__version__)
print("PhysicsNeMo:", version("nvidia-physicsnemo"))
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)
print("Output shape:", tuple(prediction.shape))
print("Initial loss:", initial_loss)
print("Final loss:", final_loss)

assert final_loss < initial_loss, "Loss did not decrease after training"

print("GPU training check passed")