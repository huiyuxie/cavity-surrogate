import torch
from importlib.metadata import version
from physicsnemo.models.mlp.fully_connected import FullyConnected


# Check CUDA availability.
assert torch.cuda.is_available(), "PyTorch could not detect an available CUDA GPU"

torch.manual_seed(42)
device = torch.device("cuda:0")

# Create a fully connected network and move it to the GPU.
model = FullyConnected(
    in_features=3,
    out_features=2,
    num_layers=4,
    layer_size=64,
    activation_fn="tanh",
).to(device)

# Generate synthetic data to test the training environment.
# Input:  (x0, x1, x2)
# Target: (x0 + x2, x1 - x2)
x = torch.rand(1024, 3, device=device)
target = torch.stack(
    (x[:, 0] + x[:, 2], x[:, 1] - x[:, 2]),
    dim=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Measure the loss before training without computing gradients.
with torch.no_grad():
    initial_loss = ((model(x) - target) ** 2).mean().item()

# Run 100 training steps.
for _ in range(100):
    optimizer.zero_grad(set_to_none=True)

    # Forward pass and mean squared error.
    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()

    assert torch.isfinite(loss).item(), "Loss contains NaN or Inf"

    # Backward pass.
    loss.backward()

    # Check gradient values.
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all().item(), (
                "Gradients contain NaN or Inf"
            )

    # Update model parameters.
    optimizer.step()

# Measure the loss after training.
with torch.no_grad():
    final_loss = ((model(x) - target) ** 2).mean().item()

# Wait for all queued CUDA operations to finish.
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