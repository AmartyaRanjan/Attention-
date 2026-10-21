import torch

assert torch.cuda.is_available(), "CUDA GPU unavailable"

device = "cuda"
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)

c = a @ b
torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("Output shape:", c.shape)
print("All finite:", torch.isfinite(c).all().item())
print("GPU matrix multiplication passed!")