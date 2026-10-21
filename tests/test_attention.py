import torch
from src.reference_attention import scaled_dot_product_attention

device = "cuda"
B, H, N, D = 2, 4, 128, 64

q = torch.randn(B, H, N, D, device=device)
k = torch.randn(B, H, N, D, device=device)
v = torch.randn(B, H, N, D, device=device)

out = scaled_dot_product_attention(q, k, v)

reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)

torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)

print("Attention output:", out.shape)
print("Maximum absolute error:", (out - reference).abs().max().item())
print("Reference correctness test passed!")