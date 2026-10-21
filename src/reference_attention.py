import math
import torch


def scaled_dot_product_attention(q, k, v, causal=False):
    """
    q: [B, H, N, D]
    k: [B, H, N, D]
    v: [B, H, N, D]

    Returns: [B, H, N, D]
    """
    d_k = q.shape[-1]

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if causal:
        n = q.shape[-2]
        mask = torch.triu(
            torch.ones(n, n, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v)

    return output