"""Exact inference attention with bounded score memory; correctness backend.

No S-by-S matrix is materialized. This is not a fused Ascend attention kernel.
"""
import math
import torch


@torch.no_grad()
def streaming_attention(q, k, v, *, scale=None, causal=False,
                        query_chunk=128, key_chunk=256, head_chunk=32):
    """[B,H,S,D] tensors; FP32 tensors and online softmax.

    Matmul accuracy also depends on the hardware backend: FP32 tensor dtype
    does not guarantee full FP32 multiplier precision on Ascend 310P.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('Attention requires [B,H,S,D]')
    if (q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1] or min(k.shape[-2], q.shape[-2]) == 0):
        raise ValueError('Incompatible or empty attention tensors')
    if len({q.device, k.device, v.device}) != 1:
        raise ValueError('Attention tensors must share a device')
    if min(query_chunk, key_chunk, head_chunk) <= 0:
        raise ValueError('Attention chunk sizes must be positive')
    scale = q.shape[-1]**-0.5 if scale is None else scale
    result = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=torch.float32)
    for hs in range(0, q.shape[1], head_chunk):
        he = min(hs + head_chunk, q.shape[1])
        for qs in range(0, q.shape[-2], query_chunk):
            qe = min(qs + query_chunk, q.shape[-2])
            queries = q[:, hs:he, qs:qe].float() * scale
            shape = (*queries.shape[:-1], 1)
            maximum = torch.full(shape, -math.inf, dtype=torch.float32, device=q.device)
            denominator = torch.zeros(shape, dtype=torch.float32, device=q.device)
            numerator = torch.zeros((*queries.shape[:-1], v.shape[-1]),
                                    dtype=torch.float32, device=q.device)
            for ks in range(0, k.shape[-2], key_chunk):
                if causal and ks >= qe:
                    break
                ke = min(ks + key_chunk, k.shape[-2])
                logits = torch.matmul(queries, k[:, hs:he, ks:ke].float().transpose(-2, -1))
                if causal:
                    mask = (torch.arange(ks, ke, device=q.device)[None, :]
                            > torch.arange(qs, qe, device=q.device)[:, None])
                    logits = logits.masked_fill(mask, -math.inf)
                next_max = torch.maximum(maximum, logits.amax(dim=-1, keepdim=True))
                safe_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
                correction = torch.exp(maximum - safe_max)
                probabilities = torch.exp(logits - safe_max)
                numerator = numerator * correction + torch.matmul(probabilities, v[:, hs:he, ks:ke].float())
                denominator = denominator * correction + probabilities.sum(dim=-1, keepdim=True)
                maximum = next_max
            result[:, hs:he, qs:qe] = numerator / denominator
    return result
