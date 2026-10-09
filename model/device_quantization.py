"""Decode cached GGUF blocks with torch operations on CPU/NPU.

Layouts are checked against independent ggml fixtures. Compressed bytes stay
resident; only a bounded GEMM row chunk is expanded into FP32.
"""
import torch


def decode_blocks(packed, kind):
    sizes = {8: 34, 12: 144, 14: 210}
    if kind not in sizes or packed.ndim != 2 or packed.shape[1] != sizes[kind]:
        raise ValueError('Unsupported or malformed quantized blocks')
    if packed.dtype != torch.uint8:
        raise ValueError('Quantized bytes must be uint8')
    count = packed.shape[0]
    if kind == 8:
        scale = packed[:, :2].contiguous().view(torch.float16).float()
        values = packed[:, 2:].contiguous().view(torch.int8).float()
        return (scale * values).reshape(-1)
    if kind == 12:
        scale = packed[:, :2].contiguous().view(torch.float16).float()
        minimum = packed[:, 2:4].contiguous().view(torch.float16).float()
        fields = packed[:, 4:16].to(torch.int32).reshape(count, 3, 4)
        low_scale, low_min, high = fields[:, 0], fields[:, 1], fields[:, 2]
        scales = torch.cat((low_scale & 63,
                            (high & 15) | ((low_scale >> 2) & 48)), dim=1).float()
        mins = torch.cat((low_min & 63,
                          (high >> 4) | ((low_min >> 2) & 48)), dim=1).float()
        quant = packed[:, 16:].to(torch.int32).reshape(count, 4, 32)
        quant = torch.stack((quant & 15, quant >> 4), dim=2).reshape(count, 8, 32).float()
        return ((scale * scales).reshape(count, 8, 1) * quant
                - (minimum * mins).reshape(count, 8, 1)).reshape(-1)
    low = packed[:, :128].to(torch.int32).reshape(count, 2, 64)
    low = torch.stack((low & 15, low >> 4), dim=2).reshape(count, 8, 32)
    high = packed[:, 128:192].to(torch.int32).reshape(count, 2, 32)
    high = torch.stack(tuple((high >> shift) & 3 for shift in (0, 2, 4, 6)), dim=2)
    quant = (low | (high.reshape(count, 8, 32) << 4)).float() - 32
    scales = packed[:, 192:208].contiguous().view(torch.int8).float()
    scale = packed[:, 208:210].contiguous().view(torch.float16).float()
    return ((scale * scales).reshape(count, 16, 1)
            * quant.reshape(count, 16, 16)).reshape(-1)
