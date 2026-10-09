"""CPU block decoders for the types present in our GGUF checkpoints.

Q4_K/Q6_K layout follows ggml-org/llama.cpp (MIT); see THIRD_PARTY.md.
Decoding is deliberately separate from the device execution backend.
"""
import numpy as np

QUANT_LAYOUTS = {0: (1, 4), 1: (1, 2), 8: (32, 34),
                 12: (256, 144), 14: (256, 210), 30: (1, 2)}


def decode(data: np.ndarray, kind: int) -> np.ndarray:
    """Return float32 values, never reinterpret BF16 as FP16."""
    if kind not in QUANT_LAYOUTS:
        raise NotImplementedError(f'Unsupported GGUF quantization type {kind}')
    data = np.ascontiguousarray(data, dtype=np.uint8)
    elements, size = QUANT_LAYOUTS[kind]
    if data.size % size:
        raise ValueError('Incomplete quantization block')
    if kind == 0:
        return data.view('<f4').reshape(-1).copy()
    if kind == 1:
        return data.view('<f2').reshape(-1).astype(np.float32)
    if kind == 30:
        return (data.view('<u2').astype(np.uint32) << 16).view(np.float32).reshape(-1)
    blocks = data.reshape(-1, size)
    if kind == 8:
        scale = blocks[:, :2].copy().view('<f2').astype(np.float32)
        return (scale * blocks[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
    if kind == 12:
        scale = blocks[:, :2].copy().view('<f2').astype(np.float32)
        minimum = blocks[:, 2:4].copy().view('<f2').astype(np.float32)
        packed = blocks[:, 4:16].reshape(-1, 3, 4)
        low_scale, low_min, high = packed[:, 0], packed[:, 1], packed[:, 2]
        scales = np.concatenate((low_scale & 63,
                                 (high & 15) | ((low_scale >> 2) & 48)), axis=1)
        mins = np.concatenate((low_min & 63,
                               (high >> 4) | ((low_min >> 2) & 48)), axis=1)
        quant = blocks[:, 16:].reshape(-1, 4, 32)
        quant = np.stack((quant & 15, quant >> 4), axis=2).reshape(-1, 8, 32)
        return ((scale * scales).reshape(-1, 8, 1) * quant.astype(np.float32)
                - (minimum * mins).reshape(-1, 8, 1)).reshape(-1)
    # Q6_K: each 128-value group combines 4-bit lows and 2-bit highs.
    low = blocks[:, :128].reshape(-1, 2, 64)
    low = np.stack((low & 15, low >> 4), axis=2).reshape(-1, 8, 32)
    high = blocks[:, 128:192].reshape(-1, 2, 32)
    high = np.stack(tuple((high >> shift) & 3 for shift in (0, 2, 4, 6)), axis=2)
    quant = ((low | (high.reshape(-1, 8, 32) << 4)).astype(np.int16) - 32)
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    scale = blocks[:, 208:210].copy().view('<f2').astype(np.float32)
    return ((scale * scales).reshape(-1, 16, 1)
            * quant.reshape(-1, 16, 16).astype(np.float32)).reshape(-1)
