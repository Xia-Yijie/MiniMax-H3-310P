"""Memory-mapped GGUF tensor store with bounded row decoding."""
from __future__ import annotations
import math
import numpy as np

from .gguf import GGUFFile
from .quantization import QUANT_LAYOUTS, decode


class TensorStore:
    def __init__(self, path):
        self.gguf = GGUFFile(path)
        self._maps = {}
        self._device_cache = {}
        self._device_cache_bytes = 0
        self._device_cache_budget = 0
        self._prepared_cache = {}
        self._prepared_cache_bytes = 0
        self._prepared_cache_budget = 0

    def enable_prepared_cache(self, max_bytes=0):
        """Bounded, immutable scaled FP16 weights; zero preserves the old path.

        Budget is total across devices. Only use with immutable inference weights.
        LoRA stays separate and is not merged or requantized.
        """
        if max_bytes < 0:
            raise ValueError('Prepared cache budget must be nonnegative')
        if self._prepared_cache:
            raise RuntimeError('Configure prepared cache before first forward')
        self._prepared_cache_budget = int(max_bytes)

    def prepared_weight(self, name, device, row_chunk=2048):
        import torch
        info = self.info(name)
        if self._prepared_cache_budget == 0 or info.quant_type not in (8, 12, 14):
            return None
        key = (name, str(device))
        if key in self._prepared_cache:
            return self._prepared_cache[key]
        if len(info.shape) != 2:
            return None
        rows, columns = info.shape
        needed = rows * columns * 2 + rows * 4
        if self._prepared_cache_bytes + needed > self._prepared_cache_budget:
            return None
        with torch.no_grad():
            weights = torch.empty((rows, columns), device=device, dtype=torch.float16)
            scales = torch.empty(rows, device=device, dtype=torch.float32)
            for start in range(0, rows, row_chunk):
                stop = min(start + row_chunk, rows)
                weight = self.device_rows(name, start, stop, device)
                if weight is None:
                    weight = torch.from_numpy(self.rows(name, start, stop).copy()).to(device)
                scale = torch.pow(2.0, torch.ceil(torch.log2(weight.abs().amax(-1).clamp(min=1))))
                weights[start:stop] = (weight / scale[:, None]).half()
                scales[start:stop] = scale
            result = (weights, scales)
            self._prepared_cache[key] = result
            self._prepared_cache_bytes += needed
            # The immutable prepared matrix supersedes its packed device copy.
            # Keeping both wastes memory needed for longer video activations.
            packed = self._device_cache.pop(key, None)
            if packed is not None:
                self._device_cache_bytes -= packed.numel() * packed.element_size()
            return result


    def enable_quant_device_cache(self, max_bytes=24 * 1024**3):
        """Keep Q4_K/Q6_K/Q8_0 bytes resident and decode bounded row chunks."""
        self._device_cache_budget = max_bytes

    def enable_q8_device_cache(self, max_bytes=24 * 1024**3):
        # Compatibility with earlier runners; now accepts all supported rungs.
        self.enable_quant_device_cache(max_bytes)

    def device_rows(self, name, start, stop, device):
        import torch
        from .device_quantization import decode_blocks
        info = self.info(name)
        if info.quant_type not in (8, 12, 14) or self._device_cache_budget <= 0 or device.type != 'npu':
            return None
        if len(info.shape) != 2 or not 0 <= start <= stop <= info.shape[0]:
            raise ValueError('Device row cache requires valid matrix rows')
        elements, nbytes = QUANT_LAYOUTS[info.quant_type]
        if info.shape[1] % elements:
            raise ValueError('Matrix rows must align to GGUF quantization blocks')
        key = (name, str(device))
        if key not in self._device_cache:
            raw = self.gguf.raw(name, elements, nbytes)
            if self._device_cache_bytes + raw.size > self._device_cache_budget:
                return None
            self._device_cache[key] = torch.from_numpy(raw.copy()).to(device)
            self._device_cache_bytes += raw.size
        row_bytes = info.shape[1] // elements * nbytes
        packed = self._device_cache[key][start * row_bytes:stop * row_bytes].reshape(-1, nbytes)
        return decode_blocks(packed, info.quant_type).reshape(stop - start, info.shape[1])

    def device_q8_rows(self, name, start, stop, device):
        return self.device_rows(name, start, stop, device)

    def info(self, name):
        return self.gguf.tensors[name]

    def rows(self, name, start=0, stop=None):
        """Decode only consecutive rows of a matrix (or elements of a vector)."""
        info = self.info(name)
        row_width = math.prod(info.shape[1:]) if len(info.shape) > 1 else 1
        stop = info.shape[0] if stop is None else stop
        if not 0 <= start <= stop <= info.shape[0]:
            raise ValueError('Invalid tensor row range')
        if info.quant_type not in QUANT_LAYOUTS:
            raise NotImplementedError(f'Unsupported quantization: {info.quant_type}')
        block_elements, block_bytes = QUANT_LAYOUTS[info.quant_type]
        if row_width % block_elements:
            raise ValueError(f'Rows of {name} do not align to quantization blocks')
        if name not in self._maps:
            self._maps[name] = self.gguf.raw(name, block_elements, block_bytes)
        bytes_per_row = row_width // block_elements * block_bytes
        raw = self._maps[name][start * bytes_per_row:stop * bytes_per_row]
        return decode(raw, info.quant_type).reshape((stop - start,) + info.shape[1:])

    def tensor(self, name, *, max_bytes=128 * 1024**2):
        info = self.info(name)
        if info.elements * 4 > max_bytes:
            raise MemoryError(f'{name}: use rows() instead of decoding the entire tensor')
        return self.rows(name)
