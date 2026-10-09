"""Read local safetensors incrementally; no entire state-dict allocation."""
import json
import math
from pathlib import Path
import struct
import numpy as np
import torch


class SafeTensorStore:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 100 * 1024**2:
                raise ValueError('Invalid safetensors header size')
            self.header = json.loads(stream.read(length))
        self.data_offset = length + 8
        self._maps = {}
        self._device_cache = {}
        self._device_cache_bytes = 0
        self._device_cache_budget = 0
        self._prepared_cache = {}
        self._prepared_cache_bytes = 0
        self._prepared_cache_budget = 0

    def enable_prepared_cache(self, max_bytes=0):
        if max_bytes < 0 or self._prepared_cache:
            raise ValueError('Set a nonnegative prepared cache budget before inference')
        self._prepared_cache_budget = int(max_bytes)

    @torch.no_grad()
    def prepared_weight(self, name, device):
        if self._prepared_cache_budget == 0:
            return None
        info = self.header[name]
        if len(info['shape']) != 2 or info['dtype'] not in ('F16', 'F32'):
            return None
        key = (name, str(device))
        if key in self._prepared_cache:
            return self._prepared_cache[key]
        rows, columns = info['shape']
        size = rows * columns * 2 + rows * 4
        if self._prepared_cache_bytes + size > self._prepared_cache_budget:
            return None
        weights = torch.empty((rows, columns), device=device, dtype=torch.float16)
        scales = torch.empty(rows, device=device, dtype=torch.float32)
        for start in range(0, rows, 2048):
            stop = min(rows, start + 2048)
            w = self.device_rows(name, start, stop, device)
            if w is None:
                w = torch.from_numpy(np.array(self.numpy(name)[start:stop], dtype=np.float32)).to(device)
            scale = torch.pow(2.0, torch.ceil(torch.log2(w.abs().amax(-1).clamp(min=1))))
            weights[start:stop] = (w / scale[:, None]).half()
            scales[start:stop] = scale
        self._prepared_cache[key] = (weights, scales)
        self._prepared_cache_bytes += size
        original = self._device_cache.pop(key, None)
        if original is not None:
            self._device_cache_bytes -= original.numel() * original.element_size()
        return weights, scales


    def enable_device_cache(self, max_bytes=8 * 1024**3):
        self._device_cache_budget = max_bytes

    def device_rows(self, name, start, stop, device):
        if self._device_cache_budget <= 0 or device.type != 'npu':
            return None
        key = (name, str(device))
        if key not in self._device_cache:
            data = self.numpy(name)
            if self.header[name]['dtype'] == 'BF16':
                return None
            if self._device_cache_bytes + data.nbytes > self._device_cache_budget:
                return None
            self._device_cache[key] = torch.from_numpy(data.copy()).to(device)
            self._device_cache_bytes += data.nbytes
        return self._device_cache[key][start:stop].float()

    def numpy(self, name):
        info = self.header[name]
        start, stop = info['data_offsets']
        types = {'F16': '<f2', 'F32': '<f4', 'BF16': '<u2'}
        dtype = types.get(info['dtype'])
        if dtype is None:
            raise NotImplementedError(info['dtype'])
        size = math.prod(info['shape']) * np.dtype(dtype).itemsize
        if size != stop - start or self.data_offset + stop > self.path.stat().st_size:
            raise ValueError(f'Invalid tensor extent: {name}')
        if name not in self._maps:
            self._maps[name] = np.memmap(self.path, mode='r', dtype=dtype,
                                       offset=self.data_offset + start, shape=tuple(info['shape']))
        return self._maps[name]

    def tensor(self, name, device='cpu', dtype=torch.float32):
        data = self.numpy(name)
        if self.header[name]['dtype'] == 'BF16':
            data = (data.astype(np.uint32) << 16).view(np.float32)
        else:
            data = data.copy()
        return torch.from_numpy(data).to(device=device, dtype=dtype)

    def linear(self, name, device='cpu', row_chunk=512):
        return SafeLinear(self, name, row_chunk).to(device)


class SafeLinear(torch.nn.Module):
    """Stream FP16/FP32 dense safetensor weights using the scaled linear backend."""
    def __init__(self, store, name, row_chunk=512):
        super().__init__()
        from .layers import GGUFLinear
        from types import SimpleNamespace
        # Reuse one tested execution implementation with a storage adapter.
        class Adapter:
            gguf = SimpleNamespace(tensors=store.header)
            def info(self, key):
                return SimpleNamespace(shape=tuple(store.header[key]['shape']))
            def tensor(self, key):
                return store.tensor(key).numpy()
            def rows(self, key, start, stop):
                return np.asarray(store.numpy(key)[start:stop], dtype=np.float32)
            def prepared_weight(self, key, device):
                return store.prepared_weight(key, device)
            def device_rows(self, key, start, stop, device):
                return store.device_rows(key, start, stop, device)
        self.linear = GGUFLinear(Adapter(), name, row_chunk=row_chunk)

    def forward(self, x):
        return self.linear(x)
