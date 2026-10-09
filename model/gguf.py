"""Read GGUF metadata and map individual tensor bytes without loading the model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # PyTorch order; GGUF stores dimensions in reverse order.
    quant_type: int
    offset: int  # Absolute file offset.

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


class GGUFFile:
    """Version 2/3 little-endian GGUF reader. No GPU/NPU initialization."""

    _formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata = {}
        self.tensors = {}
        with self.path.open('rb') as f:
            def read(fmt):
                size = struct.calcsize('<' + fmt)
                data = f.read(size)
                if len(data) != size:
                    raise ValueError('Truncated GGUF header')
                return struct.unpack('<' + fmt, data)[0]

            def string():
                size = read('Q')
                if size > self.path.stat().st_size:
                    raise ValueError('Invalid GGUF string size')
                data = f.read(size)
                if len(data) != size:
                    raise ValueError('Truncated GGUF string')
                return data.decode('utf-8')

            def value(kind):
                if kind == 8:
                    return string()
                if kind == 9:
                    item_kind, count = read('I'), read('Q')
                    if count > self.path.stat().st_size:
                        raise ValueError('Invalid GGUF array size')
                    return [value(item_kind) for _ in range(count)]
                if kind not in self._formats:
                    raise ValueError(f'Unsupported metadata type {kind}')
                return read(self._formats[kind])

            if f.read(4) != b'GGUF':
                raise ValueError('Not a little-endian GGUF file')
            self.version = read('I')
            if self.version not in (2, 3):
                raise ValueError(f'Unsupported GGUF version {self.version}')
            count, kv_count = read('Q'), read('Q')
            for _ in range(kv_count):
                key = string()
                self.metadata[key] = value(read('I'))
            pending = []
            for _ in range(count):
                name = string()
                ndim = read('I')
                if not 1 <= ndim <= 8:
                    raise ValueError(f'Invalid rank {ndim} for {name}')
                shape = tuple(reversed([read('Q') for _ in range(ndim)]))
                kind, offset = read('I'), read('Q')
                pending.append((name, shape, kind, offset))
            alignment = int(self.metadata.get('general.alignment', 32))
            if alignment <= 0:
                raise ValueError('Invalid GGUF alignment')
            self.data_offset = (f.tell() + alignment - 1) // alignment * alignment
            for name, shape, kind, offset in pending:
                if name in self.tensors:
                    raise ValueError(f'Duplicate tensor {name}')
                self.tensors[name] = TensorInfo(name, shape, kind, self.data_offset + offset)

    def raw(self, name: str, block_elements: int, block_bytes: int) -> np.memmap:
        info = self.tensors[name]
        if info.elements % block_elements:
            raise ValueError(f'Tensor {name} is not aligned to quantization blocks')
        nbytes = info.elements // block_elements * block_bytes
        if info.offset + nbytes > self.path.stat().st_size:
            raise ValueError(f'Tensor {name} extends past the end of the file')
        return np.memmap(self.path, mode='r', dtype=np.uint8, offset=info.offset, shape=(nbytes,))

    def summary(self) -> dict:
        kinds = {}
        for info in self.tensors.values():
            kinds[str(info.quant_type)] = kinds.get(str(info.quant_type), 0) + 1
        return {'path': str(self.path), 'version': self.version,
                'architecture': self.metadata.get('general.architecture'),
                'tensor_count': len(self.tensors), 'quantization_types': kinds,
                'file_bytes': self.path.stat().st_size}
