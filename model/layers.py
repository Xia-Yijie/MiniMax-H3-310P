"""GGUF-backed linear layers; FP16 matmul with FP32 scaling/residuals."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def swiglu(x, fused=False):
    if fused and x.device.type == 'npu':
        import torch_npu
        return torch_npu.npu_swiglu(x, dim=-1)
    gate, up = x.chunk(2,dim=-1)
    return F.silu(gate.float()) * up.float()


def set_fused_ops(model, enabled):
    count = 0
    for module in model.modules():
        if hasattr(module, 'fused_npu'):
            module.fused_npu = enabled
            count += 1
    return count


class GGUFLinear(nn.Module):
    """Reference streaming implementation; no full dense weight allocation.

    Default weights are mapped on the host. A TensorStore may optionally cache
    Q8 bytes on NPU and decode GEMM row tiles there. Matrix multiplication still
    uses FP16/FP32, not native INT8 GEMM acceleration.
    """
    def __init__(self, store, name, *, row_chunk=512, compute_dtype=torch.float16):
        super().__init__()
        self.store, self.name = store, name
        info = store.info(name + '.weight')
        if len(info.shape) != 2 or row_chunk <= 0:
            raise ValueError('Expected matrix weights and a positive row chunk')
        self.out_features, self.in_features = info.shape
        self.row_chunk, self.compute_dtype = row_chunk, compute_dtype
        self.direct_fp16 = False
        self.native_int8 = False
        self._unit_scale_key = None
        self._unit_scale = False
        bias_name = name + '.bias'
        self.register_buffer('bias', torch.from_numpy(store.tensor(bias_name))
                             if bias_name in store.gguf.tensors else None)

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f'{self.name}: expected {self.in_features} input features')
        if self.compute_dtype not in (torch.float16, torch.float32):
            raise ValueError('310P reference backend uses FP16 or FP32')
        flat = x.reshape(-1, self.in_features).float()
        transform = getattr(self.store, 'input_transform', None)
        if transform is not None:
            flat = transform(self.name + '.weight', flat)
        if self.native_int8 and self.compute_dtype == torch.float16:
            reader = getattr(self.store, 'native_quant_weight', None)
            quantized = reader(self.name + '.weight', x.device) if reader else None
            if quantized is not None:
                import torch_npu
                weight, scale, restore = quantized
                input_scale = flat.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
                activation = (flat / input_scale).round().clamp(-127,127).to(torch.int8)
                values = torch_npu.npu_quant_matmul(activation, weight, scale, output_dtype=torch.float16).float()
                values.mul_(restore[None]).mul_(input_scale)
                if self.bias is not None:values.add_(self.bias.to(x.device).float())
                return values.reshape(*x.shape[:-1], self.out_features)
        cached = None
        if self.compute_dtype == torch.float16:
            reader = getattr(self.store, 'prepared_weight', None)
            cached = reader(self.name + '.weight', x.device) if reader else None
            if cached is not None and self.direct_fp16:
                weight, scale = cached
                key = (id(scale),str(scale.device))
                if self._unit_scale_key != key:
                    self._unit_scale = bool((scale == 1).all())
                    self._unit_scale_key = key
                values = F.linear(flat.half(),weight).float()
                if not self._unit_scale:
                    values.mul_(scale[None])
                if self.bias is not None:
                    values.add_(self.bias.to(x.device).float())
                return values.reshape(*x.shape[:-1],self.out_features)
        if self.compute_dtype == torch.float16:
            # Bounds the magnitude of the FP16 dot products; undo in FP32.
            input_scale = torch.pow(2.0, torch.ceil(torch.log2(flat.abs().amax(-1, keepdim=True).clamp(min=1))))
            prepared = (flat / input_scale).half()
        else:
            input_scale, prepared = 1.0, flat
        if self.compute_dtype == torch.float16:
            if cached is not None:
                weight, scale = cached
                values = F.linear(prepared, weight).float()
                values.mul_(input_scale).mul_(scale[None])
                if self.bias is not None:
                    values.add_(self.bias.to(x.device).float())
                return values.reshape(*x.shape[:-1], self.out_features)
        output = torch.empty((flat.shape[0], self.out_features), device=x.device, dtype=torch.float32)
        for start in range(0, self.out_features, self.row_chunk):
            stop = min(start + self.row_chunk, self.out_features)
            reader = getattr(self.store, 'device_rows', None) or getattr(self.store, 'device_q8_rows', None)
            weight_on_device = reader(self.name + '.weight', start, stop, x.device) if reader else None
            if weight_on_device is not None:
                if self.compute_dtype == torch.float16:
                    weight_scale = torch.pow(2.0, torch.ceil(torch.log2(weight_on_device.abs().amax(-1).clamp(min=1))))
                    device_weight = (weight_on_device / weight_scale[:, None]).half()
                    values = F.linear(prepared, device_weight).float()
                    values.mul_(input_scale).mul_(weight_scale[None])
                else:
                    values = F.linear(prepared, weight_on_device)
            elif self.compute_dtype == torch.float16:
                weight = self.store.rows(self.name + '.weight', start, stop)
                weight_scale = np.exp2(np.ceil(np.log2(np.maximum(np.max(np.abs(weight), axis=1), 1))))
                weight = np.ascontiguousarray(weight / weight_scale[:, None], dtype=np.float16)
                device_weight = torch.from_numpy(weight).to(x.device)
                values = F.linear(prepared, device_weight).float()
                values.mul_(input_scale).mul_(torch.from_numpy(weight_scale.copy()).to(x.device)[None])
            else:
                weight = self.store.rows(self.name + '.weight', start, stop)
                device_weight = torch.from_numpy(weight).to(x.device)
                values = F.linear(prepared, device_weight)
            if self.bias is not None:
                values.add_(self.bias[start:stop].to(x.device).float())
            output[:, start:stop] = values
        return output.reshape(*x.shape[:-1], self.out_features)


class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-5):
        super().__init__()
        self.register_buffer('weight', torch.as_tensor(weight, dtype=torch.float32))
        self.eps = eps
        self.fused_npu = False

    def forward(self, x):
        x = x.float()
        if self.fused_npu and x.device.type == 'npu':
            import torch_npu
            return torch_npu.npu_rms_norm(x, self.weight.to(x.device), epsilon=self.eps)[0]
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight.to(x.device)
