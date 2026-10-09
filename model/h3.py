"""Inference-only pruned H3 backbone for explicitly packed audio/video tokens.

This module follows the ComfyUI checkpoint names and contiguous Q/K/V layout.
The tokenizer, VAEs, sampler and packed-position builder are connected by
infer.generate; this module exposes the backbone separately.
"""
from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F

from .attention import streaming_attention
from .layers import GGUFLinear, RMSNorm, swiglu
from .weights import TensorStore


def move_between_devices(tensor, device):
    """310P peer DMA is unavailable locally; stage inter-NPU transfers on CPU."""
    device = torch.device(device)
    if tensor.device.type == 'npu' and device.type == 'npu' and tensor.device != device:
        return tensor.cpu().to(device)
    return tensor.to(device)


@dataclass(frozen=True)
class H3Config:
    hidden_size: int
    num_heads: int
    head_dim: int
    ffn_size: int
    num_layers: int
    num_refiners: int
    time_dim: int
    text_dim: int
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5

    @classmethod
    def from_store(cls, store):
        tensors = store.gguf.tensors
        hidden = tensors['blocks.0.norm1.weight'].shape[0]
        head_dim = tensors['blocks.0.attn.q_norm.weight'].shape[0]
        inner = tensors['blocks.0.attn.qkv_proj.weight'].shape[0] // 3
        block_ids = sorted({int(m.group(1)) for n in tensors
                            if (m := re.match(r'^blocks\.(\d+)\.', n))})
        refiner_ids = sorted({int(m.group(1)) for n in tensors
                              if (m := re.match(r'^token_refiner\.blocks\.(\d+)\.', n))})
        if block_ids != list(range(len(block_ids))) or refiner_ids != list(range(len(refiner_ids))):
            raise ValueError('Non-contiguous block IDs')
        if inner % head_dim:
            raise ValueError('Incompatible attention head dimensions')
        table = tensors.get('adaln_t_table')
        if table is None:
            raise NotImplementedError('Only AdaLN-curve pruned checkpoints are supported')
        config = cls(hidden, inner // head_dim, head_dim,
                     tensors['blocks.0.mlp.fc2.weight'].shape[1],
                     len(block_ids), len(refiner_ids), table.shape[1],
                     tensors['condition_proj.weight'].shape[1])
        if tensors['blocks.0.adaln_proj.linear.weight'].shape != (hidden * 18, config.time_dim):
            raise ValueError('Unexpected three-modality AdaLN layout')
        return config


def prepare_rope(angles):
    """Cache trig tables on an immutable, local-to-forward angle tensor."""
    angles._h3_cos = angles.float().cos()
    angles._h3_sin = angles.float().sin()
    return angles


def apply_rope(x, angles):
    """Split-half RoPE on the first rotary dimensions of [S,H,D]."""
    dim = angles.shape[-1]
    if angles.shape[0] != x.shape[0] or dim % 2 or dim > x.shape[-1]:
        raise ValueError('Invalid partial RoPE dimensions')
    rotary = x[..., :dim].float()
    left, right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    cosine = getattr(angles, '_h3_cos', None)
    sine = getattr(angles, '_h3_sin', None)
    if cosine is None:
        cosine, sine = angles.float().cos(), angles.float().sin()
    result = rotary * cosine[:, None] + rotated * sine[:, None]
    return torch.cat((result, x[..., dim:].float()), dim=-1)


def normalize_rotary(norm, x, rope=None, chunk=1024):
    output = torch.empty_like(x, dtype=torch.float32)
    for start in range(0, len(x), chunk):
        end = start + chunk
        part = norm(x[start:end])
        if rope is not None:
            angle = rope[start:end]
            if hasattr(rope, '_h3_cos'):
                angle._h3_cos = rope._h3_cos[start:end]
                angle._h3_sin = rope._h3_sin[start:end]
            part = apply_rope(part, angle)
        output[start:end] = part
    return output


class H3Attention(nn.Module):
    def __init__(self, store, prefix, config, **linear_options):
        super().__init__()
        self.config = config
        self.qkv = GGUFLinear(store, prefix + '.qkv_proj', **linear_options)
        self.proj = GGUFLinear(store, prefix + '.out_proj', **linear_options)
        self.q_norm = RMSNorm(store.tensor(prefix + '.q_norm.weight'), config.qk_norm_eps)
        self.k_norm = RMSNorm(store.tensor(prefix + '.k_norm.weight'), config.qk_norm_eps)

    def forward(self, x, rope=None):
        c = self.config
        # This repack uses contiguous Q/K/V, not [head, QKV, head_dim].
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = normalize_rotary(self.q_norm, q.reshape(-1, c.num_heads, c.head_dim), rope)
        k = normalize_rotary(self.k_norm, k.reshape(-1, c.num_heads, c.head_dim), rope)
        v = v.reshape(-1, c.num_heads, c.head_dim)
        # Release the fused QKV allocation after Q/K normalization and RoPE.
        # Otherwise V's strided view retains three full matrices for attention.
        v = v.contiguous()
        attention = streaming_attention(q.transpose(0, 1)[None],
                                        k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        return self.proj(attention[0].transpose(0, 1).reshape(x.shape[0], -1))


class H3MLP(nn.Module):
    def __init__(self, store, prefix, **linear_options):
        super().__init__()
        self.up = GGUFLinear(store, prefix + '.fc1', **linear_options)
        self.down = GGUFLinear(store, prefix + '.fc2', **linear_options)
        self.token_chunk_size = 0
        self.fused_npu = False

    def forward(self, x):
        if self.token_chunk_size and x.shape[0] > self.token_chunk_size:
            output = torch.empty_like(x, dtype=torch.float32)
            for start in range(0,x.shape[0],self.token_chunk_size):
                output[start:start+self.token_chunk_size] = self._forward_chunk(x[start:start+self.token_chunk_size])
            return output
        return self._forward_chunk(x)

    def _forward_chunk(self, x):
        return self.down(swiglu(self.up(x), self.fused_npu))


def modulated_norm(norm, x, scale, shift, indices, chunk=1024):
    output = torch.empty_like(x, dtype=torch.float32)
    for start in range(0, len(x), chunk):
        end = start + chunk
        rows = indices[start:end]
        part = norm(x[start:end])
        part.mul_(1 + scale[rows]).add_(shift[rows])
        output[start:end] = part
    return output


def gated_residual(update, x, gate, indices, chunk=1024):
    # update is a fresh attention/MLP output owned by this block.
    for start in range(0, len(x), chunk):
        end = start + chunk
        update[start:end].mul_(gate[indices[start:end]]).add_(x[start:end])
    return update


class H3Block(nn.Module):
    def __init__(self, store, index, config=None, **linear_options):
        super().__init__()
        config = config or H3Config.from_store(store)
        prefix = f'blocks.{index}'
        self.hidden_size = config.hidden_size
        self.norm1 = RMSNorm(store.tensor(prefix + '.norm1.weight'), config.norm_eps)
        self.norm2 = RMSNorm(store.tensor(prefix + '.norm2.weight'), config.norm_eps)
        self.attn = H3Attention(store, prefix + '.attn', config, **linear_options)
        self.mlp = H3MLP(store, prefix + '.mlp', **linear_options)
        self.adaln = GGUFLinear(store, prefix + '.adaln_proj.linear', compute_dtype=torch.float32)

    @torch.no_grad()
    def forward(self, x, time_coordinates, modulation_indices, rope=None):
        mods = self.adaln(time_coordinates).reshape(-1, self.hidden_size * 6)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
        indices = modulation_indices.to(device=x.device, dtype=torch.long)
        if indices.shape != (x.shape[0],):
            raise ValueError('One modulation index per packed token is required')
        # Bound modulation temporaries independently of video duration.
        h = modulated_norm(self.norm1, x, scale1, shift1, indices)
        update = self.attn(h, rope)
        del h
        x = gated_residual(update, x, gate1, indices)
        h = modulated_norm(self.norm2, x, scale2, shift2, indices)
        update = self.mlp(h)
        del h
        return gated_residual(update, x, gate2, indices)


class H3RefinerBlock(nn.Module):
    def __init__(self, store, index, config, **linear_options):
        super().__init__()
        prefix = f'token_refiner.blocks.{index}'
        self.norm1 = RMSNorm(store.tensor(prefix + '.norm1.weight'), config.norm_eps)
        self.norm2 = RMSNorm(store.tensor(prefix + '.norm2.weight'), config.norm_eps)
        self.attn = H3Attention(store, prefix + '.attn', config, **linear_options)
        self.mlp = H3MLP(store, prefix + '.mlp', **linear_options)

    def forward(self, x):
        x = x.float() + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class H3Backbone(nn.Module):
    """Single packed sample. Input timesteps are t=1-sigma, not sigma.

    Token layout, modality IDs, and per-token time indices come from
    model.layout. Do not guess these values from tensor shapes.
    """
    def __init__(self, checkpoint, *, row_chunk=512, compute_dtype=torch.float16):
        super().__init__()
        if str(checkpoint).endswith('.safetensors'):
            from .community_weights import CommunityTensorStore
            self.store = CommunityTensorStore(checkpoint)
        else:
            self.store = TensorStore(checkpoint)
        self.config = H3Config.from_store(self.store)
        c = self.config
        options = {'row_chunk': row_chunk, 'compute_dtype': compute_dtype}
        self.register_buffer('time_table', torch.from_numpy(self.store.tensor('adaln_t_table')))
        self.register_buffer('rope_inv_freq', torch.from_numpy(self.store.tensor('rope.inv_freq')))
        self.condition_proj = GGUFLinear(self.store, 'condition_proj', **options)
        self.refiners = nn.ModuleList(H3RefinerBlock(self.store, i, c, **options) for i in range(c.num_refiners))
        self.refiner_norm = RMSNorm(self.store.tensor('token_refiner.final_norm.weight'), c.norm_eps)
        self.video_proj = GGUFLinear(self.store, 'video_patch_proj', row_chunk=row_chunk, compute_dtype=torch.float32)
        self.audio_proj = GGUFLinear(self.store, 'audio_patch_proj', row_chunk=row_chunk, compute_dtype=torch.float32)
        self.blocks = nn.ModuleList(H3Block(self.store, i, c, **options) for i in range(c.num_layers))
        self.final_norm = RMSNorm(self.store.tensor('final_layer.norm.weight'), c.norm_eps)
        self.final_adaln = GGUFLinear(self.store, 'final_layer.adaln_proj.linear', compute_dtype=torch.float32)
        self.video_out = GGUFLinear(self.store, 'final_layer.video_out', compute_dtype=torch.float32)
        self.audio_out = GGUFLinear(self.store, 'final_layer.audio_out', compute_dtype=torch.float32)

    def time_coordinates(self, timesteps):
        table = self.time_table.to(timesteps.device)
        position = timesteps.float().clamp(0, 1) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return table[lower] + (table[lower + 1] - table[lower]) * (position - lower)[:, None]

    def split_devices(self, primary, secondary):
        """Place the latter half of blocks on a second device, preserving order."""
        primary, secondary = torch.device(primary), torch.device(secondary)
        if primary == secondary:
            raise ValueError('Split devices must be distinct')
        self.cpu().to(primary)
        boundary = len(self.blocks) // 2
        self.block_devices = [primary] * boundary + [secondary] * (len(self.blocks) - boundary)
        for block, device in zip(self.blocks, self.block_devices):
            if device != primary:
                block.cpu().to(device)
        return self

    def rope_angles(self, position_ids):
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError('Position IDs must be [S,3]')
        angles = position_ids.float()[..., None] * self.rope_inv_freq.to(position_ids.device)
        half = angles.flatten(1)
        return torch.cat((half, half), dim=-1)

    @torch.no_grad()
    def encode_condition(self, text_states):
        if text_states.ndim != 2 or text_states.shape[-1] != self.config.text_dim:
            raise ValueError('Expected one sample of Qwen hidden states [L,5120]')
        x = self.condition_proj(text_states)
        for block in self.refiners:
            x = block(x)
        return self.refiner_norm(x)

    @torch.no_grad()
    def forward_packed(self, hidden_states, timesteps, timestep_indices, modality_ids,
                       position_ids, *, video_indices, audio_indices, progress=None):
        """Return raw projected video/audio flow rows; no schedule conversion."""
        size = hidden_states.shape[0]
        if hidden_states.shape != (size, self.config.hidden_size):
            raise ValueError('Hidden states must be [S,hidden_size]')
        if timesteps.ndim != 1 or timesteps.numel() == 0:
            raise ValueError('Timesteps must be a nonempty vector')
        if timestep_indices.shape != (size,) or modality_ids.shape != (size,):
            raise ValueError('Time indices and modality IDs must be [S]')
        if timestep_indices.dtype != torch.long or modality_ids.dtype != torch.long:
            raise ValueError('Time indices and modality IDs must be int64')
        if position_ids.shape != (size, 3):
            raise ValueError('Position IDs must be [S,3]')
        if bool(((modality_ids < 0) | (modality_ids >= 3)).any()):
            raise ValueError('Modality IDs must be in [0,3)')
        if bool(((timestep_indices < 0) | (timestep_indices >= timesteps.numel())).any()):
            raise ValueError('Invalid timestep index')
        coordinates = self.time_coordinates(timesteps)
        modulation_indices = timestep_indices * 3 + modality_ids
        rope = prepare_rope(self.rope_angles(position_ids))
        x = hidden_states.float()
        for index, block in enumerate(self.blocks):
            target = getattr(self, 'block_devices', [x.device] * len(self.blocks))[index]
            if x.device != target:
                x = move_between_devices(x, target)
                coordinates = move_between_devices(coordinates, target)
                modulation_indices = move_between_devices(modulation_indices, target)
                rope = prepare_rope(move_between_devices(rope, target))
            x = block(x, coordinates, modulation_indices, rope)
            if progress:
                progress(index + 1, len(self.blocks), x)
        x = move_between_devices(x, hidden_states.device)
        coordinates = move_between_devices(coordinates, hidden_states.device)
        shift, scale = self.final_adaln(coordinates).chunk(2, dim=-1)
        normalized = self.final_norm(x) * (1 + scale[timestep_indices]) + shift[timestep_indices]
        return {'video_rows': self.video_out(normalized[video_indices]),
                'audio_rows': self.audio_out(normalized[audio_indices])}
