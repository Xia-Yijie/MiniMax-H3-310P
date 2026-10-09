"""Qwen3-VL-32B layer-50 text/image conditioning from the local GGUF.

For text-only input all three mRoPE axes coincide, so ordinary 1D RoPE
is equivalent. Images use encode_images with the vision tower and mRoPE;
video-reference conditioning is not implemented here.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F

from .attention import streaming_attention
from .h3 import apply_rope
from .layers import GGUFLinear, RMSNorm
from .weights import TensorStore


class QwenTextLayer(nn.Module):
    def __init__(self, store, index, row_chunk=512):
        super().__init__()
        prefix = f'model.layers.{index}'
        self.input_norm = RMSNorm(store.tensor(prefix + '.input_layernorm.weight'), 1e-6)
        self.post_norm = RMSNorm(store.tensor(prefix + '.post_attention_layernorm.weight'), 1e-6)
        attn = prefix + '.self_attn'
        self.q_norm = RMSNorm(store.tensor(attn + '.q_norm.weight'), 1e-6)
        self.k_norm = RMSNorm(store.tensor(attn + '.k_norm.weight'), 1e-6)
        self.head_dim = self.q_norm.weight.numel()
        self.q_proj = GGUFLinear(store, attn + '.q_proj', row_chunk=row_chunk)
        self.k_proj = GGUFLinear(store, attn + '.k_proj', row_chunk=row_chunk)
        self.v_proj = GGUFLinear(store, attn + '.v_proj', row_chunk=row_chunk)
        self.o_proj = GGUFLinear(store, attn + '.o_proj', row_chunk=row_chunk)
        self.num_heads = self.q_proj.out_features // self.head_dim
        self.kv_heads = self.k_proj.out_features // self.head_dim
        self.gate_proj = GGUFLinear(store, prefix + '.mlp.gate_proj', row_chunk=row_chunk)
        self.up_proj = GGUFLinear(store, prefix + '.mlp.up_proj', row_chunk=row_chunk)
        self.down_proj = GGUFLinear(store, prefix + '.mlp.down_proj', row_chunk=row_chunk)

    def forward(self, x, rope):
        h = self.input_norm(x)
        q = self.q_norm(self.q_proj(h).reshape(-1, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(h).reshape(-1, self.kv_heads, self.head_dim))
        v = self.v_proj(h).reshape(-1, self.kv_heads, self.head_dim)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        repeat = self.num_heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        h = streaming_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                v.transpose(0, 1)[None], causal=True)
        x = x + self.o_proj(h[0].transpose(0, 1).reshape(x.shape[0], -1))
        h = self.post_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class QwenTextEncoder(nn.Module):
    def __init__(self, checkpoint, tokenizer_path, row_chunk=512):
        super().__init__()
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.store = TensorStore(checkpoint)
        indices = sorted({int(m.group(1)) for name in self.store.gguf.tensors
                          if (m := re.match(r'^model\.layers\.(\d+)\.', name))})
        if indices != list(range(50)):
            raise ValueError(f'Expected exactly 50 retained Qwen layers, got {indices}')
        self.layers = nn.ModuleList(QwenTextLayer(self.store, i, row_chunk) for i in indices)
        self.stream_quant_cache_bytes = 0

    @torch.no_grad()
    def encode(self, prompt, device, progress=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Prompt must be nonempty')
        if any(tag in prompt for tag in ('<|image_pad|>', '<|video_pad|>')):
            raise NotImplementedError('This entry point supports text-only conditioning')
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(ids) > 512:
            raise ValueError('Reference backend currently limits prompts to 512 tokens')
        # Look up only requested rows of the 151936 x 5120 embedding table.
        import numpy as np
        embeddings = np.concatenate([self.store.rows('model.embed_tokens.weight', i, i + 1) for i in ids])
        x = torch.from_numpy(embeddings).to(device)
        frequency = 1.0 / (5000000.0 ** (torch.arange(0, 128, 2, device=device).float() / 128))
        half = torch.arange(len(ids), device=device).float()[:, None] * frequency
        rope = torch.cat((half, half), dim=-1)
        if self.stream_quant_cache_bytes:
            self.store.enable_quant_device_cache(self.stream_quant_cache_bytes)
        for index, layer in enumerate(self.layers):
            x = layer(x, rope)
            if progress:
                progress(index + 1, len(self.layers), x)
            if self.stream_quant_cache_bytes:
                # Keep only a layer's packed weights on NPU, dequantizing there.
                # This can coexist with a resident video backbone.
                self.store._device_cache.clear()
                self.store._device_cache_bytes = 0
        # H3 consumes hidden state after layer 50, without Qwen final RMSNorm.
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError('Text encoder produced NaN/Inf')
        return x

    @torch.no_grad()
    def encode_images(self, prompt, images, device, progress=None, vision_progress=None):
        """Ordered <Picture N> presentation, vision embedding and DeepStack."""
        from transformers import Qwen2VLImageProcessor
        from .vision_encoder import QwenVisionEncoder
        if not prompt.strip() or any(tag in prompt for tag in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>')):
            raise ValueError('Use ordinary text; visual markers are inserted by the encoder')
        processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
            min_pixels=65536, max_pixels=16777216, image_mean=[.5]*3, image_std=[.5]*3)
        data = processor(images=images, return_tensors='pt')
        grids=data['image_grid_thw'].tolist()
        tower=QwenVisionEncoder(self.store,self.layers[0].q_proj.row_chunk).to(device).eval()
        ids,tags,positions=[],[],[]
        features,deep_features=[],[[],[],[]]
        offset=0;cursor=0
        image_mask=[]
        def text_part(values):
            nonlocal cursor
            ids.extend(values);tags.extend([1]*len(values));image_mask.extend([False]*len(values))
            positions.extend([[cursor+i]*3 for i in range(len(values))]);cursor+=len(values)
        for index,grid in enumerate(grids):
            t,h,w=grid;count=t*h*w
            pixels=data['pixel_values'][offset:offset+count].to(device)
            visual,deep=tower.encode_image(pixels,grid,vision_progress)
            features.append(visual)
            for i,d in enumerate(deep):deep_features[i].append(d)
            offset+=count
            text_part(self.tokenizer.encode(f'<Picture {index+1}>: ',add_special_tokens=False).ids)
            # vision_start/end are visual tags for H3 but ordinary scalar RoPE
            # positions for Qwen. image_pad has a merged 2D grid.
            text_part([self.tokenizer.token_to_id('<|vision_start|>')]);tags[-1]=0
            hh,ww=h//2,w//2
            ids.extend([self.tokenizer.token_to_id('<|image_pad|>')]*(hh*ww));tags.extend([0]*(hh*ww))
            image_mask.extend([True]*(hh*ww))
            positions.extend([[cursor,cursor+y,cursor+x] for y in range(hh) for x in range(ww)])
            cursor+=max(hh,ww)
            text_part([self.tokenizer.token_to_id('<|vision_end|>')]);tags[-1]=0
        text_part(self.tokenizer.encode(prompt,add_special_tokens=False).ids)
        del tower
        import numpy as np
        x=torch.from_numpy(np.concatenate([self.store.rows('model.embed_tokens.weight',i,i+1) for i in ids])).to(device)
        mask=torch.tensor(image_mask,device=device,dtype=torch.bool)
        x[mask]=torch.cat(features)
        rope=image_text_rope(torch.tensor(positions,device=device).float())
        deep=[torch.cat(parts) for parts in deep_features]
        for index,layer in enumerate(self.layers):
            x=layer(x,rope)
            if index<3:x[mask]=x[mask]+deep[index]
            if progress:progress(index+1,len(self.layers),x)
        if not bool(torch.isfinite(x).all()):raise FloatingPointError('Visual text encoder produced NaN/Inf')
        return x,torch.tensor(tags,device=device,dtype=torch.long)


def image_text_rope(positions):
    """Interleaved [24,20,20] mRoPE, positions [L,T/H/W]."""
    frequency=1/(5000000**(torch.arange(0,128,2,device=positions.device).float()/128))
    axes=positions.T[:,:,None]*frequency
    half=axes[0].clone()
    half[:,1:60:3]=axes[1,:,1:60:3]
    half[:,2:60:3]=axes[2,:,2:60:3]
    return torch.cat((half,half),-1)
