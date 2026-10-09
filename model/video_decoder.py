"""Video VAE decoder for the Comfy-Org H3 FP16 safetensors checkpoint.

Architecture/temporal overlap equations checked against DiffSynth-Studio
(Apache-2.0). No convolution-3D operator is needed in the decode-only path.
"""
import torch
import re
from torch import nn
from torch.nn import functional as F

from .attention import streaming_attention
from .h3 import apply_rope, prepare_rope
from .layers import RMSNorm, swiglu
from .safetensors_store import SafeTensorStore, SafeLinear


class VideoDecoderBlock(nn.Module):
    def __init__(self, store, index, row_chunk=512):
        super().__init__()
        prefix = f'decoder.transformer_blocks.{index}'
        self.norm1 = RMSNorm(store.tensor(prefix + '.norm1.weight'))
        self.norm2 = RMSNorm(store.tensor(prefix + '.norm2.weight'))
        self.register_buffer('scale1', store.tensor(prefix + '.scale1'))
        self.register_buffer('scale2', store.tensor(prefix + '.scale2'))
        self.qkv = SafeLinear(store, prefix + '.attn.to_qkv', row_chunk)
        self.out = SafeLinear(store, prefix + '.attn.to_out', row_chunk)
        self.up = SafeLinear(store, prefix + '.ff.w1', row_chunk)
        self.down = SafeLinear(store, prefix + '.ff.w2', row_chunk)
        self.fused_npu = False
        self.fused_qk_norm = False
        self.register_buffer('unit_qk_norm',torch.ones(64),persistent=False)

    def forward(self, x, rope):
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        batch, length = x.shape[:2]
        # The VAE weights keep [head, QKV, head_dim] output packing.
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 32, 192).chunk(3, -1)
        if self.fused_qk_norm and x.device.type=='npu':
            import torch_npu
            q = torch_npu.npu_rms_norm(q,self.unit_qk_norm,epsilon=1e-5)[0]
            k = torch_npu.npu_rms_norm(k,self.unit_qk_norm,epsilon=1e-5)[0]
        else:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-5)
            k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-5)
        angles = rope if batch == 1 else prepare_rope(rope.repeat(batch, 1))
        q = apply_rope(q.reshape(-1,32,64), angles).reshape(batch,length,32,64)
        k = apply_rope(k.reshape(-1,32,64), angles).reshape(batch,length,32,64)
        attended = streaming_attention(q.transpose(1,2), k.transpose(1,2),
                                        v.transpose(1,2)).transpose(1,2).reshape(batch,length,-1)
        x = x + self.out(attended) * self.scale1.to(x.device)
        hidden = swiglu(self.up(self.norm2(x)),self.fused_npu)
        result = x + self.down(hidden) * self.scale2.to(x.device)
        return result[0] if unbatched else result


class VideoDecoder(nn.Module):
    def __init__(self, checkpoint, row_chunk=512, tile_size=256, tile_overlap=64, tile_batch_size=1):
        super().__init__()
        if tile_size < 32 or tile_size % 16 or tile_overlap < 0 or tile_overlap % 16 or tile_overlap >= tile_size:
            raise ValueError('VAE tile size/overlap must be multiples of 16 with 0 <= overlap < size')
        self.tile_size, self.tile_overlap = tile_size, tile_overlap
        if tile_batch_size < 1:
            raise ValueError('Tile batch size must be positive')
        self.tile_batch_size = tile_batch_size
        self.store = SafeTensorStore(checkpoint)
        self.store.enable_device_cache()
        store = self.store
        self.x_embed = SafeLinear(store, 'decoder.x_embedder', row_chunk)
        self.out = SafeLinear(store, 'decoder.proj_out', row_chunk)
        indices = sorted({int(match.group(1)) for key in store.header
                          if (match := re.match(r'^decoder\.transformer_blocks\.(\d+)\.',key))})
        if len(indices) not in (26,36) or indices != list(range(len(indices))):
            raise ValueError('Expected complete official 36-layer or LynnReal 26-layer decoder')
        self.geometry = 'lynn_release' if len(indices)==26 else 'official'
        self.blocks = nn.ModuleList(VideoDecoderBlock(store, i, row_chunk) for i in indices)
        for name, key in [('register_tokens', 'decoder.register_tokens'),
                          ('mask_token', 'decoder.mask_token'),
                          ('norm_weight', 'decoder.norm_out.weight'),
                          ('norm_bias', 'decoder.norm_out.bias'),
                          ('mean', 'latents_mean'), ('std', 'latents_std')]:
            self.register_buffer(name, store.tensor(key))
        self.register_buffer('post_weight', store.tensor('post_quant_conv.weight').reshape(24, 24))
        self.register_buffer('post_bias', store.tensor('post_quant_conv.bias'))

    @torch.no_grad()
    def decode_clip(self, normalized_latents):
        """Decode a 7-token temporal clip to 22 RGB frames, one sample."""
        b, c, t, h, w = normalized_latents.shape
        if b < 1 or (c, t) != (24, 7):
            raise ValueError('Decoder clips must be [B,24,7,H,W]')
        device = normalized_latents.device
        z = normalized_latents.float() * self.std.to(device).view(1, 24, 1, 1, 1)
        z = z + self.mean.to(device).view(1, 24, 1, 1, 1)
        cells = z.permute(0, 2, 3, 4, 1).reshape(b, -1, 24)
        cells = F.linear(cells, self.post_weight.to(device), self.post_bias.to(device))
        x = self.x_embed(cells)
        count = x.shape[1]
        # Both maintained reference runtimes append a zero suffix token; the
        # checkpoint's legacy mask_token is not consumed in decoding.
        x = torch.cat((x, self.register_tokens.to(device)[0].unsqueeze(0).expand(b,-1,-1), torch.zeros_like(x[:,:1])), dim=1)
        axes = [(2 * (torch.arange(n).float() + .5) / n - 1) for n in (t, h, w)]
        positions = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
        positions = torch.cat((positions, torch.zeros(5, 3))).to(device)
        inv = 1.0 / (100.0 ** (torch.arange(8, device=device).float() / 8))
        half = (2 * torch.pi * positions[..., None] * inv).flatten(1)
        rope = prepare_rope(torch.cat((half, half), dim=-1))
        for block in self.blocks:
            x = block(x, rope)
        x = F.layer_norm(x, (2048,), self.norm_weight.to(device), self.norm_bias.to(device), 1e-5)
        patches = self.out(x)[:,:count]
        # Each latent cell expands into [3,4,16,16] pixels.
        raw = patches.reshape(b, t, h, w, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7)
        raw = raw.reshape(b, 3, t * 4, h * 16, w * 16)
        # Causal clip convention: discard 3 prepadded frames per 5-token group.
        reconstructed = torch.cat((raw[:, :, 3:20], raw[:, :, 23:28]), dim=2)
        if not bool(torch.isfinite(reconstructed).all()):
            raise FloatingPointError('Video decoder produced NaN/Inf')
        image_std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1)
        image_mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1)
        return reconstructed * image_std + image_mean

    @torch.no_grad()
    def decode(self, latents, progress=None, tile_progress=None, output_device=None):
        total = latents.shape[2]
        if total < 7 or (total - 2) % 5:
            raise ValueError('Temporal latent length must be 5k+2 with k>=1')
        output = None
        target = latents.device if output_device is None else torch.device(output_device)
        blend = torch.arange(5, device=target).float().view(1,1,5,1,1) / 5
        for start in range(0, total - 2, 5):
            clip = self.decode_spatial(latents[:, :, start:start + 7], tile_progress).to(target)
            frame = start // 5 * 17
            if output is None:
                frames = (total - 2) // 5 * 17 + 5
                output = torch.empty((*clip.shape[:2],frames,*clip.shape[-2:]),device=target,dtype=clip.dtype)
                output[:,:,:22] = clip
            else:
                # Blend five-frame overlap in the raw decoder domain. The RGB
                # conversion is affine, but clipping must be delayed until final.
                output[:,:,frame:frame+5] = output[:,:,frame:frame+5]*(1-blend) + clip[:,:,:5]*blend
                output[:,:,frame+5:frame+22] = clip[:,:,5:]
            if progress:
                progress(start // 5 + 1, (total - 2) // 5)
        return output.clamp_(0, 1)

    def decode_spatial(self, latents, tile_progress=None):
        """Upstream-style overlapping local VAE tiles; blend before clipping."""
        if latents.shape[0] != 1:
            raise ValueError('Spatial stitching expects one video; batches are internal tiles')
        h, w = latents.shape[-2:]
        if getattr(self,'geometry','official') == 'lynn_release':
            ys,xs = light_vae_tiles(h*16,w*16)
        else:
            ys = split_tiles(h * 16, self.tile_size, self.tile_overlap)
            xs = split_tiles(w * 16, self.tile_size, self.tile_overlap)
        if len(ys) == len(xs) == 1:
            return self.decode_clip(latents)
        # Normalized separable overlap-add. Every tile contributes at corners;
        # sequential neighbour blending can lose a diagonal contribution.
        wy = tile_axis_weights(ys, h * 16, latents.device)
        wx = tile_axis_weights(xs, w * 16, latents.device)
        canvas = None
        plans = [(yi,xi,y,yh,x,xw) for yi,(y,yh) in enumerate(ys) for xi,(x,xw) in enumerate(xs)]
        batch_size = getattr(self, 'tile_batch_size', 1)
        for start in range(0, len(plans), batch_size):
            group = plans[start:start+batch_size]
            inputs = torch.cat([latents[..., y//16:(y+yh)//16, x//16:(x+xw)//16] for _,_,y,yh,x,xw in group], dim=0)
            decoded = self.decode_clip(inputs)
            for index, (yi,xi,y,yh,x,xw) in enumerate(group):
                tile = decoded[index:index+1]
                if canvas is None:
                    canvas = torch.zeros((*tile.shape[:-2], h*16, w*16),
                                         dtype=torch.float32, device=tile.device)
                weight = wy[yi][:, None] * wx[xi][None, :]
                canvas[..., y:y+yh, x:x+xw] += tile.float() * weight
                if tile_progress:
                    tile_progress(yi*len(xs)+xi+1, len(ys)*len(xs))
        return canvas


def light_vae_tiles(height,width):
    """Distilled LynnReal release geometry: short axis 272/0, long axis 208/16.

    The student was trained for this geometry; do not use arbitrary square
    tiles merely because the checkpoint accepts a different spatial shape.
    Reference: ComfyUI-LynnReal/light_vae.py's release geometry/split equations.
    """
    import math
    def axis(length,size,overlap):
        if length<=size+overlap:return [(0,length)]
        count=math.ceil(length/size)
        required=math.ceil((length+overlap*(count-1))/count/16)*16
        if required>size+overlap:raise ValueError('Light VAE tile expansion exceeds trained geometry')
        size=max(size,required)
        while size*count-overlap*(count-1)<length:count+=1
        overlaps=[overlap]*(count-1)
        excess=size*count-sum(overlaps)-length
        for i in range(excess//16):overlaps[i%(count-1)]+=16
        starts=[0]
        for value in overlaps:starts.append(starts[-1]+size-value)
        return [(start,size) for start in starts]
    if height<width:return axis(height,272,0),axis(width,208,16)
    if height>width:return axis(height,208,16),axis(width,272,0)
    return axis(height,272,16),axis(width,272,16)


def tile_axis_weights(tiles, length, device):
    """Partition of unity, including triple overlaps (ComfyUI PR #16422)."""
    raw = []
    denominator = torch.zeros(length, dtype=torch.float32, device=device)
    for i, (start, size) in enumerate(tiles):
        position = torch.arange(size, dtype=torch.float32, device=device)
        weight = torch.ones_like(position)
        if i:
            overlap = sum(tiles[i-1]) - start
            if overlap > 0:
                weight *= (position / overlap).clamp(max=1)
        if i+1 < len(tiles):
            overlap = start + size - tiles[i+1][0]
            if overlap > 0:
                weight *= ((size-position) / overlap).clamp(max=1)
        denominator[start:start+size] += weight
        raw.append(weight)
    if not bool((denominator > 0).all()):
        raise ValueError('VAE tile plan has uncovered pixels')
    return [weight/denominator[start:start+size]
            for (start,size),weight in zip(tiles,raw)]


def split_tiles(length, tile_size, overlap):
    if length <= tile_size:
        return [(0, length)]
    import math
    count = math.ceil((length-overlap)/(tile_size-overlap))
    overlaps = [overlap]*(count-1)
    remaining = count*tile_size-sum(overlaps)-length
    for i in range(remaining//16):
        overlaps[i % (count-1)] += 16
    starts = [0]
    for value in overlaps:
        starts.append(starts[-1]+tile_size-value)
    return [(start, tile_size) for start in starts]


def blend_tiles(previous, current, overlap, dim):
    if not overlap:
        return current
    shape = [1]*current.ndim
    shape[dim] = overlap
    weight = torch.arange(overlap, device=current.device).float().reshape(shape)/overlap
    mixed = previous.narrow(dim, previous.shape[dim]-overlap, overlap)*(1-weight) + current.narrow(dim,0,overlap)*weight
    return torch.cat((mixed, current.narrow(dim,overlap,current.shape[dim]-overlap)),dim=dim)
