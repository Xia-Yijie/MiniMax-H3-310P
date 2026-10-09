"""H3 causal VAE image path. T=1 kernels exactly fold to their last temporal plane.

Architecture follows DiffSynth-Studio (Apache-2.0). No Conv3D on 310P.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .safetensors_store import SafeTensorStore
from .video_decoder import split_tiles, blend_tiles


def causal_image_kernel(weight):
    if weight.ndim != 5:
        raise ValueError('Expected Conv3D kernel')
    return weight[:, :, -1].contiguous()


class ImageEncoder(nn.Module):
    def __init__(self, checkpoint, tile_size=256, tile_overlap=64):
        super().__init__()
        self.store = SafeTensorStore(checkpoint)
        self.tile_size, self.tile_overlap = tile_size, tile_overlap
        self.register_buffer('mean', self.store.tensor('latents_mean'))
        self.register_buffer('std', self.store.tensor('latents_std'))

    def conv(self, x, name, padding=0, stride=1):
        w = causal_image_kernel(self.store.tensor(name + '.weight', x.device))
        b = self.store.tensor(name + '.bias', x.device)
        if padding:
            x = F.pad(x, (padding,)*4, mode='reflect')
        return F.conv2d(x, w, b, stride=stride)

    def norm(self, x, name):
        return F.group_norm(x.float(), 32, self.store.tensor(name+'.weight', x.device),
                            self.store.tensor(name+'.bias', x.device), eps=1e-6)

    def moments(self, x):
        x = self.conv(x, 'encoder.conv_in', 1)
        for level in range(6):
            for index in range(2):
                p = f'encoder.down.{level}.block.{index}'
                h = self.conv(F.silu(self.norm(x, p+'.norm1')), p+'.conv1', 1)
                h = self.conv(F.silu(self.norm(h, p+'.norm2')), p+'.conv2', 1)
                if p+'.nin_shortcut.weight' in self.store.header:
                    x = self.conv(x, p+'.nin_shortcut')
                x = x + h
            p = f'encoder.down.{level}.downsample.conv'
            if p+'.weight' in self.store.header:
                x = self.conv(F.pad(x, (0,1,0,1), mode='reflect'), p, stride=2)
        x = self.conv(F.silu(self.norm(x, 'encoder.norm_out')), 'encoder.conv_out', 1)
        return self.conv(x, 'quant_conv')

    @torch.no_grad()
    def encode(self, image, progress=None):
        if image.ndim != 4 or image.shape[:2] != (1,3) or any(n % 32 for n in image.shape[-2:]):
            raise ValueError('Expected RGB [1,3,H,W], H/W multiples of 32, pixels 0..1')
        mean = image.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std = image.new_tensor([.229,.224,.225]).view(1,3,1,1)
        x = (image.float()-mean)/std
        h,w = x.shape[-2:]
        ys = split_tiles(h, self.tile_size, self.tile_overlap)
        xs = split_tiles(w, self.tile_size, self.tile_overlap)
        # Keep unblended moments: upstream blending reads original neighbours,
        # not already blended tiles. In-place mutation changes corner weights.
        rows = []
        for i,(y,hh) in enumerate(ys):
            row=[]
            for j,(xx,ww) in enumerate(xs):
                tile = self.moments(x[..., y:y+hh, xx:xx+ww])
                row.append(tile)
                if progress: progress(i*len(xs)+j+1,len(ys)*len(xs))
            rows.append(row)
        joined=[]
        for i,row in enumerate(rows):
            tiles=[]
            for j,tile in enumerate(row):
                if i:
                    ov=(ys[i-1][0]+ys[i-1][1]-ys[i][0])//16
                    tile=blend_tiles(rows[i-1][j],tile,ov,-2)
                if j:
                    ov=(xs[j-1][0]+xs[j-1][1]-xs[j][0])//16
                    tile=blend_tiles(row[j-1],tile,ov,-1)
                if i+1<len(ys): tile=tile[...,:((ys[i+1][0]-ys[i][0])//16),:]
                if j+1<len(xs): tile=tile[...,:,:((xs[j+1][0]-xs[j][0])//16)]
                tiles.append(tile)
            joined.append(torch.cat(tiles,-1))
        z=torch.cat(joined,-2)[:,:24,None]
        z=(z-self.mean.to(x.device).view(1,24,1,1,1))/self.std.to(x.device).view(1,24,1,1,1)
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Image VAE produced NaN/Inf')
        return z
