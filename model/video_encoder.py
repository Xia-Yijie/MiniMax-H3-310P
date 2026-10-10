"""H3 causal video VAE encoder via Conv2D, preserving temporal kernels on 310P.

The Apache-2.0 DiffSynth encoder splits time into independent 17-frame clips,
pads the last clip by repetition, and removes three trailing latent tokens.
"""
import torch
from torch.nn import functional as F
from .image_encoder import ImageEncoder


def temporal_conv2d(x, weight, bias, spatial_padding=0, spatial_stride=1,
                    temporal_padding=0, temporal_stride=1, batch_size=2):
    """Evaluate causal Conv3D by flattening each temporal window into channels."""
    if x.ndim != 5 or x.shape[0] != 1 or weight.ndim != 5:
        raise ValueError('Expected single-batch video and 3D kernel')
    _, cin, frames, height, width = x.shape
    cout, _, kt, kh, kw = weight.shape
    if temporal_padding:
        x = torch.cat((x.new_zeros(1, cin, temporal_padding, height, width), x), 2)
    starts = list(range(0, x.shape[2] - kt + 1, temporal_stride))
    w = weight.permute(0, 2, 1, 3, 4).reshape(cout, kt * cin, kh, kw).contiguous()
    chunks = []
    for offset in range(0, len(starts), batch_size):
        windows = [x[:, :, t:t+kt].permute(0, 2, 1, 3, 4).reshape(1, kt*cin, height, width)
                   for t in starts[offset:offset+batch_size]]
        patch = torch.cat(windows)
        if spatial_padding:
            patch = F.pad(patch, (spatial_padding,)*4, mode='reflect')
        chunks.append(F.conv2d(patch, w, bias, stride=spatial_stride))
    return torch.cat(chunks).permute(1, 0, 2, 3)[None]


class VideoEncoder(ImageEncoder):
    def conv(self, x, name, padding=0, stride=1, time_stride=1, time_only=False):
        w = self.store.tensor(name+'.weight', x.device)
        b = self.store.tensor(name+'.bias', x.device)
        return temporal_conv2d(x, w, b, spatial_padding=0 if time_only else padding,
                               spatial_stride=stride, temporal_padding=2*padding,
                               temporal_stride=time_stride)

    def norm(self, x, name):
        # Upstream group normalization isolates every temporal frame.
        frames = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[-2:])
        normalized = super().norm(frames, name)
        return normalized.reshape(1, x.shape[2], x.shape[1], *x.shape[-2:]).permute(0, 2, 1, 3, 4)

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
                space_stride = 2 if level < 4 else 1
                if space_stride == 2:
                    # Fold T into the batch for spatial reflection padding.
                    shape=x.shape
                    flat=x.permute(0,2,1,3,4).reshape(-1,shape[1],*shape[-2:])
                    flat=F.pad(flat,(0,1,0,1),mode='reflect')
                    x=flat.reshape(1,shape[2],shape[1],shape[3]+1,shape[4]+1).permute(0,2,1,3,4)
                x = self.conv(x, p, padding=1, stride=space_stride,
                              time_stride=2 if level in (1,2) else 1, time_only=True)
        x = self.conv(F.silu(self.norm(x, 'encoder.norm_out')), 'encoder.conv_out', 1)
        return self.conv(x, 'quant_conv')

    def tiled_moments(self, x):
        from .video_decoder import split_tiles, blend_tiles
        height, width = x.shape[-2:]
        ys = split_tiles(height, self.tile_size, self.tile_overlap)
        xs = split_tiles(width, self.tile_size, self.tile_overlap)
        rows = [[self.moments(x[..., y:y+h, xx:xx+w]) for xx,w in xs] for y,h in ys]
        joined=[]
        for i,row in enumerate(rows):
            parts=[]
            for j,tile in enumerate(row):
                if i:
                    tile=blend_tiles(rows[i-1][j],tile,(ys[i-1][0]+ys[i-1][1]-ys[i][0])//16,-2)
                if j:
                    tile=blend_tiles(row[j-1],tile,(xs[j-1][0]+xs[j-1][1]-xs[j][0])//16,-1)
                if i+1<len(ys):tile=tile[...,:((ys[i+1][0]-ys[i][0])//16),:]
                if j+1<len(xs):tile=tile[...,:,:((xs[j+1][0]-xs[j][0])//16)]
                parts.append(tile)
            joined.append(torch.cat(parts,-1))
        return torch.cat(joined,-2)

    @torch.no_grad()
    def encode(self, video, progress=None):
        if video.ndim != 5 or video.shape[:2] != (1,3) or any(n%32 for n in video.shape[-2:]):
            raise ValueError('Expected RGB [1,3,T,H,W], pixels 0..1, H/W multiples of 32')
        if video.shape[2] < 5 or (video.shape[2]-5)%17:
            raise ValueError('Reference video frame count must be 17k+5')
        mean=video.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
        std=video.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
        x=(video.float()-mean)/std
        pad=(-x.shape[2])%17
        if pad:x=torch.cat((x,x[:,:,-1:].repeat(1,1,pad,1,1)),2)
        chunks=[]
        for start in range(0,x.shape[2],17):
            chunks.append(self.tiled_moments(x[:,:,start:start+17]).cpu())
            if progress:progress(start//17+1,x.shape[2]//17)
        z=torch.cat(chunks,2)[:,:24,:-3].to(video.device)
        z=(z-self.mean.to(z.device).view(1,24,1,1,1))/self.std.to(z.device).view(1,24,1,1,1)
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Video VAE produced NaN/Inf')
        return z
