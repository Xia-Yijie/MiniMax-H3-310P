"""Compact Real-ESRGAN SRVGG adaptation for NPU video postprocessing.

Architecture: xinntao/Real-ESRGAN, BSD-3-Clause; see licenses/Real-ESRGAN.txt.
This is pixel super-resolution, not high-resolution H3 diffusion sampling.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CompactPixelUpscaler(nn.Module):
    def __init__(self, num_conv=32, features=64, scale=4):
        super().__init__()
        self.scale = scale
        self.body = nn.ModuleList([nn.Conv2d(3, features, 3, padding=1), nn.PReLU(features)])
        for _ in range(num_conv):
            self.body.extend([nn.Conv2d(features, features, 3, padding=1), nn.PReLU(features)])
        self.body.append(nn.Conv2d(features, 3*scale*scale, 3, padding=1))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return F.pixel_shuffle(out, self.scale) + F.interpolate(x, scale_factor=self.scale, mode='nearest')

    @classmethod
    def load(cls, path, device):
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state.get('params_ema', state.get('params', state))
        model = cls()
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float16)
