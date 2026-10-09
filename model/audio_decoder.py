"""H3 BigVGAN decode-only path using folded convolution weights.

The alias-free filtering and architecture follow the Apache-2.0 DiffSynth
reference. Conv1d is expressed as Conv2d for the Ascend execution backend;
shared per-channel filters flatten channels into batch, avoiding large groups.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .safetensors_store import SafeTensorStore


class AudioDecoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.store = SafeTensorStore(checkpoint)
        self._resident = {}

    def weight(self, name, device):
        key = (name, str(device))
        if key not in self._resident:
            self._resident[key] = self.store.tensor(name, device)
        return self._resident[key]

    def conv(self, x, prefix, *, stride=1, padding=0, dilation=1, transpose=False):
        weight = self.weight(prefix + '.weight', x.device).unsqueeze(2)
        bias_key = prefix + '.bias'
        bias = self.weight(bias_key, x.device) if bias_key in self.store.header else None
        x = x.unsqueeze(2)
        if transpose:
            out = F.conv_transpose2d(x, weight, bias, stride=(1, stride), padding=(0, padding))
        else:
            out = F.conv2d(x, weight, bias, stride=(1, stride), padding=(0, padding), dilation=(1, dilation))
        return out.squeeze(2)

    def activation(self, x, prefix):
        b, c, length = x.shape
        up_filter = self.weight(prefix + '.upsample.filter', x.device)
        padded = F.pad(x, (5, 5), mode='replicate').reshape(b * c, 1, -1).unsqueeze(2)
        up = 2 * F.conv_transpose2d(padded, up_filter.unsqueeze(2), stride=(1, 2)).squeeze(2)
        # ratio=2, kernel=12 => left crop=15, right crop=15.
        up = up[..., 15:-15].reshape(b, c, -1)
        alpha = self.weight(prefix + '.act.alpha', x.device).exp().view(1, c, 1)
        beta = self.weight(prefix + '.act.beta', x.device).exp().view(1, c, 1)
        snake = up + torch.sin(alpha * up).square() / (beta + 1e-9)
        down_filter = self.weight(prefix + '.downsample.lowpass.filter', x.device)
        snake = F.pad(snake, (5, 6), mode='replicate').reshape(b * c, 1, -1).unsqueeze(2)
        out = F.conv2d(snake, down_filter.unsqueeze(2), stride=(1, 2)).squeeze(2)
        result = out.reshape(b, c, -1)
        if result.shape[-1] != length:
            raise RuntimeError('Alias-free activation changed temporal length')
        return result

    def residual_block(self, x, prefix, kernel_size):
        for index, dilation in enumerate((1, 3, 5)):
            h = self.activation(x, prefix + f'.activations.{2 * index}')
            h = self.conv(h, prefix + f'.convs1.{index}',
                          dilation=dilation, padding=(kernel_size * dilation - dilation) // 2)
            h = self.activation(h, prefix + f'.activations.{2 * index + 1}')
            h = self.conv(h, prefix + f'.convs2.{index}', padding=(kernel_size - 1) // 2)
            x = x + h
        return x

    @torch.no_grad()
    def decode(self, latents, progress=None):
        """[stereo,32,T] -> [stereo,800*T], float waveform at 32 kHz."""
        if latents.ndim != 3 or latents.shape[:2] != (2, 32):
            raise ValueError('Expected stereo latents [2,32,T]')
        device = latents.device
        mean = self.weight('latents_mean', device).view(1, 32, 1)
        std = self.weight('latents_std', device).view(1, 32, 1)
        x = latents.float() * std + mean
        x = self.conv(x, 'dec_in_proj')
        x = self.conv(x, 'decoder.conv_pre', padding=3)
        for stage, (stride, kernel) in enumerate(zip((5, 5, 2, 2, 2, 2, 2), (9, 9, 4, 4, 4, 4, 4))):
            x = self.conv(x, f'decoder.ups.{stage}.0', stride=stride,
                          padding=(kernel - stride) // 2, transpose=True)
            outputs = [self.residual_block(x, f'decoder.resblocks.{stage * 3 + index}', k)
                       for index, k in enumerate((3, 7, 11))]
            x = sum(outputs) / 3
            del outputs
            if device.type == 'npu':
                # Bound queued activation/workspace lifetimes for long audio
                # while the video backbone and VAE remain resident on 310P.
                torch.npu.synchronize(device)
                torch.npu.empty_cache()
            if progress:
                progress(stage + 1, 7)
        x = self.activation(x, 'decoder.activation_post')
        x = self.conv(x, 'decoder.conv_post', padding=3).squeeze(1)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError('Audio decoder produced NaN/Inf')
        return x.clamp(-1, 1)
