"""H3 audio reference encoder, using Conv2D for Ascend 310P.

Stereo channels are separate batches of the mono VAE. The posterior mean is
normalized and supplied to the joint transformer; Qwen receives labels only.
"""
import math
import torch
from torch.nn import functional as F
from .audio_decoder import AudioDecoder
from .attention import streaming_attention


class AudioEncoder(AudioDecoder):
    def accurate_cube(self, x, weight, operation, bias=None):
        # 310P Conv2D silently reduces FP32 operands to half precision. Snake
        # amplifies that error. Two half-representable components recover the
        # missing mantissa; scale residuals to avoid half underflow.
        if x.device.type != 'npu':return operation(x,weight,bias)
        xh=x.half().float();xl=((x-xh)*1024).half().float()
        wh=weight.half().float();wl=((weight-wh)*1024).half().float()
        y=operation(xh,wh,None)
        y=y+(operation(xl,wh,None)+operation(xh,wl,None))/1024
        y=y+operation(xl,wl,None)/(1024*1024)
        if bias is not None:y=y+bias.reshape(1,-1,*([1]*(y.ndim-2)))
        return y

    def conv(self,x,prefix,*,stride=1,padding=0,dilation=1,transpose=False):
        if transpose:raise ValueError('Audio encoder uses ordinary convolution')
        w=self.weight(prefix+'.weight',x.device).unsqueeze(2)
        b=self.weight(prefix+'.bias',x.device) if prefix+'.bias' in self.store.header else None
        def operation(a,w,b):return F.conv2d(a,w,b,stride=(1,stride),padding=(0,padding),dilation=(1,dilation))
        return self.accurate_cube(x.unsqueeze(2),w,operation,b).squeeze(2)

    def snake(self, x, name):
        alpha = self.weight(name+'.alpha', x.device)
        return x + torch.sin(alpha*x).square()/(alpha+1e-9)

    def norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.weight(name+'.weight',x.device),
                            self.weight(name+'.bias',x.device), 1e-5)

    def linear(self, x, name):
        bias=name+'.bias'
        w=self.weight(name+'.weight',x.device)
        b=self.weight(bias,x.device) if bias in self.store.header else None
        if x.device.type!='npu':return F.linear(x,w,b)
        # Move bias addition out of the low-precision cube operation.
        y=self.accurate_cube(x,w,lambda a,w,b:F.linear(a,w,b))
        return y+b if b is not None else y

    def projection(self, x):
        p='pre_block'
        q,k,v=self.linear(self.norm(x,p+'.norm1'),p+'.attn.qkv').chunk(3,-1)
        q=q+self.weight(p+'.attn.q_bias',x.device)
        k=k+self.weight(p+'.attn.zero_k_bias',x.device)
        v=v+self.weight(p+'.attn.v_bias',x.device)
        def heads(y):return y.reshape(y.shape[0],y.shape[1],8,-1).transpose(1,2)
        h=streaming_attention(heads(q),heads(k),heads(v),causal=True).mean(1)
        # Upstream averages heads, then pools the feature axis (not time).
        h=h.reshape(*h.shape[:-1],32,-1).mean(-1)
        x=self.linear(self.norm(x,p+'.norm3'),p+'.proj')+self.linear(h,p+'.attn.proj')
        h=self.norm(self.norm(x,p+'.norm2'),p+'.mlp.norm')
        h=F.gelu(self.linear(h,p+'.mlp.w0'),approximate='tanh')*self.linear(h,p+'.mlp.w1')
        return x+self.linear(h,p+'.mlp.w2')

    @torch.no_grad()
    def encode(self, waveform, progress=None):
        if waveform.ndim!=2 or waveform.shape[0]!=2 or waveform.shape[1]<1:
            raise ValueError('Expected stereo audio [2,L] at 32 kHz')
        if not bool(torch.isfinite(waveform).all()):raise ValueError('Nonfinite audio input')
        x=F.pad(waveform.float(),(0,(-waveform.shape[1])%800))[:,None]
        x=self.conv(x,'encoder.block.0',padding=3)
        for stage,stride in enumerate((2,4,4,5,5),1):
            p=f'encoder.block.{stage}.block'
            for unit,dilation in enumerate((1,3,9)):
                r=f'{p}.{unit}.block'
                h=self.conv(self.snake(x,r+'.0'),r+'.1',padding=3*dilation,dilation=dilation)
                h=self.conv(self.snake(h,r+'.2'),r+'.3')
                x=x+h
            x=self.conv(self.snake(x,p+'.3'),p+'.4',stride=stride,padding=math.ceil(stride/2))
            if progress:progress(stage,5)
        x=self.conv(self.snake(x,'encoder.block.6'),'encoder.block.7',padding=1)
        x=self.projection(x.transpose(1,2)).transpose(1,2)
        x=self.conv(x,'mean_proj')
        z=(x-self.weight('latents_mean',x.device).view(1,32,1))/self.weight('latents_std',x.device).view(1,32,1)
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Audio encoder produced NaN/Inf')
        return z
