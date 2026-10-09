"""Qwen3-VL vision tower from GGUF, with merge ordering and DeepStack.

Equations follow transformers Qwen3-VL (Apache-2.0); streamed dense kernels
avoid Conv3D and unsupported BF16 on Ascend 310P.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .layers import GGUFLinear
from .attention import streaming_attention
from .h3 import apply_rope


def merged_coordinates(h, w):
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    return torch.stack((yy,xx),-1).reshape(h//2,2,w//2,2,2).permute(0,2,1,3,4).reshape(-1,2)


class VisionNorm(nn.Module):
    def __init__(self,store,name):
        super().__init__()
        self.register_buffer('weight',torch.from_numpy(store.tensor(name+'.weight')))
        self.register_buffer('bias',torch.from_numpy(store.tensor(name+'.bias')))
    def forward(self,x):
        return F.layer_norm(x,(self.weight.numel(),),self.weight,self.bias,1e-6)


class VisionBlock(nn.Module):
    def __init__(self,store,index,row_chunk):
        super().__init__()
        p=f'visual.blocks.{index}'
        self.norm1=VisionNorm(store,p+'.norm1');self.norm2=VisionNorm(store,p+'.norm2')
        self.qkv=GGUFLinear(store,p+'.attn.qkv',row_chunk=row_chunk)
        self.out=GGUFLinear(store,p+'.attn.proj',row_chunk=row_chunk)
        self.up=GGUFLinear(store,p+'.mlp.linear_fc1',row_chunk=row_chunk)
        self.down=GGUFLinear(store,p+'.mlp.linear_fc2',row_chunk=row_chunk)
    def forward(self,x,rope):
        q,k,v=self.qkv(self.norm1(x)).reshape(-1,3,16,72).unbind(1)
        q,k=apply_rope(q,rope),apply_rope(k,rope)
        h=streaming_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None])
        x=x+self.out(h[0].transpose(0,1).reshape(len(x),-1))
        return x+self.down(F.gelu(self.up(self.norm2(x)),approximate='tanh'))


class VisionMerger(nn.Module):
    def __init__(self,store,name,row_chunk,post=False):
        super().__init__();self.post=post
        self.norm=VisionNorm(store,name+'.norm')
        self.up=GGUFLinear(store,name+'.linear_fc1',row_chunk=row_chunk)
        self.down=GGUFLinear(store,name+'.linear_fc2',row_chunk=row_chunk)
    def forward(self,x):
        x=self.norm(x.reshape(-1,4608) if self.post else x).reshape(-1,4608)
        return self.down(F.gelu(self.up(x)))


class QwenVisionEncoder(nn.Module):
    def __init__(self,store,row_chunk=512):
        super().__init__();self.store=store
        # GGUF preserves flattened input-channel/output-channel axes here.
        self.register_buffer('patch_weight',torch.from_numpy(store.tensor('visual.patch_embed.proj.weight')).reshape(1152,1536))
        self.register_buffer('patch_bias',torch.from_numpy(store.tensor('visual.patch_embed.proj.bias')))
        self.register_buffer('pos_embed',torch.from_numpy(store.tensor('visual.pos_embed.weight')))
        self.blocks=nn.ModuleList(VisionBlock(store,i,row_chunk) for i in range(27))
        self.merger=VisionMerger(store,'visual.merger',row_chunk)
        self.deep=nn.ModuleList(VisionMerger(store,f'visual.deepstack_merger_list.{i}',row_chunk,True) for i in range(3))
    def positions(self,h,w,device):
        coords=merged_coordinates(h,w).to(device)
        # align_corners=True bilinear interpolation is equivalent to the HF four
        # embedding-table lookups at linspace(0,47,h/w).
        pos=F.interpolate(self.pos_embed.reshape(48,48,1152).permute(2,0,1)[None],
                          size=(h,w),mode='bilinear',align_corners=True)[0].permute(1,2,0)
        pos=pos.reshape(h//2,2,w//2,2,1152).permute(0,2,1,3,4).reshape(-1,1152)
        inv=1/(10000**(torch.arange(0,36,2,device=device).float()/36))
        half=(coords.float()[...,None]*inv).flatten(1)
        return pos,torch.cat((half,half),-1)
    @torch.no_grad()
    def encode_image(self,patches,grid,progress=None):
        t,h,w=map(int,grid)
        if t!=1 or h%2 or w%2:raise ValueError('Image grid must be [1,evenH,evenW]')
        pos,rope=self.positions(h,w,patches.device)
        x=F.linear(patches.float(),self.patch_weight,self.patch_bias)+pos
        deep=[]
        for i,block in enumerate(self.blocks):
            x=block(x,rope)
            if i in (8,16,24):deep.append(self.deep[(8,16,24).index(i)](x))
            if progress:progress(i+1,27,x)
        return self.merger(x),deep
