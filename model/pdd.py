"""Alibaba PAI 8-step PDD adapter for the local FL2VA/Ref2VA pruned backbones.

Diffusers Q/K/V and value/gate layouts are explicitly converted. Dense AdaLN
LoRA is projected onto a verified affine basis of the pruned time curve.
Head fusion follows the modality-specific fine-grid sigma spans.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .safetensors_store import SafeTensorStore


def _scaled_linear(x, weight, scale):
    """FP16 GEMM with FP32 power-of-two range restoration."""
    x = x.float()
    xs = torch.pow(2.0, torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp(min=1))))
    return F.linear((x / xs).half(), weight).float() * xs * scale


def prepare_fast_lora(backbone):
    """Opt-in scaled FP16 low-rank GEMMs; keep residual accumulation FP32.

    Call on CPU before moving the model to its inference device. The original
    FP32 factors are replaced, avoiding duplicate resident adapter weights.
    """
    count = 0
    for module in backbone.modules():
        if isinstance(module, LowRankLinear):
            names = ('a', 'b')
        elif isinstance(module, QKVLowRankLinear):
            names = tuple(f'{letter}{i}' for i in range(3) for letter in ('a', 'b'))
        else:
            continue
        if getattr(module, 'fast_lora', False):
            continue
        for name in names:
            w = getattr(module, name).float()
            scale = torch.pow(2.0, torch.ceil(torch.log2(w.abs().amax(-1).clamp(min=1))))
            setattr(module, name, (w / scale[:, None]).half())
            module.register_buffer(name + '_scale', scale)
        module.fast_lora = True
        count += 1
    return count


def _adapter_branch(module, x, a, b):
    if getattr(module, 'fast_lora', False):
        hidden = _scaled_linear(x, getattr(module, a), getattr(module, a + '_scale'))
        return _scaled_linear(hidden, getattr(module, b), getattr(module, b + '_scale'))
    return F.linear(F.linear(x.float(), getattr(module, a).float()), getattr(module, b).float())


def prepare_half_lora_storage(backbone):
    """Store factors in FP16 while retaining the existing FP32 GEMMs."""
    count = 0
    for module in backbone.modules():
        if isinstance(module, LowRankLinear):
            names = ('a','b')
        elif isinstance(module, QKVLowRankLinear):
            names = tuple(f'{letter}{i}' for i in range(3) for letter in ('a','b'))
        else:
            continue
        if getattr(module, 'fast_lora', False):
            raise ValueError('Half storage and scaled FP16 compute are separate alternatives')
        for name in names:
            value = getattr(module,name)
            if not bool(torch.isfinite(value).all()) or float(value.abs().max()) > 65504:
                raise ValueError('Adapter factor exceeds FP16 storage range')
            setattr(module,name,value.half())
        count += 1
    return count


def install_taomate(backbone, checkpoint, strength=1.0):
    """Official TaoLive step-3000 rank-128/alpha-128 adapter.

    Upstream uses contiguous Q/K/V and gate/up packing, matching this runtime.
    Unlike PDD this adapter does not replace heads or modify the AdaLN curve.
    Its streaming KV-cache runtime is separate; this installs only the weights.
    """
    store = SafeTensorStore(checkpoint)
    metadata = store.header.get('__metadata__', {})
    if metadata.get('optimizer_step') != '3000':
        raise ValueError('Expected official TaoMate step-3000 checkpoint')
    blocks = [(b, f'blocks.{i}') for i,b in enumerate(backbone.blocks)]
    blocks += [(b, f'token_refiner.blocks.{i}') for i,b in enumerate(backbone.refiners)]
    consumed = set()
    for block, prefix in blocks:
        for owner, attr, suffix in ((block.attn,'qkv','attn.qkv_proj'),
                                   (block.attn,'proj','attn.out_proj'),
                                   (block.mlp,'up','mlp.fc1'),
                                   (block.mlp,'down','mlp.fc2')):
            base = getattr(owner, attr)
            keys = [prefix + '.' + suffix + '.lora_' + side for side in ('a','b')]
            a,b = [store.tensor(k) for k in keys]
            if a.shape != (128, base.in_features) or b.shape != (base.out_features,128):
                raise ValueError(f'TaoMate shape mismatch at {prefix}.{suffix}')
            consumed.update(keys)
            setattr(owner, attr, LowRankLinear(base,a,b,strength))
    remaining = set(store.header) - consumed - {'__metadata__'}
    if remaining:
        raise ValueError(f'Unconsumed TaoMate tensors: {sorted(remaining)}')
    return len(consumed)//2


def sigma_grid(shift, steps=32):
    t = torch.linspace(1, 0, steps+1, dtype=torch.float64)
    return shift*t/(1+(shift-1)*t)


def fuse_head_bank(weight, bias, shift, block_size=4):
    fine = sigma_grid(shift, len(weight))
    weights, biases = [], []
    for start in range(0,len(weight),block_size):
        coefficients = fine[start:start+block_size]-fine[start+1:start+block_size+1]
        coefficients = coefficients/coefficients.sum()
        weights.append((weight[start:start+block_size].double()*coefficients[:,None,None]).sum(0).float())
        biases.append((bias[start:start+block_size].double()*coefficients[:,None]).sum(0).float())
    return torch.stack(weights),torch.stack(biases)


class LowRankLinear(nn.Module):
    def __init__(self, base, a, b, scale=1.0):
        super().__init__()
        self.base,self.scale = base,scale
        self.token_chunk_size = 0
        self.register_buffer('a',a.float())
        self.register_buffer('b',b.float())

    def forward(self,x):
        if self.token_chunk_size:
            output=self.base(x)
            if output.data_ptr()==x.data_ptr():output=output.clone()
            flat=x.reshape(-1,x.shape[-1]); target=output.reshape(-1,output.shape[-1])
            for start in range(0,len(flat),self.token_chunk_size):
                end=start+self.token_chunk_size
                target[start:end].add_(_adapter_branch(self,flat[start:end],'a','b'),alpha=self.scale)
            return output
        return self.base(x)+_adapter_branch(self,x,'a','b')*self.scale


class QKVLowRankLinear(nn.Module):
    def __init__(self,base,branches,scale=1.0):
        super().__init__()
        self.base,self.scale=base,scale
        self.token_chunk_size = 0
        for index,(a,b) in enumerate(branches):
            self.register_buffer(f'a{index}',a.float())
            self.register_buffer(f'b{index}',b.float())

    def forward(self,x):
        if self.token_chunk_size:
            output=self.base(x)
            if output.data_ptr()==x.data_ptr():output=output.clone()
            flat=x.reshape(-1,x.shape[-1]); target=output.reshape(-1,output.shape[-1])
            for start in range(0,len(flat),self.token_chunk_size):
                end=start+self.token_chunk_size
                delta=[_adapter_branch(self,flat[start:end],f'a{i}',f'b{i}') for i in range(3)]
                target[start:end].add_(torch.cat(delta,-1),alpha=self.scale)
            return output
        delta=[_adapter_branch(self,x,f'a{i}',f'b{i}') for i in range(3)]
        return self.base(x)+torch.cat(delta,-1)*self.scale


class AffineDeltaLinear(nn.Module):
    def __init__(self,base,weight,bias):
        super().__init__()
        self.base=base
        self.register_buffer('weight',weight.float())
        self.register_buffer('bias',bias.float())

    def forward(self,x):
        return self.base(x)+F.linear(x.float(),self.weight,self.bias)


class PDDHead(nn.Module):
    def __init__(self,weight,bias):
        super().__init__()
        self.register_buffer('weight',weight.float())
        self.register_buffer('bias',bias.float())
        self.step=0

    def forward(self,x):
        return F.linear(x.float(),self.weight[self.step],self.bias[self.step])


def install_pdd(backbone,checkpoint,basis_path):
    store,basis=SafeTensorStore(checkpoint),SafeTensorStore(basis_path)
    metadata=store.header.get('__metadata__',{})
    fine_steps=int(metadata.get('pdd_num_steps',0))
    block_size=int(metadata.get('pdd_block_size',0))
    if (fine_steps,block_size)!=(32,4):
        raise ValueError('Expected the official 32-grid, block-4 eight-step PDD adapter')
    trunk=basis.header.get('__metadata__',{}).get('trunk')
    if trunk not in ('fl2va','ref2va'):
        raise ValueError('Expected FL2VA or Ref2VA AdaLN basis')
    if trunk not in Path(checkpoint).name.lower():
        raise ValueError('PDD adapter and AdaLN basis use different model families')
    if not torch.equal(backbone.time_table.cpu(),basis.tensor('adaln_t_table')):
        raise ValueError('PDD AdaLN basis does not exactly match this backbone time table')
    c,v=basis.tensor('c').double(),basis.tensor('V').double()
    alpha=float(metadata['lora_alpha'])
    rank=int(metadata['lora_rank'])
    scale=alpha/rank
    consumed=set()
    def take(name):
        consumed.add(name)
        return store.tensor(name)
    def pair(prefix):
        a,b=take(prefix+'.lora_down'),take(prefix+'.lora_up')
        if a.shape[0]!=rank or b.shape[1]!=rank:
            raise ValueError(f'Unexpected LoRA rank at {prefix}')
        return a,b
    def regular(base,prefix,swap=False):
        a,b=pair(prefix)
        if swap:
            value,gate=b.chunk(2,0)
            b=torch.cat((gate,value),0)
        if (a.shape[1],b.shape[0])!=(base.in_features,base.out_features):
            raise ValueError(f'LoRA dimension mismatch at {prefix}')
        return LowRankLinear(base,a,b,scale)
    blocks=[(block,f'transformer_blocks.{i}',True) for i,block in enumerate(backbone.blocks)]
    blocks += [(block,f'token_refiner.refiner_blocks.{i}',False) for i,block in enumerate(backbone.refiners)]
    for block,prefix,has_adaln in blocks:
        branches=[pair(prefix+'.attn.to_'+name) for name in ('q','k','v')]
        expected=block.attn.qkv.out_features//3
        if any(a.shape[1]!=block.attn.qkv.in_features or b.shape[0]!=expected for a,b in branches):
            raise ValueError(f'QKV adapter dimensions do not match {prefix}')
        block.attn.qkv=QKVLowRankLinear(block.attn.qkv,branches,scale)
        block.attn.proj=regular(block.attn.proj,prefix+'.attn.to_out.0')
        block.mlp.up=regular(block.mlp.up,prefix+'.ff.net.0.proj',swap=True)
        block.mlp.down=regular(block.mlp.down,prefix+'.ff.net.2')
        if has_adaln:
            a,b=pair(prefix+'.adaln_proj.linear')
            if a.shape[1]!=len(c) or b.shape[0]!=block.adaln.out_features:
                raise ValueError(f'AdaLN adapter dimensions do not match {prefix}')
            # Includes the affine DC bias, which is mandatory for a pruned model.
            delta_w=(b.double()@(a.double()@v))*scale
            delta_b=(b.double()@(a.double()@c))*scale
            block.adaln=AffineDeltaLinear(block.adaln,delta_w.float(),delta_b.float())
    for name,prefix,shift in [('video_out','proj_out',12),('audio_out','audio_proj_out',3)]:
        weight,bias=take(prefix+'.weight'),take(prefix+'.bias')
        original=getattr(backbone,name)
        if weight.shape!=(fine_steps,original.out_features,original.in_features) or bias.shape!=weight.shape[:2]:
            raise ValueError(f'Unexpected PDD head bank at {prefix}')
        weight,bias=fuse_head_bank(weight,bias,shift,block_size)
        setattr(backbone,name,PDDHead(weight,bias))
    unexpected=set(store.header)-consumed-{'__metadata__'}
    if unexpected:
        raise ValueError(f'Unconsumed PDD tensors: {sorted(unexpected)}')
    knots=torch.arange(0,fine_steps+1,block_size)
    backbone.pdd_steps=fine_steps//block_size
    return sigma_grid(12)[knots].float(),sigma_grid(3)[knots].float()
