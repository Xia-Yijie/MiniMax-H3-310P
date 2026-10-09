"""Row-wise INT8 ConvRot safetensors used by community H3 checkpoints.

The regular normalized H4 Kronecker rotation follows the published
comfy-kitchen tensor/int8_utils.py convention, not the Sylvester H2 variant.
This backend uses dequantized FP16 GEMM; it does not claim native INT8 speed.
"""
import json
from types import SimpleNamespace
import numpy as np
import torch
from .weights import TensorStore
from .safetensors_store import SafeTensorStore


def regular_hadamard(x, group_size=256):
    size=group_size
    while size>1 and size%4==0:size//=4
    if size!=1 or group_size<4 or x.shape[-1]%group_size:
        raise ValueError('ConvRot requires power-of-four groups dividing input features')
    # Rotation is independent for each token. Bound its full-length FP32
    # intermediates when a long video follows a resident VAE decode.
    flat=x.reshape(-1,x.shape[-1])
    if flat.shape[0]>4096:
        result=torch.empty(flat.shape,device=x.device,dtype=torch.float32)
        for start in range(0,len(flat),4096):
            result[start:start+4096]=regular_hadamard(flat[start:start+4096],group_size)
        return result.reshape(x.shape)
    original=x.shape
    out=x.float().reshape(-1,group_size)
    stride=1
    while stride<group_size:
        grouped=out.reshape(-1,group_size//(4*stride),4,stride)
        # H4 = all-ones minus twice the anti-diagonal, normalized by two.
        out=((grouped.sum(-2,keepdim=True)-2*grouped.flip(-2))*.5).reshape(-1,group_size)
        stride*=4
    return out.reshape(original)


def int8_output_restore(raw, scales):
    """Power-of-two guard for the kernel's FP16 output before token scaling."""
    # Bound host temporaries too: a full FP32 expansion of a large INT8
    # matrix would compete with both cards' cold-loading page cache.
    row_l1=np.empty(len(raw),dtype=np.float32)
    for start in range(0,len(raw),256):
        row_l1[start:start+256]=np.abs(np.asarray(raw[start:start+256],dtype=np.float32)).sum(-1)
    bound=row_l1*127*scales
    return np.exp2(np.ceil(np.log2(np.maximum(bound/16384,1)))).astype(np.float32)


class CommunityTensorStore(TensorStore):
    def __init__(self,path):
        self.safe=SafeTensorStore(path)
        header=self.safe.header
        meta=header.get('__metadata__',{})
        quant=json.loads(meta.get('_quantization_metadata','{}'))
        self.quant=quant.get('layers',{})
        self.gguf=SimpleNamespace(tensors={k:SimpleNamespace(shape=tuple(v['shape']),quant_type=8 if v['dtype']=='I8' else None)
                                         for k,v in header.items() if k!='__metadata__'})
        self._device_cache={};self._device_cache_bytes=0;self._device_cache_budget=0
        self._prepared_cache={};self._prepared_cache_bytes=0;self._prepared_cache_budget=0
        self._scales={}
        self._scale_device={}
        self._native_cache={}
        self.native_cache_bytes=0
        for name,info in header.items():
            if name=='__metadata__' or info['dtype']!='I8':continue
            prefix=name.removesuffix('.weight');conf=self.quant.get(prefix,{})
            if not name.endswith('.weight') or len(info['shape'])!=2 or conf.get('format')!='int8_tensorwise':
                raise ValueError(f'Unsupported community quantization at {name}')
            if conf.get('convrot') and conf.get('convrot_groupsize',256)!=256:
                raise ValueError('Only audited ConvRot group size 256 is supported')
            scale=self.safe.tensor(prefix+'.weight_scale').numpy()
            if scale.shape not in ((info['shape'][0],1),(info['shape'][0],)):
                raise ValueError(f'Expected row-wise INT8 scales at {name}')
            self._scales[name]=scale.reshape(-1,1)

    def native_quant_weight(self,name,device):
        if name not in self._scales or device.type!='npu':return None
        key=(name,str(device))
        if key not in self._native_cache:
            import torch_npu
            raw=self._numpy(name)
            scales=self._scales[name].reshape(-1)
            # Bound the dequantized INT8 accumulator before its FP16 output.
            # The token scale is applied afterwards, so even normalized inputs
            # can overflow here when a layer has large trained weight scales.
            restore=int8_output_restore(raw,scales)
            weight=torch.from_numpy(raw.copy()).to(device)
            weight=torch_npu.npu_format_cast(weight.T,29)
            scale=torch.from_numpy((scales/restore).copy()).to(device)
            restore=torch.from_numpy(restore).to(device)
            self._native_cache[key]=(weight,scale,restore)
            self.native_cache_bytes+=weight.numel()*weight.element_size()+scale.numel()*8
        return self._native_cache[key]

    def input_transform(self,name,x):
        conf=self.quant.get(name.removesuffix('.weight'),{})
        return regular_hadamard(x,conf.get('convrot_groupsize',256)) if conf.get('convrot') else x

    def _numpy(self,name):
        info=self.safe.header[name]
        if info['dtype'] in ('I8','U8'):
            start,stop=info['data_offsets']
            dtype=np.int8 if info['dtype']=='I8' else np.uint8
            if np.prod(info['shape'])!=stop-start or self.safe.data_offset+stop>self.safe.path.stat().st_size:
                raise ValueError('Invalid community tensor extent')
            if name not in self.safe._maps:
                self.safe._maps[name]=np.memmap(self.safe.path,mode='r',dtype=dtype,offset=self.safe.data_offset+start,shape=tuple(info['shape']))
            return self.safe._maps[name]
        return self.safe.numpy(name)

    def rows(self,name,start=0,stop=None):
        info=self.safe.header[name];stop=info['shape'][0] if stop is None else stop
        if not 0<=start<=stop<=info['shape'][0]:raise ValueError('Invalid tensor row range')
        data=self._numpy(name)[start:stop]
        if info['dtype']=='BF16':return (data.astype(np.uint32)<<16).view(np.float32)
        data=np.asarray(data,dtype=np.float32)
        if info['dtype']=='I8':data=data*self._scales[name][start:stop]
        return data

    def tensor(self,name,*,max_bytes=128*1024**2):
        if np.prod(self.info(name).shape)*4>max_bytes:raise MemoryError('Use rows for large tensors')
        if name in self._scales:raise ValueError('Use linear interface for rotated quantized matrices')
        return self.rows(name).copy()

    def device_rows(self,name,start,stop,device):
        if device.type!='npu' or self._device_cache_budget<=0:return None
        key=(name,str(device));info=self.safe.header[name]
        if key not in self._device_cache:
            raw=self._numpy(name)
            # Decode BF16 on the host, preserving range for AdaLN's FP32 path.
            # FP16 matrix paths apply their own scaling before the final cast.
            if info['dtype']=='BF16':raw=self.rows(name)
            if self._device_cache_bytes+raw.nbytes>self._device_cache_budget:return None
            self._device_cache[key]=torch.from_numpy(raw.copy()).to(device)
            self._device_cache_bytes+=raw.nbytes
        result=self._device_cache[key][start:stop].float()
        if info['dtype']=='I8':
            if key not in self._scale_device:
                self._scale_device[key]=torch.from_numpy(self._scales[name].copy()).to(device)
            result=result*self._scale_device[key][start:stop]
        return result
