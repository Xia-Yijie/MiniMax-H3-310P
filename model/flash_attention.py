"""Explicit 310P PromptFlashAttention backend (FP16 compute, FP32 output)."""
import os
import json
import torch


def scaled_half_values(v, head_chunk=8):
    # Per-head reductions are independent. Bound temporary FP32 copies for
    # long clips, while keeping exactly the same scale and FP16 values.
    maxima=[]
    for start in range(0,v.shape[1],head_chunk):
        maxima.append(v[:,start:start+head_chunk].abs().amax(dim=(-2,-1),keepdim=True))
    value_max=torch.cat(maxima,dim=1)
    scale=torch.pow(2.0,torch.ceil(torch.log2(value_max.clamp(min=1))))
    half=torch.empty_like(v,dtype=torch.float16,memory_format=torch.contiguous_format)
    for start in range(0,v.shape[1],head_chunk):
        half[:,start:start+head_chunk]=(v[:,start:start+head_chunk]/scale[:,start:start+head_chunk]).half()
    return half,scale,value_max


@torch.no_grad()
def prompt_flash_attention(q, k, v, *, scale=None, causal=False,
                           query_chunk=512, key_chunk=1024, head_chunk=32):
    if causal:
        raise NotImplementedError('This backend is validated only for noncausal video attention')
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[:2] != k.shape[:2]
            or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]):
        raise ValueError('Incompatible attention shapes')
    if q.device.type != 'npu' or len({q.device, k.device, v.device}) != 1:
        raise ValueError('PromptFlashAttention requires tensors on the same NPU')
    if max(q.shape[-2], k.shape[-2]) < 128:
        # Preserve the FP32 text-refiner path; fusion overhead dominates here.
        from .attention import streaming_attention
        return streaming_attention(q, k, v, scale=scale, query_chunk=query_chunk,
                                   key_chunk=key_chunk, head_chunk=head_chunk)
    import torch_npu
    scale = q.shape[-1] ** -.5 if scale is None else scale
    # 310P can overflow the unscaled FP16 QK dot product before applying
    # scale_value. Real H3 late-layer inputs (Q~210, K~380) reproduce NaNs
    # in heads11/47 despite finite inputs. Exact power-of-two Q/K scaling
    # keeps that intermediate in range; compensate in the softmax scale.
    qk_downscale = 16.0
    half_queries = (q / qk_downscale).to(dtype=torch.float16, memory_format=torch.contiguous_format)
    half_keys = (k / qk_downscale).to(dtype=torch.float16, memory_format=torch.contiguous_format)
    # Attention is linear in V. A per-head power-of-two scale prevents half
    # input/output overflow without changing the exact softmax(QK) operation.
    half_values,value_scale,value_max=scaled_half_values(v)
    if os.environ.get('H3_FLASH_DIAGNOSTICS') == '1':
        print(json.dumps({'stage':'flash_input_ranges','device':str(q.device),
                          'q_max':float(q.abs().amax()),'k_max':float(k.abs().amax()),
                          'v_max':float(value_max.amax()),'v_scale_max':float(value_scale.amax())}),flush=True)
    with torch.npu.device(q.device):
        result = torch_npu.npu_prompt_flash_attention(
            half_queries, half_keys, half_values,
            num_heads=q.shape[1], scale_value=scale * qk_downscale**2, input_layout='BNSD',
            pre_tokens=2147483647, next_tokens=2147483647)
    if os.environ.get('H3_FLASH_DIAGNOSTICS') == '1':
        print(json.dumps({'stage':'flash_output_check','device':str(result.device),
                          'finite':bool(torch.isfinite(result).all())}),flush=True)
    return result.float().mul_(value_scale)
