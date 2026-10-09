"""Probe 310P PromptFlashAttention and compare full-sequence attention."""
import json, time
from pathlib import Path
import torch
from model.runtime import initialize_npu
from model.attention import streaming_attention
from model.dual_attention import DualNPUAttention

OUT=Path('/tmp/minimax-h3-run/attention-operators.json')
def save(report):
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

def flash(q,k,v,layout):
 import torch_npu
 heads=q.shape[1]
 if layout=='BSH':
  q,k,v=[t.transpose(1,2).reshape(t.shape[0],t.shape[2],-1).half().contiguous() for t in (q,k,v)]
 else:
  q,k,v=[t.half().contiguous() for t in (q,k,v)]
 result=torch_npu.npu_prompt_flash_attention(q,k,v,num_heads=heads,scale_value=128**-.5,input_layout=layout,pre_tokens=2147483647,next_tokens=2147483647)
 if layout=='BSH':
  result=result.reshape(result.shape[0],result.shape[1],heads,128).transpose(1,2)
 return result.float()

@torch.no_grad()
def main():
 torch.set_num_threads(4)
 devices=[initialize_npu(0),initialize_npu(1)];initialize_npu(0)
 report={'completed':False,'native_attempts':[],'results':[]};chosen=None
 q,k,v=[torch.randn(1,8,256,128,device=devices[0]) for _ in range(3)]
 ref=streaming_attention(q,k,v,query_chunk=128,key_chunk=256)
 for layout in ['BNSD','BSH']:
  try:
   start=time.monotonic();actual=flash(q,k,v,layout);torch.npu.synchronize(0)
   err=float((actual-ref).abs().max());rel=float((actual-ref).square().mean().sqrt()/ref.square().mean().sqrt())
   report['native_attempts'].append({'layout':layout,'seconds':time.monotonic()-start,'max_abs_error':err,'relative_rms':rel,'finite':bool(torch.isfinite(actual).all())})
   torch.testing.assert_close(actual,ref,rtol=.01,atol=.003)
   chosen=layout;break
  except Exception as e:
   report['native_attempts'].append({'layout':layout,'error':str(e)[:3500]});save(report)
 if chosen is None:
  report['completed']=True;report['native_supported_for_tested_inputs']=False;save(report);return
 report['native_supported_for_tested_inputs']=True;report['layout']=chosen
 del q,k,v,ref,actual
 parallel=DualNPUAttention(devices,query_chunk=512,key_chunk=1024)
 try:
  for size,frames in [(14364,22),(55380,90)]:
   q,k,v=[torch.randn(1,56,size,128,device=devices[0]) for _ in range(3)]
   torch.npu.synchronize(0);start=time.monotonic();ref=parallel(q,k,v);torch.npu.synchronize(0);baseline=time.monotonic()-start
   start=time.monotonic();actual=flash(q,k,v,chosen);torch.npu.synchronize(0);elapsed=time.monotonic()-start
   error=float((actual-ref).abs().max());relative=float((actual-ref).square().mean().sqrt()/ref.square().mean().sqrt())
   finite=bool(torch.isfinite(actual).all());passed=finite and relative<.01
   report['results'].append({'tokens':size,'approximate_frames':frames,'dual_streaming_seconds':baseline,'single_native_seconds':elapsed,'speedup_over_dual_streaming':baseline/elapsed,'max_abs_error':error,'relative_rms':relative,'numerical_screen_passed':passed})
   save(report)
   if not passed: break
   del q,k,v,ref,actual;torch.npu.empty_cache()
 finally: parallel.close()
 report['completed']=True;save(report)
if __name__=='__main__':main()
