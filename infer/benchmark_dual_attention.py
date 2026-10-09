"""Time exact attention across one/two NPUs at real 1080P token counts."""
import json
import argparse
from pathlib import Path
import time
import torch
from model.runtime import initialize_npu
from model.attention import streaming_attention
from model.dual_attention import DualNPUAttention

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--large-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('/tmp/minimax-h3-run/dual-attention-benchmark.json'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    devices=[initialize_npu(0),initialize_npu(1)]
    initialize_npu(0)
    options=dict(query_chunk=512,key_chunk=1024)
    parallel=DualNPUAttention(devices,**options)
    results=[]
    try:
        cases=[(256,0),(14364,22),(55380,90)]
        if args.large_only:
            cases=[(55380,90)]
        for size,frames in cases:
            q,k,v=[torch.randn(1,56,size,128,device=devices[0])*.03 for _ in range(3)]
            torch.npu.synchronize(0)
            started=time.monotonic()
            reference=streaming_attention(q,k,v,**options)
            torch.npu.synchronize(0)
            serial=time.monotonic()-started
            started=time.monotonic()
            actual=parallel(q,k,v)
            torch.npu.synchronize(0)
            dual=time.monotonic()-started
            error=float((reference-actual).abs().max())
            torch.testing.assert_close(actual,reference,rtol=.003,atol=.0001)
            result={'tokens':size,'approximate_video_frames':frames,'single_seconds':serial,'dual_seconds':dual,'speedup':serial/dual,'max_abs_error':error,'memory':{str(d):{'allocated_bytes':torch.npu.memory_allocated(d),'peak_allocated_bytes':torch.npu.max_memory_allocated(d)} for d in devices}}
            results.append(result)
            print(json.dumps(result),flush=True)
            args.output.write_text(json.dumps({'completed':frames==90,'results':results},indent=2)+'\n')
            del q,k,v,reference,actual
            torch.npu.empty_cache()
    finally:
        parallel.close()
if __name__=='__main__':main()
