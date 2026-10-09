import inspect,json,time
from pathlib import Path
import numpy as np
import torch
from model.runtime import initialize_npu
from model.device_quantization import decode_blocks
from model.gguf import GGUFFile
@torch.no_grad()
def main():
 torch.set_num_threads(4);device=initialize_npu(0)
 fixture=np.load(Path(__file__).resolve().parents[1]/'tests/fixtures/ggml_decode.npz')
 source=inspect.getsource(decode_blocks).replace('fields = packed[:, 4:16].to(torch.int32).reshape','fields = packed[:, 4:16].reshape').replace('quant = packed[:, 16:].to(torch.int32).reshape','quant = packed[:, 16:].reshape')
 namespace={'torch':torch};exec(source,namespace);candidate=namespace['decode_blocks']
 actual=candidate(torch.from_numpy(fixture['raw_12'].copy()).reshape(-1,144).to(device),12).cpu().numpy()
 np.testing.assert_array_equal(actual,fixture['expected_12'])
 original=decode_blocks
 gguf=GGUFFile(Path(__file__).resolve().parents[1]/'weights/minimax_h3_fl2va_pruned-Q4_K.gguf')
 raw=gguf.raw('blocks.0.attn.out_proj.weight',256,144)
 blocks=torch.from_numpy(raw[:2048*7168//256*144].copy()).reshape(-1,144).to(device)
 ref=original(blocks,12);new=candidate(blocks,12)
 torch.testing.assert_close(ref,new,rtol=0,atol=0)
 del ref,new
 times={}
 for name,fn in [('int32_original',original),('uint8_optimized',candidate)]:
  fn(blocks,12);torch.npu.synchronize(0);start=time.monotonic()
  for _ in range(20):fn(blocks,12)
  torch.npu.synchronize(0);times[name]=(time.monotonic()-start)/20
 report={'completed':True,'fixture_exact_match':True,'real_rows_exact_match':True,'seconds_per_decode':times,'speedup':times['int32_original']/times['uint8_optimized']}
 output=Path('/tmp/minimax-h3-run/q4-decode-operators.json');output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
