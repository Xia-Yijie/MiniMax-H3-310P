"""Prepare reusable first/last or multiple-image conditions on Ascend."""
import argparse
import json
from pathlib import Path
import time
import torch
from model.runtime import initialize_npu
from infer.image_condition import add_image_arguments,request_spec,prepare,save_cache

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prompt',required=True)
    p.add_argument('--width',type=int,default=256);p.add_argument('--height',type=int,default=160)
    p.add_argument('--device',default='npu:0');p.add_argument('--row-chunk',type=int,default=2048)
    p.add_argument('--output',type=Path,required=True)
    add_image_arguments(p);args=p.parse_args()
    if min(args.width,args.height)<32 or args.width%2 or args.height%2:p.error('Width/height must be even and at least 32')
    root=Path(__file__).resolve().parents[1];spec=request_spec(args,root)
    if not spec['images']:p.error('At least one image is required')
    torch.set_num_threads(4)
    device=initialize_npu(int(args.device.split(':')[1]));started=time.monotonic()
    def log(stage,**values):print(json.dumps({'stage':stage,'elapsed':time.monotonic()-started,**values}),flush=True)
    values=prepare(spec,root,device,device,args.row_chunk,log)
    save_cache(args.output,spec,values)
    log('complete',path=str(args.output),hidden_shape=list(values[0].shape))
if __name__=='__main__':main()
