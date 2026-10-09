"""Measure one real H3 block at larger resolutions, not a generation score."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model.h3 import H3Backbone
from model.layout import TextToVideoLayout
from model.runtime import initialize_npu


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=Path('/tmp/minimax-h3-run/resolution-benchmark.json'))
    parser.add_argument('--query-chunk', type=int, default=128)
    parser.add_argument('--key-chunk', type=int, default=256)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    import functools
    import model.h3 as h3_module
    h3_module.streaming_attention = functools.partial(h3_module.streaming_attention,
                                                     query_chunk=args.query_chunk, key_chunk=args.key_chunk)
    root = Path(__file__).resolve().parents[1]
    device = initialize_npu(args.device)
    backbone = H3Backbone(root/'weights/minimax_h3_fl2va_pruned-Q8_0.gguf', row_chunk=2048).to(device)
    backbone.store.enable_q8_device_cache()
    measured = {}
    if args.profile:
        from model.layers import GGUFLinear
        clocks = {}
        for name, module in backbone.blocks[0].named_modules():
            if isinstance(module, GGUFLinear):
                def before(module, inputs, name=name):
                    torch.npu.synchronize(device)
                    clocks[name] = time.monotonic()
                def after(module, inputs, output, name=name):
                    torch.npu.synchronize(device)
                    measured[name] = measured.get(name, 0) + time.monotonic()-clocks[name]
                module.register_forward_pre_hook(before)
                module.register_forward_hook(after)
        raw_attention = h3_module.streaming_attention
        def timed_attention(*values, **options):
            torch.npu.synchronize(device)
            started = time.monotonic()
            output = raw_attention(*values, **options)
            torch.npu.synchronize(device)
            measured['attention_kernel'] = measured.get('attention_kernel',0)+time.monotonic()-started
            return output
        h3_module.streaming_attention = timed_attention
    hidden = torch.from_numpy(np.load(args.text_cache)).to(device)
    text = backbone.encode_condition(hidden)
    results = []
    for width, height in [(512,288),(1280,736),(1920,1088)]:
        shape = (1,24,7,height//16,width//16)
        audio_shape = (2,32,37)
        layout = TextToVideoLayout.build(len(text),shape,audio_shape,device)
        video = torch.randn(shape,device=device)
        audio = torch.randn(audio_shape,device=device)
        x = layout.embed(backbone,text,video,audio)
        times, indices = layout.time_inputs(.1,.2,device)
        coordinates = backbone.time_coordinates(times)
        rope = backbone.rope_angles(layout.positions)
        # Warm disk/cache and compiled operators before timing at this shape.
        torch.npu.reset_peak_memory_stats(device)
        start = time.monotonic()
        y = backbone.blocks[0](x,coordinates,indices*3+layout.modalities,rope)
        torch.npu.synchronize(device)
        cold = time.monotonic()-start
        measured.clear()
        start = time.monotonic()
        y = backbone.blocks[0](x,coordinates,indices*3+layout.modalities,rope)
        torch.npu.synchronize(device)
        item = {'width':width,'height':height,'frames':22,'tokens':len(x),
                'cold_block_seconds':cold,'warm_block_seconds':time.monotonic()-start,
                'finite':bool(torch.isfinite(y).all()),
                'peak_allocated_bytes':torch.npu.max_memory_allocated(device),
                'component_seconds':dict(measured) if args.profile else None,
                'note':'One block only; full-model memory and final image quality are not validated.'}
        results.append(item)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(item),flush=True)
        del x,y,video,audio,layout,rope
        torch.npu.empty_cache()


if __name__=='__main__':
    main()
