"""Conservative two-NPU block capacity screening, not a generation result."""
import argparse
import json
from pathlib import Path
import time
import functools
import numpy as np
import torch
from model.runtime import initialize_npu
from model.h3 import H3Backbone, move_between_devices
from model.pdd import install_pdd
from model.layers import GGUFLinear
from model.layout import TextToVideoLayout
from model.attention import streaming_attention
import model.h3 as h3

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--text-tokens',type=int,default=128,help='Synthetic text length for capacity screening')
    parser.add_argument('--text-cache',type=Path,help='Optional real prompt hidden-state .npy')
    parser.add_argument('--parallel-attention',action='store_true')
    parser.add_argument('--attention-backend',choices=['streaming','flash'],default='streaming')
    parser.add_argument('--compare-streaming',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    start=time.monotonic()
    devices=[initialize_npu(0),initialize_npu(1)]
    initialize_npu(0)
    h3.streaming_attention=functools.partial(streaming_attention,query_chunk=512,key_chunk=1024)
    attention_fn=streaming_attention
    if args.attention_backend=='flash':
        from model.flash_attention import prompt_flash_attention
        attention_fn=prompt_flash_attention
        h3.streaming_attention=attention_fn
    if args.parallel_attention:
        from model.dual_attention import DualNPUAttention
        parallel=DualNPUAttention(devices,attention_fn=attention_fn,query_chunk=512,key_chunk=1024)
        h3.streaming_attention=parallel
    selected_attention=h3.streaming_attention
    comparison=[]
    if args.compare_streaming:
        from model.dual_attention import DualNPUAttention
        reference_attention=DualNPUAttention(devices,query_chunk=512,key_chunk=1024)
    root=Path(__file__).resolve().parents[1]
    model=H3Backbone(root/'weights/minimax_h3_fl2va_pruned-Q4_K.gguf',row_chunk=2048)
    install_pdd(model,root/'weights/pdd/MiniMax-H3-FL2VA-Acc-8Step.safetensors',root/'weights/pdd/basis_fl2va.safetensors')
    model.split_devices(*devices).eval()
    model.store.enable_quant_device_cache()
    if args.smoke:
        # Check host staging exactly and isolate cross-card arithmetic per layer.
        x=torch.randn(4,model.config.hidden_size,device=devices[0])*.01
        times=torch.tensor([.1,.2],device=devices[0])
        indices=torch.tensor([0,1,0,1],device=devices[0])
        modalities=torch.tensor([1,2,0,0],device=devices[0])
        positions=torch.zeros(4,3,device=devices[0])
        kw=dict(video_indices=slice(2,4),audio_indices=slice(1,2))
        split=model.forward_packed(x,times,indices,modalities,positions,**kw)
        assert all(bool(torch.isfinite(v).all()) for v in split.values())
        torch.testing.assert_close(move_between_devices(move_between_devices(x,devices[1]),devices[0]),x,rtol=0,atol=0)
        coords=model.time_coordinates(times);mods=indices*3+modalities;rope=model.rope_angles(positions)
        block=model.blocks[25]
        second=block(move_between_devices(x,devices[1]),move_between_devices(coords,devices[1]),move_between_devices(mods,devices[1]),move_between_devices(rope,devices[1])).cpu()
        block.cpu().to(devices[0])
        first=block(x,coords,mods,rope).cpu()
        repeated=block(x,coords,mods,rope).cpu()
        errors={'isolated_block_cross_card_max_abs':float((first-second).abs().max()),'isolated_block_same_card_repeat_max_abs':float((first-repeated).abs().max())}
        print(json.dumps(errors),flush=True)
        torch.testing.assert_close(first,second,rtol=.003,atol=.003)
        report={'completed':True,'kind':'dual_npu_finite_forward_and_isolated_layer_smoke','errors':errors,'seconds':time.monotonic()-start,'full_sequence_numerical_equivalence_established':False,'notes':'Earlier 50-layer comparison failed 0.003 tolerance (max absolute discrepancy 0.1083); full-sequence drift requires further diagnosis. This check only validates exact transport, isolated layer agreement and finite split forward.'}
    else:
        # Preload every owned compressed matrix to include full resident weights.
        for name,module in model.named_modules():
            if isinstance(module,GGUFLinear):
                device=devices[0]
                if name.startswith('blocks.'):
                    device=model.block_devices[int(name.split('.')[1])]
                model.store.device_rows(module.name+'.weight',0,1,device)
        for device in devices:
            torch.npu.synchronize(device);torch.npu.empty_cache();torch.npu.reset_peak_memory_stats(device)
        ntext=np.load(args.text_cache).shape[0] if args.text_cache else args.text_tokens
        vshape=(1,24,((args.frames-5)//17)*5+2,68,120)
        ashape=(2,32,round(args.frames/24*40))
        layout=TextToVideoLayout.build(ntext,vshape,ashape,devices[0])
        original=torch.randn(layout.positions.shape[0],model.config.hidden_size,device=devices[0])*.01
        coords=model.time_coordinates(torch.tensor([.1,.2],device=devices[0]))
        _,indices=layout.time_inputs(.1,.2,devices[0])
        mods=indices*3+layout.modalities
        rope=model.rope_angles(layout.positions)
        x=original
        for index,device in [(0,devices[0]),(25,devices[1])]:
            inputs=(move_between_devices(x,device),move_between_devices(coords,device),move_between_devices(mods,device),move_between_devices(rope,device))
            if args.compare_streaming:
                h3.streaming_attention=reference_attention
                t=time.monotonic();reference=model.blocks[index](*inputs)
                torch.npu.synchronize(device);reference_seconds=time.monotonic()-t
                reference=reference.cpu()
            h3.streaming_attention=selected_attention
            t=time.monotonic();x=model.blocks[index](*inputs)
            torch.npu.synchronize(device);selected_seconds=time.monotonic()-t
            if args.compare_streaming:
                reference=reference.to(device)
                diff=x-reference
                rms=float(diff.square().mean().sqrt()/reference.square().mean().sqrt().clamp(min=1e-10))
                comparison.append({'block':index,'device':str(device),'streaming_seconds':reference_seconds,'selected_seconds':selected_seconds,'speedup':reference_seconds/selected_seconds,'relative_rms':rms,'max_abs_error':float(diff.abs().max())})
                print(json.dumps(comparison[-1]),flush=True)
                assert rms<.01,comparison[-1]
                del reference,diff
            assert bool(torch.isfinite(x).all())
            torch.npu.synchronize(device)
            print(json.dumps({'stage':'probe_block_complete','frames':args.frames,'device':str(device),'seconds':time.monotonic()-start}),flush=True)
        memory={}
        # Reserve room for decoder weights, RGB accumulation/concatenation, and allocator overhead.
        allowance=6*1024**3+3*(args.frames*1920*1088*3*4)+2*1024**3
        for device in devices:
            free,total=torch.npu.mem_get_info(device)
            peak=torch.npu.max_memory_allocated(device)
            memory[str(device)]={'peak_allocated_bytes':peak,'peak_reserved_bytes':torch.npu.max_memory_reserved(device),'total_bytes':total,'free_bytes':free,'decoder_and_safety_allowance_bytes':allowance,'conservative_fit':peak+allowance<total}
        report={'completed':True,'kind':'representative_block_screen_only','frames':args.frames,'video_seconds':args.frames/24,'seconds':time.monotonic()-start,'memory':memory,'conservative_fit':all(v['conservative_fit'] for v in memory.values()),'full_generation_verified':False,'parallel_attention':args.parallel_attention,'attention_backend':args.attention_backend}
        report['block_comparison']=comparison
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if args.parallel_attention:
        parallel.close()
    if args.compare_streaming:
        reference_attention.close()
if __name__=='__main__': main()
