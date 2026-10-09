"""Experimental learned latent upscaling and low-noise H3 refinement.

Uses the original (non-PDD) flow head for arbitrary low-noise refinement.
PDD eight-step heads must not be reused at arbitrary noise levels.
"""
import argparse, gc, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from model.runtime import initialize_npu
from model.latent_upscaler import load_upscaler, upscale_normalized
from model.layout import TextToVideoLayout, unpatchify_video, unpack_audio
from model.h3 import H3Backbone
from model.flash_attention import prompt_flash_attention
from infer.generate import main as decode_video

@torch.no_grad()
def main(argv=None, resident=None):
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True,help='Clean normalized source .latents.npz')
    p.add_argument('--upscaler',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True)
    p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--strength',type=float,default=.15)
    p.add_argument('--upscale-method',choices=['learned','bilinear'],default='learned')
    p.add_argument('--pre-upscale-factor',type=float,default=0,
                   help='Learned intermediate spatial scale, then bilinear resize to target; 0 means direct learned resize')
    p.add_argument('--device',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vae-tile-size',type=int,default=256)
    p.add_argument('--vae-tile-overlap',type=int,default=64)
    p.add_argument('--vae-tile-batch-size',type=int,default=1)
    p.add_argument('--vae-output-device',choices=['npu','cpu'],default='npu')
    p.add_argument('--video-vae-checkpoint',type=Path)
    p.add_argument('--vae-direct-fp16',action='store_true')
    p.add_argument('--fused-npu-ops',action='store_true')
    p.add_argument('--vae-attention-backend',choices=['streaming','flash'],default='streaming')
    p.add_argument('--vae-prepared-cache-gib',type=float,default=0)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[1]/'weights/minimax_h3_fl2va_pruned-Q8_0.gguf')
    p.add_argument('--refine-prepared-cache-gib',type=float,default=0)
    p.add_argument('--backbone-family',choices=['fl2va','ref2va','hybrid'])
    a=p.parse_args(argv)
    if a.steps<0 or not 0<a.strength<=1 or min(a.width,a.height)<32 or a.width%2 or a.height%2 or (a.pre_upscale_factor and not 1<=a.pre_upscale_factor<=4): p.error('Invalid refinement settings')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();events=[]
    def log(stage,**data):
        record={'stage':stage,'seconds':time.monotonic()-started,**data};events.append(record)
        with a.output.with_suffix('.cascade.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    torch.set_num_threads(2);device=initialize_npu(a.device);torch.npu.reset_peak_memory_stats(device)
    meta=json.loads(a.source.with_suffix('.json').read_text())
    if meta.get('visual_request'):raise ValueError('Initial cascade experiment only supports text-only sources')
    text_meta=json.loads(a.text_cache.with_suffix('.json').read_text())
    if text_meta['prompt']!=meta['prompt']:raise ValueError('Source and text prompt mismatch')
    root=Path(__file__).resolve().parents[1]
    if text_meta['tokenizer_sha256']!=hashlib.sha256((root/'weights/processor/tokenizer.json').read_bytes()).hexdigest():raise ValueError('Tokenizer mismatch')
    with np.load(a.source) as f:
        video=torch.from_numpy(f['video'].copy()).to(device);audio=torch.from_numpy(f['audio'].copy()).to(device)
    cw=math.ceil(a.width/32)*32;ch=math.ceil(a.height/32)*32
    if a.upscale_method == 'bilinear':
        model = None
    elif resident is not None and 'upscaler' in resident:
        if resident['upscaler_checkpoint'] != str(a.upscaler.resolve()):
            raise ValueError('Resident upscaler checkpoint changed')
        model=resident['upscaler'];log('reused_resident_upscaler')
    else:
        model=load_upscaler(a.upscaler,device);log('upscaler_loaded')
        if resident is not None:
            resident.update(upscaler=model,upscaler_checkpoint=str(a.upscaler.resolve()))
    target = (ch//16,cw//16)
    if model is not None:
        intermediate = target
        if a.pre_upscale_factor:
            intermediate = tuple(min(dst, round(src*a.pre_upscale_factor))
                                 for src,dst in zip(video.shape[-2:],target))
        video=upscale_normalized(model,video,*intermediate)
    if tuple(video.shape[-2:]) != target:
        # Resize each latent time slice independently, preserving temporal samples.
        b,c,t,h,w=video.shape
        frames=video.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        frames=torch.nn.functional.interpolate(frames,size=target,mode='bilinear',align_corners=False)
        video=frames.reshape(b,t,c,*target).permute(0,2,1,3,4).contiguous()
        del frames
    log('upscaled',shape=list(video.shape));del model;gc.collect();torch.npu.empty_cache()
    up_peak=torch.npu.max_memory_allocated(device)
    if a.steps:
        import model.h3 as h3module
        h3module.streaming_attention=prompt_flash_attention
        model=H3Backbone(a.checkpoint,row_chunk=2048).to(device).eval()
        for block in model.blocks:
            block.mlp.token_chunk_size=1024
        from model.layers import set_fused_ops
        set_fused_ops(model,a.fused_npu_ops)
        if a.refine_prepared_cache_gib:
            model.store.enable_prepared_cache(int(a.refine_prepared_cache_gib*1024**3))
            # Decode remaining matrices on NPU rather than repeatedly on CPU.
            model.store.enable_quant_device_cache(6*1024**3)
        else:
            model.store.enable_quant_device_cache()
        hidden=torch.from_numpy(np.load(a.text_cache)).to(device)
        text=model.encode_condition(hidden)
        shape=tuple(video.shape);ashape=tuple(audio.shape)
        layout=TextToVideoLayout.build(len(text),shape,ashape,device)
        # Standard flow matching bridge: x_sigma=(1-sigma)*clean+sigma*noise.
        noise=torch.randn(shape,generator=torch.Generator().manual_seed(a.seed)).to(device)
        anoise=torch.randn(ashape,generator=torch.Generator().manual_seed(a.seed)).to(device)
        video=(1-a.strength)*video+a.strength*noise
        audio=(1-a.strength)*audio+a.strength*anoise
        del noise,anoise
        sigmas=torch.linspace(a.strength,0,a.steps+1)
        log('refine_start',steps=a.steps,strength=a.strength,adapter='none_original_flow_head')
        for step in range(a.steps):
            packed=layout.embed(model,text,video,audio)
            times,indices=layout.time_inputs(1-float(sigmas[step]),1-float(sigmas[step]),device)
            prediction=model.forward_packed(packed,times,indices,layout.modalities,layout.positions,
                video_indices=layout.video_slice,audio_indices=layout.audio_slice,
                progress=lambda done,total,x: log('refine_block',step=step+1,layer=done) if done%10==0 else None)
            delta=float(sigmas[step]-sigmas[step+1])
            video=video+unpatchify_video(prediction['video_rows'],shape)*delta
            audio=audio+unpack_audio(prediction['audio_rows'],ashape)*delta
            if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):raise FloatingPointError('Nonfinite refinement')
            log('refine_step',step=step+1)
        del model,hidden,text,layout,packed,prediction;gc.collect();torch.npu.empty_cache()
    latent=a.output.with_suffix('.latents.npz')
    np.savez(latent,video=video.cpu().numpy(),audio=audio.cpu().numpy())
    refine_peak=torch.npu.max_memory_allocated(device)
    metadata={'prompt':meta['prompt'],'steps':max(a.steps,1),'seed':a.seed,'width':a.width,'height':a.height,'frames':meta['frames'],
        'visual_request':None,'pdd_checkpoint':None,'backbone_checkpoint':str(a.checkpoint),
        'cascade_source':str(a.source),'refine_steps':a.steps,'refine_strength':a.strength,'upscaler':str(a.upscaler),
        'upscale_method':a.upscale_method,'pre_upscale_factor':a.pre_upscale_factor}
    latent.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    del video,audio;gc.collect();torch.npu.empty_cache();log('decode_start')
    report=decode_video(['--prompt',meta['prompt'],'--width',str(a.width),'--height',str(a.height),'--frames',str(meta['frames']),
        '--steps',str(max(a.steps,1)),'--seed',str(a.seed),'--generation-device',str(device),'--text-device',str(device),
        '--backbone-checkpoint',str(a.checkpoint),'--latents-cache',str(latent),'--output',str(a.output),
        '--attention-query-chunk','512','--attention-key-chunk','1024',
        '--vae-tile-size',str(a.vae_tile_size),'--vae-tile-overlap',str(a.vae_tile_overlap),
        '--vae-tile-batch-size',str(a.vae_tile_batch_size),
        '--vae-output-device',a.vae_output_device,
        '--vae-attention-backend',a.vae_attention_backend,'--vae-prepared-cache-gib',str(a.vae_prepared_cache_gib),
        *(['--video-vae-checkpoint',str(a.video_vae_checkpoint)] if a.video_vae_checkpoint else []),
        *(['--fused-npu-ops'] if a.fused_npu_ops else []),
        *(['--vae-direct-fp16'] if a.vae_direct_fp16 else []),
        *(['--backbone-family',a.backbone_family] if a.backbone_family else [])],resident=resident)
    report.update(cascade_source=str(a.source),cascade_steps=a.steps,cascade_strength=a.strength,
        cascade_total_seconds=time.monotonic()-started,upscale_refine_peak_allocated_bytes=refine_peak,upscale_peak_allocated_bytes=up_peak,
        scope='Cascade time excludes generation of the source video',events=events)
    a.output.with_suffix('.cascade.json').write_text(json.dumps(report,indent=2));log('complete',total_seconds=report['cascade_total_seconds'])
    return report
if __name__=='__main__':main()
